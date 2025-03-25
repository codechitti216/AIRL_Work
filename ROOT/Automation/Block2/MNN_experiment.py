import os
import json
import re
import sys
import pandas as pd
import numpy as np
import time
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import the Memory Neural Network model from MNN.py
from MNN import MemoryNeuralNetwork

##########################
# Experiment Folder Setup
##########################

RESULTS_ROOT = "Results/MNN_Results"  # Root folder for MNN experiments

ABBREVIATIONS = {
    "learning_rate": "lr",
    "dropout_rate": "dr",
    "hidden_neurons": "hn",
    "regularization": "reg",
    "beam_fill_window": "bfw",
    # Run-specific parameters:
    "epochs": "ep",
    "num_past_beam_instances": "npbi",
    "num_imu_instances": "niu",
    "num_layers": "nl",
    "training_trajectories": "trtraj",
    "testing_trajectories": "ttraj"
}

def sanitize(text):
    text = str(text).replace(":", "_")
    text = re.sub(r'\s+', '_', text)
    text = re.sub(r'[^\w\-_\.]', '', text)
    return text

def stringify_value(value):
    if isinstance(value, list):
        if all(isinstance(item, (list, tuple)) for item in value):
            return "_".join(["-".join(map(str, item)) for item in value])
        else:
            return "-".join(map(str, value))
    elif isinstance(value, bool):
        return "T" if value else "F"
    else:
        return str(value)

CORE_KEYS = ["learning_rate", "dropout_rate", "hidden_neurons", "regularization", "beam_fill_window"]

def generate_experiment_folder_name(config):
    parts = []
    for key in CORE_KEYS:
        if key in config:
            abbr_key = ABBREVIATIONS.get(key, sanitize(key))
            value_str = stringify_value(config[key])
            sanitized_value = sanitize(value_str)
            parts.append(f"{abbr_key}{sanitized_value}")
    folder_name = "Exp_" + "_".join(parts)
    print(f"[DEBUG] Generated experiment folder name: {folder_name}")
    return folder_name

def create_experiment_folder(config):
    folder_name = generate_experiment_folder_name(config)
    folder_name = folder_name.replace(":", "_").replace("*", "_").replace("?", "_")\
                              .replace("<", "_").replace(">", "_").replace("|", "_")
    folder_name = folder_name.rstrip(" .").lower()
    exp_folder = os.path.join(RESULTS_ROOT, folder_name)
    if os.path.exists(exp_folder):
        print(f"[DEBUG] Experiment folder already exists: {exp_folder}. Reusing this folder.")
    else:
        os.makedirs(exp_folder, exist_ok=True)
    subdirs = {
        "CHECKPOINTS_DIR": "Checkpoints",
        "TRAINING_SUMMARIES_DIR": "TrainingSummaries",
        "TEST_SUMMARIES_DIR": "TestSummaries",
        "PLOTS_DIR": "Plots",
        "PREDICTIONS_DIR": "Predictions"
    }
    global CHECKPOINTS_DIR, TRAINING_SUMMARIES_DIR, TEST_SUMMARIES_DIR, PLOTS_DIR, PREDICTIONS_DIR, GLOBAL_LOG_FILE
    for key, sub in subdirs.items():
        path = os.path.join(exp_folder, sub)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        globals()[key] = path
        print(f"[DEBUG] Subdirectory for {key}: {path}")
    GLOBAL_LOG_FILE = os.path.join(exp_folder, "experiment_global_log.txt")
    print(f"[DEBUG] Global log file set to: {GLOBAL_LOG_FILE}")
    return exp_folder

##########################
# Run-Specific Identifier
##########################

def generate_run_specific_identifier(config):
    train_parts = []
    for idx, pair in enumerate(config.get("training_trajectories", []), start=1):
        ep = str(pair[1])
        train_parts.append(f"T{idx}-{ep}")
    trtraj_str = "-".join(train_parts) if train_parts else "None"
    ttraj_list = config.get("testing_trajectories", [])
    ttraj_str = "-".join([sanitize(t) for t in ttraj_list]) if ttraj_list else "None"
    npbi = config.get("num_past_beam_instances", "NA")
    niu = config.get("num_imu_instances", "NA")
    nl = config.get("num_layers", "NA")
    raw_run_id = f"{ABBREVIATIONS.get('num_past_beam_instances','npbi')}{npbi}_" \
                 f"{ABBREVIATIONS.get('num_imu_instances','niu')}{niu}_" \
                 f"{ABBREVIATIONS.get('num_layers','nl')}{nl}_" \
                 f"trtraj{trtraj_str}_ttraj{ttraj_str}"
    run_id = sanitize(raw_run_id)
    return run_id

##########################
# Duplicate Check Function
##########################

def check_duplicate_file(subfolder, filename):
    file_path = os.path.join(subfolder, filename)
    print(f"[DEBUG] Checking duplicate for file: {file_path}")
    if os.path.exists(file_path):
        log_global(f"Duplicate file found for configuration: {file_path}. Aborting experiment run.")
        sys.exit(0)
    else:
        print(f"[DEBUG] No duplicate file found: {file_path}")
    return file_path

##########################
# Logging
##########################

def log_global(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(GLOBAL_LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(message)

##########################
# Data Loading & Preprocessing
##########################

DATA_DIR = "../../Data"
REQUIRED_VELOCITY_COLS = ["V North", "V East", "V Down"]

def load_csv_files(traj_path):
    print(f"[INFO] Loading CSV files from trajectory folder: {traj_path}")
    beams_gt_path = os.path.join(traj_path, "beams_gt.csv")
    velocity_gt_path = os.path.join(traj_path, "velocity_gt.csv")
    
    try:
        beams_gt = pd.read_csv(beams_gt_path, na_values=[''])
    except Exception as e:
        log_global(f"ERROR: Failed to load beams_gt from {beams_gt_path}: {e}")
        raise e

    try:
        velocity_gt = pd.read_csv(velocity_gt_path, na_values=[''])
    except Exception as e:
        log_global(f"ERROR: Failed to load velocity_gt from {velocity_gt_path}: {e}")
        raise e

    # Check required velocity columns
    if not all(col in velocity_gt.columns for col in REQUIRED_VELOCITY_COLS):
        err_msg = f"ERROR: velocity_gt file missing required columns. Expected: {REQUIRED_VELOCITY_COLS}, Found: {list(velocity_gt.columns)}"
        log_global(err_msg)
        raise ValueError(err_msg)
    
    beams_training = beams_gt.copy()
    imu_files = [f for f in os.listdir(traj_path) if f.startswith("IMU_") and f.endswith(".csv")]
    if not imu_files:
        err_msg = f"ERROR: No IMU file found in {traj_path}"
        log_global(err_msg)
        raise ValueError(err_msg)
    try:
        imu = pd.read_csv(os.path.join(traj_path, imu_files[0]))
    except Exception as e:
        log_global(f"ERROR: Failed to load IMU file: {e}")
        raise e

    # Sort based on time
    beams_gt.sort_values("Time", inplace=True)
    beams_training.sort_values("Time", inplace=True)
    velocity_gt.sort_values("Time", inplace=True)
    imu.sort_values("Time [s]", inplace=True)

    # Reset indexes
    beams_gt.reset_index(drop=True, inplace=True)
    beams_training.reset_index(drop=True, inplace=True)
    velocity_gt.reset_index(drop=True, inplace=True)
    imu.reset_index(drop=True, inplace=True)

    # Check if beams_gt and velocity_gt have same length
    if len(beams_gt) != len(velocity_gt):
        log_global(f"WARNING: beams_gt length ({len(beams_gt)}) != velocity_gt length ({len(velocity_gt)}). Data may not be aligned.")
    else:
        print(f"[INFO] beams_gt and velocity_gt have matching lengths: {len(beams_gt)}")

    log_global(f"Loaded files: beams_gt={len(beams_gt)} rows, velocity_gt={len(velocity_gt)} rows, imu={len(imu)} rows")
    return beams_gt, beams_training, imu, velocity_gt

##########################
# Functions for Testing-Only Missing Data (Range Removal)
##########################

def apply_range_removal(beams_df, config):
    """
    Instead of random removal, for testing we remove (set to NaN) any beam value
    that is not within the specified range. The range is taken from config['beam_range'].
    This function returns the modified DataFrame and a dictionary with the count of removals per beam.
    """
    lower, upper = config.get("beam_range", [-1.5, 1.5])
    count_removed = {'b1': 0, 'b2': 0, 'b3': 0, 'b4': 0}
    start_idx = config.get("beam_fill_window", 3)
    print(f"[INFO] Applying range removal from row {start_idx} with range [{lower}, {upper}]")
    for idx in range(start_idx, beams_df.shape[0]):
        for beam in ['b1', 'b2', 'b3', 'b4']:
            value = beams_df.loc[idx, beam]
            if pd.notna(value):
                if not (lower <= value <= upper):
                    beams_df.loc[idx, beam] = np.nan
                    count_removed[beam] += 1
    total_removed = sum(count_removed.values())
    print(f"[DEBUG] Applied range removal from row {start_idx} onward. Total beams removed: {total_removed}")
    return beams_df, count_removed

def plot_missing_beams_frequency(count_removed, traj, run_id):
    fig, ax = plt.subplots(figsize=(8, 6))
    beams = list(count_removed.keys())
    counts = list(count_removed.values())
    ax.bar(beams, counts, color='salmon')
    ax.set_xlabel("Beam")
    ax.set_ylabel("Number of beams removed")
    ax.set_title(f"Number of beams removed (out-of-range) for {traj}")
    plot_path = os.path.join(PLOTS_DIR, f"BeamRemovalCount_{sanitize(traj)}_{run_id}.png")
    fig.savefig(plot_path)
    plt.close(fig)
    log_global(f"[{traj}] Beam removal count plot saved to {plot_path}")

##########################
# Fill Missing Values (Sequential Moving Average)
##########################

def fill_missing_beams(beams_df, beam_fill_window, beam_cols=["b1", "b2", "b3", "b4"]):
    filled = beams_df.copy()
    for i in range(beam_fill_window, len(filled)):
        for col in beam_cols:
            if pd.isna(filled.loc[i, col]):
                window = filled.loc[i - beam_fill_window:i - 1, col]
                if window.isna().all():
                    last_val = filled[col].ffill().iloc[i - 1]
                    log_global(f"DEBUG: All previous values missing for {col} at row {i}; using last valid value {last_val}")
                    filled.loc[i, col] = last_val
                else:
                    avg_val = window.mean()
                    log_global(f"DEBUG: Filling missing {col} at row {i} with moving average {avg_val}")
                    filled.loc[i, col] = avg_val
    return filled, beam_fill_window

##########################
# Construct Input-Target
##########################

def construct_input_target(filled_beams, velocity_gt, imu, t, num_past_beam_instances, num_imu_instances):
    # Check if t is within bounds for velocity_gt
    if t >= len(velocity_gt):
        err_msg = f"ERROR: Index {t} is out of bounds for velocity_gt (length {len(velocity_gt)})"
        log_global(err_msg)
        raise IndexError(err_msg)
    
    if t < num_past_beam_instances or t < (num_imu_instances - 1):
        err_msg = f"ERROR: Index {t} does not satisfy minimum history requirements (num_past_beam_instances={num_past_beam_instances}, num_imu_instances={num_imu_instances})"
        log_global(err_msg)
        raise ValueError(err_msg)
    
    try:
        current_beams = filled_beams.loc[t, ["b1", "b2", "b3", "b4"]].values.astype(float)
    except Exception as e:
        err_msg = f"ERROR at index {t} fetching current beams: {e}"
        log_global(err_msg)
        raise e

    past_beams = []
    for i in range(1, num_past_beam_instances + 1):
        idx = t - i
        if idx < 0:
            err_msg = f"ERROR: Negative index {idx} computed for past beams at t={t}"
            log_global(err_msg)
            raise ValueError(err_msg)
        try:
            past_row = filled_beams.loc[idx, ["b1", "b2", "b3", "b4"]].values.astype(float)
            past_beams.extend(past_row)
        except Exception as e:
            err_msg = f"ERROR at index {idx} fetching past beams: {e}"
            log_global(err_msg)
            raise e

    imu_cols = ['ACC X [m/s^2]', 'ACC Y [m/s^2]', 'ACC Z [m/s^2]',
                'GYRO X [rad/s]', 'GYRO Y [rad/s]', 'GYRO Z [rad/s]']
    for col in imu_cols:
        if col not in imu.columns:
            err_msg = f"ERROR: Expected IMU column {col} not found in IMU data."
            log_global(err_msg)
            raise ValueError(err_msg)
    past_imu = []
    for i in range(num_imu_instances - 1, -1, -1):
        idx = t - i
        if idx < 0:
            err_msg = f"ERROR: Negative index {idx} computed for IMU data at t={t}"
            log_global(err_msg)
            raise ValueError(err_msg)
        try:
            imu_row = imu.loc[idx, imu_cols].values.astype(float)
            past_imu.extend(imu_row)
        except Exception as e:
            err_msg = f"ERROR at index {idx} fetching IMU data: {e}"
            log_global(err_msg)
            raise e

    input_vector = np.concatenate([current_beams, np.array(past_beams), np.array(past_imu)])
    
    try:
        target_vector = velocity_gt.loc[t, REQUIRED_VELOCITY_COLS].values.astype(float)
    except Exception as e:
        err_msg = f"ERROR at index {t} fetching target velocities: {e}"
        log_global(err_msg)
        raise e
    return input_vector, target_vector

def get_missing_mask(beams_training, t, target_cols=["b1", "b2", "b3", "b4"]):
    row = beams_training.loc[t, target_cols]
    return row.isna().values

##########################
# Plotting Functions
##########################

def plot_velocity_predictions(predictions, traj, beam_fill_window, title_suffix=""):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    samples = [pred["Sample"] for pred in predictions]
    for i, comp in enumerate(["V_North", "V_East", "V_Down"]):
        pred_vals = [pred[f"Pred_{comp}"] for pred in predictions]
        gt_vals = [pred[f"GT_{comp}"] for pred in predictions]
        gt_series = pd.Series(gt_vals)
        moving_avg = gt_series.rolling(window=beam_fill_window, min_periods=1).mean().tolist()
        print(f"[DEBUG Plot] {comp}: First 5 Predicted: {pred_vals[:5]}")
        print(f"[DEBUG Plot] {comp}: First 5 Ground Truth: {gt_vals[:5]}")
        axes[i].plot(samples, gt_vals, label=f"Ground Truth {comp}", marker='x')
        axes[i].plot(samples, pred_vals, label=f"Predicted {comp}", marker='o')
        axes[i].set_ylabel("Value")
        axes[i].legend(loc="upper right")
        axes[i].grid(True)
    axes[-1].set_xlabel("Sample Index")
    fig.suptitle(f"Predicted vs Ground Truth & Moving Avg for {traj} {title_suffix}")
    return fig

##########################
# Finding the Starting Index
##########################

def find_first_valid_index(filled_beams, beams_gt, imu, num_past_beam_instances, num_imu_instances, velocity_gt):
    start = max(num_past_beam_instances, num_imu_instances - 1)
    log_global(f"Starting search for valid sample from index {start}")
    for t in range(start, min(len(filled_beams), len(velocity_gt))):
        try:
            # Verify that t works for both beams and velocity data.
            inp, tar = construct_input_target(filled_beams, velocity_gt, imu, t, num_past_beam_instances, num_imu_instances)
            log_global(f"Valid input-target pair found at index {t}.")
            return t
        except Exception as e:
            log_global(f"Index {t} invalid: {e}")
            continue
    return None

##########################
# Sequential Training Routine
##########################

def sequential_train(training_trajectory_pairs, config, model, run_id, trained_list):
    global_training_summary = []
    processed_training_info = []
    
    for traj_pair in training_trajectory_pairs:
        traj, traj_epochs = traj_pair
        traj_path = os.path.join(DATA_DIR, traj)
        log_global(f"=== Training on Trajectory: {traj} for {traj_epochs} epochs ===")
        try:
            beams_gt, beams_training, imu, velocity_gt = load_csv_files(traj_path)
        except Exception as e:
            log_global(f"Error loading files in {traj}: {e}")
            continue

        # For training, we do NOT remove any beams based on range.
        orig_missing = beams_training[["b1", "b2", "b3", "b4"]].isna().sum().sum()
        log_global(f"Original missing count in {traj}: {orig_missing}")
        
        filled_beams, _ = fill_missing_beams(beams_training, config["beam_fill_window"])
        min_history = find_first_valid_index(filled_beams, beams_gt, imu, config["num_past_beam_instances"], config["num_imu_instances"], velocity_gt)
        if min_history is None:
            log_global("Not enough training data in the trajectory to determine input size.")
            continue
        log_global(f"Using starting index {min_history} for constructing input-target pairs.")
        
        inputs, targets, masks = [], [], []
        for t in range(min_history, min(len(filled_beams), len(velocity_gt))):
            try:
                inp, tar = construct_input_target(filled_beams, velocity_gt, imu, t, config["num_past_beam_instances"], config["num_imu_instances"])
                inputs.append(inp)
                targets.append(tar)
                masks.append(get_missing_mask(beams_training, t))
                log_global(f"Constructed sample at index {t}.")
            except Exception as e:
                log_global(f"Skipping index {t}: {e}")
                continue
        if len(inputs) == 0:
            log_global("Not enough training data in the trajectory to determine input size.")
            continue
        inputs = np.array(inputs)
        targets = np.array(targets)
        masks = np.array(masks)
        num_samples = len(inputs)
        input_size = inputs.shape[1]
        log_global(f"[{traj}] Training Samples: {num_samples}, Input size: {input_size}")
        
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["regularization"])
        loss_fn = nn.MSELoss()
        
        evolution = []
        t0 = time.time()
        for epoch in range(1, traj_epochs + 1):
            optimizer.zero_grad()
            losses = []
            epoch_squared_errors = np.zeros(3)  # 3 outputs for velocities
            for i in range(num_samples):
                model.train()
                x = torch.tensor(inputs[i], dtype=torch.float32, device=model.device).unsqueeze(0).unsqueeze(0)
                y = torch.tensor(targets[i], dtype=torch.float32, device=model.device)
                y_pred = model(x).squeeze().view(-1)
                y = y.view(-1)
                sample_loss = loss_fn(y_pred, y)
                losses.append(sample_loss)
                error = (y_pred - y).detach().cpu().numpy() ** 2
                epoch_squared_errors += error
            epoch_loss = torch.stack(losses).sum()
            epoch_loss.backward()
            optimizer.step()
            epoch_rmse = np.sqrt(epoch_squared_errors / num_samples)
            evolution.append([epoch] + epoch_rmse.tolist())
            log_global(f"[{traj}] Epoch {epoch}: RMSE per output: {epoch_rmse}, Avg RMSE: {np.mean(epoch_rmse):.5f}")
        training_time = time.time() - t0
        
        model.eval()
        final_predictions = []
        for i in range(num_samples):
            x = torch.tensor(inputs[i], dtype=torch.float32, device=model.device).unsqueeze(0).unsqueeze(0)
            y = torch.tensor(targets[i], dtype=torch.float32, device=model.device)
            with torch.no_grad():
                y_pred = model(x).squeeze().view(-1)
            y = y.view(-1)
            final_predictions.append({
                "Sample": i,
                "Pred_V_North": y_pred[0].item(),
                "Pred_V_East": y_pred[1].item(),
                "Pred_V_Down": y_pred[2].item(),
                "GT_V_North": y[0].item(),
                "GT_V_East": y[1].item(),
                "GT_V_Down": y[2].item()
            })
        print(f"[DEBUG] Final predictions for {traj} (first 5 samples): {final_predictions[:5]}")
        train_plot_path = os.path.join(PLOTS_DIR, f"FinalOutputPredictions_{sanitize(traj)}_{run_id}_range_removed.png")
        train_fig = plot_velocity_predictions(final_predictions, traj, config["beam_fill_window"], title_suffix="(Training)")
        train_fig.savefig(train_plot_path)
        plt.close(train_fig)
        log_global(f"[{traj}] Training predictions plot saved to {train_plot_path}")
        
        current_trained_on = ",".join(trained_list) if trained_list else "NONE"
        summary = {
            "Trajectory": traj,
            "NumSamples": num_samples,
            "InputSize": input_size,
            "EpochsTrained": traj_epochs,
            "AvgBestRMSE": np.mean(evolution[-1][1:]),
            "TrainingTime": training_time,
            "TrainedOn": current_trained_on,
            "RemovalMethod": "RangeRemoval"
        }
        global_training_summary.append(summary)
        trained_list.append(f"{traj}:{traj_epochs}")
    return global_training_summary

def test_on_trajectory(traj, config, checkpoint_filename, run_id, base_trained_on):
    traj_path = os.path.join(DATA_DIR, traj)
    beams_gt, beams_training, imu, velocity_gt = load_csv_files(traj_path)
    # For testing, apply range removal: remove any beam value not between -1.5 and 1.5.
    beams_training, count_removed = apply_range_removal(beams_training, config)
    filled_beams, _ = fill_missing_beams(beams_training, config["beam_fill_window"])
    log_global(f"[{traj}] Total beams removed due to out-of-range: {sum(count_removed.values())}")
    plot_missing_beams_frequency(count_removed, traj, run_id)
    
    min_history = max(config["beam_fill_window"], config["num_past_beam_instances"], config["num_imu_instances"] - 1)
    inputs, targets, masks = [], [], []
    for t in range(min_history, min(len(filled_beams), len(velocity_gt))):
        try:
            inp, tar = construct_input_target(filled_beams, velocity_gt, imu, t, config["num_past_beam_instances"], config["num_imu_instances"])
            inputs.append(inp)
            targets.append(tar)
            masks.append(get_missing_mask(beams_training, t, target_cols=["b1", "b2", "b3", "b4"]))
        except Exception as e:
            log_global(f"Skipping index {t} in testing: {e}")
            continue
    if len(inputs) == 0:
        log_global(f"[Test] Not enough test data in {traj} after history constraints. Skipping.")
        return None
    inputs = np.array(inputs)
    targets = np.array(targets)
    num_samples = len(inputs)
    input_size = inputs.shape[1]
    log_global(f"[{traj}] Testing Samples: {num_samples}, Input size: {input_size}")
    
    model = MemoryNeuralNetwork(number_of_input_neurons=input_size,
                                number_of_hidden_neurons=config["hidden_neurons"],
                                number_of_output_neurons=3,  # 3 outputs for velocities
                                dropout_rate=config["dropout_rate"])
    cp_full_path = os.path.join(CHECKPOINTS_DIR, checkpoint_filename)
    try:
        state = torch.load(cp_full_path, map_location=model.device, weights_only=True)
    except Exception as e:
        log_global(f"[Test] Error loading checkpoint for {traj}: {e}")
        return None
    model.load_state_dict(state)
    model.eval()
    
    predictions = []
    squared_errors = np.zeros(3)
    for i in range(num_samples):
        x = torch.tensor(inputs[i], dtype=torch.float32, device=model.device).unsqueeze(0).unsqueeze(0)
        y = torch.tensor(targets[i], dtype=torch.float32, device=model.device)
        with torch.no_grad():
            y_pred = model(x).squeeze().view(-1)
        predictions.append({
            "Sample": i,
            "Pred_V_North": y_pred[0].item(),
            "Pred_V_East": y_pred[1].item(),
            "Pred_V_Down": y_pred[2].item(),
            "GT_V_North": y[0].item(),
            "GT_V_East": y[1].item(),
            "GT_V_Down": y[2].item()
        })
        err = (y_pred - y.view(-1)).detach().cpu().numpy() ** 2
        squared_errors += err
    test_rmse = np.sqrt(squared_errors / num_samples)
    log_global(f"[{traj}] Test RMSE per output: {test_rmse}")
    
    test_pred_csv = os.path.join(PREDICTIONS_DIR, f"TestPredictions_{sanitize(traj)}_{run_id}_range_removed.csv")
    pd.DataFrame(predictions).to_csv(test_pred_csv, index=False)
    log_global(f"[{traj}] Test predictions saved to {test_pred_csv}")
    
    plot_file = os.path.join(PLOTS_DIR, f"VelocityPredictions_{sanitize(traj)}_{run_id}_range_removed.png")
    fig = plot_velocity_predictions(predictions, traj, config["beam_fill_window"], title_suffix="(Testing Data)")
    fig.savefig(plot_file)
    plt.close(fig)
    log_global(f"[{traj}] Velocity predictions plot saved to {plot_file}")
    
    test_summary = {
        "Trajectory": traj,
        "NumSamples": num_samples,
        "Test_RMSE_V_North": test_rmse[0],
        "Test_RMSE_V_East": test_rmse[1],
        "Test_RMSE_V_Down": test_rmse[2],
        "AvgTest_RMSE": np.mean(test_rmse),
        "TrainedOn": base_trained_on,
        "RemovalMethod": "RangeRemoval"
    }
    return test_summary

##########################
# Main Routine
##########################

def main():
    with open(GLOBAL_LOG_FILE, "w") as f:
        f.write("Experiment Log\n")
    
    with open("MNN.json", "r") as f:
        config = json.load(f)
    
    training_trajectory_pairs = config.get("training_trajectories", [])
    testing_list = config.get("testing_trajectories", [])
    
    if not training_trajectory_pairs:
        log_global("No training trajectories provided.")
        return

    # Since we are no longer using random removal percentages,
    # we simply denote the removal method in file names.
    missing_freq_str = "range_removed"
    print(f"[DEBUG] Removal method: {missing_freq_str}")
    
    cumulative_trained_on = []  
    global_training_summary = []
    
    # Determine input size from first training trajectory.
    first_traj = training_trajectory_pairs[0][0]
    first_traj_path = os.path.join(DATA_DIR, first_traj)
    beams_gt, beams_training, imu, velocity_gt = load_csv_files(first_traj_path)
    filled_beams, _ = fill_missing_beams(beams_training, config["beam_fill_window"])
    min_history = find_first_valid_index(filled_beams, beams_gt, imu, config["num_past_beam_instances"], config["num_imu_instances"], velocity_gt)
    if min_history is None:
        log_global("Not enough training data in the first trajectory to determine input size.")
        return
    log_global(f"Using starting index {min_history} from the first trajectory.")
    
    if len(velocity_gt) <= min_history:
        log_global(f"ERROR: Not enough rows in velocity_gt (length {len(velocity_gt)}) to satisfy the min_history requirement ({min_history}).")
        return

    inputs = []
    for t in range(min_history, min(len(filled_beams), len(velocity_gt))):
        try:
            inp, _ = construct_input_target(filled_beams, velocity_gt, imu, t, config["num_past_beam_instances"], config["num_imu_instances"])
            inputs.append(inp)
            log_global(f"Constructed sample at index {t}.")
        except Exception as e:
            log_global(f"Skipping index {t} in main: {e}")
            continue
    if len(inputs) == 0:
        log_global("Not enough training data in the first trajectory to determine input size.")
        return
    input_size = np.array(inputs).shape[1]
    model = MemoryNeuralNetwork(number_of_input_neurons=input_size,
                                number_of_hidden_neurons=config["hidden_neurons"],
                                number_of_output_neurons=3,  # 3 outputs for velocities
                                dropout_rate=config["dropout_rate"],
                                learning_rate=config["learning_rate"],
                                learning_rate_2=config["learning_rate_2"],
                                lipschitz_constant=config["lipschitz_constant"])
    
    run_id = generate_run_specific_identifier(config)
    
    log_global("=== Sequential Training Phase ===")
    for traj_pair in training_trajectory_pairs:
        traj, traj_epochs = traj_pair
        log_global(f"Processing training trajectory: {traj} for {traj_epochs} epochs")
        current_trained_on = ",".join(cumulative_trained_on) if cumulative_trained_on else "NONE"
        summary = sequential_train([traj_pair], config, model, run_id, cumulative_trained_on)
        if summary:
            summary[0]["TrainedOn"] = current_trained_on
            global_training_summary.append(summary[0])
            cumulative_trained_on.append(f"{traj}:{traj_epochs}")
    
    if not global_training_summary:
        log_global("No training trajectories processed; aborting.")
        return

    final_checkpoint_filename = f"MNN_{run_id}_{missing_freq_str}_final.pth"
    cp_path = check_duplicate_file(CHECKPOINTS_DIR, final_checkpoint_filename)
    torch.save(model.state_dict(), cp_path)
    log_global(f"Final checkpoint saved to {cp_path}")
    
    train_summary_filename = f"GlobalTrainingSummary_{run_id}_{missing_freq_str}.csv"
    ts_path = check_duplicate_file(TRAINING_SUMMARIES_DIR, train_summary_filename)
    pd.DataFrame(global_training_summary).to_csv(ts_path, index=False)
    log_global(f"Global training summary saved to {ts_path}")
    
    log_global("=== Testing Phase ===")
    global_test_summary = []
    base_trained_on = ",".join(dict.fromkeys(cumulative_trained_on)) if cumulative_trained_on else "NONE"
    for traj in testing_list:
        log_global(f"Processing testing trajectory: {traj}")
        test_summary = test_on_trajectory(traj, config, final_checkpoint_filename, run_id, base_trained_on)
        if test_summary:
            global_test_summary.append(test_summary)
    if global_test_summary:
        test_summary_filename = f"GlobalTestSummary_{run_id}_{missing_freq_str}.csv"
        tst_path = check_duplicate_file(TEST_SUMMARIES_DIR, test_summary_filename)
        pd.DataFrame(global_test_summary).to_csv(tst_path, index=False)
        log_global(f"Global test summary saved to {tst_path}")

if __name__ == "__main__":
    with open("MNN.json", "r") as f:
        config = json.load(f)
    EXPERIMENT_FOLDER = create_experiment_folder(config)
    global CHECKPOINTS_DIR, TRAINING_SUMMARIES_DIR, TEST_SUMMARIES_DIR, PLOTS_DIR, PREDICTIONS_DIR, GLOBAL_LOG_FILE
    CHECKPOINTS_DIR = os.path.join(EXPERIMENT_FOLDER, "Checkpoints")
    TRAINING_SUMMARIES_DIR = os.path.join(EXPERIMENT_FOLDER, "TrainingSummaries")
    TEST_SUMMARIES_DIR = os.path.join(EXPERIMENT_FOLDER, "TestSummaries")
    PLOTS_DIR = os.path.join(EXPERIMENT_FOLDER, "Plots")
    PREDICTIONS_DIR = os.path.join(EXPERIMENT_FOLDER, "Predictions")
    GLOBAL_LOG_FILE = os.path.join(EXPERIMENT_FOLDER, "experiment_global_log.txt")
    
    log_global(f"Experiment folder created: {EXPERIMENT_FOLDER}")
    main()
