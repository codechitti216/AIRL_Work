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

# Import the TCN model.
from TCN import TCN

##########################
# Experiment Folder Setup
##########################

RESULTS_ROOT = "Results/TCN_Results"

# Use the same core keys as in LSTM_experiment.py
CORE_KEYS = ["learning_rate", "dropout_rate", "hidden_neurons", "regularization", "beam_fill_window"]

ABBREVIATIONS = {
    "learning_rate": "lr",
    "dropout_rate": "dr",
    "hidden_neurons": "hn",   # used as num_channels in TCN
    "regularization": "reg",
    "beam_fill_window": "bfw",
    # Run-specific parameters:
    "num_past_beam_instances": "npbi",
    "num_imu_instances": "nimu",
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

def generate_experiment_folder_name(config):
    parts = []
    for key in CORE_KEYS:
        if key in config:
            abbr_key = ABBREVIATIONS.get(key, sanitize(key))
            value_str = stringify_value(config[key])
            sanitized_value = sanitize(value_str)
            parts.append(f"{abbr_key}{sanitized_value}")
    # Mimic LSTM_experiment.py: folder name with lower-case "exp_"
    folder_name = "exp_" + "_".join(parts)
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
        print(f"[DEBUG] Created subdirectory: {key} -> {path}")
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
    nimu = config.get("num_imu_instances", "NA")
    nl = config.get("num_layers", "NA")
    raw_run_id = f"{ABBREVIATIONS.get('num_past_beam_instances','npbi')}{npbi}_" \
                 f"{ABBREVIATIONS.get('num_imu_instances','nimu')}{nimu}_" \
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

def load_csv_files(traj_path):
    beams_gt_path = os.path.join(traj_path, "beams_gt.csv")
    beams_gt = pd.read_csv(beams_gt_path, na_values=[''])
    # In LSTM_experiment.py, beams_training is simply a copy of beams_gt.
    beams_training = beams_gt.copy()
    imu_files = [f for f in os.listdir(traj_path) if f.startswith("IMU_") and f.endswith(".csv")]
    if not imu_files:
        raise ValueError(f"No IMU file found in {traj_path}")
    imu = pd.read_csv(os.path.join(traj_path, imu_files[0]))
    beams_gt.sort_values("Time", inplace=True)
    beams_training.sort_values("Time", inplace=True)
    imu.sort_values("Time [s]", inplace=True)
    beams_gt.reset_index(drop=True, inplace=True)
    beams_training.reset_index(drop=True, inplace=True)
    imu.reset_index(drop=True, inplace=True)
    log_global(f"DEBUG: beams_training length: {len(beams_training)}, beams_gt length: {len(beams_gt)}, imu length: {len(imu)}")
    return beams_gt, beams_training, imu

##########################
# New Functions for Missing Data
##########################

def apply_random_removal(beams_training, config):
    """
    Applies random removal of beam values based on beam-specific probabilities.
    Random removal now starts at index = beam_fill_window.
    """
    probs = config.get("missing_beam_probability", {"b1": 0.0, "b2": 0.0, "b3": 0.0, "b4": 0.0})
    start_idx = config.get("beam_fill_window", 3)
    for idx in range(start_idx, beams_training.shape[0]):
        for beam in ['b1', 'b2', 'b3', 'b4']:
            if np.random.rand() < probs.get(beam, 0.0):
                beams_training.loc[idx, beam] = np.nan
    print(f"[DEBUG] Applied random removal from row {start_idx} onward.")
    return beams_training

def compute_missing_frequencies(config):
    """
    Computes the missing percentage string directly from the JSON.
    For example, if b1:0.2, b2:0.15, b3:0.1, b4:0.05 then returns "b1_20-b2_15-b3_10-b4_5".
    """
    probs = config.get("missing_beam_probability", {})
    missing_freq_str = "-".join([f"{beam}_{int(round(probs.get(beam, 0)*100))}" for beam in sorted(probs.keys())])
    print(f"[DEBUG] Missing percentage string from JSON: {missing_freq_str}")
    return missing_freq_str

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

def construct_input_target(filled_beams, beams_gt, imu, t, num_past_beam_instances, num_imu_instances):
    if t < num_past_beam_instances or t < (num_imu_instances - 1):
        raise ValueError(f"Index {t} does not have enough history.")
    try:
        current_beams = filled_beams.loc[t, ["b1", "b2", "b3", "b4"]].values.astype(float)
    except Exception as e:
        log_global(f"ERROR at index {t} fetching current beams: {e}")
        raise e
    past_beams = []
    for i in range(1, num_past_beam_instances + 1):
        idx = t - i
        if idx < 0:
            raise ValueError(f"Index {t} results in negative history index.")
        try:
            past_row = filled_beams.loc[idx, ["b1", "b2", "b3", "b4"]].values.astype(float)
            past_beams.extend(past_row)
        except Exception as e:
            log_global(f"ERROR at index {idx} fetching past beams: {e}")
            raise e
    imu_cols = ['ACC X [m/s^2]', 'ACC Y [m/s^2]', 'ACC Z [m/s^2]',
                'GYRO X [rad/s]', 'GYRO Y [rad/s]', 'GYRO Z [rad/s]']
    for col in imu_cols:
        if col not in imu.columns:
            log_global(f"ERROR: Expected IMU column {col} not found.")
            raise ValueError(f"IMU column {col} not found.")
    past_imu = []
    for i in range(num_imu_instances - 1, -1, -1):
        idx = t - i
        if idx < 0:
            raise ValueError(f"Index {t} results in negative IMU index.")
        try:
            imu_row = imu.loc[idx, imu_cols].values.astype(float)
            past_imu.extend(imu_row)
        except Exception as e:
            log_global(f"ERROR at index {idx} fetching IMU data: {e}")
            raise e
    input_vector = np.concatenate([current_beams, np.array(past_beams), np.array(past_imu)])
    try:
        target_vector = beams_gt.loc[t, ["b1", "b2", "b3", "b4"]].values.astype(float)
    except Exception as e:
        log_global(f"ERROR at index {t} fetching target: {e}")
        raise e
    return input_vector, target_vector

def get_missing_mask(beams_training, t, target_cols=["b1", "b2", "b3", "b4"]):
    row = beams_training.loc[t, target_cols]
    return row.isna().values

def plot_velocity_predictions(predictions, traj, beam_fill_window, title_suffix=""):
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    samples = [pred["Sample"] for pred in predictions]
    for i, beam in enumerate(["b1", "b2", "b3", "b4"]):
        pred_vals = [pred[f"Pred_{beam}"] for pred in predictions]
        gt_vals = [pred[f"GT_{beam}"] for pred in predictions]
        gt_series = pd.Series(gt_vals)
        moving_avg = gt_series.rolling(window=beam_fill_window, min_periods=1).mean().tolist()
        print(f"[DEBUG Plot] Beam {beam}: First 5 Predicted: {pred_vals[:5]}")
        print(f"[DEBUG Plot] Beam {beam}: First 5 Ground Truth: {gt_vals[:5]}")
        print(f"[DEBUG Plot] Beam {beam}: First 5 Moving Avg: {moving_avg[:5]}")
        axes[i].plot(samples, pred_vals, label=f"Predicted {beam}", marker='o')
        axes[i].plot(samples, gt_vals, label=f"Ground Truth {beam}", marker='x')
        axes[i].plot(samples, moving_avg, label=f"Moving Avg {beam}", linewidth=2, color='purple')
        axes[i].set_ylabel("Value")
        axes[i].legend(loc="upper right")
        axes[i].grid(True)
    axes[-1].set_xlabel("Sample Index")
    fig.suptitle(f"Predicted vs Ground Truth & Moving Avg for {traj} {title_suffix}")
    return fig

def plot_missing_beams_frequency(beams_training, traj, run_id):
    probs = config.get("missing_beam_probability", {})
    missing_percentages = {beam: int(round(probs.get(beam, 0)*100)) for beam in sorted(probs.keys())}
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(missing_percentages.keys(), missing_percentages.values(), color='salmon')
    ax.set_xlabel("Beam")
    ax.set_ylabel("Missing Percentage")
    ax.set_title(f"Missing Beams Percentage for {traj}")
    plot_path = os.path.join(PLOTS_DIR, f"MissingBeamsPercentage_{sanitize(traj)}_{run_id}.png")
    fig.savefig(plot_path)
    plt.close(fig)
    log_global(f"[{traj}] Missing beams percentage plot saved to {plot_path}")

def find_first_valid_index(filled_beams, beams_gt, imu, num_past_beam_instances, num_imu_instances):
    start = max(num_past_beam_instances, num_imu_instances - 1)
    log_global(f"Starting search from index {start}")
    for t in range(start, len(filled_beams)):
        try:
            inp, tar = construct_input_target(filled_beams, beams_gt, imu, t, num_past_beam_instances, num_imu_instances)
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
    probs = config.get("missing_beam_probability", {})
    missing_freq_str = "-".join([f"{beam}_{int(round(probs.get(beam, 0)*100))}" for beam in sorted(probs.keys())])
    log_global(f"[DEBUG] Missing percentage string: {missing_freq_str}")
    
    for traj_pair in training_trajectory_pairs:
        traj, traj_epochs = traj_pair
        traj_path = os.path.join(DATA_DIR, traj)
        log_global(f"=== Training on Trajectory: {traj} for {traj_epochs} epochs ===")
        try:
            beams_gt, beams_training, imu = load_csv_files(traj_path)
        except Exception as e:
            log_global(f"Error loading files in {traj}: {e}")
            continue

        beams_training = apply_random_removal(beams_training, config)
        orig_missing = beams_training[["b1", "b2", "b3", "b4"]].isna().sum().sum()
        log_global(f"Original missing count in {traj}: {orig_missing}")
        if orig_missing == 0:
            log_global(f"[Train] No missing beams in {traj}. Skipping training on this trajectory.")
            continue
        filled_beams, _ = fill_missing_beams(beams_training, config["beam_fill_window"])
        min_history = find_first_valid_index(filled_beams, beams_gt, imu, config["num_past_beam_instances"], config["num_imu_instances"])
        if min_history is None:
            log_global("Not enough training data in the trajectory to determine input size.")
            continue
        log_global(f"Using starting index {min_history} for constructing input-target pairs.")
        
        inputs, targets, masks = [], [], []
        for t in range(min_history, len(filled_beams)):
            try:
                inp, tar = construct_input_target(filled_beams, beams_gt, imu, t, config["num_past_beam_instances"], config["num_imu_instances"])
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
            epoch_squared_errors = np.zeros(4)
            for i in range(num_samples):
                model.train()
                # For TCN, unsqueeze only once.
                x = torch.tensor(inputs[i], dtype=torch.float32, device=model.device).unsqueeze(0)
                y = torch.tensor(targets[i], dtype=torch.float32, device=model.device)
                y_pred = model(x).squeeze().view(-1)
                y = y.view(-1)
                mask = get_missing_mask(beams_training, i, target_cols=["b1", "b2", "b3", "b4"])
                sample_loss = loss_fn(y_pred, y)
                losses.append(sample_loss)
                error = (y_pred - y).detach().cpu().numpy() ** 2
                epoch_squared_errors += error
            epoch_loss = torch.stack(losses).sum()
            epoch_loss.backward()
            optimizer.step()
            epoch_rmse = np.sqrt(epoch_squared_errors / num_samples)
            evolution.append([epoch] + epoch_rmse.tolist())
            log_global(f"[{traj}] Epoch {epoch}: RMSE per beam: {epoch_rmse}, Avg RMSE: {np.mean(epoch_rmse):.5f}")
        training_time = time.time() - t0
        
        model.eval()
        final_predictions = []
        for i in range(num_samples):
            x = torch.tensor(inputs[i], dtype=torch.float32, device=model.device).unsqueeze(0)
            y = torch.tensor(targets[i], dtype=torch.float32, device=model.device)
            with torch.no_grad():
                y_pred = model(x).squeeze().view(-1)
            y = y.view(-1)
            final_predictions.append({
                "Sample": i,
                "Pred_b1": y_pred[0].item(),
                "Pred_b2": y_pred[1].item(),
                "Pred_b3": y_pred[2].item(),
                "Pred_b4": y_pred[3].item(),
                "GT_b1": y[0].item(),
                "GT_b2": y[1].item(),
                "GT_b3": y[2].item(),
                "GT_b4": y[3].item()
            })
        print(f"[DEBUG] Final predictions for {traj} (first 5 samples): {final_predictions[:5]}")
        train_plot_path = os.path.join(PLOTS_DIR, f"FinalOutputPredictions_{sanitize(traj)}_{run_id}_{missing_freq_str}_final.png")
        train_fig = plot_velocity_predictions(final_predictions, traj, config["beam_fill_window"], title_suffix="(Training)")
        train_fig.savefig(train_plot_path)
        plt.close(train_fig)
        log_global(f"[{traj}] Training predictions plot saved to {train_plot_path}")
        
        current_trained_on = ", ".join(trained_list) if trained_list else "NONE"
        summary = {
            "Trajectory": traj,
            "NumSamples": num_samples,
            "InputSize": input_size,
            "EpochsTrained": traj_epochs,
            "AvgBestRMSE": np.mean(evolution[-1][1:]),
            "TrainingTime": training_time,
            "TrainedOn": current_trained_on,
            "MissingPercentage": missing_freq_str
        }
        global_training_summary.append(summary)
        processed_training_info.append(f"{traj}:{traj_epochs}")
        if f"{traj}:{traj_epochs}" not in trained_list:
            trained_list.append(f"{traj}:{traj_epochs}")
    return global_training_summary, processed_training_info

def test_on_trajectory(traj, config, checkpoint_filename, run_id, base_trained_on, missing_freq_str):
    traj_path = os.path.join(DATA_DIR, traj)
    beams_gt, beams_training, imu = load_csv_files(traj_path)
    # For testing, do not apply random removal.
    filled_beams, _ = fill_missing_beams(beams_training, config["beam_fill_window"])
    min_history = max(_, config["num_past_beam_instances"], config["num_imu_instances"] - 1)
    inputs, targets, masks = [], [], []
    for t in range(min_history, len(filled_beams)):
        try:
            inp, tar = construct_input_target(filled_beams, beams_gt, imu, t, config["num_past_beam_instances"], config["num_imu_instances"])
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
    
    model = TCN(num_inputs=input_size,
                num_channels=config["hidden_neurons"],
                kernel_size=config["kernel_size"],
                dropout=config["dropout_rate"],
                num_layers=config["num_layers"],
                dilation_base=config["dilation_base"],
                num_output_neurons=4)
    cp_full_path = os.path.join(CHECKPOINTS_DIR, checkpoint_filename)
    try:
        state = torch.load(cp_full_path, map_location=model.device, weights_only=True)
    except Exception as e:
        log_global(f"[Test] Error loading checkpoint for {traj}: {e}")
        return None
    model.load_state_dict(state)
    model.eval()
    
    predictions = []
    squared_errors = np.zeros(4)
    for i in range(num_samples):
        x = torch.tensor(inputs[i], dtype=torch.float32, device=model.device).unsqueeze(0)
        y = torch.tensor(targets[i], dtype=torch.float32, device=model.device)
        with torch.no_grad():
            y_pred = model(x).squeeze().view(-1)
        predictions.append({
            "Sample": i,
            "Pred_b1": y_pred[0].item(),
            "Pred_b2": y_pred[1].item(),
            "Pred_b3": y_pred[2].item(),
            "Pred_b4": y_pred[3].item(),
            "GT_b1": y[0].item(),
            "GT_b2": y[1].item(),
            "GT_b3": y[2].item(),
            "GT_b4": y[3].item()
        })
        err = (y_pred - y.view(-1)).detach().cpu().numpy() ** 2
        squared_errors += err
    test_rmse = np.sqrt(squared_errors / num_samples)
    log_global(f"[{traj}] Test RMSE per beam: {test_rmse}")
    
    test_pred_csv = os.path.join(PREDICTIONS_DIR, f"TestPredictions_{sanitize(traj)}_{run_id}_{missing_freq_str}.csv")
    pd.DataFrame(predictions).to_csv(test_pred_csv, index=False)
    log_global(f"[{traj}] Test predictions saved to {test_pred_csv}")
    
    plot_file = os.path.join(PLOTS_DIR, f"VelocityPredictions_{sanitize(traj)}_{run_id}_{missing_freq_str}.png")
    fig = plot_velocity_predictions(predictions, traj, config["beam_fill_window"], title_suffix="(Testing Data)")
    fig.savefig(plot_file)
    plt.close(fig)
    log_global(f"[{traj}] Velocity predictions plot saved to {plot_file}")
    
    test_summary = {
        "Trajectory": traj,
        "NumSamples": num_samples,
        "Test_RMSE_b1": test_rmse[0],
        "Test_RMSE_b2": test_rmse[1],
        "Test_RMSE_b3": test_rmse[2],
        "Test_RMSE_b4": test_rmse[3],
        "AvgTest_RMSE": np.mean(test_rmse),
        "TrainedOn": base_trained_on,
        "MissingPercentage": missing_freq_str
    }
    return test_summary

##########################
# Main Routine
##########################

def main():
    with open(GLOBAL_LOG_FILE, "w") as f:
        f.write("Experiment Log\n")
    
    with open("TCN.json", "r") as f:
        config = json.load(f)
    
    training_trajectory_pairs = config.get("training_trajectories", [])
    testing_list = config.get("testing_trajectories", [])
    
    global_training_summary = []
    cumulative_trained_on = []  # e.g., ["Trajectory1:30", "Trajectory2:40", ...]
    processed_training_info = []
    
    if not training_trajectory_pairs:
        log_global("No training trajectories provided.")
        return
    first_traj = training_trajectory_pairs[0][0]
    first_traj_path = os.path.join(DATA_DIR, first_traj)
    beams_gt, beams_training, imu = load_csv_files(first_traj_path)
    filled_beams, _ = fill_missing_beams(beams_training, config["beam_fill_window"])
    min_history = find_first_valid_index(filled_beams, beams_gt, imu, config["num_past_beam_instances"], config["num_imu_instances"])
    if min_history is None:
        log_global("Not enough training data in the first trajectory to determine input size.")
        return
    log_global(f"Using starting index {min_history} from the first trajectory.")
    
    inputs = []
    for t in range(min_history, len(filled_beams)):
        try:
            inp, _ = construct_input_target(filled_beams, beams_gt, imu, t, config["num_past_beam_instances"], config["num_imu_instances"])
            inputs.append(inp)
            log_global(f"Constructed sample at index {t}.")
        except Exception as e:
            log_global(f"Skipping index {t} in main: {e}")
            continue
    if len(inputs) == 0:
        log_global("Not enough training data in the first trajectory to determine input size.")
        return
    input_size = np.array(inputs).shape[1]
    model = TCN(num_inputs=input_size,
                num_channels=config["hidden_neurons"],
                kernel_size=config["kernel_size"],
                dropout=config["dropout_rate"],
                num_layers=config["num_layers"],
                dilation_base=config["dilation_base"],
                num_output_neurons=4)
    
    # Compute missing percentage string from JSON
    missing_freq_str = compute_missing_frequencies(config)
    
    log_global("=== Sequential Training Phase ===")
    for traj_pair in training_trajectory_pairs:
        traj, traj_epochs = traj_pair
        log_global(f"Processing training trajectory: {traj} for {traj_epochs} epochs")
        current_trained_on = ", ".join(cumulative_trained_on) if cumulative_trained_on else "NONE"
        summary, proc_info = sequential_train([traj_pair], config, model, run_id=generate_run_specific_identifier(config), trained_list=cumulative_trained_on)
        if summary:
            summary[0]["TrainedOn"] = current_trained_on
            global_training_summary.append(summary[0])
        processed_training_info.extend([f"{traj}:{traj_epochs}"])
        if f"{traj}:{traj_epochs}" not in cumulative_trained_on:
            cumulative_trained_on.append(f"{traj}:{traj_epochs}")
    
    if not global_training_summary:
        log_global("No training trajectories processed; aborting.")
        return
    
    final_checkpoint_filename = f"TCN_{'_'.join(cumulative_trained_on)}_lr{config['learning_rate']}_dr{config['dropout_rate']}_hn{config['hidden_neurons']}_nl{config['num_layers']}_ks{config['kernel_size']}_db{config['dilation_base']}_bfw{config['beam_fill_window']}_npbi{config['num_past_beam_instances']}_nimu{config['num_imu_instances']}_pr{str(config['partial_rmse'])}_{missing_freq_str}_final.pth"
    final_checkpoint_path = os.path.join(CHECKPOINTS_DIR, final_checkpoint_filename)
    check_duplicate_file(CHECKPOINTS_DIR, final_checkpoint_filename)
    torch.save(model.state_dict(), final_checkpoint_path)
    log_global(f"Final checkpoint saved to {final_checkpoint_path}")
    
    train_summary_filename = f"GlobalTrainingSummary_{final_checkpoint_filename[:-4]}.csv"
    ts_path = check_duplicate_file(TRAINING_SUMMARIES_DIR, train_summary_filename)
    pd.DataFrame(global_training_summary).to_csv(ts_path, index=False)
    log_global(f"Global training summary saved to {ts_path}")
    
    log_global("=== Testing Phase ===")
    global_test_summary = []
    base_trained_on = ", ".join(dict.fromkeys(cumulative_trained_on).keys()) if cumulative_trained_on else "NONE"
    for traj in testing_list:
        log_global(f"Processing testing trajectory: {traj}")
        test_summary = test_on_trajectory(traj, config, final_checkpoint_filename, 
                                          run_id=generate_run_specific_identifier(config), 
                                          base_trained_on=base_trained_on, 
                                          missing_freq_str=missing_freq_str)
        if test_summary:
            global_test_summary.append(test_summary)
    if global_test_summary:
        test_summary_filename = f"GlobalTestSummary_{final_checkpoint_filename[:-4]}.csv"
        tst_path = check_duplicate_file(TEST_SUMMARIES_DIR, test_summary_filename)
        pd.DataFrame(global_test_summary).to_csv(tst_path, index=False)
        log_global(f"Global test summary saved to {tst_path}")

if __name__ == "__main__":
    with open("TCN.json", "r") as f:
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
