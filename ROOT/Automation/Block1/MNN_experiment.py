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
# Global Results Root
##########################
RESULTS_ROOT = "Results/MNN_Results"  # Root folder for MNN experiments

ABBREVIATIONS = {
    "learning_rate": "lr",
    "dropout_rate": "dr",
    "hidden_neurons": "hn",
    "regularization": "reg",
    "beam_fill_window": "bfw",
    # Run-specific parameters:
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

#######################################
# Generate Top-Level Folder Name
#######################################
def generate_top_level_folder_name(config):
    # Build training part: for each training trajectory, extract digits from name and append epoch.
    train_trajs = config.get("training_trajectories", [])
    train_parts = []
    for traj_pair in train_trajs:
        traj_name, epochs = traj_pair
        digits = re.findall(r'\d+', traj_name)
        train_parts.append("_".join(digits) + f"_{epochs}")
    train_part = "TrTj_" + "_".join(train_parts) if train_parts else ""
    
    # Build testing part: for each testing trajectory, extract digits.
    test_trajs = config.get("testing_trajectories", [])
    test_parts = []
    for traj in test_trajs:
        digits = re.findall(r'\d+', traj)
        test_parts.append("_".join(digits))
    test_part = "TTj_" + "_".join(test_parts) if test_parts else ""
    
    # Concatenate both parts with an underscore between if both exist.
    folder_name = f"{train_part}_{test_part}" if train_part and test_part else train_part or test_part
    folder_name = sanitize(folder_name)
    print(f"[DEBUG] Top-level folder name: {folder_name}")
    return folder_name

#######################################
# Generate Second-Level Folder Name
#######################################
def generate_second_level_folder_name(config):
    # Use beam_fill_window, num_past_beam_instances, num_imu_instances
    bfw = config.get("beam_fill_window")
    npbi = config.get("num_past_beam_instances")
    niu = config.get("num_imu_instances")
    folder_name = f"bfw_{bfw}_npbi_{npbi}_niu_{niu}"
    folder_name = sanitize(folder_name)
    print(f"[DEBUG] Second-level folder name: {folder_name}")
    return folder_name

#######################################
# Create Full Experiment Folder Structure
#######################################
def create_experiment_folders(config):
    top_level_folder = os.path.join(RESULTS_ROOT, generate_top_level_folder_name(config))
    second_level_folder = os.path.join(top_level_folder, generate_second_level_folder_name(config))
    os.makedirs(second_level_folder, exist_ok=True)
    
    # Create sub-folders
    subfolders = {
        "Checkpoints": os.path.join(second_level_folder, "Checkpoints"),
        "Plots": os.path.join(second_level_folder, "Plots"),
        "Predictions": os.path.join(second_level_folder, "Predictions"),
        "TestSummaries": os.path.join(second_level_folder, "Test Summaries"),
        "TrainingSummaries": os.path.join(second_level_folder, "Training Summaries")
    }
    for folder in subfolders.values():
        os.makedirs(folder, exist_ok=True)
    
    # Create README file with configuration details that are not included in folder names.
    keys_in_folder = ["training_trajectories", "testing_trajectories", "beam_fill_window", "num_past_beam_instances", "num_imu_instances"]
    config_for_readme = {k: v for k, v in config.items() if k not in keys_in_folder}
    readme_content = "Training Session Configuration:\n"
    for key, value in config_for_readme.items():
        readme_content += f"{key}: {value}\n"
    readme_path = os.path.join(second_level_folder, "README.txt")
    with open(readme_path, "w") as f:
        f.write(readme_content)
    
    # Create an empty experimental global log file.
    global_log_path = os.path.join(second_level_folder, "experimental_global_log.txt")
    with open(global_log_path, "w") as f:
        f.write("Experiment Log\n")
    
    print(f"[DEBUG] Created experiment folder structure in: {second_level_folder}")
    return second_level_folder, subfolders, global_log_path

#######################################
# Logging Function
#######################################
def log_global(message, log_file):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(message)

#######################################
# Data Loading & Preprocessing Functions
#######################################
DATA_DIR = "../../Data"

def load_csv_files(traj_path):
    beams_gt_path = os.path.join(traj_path, "beams_gt.csv")
    beams_gt = pd.read_csv(beams_gt_path, na_values=[''])
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
    log_global(f"DEBUG: beams_training length: {len(beams_training)}, beams_gt length: {len(beams_gt)}, imu length: {len(imu)}", GLOBAL_LOG_FILE)
    return beams_gt, beams_training, imu

def apply_random_removal(beams_training, config):
    probs = config.get("missing_beam_probability", {"b1": 0.0, "b2": 0.0, "b3": 0.0, "b4": 0.0})
    start_idx = config.get("beam_fill_window", 40)
    for idx in range(start_idx, beams_training.shape[0]):
        for beam in ['b1', 'b2', 'b3', 'b4']:
            if np.random.rand() < probs.get(beam, 0.0):
                beams_training.loc[idx, beam] = np.nan
    log_global(f"Applied random removal from row {start_idx} onward.", GLOBAL_LOG_FILE)
    return beams_training

def fill_missing_beams(beams_df, beam_fill_window, beam_cols=["b1", "b2", "b3", "b4"]):
    filled = beams_df.copy()
    for i in range(beam_fill_window, len(filled)):
        for col in beam_cols:
            if pd.isna(filled.loc[i, col]):
                window = filled.loc[i - beam_fill_window:i - 1, col]
                if window.isna().all():
                    last_val = filled[col].ffill().iloc[i - 1]
                    log_global(f"All previous values missing for {col} at row {i}; using last valid value {last_val}", GLOBAL_LOG_FILE)
                    filled.loc[i, col] = last_val
                else:
                    avg_val = window.mean()
                    log_global(f"Filling missing {col} at row {i} with moving average {avg_val}", GLOBAL_LOG_FILE)
                    filled.loc[i, col] = avg_val
    return filled, beam_fill_window

def construct_input_target(filled_beams, beams_gt, imu, t, num_past_beam_instances, num_imu_instances):
    if t < num_past_beam_instances or t < (num_imu_instances - 1):
        raise ValueError(f"Index {t} does not have enough history.")
    try:
        current_beams = filled_beams.loc[t, ["b1", "b2", "b3", "b4"]].values.astype(float)
    except Exception as e:
        log_global(f"ERROR at index {t} fetching current beams: {e}", GLOBAL_LOG_FILE)
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
            log_global(f"ERROR at index {idx} fetching past beams: {e}", GLOBAL_LOG_FILE)
            raise e
    imu_cols = ['ACC X [m/s^2]', 'ACC Y [m/s^2]', 'ACC Z [m/s^2]',
                'GYRO X [rad/s]', 'GYRO Y [rad/s]', 'GYRO Z [rad/s]']
    for col in imu_cols:
        if col not in imu.columns:
            log_global(f"ERROR: Expected IMU column {col} not found.", GLOBAL_LOG_FILE)
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
            log_global(f"ERROR at index {idx} fetching IMU data: {e}", GLOBAL_LOG_FILE)
            raise e
    input_vector = np.concatenate([current_beams, np.array(past_beams), np.array(past_imu)])
    try:
        target_vector = beams_gt.loc[t, ["b1", "b2", "b3", "b4"]].values.astype(float)
    except Exception as e:
        log_global(f"ERROR at index {t} fetching target: {e}", GLOBAL_LOG_FILE)
        raise e
    return input_vector, target_vector

def get_missing_mask(beams_training, t, target_cols=["b1", "b2", "b3", "b4"]):
    row = beams_training.loc[t, target_cols]
    return row.isna().values

#######################################
# Plotting Functions
#######################################
def plot_velocity_predictions(predictions, traj, beam_fill_window, title_suffix="", clip_ground_truth=False):
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    samples = [pred["Sample"] for pred in predictions]
    for i, beam in enumerate(["b1", "b2", "b3", "b4"]):
        pred_vals = [pred[f"Pred_{beam}"] for pred in predictions]
        gt_vals = [pred[f"GT_{beam}"] for pred in predictions]
        if clip_ground_truth:
            gt_vals = [(-5 if g <= -5 else g) for g in gt_vals]
        gt_series = pd.Series(gt_vals)
        moving_avg = gt_series.rolling(window=beam_fill_window, min_periods=1).mean().tolist()
        axes[i].plot(samples, gt_vals, label=f"Ground Truth {beam}", linestyle="-")
        axes[i].plot(samples, pred_vals, label=f"Predicted {beam}", linestyle="-")
        axes[i].plot(samples, moving_avg, label=f"Moving Avg {beam}", linewidth=2, color='purple')
        axes[i].set_ylabel("Value")
        axes[i].legend(loc="upper right")
        axes[i].grid(True)
    axes[-1].set_xlabel("Sample Index")
    fig.suptitle(f"Predicted vs Ground Truth & Moving Avg for {traj} {title_suffix}")
    return fig

def plot_square_error(predictions, traj, beam_fill_window, title_suffix="", clip_ground_truth=False):
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    samples = [pred["Sample"] for pred in predictions]
    for i, beam in enumerate(["b1", "b2", "b3", "b4"]):
        gt_vals = [pred[f"GT_{beam}"] for pred in predictions]
        if clip_ground_truth:
            gt_vals = [(-5 if g <= -5 else g) for g in gt_vals]
        pred_vals = [pred[f"Pred_{beam}"] for pred in predictions]
        gt_series = pd.Series(gt_vals)
        moving_avg = gt_series.rolling(window=beam_fill_window, min_periods=1).mean().tolist()
        sq_error_pred = [(p - g) ** 2 for p, g in zip(pred_vals, gt_vals)]
        sq_error_moving = [(m - g) ** 2 for m, g in zip(moving_avg, gt_vals)]
        axes[i].plot(samples, sq_error_pred, label="Squared Error (Pred vs GT)", linestyle="-")
        axes[i].plot(samples, sq_error_moving, label="Squared Error (Moving Avg vs GT)", linestyle="-")
        axes[i].set_ylabel("Squared Error")
        axes[i].legend(loc="upper right")
        axes[i].grid(True)
    axes[-1].set_xlabel("Sample Index")
    fig.suptitle(f"Square Error for {traj} {title_suffix}")
    return fig

#######################################
# Finding the Starting Index
#######################################
def find_first_valid_index(filled_beams, beams_gt, imu, num_past_beam_instances, num_imu_instances):
    start = max(num_past_beam_instances, num_imu_instances - 1)
    log_global(f"Starting search from index {start}", GLOBAL_LOG_FILE)
    for t in range(start, len(filled_beams)):
        try:
            inp, tar = construct_input_target(filled_beams, beams_gt, imu, t, num_past_beam_instances, num_imu_instances)
            log_global(f"Valid input-target pair found at index {t}.", GLOBAL_LOG_FILE)
            return t
        except Exception as e:
            log_global(f"Index {t} invalid: {e}", GLOBAL_LOG_FILE)
            continue
    return None

#######################################
# Sequential Training Routine
#######################################
def sequential_train(training_trajectory_pairs, config, model, run_id, log_file):
    global_training_summary = []
    probs = config.get("missing_beam_probability", {})
    missing_freq_str = "-".join([f"{beam}_{int(round(probs.get(beam, 0)*100))}" for beam in sorted(probs.keys())])
    log_global(f"[DEBUG] Missing percentage string: {missing_freq_str}", log_file)
    
    for traj_pair in training_trajectory_pairs:
        traj, traj_epochs = traj_pair
        traj_path = os.path.join(DATA_DIR, traj)
        log_global(f"=== Training on Trajectory: {traj} for {traj_epochs} epochs ===", log_file)
        try:
            beams_gt, beams_training, imu = load_csv_files(traj_path)
        except Exception as e:
            log_global(f"Error loading files in {traj}: {e}", log_file)
            continue

        beams_training = apply_random_removal(beams_training, config)
        orig_missing = beams_training[["b1", "b2", "b3", "b4"]].isna().sum().sum()
        log_global(f"Original missing count in {traj}: {orig_missing}", log_file)
        if orig_missing == 0:
            log_global(f"[Train] No missing beams in {traj}. Skipping training on this trajectory.", log_file)
            continue
        filled_beams, _ = fill_missing_beams(beams_training, config["beam_fill_window"])
        min_history = find_first_valid_index(filled_beams, beams_gt, imu, config["num_past_beam_instances"], config["num_imu_instances"])
        if min_history is None:
            log_global("Not enough training data in the trajectory to determine input size.", log_file)
            continue
        log_global(f"Using starting index {min_history} for constructing input-target pairs.", log_file)
        
        inputs, targets = [], []
        for t in range(min_history, len(filled_beams)):
            try:
                inp, tar = construct_input_target(filled_beams, beams_gt, imu, t, config["num_past_beam_instances"], config["num_imu_instances"])
                inputs.append(inp)
                targets.append(tar)
                log_global(f"Constructed sample at index {t}.", log_file)
            except Exception as e:
                log_global(f"Skipping index {t}: {e}", log_file)
                continue
        if len(inputs) == 0:
            log_global("Not enough training data in the trajectory to determine input size.", log_file)
            continue
        inputs = np.array(inputs)
        targets = np.array(targets)
        num_samples = len(inputs)
        input_size = inputs.shape[1]
        log_global(f"[{traj}] Training Samples: {num_samples}, Input size: {input_size}", log_file)
        
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
            log_global(f"[{traj}] Epoch {epoch}: RMSE per output: {epoch_rmse}, Avg RMSE: {np.mean(epoch_rmse):.5f}", log_file)
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
                "Pred_b1": y_pred[0].item(),
                "Pred_b2": y_pred[1].item(),
                "Pred_b3": y_pred[2].item(),
                "Pred_b4": y_pred[3].item(),
                "GT_b1": y[0].item(),
                "GT_b2": y[1].item(),
                "GT_b3": y[2].item(),
                "GT_b4": y[3].item()
            })
        train_plot_path = os.path.join(SUBFOLDERS["Checkpoints"], f"FinalOutputPredictions_{sanitize(traj)}_{run_id}.png")
        train_fig = plot_velocity_predictions(final_predictions, traj, config["beam_fill_window"], title_suffix="(Training)", clip_ground_truth=False)
        train_fig.savefig(train_plot_path)
        plt.close(train_fig)
        log_global(f"[{traj}] Training predictions plot saved to {train_plot_path}", log_file)

        error_plot_path = os.path.join(SUBFOLDERS["Plots"], f"SquareError_{sanitize(traj)}_{run_id}.png")
        square_error_fig = plot_square_error(final_predictions, traj, config["beam_fill_window"], title_suffix="(Training)", clip_ground_truth=False)
        square_error_fig.savefig(error_plot_path)
        plt.close(square_error_fig)
        log_global(f"[{traj}] Square error plot saved to {error_plot_path}", log_file)

        summary = {
            "Trajectory": traj,
            "NumSamples": num_samples,
            "InputSize": input_size,
            "EpochsTrained": traj_epochs,
            "AvgBestRMSE": np.mean(evolution[-1][1:]),
            "TrainingTime": training_time
        }
        global_training_summary.append(summary)
    return global_training_summary

#######################################
# Testing Routine
#######################################
def test_on_trajectory(traj, config, checkpoint_filename, run_id, base_trained_on, log_file):
    traj_path = os.path.join(DATA_DIR, traj)
    beams_gt, beams_training, imu = load_csv_files(traj_path)
    filled_beams, _ = fill_missing_beams(beams_training, config["beam_fill_window"])
    min_history = max(_, config["num_past_beam_instances"], config["num_imu_instances"] - 1)
    inputs, targets = [], []
    for t in range(min_history, len(filled_beams)):
        try:
            inp, tar = construct_input_target(filled_beams, beams_gt, imu, t, config["num_past_beam_instances"], config["num_imu_instances"])
            inputs.append(inp)
            targets.append(tar)
        except Exception as e:
            log_global(f"Skipping index {t} in testing: {e}", log_file)
            continue
    if len(inputs) == 0:
        log_global(f"[Test] Not enough test data in {traj} after history constraints. Skipping.", log_file)
        return None
    inputs = np.array(inputs)
    targets = np.array(targets)
    num_samples = len(inputs)
    input_size = inputs.shape[1]
    log_global(f"[{traj}] Testing Samples: {num_samples}, Input size: {input_size}", log_file)
    
    model = MemoryNeuralNetwork(number_of_input_neurons=input_size,
                                number_of_hidden_neurons=config["hidden_neurons"],
                                number_of_output_neurons=4,
                                dropout_rate=config["dropout_rate"])
    cp_full_path = os.path.join(SUBFOLDERS["Checkpoints"], checkpoint_filename)
    try:
        state = torch.load(cp_full_path, map_location=model.device, weights_only=True)
    except Exception as e:
        log_global(f"[Test] Error loading checkpoint for {traj}: {e}", log_file)
        return None
    model.load_state_dict(state)
    model.eval()
    
    predictions = []
    squared_errors = np.zeros(4)
    for i in range(num_samples):
        x = torch.tensor(inputs[i], dtype=torch.float32, device=model.device).unsqueeze(0).unsqueeze(0)
        y = torch.tensor(targets[i], dtype=torch.float32, device=model.device)
        with torch.no_grad():
            y_pred = model(x).squeeze().view(-1)
        pred_dict = {
            "Sample": i,
            "Pred_b1": y_pred[0].item(),
            "Pred_b2": y_pred[1].item(),
            "Pred_b3": y_pred[2].item(),
            "Pred_b4": y_pred[3].item(),
            "GT_b1": y[0].item(),
            "GT_b2": y[1].item(),
            "GT_b3": y[2].item(),
            "GT_b4": y[3].item()
        }
        if (pred_dict["GT_b1"] > -5 and pred_dict["GT_b2"] > -5 and 
            pred_dict["GT_b3"] > -5 and pred_dict["GT_b4"] > -5):
            pred_dict["All_Beams_Gt"] = 1
        else:
            pred_dict["All_Beams_Gt"] = 0
        predictions.append(pred_dict)
        err = (y_pred - y.view(-1)).detach().cpu().numpy() ** 2
        squared_errors += err
    test_rmse = np.sqrt(squared_errors / num_samples)
    log_global(f"[{traj}] Test RMSE per output: {test_rmse}", log_file)
    
    test_pred_csv = os.path.join(SUBFOLDERS["Predictions"], f"TestPredictions_{sanitize(traj)}_{run_id}.csv")
    pd.DataFrame(predictions).to_csv(test_pred_csv, index=False)
    log_global(f"[{traj}] Test predictions saved to {test_pred_csv}", log_file)
    
    plot_file = os.path.join(SUBFOLDERS["Plots"], f"VelocityPredictions_{sanitize(traj)}_{run_id}.png")
    fig = plot_velocity_predictions(predictions, traj, config["beam_fill_window"], title_suffix="(Testing Data)", clip_ground_truth=True)
    fig.savefig(plot_file)
    plt.close(fig)
    log_global(f"[{traj}] Velocity predictions plot saved to {plot_file}", log_file)

    error_plot_path = os.path.join(SUBFOLDERS["Plots"], f"SquareError_{sanitize(traj)}_{run_id}.png")
    square_error_fig = plot_square_error(predictions, traj, config["beam_fill_window"], title_suffix="(Testing Data)", clip_ground_truth=True)
    square_error_fig.savefig(error_plot_path)
    plt.close(square_error_fig)
    log_global(f"[{traj}] Square error plot saved to {error_plot_path}", log_file)
    
    test_summary = {
        "Trajectory": traj,
        "NumSamples": num_samples,
        "Test_RMSE_b1": test_rmse[0],
        "Test_RMSE_b2": test_rmse[1],
        "Test_RMSE_b3": test_rmse[2],
        "Test_RMSE_b4": test_rmse[3],
        "AvgTest_RMSE": np.mean(test_rmse)
    }
    return test_summary

#######################################
# Main Routine
#######################################
def main():
    with open("MNN.json", "r") as f:
        config = json.load(f)
    
    training_trajectory_pairs = config.get("training_trajectories", [])
    testing_list = config.get("testing_trajectories", [])
    
    if not training_trajectory_pairs:
        log_global("No training trajectories provided.", GLOBAL_LOG_FILE)
        return

    # Determine input size from first trajectory.
    first_traj = training_trajectory_pairs[0][0]
    first_traj_path = os.path.join(DATA_DIR, first_traj)
    beams_gt, beams_training, imu = load_csv_files(first_traj_path)
    filled_beams, _ = fill_missing_beams(beams_training, config["beam_fill_window"])
    min_history = find_first_valid_index(filled_beams, beams_gt, imu, config["num_past_beam_instances"], config["num_imu_instances"])
    if min_history is None:
        log_global("Not enough training data in the first trajectory to determine input size.", GLOBAL_LOG_FILE)
        return
    log_global(f"Using starting index {min_history} from the first trajectory.", GLOBAL_LOG_FILE)
    
    inputs = []
    for t in range(min_history, len(filled_beams)):
        try:
            inp, _ = construct_input_target(filled_beams, beams_gt, imu, t, config["num_past_beam_instances"], config["num_imu_instances"])
            inputs.append(inp)
            log_global(f"Constructed sample at index {t}.", GLOBAL_LOG_FILE)
        except Exception as e:
            log_global(f"Skipping index {t} in main: {e}", GLOBAL_LOG_FILE)
            continue
    if len(inputs) == 0:
        log_global("Not enough training data in the first trajectory to determine input size.", GLOBAL_LOG_FILE)
        return
    input_size = np.array(inputs).shape[1]
    
    model = MemoryNeuralNetwork(number_of_input_neurons=input_size,
                                number_of_hidden_neurons=config["hidden_neurons"],
                                number_of_output_neurons=4,
                                dropout_rate=config["dropout_rate"],
                                learning_rate=config["learning_rate"],
                                learning_rate_2=config["learning_rate_2"],
                                lipschitz_constant=config["lipschitz_constant"])
    
    run_id = sanitize(generate_top_level_folder_name(config) + "_" + generate_second_level_folder_name(config))
    
    log_global("=== Sequential Training Phase ===", GLOBAL_LOG_FILE)
    training_summary = sequential_train(training_trajectory_pairs, config, model, run_id, GLOBAL_LOG_FILE)
    
    if not training_summary:
        log_global("No training trajectories processed; aborting.", GLOBAL_LOG_FILE)
        return

    final_checkpoint_filename = f"MNN_{run_id}_final.pth"
    checkpoint_path = os.path.join(SUBFOLDERS["Checkpoints"], final_checkpoint_filename)
    if os.path.exists(checkpoint_path):
        log_global(f"Duplicate file found for configuration: {checkpoint_path}. Aborting experiment run.", GLOBAL_LOG_FILE)
        sys.exit(0)
    torch.save(model.state_dict(), checkpoint_path)
    log_global(f"Final checkpoint saved to {checkpoint_path}", GLOBAL_LOG_FILE)
    
    train_summary_filename = f"GlobalTrainingSummary_{run_id}.csv"
    train_summary_path = os.path.join(SUBFOLDERS["TrainingSummaries"], train_summary_filename)
    if os.path.exists(train_summary_path):
        log_global(f"Duplicate file found for configuration: {train_summary_path}. Aborting experiment run.", GLOBAL_LOG_FILE)
        sys.exit(0)
    pd.DataFrame(training_summary).to_csv(train_summary_path, index=False)
    log_global(f"Global training summary saved to {train_summary_path}", GLOBAL_LOG_FILE)
    
    log_global("=== Testing Phase ===", GLOBAL_LOG_FILE)
    global_test_summary = []
    for traj in testing_list:
        log_global(f"Processing testing trajectory: {traj}", GLOBAL_LOG_FILE)
        test_summary = test_on_trajectory(traj, config, final_checkpoint_filename, run_id, "", GLOBAL_LOG_FILE)
        if test_summary:
            global_test_summary.append(test_summary)
    if global_test_summary:
        test_summary_filename = f"GlobalTestSummary_{run_id}.csv"
        test_summary_path = os.path.join(SUBFOLDERS["TestSummaries"], test_summary_filename)
        if os.path.exists(test_summary_path):
            log_global(f"Duplicate file found for configuration: {test_summary_path}. Aborting experiment run.", GLOBAL_LOG_FILE)
            sys.exit(0)
        pd.DataFrame(global_test_summary).to_csv(test_summary_path, index=False)
        log_global(f"Global test summary saved to {test_summary_path}", GLOBAL_LOG_FILE)

if __name__ == "__main__":
    # Create folder structure
    second_level_folder, SUBFOLDERS, GLOBAL_LOG_FILE = create_experiment_folders(json.load(open("MNN.json")))
    log_global(f"Experiment folder created at: {second_level_folder}", GLOBAL_LOG_FILE)
    main()
