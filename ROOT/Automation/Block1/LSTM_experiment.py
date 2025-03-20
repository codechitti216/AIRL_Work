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

# Import the LSTM model defined in LSTM.py
from LSTM import LSTMNetwork

##########################
# Experiment Folder Setup
##########################

RESULTS_ROOT = "ExperimentResults"

# Define abbreviations for common keys.
ABBREVIATIONS = {
    "beam_fill_window": "bfw",
    "dropout_rate": "dr",
    "hidden_neurons": "hn",
    "learning_rate": "lr",
    "num_imu_instances": "nimu",
    "num_layers": "nl",
    "num_past_beam_instances": "npb",
    "partial_rmse": "pr",
    "regularization": "reg",
    "testing_trajectories": "ttraj",
    "training_trajectories": "trtraj",
    "epochs": "ep",
    "seed": "sd"
}

def sanitize(text):
    """
    Sanitize a text string for use in file/folder names:
    - Replace spaces with underscores.
    - Remove any non-alphanumeric characters except underscores, hyphens, and dots.
    """
    text = re.sub(r'\s+', '_', text)
    text = re.sub(r'[^\w\-_\.]', '', text)
    return text

def stringify_value(value):
    """
    Convert a value to a string suitable for a folder name.
    For lists (like training trajectories), join items with a dash.
    For booleans, use T/F.
    """
    if isinstance(value, list):
        # If it's a list of lists/tuples (e.g. training trajectories), join inner elements with a hyphen.
        if all(isinstance(item, (list, tuple)) for item in value):
            return "_".join(["-".join(map(str, item)) for item in value])
        else:
            return "-".join(map(str, value))
    elif isinstance(value, bool):
        return "T" if value else "F"
    else:
        return str(value)

def generate_experiment_folder_name(config):
    """
    Generate a folder name from the JSON configuration by:
      1. Sorting the keys alphabetically.
      2. Converting each key to its abbreviation (if available).
      3. Concatenating each abbreviated key and its sanitized value.
      4. Joining all key–value pairs with underscores and prepending "Exp_".
    """
    sorted_items = sorted(config.items())
    parts = []
    for key, value in sorted_items:
        abbr_key = ABBREVIATIONS.get(key, sanitize(key))
        value_str = stringify_value(value)
        sanitized_value = sanitize(value_str)
        parts.append(f"{abbr_key}{sanitized_value}")
    folder_name = "Exp_" + "_".join(parts)
    return folder_name

def create_experiment_folder(arch_folder, config):
    """
    Create an experiment folder inside the given architecture folder.
    If an identical folder (based on the configuration) already exists, exit the program.
    Standard subfolders are created inside the experiment folder.
    """
    folder_name = generate_experiment_folder_name(config)
    full_path = os.path.join(arch_folder, folder_name)
    
    if os.path.exists(full_path):
        print(f"Duplicate experiment folder found: {full_path}. Skipping run.")
        sys.exit(0)
    
    os.makedirs(full_path, exist_ok=True)
    
    # Create standard subfolders
    subfolders = ["Checkpoints", "TrainingSummaries", "TestSummaries", "Plots", "Predictions"]
    for sub in subfolders:
        os.makedirs(os.path.join(full_path, sub), exist_ok=True)
    
    print(f"Experiment folder created: {full_path}")
    return full_path

##########################
# Logging and Duplicate Check
##########################

def log_global(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(GLOBAL_LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(message)

def check_duplicate(file_path):
    print(f"[DEBUG] Checking duplicate for file: {file_path}")
    if os.path.exists(file_path):
        log_global(f"Duplicate file found: {file_path}. Aborting experiment.")
        sys.exit(0)
    else:
        print(f"[DEBUG] No duplicate found for file: {file_path}")

##########################
# Data Loading & Preprocessing
##########################

DATA_DIR = "../../Data"

def load_csv_files(traj_path):
    beams_gt_path = os.path.join(traj_path, "beams_gt.csv")
    beams_training_path = os.path.join(traj_path, "beams_training.csv")
    beams_gt = pd.read_csv(beams_gt_path, na_values=[''])
    beams_training = pd.read_csv(beams_training_path, na_values=[''])
    log_global(f"DEBUG: beams_gt columns: {beams_gt.columns.tolist()}")
    log_global(f"DEBUG: beams_training columns: {beams_training.columns.tolist()}")
    log_global("DEBUG: beams_training head:")
    log_global(str(beams_training.head()))
    beam_cols = ["b1", "b2", "b3", "b4"]
    missing_counts = beams_training[beam_cols].isna().sum()
    log_global("DEBUG: Missing counts in beams_training:")
    log_global(str(missing_counts))
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
    return beams_gt, beams_training, imu

def fill_missing_beams(beams_df, beam_fill_window, beam_cols=["b1", "b2", "b3", "b4"]):
    filled = beams_df.copy()
    missing_mask = filled[beam_cols].isna().any(axis=1)
    missing_count = missing_mask.sum()
    log_global(f"DEBUG: Total rows with missing beams: {missing_count}")
    if missing_count == 0:
        log_global("DEBUG: No missing beams detected.")
        return filled, None
    start_index = missing_mask.idxmax()
    log_global(f"DEBUG: First row with missing beam detected at index: {start_index}")
    for i in range(start_index, len(filled)):
        for col in beam_cols:
            if pd.isna(filled.loc[i, col]):
                if i < beam_fill_window:
                    log_global(f"DEBUG: Not enough history to fill {col} at row {i}")
                    continue
                prev_vals = filled.loc[i - beam_fill_window:i - 1, col]
                avg_val = prev_vals.mean()
                log_global(f"DEBUG: Filling missing {col} at row {i} with average value {avg_val} from rows {i - beam_fill_window} to {i - 1}")
                filled.loc[i, col] = avg_val
    return filled, start_index

def construct_input_target(filled_beams, beams_gt, imu, t, num_past_beam_instances, num_imu_instances):
    current_beams = filled_beams.loc[t, ["b1", "b2", "b3", "b4"]].values.astype(float)
    past_beams = []
    for i in range(1, num_past_beam_instances + 1):
        past_beams.extend(filled_beams.loc[t - i, ["b1", "b2", "b3", "b4"]].values.astype(float))
    imu_cols = ['ACC X [m/s^2]', 'ACC Y [m/s^2]', 'ACC Z [m/s^2]',
                'GYRO X [rad/s]', 'GYRO Y [rad/s]', 'GYRO Z [rad/s]']
    past_imu = []
    for i in range(num_imu_instances - 1, -1, -1):
        past_imu.extend(imu.loc[t - i, imu_cols].values.astype(float))
    input_vector = np.concatenate([current_beams, np.array(past_beams), np.array(past_imu)])
    target_vector = beams_gt.loc[t, ["b1", "b2", "b3", "b4"]].values.astype(float)
    return input_vector, target_vector

def get_missing_mask(beams_training, t, beam_cols=["b1", "b2", "b3", "b4"]):
    row = beams_training.loc[t, beam_cols]
    return row.isna().values

##########################
# Sequential Training Routine
##########################

def sequential_train(training_trajectory_pairs, config, model):
    global_training_summary = []
    # Process each training trajectory separately.
    for traj_pair in training_trajectory_pairs:
        traj, traj_epochs = traj_pair
        traj_path = os.path.join(DATA_DIR, traj)
        log_global(f"=== Training on Trajectory: {traj} for {traj_epochs} epochs ===")
        try:
            beams_gt, beams_training, imu = load_csv_files(traj_path)
        except Exception as e:
            log_global(f"Error loading files in {traj}: {e}")
            continue
        beam_cols = ["b1", "b2", "b3", "b4"]
        orig_missing = beams_training[beam_cols].isna().sum().sum()
        if orig_missing == 0:
            log_global(f"[Train] No missing beams in {traj}. Skipping training on this trajectory.")
            continue
        filled_beams, start_missing = fill_missing_beams(beams_training, config["beam_fill_window"])
        if start_missing is None:
            log_global(f"[Train] No missing beams detected after filling in {traj}. Skipping trajectory.")
            continue
        log_global(f"DEBUG: Filled beams from index {start_missing} onward in {traj}")
        min_history = max(start_missing, config["num_past_beam_instances"], config["num_imu_instances"] - 1)
        inputs, targets, masks = [], [], []
        for t in range(min_history, len(filled_beams)):
            if t - config["num_past_beam_instances"] < 0 or t - (config["num_imu_instances"] - 1) < 0:
                continue
            inp, tar = construct_input_target(filled_beams, beams_gt, imu, t, config["num_past_beam_instances"], config["num_imu_instances"])
            inputs.append(inp)
            targets.append(tar)
            masks.append(get_missing_mask(beams_training, t))
        if len(inputs) == 0:
            log_global(f"[Train] Not enough data in {traj} after history constraints. Skipping trajectory.")
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
            epoch_squared_errors = np.zeros(4)
            for i in range(num_samples):
                model.train()
                optimizer.zero_grad()
                x = torch.tensor(inputs[i], dtype=torch.float32, device=model.device).unsqueeze(0).unsqueeze(0)
                y = torch.tensor(targets[i], dtype=torch.float32, device=model.device)
                y_pred = model(x).squeeze(0)  # shape: (4,)
                mask = masks[i]
                if config["partial_rmse"] and mask.any():
                    indices = torch.tensor(np.where(mask)[0], dtype=torch.long, device=model.device)
                    if indices.numel() > 0:
                        y_pred_masked = torch.index_select(y_pred, 0, indices)
                        y_masked = torch.index_select(y, 0, indices)
                        loss = loss_fn(y_pred_masked, y_masked)
                    else:
                        loss = loss_fn(y_pred, y)
                else:
                    loss = loss_fn(y_pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                error = (y_pred - y).detach().cpu().numpy() ** 2
                epoch_squared_errors += error
            epoch_rmse = np.sqrt(epoch_squared_errors / num_samples)
            evolution.append([epoch] + epoch_rmse.tolist())
            log_global(f"[{traj}] Epoch {epoch}: RMSE per beam: {epoch_rmse}, Avg RMSE: {np.mean(epoch_rmse):.5f}")
        
        training_time = time.time() - t0
        
        # **** New: Evaluate and plot final epoch predictions for this trajectory ****
        model.eval()
        final_predictions = []
        for i in range(num_samples):
            x = torch.tensor(inputs[i], dtype=torch.float32, device=model.device).unsqueeze(0).unsqueeze(0)
            y = torch.tensor(targets[i], dtype=torch.float32, device=model.device)
            with torch.no_grad():
                y_pred = model(x).squeeze(0)
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
        final_fig = plot_velocity_predictions(final_predictions, traj)
        final_plot_path = os.path.join(PLOTS_DIR, f"FinalBeamPredictions_{traj}.png")
        final_fig.savefig(final_plot_path)
        plt.close(final_fig)
        log_global(f"[{traj}] Final epoch beam predictions plot saved to {final_plot_path}")
        # **** End New ****
        
        # Prepare summary row without including the current trajectory in "TrainedOn"
        summary = {
            "Trajectory": traj,
            "NumSamples": num_samples,
            "InputSize": input_size,
            "EpochsTrained": traj_epochs,
            "AvgBestRMSE": np.mean(evolution[-1][1:]),
            "TrainingTime": training_time,
            "TrainedOn": ""  # to be filled later in main()
        }
        global_training_summary.append(summary)
    return global_training_summary

def plot_velocity_predictions(predictions, traj):
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    samples = [pred["Sample"] for pred in predictions]
    for i, beam in enumerate(["b1", "b2", "b3", "b4"]):
        pred_vals = [pred[f"Pred_{beam}"] for pred in predictions]
        gt_vals = [pred[f"GT_{beam}"] for pred in predictions]
        axes[i].plot(samples, pred_vals, label=f"Predicted {beam}", marker='o')
        axes[i].plot(samples, gt_vals, label=f"Ground Truth {beam}", marker='x')
        axes[i].set_ylabel("Velocity")
        axes[i].legend(loc="upper right")
        axes[i].grid(True)
    axes[-1].set_xlabel("Sample Index")
    fig.suptitle(f"Predicted vs Ground Truth Beam Velocities for {traj}")
    return fig

def test_on_trajectory(traj, config, checkpoint_filename, trained_on):
    traj_path = os.path.join(DATA_DIR, traj)
    beams_gt, beams_training, imu = load_csv_files(traj_path)
    filled_beams, start_missing = fill_missing_beams(beams_training, config["beam_fill_window"])
    if start_missing is None:
        log_global(f"[Test] No missing beams in {traj}. Skipping testing.")
        return None
    min_history = max(start_missing, config["num_past_beam_instances"], config["num_imu_instances"] - 1)
    inputs, targets, masks = [], [], []
    for t in range(min_history, len(filled_beams)):
        if t - config["num_past_beam_instances"] < 0 or t - (config["num_imu_instances"] - 1) < 0:
            continue
        inp, tar = construct_input_target(filled_beams, beams_gt, imu, t, config["num_past_beam_instances"], config["num_imu_instances"])
        inputs.append(inp)
        targets.append(tar)
        masks.append(get_missing_mask(beams_training, t))
    if len(inputs) == 0:
        log_global(f"[Test] Not enough test data in {traj} after history constraints. Skipping.")
        return None
    inputs = np.array(inputs)
    targets = np.array(targets)
    num_samples = len(inputs)
    input_size = inputs.shape[1]
    log_global(f"[{traj}] Testing Samples: {num_samples}, Input size: {input_size}")
    
    model = LSTMNetwork(number_of_input_neurons=input_size,
                        number_of_hidden_neurons=config["hidden_neurons"],
                        number_of_output_neurons=4,
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
    squared_errors = np.zeros(4)
    for i in range(num_samples):
        x = torch.tensor(inputs[i], dtype=torch.float32, device=model.device).unsqueeze(0).unsqueeze(0)
        y = torch.tensor(targets[i], dtype=torch.float32, device=model.device)
        with torch.no_grad():
            y_pred = model(x).squeeze(0)
        predictions.append({
            "Sample": i,
            "Pred_b1": y_pred[0].item(),
            "Pred_b2": y_pred[1].item(),
            "Pred_b3": y_pred[2].item(),
            "Pred_b4": y_pred[3].item(),
            "GT_b1": y[0].item(),
            "GT_b2": y[1].item(),
            "GT_b3": y[2].item(),
            "GT_b4": y[3].item(),
            "Error_b1": (y_pred[0] - y[0]).item()**2,
            "Error_b2": (y_pred[1] - y[1]).item()**2,
            "Error_b3": (y_pred[2] - y[2]).item()**2,
            "Error_b4": (y_pred[3] - y[3]).item()**2
        })
        err = (y_pred - y).detach().cpu().numpy() ** 2
        squared_errors += err
    test_rmse = np.sqrt(squared_errors / num_samples)
    log_global(f"[{traj}] Test RMSE per beam: {test_rmse}")
    
    pred_df = pd.DataFrame(predictions)
    test_pred_csv = os.path.join(PREDICTIONS_DIR, f"TestPredictions_{traj}.csv")
    pred_df.to_csv(test_pred_csv, index=False)
    log_global(f"[{traj}] Test predictions saved to {test_pred_csv}")
    
    fig = plot_velocity_predictions(predictions, traj)
    plot_file = os.path.join(PLOTS_DIR, f"VelocityPredictions_{traj}.png")
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
        "TrainedOn": trained_on
    }
    return test_summary

##########################
# Main Experiment Routine
##########################

def main():
    with open(GLOBAL_LOG_FILE, "w") as f:
        f.write("Experiment Log\n")
    
    with open("LSTM.json", "r") as f:
        config = json.load(f)
    
    training_trajectory_pairs = config.get("training_trajectories", [])
    testing_list = config.get("testing_trajectories", [])
    
    global_training_summary = []
    cumulative_trained_on = []  # List of trajectories completed so far (excluding the current one)
    
    if not training_trajectory_pairs:
        log_global("No training trajectories provided.")
        return
    first_traj = training_trajectory_pairs[0][0]
    first_traj_path = os.path.join(DATA_DIR, first_traj)
    beams_gt, beams_training, imu = load_csv_files(first_traj_path)
    filled_beams, start_missing = fill_missing_beams(beams_training, config["beam_fill_window"])
    if start_missing is None:
        log_global("Not enough missing data in the first trajectory to determine input size.")
        return
    min_history = max(start_missing, config["num_past_beam_instances"], config["num_imu_instances"] - 1)
    inputs = []
    for t in range(min_history, len(filled_beams)):
        try:
            inp, _ = construct_input_target(filled_beams, beams_gt, imu, t, config["num_past_beam_instances"], config["num_imu_instances"])
            inputs.append(inp)
        except Exception as e:
            continue
    if len(inputs) == 0:
        log_global("Not enough training data in the first trajectory to determine input size.")
        return
    input_size = np.array(inputs).shape[1]
    model = LSTMNetwork(number_of_input_neurons=input_size,
                        number_of_hidden_neurons=config["hidden_neurons"],
                        number_of_output_neurons=4,
                        dropout_rate=config["dropout_rate"])
    
    log_global("=== Sequential Training Phase ===")
    # Process each training trajectory in order.
    for traj_pair in training_trajectory_pairs:
        traj, traj_epochs = traj_pair
        log_global(f"Processing training trajectory: {traj} for {traj_epochs} epochs")
        # Before training the current trajectory, capture the cumulative list so far.
        current_trained_on = ", ".join(cumulative_trained_on)
        traj_summary = sequential_train([[traj, traj_epochs]], config, model)
        if traj_summary:
            # Set "TrainedOn" field to the list BEFORE this trajectory is added.
            row = traj_summary[0]
            row["TrainedOn"] = current_trained_on
            global_training_summary.append(row)
            # Then add the current trajectory to the cumulative list.
            cumulative_trained_on.append(traj)
    
    if not global_training_summary:
        log_global("No training trajectories processed; aborting.")
        return
    
    final_checkpoint_filename = (
        f"LSTM_{'_'.join(cumulative_trained_on)}_lr{config['learning_rate']}_dr{config['dropout_rate']}_"
        f"hn{config['hidden_neurons']}_nl{config['num_layers']}_reg{config['regularization']}_bfw{config['beam_fill_window']}_"
        f"npb{config['num_past_beam_instances']}_nimu{config['num_imu_instances']}_pr{config['partial_rmse']}_final.pth"
    )
    final_checkpoint_path = os.path.join(CHECKPOINTS_DIR, final_checkpoint_filename)
    check_duplicate(final_checkpoint_path)
    torch.save(model.state_dict(), final_checkpoint_path)
    log_global(f"Final checkpoint saved to {final_checkpoint_path}")
    
    train_summary_df = pd.DataFrame(global_training_summary)
    train_summary_csv = os.path.join(TRAINING_SUMMARIES_DIR, f"GlobalTrainingSummary_{final_checkpoint_filename[:-4]}.csv")
    check_duplicate(train_summary_csv)
    train_summary_df.to_csv(train_summary_csv, index=False)
    log_global(f"Global training summary saved to {train_summary_csv}")
    
    log_global("=== Testing Phase ===")
    global_test_summary = []
    for traj in testing_list:
        log_global(f"Processing testing trajectory: {traj}")
        test_summary = test_on_trajectory(traj, config, final_checkpoint_filename, ", ".join(cumulative_trained_on))
        if test_summary:
            global_test_summary.append(test_summary)
    if global_test_summary:
        test_summary_df = pd.DataFrame(global_test_summary)
        test_summary_csv = os.path.join(TEST_SUMMARIES_DIR, f"GlobalTestSummary_{final_checkpoint_filename[:-4]}.csv")
        check_duplicate(test_summary_csv)
        test_summary_df.to_csv(test_summary_csv, index=False)
        log_global(f"Global test summary saved to {test_summary_csv}")

if __name__ == "__main__":
    # Load the configuration and create the experiment folder
    with open("LSTM.json", "r") as f:
        config = json.load(f)
    
    # Define the architecture-specific results folder (for LSTM)
    arch_folder = "LSTM_Results"
    if not os.path.exists(arch_folder):
        os.makedirs(arch_folder, exist_ok=True)
    
    # Create the experiment folder based on the full configuration with abbreviations
    EXPERIMENT_FOLDER = create_experiment_folder(arch_folder, config)
    
    # Set global directories for saving files based on the experiment folder
    global CHECKPOINTS_DIR, TRAINING_SUMMARIES_DIR, TEST_SUMMARIES_DIR, PLOTS_DIR, PREDICTIONS_DIR, GLOBAL_LOG_FILE
    CHECKPOINTS_DIR = os.path.join(EXPERIMENT_FOLDER, "Checkpoints")
    TRAINING_SUMMARIES_DIR = os.path.join(EXPERIMENT_FOLDER, "TrainingSummaries")
    TEST_SUMMARIES_DIR = os.path.join(EXPERIMENT_FOLDER, "TestSummaries")
    PLOTS_DIR = os.path.join(EXPERIMENT_FOLDER, "Plots")
    PREDICTIONS_DIR = os.path.join(EXPERIMENT_FOLDER, "Predictions")
    GLOBAL_LOG_FILE = os.path.join(EXPERIMENT_FOLDER, "experiment_global_log.txt")
    
    log_global(f"Experiment folder created: {EXPERIMENT_FOLDER}")
    main()
