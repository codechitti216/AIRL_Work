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

# Import the FAN model.
from FAN import FourierAnalysisNetwork

##########################
# Experiment Folder Setup
##########################

RESULTS_ROOT = "FAN_results"  # Root folder for FAN experiments

# Define abbreviations for common config keys.
ABBREVIATIONS = {
    "beam_fill_window": "bfw",
    "dropout_rate": "dr",
    "hidden_neurons": "hn",
    "learning_rate": "lr",  # For naming only
    "learning_rate_2": "lr2",
    "lipschitz_constant": "lc",
    "regularization": "reg",
    "num_past_beam_instances": "npb",
    "num_imu_instances": "nimu",
    "partial_rmse": "pr",
    "training_trajectories": "trtraj",
    "testing_trajectories": "ttraj"
}

def sanitize(text):
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
    sorted_items = sorted(config.items())
    parts = []
    for key, value in sorted_items:
        abbr_key = ABBREVIATIONS.get(key, sanitize(key))
        value_str = stringify_value(value)
        sanitized_value = sanitize(value_str)
        parts.append(f"{abbr_key}{sanitized_value}")
    folder_name = "Exp_" + "_".join(parts)
    print(f"[DEBUG] Generated experiment folder name: {folder_name}")
    return folder_name

def create_experiment_folder(config):
    folder_name = generate_experiment_folder_name(config)
    exp_folder = os.path.join(RESULTS_ROOT, folder_name)
    print(f"[DEBUG] Checking if experiment folder exists: {exp_folder}")
    if os.path.exists(exp_folder):
        print(f"Duplicate experiment folder found: {exp_folder}. Aborting experiment run.")
        sys.exit(0)
    os.makedirs(exp_folder, exist_ok=True)
    subdirs = {
        "CHECKPOINTS_DIR": "Checkpoints",
        "TRAINING_SUMMARIES_DIR": "TrainingSummaries",
        "TEST_SUMMARIES_DIR": "TestSummaries",
        "PLOTS_DIR": "Plots",
        "PREDICTIONS_DIR": "Predictions",
        "TRAINING_VELOCITY_PLOTS_DIR": "TrainingVelocityPlots"
    }
    global CHECKPOINTS_DIR, TRAINING_SUMMARIES_DIR, TEST_SUMMARIES_DIR, PLOTS_DIR, PREDICTIONS_DIR, TRAINING_VELOCITY_PLOTS_DIR, GLOBAL_LOG_FILE
    for key, sub in subdirs.items():
        path = os.path.join(exp_folder, sub)
        os.makedirs(path, exist_ok=True)
        globals()[key] = path
        print(f"[DEBUG] Created subdirectory: {key} -> {path}")
    GLOBAL_LOG_FILE = os.path.join(exp_folder, "experiment_global_log.txt")
    print(f"[DEBUG] Global log file set to: {GLOBAL_LOG_FILE}")
    return exp_folder

##########################
# Logging & Duplicate Check
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

def get_missing_mask(beams_training, t, target_cols=["b1", "b2", "b3", "b4"]):
    row = beams_training.loc[t, target_cols]
    return row.isna().values

def plot_velocity_predictions(predictions, traj, title_suffix=""):
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    samples = [pred["Sample"] for pred in predictions]
    for i, beam in enumerate(["b1", "b2", "b3", "b4"]):
        pred_vals = [pred[f"Pred_{beam}"] for pred in predictions]
        gt_vals = [pred[f"GT_{beam}"] for pred in predictions]
        axes[i].plot(samples, pred_vals, label=f"Predicted {beam}", marker='o')
        axes[i].plot(samples, gt_vals, label=f"Ground Truth {beam}", marker='x')
        axes[i].set_ylabel("Value")
        axes[i].legend(loc="upper right")
        axes[i].grid(True)
    axes[-1].set_xlabel("Sample Index")
    fig.suptitle(f"Predicted vs Ground Truth Beam Velocities for {traj} {title_suffix}")
    return fig

##########################
# Finding the Starting Index
##########################

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

def sequential_train(training_trajectory_pairs, config, model):
    global_training_summary = []
    processed_training_info = []
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
        log_global(f"Original missing count in {traj}: {orig_missing}")
        if orig_missing == 0:
            log_global(f"[Train] No missing beams in {traj}. Skipping training on this trajectory.")
            continue
        filled_beams, start_missing = fill_missing_beams(beams_training, config["beam_fill_window"])
        if start_missing is None:
            log_global(f"[Train] No missing beams detected after filling in {traj}. Skipping trajectory.")
            continue
        log_global(f"Using filled beams starting from index {start_missing} in {traj}")
        min_history = max(start_missing, config["num_past_beam_instances"], config["num_imu_instances"] - 1)
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
            optimizer.zero_grad()
            losses = []
            epoch_squared_errors = np.zeros(4)
            for i in range(num_samples):
                model.train()
                x = torch.tensor(inputs[i], dtype=torch.float32, device=model.device).unsqueeze(0).unsqueeze(0)
                y = torch.tensor(targets[i], dtype=torch.float32, device=model.device)
                y_pred = model(x).squeeze().view(-1)
                y = y.view(-1)
                mask = get_missing_mask(beams_training, i, target_cols=["b1", "b2", "b3", "b4"])
                if config.get("partial_rmse", False) and mask.any():
                    indices = torch.tensor(np.where(mask)[0], dtype=torch.long, device=model.device)
                    if indices.numel() > 0:
                        y_pred_masked = y_pred[indices]
                        y_masked = y[indices]
                        sample_loss = loss_fn(y_pred_masked, y_masked)
                    else:
                        sample_loss = loss_fn(y_pred, y)
                else:
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
        final_fig = plot_velocity_predictions(final_predictions, traj)
        final_plot_path = os.path.join(PLOTS_DIR, f"FinalBeamPredictions_{traj}.png")
        final_fig.savefig(final_plot_path)
        plt.close(final_fig)
        log_global(f"[{traj}] Final epoch beam predictions plot saved to {final_plot_path}")
        
        summary = {
            "Trajectory": traj,
            "NumSamples": num_samples,
            "InputSize": input_size,
            "EpochsTrained": traj_epochs,
            "AvgBestRMSE": np.mean(evolution[-1][1:]),
            "TrainingTime": training_time,
            "TrainedOn": ""  # To be set in main()
        }
        global_training_summary.append(summary)
        processed_training_info.append(f"{traj}:{traj_epochs}")
    return global_training_summary, processed_training_info

def test_on_trajectory(traj, config, checkpoint_filename, trained_on):
    traj_path = os.path.join(DATA_DIR, traj)
    beams_gt, beams_training, imu = load_csv_files(traj_path)
    filled_beams, _ = fill_missing_beams(beams_training, config["beam_fill_window"])
    min_history = max(_, config["num_past_beam_instances"], config["num_imu_instances"] - 1)
    inputs, targets, masks = [], [], []
    for t in range(min_history, len(filled_beams)):
        inp, tar = construct_input_target(filled_beams, beams_gt, imu, t, config["num_past_beam_instances"], config["num_imu_instances"])
        inputs.append(inp)
        targets.append(tar)
        masks.append(get_missing_mask(beams_training, t, target_cols=["b1", "b2", "b3", "b4"]))
    if len(inputs) == 0:
        log_global(f"[Test] Not enough test data in {traj} after history constraints. Skipping.")
        return None
    inputs = np.array(inputs)
    targets = np.array(targets)
    num_samples = len(inputs)
    input_size = inputs.shape[1]
    log_global(f"[{traj}] Testing Samples: {num_samples}, Input size: {input_size}")
    
    model = FourierAnalysisNetwork(number_of_input_neurons=input_size,
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
    
    pred_df = pd.DataFrame(predictions)
    test_pred_csv = os.path.join(PREDICTIONS_DIR, f"TestPredictions_{traj}.csv")
    pred_df.to_csv(test_pred_csv, index=False)
    log_global(f"[{traj}] Test predictions saved to {test_pred_csv}")
    
    fig = plot_velocity_predictions(predictions, traj, title_suffix="(Testing Data)")
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
        "TrainedOn": ", ".join(trained_on)
    }
    return test_summary

def main():
    with open(GLOBAL_LOG_FILE, "w") as f:
        f.write("Experiment Log\n")
    
    with open("FAN.json", "r") as f:
        config = json.load(f)
    
    training_trajectory_pairs = config.get("training_trajectories", [])
    testing_list = config.get("testing_trajectories", [])
    
    global_training_summary = []
    cumulative_trained_on = []  # Stores strings like "Trajectory:Epochs" in order
    
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
            log_global(f"Constructed sample at index {t}.")
        except Exception as e:
            log_global(f"Skipping index {t} in main: {e}")
            continue
    if len(inputs) == 0:
        log_global("Not enough training data in the first trajectory to determine input size.")
        return
    input_size = np.array(inputs).shape[1]
    model = FourierAnalysisNetwork(number_of_input_neurons=input_size,
                                   number_of_hidden_neurons=config["hidden_neurons"],
                                   number_of_output_neurons=4,
                                   dropout_rate=config["dropout_rate"])
    
    log_global("=== Sequential Training Phase ===")
    processed_training_info = []
    for traj_pair in training_trajectory_pairs:
        traj, traj_epochs = traj_pair
        log_global(f"Processing training trajectory: {traj} for {traj_epochs} epochs")
        current_trained_on = cumulative_trained_on.copy()
        traj_summary, processed = sequential_train([[traj, traj_epochs]], config, model)
        if traj_summary:
            row = traj_summary[0]
            row["TrainedOn"] = ", ".join(current_trained_on) if current_trained_on else "None"
            global_training_summary.append(row)
            processed_training_info.append(f"{traj}:{traj_epochs}")
            cumulative_trained_on.append(f"{traj}:{traj_epochs}")
    
    if not global_training_summary:
        log_global("No training trajectories processed; aborting.")
        return
    
    final_checkpoint_filename = (
        f"FAN_{'_'.join(cumulative_trained_on)}_lr{config['learning_rate']}_drop{config['dropout_rate']}_"
        f"hn{config['hidden_neurons']}_reg{config['regularization']}_bfw{config['beam_fill_window']}_"
        f"npb{config['num_past_beam_instances']}_nimu{config['num_imu_instances']}_pr{str(config['partial_rmse'])}_final.pth"
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
        test_summary = test_on_trajectory(traj, config, final_checkpoint_filename, cumulative_trained_on)
        if test_summary:
            global_test_summary.append(test_summary)
    if global_test_summary:
        test_summary_df = pd.DataFrame(global_test_summary)
        test_summary_csv = os.path.join(TEST_SUMMARIES_DIR, f"GlobalTestSummary_{final_checkpoint_filename[:-4]}.csv")
        check_duplicate(test_summary_csv)
        test_summary_df.to_csv(test_summary_csv, index=False)
        log_global(f"Global test summary saved to {test_summary_csv}")

if __name__ == "__main__":
    with open("FAN.json", "r") as f:
        config = json.load(f)
    arch_folder = "FAN_results"
    if not os.path.exists(arch_folder):
        os.makedirs(arch_folder, exist_ok=True)
    EXPERIMENT_FOLDER = create_experiment_folder(config)
    global CHECKPOINTS_DIR, TRAINING_SUMMARIES_DIR, TEST_SUMMARIES_DIR, PLOTS_DIR, PREDICTIONS_DIR, TRAINING_VELOCITY_PLOTS_DIR, GLOBAL_LOG_FILE
    CHECKPOINTS_DIR = os.path.join(EXPERIMENT_FOLDER, "Checkpoints")
    TRAINING_SUMMARIES_DIR = os.path.join(EXPERIMENT_FOLDER, "TrainingSummaries")
    TEST_SUMMARIES_DIR = os.path.join(EXPERIMENT_FOLDER, "TestSummaries")
    PLOTS_DIR = os.path.join(EXPERIMENT_FOLDER, "Plots")
    PREDICTIONS_DIR = os.path.join(EXPERIMENT_FOLDER, "Predictions")
    TRAINING_VELOCITY_PLOTS_DIR = os.path.join(EXPERIMENT_FOLDER, "TrainingVelocityPlots")
    GLOBAL_LOG_FILE = os.path.join(EXPERIMENT_FOLDER, "experiment_global_log.txt")
    
    log_global(f"Experiment folder created: {EXPERIMENT_FOLDER}")
    main()
