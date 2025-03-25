#!/usr/bin/env python3
"""
MNN_experiment.py

This script sets up and runs experiments with the Memory Neural Network (MNN) model.
It includes experiment folder setup, detailed logging, data loading/synchronization,
and updated preprocessing steps to address NaN predictions.

Key Updates:
1. The random and range removals (and the subsequent moving average filling) are now applied 
   only on data starting from the first valid index.
2. For test evaluation:
   - Test RMSE: Samples with ground truth velocities outside the specified velocity range 
     are marked with NaN errors and ignored in the overall RMSE calculation.
   - Plotting: Ground truth values outside the velocity range are clipped to the nearest boundary.
   
Configuration is read from MNN.json.
"""

import os
import sys
import json
import re
import time
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import the Memory Neural Network model from MNN.py
from MNN import MemoryNeuralNetwork

# Debug flag (always enabled here)
DEBUG = True

def debug_print(message):
    if DEBUG:
        print(f"[DEBUG] {message}")

##########################
# Experiment Folder Setup
##########################

RESULTS_ROOT = "Results"  # Top-level folder for experiments

def sanitize(text):
    """Sanitize a given string to remove unwanted characters."""
    text = str(text).replace(":", "")
    text = re.sub(r'[^\w_.-]', '', text)
    return text

def stringify_value(value):
    """Convert a value into a string representation."""
    if isinstance(value, list):
        if all(isinstance(item, (list, tuple)) for item in value):
            return "_".join(["-".join(map(str, item)) for item in value])
        else:
            return "-".join(map(str, value))
    elif isinstance(value, bool):
        return "T" if value else "F"
    else:
        return str(value)

def generate_first_level_folder_name(config):
    tr_parts = []
    for traj_pair in config.get("training_trajectories", []):
        traj, epochs = traj_pair
        traj_id = traj.lower().replace("trajectory", "")
        tr_parts.append(f"{traj_id}{epochs}")
    tr_str = "".join(tr_parts)

    tt_parts = []
    for traj in config.get("testing_trajectories", []):
        traj_id = traj.lower().replace("trajectory", "")
        tt_parts.append(traj_id)
    tt_str = "".join(tt_parts)

    folder_name = f"TrTj{tr_str}TTj{tt_str}"
    return sanitize(folder_name)

def generate_second_level_folder_name(config):
    bfw = config.get("beam_fill_window")
    npbi = config.get("num_past_beam_instances")
    niu = config.get("num_imu_instances")
    rr = config.get("random_removal_percentages", {})
    folder_name = (f"bfw_{bfw}npbi{npbi}niu{niu}"
                   f"b1{rr.get('b1', 0)}b2{rr.get('b2', 0)}"
                   f"b3{rr.get('b3', 0)}b4{rr.get('b4', 0)}")
    return sanitize(folder_name)

def create_experiment_folder(config):
    first_level = generate_first_level_folder_name(config)
    second_level = generate_second_level_folder_name(config)
    exp_folder = os.path.join(RESULTS_ROOT, first_level, second_level)
    
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
        os.makedirs(path, exist_ok=True)
        globals()[key] = path
        debug_print(f"Subdirectory for {key}: {path}")

    readme_path = os.path.join(exp_folder, "Readme.md")
    with open(readme_path, "w") as f:
        f.write("Experiment Details\n")
        f.write("==================\n\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    debug_print(f"Readme.md created at: {readme_path}")

    GLOBAL_LOG_FILE = os.path.join(exp_folder, "experiment_global_log.txt")
    debug_print(f"Global log file set to: {GLOBAL_LOG_FILE}")
    return exp_folder

##########################
# Logging
##########################

def log_global(message):
    """Log a message to the global log file with a timestamp and print it."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(GLOBAL_LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(message)

##########################
# Data Loading & Synchronization
##########################

DATA_DIR = "../../Data"
REQUIRED_VELOCITY_COLS = ["V North", "V East", "V Down"]

def synchronize_data(beams_df, velocity_df, imu_df):
    debug_print("Starting synchronization of data.")
    beams_df = beams_df.copy()
    velocity_df = velocity_df.copy()
    imu_df = imu_df.copy()
    
    beams_df['Time'] = beams_df['Time'].astype(str)
    velocity_df['Time'] = velocity_df['Time'].astype(str)
    imu_df['Time'] = imu_df['Time [s]'].astype(str)
    
    common_times = set(beams_df['Time']) & set(velocity_df['Time']) & set(imu_df['Time'])
    debug_print(f"Found {len(common_times)} common timestamps.")
    if not common_times:
        raise ValueError("No common timestamps found across data files.")
    
    beams_df = beams_df[beams_df['Time'].isin(common_times)].sort_values("Time").reset_index(drop=True)
    velocity_df = velocity_df[velocity_df['Time'].isin(common_times)].sort_values("Time").reset_index(drop=True)
    imu_df = imu_df[imu_df['Time'].isin(common_times)].sort_values("Time").reset_index(drop=True)
    
    debug_print(f"Synchronization complete. beams_df: {beams_df.shape}, velocity_df: {velocity_df.shape}, imu_df: {imu_df.shape}")
    return beams_df, velocity_df, imu_df

def load_csv_files(traj_path):
    print(f"[INFO] Loading CSV files from trajectory folder: {traj_path}")
    beams_gt_path = os.path.join(traj_path, "beams_gt.csv")
    velocity_gt_path = os.path.join(traj_path, "velocity_gt.csv")
    
    try:
        beams_gt = pd.read_csv(beams_gt_path, na_values=[''])
        debug_print(f"Loaded beams_gt.csv with shape: {beams_gt.shape}")
    except Exception as e:
        log_global(f"ERROR: Failed to load beams_gt from {beams_gt_path}: {e}")
        raise e

    try:
        velocity_gt = pd.read_csv(velocity_gt_path, na_values=[''])
        debug_print(f"Loaded velocity_gt.csv with shape: {velocity_gt.shape}")
    except Exception as e:
        log_global(f"ERROR: Failed to load velocity_gt from {velocity_gt_path}: {e}")
        raise e

    if not all(col in velocity_gt.columns for col in REQUIRED_VELOCITY_COLS):
        err_msg = (f"ERROR: velocity_gt file missing required columns. Expected: {REQUIRED_VELOCITY_COLS}, "
                   f"Found: {list(velocity_gt.columns)}")
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
        debug_print(f"Loaded IMU file with shape: {imu.shape}")
    except Exception as e:
        log_global(f"ERROR: Failed to load IMU file: {e}")
        raise e

    beams_gt, velocity_gt, imu = synchronize_data(beams_gt, velocity_gt, imu)
    beams_training, _, _ = synchronize_data(beams_training, velocity_gt, imu)

    log_global(f"Loaded files after synchronization: beams_gt={len(beams_gt)} rows, "
               f"velocity_gt={len(velocity_gt)} rows, imu={len(imu)} rows")
    return beams_gt, beams_training, imu, velocity_gt

##########################
# Missing Value Removal & Filling (with Moving Average)
##########################

def fill_missing_beams(beams_df, beam_fill_window, beam_cols=["b1", "b2", "b3", "b4"]):
    debug_print("Starting missing value filling.")
    filled = beams_df.copy()
    for i in range(beam_fill_window, len(filled)):
        for col in beam_cols:
            if pd.isna(filled.loc[i, col]):
                window = filled.loc[i - beam_fill_window:i - 1, col]
                debug_print(f"Row {i}, Column {col}: window values = {window.values}")
                if window.isna().all():
                    last_val = filled[col].ffill().iloc[i - 1]
                    log_global(f"DEBUG: All previous values missing for {col} at row {i}; using last valid value {last_val}")
                    filled.loc[i, col] = last_val
                else:
                    avg_val = window.mean()
                    log_global(f"DEBUG: Filling missing {col} at row {i} with moving average {avg_val}")
                    filled.loc[i, col] = avg_val
    debug_print("Missing value filling complete.")
    return filled, beam_fill_window

##########################
# First Valid Sample Determination & Data Preprocessing
##########################

def construct_input_target(filled_beams, velocity_gt, imu, t, num_past_beam_instances, num_imu_instances):
    if t >= len(velocity_gt):
        err_msg = f"ERROR: Index {t} is out of bounds for velocity_gt (length {len(velocity_gt)})"
        log_global(err_msg)
        raise IndexError(err_msg)

    if t < num_past_beam_instances or t < (num_imu_instances - 1):
        err_msg = (f"ERROR: Index {t} does not satisfy minimum history requirements "
                   f"(num_past_beam_instances={num_past_beam_instances}, num_imu_instances={num_imu_instances})")
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
            debug_print(f"Past beam at index {idx} for t={t}: {past_row}")
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
            debug_print(f"IMU data at index {idx} for t={t}: {imu_row}")
        except Exception as e:
            err_msg = f"ERROR at index {idx} fetching IMU data: {e}"
            log_global(err_msg)
            raise e

    input_vector = np.concatenate([current_beams, np.array(past_beams), np.array(past_imu)])
    debug_print(f"Constructed input vector at t={t}: {input_vector}")

    try:
        target_vector = velocity_gt.loc[t, REQUIRED_VELOCITY_COLS].values.astype(float)
        debug_print(f"Constructed target vector at t={t}: {target_vector}")
    except Exception as e:
        err_msg = f"ERROR at index {t} fetching target velocities: {e}"
        log_global(err_msg)
        raise e

    return input_vector, target_vector

def find_first_valid_index(filled_beams, beams_gt, imu, num_past_beam_instances, num_imu_instances, velocity_gt):
    start = max(num_past_beam_instances, num_imu_instances - 1)
    log_global(f"Starting search for valid sample from index {start}")
    for t in range(start, min(len(filled_beams), len(velocity_gt))):
        try:
            inp, tar = construct_input_target(filled_beams, velocity_gt, imu, t, num_past_beam_instances, num_imu_instances)
            debug_print(f"At index {t}: Constructed input = {inp}, target = {tar}")
            log_global(f"Valid input-target pair found at index {t}.")
            return t
        except Exception as e:
            log_global(f"Index {t} invalid: {e}")
            continue
    return None

def process_data_with_removals(beams_df, velocity_df, imu_df, config):
    num_past_beams = config.get("num_past_beam_instances")
    num_imus = config.get("num_imu_instances")
    
    first_valid = find_first_valid_index(fill_missing_beams(beams_df, config["beam_fill_window"])[0],
                                           beams_df, imu_df, num_past_beams, num_imus, velocity_df)
    log_global(f"Using starting index {first_valid} for processing data.")
    
    # Slice data from the first valid index onward
    beams_df = beams_df.iloc[first_valid:].reset_index(drop=True)
    velocity_df = velocity_df.iloc[first_valid:].reset_index(drop=True)
    imu_df = imu_df.iloc[first_valid:].reset_index(drop=True)
    
    # Apply removal based on method specified in config
    removal_method = config.get("removal_method", "random")
    if removal_method == "random":
        log_global("Applying random removal on beams data...")
        beams_df, _ = apply_random_removal(beams_df, config, 0)
    elif removal_method == "range":
        log_global("Applying range removal on beams data...")
        beams_df, _ = apply_range_removal(beams_df, config, 0)
    
    # Fill missing values after removal
    beams_df, _ = fill_missing_beams(beams_df, config["beam_fill_window"])
    return beams_df, velocity_df, imu_df

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
        # Clip ground truth values to velocity range boundaries for plotting
        gt_vals_clipped = []
        for val in gt_vals:
            # Assuming velocity_range is [-3, 3]
            if val < -3:
                gt_vals_clipped.append(-3)
            elif val > 3:
                gt_vals_clipped.append(3)
            else:
                gt_vals_clipped.append(val)
        axes[i].plot(samples, gt_vals_clipped, label=f"Ground Truth {comp}", linestyle='-')
        axes[i].plot(samples, pred_vals, label=f"Predicted {comp}", linestyle='--')
        axes[i].set_ylabel("Value")
        axes[i].legend(loc="upper right")
        axes[i].grid(True)
    axes[-1].set_xlabel("Sample Index")
    fig.suptitle(f"Predicted vs Ground Truth for {traj} {title_suffix}")
    return fig

def plot_pred_error(predictions, traj, title_suffix=""):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    samples = [pred["Sample"] for pred in predictions]
    for i, comp in enumerate(["V_North", "V_East", "V_Down"]):
        pred_vals = [pred[f"Pred_{comp}"] for pred in predictions]
        gt_vals = [pred[f"GT_{comp}"] for pred in predictions]
        sq_error = [(p - g)**2 for p, g in zip(pred_vals, gt_vals)]
        axes[i].plot(samples, sq_error, label=f"Squared Error ({comp})", linestyle='--')
        axes[i].set_ylabel("Squared Error")
        axes[i].legend(loc="upper right")
        axes[i].grid(True)
    axes[-1].set_xlabel("Sample Index")
    fig.suptitle(f"Squared Error (Predicted vs GT) for {traj} {title_suffix}")
    return fig

##########################
# Sequential Training Routine
##########################

def apply_random_removal(beams_df, config, start_idx):
    percentages = config.get("random_removal_percentages", {"b1": 0, "b2": 0, "b3": 0, "b4": 0})
    count_removed = {'b1': 0, 'b2': 0, 'b3': 0, 'b4': 0}
    print(f"[INFO] Applying random removal from row {start_idx} with percentages: {percentages}")
    for beam in ['b1', 'b2', 'b3', 'b4']:
        valid_indices = beams_df.index[start_idx:][beams_df.loc[start_idx:, beam].notna()]
        num_to_remove = int(len(valid_indices) * (percentages.get(beam, 0) / 100.0))
        debug_print(f"For beam {beam} from index {start_idx}: {len(valid_indices)} valid indices, removing {num_to_remove}")
        if num_to_remove > 0:
            remove_indices = np.random.choice(valid_indices, size=num_to_remove, replace=False)
            beams_df.loc[remove_indices, beam] = np.nan
            count_removed[beam] = num_to_remove
    total_removed = sum(count_removed.values())
    print(f"[DEBUG] Applied random removal. Total beams removed: {total_removed}")
    beams_df, _ = fill_missing_beams(beams_df, config["beam_fill_window"])
    return beams_df, count_removed

def apply_range_removal(beams_df, config, start_idx):
    lower, upper = config.get("beam_range", [-1.5, 1.5])
    count_removed = {'b1': 0, 'b2': 0, 'b3': 0, 'b4': 0}
    print(f"[INFO] Applying range removal from row {start_idx} with range [{lower}, {upper}]")
    for idx in range(start_idx, beams_df.shape[0]):
        for beam in ['b1', 'b2', 'b3', 'b4']:
            value = beams_df.loc[idx, beam]
            if pd.notna(value) and not (lower <= value <= upper):
                beams_df.loc[idx, beam] = np.nan
                count_removed[beam] += 1
    total_removed = sum(count_removed.values())
    print(f"[DEBUG] Applied range removal from row {start_idx} onward. Total beams removed: {total_removed}")
    beams_df, _ = fill_missing_beams(beams_df, config["beam_fill_window"])
    return beams_df, count_removed

def sequential_train(training_trajectory_pairs, config, model, run_id, trained_list):
    global_training_summary = []
    for traj_pair in training_trajectory_pairs:
        traj, traj_epochs = traj_pair
        traj_path = os.path.join(DATA_DIR, traj)
        log_global(f"=== Training on Trajectory: {traj} for {traj_epochs} epochs ===")
        try:
            beams_gt, beams_training, imu, velocity_gt = load_csv_files(traj_path)
        except Exception as e:
            log_global(f"Error loading files in {traj}: {e}")
            continue

        # Determine the first valid index on the original filled data
        filled_beams_initial, _ = fill_missing_beams(beams_training, config["beam_fill_window"])
        min_history = find_first_valid_index(filled_beams_initial, beams_gt, imu,
                                               config["num_past_beam_instances"],
                                               config["num_imu_instances"],
                                               velocity_gt)
        if min_history is None:
            log_global("Not enough training data in the trajectory to determine input size.")
            continue
        log_global(f"Using starting index {min_history} for removal and input-target construction.")
        
        # For training, apply random removal after min_history
        beams_training_modified, count_removed = apply_random_removal(beams_training, config, min_history)
        log_global(f"Beam removal counts: {count_removed}")
        
        filled_beams, _ = fill_missing_beams(beams_training_modified, config["beam_fill_window"])
        
        inputs, targets, masks = [], [], []
        for t in range(min_history, min(len(filled_beams), len(velocity_gt))):
            try:
                inp, tar = construct_input_target(filled_beams, velocity_gt, imu, t,
                                                    config["num_past_beam_instances"],
                                                    config["num_imu_instances"])
                debug_print(f"Sample index {t}: input = {inp}, target = {tar}")
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
            epoch_squared_errors = np.zeros(3)
            for i in range(num_samples):
                model.train()
                x = torch.tensor(inputs[i], dtype=torch.float32, device=model.device).unsqueeze(0).unsqueeze(0)
                y = torch.tensor(targets[i], dtype=torch.float32, device=model.device)
                debug_print(f"Epoch {epoch}, Sample {i}: input = {x}, target = {y}")
                y_pred = model(x).squeeze().view(-1)
                debug_print(f"Epoch {epoch}, Sample {i}: prediction = {y_pred}")
                y = y.view(-1)
                sample_loss = loss_fn(y_pred, y)
                losses.append(sample_loss)
                error = (y_pred - y).detach().cpu().numpy() ** 2
                debug_print(f"Epoch {epoch}, Sample {i}: error = {error}")
                epoch_squared_errors += error
            epoch_loss = torch.stack(losses).sum()
            epoch_loss.backward()
            optimizer.step()
            epoch_rmse = np.sqrt(epoch_squared_errors / num_samples)
            evolution.append([epoch] + epoch_rmse.tolist())
            log_global(f"[{traj}] Epoch {epoch}: RMSE per output: {epoch_rmse}, Avg RMSE: {np.mean(epoch_rmse):.5f}")
        training_time = time.time() - t0
        
        # Evaluation on training data
        model.eval()
        final_predictions = []
        for i in range(num_samples):
            x = torch.tensor(inputs[i], dtype=torch.float32, device=model.device).unsqueeze(0).unsqueeze(0)
            y = torch.tensor(targets[i], dtype=torch.float32, device=model.device)
            with torch.no_grad():
                y_pred = model(x).squeeze().view(-1)
            final_predictions.append({
                "Sample": i,
                "Pred_V_North": y_pred[0].item(),
                "Pred_V_East": y_pred[1].item(),
                "Pred_V_Down": y_pred[2].item(),
                "GT_V_North": y[0].item(),
                "GT_V_East": y[1].item(),
                "GT_V_Down": y[2].item()
            })
        train_plot_path = os.path.join(PLOTS_DIR, f"FinalOutputPredictions_{sanitize(traj)}_{run_id}_range_removed.png")
        train_fig = plot_velocity_predictions(final_predictions, traj, config["beam_fill_window"], title_suffix="(Training)")
        train_fig.savefig(train_plot_path)
        plt.close(train_fig)
        log_global(f"[{traj}] Training predictions plot saved to {train_plot_path}")

        error_plot_path = os.path.join(PLOTS_DIR, f"SquaredError_{sanitize(traj)}_{run_id}_error.png")
        error_fig = plot_pred_error(final_predictions, traj, title_suffix="(Training)")
        error_fig.savefig(error_plot_path)
        plt.close(error_fig)
        log_global(f"[{traj}] Squared error plot saved to {error_plot_path}")

        current_trained_on = ",".join(trained_list) if trained_list else "NONE"
        summary = {
            "Trajectory": traj,
            "NumSamples": num_samples,
            "InputSize": input_size,
            "EpochsTrained": traj_epochs,
            "AvgBestRMSE": np.mean(evolution[-1][1:]),
            "TrainingTime": training_time,
            "TrainedOn": current_trained_on,
            "RemovalMethod": "Random"
        }
        global_training_summary.append(summary)
        trained_list.append(f"{traj}:{traj_epochs}")
    return global_training_summary

def test_on_trajectory(traj, config, checkpoint_filename, run_id, base_trained_on):
    traj_path = os.path.join(DATA_DIR, traj)
    beams_gt, beams_training, imu, velocity_gt = load_csv_files(traj_path)
    
    filled_beams_initial, _ = fill_missing_beams(beams_training, config["beam_fill_window"])
    min_history = find_first_valid_index(filled_beams_initial, beams_gt, imu,
                                           config["num_past_beam_instances"],
                                           config["num_imu_instances"],
                                           velocity_gt)
    if min_history is None:
        log_global("Not enough testing data in the trajectory to determine input size.")
        return None

    testing_removal_method = config.get("testing_removal_method", "range")
    if testing_removal_method == "random":
        beams_training_modified, count_removed = apply_random_removal(beams_training, config, min_history)
    else:
        beams_training_modified, count_removed = apply_range_removal(beams_training, config, min_history)
    log_global(f"Beam removal counts (testing): {count_removed}")
    
    filled_beams, _ = fill_missing_beams(beams_training_modified, config["beam_fill_window"])
    
    inputs, targets, masks = [], [], []
    for t in range(min_history, min(len(filled_beams), len(velocity_gt))):
        try:
            inp, tar = construct_input_target(filled_beams, velocity_gt, imu, t,
                                                config["num_past_beam_instances"],
                                                config["num_imu_instances"])
            debug_print(f"Testing sample index {t}: input = {inp}, target = {tar}")
            inputs.append(inp)
            targets.append(tar)
            masks.append(get_missing_mask(beams_training, t))
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
                                number_of_output_neurons=3,
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
    sample_errors = []  # Compute error per sample
    velocity_min, velocity_max = config.get("velocity_range", [-3, 3])
    for i in range(num_samples):
        x = torch.tensor(inputs[i], dtype=torch.float32, device=model.device).unsqueeze(0).unsqueeze(0)
        y = torch.tensor(targets[i], dtype=torch.float32, device=model.device).view(-1)
        debug_print(f"[Test] Sample {i}: input = {x}, target = {y}")
        with torch.no_grad():
            y_pred = model(x).squeeze().view(-1)
        debug_print(f"[Test] Sample {i}: prediction = {y_pred}")
        sample_pred = {
            "Sample": i,
            "Pred_V_North": y_pred[0].item(),
            "Pred_V_East": y_pred[1].item(),
            "Pred_V_Down": y_pred[2].item(),
            "GT_V_North": y[0].item(),
            "GT_V_East": y[1].item(),
            "GT_V_Down": y[2].item()
        }
        predictions.append(sample_pred)
        gt = y.detach().cpu().numpy()
        # Mark error as NaN if any GT component is out of range
        if (gt[0] < velocity_min or gt[0] > velocity_max or 
            gt[1] < velocity_min or gt[1] > velocity_max or 
            gt[2] < velocity_min or gt[2] > velocity_max):
            debug_print(f"[Test] Sample {i}: Ground truth {gt} out of range {velocity_min} to {velocity_max}. Error marked as NaN.")
            sample_errors.append(np.nan)
        else:
            error = (y_pred - y).detach().cpu().numpy() ** 2
            debug_print(f"[Test] Sample {i}: error = {error}")
            sample_errors.append(np.mean(error))
    if any(np.isnan(sample_errors)):
        overall_rmse = np.nan
    else:
        overall_rmse = np.sqrt(np.mean(sample_errors))
    log_global(f"[{traj}] Test RMSE per sample (NaN if any GT component out-of-range): {overall_rmse}")
    
    test_pred_csv = os.path.join(PREDICTIONS_DIR, f"TestPredictions_{sanitize(traj)}_{run_id}_range_removed.csv")
    pd.DataFrame(predictions).to_csv(test_pred_csv, index=False)
    log_global(f"[{traj}] Test predictions saved to {test_pred_csv}")
    
    plot_file = os.path.join(PLOTS_DIR, f"VelocityPredictions_{sanitize(traj)}_{run_id}_range_removed.png")
    fig = plot_velocity_predictions(predictions, traj, config["beam_fill_window"], title_suffix="(Testing Data)")
    fig.savefig(plot_file)
    plt.close(fig)
    log_global(f"[{traj}] Velocity predictions plot saved to {plot_file}")
    
    error_plot_path = os.path.join(PLOTS_DIR, f"SquaredError_{sanitize(traj)}_{run_id}_range_removed_error.png")
    error_fig = plot_pred_error(predictions, traj, title_suffix="(Testing Data)")
    error_fig.savefig(error_plot_path)
    plt.close(error_fig)
    log_global(f"[{traj}] Squared error plot saved to {error_plot_path}")
    
    test_summary = {
        "Trajectory": traj,
        "NumSamples": num_samples,
        "Test_RMSE": overall_rmse,
        "TrainedOn": base_trained_on,
        "RemovalMethod": testing_removal_method
    }
    return test_summary

def sequential_training_and_testing(config):
    with open(GLOBAL_LOG_FILE, "w") as f:
        f.write("Experiment Log\n")
    
    training_trajectory_pairs = config.get("training_trajectories", [])
    testing_list = config.get("testing_trajectories", [])
    
    if training_trajectory_pairs:
        removal_method = config.get("removal_method", "random")
        cumulative_trained_on = []  
        global_training_summary = []
        
        # Determine input size using the first training trajectory
        first_traj = training_trajectory_pairs[0][0]
        first_traj_path = os.path.join(DATA_DIR, first_traj)
        beams_gt, beams_training, imu, velocity_gt = load_csv_files(first_traj_path)
        filled_beams_initial, _ = fill_missing_beams(beams_training, config["beam_fill_window"])
        min_history = find_first_valid_index(filled_beams_initial, beams_gt, imu,
                                               config["num_past_beam_instances"],
                                               config["num_imu_instances"],
                                               velocity_gt)
        if min_history is None:
            log_global("Not enough training data in the first trajectory to determine input size.")
            return
        log_global(f"Using starting index {min_history} from the first trajectory.")
        
        inputs = []
        for t in range(min_history, min(len(filled_beams_initial), len(velocity_gt))):
            try:
                inp, _ = construct_input_target(filled_beams_initial, velocity_gt, imu, t,
                                                config["num_past_beam_instances"],
                                                config["num_imu_instances"])
                debug_print(f"Training sample index {t}: input = {inp}")
                inputs.append(inp)
                log_global(f"Constructed sample at index {t}.")
            except Exception as e:
                log_global(f"Skipping index {t} in training: {e}")
                continue

        if len(inputs) == 0:
            log_global("Not enough training data in the first trajectory to determine input size.")
            return
        input_size = np.array(inputs).shape[1]
        
        model = MemoryNeuralNetwork(number_of_input_neurons=input_size,
                                    number_of_hidden_neurons=config["hidden_neurons"],
                                    number_of_output_neurons=3,
                                    dropout_rate=config["dropout_rate"],
                                    learning_rate=config["learning_rate"],
                                    learning_rate_2=config["learning_rate_2"],
                                    lipschitz_constant=config["lipschitz_constant"])
        
        run_id = f"MNN_{sanitize(generate_first_level_folder_name(config))}_{sanitize(generate_second_level_folder_name(config))}"
        
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

        final_checkpoint_filename = f"MNN_{run_id}_{removal_method}_removed_final.pth"
        cp_path = os.path.join(CHECKPOINTS_DIR, final_checkpoint_filename)
        if os.path.exists(cp_path):
            log_global(f"Duplicate file found for configuration: {cp_path}. Aborting experiment run.")
            sys.exit(0)
        torch.save(model.state_dict(), cp_path)
        log_global(f"Final checkpoint saved to {cp_path}")
        
        train_summary_filename = f"GlobalTrainingSummary_{run_id}_{removal_method}_removed.csv"
        ts_path = os.path.join(TRAINING_SUMMARIES_DIR, train_summary_filename)
        if os.path.exists(ts_path):
            log_global(f"Duplicate file found for configuration: {ts_path}. Aborting experiment run.")
            sys.exit(0)
        pd.DataFrame(global_training_summary).to_csv(ts_path, index=False)
        log_global(f"Global training summary saved to {ts_path}")
    
    else:
        checkpoint_file = config.get("checkpoint_file")
        if not checkpoint_file or not os.path.exists(checkpoint_file):
            raise ValueError("Checkpoint file not provided in config or file does not exist.")
        final_checkpoint_filename = os.path.basename(checkpoint_file)
        log_global(f"No training trajectories provided. Loading checkpoint {checkpoint_file} for testing.")
        testing_list = config.get("testing_trajectories", [])
        if not testing_list:
            raise ValueError("No testing trajectories provided.")
        first_traj = testing_list[0]
        first_traj_path = os.path.join(DATA_DIR, first_traj)
        beams_gt, beams_training, imu, velocity_gt = load_csv_files(first_traj_path)
        filled_beams_initial, _ = fill_missing_beams(beams_training, config["beam_fill_window"])
        min_history = find_first_valid_index(filled_beams_initial, beams_gt, imu,
                                               config["num_past_beam_instances"],
                                               config["num_imu_instances"],
                                               velocity_gt)
        if min_history is None:
            log_global("Not enough testing data in the first trajectory to determine input size.")
            return
        log_global(f"Using starting index {min_history} from the first testing trajectory.")
        inputs = []
        for t in range(min_history, min(len(filled_beams_initial), len(velocity_gt))):
            try:
                inp, _ = construct_input_target(filled_beams_initial, velocity_gt, imu, t,
                                                config["num_past_beam_instances"],
                                                config["num_imu_instances"])
                debug_print(f"Testing sample index {t}: input = {inp}")
                inputs.append(inp)
                log_global(f"Constructed sample at index {t} for testing.")
            except Exception as e:
                log_global(f"Skipping index {t} in testing: {e}")
                continue
        if len(inputs) == 0:
            log_global("Not enough testing data in the first trajectory to determine input size.")
            return
        input_size = np.array(inputs).shape[1]
        
        model = MemoryNeuralNetwork(number_of_input_neurons=input_size,
                                    number_of_hidden_neurons=config["hidden_neurons"],
                                    number_of_output_neurons=3,
                                    dropout_rate=config["dropout_rate"])
        run_id = f"MNN_{sanitize(generate_first_level_folder_name(config))}_{sanitize(generate_second_level_folder_name(config))}"
    
    log_global("=== Testing Phase ===")
    global_test_summary = []
    base_trained_on = ",".join(cumulative_trained_on) if training_trajectory_pairs else "NONE"
    for traj in config.get("testing_trajectories", []):
        log_global(f"Processing testing trajectory: {traj}")
        test_summary = test_on_trajectory(traj, config, final_checkpoint_filename, run_id, base_trained_on)
        if test_summary:
            global_test_summary.append(test_summary)
    if global_test_summary:
        test_summary_filename = f"GlobalTestSummary_{run_id}_{config.get('testing_removal_method', 'range')}_removed.csv"
        tst_path = os.path.join(TEST_SUMMARIES_DIR, test_summary_filename)
        if os.path.exists(tst_path):
            log_global(f"Duplicate file found for configuration: {tst_path}. Aborting experiment run.")
            sys.exit(0)
        pd.DataFrame(global_test_summary).to_csv(tst_path, index=False)
        log_global(f"Global test summary saved to {tst_path}")

##########################
# Entry Point
##########################

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
    sequential_training_and_testing(config)
