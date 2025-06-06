#!/usr/bin/env python3
"""
MNN_experiment.py

This script trains the Memory Neural Network (MNN) on all specified training trajectories,
processing each trajectory by:
  - Loading beams, velocity, and IMU files from the trajectory folder.
  - Synchronizing them based on common timestamps.
  - Starting from the first valid index, applying random removal to beam values based on percentages
    specified in the JSON configuration.
  - Immediately replacing removed beam values (NaN) with the moving average of the past beam_fill_window values.
  - Constructing training pairs where the inputs are the processed beams (and historical data) and
    targets are the velocity components.
  - Training the model sequentially over trajectories.
  
Additionally, after each training epoch the script calculates and logs the RMSE per output and the average RMSE.
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

# Data directory
DATA_DIR = "../../Data"
# Required columns in velocity CSV
REQUIRED_VELOCITY_COLS = ["V North", "V East", "V Down"]

def load_csv_files(traj_path):
    print(f"[INFO] Loading CSV files from trajectory folder: {traj_path}")
    beams_df = pd.read_csv(os.path.join(traj_path, "beams_gt.csv"), na_values=[''])
    velocity_df = pd.read_csv(os.path.join(traj_path, "velocity_gt.csv"), na_values=[''])
    
    if not all(col in velocity_df.columns for col in REQUIRED_VELOCITY_COLS):
        raise ValueError(f"velocity_gt file missing required columns. Expected: {REQUIRED_VELOCITY_COLS}, Found: {list(velocity_df.columns)}")
    
    imu_files = [f for f in os.listdir(traj_path) if f.startswith("IMU_") and f.endswith(".csv")]
    if not imu_files:
        raise ValueError(f"No IMU file found in {traj_path}")
    imu_df = pd.read_csv(os.path.join(traj_path, imu_files[0]))
    
    # Convert time columns to strings for synchronization
    beams_df['Time'] = beams_df['Time'].astype(str)
    velocity_df['Time'] = velocity_df['Time'].astype(str)
    imu_df['Time'] = imu_df['Time [s]'].astype(str)
    
    # Find common timestamps across all three dataframes
    common_times = set(beams_df['Time']) & set(velocity_df['Time']) & set(imu_df['Time'])
    if not common_times:
        raise ValueError("No common timestamps found across beams, velocity, and IMU data.")
    
    beams_df = beams_df[beams_df['Time'].isin(common_times)].sort_values("Time").reset_index(drop=True)
    velocity_df = velocity_df[velocity_df['Time'].isin(common_times)].sort_values("Time").reset_index(drop=True)
    imu_df = imu_df[imu_df['Time'].isin(common_times)].sort_values("Time").reset_index(drop=True)
    
    print(f"[INFO] Loaded and synchronized data: beams={beams_df.shape}, velocity={velocity_df.shape}, imu={imu_df.shape}")
    return beams_df, imu_df, velocity_df

def fill_missing_beams(beams_df, beam_fill_window, beam_cols=["b1", "b2", "b3", "b4"]):
    filled = beams_df.copy()
    for i in range(beam_fill_window, len(filled)):
        for col in beam_cols:
            if pd.isna(filled.loc[i, col]):
                window = filled.loc[i - beam_fill_window:i - 1, col]
                if window.isna().all():
                    filled.loc[i, col] = filled[col].ffill().iloc[i - 1]
                else:
                    filled.loc[i, col] = window.mean()
    return filled

def apply_random_removal(beams_df, config, start_idx):
    percentages = config.get("random_removal_percentages", {"b1": 0, "b2": 0, "b3": 0, "b4": 0})
    beam_fill_window = config.get("beam_fill_window", 5)
    for beam in ['b1', 'b2', 'b3', 'b4']:
        candidate_indices = []
        for i in beams_df.index[start_idx:]:
            if pd.notna(beams_df.loc[i, beam]):
                if i - beam_fill_window >= 0:
                    window = beams_df.loc[i - beam_fill_window:i - 1, beam]
                    if not window.isna().any():
                        candidate_indices.append(i)
        num_candidates = len(candidate_indices)
        num_to_remove = int(num_candidates * (percentages.get(beam, 0) / 100.0))
        if num_to_remove > 0 and candidate_indices:
            remove_indices = np.random.choice(candidate_indices, size=num_to_remove, replace=False)
            beams_df.loc[remove_indices, beam] = np.nan
    # Replace removed values with moving averages
    beams_df = fill_missing_beams(beams_df, beam_fill_window)
    return beams_df

def construct_input_target(filled_beams, velocity_df, imu_df, t, num_past_beam_instances, num_imu_instances):
    current_beams = filled_beams.loc[t, ["b1", "b2", "b3", "b4"]].values.astype(float)
    past_beams = []
    for i in range(1, num_past_beam_instances + 1):
        past_beams.extend(filled_beams.loc[t - i, ["b1", "b2", "b3", "b4"]].values.astype(float))
    imu_cols = ['ACC X [m/s^2]', 'ACC Y [m/s^2]', 'ACC Z [m/s^2]',
                'GYRO X [rad/s]', 'GYRO Y [rad/s]', 'GYRO Z [rad/s]']
    past_imu = []
    for i in range(num_imu_instances - 1, -1, -1):
        past_imu.extend(imu_df.loc[t - i, imu_cols].values.astype(float))
    input_vector = np.concatenate([current_beams, np.array(past_beams), np.array(past_imu)])
    target_vector = velocity_df.loc[t, REQUIRED_VELOCITY_COLS].values.astype(float)
    return input_vector, target_vector

def find_first_valid_index(beams_df, velocity_df, imu_df, num_past_beam_instances, num_imu_instances):
    start = max(num_past_beam_instances, num_imu_instances - 1)
    for t in range(start, min(len(beams_df), len(velocity_df))):
        try:
            _ = construct_input_target(beams_df, velocity_df, imu_df, t, num_past_beam_instances, num_imu_instances)
            return t
        except Exception:
            continue
    return None

def process_training_trajectory(traj, traj_epochs, config):
    traj_path = os.path.join(DATA_DIR, traj)
    print(f"[INFO] Training on Trajectory: {traj} for {traj_epochs} epoch(s)")
    
    beams_df, imu_df, velocity_df = load_csv_files(traj_path)
    beams_filled = fill_missing_beams(beams_df, config["beam_fill_window"])
    first_valid = find_first_valid_index(beams_filled, velocity_df, imu_df,
                                           config["num_past_beam_instances"],
                                           config["num_imu_instances"])
    if first_valid is None:
        raise ValueError("No valid starting index found in trajectory " + traj)
    
    beams_filled = beams_filled.iloc[first_valid:].reset_index(drop=True)
    velocity_df = velocity_df.iloc[first_valid:].reset_index(drop=True)
    imu_df = imu_df.iloc[first_valid:].reset_index(drop=True)
    
    beams_filled = apply_random_removal(beams_filled, config, 0)
    
    inputs, targets = [], []
    start_t = max(config["num_past_beam_instances"], config["num_imu_instances"] - 1)
    print(f"[INFO] Constructing input-target pairs from trajectory: {traj}")
    for t in tqdm(range(start_t, min(len(beams_filled), len(velocity_df))), desc=f"[{traj}] Samples"):

        try:
            inp, tar = construct_input_target(beams_filled, velocity_df, imu_df, t,
                                              config["num_past_beam_instances"],
                                              config["num_imu_instances"])
            inputs.append(inp)
            targets.append(tar)
        except Exception as e:
            print(f"[WARNING] Skipping index {t}: {e}")
            continue

    if len(inputs) == 0:
        raise ValueError("No valid training samples in trajectory " + traj)
    
    inputs = np.array(inputs)
    targets = np.array(targets)
    return inputs, targets

def check_model_params(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"[DEBUG] NaN detected in parameter: {name}")
        else:
            print(f"[DEBUG] Parameter {name} norm: {param.norm().item():.5f}")

def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"[DEBUG] No gradient for parameter: {name}")
        elif torch.isnan(param.grad).any():
            print(f"[DEBUG] NaN detected in gradient of: {name}")
        else:
            print(f"[DEBUG] Gradient norm for {name}: {param.grad.norm().item():.5f}")

from tqdm import tqdm
import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def train_model_on_trajectory(model, inputs, targets, traj_name, traj_epochs, learning_rate):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_samples = inputs.shape[0]

    results_dir = 'Results'
    os.makedirs(results_dir, exist_ok=True)

    for epoch in range(traj_epochs):
        model.train()
        epoch_predictions = []
        epoch_targets = []

        epoch_squared_error = np.zeros(3)  # For V_North, V_East, V_Down

        with tqdm(total=num_samples, desc=f"[{traj_name}] Epoch {epoch + 1}/{traj_epochs}", leave=False) as pbar:
            for i in range(num_samples):
                input_tensor = torch.tensor(inputs[i], dtype=torch.float32, device=model.device).unsqueeze(0)
                target_tensor = torch.tensor(targets[i], dtype=torch.float32, device=model.device).unsqueeze(0)

                optimizer.zero_grad()
                output = model(input_tensor)
                loss = criterion(output, target_tensor)
                loss.backward()
                optimizer.step()

                if epoch == traj_epochs - 1:
                    epoch_predictions.append(output.detach().cpu().numpy().flatten())
                    epoch_targets.append(target_tensor.cpu().numpy().flatten())

                # Accumulate squared error
                error = (output.detach().cpu().numpy().flatten() - target_tensor.cpu().numpy().flatten()) ** 2
                epoch_squared_error += error

                pbar.update(1)

        epoch_rmse = np.sqrt(epoch_squared_error / num_samples)
        tqdm.write(f"[{traj_name}] Epoch {epoch + 1}/{traj_epochs} RMSE: V_North={epoch_rmse[0]:.4f}, V_East={epoch_rmse[1]:.4f}, V_Down={epoch_rmse[2]:.4f}")

    # Final epoch: plot results
    epoch_predictions = np.array(epoch_predictions)
    epoch_targets = np.array(epoch_targets)
    time_axis = np.arange(len(epoch_predictions))
    components = ['V_North', 'V_East', 'V_Down']

    plt.figure(figsize=(15, 10))
    for idx in range(3):
        plt.subplot(3, 1, idx + 1)
        plt.plot(time_axis, epoch_targets[:, idx], label='Ground Truth', color='blue')
        plt.plot(time_axis, epoch_predictions[:, idx], label='Predicted', color="magenta")
        plt.title(f'{components[idx]}: Predicted vs Ground Truth')
        plt.xlabel('Sample Index')
        plt.ylabel('Velocity (m/s)')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    fig_path1 = os.path.join(results_dir, f'{traj_name}_final_epoch_comparison.png')
    plt.savefig(fig_path1)
    plt.close()
    print(f"[INFO] Saved: {fig_path1}")

    # -------- Plot 2: RMSE over time (per sample) --------
    rmse_over_time = np.abs(epoch_predictions - epoch_targets)
    plt.figure(figsize=(15, 10))
    for idx in range(3):
        plt.subplot(3, 1, idx + 1)
        plt.plot(time_axis, rmse_over_time[:, idx], color="red")
        plt.title(f'{components[idx]} RMSE over Time')
        plt.xlabel('Sample Index')
        plt.ylabel('RMSE (m/s)')
        plt.grid(True)

    plt.tight_layout()
    fig_path2 = os.path.join(results_dir, f'{traj_name}_final_epoch_rmse_curve.png')
    plt.savefig(fig_path2)
    plt.close()
    print(f"[INFO] Saved: {fig_path2}")

    return model

def main():
    with open("MNN.json", "r") as f:
        config = json.load(f)
    
    training_trajectory_pairs = config.get("training_trajectories", [])
    if not training_trajectory_pairs:
        print("No training trajectories provided in configuration.")
        sys.exit(1)

    first_traj = training_trajectory_pairs[0][0]
    beams_df, imu_df, velocity_df = load_csv_files(os.path.join(DATA_DIR, first_traj))
    beams_filled = fill_missing_beams(beams_df, config["beam_fill_window"])
    first_valid = find_first_valid_index(beams_filled, velocity_df, imu_df,
                                         config["num_past_beam_instances"],
                                         config["num_imu_instances"])
    if first_valid is None:
        print("Not enough valid training data in the first trajectory.")
        sys.exit(1)

    sample_inp, _ = construct_input_target(beams_filled, velocity_df, imu_df, first_valid,
                                           config["num_past_beam_instances"],
                                           config["num_imu_instances"])
    input_size = sample_inp.shape[0]

    model = MemoryNeuralNetwork(
        number_of_input_neurons=input_size,
        number_of_hidden_neurons=config["hidden_neurons"],
        number_of_output_neurons=3,
        dropout_rate=config["dropout_rate"],
        learning_rate=config["learning_rate"],
        learning_rate_2=config["learning_rate_2"],
        lipschitz_constant=config["lipschitz_constant"]
    )

    for traj, traj_epochs in training_trajectory_pairs:
        inputs, targets = process_training_trajectory(traj, traj_epochs, config)
        model = train_model_on_trajectory(model, inputs, targets, traj, traj_epochs, config["learning_rate"])

    # 🔒 Save ONE final checkpoint
    os.makedirs("Checkpoints", exist_ok=True)
    checkpoint_path = os.path.join("Checkpoints", "final_model_checkpoint.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"[INFO] Final model checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
