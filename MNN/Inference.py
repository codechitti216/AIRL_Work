#!/usr/bin/env python3
"""
MNN_inference.py

This script loads a trained MNN checkpoint and runs inference on a specified trajectory.
Functionality:
  - Loads beams, IMU, and Time from the trajectory folder.
  - Synchronizes the data based on common timestamps.
  - Starting from the first valid index (where enough history exists), it checks each beam:
      if a beam value is outside the range [-1.5, 1.5], it is set to NaN.
  - Missing values are then filled progressively using the moving average of the past beam_fill_window rows.
  - Constructs input samples (using current beams, past beams, and past IMU data) and runs inference.
  - Saves the predicted velocities along with the corresponding Time into a CSV file.
  
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

from MNN import MemoryNeuralNetwork

# Data directory (assumed relative)
DATA_DIR = "../../Data"
# Range for valid beam values
VALID_BEAM_MIN = -1.5
VALID_BEAM_MAX = 1.5
# Required velocity columns (for reference – not used for inference)
REQUIRED_VELOCITY_COLS = ["V North", "V East", "V Down"]

def sanitize(text):
    text = str(text).replace(":", "")
    text = re.sub(r'[^\w_.-]', '', text)
    return text

def load_csv_files(traj_path):
    """
    Loads beams, velocity, and IMU files from the trajectory folder.
    Although velocity is not used for inference inputs,
    it may be loaded if needed; here we focus on beams and IMU.
    """
    beams_df = pd.read_csv(os.path.join(traj_path, "beams_gt.csv"), na_values=[''])
    # Load the first IMU CSV file found
    imu_files = [f for f in os.listdir(traj_path) if f.startswith("IMU_") and f.endswith(".csv")]
    if not imu_files:
        raise ValueError(f"No IMU file found in {traj_path}")
    imu_df = pd.read_csv(os.path.join(traj_path, imu_files[0]))
    
    # Convert time columns to string (assuming beams has "Time" and IMU has "Time [s]")
    beams_df['Time'] = beams_df['Time'].astype(str)
    imu_df['Time'] = imu_df['Time [s]'].astype(str)
    
    # Synchronize using common timestamps
    common_times = set(beams_df['Time']) & set(imu_df['Time'])
    if not common_times:
        raise ValueError("No common timestamps found between beams and IMU data.")
    
    beams_df = beams_df[beams_df['Time'].isin(common_times)].sort_values("Time").reset_index(drop=True)
    imu_df = imu_df[imu_df['Time'].isin(common_times)].sort_values("Time").reset_index(drop=True)
    
    return beams_df, imu_df

def remove_invalid_beam_values(beams_df, beam_cols=["b1", "b2", "b3", "b4"]):
    """
    For each beam column, if a value is outside the valid range, set it to NaN.
    """
    beams_df = beams_df.copy()
    for col in beam_cols:
        beams_df.loc[(beams_df[col] < VALID_BEAM_MIN) | (beams_df[col] > VALID_BEAM_MAX), col] = np.nan
    return beams_df

def fill_missing_beams(beams_df, beam_fill_window, beam_cols=["b1", "b2", "b3", "b4"]):
    """
    Replace NaN values with the moving average of the past beam_fill_window rows.
    If all previous values are missing, use forward-fill (last valid value).
    """
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

def construct_input_sample(filled_beams, imu_df, t, num_past_beam_instances, num_imu_instances):
    """
    Constructs an input vector at index t.
      - Current beam values from columns b1, b2, b3, b4.
      - Past beams: concatenated values from previous num_past_beam_instances rows.
      - Past IMU: concatenated values from previous num_imu_instances rows for required IMU columns.
    """
    # Current beams
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
    return input_vector

def find_first_valid_index(filled_beams, imu_df, num_past_beam_instances, num_imu_instances):
    """
    Find the first index t where there is sufficient history to construct an input sample.
    """
    start = max(num_past_beam_instances, num_imu_instances - 1)
    for t in range(start, len(filled_beams)):
        try:
            _ = construct_input_sample(filled_beams, imu_df, t, num_past_beam_instances, num_imu_instances)
            return t
        except Exception:
            continue
    return None

def run_inference(trajectory, config, checkpoint_path, output_csv="inference_results.csv"):
    """
    Runs inference on the given trajectory folder using the trained model checkpoint.
    Saves the predicted velocities along with the Time stamp.
    """
    traj_path = os.path.join(DATA_DIR, trajectory)
    print(f"[INFO] Running inference on trajectory: {trajectory}")
    beams_df, imu_df = load_csv_files(traj_path)
    
    # Remove invalid beam values (outside [-1.5, 1.5])
    beams_df = remove_invalid_beam_values(beams_df)
    
    # Fill missing values progressively using moving average
    beam_fill_window = config.get("beam_fill_window", 5)
    filled_beams = fill_missing_beams(beams_df, beam_fill_window)
    
    # Find first valid index (to have enough history)
    first_valid = find_first_valid_index(filled_beams, imu_df,
                                           config["num_past_beam_instances"],
                                           config["num_imu_instances"])
    if first_valid is None:
        raise ValueError("No valid starting index found in trajectory " + trajectory)
    
    # Slice data from the first valid index onward (reset index but keep Time column)
    filled_beams = filled_beams.iloc[first_valid:].reset_index(drop=True)
    imu_df = imu_df.iloc[first_valid:].reset_index(drop=True)
    time_series = filled_beams["Time"].tolist()
    
    # Construct input samples and run inference
    num_samples = len(filled_beams)
    inputs = []
    valid_indices = []
    start_t = max(config["num_past_beam_instances"], config["num_imu_instances"] - 1)
    for t in range(start_t, num_samples):
        try:
            sample = construct_input_sample(filled_beams, imu_df, t,
                                            config["num_past_beam_instances"],
                                            config["num_imu_instances"])
            inputs.append(sample)
            valid_indices.append(t)
        except Exception as e:
            print(f"[WARNING] Skipping index {t}: {e}")
            continue

    if len(inputs) == 0:
        raise ValueError("No valid inference samples in trajectory " + trajectory)
    
    inputs = np.array(inputs)
    
    # Determine input size
    input_size = inputs.shape[1]
    
    # Initialize the model with the same architecture used in training
    model = MemoryNeuralNetwork(number_of_input_neurons=input_size,
                                number_of_hidden_neurons=config["hidden_neurons"],
                                number_of_output_neurons=3,  # Predicting V North, V East, V Down
                                dropout_rate=config["dropout_rate"],
                                learning_rate=config["learning_rate"],
                                learning_rate_2=config["learning_rate_2"],
                                lipschitz_constant=config["lipschitz_constant"])
    
    # Load the trained checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=model.device))
    model.eval()
    
    predictions = []
    for idx, sample in enumerate(inputs):
        # Use the corresponding Time from the filled_beams; note that valid_indices offset the row index
        sample_time = filled_beams.loc[valid_indices[idx], "Time"]
        x = torch.tensor(sample, dtype=torch.float32, device=model.device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            y_pred = model(x).squeeze().view(-1)
        pred_velocities = y_pred.cpu().numpy().tolist()
        predictions.append({
            "Time": sample_time,
            "V North": pred_velocities[0],
            "V East": pred_velocities[1],
            "V Down": pred_velocities[2]
        })
    
    # Save predictions to CSV
    output_path = os.path.join(os.getcwd(), output_csv)
    pd.DataFrame(predictions).to_csv(output_path, index=False)
    print(f"[INFO] Inference complete. Predictions saved to {output_path}")

def main():
    # Load configuration
    with open("MNN.json", "r") as f:
        config = json.load(f)
    
    # Specify the trajectory to run inference on (e.g., "Trajectory1" or "Train Trajectory1")
    trajectory = "Trajectory14"  # Modify as needed
    
    # Path to the trained checkpoint (modify if necessary)
    checkpoint_path = "checkpoint.pth"  # Update with the correct checkpoint filename
    
    run_inference(trajectory, config, checkpoint_path)

if __name__ == "__main__":
    main()
