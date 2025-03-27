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
DATA_DIR = "../Data"
# Range for valid beam values
VALID_BEAM_MIN = -1.1
VALID_BEAM_MAX = 1.5
# Required velocity columns (for reference – not used for inference)
REQUIRED_VELOCITY_COLS = ["V North", "V East", "V Down"]

# Define the starting row for modifying beam values (i.e. 11th row, index 10)

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
    (This is applied to all rows for diagnostic purposes.)
    """
    beams_df = beams_df.copy()
    for col in beam_cols:
        beams_df.loc[(beams_df[col] < VALID_BEAM_MIN) | (beams_df[col] > VALID_BEAM_MAX), col] = np.nan
    return beams_df

def compute_switch_value(beams_df, beam_cols=["b1", "b2", "b3", "b4"]):
    """
    Computes the SWITCH VALUE for each row.
    Returns 0 if at least 2 beams are missing (NaN), otherwise 1.
    """
    beams_df["SWITCH VALUE"] = beams_df[beam_cols].isna().sum(axis=1).apply(lambda x: 0 if x >= 2 else 1)
    return beams_df

def fill_missing_beams(beams_df, beam_fill_window, beam_cols=["b1", "b2", "b3", "b4"]):
    """
    For rows starting at START_FILL_INDEX, replace NaN values with the moving average of the past beam_fill_window rows.
    If the window is entirely NaN, use forward-fill (last valid value).
    If the replacement is still NaN, exit immediately.
    """
    filled = beams_df.copy()
    for i in range(beam_fill_window + 3, len(filled)):
        for col in beam_cols:
            if pd.isna(filled.loc[i, col]):
                window = filled.loc[i - beam_fill_window:i - 1, col]
                print(f"[DEBUG] At index {i}, filling '{col}' using window values: {window.tolist()}")
                if window.isna().all():
                    ffill_value = filled[col].ffill().iloc[i - 1]
                    print(f"[DEBUG] Forward-filling '{col}' at index {i} with value: {ffill_value}")
                    filled.loc[i, col] = ffill_value
                else:
                    mean_val = window.mean()
                    print(f"[DEBUG] Filling '{col}' at index {i} with mean: {mean_val}")
                    filled.loc[i, col] = mean_val
                if pd.isna(filled.loc[i, col]):
                    print(f"[ERROR] Still NaN after filling '{col}' at index {i}. Exiting for debugging.")
                    sys.exit(1)
    return filled

def construct_input_sample(filled_beams, imu_df, t, num_past_beam_instances, num_imu_instances):
    """
    Constructs an input vector at index t:
      - Current beam values from columns b1, b2, b3, b4.
      - Past beams: concatenated values from the previous num_past_beam_instances rows.
      - Past IMU: concatenated values from the previous num_imu_instances rows for required IMU columns.
    Exits if any component contains NaN.
    """
    current_beams = filled_beams.loc[t, ["b1", "b2", "b3", "b4"]].values.astype(float)
    if np.isnan(current_beams).any():
        print(f"[ERROR] NaN detected in current beams at index {t}: {current_beams}")
        sys.exit(1)
        
    past_beams = []
    for i in range(1, num_past_beam_instances + 1):
        past_row = filled_beams.loc[t - i, ["b1", "b2", "b3", "b4"]].values.astype(float)
        if np.isnan(past_row).any():
            print(f"[ERROR] NaN detected in past beams at index {t - i}: {past_row}")
            sys.exit(1)
        past_beams.extend(past_row)
        
    imu_cols = ['ACC X [m/s^2]', 'ACC Y [m/s^2]', 'ACC Z [m/s^2]',
                'GYRO X [rad/s]', 'GYRO Y [rad/s]', 'GYRO Z [rad/s]']
    past_imu = []
    for i in range(num_imu_instances - 1, -1, -1):
        imu_values = imu_df.loc[t - i, imu_cols].values.astype(float)
        if np.isnan(imu_values).any():
            print(f"[ERROR] NaN detected in IMU values at index {t - i}: {imu_values}")
            sys.exit(1)
        past_imu.extend(imu_values)
        
    input_vector = np.concatenate([current_beams, np.array(past_beams), np.array(past_imu)])
    if np.isnan(input_vector).any():
        print(f"[ERROR] NaN detected in final input vector at index {t}: {input_vector}")
        sys.exit(1)
    return input_vector

def find_first_valid_index(filled_beams, imu_df, num_past_beam_instances, num_imu_instances):
    """
    Finds the first index t (starting from the minimal required index) where a complete, valid input sample can be constructed.
    Used for reporting/debugging purposes.
    """
    start = max(num_past_beam_instances, num_imu_instances - 1)
    for t in range(start, len(filled_beams)):
        try:
            _ = construct_input_sample(filled_beams, imu_df, t, num_past_beam_instances, num_imu_instances)
            return t
        except Exception as e:
            print(f"[DEBUG] Exception at index {t}: {e}")
            continue
    return None

START_FILL_INDEX = 10

def run_inference(trajectory, config, checkpoint_path, output_csv="inference_results.csv"):
    """
    Runs inference on the given trajectory folder using the trained model checkpoint.
    Processing logic:
      1. Load the full dataset (all rows preserved).
      2. Remove (mark as NaN) any invalid beam values and compute SWITCH VALUE.
      3. Starting from row index START_FILL_INDEX (i.e. the 11th row), replace invalid beam values with moving averages.
      4. Slice the dataset from row START_FILL_INDEX onward.
      5. Construct input samples from the sliced, filled data and run inference.
    """
    traj_path = os.path.join(DATA_DIR, trajectory)
    print(f"[INFO] Running inference on trajectory: {trajectory}")
    beams_df, imu_df = load_csv_files(traj_path)
    
    # Mark invalid beams as NaN and compute SWITCH VALUE for all rows.
    beams_df = remove_invalid_beam_values(beams_df)
    beams_df = compute_switch_value(beams_df)
    
    # Fill missing beam values only starting from START_FILL_INDEX (i.e. from the 11th row onward)
    beam_fill_window = config.get("beam_fill_window", 5)
    filled_beams = fill_missing_beams(beams_df, beam_fill_window)
    
    # Now, only keep rows from START_FILL_INDEX onward for inference.
    print(f"[INFO] Slicing data from row {START_FILL_INDEX} onward for inference.")
    filled_beams = filled_beams.iloc[START_FILL_INDEX:].reset_index(drop=True)
    imu_df = imu_df.iloc[START_FILL_INDEX:].reset_index(drop=True)
    
    first_valid = find_first_valid_index(filled_beams, imu_df,
                                           config["num_past_beam_instances"],
                                           config["num_imu_instances"])
    if first_valid is None:
        raise ValueError("No valid starting index found in trajectory " + trajectory)
    else:
        print(f"[INFO] First valid index (in sliced data): {first_valid}")
    
    start_t = max(config["num_past_beam_instances"], config["num_imu_instances"] - 1)
    num_samples = len(filled_beams)
    inputs = []
    valid_indices = []
    for t in range(start_t, num_samples):
        sample = construct_input_sample(filled_beams, imu_df, t,
                                        config["num_past_beam_instances"],
                                        config["num_imu_instances"])
        inputs.append(sample)
        valid_indices.append(t)
    
    if len(inputs) == 0:
        raise ValueError("No valid inference samples in trajectory " + trajectory)
    
    inputs = np.array(inputs)
    input_size = inputs.shape[1]
    
    model = MemoryNeuralNetwork(number_of_input_neurons=input_size,
                                number_of_hidden_neurons=config["hidden_neurons"],
                                number_of_output_neurons=3,  # Predicting V North, V East, V Down
                                dropout_rate=config["dropout_rate"],
                                learning_rate=config["learning_rate"],
                                learning_rate_2=config["learning_rate_2"],
                                lipschitz_constant=config["lipschitz_constant"])
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=model.device))
    model.eval()
    
    predictions = []
    for idx, sample in enumerate(inputs):
        sample_time = filled_beams.loc[valid_indices[idx], "Time"]
        switch_value = filled_beams.loc[valid_indices[idx], "SWITCH VALUE"]
        x = torch.tensor(sample, dtype=torch.float32, device=model.device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            y_pred = model(x).squeeze().view(-1)
            print(f"[DEBUG] Input sample at index {valid_indices[idx]}: {sample}")
            print(f"[DEBUG] Model output: {y_pred}")
            if torch.isnan(y_pred).any():
                print(f"[ERROR] NaN detected in prediction at index {valid_indices[idx]}. Exiting.")
                sys.exit(1)
            pred_velocities = y_pred.cpu().numpy().tolist()
            predictions.append({
                "Time": sample_time,
                "V_X": pred_velocities[0],
                "V Y": pred_velocities[1],
                "V_Z": pred_velocities[2],
                "SWITCH VALUE": switch_value
            })
    
    output_path = os.path.join(os.getcwd(), output_csv)
    pd.DataFrame(predictions).to_csv(output_path, index=False)
    print(f"[INFO] Inference complete. Predictions saved to {output_path}")

def main():
    with open("MNN.json", "r") as f:
        config = json.load(f)
    
    trajectory = "Trajectory15"  # Modify as needed
    checkpoint_path = "checkpoint.pth"  # Update with the correct checkpoint filename
    run_inference(trajectory, config, checkpoint_path)

if __name__ == "__main__":
    main()
