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
VALID_BEAM_MIN = -1.1
VALID_BEAM_MAX = 1.5
# Required velocity columns (for reference – not used for inference)
REQUIRED_VELOCITY_COLS = ["V North", "V East", "V Down"]

def load_csv_files(traj_path):
    """
    Loads beams and IMU files from the trajectory folder.
    Converts time columns to strings and synchronizes beams and IMU using common timestamps.
    """
    beams_df = pd.read_csv(os.path.join(traj_path, "beams_gt.csv"), na_values=[''])
    imu_files = [f for f in os.listdir(traj_path) if f.startswith("IMU_") and f.endswith(".csv")]
    if not imu_files:
        raise ValueError(f"No IMU file found in {traj_path}")
    imu_df = pd.read_csv(os.path.join(traj_path, imu_files[0]))
    
    beams_df['Time'] = beams_df['Time'].astype(str)
    imu_df['Time'] = imu_df['Time [s]'].astype(str)
    
    common_times = set(beams_df['Time']) & set(imu_df['Time'])
    if not common_times:
        raise ValueError("No common timestamps found between beams and IMU data.")
    
    beams_df = beams_df[beams_df['Time'].isin(common_times)].sort_values("Time").reset_index(drop=True)
    imu_df = imu_df[imu_df['Time'].isin(common_times)].sort_values("Time").reset_index(drop=True)
    
    return beams_df, imu_df

def remove_invalid_beam_values(beams_df, beam_fill_window, beam_cols=["b1", "b2", "b3", "b4"]):
    """
    For rows with index >= beam_fill_window, mark beam values outside the valid range as NaN.
    Rows with index < beam_fill_window remain unmodified.
    """
    df = beams_df.copy()
    for idx in df.index:
        if idx >= beam_fill_window:
            for col in beam_cols:
                if (df.loc[idx, col] < VALID_BEAM_MIN) or (df.loc[idx, col] > VALID_BEAM_MAX):
                    df.loc[idx, col] = np.nan
    return df

def compute_switch_value(beams_df, beam_fill_window, beam_cols=["b1", "b2", "b3", "b4"]):
    """
    For rows with index >= beam_fill_window, compute the SWITCH VALUE.
    SWITCH VALUE is 0 if at least 2 beam values are missing (NaN), otherwise 1.
    For rows with index < beam_fill_window, the SWITCH VALUE remains 1 (default).
    """
    df = beams_df.copy()
    df["SWITCH VALUE"] = 1  # default for early rows
    for idx in df.index:
        if idx >= beam_fill_window:
            missing = df.loc[idx, beam_cols].isna().sum()
            df.loc[idx, "SWITCH VALUE"] = 0 if missing >= 2 else 1
    return df

def fill_missing_beams(beams_df, beam_fill_window, beam_cols=["b1", "b2", "b3", "b4"]):
    """
    For rows with index >= beam_fill_window, replace NaN beam values with the moving average computed
    from the previous beam_fill_window rows.
    
    If the window is entirely NaN, a forward-fill is used.
    Extra debugging is added: when the current row equals the specified debug index (e.g. 15), it prints detailed window information.
    Immediately exits if replacement still results in NaN.
    Rows with indices < beam_fill_window remain unchanged.
    """
    filled = beams_df.copy()
    for i in range(beam_fill_window, len(filled)):
        for col in beam_cols:
            if pd.isna(filled.loc[i, col]):
                window = filled.loc[i - beam_fill_window:i - 1, col]
                
                # Extra debugging for a specific row (e.g., row 15)
                if i == 15:
                    print(f"[DEBUG] For row {i} and column '{col}', history window details:")
                    for idx, val in window.items():
                        print(f"   Index {idx}: Value {val}")
                
                nan_indices = window.index[window.isna()].tolist()
                if nan_indices:
                    print(f"[DEBUG] For row {i} and column '{col}', NaN found in history at indices: {nan_indices}")
                
                if window.isna().all():
                    ffill_value = filled[col].ffill().iloc[i - 1]
                    print(f"[DEBUG] Row {i} column '{col}': Entire window is NaN. Forward-filling with value: {ffill_value}")
                    filled.loc[i, col] = ffill_value
                else:
                    mean_val = window.mean()
                    print(f"[DEBUG] Row {i} column '{col}': Replacing NaN with moving average: {mean_val}")
                    filled.loc[i, col] = mean_val
                    
                if pd.isna(filled.loc[i, col]):
                    print(f"[ERROR] Still NaN after filling '{col}' at row {i}. Exiting for debugging.")
                    sys.exit(1)
    return filled



def compute_first_valid_index(filled_beams, imu_df, num_past_beam_instances, num_imu_instances):
    """
    Computes the first index t (starting from max(num_past_beam_instances, num_imu_instances))
    where a complete, valid input sample can be constructed from the filled beam data.
    """
    start = max(num_past_beam_instances, num_imu_instances)
    for t in range(start, len(filled_beams)):
        try:
            _ = construct_input_sample(filled_beams, imu_df, t, num_past_beam_instances, num_imu_instances)
            return t
        except Exception as e:
            print(f"[DEBUG] Exception at index {t}: {e}")
            continue
    return None

def construct_input_sample(final_beams, imu_df, t, num_past_beam_instances, num_imu_instances):
    """
    Constructs an input vector at index t:
      - Current beam values from columns b1, b2, b3, and b4.
      - Past beams: concatenated values from the previous num_past_beam_instances rows.
      - Past IMU: concatenated values from the previous num_imu_instances rows.
    If there is insufficient history, zeros are padded.
    Exits if any required data is still NaN.
    """
    # Current beams
    current_beams = final_beams.loc[t, ["b1", "b2", "b3", "b4"]].values.astype(float)
    if np.isnan(current_beams).any():
        print(f"[ERROR] NaN detected in current beams at index {t}: {current_beams}")
        sys.exit(1)
    
    past_beams = []
    for i in range(1, num_past_beam_instances + 1):
        idx = t - i
        if idx < 0:
            past_beams.extend([0.0] * 4)
        else:
            row = final_beams.loc[idx, ["b1", "b2", "b3", "b4"]].values.astype(float)
            past_beams.extend(row)
    
    imu_cols = ['ACC X [m/s^2]', 'ACC Y [m/s^2]', 'ACC Z [m/s^2]',
                'GYRO X [rad/s]', 'GYRO Y [rad/s]', 'GYRO Z [rad/s]']
    past_imu = []
    for i in range(num_imu_instances):
        idx = t - i
        if idx < 0:
            past_imu.extend([0.0] * 6)
        else:
            row = imu_df.loc[idx, imu_cols].values.astype(float)
            past_imu.extend(row)
    
    input_vector = np.concatenate([current_beams, np.array(past_beams), np.array(past_imu)])
    if np.isnan(input_vector).any():
        print(f"[ERROR] NaN detected in final input vector at index {t}: {input_vector}")
        sys.exit(1)
    return input_vector

def run_inference(trajectory, config, checkpoint_path, output_csv="inference_results.csv"):
    """
    Runs inference on the given trajectory folder using the trained model checkpoint.
    
    Processing logic:
      1. Load the full dataset (all rows preserved).
      2. For rows with index >= beam_fill_window, mark invalid beam values as NaN,
         compute SWITCH VALUE, and apply fill_missing_beams.
         (Rows with index < beam_fill_window remain unmodified.)
      3. Compute first_valid_index from the filled data (starting from max(num_past_beam_instances, num_imu_instances)).
      4. Construct input samples for every row in the complete dataset.
         - For rows with index < first_valid_index, raw (unmodified) values are used.
         - For rows with index >= first_valid_index, the processed (filled) values are used.
         - Missing history is padded with zeros.
      5. Run inference on each input sample and save predictions (including Time and SWITCH VALUE) to CSV.
    """
    traj_path = os.path.join(DATA_DIR, trajectory)
    print(f"[INFO] Running inference on trajectory: {trajectory}")
    beams_df, imu_df = load_csv_files(traj_path)
    
    # Step 2: For rows with index >= beam_fill_window, process invalid beam removal and compute SWITCH VALUE.
    beam_fill_window = config.get("beam_fill_window", 5)
    processed_beams = beams_df.copy()
    if len(processed_beams) > beam_fill_window:
        proc_part = remove_invalid_beam_values(processed_beams.iloc[beam_fill_window:], beam_fill_window, beam_cols=["b1", "b2", "b3", "b4"])
        proc_part = compute_switch_value(proc_part, beam_fill_window, beam_cols=["b1", "b2", "b3", "b4"])
        proc_part = fill_missing_beams(proc_part, beam_fill_window, beam_cols=["b1", "b2", "b3", "b4"])
        processed_beams.iloc[beam_fill_window:] = proc_part
    else:
        processed_beams = compute_switch_value(processed_beams, beam_fill_window, beam_cols=["b1", "b2", "b3", "b4"])
    
    # Step 3: Compute the first valid index from the processed data.
    first_valid = compute_first_valid_index(processed_beams, imu_df,
                                            config["num_past_beam_instances"],
                                            config["num_imu_instances"])
    if first_valid is None:
        raise ValueError("No valid starting index found in trajectory " + trajectory)
    else:
        print(f"[INFO] First valid index determined: {first_valid}")
    
    # Step 4: Build the final beams dataframe.
    # For indices < first_valid, use raw beam values; for indices >= first_valid, use processed (filled) values.
    final_beams = beams_df.copy()
    final_beams.iloc[first_valid:] = processed_beams.iloc[first_valid:]
    
    # Step 5: Construct input samples for every row (with padding for missing history)
    num_samples = len(final_beams)
    inputs = []
    for t in range(num_samples):
        sample = construct_input_sample(final_beams, imu_df, t,
                                        config["num_past_beam_instances"],
                                        config["num_imu_instances"])
        inputs.append(sample)
    
    inputs = np.array(inputs)
    input_size = inputs.shape[1]
    
    # Step 6: Initialize the model and load the checkpoint.
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
    for t in range(num_samples):
        sample_time = beams_df.loc[t, "Time"]
        switch_value = beams_df.loc[t, "SWITCH VALUE"] if "SWITCH VALUE" in beams_df.columns else 1
        x = torch.tensor(inputs[t], dtype=torch.float32, device=model.device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            y_pred = model(x).squeeze().view(-1)
            print(f"[DEBUG] Input sample at index {t}: {inputs[t]}")
            print(f"[DEBUG] Model output at index {t}: {y_pred}")
            if torch.isnan(y_pred).any():
                print(f"[ERROR] NaN detected in prediction at index {t}. Exiting.")
                sys.exit(1)
            pred_velocities = y_pred.cpu().numpy().tolist()
            predictions.append({
                "Time": sample_time,
                "V_X": pred_velocities[0],
                "V_Y": pred_velocities[1],
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
