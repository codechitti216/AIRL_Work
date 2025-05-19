import os
import sys
import pandas as pd
import numpy as np  # Added for random selection

DATA_FOLDER = "../Data/"

# Hard-coded input file paths
DVL_INPUT_FILE = "../../Confidential/DVL_time_range_05_58_to_06_08.csv"
IMU_INPUT_FILE = "../../Confidential/IMU_time_range_05_58_to_06_08.csv"

def get_latest_trajectory_number():
    """
    Finds the highest trajectory number in the Data folder.
    """
    if not os.path.exists(DATA_FOLDER):
        print(f"DEBUG: Data folder '{DATA_FOLDER}' does not exist.")
        return 0

    trajectory_folders = [f for f in os.listdir(DATA_FOLDER) if f.startswith("Trajectory")]
    print(f"DEBUG: Found trajectory folders: {trajectory_folders}")

    trajectory_numbers = []
    for folder in trajectory_folders:
        num_str = folder[len("Trajectory"):]
        try:
            num = int(num_str)
            trajectory_numbers.append(num)
        except ValueError:
            print(f"DEBUG: Skipping folder '{folder}' due to invalid number '{num_str}'.")
            continue

    print(f"DEBUG: Extracted trajectory numbers: {trajectory_numbers}")
    return max(trajectory_numbers, default=0)

def corrupt_beam_data(df, beam_cols, corruption_value=-32.76481, corruption_ratio=0.1):
    """
    Randomly replaces 10% of the data points in each beam column with a corruption value.
    """
    df_corrupted = df.copy()
    for col in beam_cols:
        total_rows = len(df_corrupted)
        num_corrupt = int(total_rows * corruption_ratio)
        corrupt_indices = np.random.choice(df_corrupted.index, num_corrupt, replace=False)
        df_corrupted.loc[corrupt_indices, col] = corruption_value
        print(f"DEBUG: Corrupted {num_corrupt} values in column '{col}' with {corruption_value}")
    return df_corrupted

def process_trajectory(latest_traj_num, dvl_input_file, imu_input_file):
    """
    Processes the DVL and IMU CSV files.
    """
    new_traj_num = latest_traj_num + 1
    new_traj_path = os.path.join(DATA_FOLDER, f"Trajectory{new_traj_num}")
    print(f"DEBUG: Creating new trajectory folder: {new_traj_path}")
    os.makedirs(new_traj_path, exist_ok=True)

    # --- Process DVL file for beams_gt.csv and velocity_gt.csv ---
    print(f"DEBUG: Using DVL file: {dvl_input_file}")
    if not os.path.exists(dvl_input_file):
        raise FileNotFoundError(f"DEBUG: DVL file '{dvl_input_file}' is missing.")
    dvl_df = pd.read_csv(dvl_input_file)
    print("DEBUG: DVL CSV read successfully.")

    # Validate required columns for beams output
    required_beams_cols = {"Time", "b_vel0", "b_vel1", "b_vel2", "b_vel3"}
    missing_beams_cols = required_beams_cols - set(dvl_df.columns)
    if missing_beams_cols:
        raise ValueError(f"DEBUG: Missing required beams columns in DVL file '{dvl_input_file}'. Missing columns: {missing_beams_cols}")

    # Corrupt beam data with 10% replaced by -32.76481
    beam_columns = ["b_vel0", "b_vel1", "b_vel2", "b_vel3"]
    dvl_df = corrupt_beam_data(dvl_df, beam_columns)

    # Create output with renamed columns
    beams_output = dvl_df[["Time"] + beam_columns].rename(
        columns={"b_vel0": "b1", "b_vel1": "b2", "b_vel2": "b3", "b_vel3": "b4"}
    )
    beams_output_path = os.path.join(new_traj_path, "beams_gt.csv")
    beams_output.to_csv(beams_output_path, index=False)
    print(f"DEBUG: Processed and corrupted beams data saved to: {beams_output_path}")

    # Validate required columns for velocity output
    required_velocity_cols = {"Time", "vel_x", "vel_y", "vel_z1"}
    missing_velocity_cols = required_velocity_cols - set(dvl_df.columns)
    if missing_velocity_cols:
        raise ValueError(f"DEBUG: Missing required velocity columns in DVL file '{dvl_input_file}'. Missing columns: {missing_velocity_cols}")

    velocity_output = dvl_df[["Time", "vel_x", "vel_y", "vel_z1"]].rename(
        columns={"vel_x": "V North", "vel_y": "V East", "vel_z1": "V Down"}
    )
    velocity_output_path = os.path.join(new_traj_path, "velocity_gt.csv")
    velocity_output.to_csv(velocity_output_path, index=False)
    print(f"DEBUG: Processed velocity data saved to: {velocity_output_path}")

    # --- Process IMU file for IMU_trajectory_<n>.csv ---
    print(f"DEBUG: Using IMU file: {imu_input_file}")
    if not os.path.exists(imu_input_file):
        raise FileNotFoundError(f"DEBUG: IMU file '{imu_input_file}' is missing.")
    imu_df = pd.read_csv(imu_input_file)
    print("DEBUG: IMU CSV read successfully.")
    
    required_imu_cols = {"Time", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"}
    missing_imu_cols = required_imu_cols - set(imu_df.columns)
    if missing_imu_cols:
        raise ValueError(f"DEBUG: Missing required IMU columns in IMU file '{imu_input_file}'. Missing columns: {missing_imu_cols}")

    imu_output = imu_df[["Time", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]].rename(
        columns={
            "Time": "Time [s]",
            "acc_x": "ACC X [m/s^2]",
            "acc_y": "ACC Y [m/s^2]",
            "acc_z": "ACC Z [m/s^2]",
            "gyro_x": "GYRO X [rad/s]",
            "gyro_y": "GYRO Y [rad/s]",
            "gyro_z": "GYRO Z [rad/s]"
        }
    )
    imu_output.sort_values(by="Time [s]", inplace=True)
    imu_output_path = os.path.join(new_traj_path, f"IMU_trajectory{new_traj_num}.csv")
    imu_output.to_csv(imu_output_path, index=False)
    print(f"DEBUG: Processed IMU trajectory data saved to: {imu_output_path}")

    print(f"Processed data saved in {new_traj_path}")

if __name__ == "__main__":
    latest_number = get_latest_trajectory_number()
    if latest_number == 0:
        print("No existing trajectory found. Please ensure the Data folder contains valid trajectory folders.")
        process_trajectory(latest_number, DVL_INPUT_FILE, IMU_INPUT_FILE)
    else:
        print(f"DEBUG: Latest trajectory number found: {latest_number}")
        process_trajectory(latest_number, DVL_INPUT_FILE, IMU_INPUT_FILE)
