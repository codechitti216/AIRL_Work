import os
import pandas as pd
import numpy as np

def calculate_relative_changes(df, columns):
    """
    Calculate absolute differences for specified columns and remove first row
    """
    # Create a copy of the dataframe to avoid modifying the original
    df_new = df.copy()
    
    # Calculate differences for each column
    for col in columns:
        # Calculate differences between consecutive values
        df_new[col] = df_new[col].diff()
    
    # Remove first row (contains NaN due to diff)
    df_new = df_new.iloc[1:].reset_index(drop=True)
    
    return df_new

def process_trajectory(traj_path):
    """
    Process DVL and GT data for a single trajectory
    """
    # Get trajectory number from folder name (handles both single and double digit numbers)
    traj_name = os.path.basename(traj_path)
    traj_num = ''.join(filter(str.isdigit, traj_name))
    
    # Process DVL data
    dvl_file = os.path.join(traj_path, f"DVL_trajectory{traj_num}.csv")
    if os.path.exists(dvl_file):
        dvl_data = pd.read_csv(dvl_file)
        dvl_columns = ['DVL X [m/s]', 'DVL Y [m/s]', 'DVL Z [m/s]']
        dvl_data = calculate_relative_changes(dvl_data, dvl_columns)
        dvl_data.to_csv(dvl_file, index=False)
        print(f"Processed DVL data in {traj_name}")
    
    # Process GT data
    gt_file = os.path.join(traj_path, f"GT_trajectory{traj_num}.csv")
    if os.path.exists(gt_file):
        gt_data = pd.read_csv(gt_file)
        gt_columns = ['V North [m/s]', 'V East [m/s]', 'V Down [m/s]']
        gt_data = calculate_relative_changes(gt_data, gt_columns)
        gt_data.to_csv(gt_file, index=False)
        print(f"Processed GT data in {traj_name}")

def main():
    # Get the data directory path (current directory where the script is located)
    data_folder = os.path.dirname(os.path.abspath(__file__))
    
    # Process each trajectory folder
    for item in os.listdir(data_folder):
        traj_path = os.path.join(data_folder, item)
        if os.path.isdir(traj_path) and item.startswith('Trajectory'):
            print(f"\nProcessing {item}...")
            process_trajectory(traj_path)

if __name__ == "__main__":
    main()