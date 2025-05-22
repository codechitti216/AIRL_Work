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
    Process DVL beams and GT data for a single trajectory
    """
    # Get trajectory number from folder name (handles both single and double digit numbers)
    traj_name = os.path.basename(traj_path)
    traj_num = ''.join(filter(str.isdigit, traj_name))
    
    # Process DVL beams data
    beams_file = os.path.join(traj_path, "beams_gt.csv")
    if os.path.exists(beams_file):
        beams_data = pd.read_csv(beams_file)
        beams_columns = ['b1', 'b2', 'b3', 'b4']
        beams_data = calculate_relative_changes(beams_data, beams_columns)
        beams_data.to_csv(beams_file, index=False)
        print(f"Processed beams data in {traj_name}")
    
    # Process GT data
    velocity_gt_file = os.path.join(traj_path, "velocity_gt.csv")
    if os.path.exists(velocity_gt_file):
        gt_data = pd.read_csv(velocity_gt_file)
        gt_columns = ['V North', 'V East', 'V Down']
        gt_data = calculate_relative_changes(gt_data, gt_columns)
        gt_data.to_csv(velocity_gt_file, index=False)
        print(f"Processed velocity data in {traj_name}")

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