import os
import subprocess
import pandas as pd
import re
import time
import torch
import traceback
from termcolor import colored
from tqdm import tqdm

data_path = "../../../Data"
results_file = "../../Experiments/Results.csv"

def print_timestamp():
    return time.strftime('%Y-%m-%d %H:%M:%S')

def check_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(colored(f"[{print_timestamp()}] Using device: {device}", 'cyan'))
    torch.cuda.empty_cache()
    return device

def print_status(message, color='white'):
    print(colored(f"[{print_timestamp()}] {message}", color))

# Ensure results file exists
if not os.path.exists(results_file):
    results_df = pd.DataFrame(columns=[
        'Model Type', 'Learning Rate', 'Dropout Rate', 'Trajectory ID', 'Hidden Neurons',
        'Loss Function', 'Regularization', '6()', '4()', 'Epochs', 'Stacking Count', 'Time Taken', 'Best RMSE'
    ])
    results_df.to_csv(results_file, index=False)
else:
    results_df = pd.read_csv(results_file)

file_pattern = re.compile(r'combined_(\d+)_(\d+)\.csv')
trajectory_pattern = re.compile(r'Trajectory(\d+)')

experiment_scripts = [
    ("LSTM_experiment.py", "LSTM"),
    ("MNN_experiment.py", "MNN"),
    ("FAN_experiment.py", "FAN")
]

def run_experiment(script, model_type, traj_id, a, b):
    """Runs the given experiment script and logs results."""
    print_status(f"Running {model_type} experiment for Trajectory {traj_id}, a={a}, b={b}", 'green')
    check_device()
    try:
        print_status(f"Starting subprocess for {model_type}...", 'cyan')
        result = subprocess.run(["python", script], capture_output=True, text=True)
        print_status(f"Subprocess completed for {model_type}. Checking output...", 'cyan')
        print(result.stdout)
        if result.stderr:
            print_status(f"Error in {model_type}: {result.stderr}", 'red')
        return extract_results(model_type, traj_id, a, b)
    except Exception as e:
        print_status(f"Error running {script}: {str(e)}", 'red')
        traceback.print_exc()
    return None

def extract_results(model_type, traj_id, a, b):
    """Extracts the latest results from the model-specific results file."""
    model_results_path = f"../../Experiments/Results_New/{model_type}/Results.csv"
    if os.path.exists(model_results_path):
        df = pd.read_csv(model_results_path)
        df_filtered = df[(df['Trajectory ID'] == traj_id) & (df['6()'] == a) & (df['4()'] == b)]
        if not df_filtered.empty:
            best_row = df_filtered.sort_values(by='Best RMSE').iloc[0]
            return best_row.to_dict()
    return None

def update_results(new_entry):
    """Updates the results CSV with new experiment results, ensuring duplicates are handled correctly."""
    global results_df
    duplicate_mask = (
        (results_df['Trajectory ID'] == new_entry['Trajectory ID']) &
        (results_df['6()'] == new_entry['6()']) &
        (results_df['4()'] == new_entry['4()']) &
        (results_df['Learning Rate'] == new_entry['Learning Rate']) &
        (results_df['Dropout Rate'] == new_entry['Dropout Rate']) &
        (results_df['Hidden Neurons'] == new_entry['Hidden Neurons']) &
        (results_df['Stacking Count'] == new_entry['Stacking Count']) &
        (results_df['Loss Function'] == new_entry['Loss Function']) &
        (results_df['Regularization'] == new_entry['Regularization'])
    )
    if not results_df[duplicate_mask].empty:
        print_status("Skipping duplicate entry in Results.csv", 'yellow')
        return

    results_df = pd.concat([results_df, pd.DataFrame([new_entry])], ignore_index=True)
    try:
        results_df.to_csv(results_file, index=False)
        print_status("Results successfully logged in Results.csv", 'green')
    except Exception as e:
        print_status(f"Warning: Issue writing to CSV - {str(e)}", 'yellow')

# Main execution loop
start_time = time.time()
for traj_folder in os.listdir(data_path):
    traj_path = os.path.join(data_path, traj_folder)
    if not os.path.isdir(traj_path):
        continue
    match = trajectory_pattern.search(traj_folder)
    if not match:
        print_status(f"Skipping {traj_folder} (invalid trajectory format)", 'yellow')
        continue
    trajectory_id = int(match.group(1))
    print_status(f"Processing Trajectory: {trajectory_id}", 'cyan')
    all_files = os.listdir(traj_path)
    print_status(f"Files found in {traj_folder}: {all_files}", 'cyan')
    csv_files = [f for f in all_files if file_pattern.match(f)]
    if not csv_files:
        print_status(f"No valid combination files found in {traj_folder}.", 'yellow')
    else:
        print_status(f"Valid combination files detected: {csv_files}", 'green')
        for file_name in csv_files:
            match = file_pattern.search(file_name)
            if match:
                a, b = int(match.group(1)), int(match.group(2))
                print_status(f"Processing file: {file_name} with a={a}, b={b}", 'cyan')
                for script, model_type in experiment_scripts:
                    print_status(f"Starting training for {model_type} on {file_name}", 'green')
                    run_experiment(script, model_type, trajectory_id, a, b)

print_status("All experiments completed successfully.", 'green')
end_time = time.time()
print_status(f"Total execution time: {end_time - start_time:.2f} seconds", 'cyan')
