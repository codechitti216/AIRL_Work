import torch
import pandas as pd
import numpy as np
import os
import glob
import time
import re
import sys
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Logging setup
log_file = "../../Experiments/experiment_log_LSTM.txt"
def log(message):
    print(message)
    with open(log_file, "a") as f:
        f.write(message + "\n")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Architecture_Codes')))
from LSTM import LSTMNetwork

# Hyperparameters
ID = 0
neeta = 1.2e-3
epochs = 75
stacking_count = 2  
log(f"Stacking count set to: {stacking_count}")

# Base paths
base_path = "../../../Data"
checkpoint_base = "../../Experiments/Results_New/LSTM/Checkpoints_LSTM"
results_file = "../../Experiments/Results_New/LSTM/Results_LSTM.csv"

# Create results DataFrame
if os.path.exists(results_file):
    results_df = pd.read_csv(results_file)
else:
    results_df = pd.DataFrame(columns=['ID', 'Trajectory', 'Learning Rate', 'Stacking Count', '6()', '4()', 'Best RMSE Loss', 'Time Taken', 'Epochs'])

file_pattern = re.compile(r'combined_(\d+)_(\d+)\.csv')
log("Starting experiment script...")

for traj_folder in os.listdir(base_path):
    traj_path = os.path.join(base_path, traj_folder)
    log(f"Checking directory: {traj_path}")

    if not os.path.isdir(traj_path):
        continue

    trajectory_id = int(''.join(filter(str.isdigit, traj_folder)))
    csv_files = glob.glob(os.path.join(traj_path, 'combined_*.csv'))

    for file_path in csv_files:
        log(f"Processing file: {file_path}")
        match = file_pattern.search(os.path.basename(file_path))
        if not match:
            continue

        a, b = int(match.group(1)), int(match.group(2))
        num_inputs = (6 * a) + (4 * b)
        log(f"Parsed values - a: {a}, b: {b}, input size: {num_inputs}")

        # Initialize LSTM Network
        lstm = LSTMNetwork(number_of_input_neurons=num_inputs, number_of_hidden_neurons=60, number_of_output_neurons=3, neeta=neeta)

        # Load dataset
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            log(f"Error loading {file_path}: {e}")
            continue

        input_columns = [f'ACC X_{i}' for i in range(a)] + [f'ACC Y_{i}' for i in range(a)] + [f'ACC Z_{i}' for i in range(a)] + \
                        [f'GYRO X_{i}' for i in range(a)] + [f'GYRO Y_{i}' for i in range(a)] + [f'GYRO Z_{i}' for i in range(a)] + \
                        [f'DVL{j}_1' for j in range(b)] + [f'DVL{j}_2' for j in range(b)] + [f'DVL{j}_3' for j in range(b)] + [f'DVL{j}_4' for j in range(b)]
        output_columns = ['V North', 'V East', 'V Down']

        if not set(input_columns).issubset(df.columns) or not set(output_columns).issubset(df.columns):
            log(f"Skipping {file_path} due to missing columns")
            continue

        input_data = df[input_columns].values[:-1]
        target_data = df[output_columns].shift(-1).dropna().values[:-1]

        input_data_stacked = np.tile(input_data, (stacking_count, 1))
        target_data_stacked = np.tile(target_data, (stacking_count, 1))
        
        train_samples = 2 * len(input_data_stacked) // 3
        X_train, y_train = input_data_stacked[:train_samples], target_data_stacked[:train_samples]

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(lstm.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(lstm.device)

        best_rmse = float('inf')
        losses = []
        checkpoint_path = f"{checkpoint_base}/Trajectory{trajectory_id}/checkpoint_{a}_{b}.pth"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        # Load checkpoint if exists
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            lstm.load_state_dict(checkpoint['model_state'])
            best_rmse = checkpoint['best_rmse']
            log(f"Resuming training from {checkpoint_path} with best RMSE: {best_rmse:.6f}")

        start_time = time.time()
        for epoch in range(epochs):
            error_sum = 0.0
            for i in range(len(X_train_tensor)):
                loss = lstm.backprop(X_train_tensor[i], y_train_tensor[i])
                error_sum += loss

            epoch_rmse = error_sum / len(X_train_tensor)
            losses.append(epoch_rmse)

            if epoch_rmse < best_rmse:
                best_rmse = epoch_rmse
                torch.save({'model_state': lstm.state_dict(), 'best_rmse': best_rmse}, checkpoint_path)
                log(f"Checkpoint saved: {checkpoint_path} (Best RMSE: {best_rmse:.6f})")

        time_taken = time.time() - start_time
        log(f"Training completed in {time_taken:.2f} seconds. Best RMSE: {best_rmse:.6f}")

        # Save RMSE plot
        plt.figure()
        plt.plot(range(len(losses)), losses, label='RMSE Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Loss (Trajectory {trajectory_id}, a={a}, b={b})')
        plt.legend()
        plt.savefig(f"../../Experiments/Results_New/LSTM/RMSE_Trajectory{trajectory_id}_{a}_{b}.png")
        plt.close()

        # Append results (eliminating duplicates)
        new_entry = pd.DataFrame([{'ID': ID, 'Trajectory': trajectory_id, 'Learning Rate': neeta, 'Stacking Count': stacking_count, '6()': a, '4()': b, 'Best RMSE Loss': best_rmse, 'Time Taken': time_taken, 'Epochs': epochs}])
        results_df = pd.concat([results_df, new_entry]).drop_duplicates(subset=['Trajectory', 'Learning Rate', 'Stacking Count', '6()', '4()', 'Epochs'], keep='last')

# Save final results
results_df.to_csv(results_file, index=False)
log("All experiments completed. Results saved.")
