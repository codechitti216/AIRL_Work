import torch
import pandas as pd
import numpy as np
import os
import glob
import time
import re
from sklearn.metrics import mean_squared_error
import sys
import random

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Architecture_Codes')))
from Memory_Neural_Network import MemoryNeuralNetwork
from LSTM import LSTMNetwork

ID = 0

neeta = 1.2e-3
neeta_dash = 5e-4
lipschitz_constant = 1.2
epochs = 70

# User-defined stacking count
stacking_count = 5  # Change this value as needed
print(f"Stacking count set to: {stacking_count}")

base_path = "../../../Data"

results_df = pd.DataFrame(columns=['ID', 'Trajectory', 'Learning Rate 1', 'Learning Rate 2', 
                                   'Stacking Count', '6()', '4()', 'Best RMSE Loss', 'Time Taken', 'Epochs'])

file_pattern = re.compile(r'combined_(\d+)_(\d+)\.csv')

print("Starting experiment script...")

for traj_folder in os.listdir(base_path):
    traj_path = os.path.join(base_path, traj_folder)
    print(f"Checking directory: {traj_path}")

    if not os.path.isdir(traj_path):
        print(f"Skipping {traj_path} (not a directory)")
        continue

    csv_files = glob.glob(os.path.join(traj_path, 'combined_*.csv'))
    print(f"Found {len(csv_files)} CSV files in {traj_path}")

    if not csv_files:
        print(f"No CSV files found in {traj_path}")
        continue

    for file_path in csv_files:
        print(f"Processing file: {file_path}")
        match = file_pattern.search(os.path.basename(file_path))
        if not match:
            print(f"Skipping {file_path} (filename doesn't match expected pattern)")
            continue

        a, b = int(match.group(1)), int(match.group(2))
        num_inputs = (6 * a) + (4 * b)
        print(f"Parsed values - a: {a}, b: {b}, input size: {num_inputs}")

        print("Initializing Memory Neural Network...")
        mnn = MemoryNeuralNetwork(number_of_input_neurons=num_inputs, number_of_output_neurons=3, 
                                  neeta=neeta, neeta_dash=neeta_dash, lipschitz_norm=lipschitz_constant, 
                                  spectral_norm=True)

        try:
            df = pd.read_csv(file_path)
            print(f"Dataset loaded. Shape: {df.shape}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

        input_columns = [item for i in range(a) for item in [f'ACC X_{i}', f'ACC Y_{i}', f'ACC Z_{i}', f'GYRO X_{i}', f'GYRO Y_{i}', f'GYRO Z_{i}']]
        input_columns += [item for j in range(b) for item in [f'DVL{j}_1', f'DVL{j}_2', f'DVL{j}_3', f'DVL{j}_4']]
        output_columns = ['V North', 'V East', 'V Down']

        missing_inputs = [col for col in input_columns if col not in df.columns]
        missing_outputs = [col for col in output_columns if col not in df.columns]
        if missing_inputs or missing_outputs:
            print(f"Missing columns in {file_path}: Inputs {missing_inputs}, Outputs {missing_outputs}")
            continue

        input_data = df[input_columns].values
        target_data = df[output_columns].shift(-1).dropna().values  # Shift target data by one row

        # Ensure input and target data have the same length
        input_data = input_data[:-1]
        target_data = target_data[:-1]

        # Stack the input data along the batch dimension
        input_data_stacked = np.tile(input_data, (stacking_count, 1))
        target_data_stacked = np.tile(target_data, (stacking_count, 1))
        print(f"Input data stacked {stacking_count} times along the batch dimension. New shape: {input_data_stacked.shape}")

        num_samples = len(input_data_stacked)
        train_samples = 2 * num_samples // 3
        X_train, y_train = input_data_stacked[:train_samples], target_data_stacked[:train_samples]
        X_test, y_test = input_data_stacked[train_samples:], target_data_stacked[train_samples:]

        print(f"Dataset split: {train_samples} training samples, {num_samples - train_samples} testing samples")

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        best_rmse = float('inf')
        start_time = time.time()
        print(f"Training started for {file_path}")

        for epoch in range(epochs):
            error_sum = 0.0

            for i in range(len(X_train)):
                pred = mnn.feedforward(X_train[i, :])
                mnn.backprop(y_train[i, :])
                mse = mean_squared_error(y_train[i, :], pred.cpu().detach().numpy())
                error_sum += mse
                if i % 100 == 0:
                    print(f"Epoch {epoch + 1}, Sample {i}: MSE = {mse:.6f}")

            epoch_error = error_sum / len(X_train)
            if epoch_error < best_rmse:
                best_rmse = epoch_error

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Best RMSE so far: {best_rmse:.6f}")

        time_taken = time.time() - start_time
        print(f"Training completed for {file_path} in {time_taken:.2f} seconds. Best RMSE: {best_rmse:.6f}")

        ID += 1

        results_df = pd.concat([results_df, pd.DataFrame([{
            'ID': ID,
            'Trajectory': int(''.join(filter(str.isdigit, traj_folder))),
            'Learning Rate 1': neeta,
            'Learning Rate 2': neeta_dash,
            'Stacking Count': stacking_count,
            '6()': a,
            '4()': b,
            'Best RMSE Loss': best_rmse,
            'Time Taken': time_taken,
            'Epochs' : epochs
        }])], ignore_index=True)

results_df.to_csv(f'../../../Scripts/Experiments/Results/MNN_{epochs}_experiment_results.csv', index=False)
print("All experiments completed. Results saved in 'MNN_100_experiment_results.csv'.")