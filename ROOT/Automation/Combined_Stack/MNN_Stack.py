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

# Import the Memory Neural Network model
from MNN import MemoryNeuralNetwork

# Data directory
DATA_DIR = "../../Data"
# Required columns in velocity CSV
REQUIRED_VELOCITY_COLS = ["V North", "V East", "V Down"]

class BeamVelocityStack:
    def __init__(self, config):
        self.config = config
        self.beam_model = None
        self.velocity_model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def load_csv_files(self, traj_path):
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

    def fill_missing_beams(self, beams_df, beam_fill_window, beam_cols=["b1", "b2", "b3", "b4"]):
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

    def apply_random_removal(self, beams_df, config, start_idx):
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
        beams_df = self.fill_missing_beams(beams_df, beam_fill_window)
        return beams_df

    def construct_beam_input_target(self, filled_beams, imu_df, t, num_past_beam_instances, num_imu_instances):
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
        target_vector = filled_beams.loc[t, ["b1", "b2", "b3", "b4"]].values.astype(float)
        return input_vector, target_vector

    def construct_velocity_input_target(self, predicted_beams, imu_df, velocity_df, t, num_past_beam_instances, num_imu_instances):
        current_beams = predicted_beams[t]
        past_beams = []
        for i in range(1, num_past_beam_instances + 1):
            past_beams.extend(predicted_beams[t - i])
        imu_cols = ['ACC X [m/s^2]', 'ACC Y [m/s^2]', 'ACC Z [m/s^2]',
                    'GYRO X [rad/s]', 'GYRO Y [rad/s]', 'GYRO Z [rad/s]']
        past_imu = []
        for i in range(num_imu_instances - 1, -1, -1):
            past_imu.extend(imu_df.loc[t - i, imu_cols].values.astype(float))
        input_vector = np.concatenate([current_beams, np.array(past_beams), np.array(past_imu)])
        target_vector = velocity_df.loc[t, REQUIRED_VELOCITY_COLS].values.astype(float)
        return input_vector, target_vector

    def find_first_valid_index(self, beams_df, imu_df, num_past_beam_instances, num_imu_instances):
        start = max(num_past_beam_instances, num_imu_instances - 1)
        for t in range(start, len(beams_df)):
            try:
                _ = self.construct_beam_input_target(beams_df, imu_df, t, num_past_beam_instances, num_imu_instances)
                return t
            except Exception:
                continue
        return None

    def train_beam_model(self, beams_df, imu_df, traj_name, traj_epochs):
        print(f"[INFO] Training beam prediction model on trajectory: {traj_name}")
        
        # Initialize beam model
        input_size = 4 + (4 * self.config["num_past_beam_instances"]) + (6 * self.config["num_imu_instances"])
        self.beam_model = MemoryNeuralNetwork(
            number_of_input_neurons=input_size,
            number_of_hidden_neurons=self.config["hidden_neurons"],
            number_of_output_neurons=4,  # 4 beam values
            dropout_rate=self.config["dropout_rate"],
            learning_rate=self.config["learning_rate"],
            learning_rate_2=self.config["learning_rate_2"],
            lipschitz_constant=self.config["lipschitz_constant"]
        ).to(self.device)

        # Prepare training data
        first_valid = self.find_first_valid_index(beams_df, imu_df, 
                                                self.config["num_past_beam_instances"],
                                                self.config["num_imu_instances"])
        if first_valid is None:
            raise ValueError("No valid starting index found in trajectory " + traj_name)

        beams_df = beams_df.iloc[first_valid:].reset_index(drop=True)
        imu_df = imu_df.iloc[first_valid:].reset_index(drop=True)

        inputs, targets = [], []
        start_t = max(self.config["num_past_beam_instances"], self.config["num_imu_instances"] - 1)
        
        for t in range(start_t, len(beams_df)):
            try:
                inp, tar = self.construct_beam_input_target(beams_df, imu_df, t,
                                                          self.config["num_past_beam_instances"],
                                                          self.config["num_imu_instances"])
                inputs.append(inp)
                targets.append(tar)
            except Exception as e:
                print(f"[WARNING] Skipping index {t}: {e}")
                continue

        if len(inputs) == 0:
            raise ValueError("No valid training samples in trajectory " + traj_name)

        inputs = np.array(inputs)
        targets = np.array(targets)

        # Train beam model
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.beam_model.parameters(), lr=self.config["learning_rate"])

        for epoch in range(traj_epochs):
            self.beam_model.train()
            epoch_loss = 0
            for i in range(len(inputs)):
                input_tensor = torch.tensor(inputs[i], dtype=torch.float32, device=self.device).unsqueeze(0)
                target_tensor = torch.tensor(targets[i], dtype=torch.float32, device=self.device).unsqueeze(0)

                optimizer.zero_grad()
                output = self.beam_model(input_tensor)
                loss = criterion(output, target_tensor)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(inputs)
            print(f"[{traj_name}] Beam Model Epoch {epoch + 1}/{traj_epochs} Loss: {avg_loss:.6f}")

        return self.beam_model

    def train_velocity_model(self, predicted_beams, imu_df, velocity_df, traj_name, traj_epochs):
        print(f"[INFO] Training velocity prediction model on trajectory: {traj_name}")
        
        # Initialize velocity model
        input_size = 4 + (4 * self.config["num_past_beam_instances"]) + (6 * self.config["num_imu_instances"])
        self.velocity_model = MemoryNeuralNetwork(
            number_of_input_neurons=input_size,
            number_of_hidden_neurons=self.config["hidden_neurons"],
            number_of_output_neurons=3,  # 3 velocity components
            dropout_rate=self.config["dropout_rate"],
            learning_rate=self.config["learning_rate"],
            learning_rate_2=self.config["learning_rate_2"],
            lipschitz_constant=self.config["lipschitz_constant"]
        ).to(self.device)

        # Prepare training data
        inputs, targets = [], []
        start_t = max(self.config["num_past_beam_instances"], self.config["num_imu_instances"] - 1)
        
        for t in range(start_t, len(predicted_beams)):
            try:
                inp, tar = self.construct_velocity_input_target(predicted_beams, imu_df, velocity_df, t,
                                                             self.config["num_past_beam_instances"],
                                                             self.config["num_imu_instances"])
                inputs.append(inp)
                targets.append(tar)
            except Exception as e:
                print(f"[WARNING] Skipping index {t}: {e}")
                continue

        if len(inputs) == 0:
            raise ValueError("No valid training samples in trajectory " + traj_name)

        inputs = np.array(inputs)
        targets = np.array(targets)

        # Train velocity model
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.velocity_model.parameters(), lr=self.config["learning_rate"])

        for epoch in range(traj_epochs):
            self.velocity_model.train()
            epoch_loss = 0
            for i in range(len(inputs)):
                input_tensor = torch.tensor(inputs[i], dtype=torch.float32, device=self.device).unsqueeze(0)
                target_tensor = torch.tensor(targets[i], dtype=torch.float32, device=self.device).unsqueeze(0)

                optimizer.zero_grad()
                output = self.velocity_model(input_tensor)
                loss = criterion(output, target_tensor)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(inputs)
            print(f"[{traj_name}] Velocity Model Epoch {epoch + 1}/{traj_epochs} Loss: {avg_loss:.6f}")

        return self.velocity_model

    def predict(self, beams_df, imu_df):
        """Predict both beams and velocities for new data"""
        self.beam_model.eval()
        self.velocity_model.eval()
        
        # First predict beams
        predicted_beams = []
        start_t = max(self.config["num_past_beam_instances"], self.config["num_imu_instances"] - 1)
        
        with torch.no_grad():
            for t in range(start_t, len(beams_df)):
                try:
                    inp, _ = self.construct_beam_input_target(beams_df, imu_df, t,
                                                            self.config["num_past_beam_instances"],
                                                            self.config["num_imu_instances"])
                    input_tensor = torch.tensor(inp, dtype=torch.float32, device=self.device).unsqueeze(0)
                    beam_pred = self.beam_model(input_tensor).cpu().numpy().flatten()
                    predicted_beams.append(beam_pred)
                except Exception as e:
                    print(f"[WARNING] Skipping beam prediction at index {t}: {e}")
                    continue

        # Then predict velocities using predicted beams
        predicted_velocities = []
        for t in range(start_t, len(predicted_beams)):
            try:
                inp, _ = self.construct_velocity_input_target(predicted_beams, imu_df, None, t,
                                                           self.config["num_past_beam_instances"],
                                                           self.config["num_imu_instances"])
                input_tensor = torch.tensor(inp, dtype=torch.float32, device=self.device).unsqueeze(0)
                vel_pred = self.velocity_model(input_tensor).cpu().numpy().flatten()
                predicted_velocities.append(vel_pred)
            except Exception as e:
                print(f"[WARNING] Skipping velocity prediction at index {t}: {e}")
                continue

        # Save predictions to CSV files
        beam_predictions = pd.DataFrame(predicted_beams, columns=['b1', 'b2', 'b3', 'b4'])
        beam_predictions['Time'] = beams_df['Time'].iloc[start_t:start_t + len(predicted_beams)].values
        beam_predictions = beam_predictions[['Time', 'b1', 'b2', 'b3', 'b4']]
        beam_predictions.to_csv('predicted_beams.csv', index=False)

        velocity_predictions = pd.DataFrame(predicted_velocities, columns=['V North', 'V East', 'V Down'])
        velocity_predictions['Time'] = beams_df['Time'].iloc[start_t:start_t + len(predicted_velocities)].values
        velocity_predictions = velocity_predictions[['Time', 'V North', 'V East', 'V Down']]
        velocity_predictions.to_csv('predicted_velocities.csv', index=False)

        return np.array(predicted_beams), np.array(predicted_velocities)

    def save_models(self, path):
        """Save both models"""
        os.makedirs(path, exist_ok=True)
        torch.save(self.beam_model.state_dict(), os.path.join(path, "beam_model.pth"))
        torch.save(self.velocity_model.state_dict(), os.path.join(path, "velocity_model.pth"))

    def load_models(self, path):
        """Load both models"""
        self.beam_model.load_state_dict(torch.load(os.path.join(path, "beam_model.pth")))
        self.velocity_model.load_state_dict(torch.load(os.path.join(path, "velocity_model.pth"))) 