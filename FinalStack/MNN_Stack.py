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
DATA_DIR = "../Data_XYZ_change"
# Required columns in velocity CSV
REQUIRED_VELOCITY_COLS = ["V North", "V East", "V Down"]

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, yhat, y):
        # Increase epsilon to prevent numerical instability
        return torch.sqrt(self.mse(yhat, y) + 1e-6)

class BeamVelocityStack:
    def __init__(self, config):
        self.config = config
        
        # Set device based on config
        if "force_cpu" in self.config and self.config["force_cpu"]:
            self.device = torch.device("cpu")
            print("[INFO] Forcing CPU usage based on configuration.")
        elif "device" in self.config and self.config["device"] != "auto":
            # Use specific device from config
            try:
                self.device = torch.device(self.config["device"])
                print(f"[INFO] Using device specified in config: {self.device}")
            except Exception as e:
                print(f"[WARNING] Error setting device from config: {e}. Falling back to auto-detection.")
                self._auto_detect_device()
        else:
            # Auto-detect best device
            self._auto_detect_device()
            
        # Initialize models
        self.beam_model = None
        self.velocity_model = None
        
        # Storage for predictions and original data
        self.beam_predictions = {}  # Store final beam predictions for each trajectory
        self.original_beams = {}    # Store original beams for each trajectory
        
        # Additional storage for two-phase training
        self.hybrid_beams = {}      # Store hybrid beams (original + predictions) for each trajectory
        self.imu_data = {}          # Store IMU data for reuse in velocity training
        self.velocity_data = {}     # Store velocity data for reuse in velocity training
        
        print(f"[INFO] Initialized BeamVelocityStack on device: {self.device}")
        
    def _auto_detect_device(self):
        """Auto-detect the best available device"""
        self.device = torch.device("cpu")
        try:
            if torch.cuda.is_available():
                device_id = 0
                self.device = torch.device(f"cuda:{device_id}")
                print(f"[INFO] CUDA is available! Using GPU: {torch.cuda.get_device_name(device_id)}")
                print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(device_id).total_memory / 1e9:.2f} GB")
            else:
                print("[INFO] CUDA is not available. Using CPU.")
        except Exception as e:
            print(f"[WARNING] Error initializing CUDA: {e}. Falling back to CPU.")
            self.device = torch.device("cpu")
        
    def load_csv_files(self, traj_path):
        """Standard load method for backward compatibility"""
        beams_df, imu_df, velocity_df, original_beams, _ = self.load_csv_files_enhanced(traj_path)
        return beams_df, imu_df, velocity_df, original_beams
        
    def load_csv_files_enhanced(self, traj_path):
        """Load and synchronize CSV files with enhanced error handling"""
        try:
            # Get the absolute path to the data directory
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'FinalStack', 'Data_XYZ_change')
            traj_path = os.path.join(data_dir, os.path.basename(traj_path))
            
            print(f"[INFO] Loading CSV files from trajectory folder: {traj_path}")
            
            # Load beams data
            beams_df = pd.read_csv(os.path.join(traj_path, "beams_gt.csv"), na_values=[''])
            imu_df = pd.read_csv(os.path.join(traj_path, f"IMU_{os.path.basename(traj_path)}.csv"), na_values=[''])
            velocity_df = pd.read_csv(os.path.join(traj_path, "velocity_gt.csv"), na_values=[''])
            
            # Store original beams for comparison
            original_beams = beams_df.copy()
            
            # Create missed beams DataFrame
            missed_beams = beams_df.copy()
            missed_beams[['b1', 'b2', 'b3', 'b4']] = np.nan
            
            # Fill missing beams with moving averages
            beams_df = self.fill_missing_beams(beams_df, self.config["beam_fill_window"])
            
            print(f"[INFO] Loaded and synchronized data: beams={beams_df.shape}, velocity={velocity_df.shape}, imu={imu_df.shape}")
            return beams_df, imu_df, velocity_df, original_beams, missed_beams
            
        except Exception as e:
            print(f"[ERROR] Failed to load data from {traj_path}: {e}")
            return None, None, None, None, None

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

    def construct_beam_input_target(self, beams_df, imu_df, original_beams, t, num_past_beam_instances, num_imu_instances):
        current_beams = beams_df.loc[t, ["b1", "b2", "b3", "b4"]].values.astype(float)
        past_beams = []
        for i in range(1, num_past_beam_instances + 1):
            past_beams.extend(beams_df.loc[t - i, ["b1", "b2", "b3", "b4"]].values.astype(float))
        imu_cols = ['ACC X [m/s^2]', 'ACC Y [m/s^2]', 'ACC Z [m/s^2]',
                    'GYRO X [rad/s]', 'GYRO Y [rad/s]', 'GYRO Z [rad/s]']
        # Only use current IMU
        current_imu = imu_df.loc[t, imu_cols].values.astype(float)
        input_vector = np.concatenate([current_beams, np.array(past_beams), current_imu])
        
        if original_beams is not None:
            target_vector = original_beams.loc[t, ["b1", "b2", "b3", "b4"]].values.astype(float)
        else:
            target_vector = None
        return input_vector, target_vector

    def construct_velocity_input_target(self, beams_df, hybrid_beams, imu_df, velocity_df, t, num_past_beam_instances, num_past_imu_instances):
        """Construct input vector for velocity model and target velocity."""
        # Get current IMU data
        imu_cols = ['ACC X [m/s^2]', 'ACC Y [m/s^2]', 'ACC Z [m/s^2]',
                    'GYRO X [rad/s]', 'GYRO Y [rad/s]', 'GYRO Z [rad/s]']
        current_imu = imu_df.iloc[t][imu_cols].values.astype(float)
        
        # Get past beam data from hybrid beams
        past_beams = []
        for i in range(num_past_beam_instances):
            past_idx = t - (i + 1)
            if past_idx >= 0:
                past_beam = hybrid_beams.iloc[past_idx][["b1", "b2", "b3", "b4"]].values.astype(float)
                past_beams.append(past_beam)
            else:
                # Pad with zeros if not enough past data
                past_beams.append(np.zeros(4))
        
        # Flatten past beams
        past_beams = np.array(past_beams).flatten()
        
        # Get current beam data
        current_beam = hybrid_beams.iloc[t][["b1", "b2", "b3", "b4"]].values.astype(float)
        
        # Combine all features
        input_vector = np.concatenate([current_beam, past_beams, current_imu])
        
        # Get target velocity
        vel_cols = ['V North', 'V East', 'V Down']
        target = velocity_df.iloc[t][vel_cols].values.astype(float)
        
        return input_vector, target

    def find_first_valid_index(self, beams_df, imu_df, num_past_beam_instances, num_imu_instances):
        start = max(num_past_beam_instances, num_imu_instances - 1)
        for t in range(start, len(beams_df)):
            try:
                _ = self.construct_beam_input_target(beams_df, imu_df, None, t, num_past_beam_instances, num_imu_instances)
                return t
            except Exception:
                continue
        return None

    def train_beam_model(self, beams_df, imu_df, original_beams, traj_name, epochs, model=None, target_traj_name=None):
        """Train beam model on a single trajectory
        
        Args:
            beams_df: DataFrame containing beam data
            imu_df: DataFrame containing IMU data
            original_beams: DataFrame containing original beam data
            traj_name: Name of the trajectory being trained on
            epochs: Number of epochs to train for
            model: Optional pre-existing model to train
            target_traj_name: Name of the target trajectory (for leave-one-out)
        """
        # Skip if this is the target trajectory
        if traj_name == target_traj_name:
            print(f"[INFO] Skipping training on target trajectory {traj_name}")
            return model, None
            
        # Use provided model or create new one
        if model is None:
            input_size = 4 + (4 * self.config["num_past_beam_instances"]) + 6  # Current IMU only
            model = MemoryNeuralNetwork(
                number_of_input_neurons=input_size,
                number_of_hidden_neurons=self.config["hidden_neurons"],
                number_of_output_neurons=4,
                dropout_rate=self.config["dropout_rate"],
                learning_rate=self.config["learning_rate"],
                learning_rate_2=self.config["learning_rate_2"],
                learning_rate_3=self.config["learning_rate_3"],
                lipschitz_constant=self.config["lipschitz_constant"]
            ).to(self.device)
        
        # Prepare training data
        first_valid = self.find_first_valid_index(beams_df, imu_df, 
                                                self.config["num_past_beam_instances"],
                                                1)  # Only need 1 IMU instance
        if first_valid is None:
            return model, None

        beams_df = beams_df.iloc[first_valid:].reset_index(drop=True)
        imu_df = imu_df.iloc[first_valid:].reset_index(drop=True)
        original_beams = original_beams.iloc[first_valid:].reset_index(drop=True)

        inputs, targets = [], []
        start_t = self.config["num_past_beam_instances"]
        
        for t in range(start_t, len(beams_df)):
            try:
                inp, _ = self.construct_beam_input_target(beams_df, imu_df, original_beams, t,
                                                        self.config["num_past_beam_instances"],
                                                        1)  # Only need 1 IMU instance
                # Use original beams as target
                tar = original_beams.loc[t, ["b1", "b2", "b3", "b4"]].values.astype(float)
                
                # Check for NaN values in input and target
                if np.isnan(inp).any() or np.isnan(tar).any():
                    continue
                    
                inputs.append(inp)
                targets.append(tar)
            except Exception:
                continue

        if len(inputs) == 0:
            return model, None

        # Convert inputs and targets to numpy arrays
        inputs = np.array(inputs)
        targets = np.array(targets)
        
        # Train beam model with RMSE loss
        criterion = RMSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config["learning_rate"])
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            
            # Process sample by sample
            for i in range(len(inputs)):
                input_tensor = torch.tensor(inputs[i], dtype=torch.float32).unsqueeze(0).to(self.device)
                target_tensor = torch.tensor(targets[i], dtype=torch.float32).unsqueeze(0).to(self.device)
                
                optimizer.zero_grad()
                output = model(input_tensor)
                # Ensure output and target_tensor shapes match
                if output.shape != target_tensor.shape:
                    if output.dim() == 1:
                        output = output.unsqueeze(0)
                    if target_tensor.dim() == 1:
                        target_tensor = target_tensor.unsqueeze(0)
                loss = criterion(output, target_tensor)
                loss.backward()
                
                # Add gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(inputs)
            print(f"Beam Model [{traj_name}] Epoch {epoch+1}/{epochs} RMSE: {avg_loss:.6f}")
        
        # Calculate training RMSE on the training data
        model.eval()
        with torch.no_grad():
            train_predictions = []
            train_targets = []
            for i in range(len(inputs)):
                input_tensor = torch.tensor(inputs[i], dtype=torch.float32).unsqueeze(0).to(self.device)
                output = model(input_tensor)
                train_predictions.append(output.cpu().numpy().flatten())
                train_targets.append(targets[i])
            
            train_predictions = np.array(train_predictions)
            train_targets = np.array(train_targets)
            train_rmse = np.sqrt(np.mean((train_predictions - train_targets)**2))
            if target_traj_name is not None:
                self.training_rmse[target_traj_name] = train_rmse
                print(f"[INFO] Training RMSE for {target_traj_name}: {train_rmse:.6f}")
        
        # Generate predictions
        model.eval()
        predictions = np.zeros((len(beams_df), 4))
        
        with torch.no_grad():
            for t in range(start_t, len(beams_df)):
                try:
                    input_vector, _ = self.construct_beam_input_target(
                        beams_df, imu_df, None, t,
                        self.config["num_past_beam_instances"], 1
                    )
                    
                    input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
                    output = model(input_tensor)
                    predictions[t] = output.cpu().numpy().flatten()
                except Exception:
                    continue
        
        # Return only the non-zero predictions
        valid_predictions = predictions[start_t:]
        non_zero_mask = np.any(valid_predictions != 0, axis=1)
        final_predictions = valid_predictions[non_zero_mask]
        
        return model, final_predictions

    def train_velocity_models_leave_one_out(self, training_trajectories):
        """
        For each training trajectory, train a velocity model on (n-1) other trajectories (leave-one-out),
        test on the left-out, and store all models, predictions, and RMSEs.
        """
        print("\n[INFO] PHASE 2: VELOCITY MODEL TRAINING (LEAVE-ONE-OUT)")
        print("===================================================")
        self.velocity_models = {}
        self.velocity_predictions = {}
        self.velocity_rmse = {}
        n = len(training_trajectories)
        for target_idx, target_traj_config in enumerate(training_trajectories):
            target_traj_name = target_traj_config[0]
            target_epochs = target_traj_config[1]
            print(f"\n[INFO] Training velocity model for target trajectory: {target_traj_name}")
            # Collect training data from all other trajectories
            train_inputs, train_targets = [], []
            for other_idx, other_traj_config in enumerate(training_trajectories):
                other_traj_name = other_traj_config[0]
                if other_traj_name == target_traj_name:
                    continue
                # Use hybrid beams, IMU, and velocity data from phase 1
                hybrid_beams = self.hybrid_beams[other_traj_name]
                imu_df = self.imu_data[other_traj_name]
                velocity_df = self.velocity_data[other_traj_name]
                start_t = self.config["num_past_beam_instances"]
                for t in range(start_t, len(hybrid_beams)):
                    try:
                        inp, tar = self.construct_velocity_input_target(
                            hybrid_beams, hybrid_beams, imu_df, velocity_df, t,
                            self.config["num_past_beam_instances"], 1
                        )
                        train_inputs.append(inp)
                        train_targets.append(tar)
                    except Exception:
                        continue
            if len(train_inputs) == 0:
                print(f"[WARNING] No training data for velocity model {target_traj_name}")
                continue
            train_inputs = np.array(train_inputs)
            train_targets = np.array(train_targets)
            # Initialize and train velocity model
            input_size = 4 + (4 * self.config["num_past_beam_instances"]) + 6
            model = MemoryNeuralNetwork(
                number_of_input_neurons=input_size,
                number_of_hidden_neurons=self.config["hidden_neurons"],
                number_of_output_neurons=3,
                dropout_rate=self.config["dropout_rate"],
                learning_rate=self.config["learning_rate"],
                learning_rate_2=self.config["learning_rate_2"],
                lipschitz_constant=self.config["lipschitz_constant"]
            ).to(self.device)
            criterion = RMSELoss()
            optimizer = optim.Adam(model.parameters(), lr=self.config["learning_rate"])
            for epoch in range(target_epochs):
                model.train()
                epoch_loss = 0
                for i in range(len(train_inputs)):
                    input_tensor = torch.tensor(train_inputs[i], dtype=torch.float32).unsqueeze(0).to(self.device)
                    target_tensor = torch.tensor(train_targets[i], dtype=torch.float32).unsqueeze(0).to(self.device)
                    optimizer.zero_grad()
                    output = model(input_tensor)
                    loss = criterion(output, target_tensor)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                avg_loss = epoch_loss / len(train_inputs)
                print(f"Velocity Model [{target_traj_name}] Epoch {epoch+1}/{target_epochs} RMSE: {avg_loss:.6f}")
            # Test on the left-out trajectory
            hybrid_beams = self.hybrid_beams[target_traj_name]
            imu_df = self.imu_data[target_traj_name]
            velocity_df = self.velocity_data[target_traj_name]
            start_t = self.config["num_past_beam_instances"]
            predictions = np.zeros((len(hybrid_beams), 3))
            with torch.no_grad():
                for t in range(start_t, len(hybrid_beams)):
                    try:
                        input_vector, _ = self.construct_velocity_input_target(
                            hybrid_beams, hybrid_beams, imu_df, velocity_df, t,
                            self.config["num_past_beam_instances"], 1
                        )
                        input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
                        output = model(input_tensor)
                        predictions[t] = output.cpu().numpy().flatten()
                    except Exception as e:
                        print(f"[WARNING] Error predicting velocity at index {t}: {e}")
                        continue
            valid_predictions = predictions[start_t:]
            non_zero_mask = np.any(valid_predictions != 0, axis=1)
            velocity_predictions = valid_predictions[non_zero_mask]
            # Pad/truncate to match length
            total_length = len(velocity_df)
            remaining_length = total_length - 2
            if len(velocity_predictions) < remaining_length:
                padding = np.zeros((remaining_length - len(velocity_predictions), 3))
                velocity_predictions = np.vstack([velocity_predictions, padding])
            elif len(velocity_predictions) > remaining_length:
                velocity_predictions = velocity_predictions[:remaining_length]
            full_velocity_predictions = np.vstack([
                velocity_df[['V North', 'V East', 'V Down']].values[:2],
                velocity_predictions
            ])
            # Compute RMSE
            rmse = np.sqrt(np.mean((full_velocity_predictions - velocity_df[['V North', 'V East', 'V Down']].values) ** 2))
            self.velocity_models[target_traj_name] = model
            self.velocity_predictions[target_traj_name] = full_velocity_predictions
            self.velocity_rmse[target_traj_name] = rmse
            print(f"[INFO] Velocity RMSE for {target_traj_name}: {rmse:.6f}")
        # After training all models, select the best one based on lowest RMSE
        best_traj = min(self.velocity_rmse.items(), key=lambda x: x[1])[0]
        self.velocity_model = self.velocity_models[best_traj]
        print(f"\n[INFO] Selected best velocity model from {best_traj} with RMSE: {self.velocity_rmse[best_traj]:.6f}")
        
        print("\n[INFO] Leave-one-out velocity model training completed for all training trajectories.")

    def train(self, training_trajectories):
        """Train the model stack using a strict two-phase approach with leave-one-out.
        
        Phase 1: Train beam model and get predictions for each trajectory using leave-one-out
        Phase 2: Train velocity model using hybrid beams from phase 1
        
        Args:
            training_trajectories: List of [trajectory_name, epochs] pairs
            
        Returns:
            velocity_model: The trained velocity model
        """
        print("[INFO] Starting training process with strict two-phase approach and leave-one-out...")
        
        # PHASE 1: Train beam model and get predictions for each trajectory using leave-one-out
        print("\n[INFO] PHASE 1: BEAM MODEL TRAINING AND PREDICTIONS USING LEAVE-ONE-OUT")
        print("===================================================")
        
        # Store predictions and data for phase 2
        self.beam_predictions = {}
        self.original_beams = {}
        self.imu_data = {}
        self.velocity_data = {}
        self.hybrid_beams = {}
        self.training_models = {}  # Store models for each training trajectory
        self.training_rmse = {}  # Store training RMSE for each model
        self.testing_rmse = {}   # Store testing RMSE for each model
        self.average_rmse = {}   # Store average of training and testing RMSE
        
        # First, get beam predictions for each trajectory using leave-one-out
        for target_traj_config in tqdm(training_trajectories, desc="Leave-One-Out Beam Predictions", leave=True):
            target_traj_name = target_traj_config[0]
            print(f"\n[INFO] Processing target trajectory: {target_traj_name}")
            
            # Create a new beam model for this target trajectory
            input_size = 4 + (4 * self.config["num_past_beam_instances"]) + 6  # Current IMU only
            local_beam_model = MemoryNeuralNetwork(
                number_of_input_neurons=input_size,
                number_of_hidden_neurons=self.config["hidden_neurons"],
                number_of_output_neurons=4,
                dropout_rate=self.config["dropout_rate"],
                learning_rate=self.config["learning_rate"],
                learning_rate_2=self.config["learning_rate_2"],
                learning_rate_3=self.config["learning_rate_3"],
                lipschitz_constant=self.config["lipschitz_constant"]
            ).to(self.device)
            
            # Train on all other trajectories except the target
            print(f"[INFO] Training beam model on all trajectories except {target_traj_name}")
            for traj_config in training_trajectories:
                traj_name = traj_config[0]
                if traj_name == target_traj_name:
                    continue
                    
                print(f"[INFO] Training on trajectory: {traj_name}")
                traj_path = os.path.join("Data_XYZ_change", traj_name)
                beams_df, imu_df, velocity_df, original_beams, missed_beams = self.load_csv_files_enhanced(traj_path)
                
                if beams_df is None or imu_df is None or velocity_df is None:
                    print(f"[ERROR] Failed to load data for trajectory {traj_name}")
                    continue
                
                # Train beam model on this trajectory
                try:
                    _, _ = self.train_beam_model(beams_df, imu_df, original_beams, traj_name, 
                                               traj_config[1], model=local_beam_model, target_traj_name=target_traj_name)
                    print(f"[INFO] Successfully trained beam model on {traj_name}")
                except Exception as e:
                    print(f"[ERROR] Failed training beam model on {traj_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Now predict for the target trajectory
            print(f"[INFO] Predicting beams for target trajectory: {target_traj_name}")
            traj_path = os.path.join("Data_XYZ_change", target_traj_name)
            beams_df, imu_df, velocity_df, original_beams, missed_beams = self.load_csv_files_enhanced(traj_path)
            
            if beams_df is None or imu_df is None or velocity_df is None:
                print(f"[ERROR] Failed to load data for target trajectory {target_traj_name}")
                continue
            
            # Store data for phase 2
            self.original_beams[target_traj_name] = original_beams
            self.imu_data[target_traj_name] = imu_df
            self.velocity_data[target_traj_name] = velocity_df
            
            try:
                # Set model to evaluation mode
                local_beam_model.eval()
                
                # Reset model state
                if hasattr(local_beam_model, 'prev_output_of_nn'):
                    batch_size = 1
                    output_size = 4
                    local_beam_model.prev_output_of_nn = torch.zeros(batch_size, output_size, device=self.device)
                
                # Predict beams sample by sample
                predictions = np.zeros((len(beams_df), 4))
                start_idx = self.config["num_past_beam_instances"]
                
                with torch.no_grad():
                    for t in range(start_idx, len(beams_df)):
                        try:
                            input_vector, _ = self.construct_beam_input_target(
                                beams_df, imu_df, None, t,
                                self.config["num_past_beam_instances"], 1
                            )
                            
                            input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
                            output = local_beam_model(input_tensor)
                            predictions[t] = output.cpu().numpy().flatten()
                        except Exception as e:
                            print(f"[WARNING] Error predicting beam at index {t}: {e}")
                            continue
                
                # Return only the non-zero predictions
                valid_predictions = predictions[start_idx:]
                non_zero_mask = np.any(valid_predictions != 0, axis=1)
                final_predictions = valid_predictions[non_zero_mask]
                
                # Calculate RMSE for this model
                rmse = np.sqrt(np.mean((final_predictions - original_beams.iloc[start_idx:][['b1', 'b2', 'b3', 'b4']].values[non_zero_mask])**2))
                self.testing_rmse[target_traj_name] = rmse
                print(f"[INFO] Testing RMSE for {target_traj_name}: {rmse:.6f}")
                
                # Calculate average RMSE (training + testing) / 2
                if target_traj_name in self.training_rmse:
                    avg_rmse = (self.training_rmse[target_traj_name] + rmse) / 2
                    self.average_rmse[target_traj_name] = avg_rmse
                    print(f"[INFO] Average RMSE for {target_traj_name}: {avg_rmse:.6f}")
                
                self.beam_predictions[target_traj_name] = final_predictions
                self.training_models[target_traj_name] = local_beam_model
                print(f"[INFO] Successfully generated beam predictions for {target_traj_name} - shape: {final_predictions.shape}")
                
                # Create hybrid beams (original where available, predicted where missing)
                hybrid_beams = original_beams.copy()
                beam_offset = len(original_beams) - len(final_predictions)
                
                for i in range(len(final_predictions)):
                    orig_idx = i + beam_offset
                    if orig_idx < len(original_beams):
                        for j, beam in enumerate(['b1', 'b2', 'b3', 'b4']):
                            if pd.isna(missed_beams.loc[orig_idx, beam]):
                                hybrid_beams.loc[orig_idx, beam] = final_predictions[i, j]
                
                self.hybrid_beams[target_traj_name] = hybrid_beams
                
            except Exception as e:
                print(f"[ERROR] Failed generating beam predictions for {target_traj_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Select best model based on average RMSE
        best_traj = min(self.average_rmse.items(), key=lambda x: x[1])[0]
        self.beam_model = self.training_models[best_traj]
        print(f"\n[INFO] Selected best model from {best_traj} with average RMSE: {self.average_rmse[best_traj]:.6f}")
        print(f"[INFO] Training RMSE: {self.training_rmse[best_traj]:.6f}")
        print(f"[INFO] Testing RMSE: {self.testing_rmse[best_traj]:.6f}")
        
        print("\n[INFO] Beam model training and predictions completed using leave-one-out")
        print("===================================================")
        
        # PHASE 2: Train velocity model using hybrid beams from phase 1
        self.train_velocity_models_leave_one_out(training_trajectories)
        
        print("\n[INFO] Two-phase training completed successfully")
        
        # After training is complete, save results for each training trajectory
        results_dir = "training_results"
        os.makedirs(results_dir, exist_ok=True)
        
        for traj_config in training_trajectories:
            traj_name = traj_config[0]
            self.save_trajectory_results(traj_name, results_dir, is_training=True)
        
        return self.velocity_model

    def predict(self, traj_name):
        """Standard predict method for backward compatibility"""
        beam_predictions, velocity_predictions, _ = self.predict_enhanced(traj_name)
        return beam_predictions, velocity_predictions
        
    def predict_enhanced(self, traj_name):
        """Enhanced prediction method that handles both training and testing trajectories."""
        print(f"[INFO] Starting enhanced prediction for trajectory: {traj_name}")
        
        # Load data
        traj_path = os.path.join("Data_XYZ_change", traj_name)
        beams_df, imu_df, velocity_df, original_beams, missed_beams = self.load_csv_files_enhanced(traj_path)
        
        if beams_df is None or imu_df is None or velocity_df is None:
            print(f"[ERROR] Failed to load data for trajectory {traj_name}")
            return None, None, None
        
        # Check if this is a training trajectory
        is_training = any(traj_config[0] == traj_name for traj_config in self.config["training_trajectories"])
        
        if is_training:
            print(f"[INFO] {traj_name} is a training trajectory, using leave-one-out approach")
            # Get predictions using leave-one-out
            beam_predictions = self.beam_predictions.get(traj_name)
            if beam_predictions is None:
                print(f"[ERROR] No beam predictions available for training trajectory {traj_name}")
                return None, None, None
            # Use the correct leave-one-out velocity model
            velocity_model = self.velocity_models.get(traj_name)
            if velocity_model is None:
                print(f"[ERROR] No velocity model available for training trajectory {traj_name}")
                return beam_predictions, None, None
        else:
            print(f"[INFO] {traj_name} is a testing trajectory, using best beam model")
            # Use the best beam model for predictions
            if self.beam_model is None:
                print(f"[ERROR] No beam model available for testing trajectory {traj_name}")
                return None, None, None
            # Set model to evaluation mode
            self.beam_model.eval()
            if hasattr(self.beam_model, 'prev_output_of_nn'):
                batch_size = 1
                output_size = 4
                self.beam_model.prev_output_of_nn = torch.zeros(batch_size, output_size, device=self.device)
            predictions = np.zeros((len(beams_df), 4))
            start_idx = self.config["num_past_beam_instances"]
            with torch.no_grad():
                for t in range(start_idx, len(beams_df)):
                    try:
                        input_vector, _ = self.construct_beam_input_target(
                            beams_df, imu_df, None, t,
                            self.config["num_past_beam_instances"], 1
                        )
                        input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
                        output = self.beam_model(input_tensor)
                        predictions[t] = output.cpu().numpy().flatten()
                    except Exception as e:
                        print(f"[WARNING] Error predicting beam at index {t}: {e}")
                        continue
            valid_predictions = predictions[start_idx:]
            non_zero_mask = np.any(valid_predictions != 0, axis=1)
            beam_predictions = valid_predictions[non_zero_mask]
            if len(beam_predictions) == 0:
                print(f"[ERROR] No valid beam predictions generated for {traj_name}")
                return None, None, None
            # Use the best velocity model for test
            velocity_model = self.velocity_model
            if velocity_model is None:
                print(f"[ERROR] No velocity model available for {traj_name}")
                return beam_predictions, None, None
        # Create full beam predictions array with first 2 instances from GT
        total_length = len(beams_df)
        remaining_length = total_length - 2
        if len(beam_predictions) < remaining_length:
            padding = np.zeros((remaining_length - len(beam_predictions), 4))
            beam_predictions = np.vstack([beam_predictions, padding])
        elif len(beam_predictions) > remaining_length:
            beam_predictions = beam_predictions[:remaining_length]
        full_beam_predictions = np.vstack([
            original_beams[['b1', 'b2', 'b3', 'b4']].values[:2],
            beam_predictions
        ])
        print(f"[INFO] Creating hybrid beams for {traj_name}")
        hybrid_beams = original_beams.copy()
        for i in range(len(hybrid_beams)):
            for j, beam in enumerate(['b1', 'b2', 'b3', 'b4']):
                if pd.isna(missed_beams.loc[i, beam]):
                    hybrid_beams.loc[i, beam] = full_beam_predictions[i, j]
        print(f"[INFO] Predicting velocities for {traj_name}")
        velocity_model.eval()
        if hasattr(velocity_model, 'prev_output_of_nn'):
            batch_size = 1
            output_size = 3
            velocity_model.prev_output_of_nn = torch.zeros(batch_size, output_size, device=self.device)
        predictions = np.zeros((len(beams_df), 3))
        start_idx = self.config["num_past_beam_instances"]
        with torch.no_grad():
            for t in range(start_idx, len(beams_df)):
                try:
                    input_vector, _ = self.construct_velocity_input_target(
                        beams_df, hybrid_beams, imu_df, velocity_df, t,
                        self.config["num_past_beam_instances"], 1
                    )
                    input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
                    output = velocity_model(input_tensor)
                    predictions[t] = output.cpu().numpy().flatten()
                except Exception as e:
                    print(f"[WARNING] Error predicting velocity at index {t}: {e}")
                    continue
        valid_predictions = predictions[start_idx:]
        non_zero_mask = np.any(valid_predictions != 0, axis=1)
        velocity_predictions = valid_predictions[non_zero_mask]
        total_length = len(velocity_df)
        remaining_length = total_length - 2
        if len(velocity_predictions) < remaining_length:
            padding = np.zeros((remaining_length - len(velocity_predictions), 3))
            velocity_predictions = np.vstack([velocity_predictions, padding])
        elif len(velocity_predictions) > remaining_length:
            velocity_predictions = velocity_predictions[:remaining_length]
        full_velocity_predictions = np.vstack([
            velocity_df[['V North', 'V East', 'V Down']].values[:2],
            velocity_predictions
        ])
        print(f"[INFO] Completed predictions for {traj_name}")
        print(f"Beam predictions shape: {full_beam_predictions.shape}")
        print(f"Velocity predictions shape: {full_velocity_predictions.shape}")
        results_dir = "testing_results" if not is_training else "training_results"
        os.makedirs(results_dir, exist_ok=True)
        self.save_trajectory_results(traj_name, results_dir, 
                                   is_training=is_training,
                                   beam_predictions=full_beam_predictions,
                                   velocity_predictions=full_velocity_predictions,
                                   hybrid_beams=hybrid_beams)
        return full_beam_predictions, full_velocity_predictions, hybrid_beams

    def save_models(self, path):
        """Save both models"""
        os.makedirs(path, exist_ok=True)
        torch.save(self.beam_model.state_dict(), os.path.join(path, "beam_model.pth"))
        torch.save(self.velocity_model.state_dict(), os.path.join(path, "velocity_model.pth"))

    def load_models(self, path):
        """Load both models"""
        # Instantiate models before loading weights
        # Only use current IMU (6 values), consistent with training
        input_size_beam = 4 + (4 * self.config["num_past_beam_instances"]) + 6  # Current IMU only
        input_size_vel = 4 + (4 * self.config["num_past_beam_instances"]) + 6  # Current IMU only
        self.beam_model = MemoryNeuralNetwork(
            number_of_input_neurons=input_size_beam,
            number_of_hidden_neurons=self.config["hidden_neurons"],
            number_of_output_neurons=4,
            dropout_rate=self.config["dropout_rate"],
            learning_rate=self.config["learning_rate"],
            learning_rate_2=self.config["learning_rate_2"],
            lipschitz_constant=self.config["lipschitz_constant"]
        ).to(self.device)
        self.velocity_model = MemoryNeuralNetwork(
            number_of_input_neurons=input_size_vel,
            number_of_hidden_neurons=self.config["hidden_neurons"],
            number_of_output_neurons=3,
            dropout_rate=self.config["dropout_rate"],
            learning_rate=self.config["learning_rate"],
            learning_rate_2=self.config["learning_rate_2"],
            lipschitz_constant=self.config["lipschitz_constant"]
        ).to(self.device)
        self.beam_model.load_state_dict(torch.load(os.path.join(path, "beam_model.pth")))
        self.velocity_model.load_state_dict(torch.load(os.path.join(path, "velocity_model.pth")))

    def predict_beams_leave_one_out(self, target_trajectory):
        """Predict beams for a target trajectory using a model trained on all other trajectories.
        
        Args:
            target_trajectory: Name of the trajectory to predict beams for
            
        Returns:
            predictions: numpy array of beam predictions
        """
        print(f"[INFO] Predicting beams for {target_trajectory} using leave-one-out approach")
        
        # Create a new beam model for this target trajectory
        input_size = 4 + (4 * self.config["num_past_beam_instances"]) + 6  # Current IMU only
        local_beam_model = MemoryNeuralNetwork(
            number_of_input_neurons=input_size,
            number_of_hidden_neurons=self.config["hidden_neurons"],
            number_of_output_neurons=4,
            dropout_rate=self.config["dropout_rate"],
            learning_rate=self.config["learning_rate"],
            learning_rate_2=self.config["learning_rate_2"],
            lipschitz_constant=self.config["lipschitz_constant"]
        ).to(self.device)
        print(f"[INFO] Created new beam model for {target_trajectory}")
        
        # Train on all other trajectories
        for traj_name in self.config["training_trajectories"]:
            if traj_name == target_trajectory:
                continue
                
            print(f"[INFO] Training on trajectory: {traj_name}")
            traj_path = os.path.join("Data_XYZ_change", traj_name)
            beams_df, imu_df, velocity_df, original_beams = self.load_csv_files(traj_path)
            
            if beams_df is None or imu_df is None or velocity_df is None:
                print(f"[ERROR] Failed to load data for trajectory {traj_name}")
                continue
            
            # Train beam model
            try:
                _, _ = self.train_beam_model(beams_df, imu_df, original_beams, traj_name, self.config["epochs"], model=local_beam_model, target_traj_name=target_trajectory)
                print(f"[INFO] Successfully trained beam model on {traj_name}")
            except Exception as e:
                print(f"[ERROR] Failed training beam model on {traj_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Now predict for the target trajectory
        print(f"[INFO] Predicting beams for target trajectory: {target_trajectory}")
        traj_path = os.path.join("Data_XYZ_change", target_trajectory)
        beams_df, imu_df, velocity_df, original_beams = self.load_csv_files(traj_path)
        
        if beams_df is None or imu_df is None or velocity_df is None:
            print(f"[ERROR] Failed to load data for target trajectory {target_trajectory}")
            return None
        
        try:
            # Set model to evaluation mode
            local_beam_model.eval()
            
            # Reset model state
            if hasattr(local_beam_model, 'prev_output_of_nn'):
                batch_size = 1
                output_size = 4
                local_beam_model.prev_output_of_nn = torch.zeros(batch_size, output_size, device=self.device)
            
            # Predict beams sample by sample
            predictions = np.zeros((len(beams_df), 4))
            start_idx = self.config["num_past_beam_instances"]
            
            with torch.no_grad():
                for t in range(start_idx, len(beams_df)):
                    try:
                        input_vector, _ = self.construct_beam_input_target(
                            beams_df, imu_df, None, t,
                            self.config["num_past_beam_instances"], 1
                        )
                        
                        input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
                        output = local_beam_model(input_tensor)
                        predictions[t] = output.cpu().numpy().flatten()
                    except Exception as e:
                        print(f"[WARNING] Error predicting beam at index {t}: {e}")
                        continue
            
            # Return only the non-zero predictions
            valid_predictions = predictions[start_idx:]
            non_zero_mask = np.any(valid_predictions != 0, axis=1)
            final_predictions = valid_predictions[non_zero_mask]
            
            print(f"[INFO] Successfully generated beam predictions for {target_trajectory} - shape: {final_predictions.shape}")
            return final_predictions
            
        except Exception as e:
            print(f"[ERROR] Failed generating beam predictions for {target_trajectory}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_all_training_predictions(self):
        """
        Get beam predictions for all training trajectories using leave-one-out approach.
        Returns a dictionary mapping trajectory names to their predictions.
        """
        print("[INFO] Starting leave-one-out predictions for all training trajectories")
        predictions = {}
        
        for traj_config in self.config["training_trajectories"]:
            traj_name = traj_config[0]
            print(f"\n[INFO] Processing trajectory: {traj_name}")
            
            # Get predictions using leave-one-out approach
            _, traj_predictions = self.predict_beams_leave_one_out(traj_name)
            
            if traj_predictions is not None:
                predictions[traj_name] = traj_predictions
                print(f"[INFO] Successfully obtained predictions for {traj_name}")
            else:
                print(f"[ERROR] Failed to get predictions for {traj_name}")
        
        return predictions

    def save_trajectory_results(self, traj_name, results_dir, is_training=True, beam_predictions=None, velocity_predictions=None, hybrid_beams=None):
        """Save all results for a trajectory with organized directory structure
        
        Args:
            traj_name: Name of the trajectory
            results_dir: Directory to save results
            is_training: Whether this is a training trajectory
            beam_predictions: Optional pre-computed beam predictions
            velocity_predictions: Optional pre-computed velocity predictions
            hybrid_beams: Optional pre-computed hybrid beams
        """
        # Create main directory for this trajectory
        traj_dir = os.path.join(results_dir, traj_name)
        os.makedirs(traj_dir, exist_ok=True)
        
        # Create subdirectories
        beams_dir = os.path.join(traj_dir, "Beams")
        velocities_dir = os.path.join(traj_dir, "Velocities")
        plots_dir = os.path.join(traj_dir, "Plots")
        
        os.makedirs(beams_dir, exist_ok=True)
        os.makedirs(velocities_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        
        # Get data
        traj_path = os.path.join("Data_XYZ_change", traj_name)
        beams_df, imu_df, velocity_df, original_beams, missed_beams = self.load_csv_files_enhanced(traj_path)
        
        if beams_df is None or imu_df is None or velocity_df is None:
            print(f"[ERROR] Failed to load data for trajectory {traj_name}")
            return
        
        # If predictions not provided, compute them
        if beam_predictions is None or velocity_predictions is None or hybrid_beams is None:
            beam_predictions, velocity_predictions, hybrid_beams = self.predict_enhanced(traj_name)
            
            if beam_predictions is None or velocity_predictions is None or hybrid_beams is None:
                print(f"[ERROR] Failed to get predictions for trajectory {traj_name}")
                return
        
        # Save beam data
        original_beams.to_csv(os.path.join(beams_dir, "GT.csv"), index=False)
        missed_beams.to_csv(os.path.join(beams_dir, "Missed.csv"), index=False)
        beams_df.to_csv(os.path.join(beams_dir, "Filled.csv"), index=False)
        np.savetxt(os.path.join(beams_dir, "Predicted.csv"), beam_predictions, delimiter=",")
        hybrid_beams.to_csv(os.path.join(beams_dir, "Hybrid.csv"), index=False)
        
        # Save velocity data
        velocity_df.to_csv(os.path.join(velocities_dir, "GT.csv"), index=False)
        np.savetxt(os.path.join(velocities_dir, "Predicted.csv"), velocity_predictions, delimiter=",")
        
        # Create and save plots
        # 1. Predicted vs GT (Beams)
        plt.figure(figsize=(15, 10))
        for i, beam in enumerate(['b1', 'b2', 'b3', 'b4']):
            plt.subplot(2, 2, i+1)
            plt.plot(original_beams[beam].values, label='GT', alpha=0.7)
            plt.plot(beam_predictions[:, i], label='Predicted', alpha=0.7)
            plt.title(f'{beam} - GT vs Predicted')
            plt.legend()
            plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'Beams_GT_vs_Predicted.png'))
        plt.close()
        
        # 2. Individual BEAM RMSE over time
        plt.figure(figsize=(15, 10))
        for i, beam in enumerate(['b1', 'b2', 'b3', 'b4']):
            plt.subplot(2, 2, i+1)
            rmse = np.sqrt(np.mean((beam_predictions[:, i] - original_beams[beam].values)**2))
            plt.plot(np.abs(beam_predictions[:, i] - original_beams[beam].values), label=f'RMSE: {rmse:.4f}')
            plt.title(f'{beam} - Absolute Error over time')
            plt.legend()
            plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'Beams_RMSE_over_time.png'))
        plt.close()
        
        # 3. Predicted vs GT (Velocities)
        plt.figure(figsize=(15, 10))
        for i, vel in enumerate(['V North', 'V East', 'V Down']):
            plt.subplot(3, 1, i+1)
            plt.plot(velocity_df[vel].values, label='GT', alpha=0.7)
            plt.plot(velocity_predictions[:, i], label='Predicted', alpha=0.7)
            plt.title(f'{vel} - GT vs Predicted')
            plt.legend()
            plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'Velocities_GT_vs_Predicted.png'))
        plt.close()
        
        # 4. Individual Component RMSE over time
        plt.figure(figsize=(15, 10))
        for i, vel in enumerate(['V North', 'V East', 'V Down']):
            plt.subplot(3, 1, i+1)
            rmse = np.sqrt(np.mean((velocity_predictions[:, i] - velocity_df[vel].values)**2))
            plt.plot(np.abs(velocity_predictions[:, i] - velocity_df[vel].values), label=f'RMSE: {rmse:.4f}')
            plt.title(f'{vel} - Absolute Error over time')
            plt.legend()
            plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'Velocities_RMSE_over_time.png'))
        plt.close()
        
        # Save summary
        summary = {
            'trajectory': traj_name,
            'is_training': is_training,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'beam_rmse': {
                beam: float(np.sqrt(np.mean((beam_predictions[:, i] - original_beams[beam].values)**2)))
                for i, beam in enumerate(['b1', 'b2', 'b3', 'b4'])
            },
            'velocity_rmse': {
                vel: float(np.sqrt(np.mean((velocity_predictions[:, i] - velocity_df[vel].values)**2)))
                for i, vel in enumerate(['V North', 'V East', 'V Down'])
            }
        }
        
        with open(os.path.join(traj_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"[INFO] Saved all results for {traj_name} in {traj_dir}")
        print(f"[INFO] Original beams shape: {original_beams.shape}, Predicted beams shape: {beam_predictions.shape}")
        print(f"[INFO] Original velocity shape: {velocity_df[['V North', 'V East', 'V Down']].shape}, Predicted velocity shape: {velocity_predictions.shape}")
        print(f"[DEBUG] Beam start: {0}, Velocity start: {0}")
        print(f"[DEBUG] Original beams: {len(original_beams)}, Predicted beams: {len(beam_predictions)}")
        print(f"[DEBUG] Original velocities: {len(velocity_df)}, Predicted velocities: {len(velocity_predictions)}")
        print(f"[INFO] All data files and plots saved for {traj_name}")