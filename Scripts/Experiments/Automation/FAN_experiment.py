import torch
import os
import json
import pandas as pd
import time
import re
import numpy as np
import matplotlib.pyplot as plt
import glob
import random
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Logging setup
LOG_FILE = "../../Experiments/experiment_log_FAN.txt"
def log_message(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"[{timestamp}] {message}\n")
    print(message)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Architecture_Codes')))
from FAN import FAN

CONFIG_PATH = "FAN.json"
RESULTS_PATH = "../../Experiments/Results_New/FAN/Results_FAN.csv"
CHECKPOINT_DIR = "../../Experiments/Results_New/FAN/Checkpoints_FAN/"
PLOTS_DIR = "../../Experiments/Results_New/FAN/Plots/"
DATA_DIR = "../../../Data"

# Load configuration
log_message("[INFO] Loading configuration...")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
log_message("[INFO] Configuration loaded successfully.")

# Load hyperparameters
learning_rate = config["learning_rate"]
dropout_rate = config["dropout_rate"]
hidden_neurons = config["hidden_neurons"]
stacking_count = config["stacking_count"]
epochs = config["epochs"]
regularization = config["regularization"]
rmse_threshold = config.get("Threshold")
number_of_trials = config["Trials"]

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def check_duplicate(trajectory_id, a, b):
    if os.path.exists(RESULTS_PATH):
        df = pd.read_csv(RESULTS_PATH)
        match = df[
            (df["Learning Rate"] == learning_rate) & (df["Dropout Rate"] == dropout_rate) &
            (df["Trajectory ID"] == trajectory_id) & (df["Number of Hidden Neurons"] == hidden_neurons) &
            (df["Epochs"] == epochs) & (df["Stacking Count"] == stacking_count) &
            (df["6()"] == a) & (df["4()"] == b)
        ]
        if not match.empty:
            log_message("[WARNING] Duplicate experiment found. Skipping training.")
            return True
    return False

def train_model(trajectory_id, data_file, a, b):
    global learning_rate, dropout_rate, hidden_neurons, stacking_count, epochs
    if check_duplicate(trajectory_id, a, b):
        return

    log_message(f"[INFO] Loading data for Trajectory: {trajectory_id}, a={a}, b={b}")
    data = pd.read_csv(data_file)
    num_inputs = (6 * a) + (4 * b)
    input_data = data.iloc[:-1, :num_inputs].values
    target_data = data.iloc[1:, -3:].values

    attempt, best_rmse = 0, float("inf")
    start_time = time.time()

    while attempt < number_of_trials:
        fan = FAN(
            number_of_input_neurons=num_inputs,
            number_of_output_neurons=3,
            learning_rate=learning_rate,
            lambda_reg=regularization
        ).to("cuda")

        optimizer = torch.optim.AdamW(fan.parameters(), lr=learning_rate, weight_decay=regularization)
        loss_fn = torch.nn.MSELoss()

        checkpoint_path = os.path.join(
    CHECKPOINT_DIR, 
    f"FAN_lr{learning_rate}_drop{dropout_rate}_{trajectory_id}_hid{hidden_neurons}_ep{epochs}_stack{stacking_count}_a{a}_b{b}.pth"
)
        fan.train()
        rmse_per_epoch = []

        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(len(input_data)):
                x = torch.tensor(input_data[i], dtype=torch.float32, device="cuda")
                y = torch.tensor(target_data[i], dtype=torch.float32, device="cuda")
                optimizer.zero_grad()
                output = fan.feedforward(x)
                loss = loss_fn(output, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fan.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_rmse = (epoch_loss / len(input_data)) ** 0.5
            rmse_per_epoch.append(avg_rmse)
            best_rmse = min(best_rmse, avg_rmse)
            log_message(f"[INFO] Epoch {epoch + 1}: RMSE = {avg_rmse:.5f}")
            torch.save(fan.state_dict(), checkpoint_path)

            if best_rmse <= rmse_threshold:
                log_message("[SUCCESS] RMSE within threshold! Stopping training.")
                break

        total_time = time.time() - start_time
        plot_path = os.path.join(PLOTS_DIR, f"FAN_{trajectory_id}_a{a}_b{b}.png")
        plt.figure()
        plt.plot(range(1, len(rmse_per_epoch) + 1), rmse_per_epoch, marker='o', linestyle='-', color='b')
        plt.xlabel("Epochs")
        plt.ylabel("RMSE")
        plt.title(f"FAN RMSE - Trajectory {trajectory_id}, a={a}, b={b}, Attempt {attempt}")
        plt.grid()
        plt.savefig(plot_path)
        plt.close()
        log_message(f"[INFO] Saved RMSE plot to {plot_path}")

if __name__ == "__main__":
    for traj_folder in os.listdir(DATA_DIR):
        traj_path = os.path.join(DATA_DIR, traj_folder)
        if not os.path.isdir(traj_path):
            continue
        for file in os.listdir(traj_path):
            file_path = os.path.join(traj_path, file)
            trajectory_id, a, b = re.findall(r'\d+', file_path)[-3:]
            train_model(int(trajectory_id), file_path, int(a), int(b))
