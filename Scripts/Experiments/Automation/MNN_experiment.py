import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import pandas as pd
import time
import re
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Architecture_Codes')))
from MNN import MemoryNeuralNetwork


CONFIG_PATH = "MNN.json"
RESULTS_PATH = "../../Experiments/Results_New/MNN/Results_MNN.csv"
CHECKPOINT_DIR = "../../Experiments/Results_New/MNN/Checkpoints_MNN/"
LOG_FILE = "experiment_log_MNN.txt"
DATA_DIR = "../../../Data"

print("Loading configuration...")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
print("Configuration loaded successfully.")


learning_rate = config["learning_rate"]
dropout_rate = config["dropout_rate"]
hidden_neurons = config["hidden_neurons"]
stacking_count = config["stacking_count"]
epochs = config["epochs"]
lipschitz_constant = config["lipschitz_constant"]
rmse_threshold = config["Threshold"]

print(f"Hyperparameters: LR={learning_rate}, Dropout={dropout_rate}, Hidden Neurons={hidden_neurons}, Stacking={stacking_count}, Epochs={epochs}, Lipschitz={lipschitz_constant}, RMSE Threshold={rmse_threshold}")


os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def log_message(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(LOG_FILE, "a") as log:
        log.write(f"[{timestamp}] {message}\n")
    print(message)


def parse_filename(filename):
    match = re.search(r'combined_(\d+)_(\d+)\.csv', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def check_duplicate(trajectory_id, a, b):
    print(f"Checking for duplicates for trajectory: {trajectory_id}, a={a}, b={b}")
    if os.path.exists(RESULTS_PATH):
        df = pd.read_csv(RESULTS_PATH)
        match = df[
            (df["Learning Rate"] == learning_rate) &
            (df["Dropout Rate"] == dropout_rate) &
            (df["Trajectory Id"] == trajectory_id) &
            (df["Number of Hidden Neurons"] == hidden_neurons) &
            (df["Epochs"] == epochs) &
            (df["Stacking Count"] == stacking_count) &
            (df["6()"] == a) &
            (df["4()"] == b)
        ]
        if not match.empty:
            print("Duplicate experiment found. Skipping training.")
            return True
    print("No duplicate found. Proceeding with training.")
    return False


def train_model(trajectory_id, data_file, a, b):
    global learning_rate, hidden_neurons, stacking_count, epochs, dropout_rate

    print(f"Loading data for trajectory: {trajectory_id}, a={a}, b={b}")
    try:
        data = pd.read_csv(data_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print(f"Data shape: {data.shape}")
    data.drop(data.index[-1], inplace=True)  
    num_inputs = (6 * a) + (4 * b)
    input_data = data.iloc[:, :num_inputs].values
    target_data = data.iloc[:, -3:].values

    attempt = 0
    while attempt < 2:
        best_rmse = float('inf')
        print("Initializing model...")
        model = MemoryNeuralNetwork(
            number_of_input_neurons=num_inputs,
            number_of_hidden_neurons=hidden_neurons,
            number_of_output_neurons=3,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            lipschitz_constant=lipschitz_constant
        ).to("cuda")
        print("Model initialized successfully.")

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        
        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            f"MNN_lr{learning_rate}_drop{dropout_rate}_{trajectory_id}_hid{hidden_neurons}_ep{epochs}_stack{stacking_count}_a{a}_b{b}.pth"
        )
        
        torch.autograd.set_detect_anomaly(True)
        model.train()
        start_time = time.time()
        print(f"Starting training attempt {attempt + 1} for trajectory: {trajectory_id}, a={a}, b={b}")
        
        rmse_per_epoch = []
        for epoch in range(epochs):
            epoch_loss = 0
            print(f"Epoch {epoch + 1}/{epochs} started...")
            for i in tqdm(range(len(input_data)), desc=f"Epoch {epoch + 1}/{epochs}"):
                x = torch.tensor(input_data[i], dtype=torch.float32, device="cuda")
                y = torch.tensor(target_data[i], dtype=torch.float32, device="cuda")
                optimizer.zero_grad()
                output = model(x)
                loss = loss_fn(output, y)
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            avg_rmse = (epoch_loss / len(input_data)) ** 0.5
            rmse_per_epoch.append(avg_rmse)
            best_rmse = min(best_rmse, avg_rmse)
            log_message(f"Epoch {epoch + 1}: RMSE = {avg_rmse:.5f}")
            torch.save(model.state_dict(), checkpoint_path)
        
        
        results_df = pd.DataFrame({
            "Learning Rate": [learning_rate],
            "Dropout Rate": [dropout_rate],
            "Trajectory Id": [trajectory_id],
            "Number of Hidden Neurons": [hidden_neurons],
            "Epochs": [epochs],
            "Stacking Count": [stacking_count],
            "6()": [a],
            "4()": [b],
            "Best RMSE": [best_rmse]
        })
        if os.path.exists(RESULTS_PATH):
            results_df.to_csv(RESULTS_PATH, mode='a', header=False, index=False)
        else:
            results_df.to_csv(RESULTS_PATH, mode='w', header=True, index=False)
        
        
        plot_dir = "../../Experiments/Results_New/MNN/Plots/"
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"MNN_lr{learning_rate}_drop{dropout_rate}_{trajectory_id}_hid{hidden_neurons}_ep{epochs}_stack{stacking_count}_a{a}_b{b}.png")
        plt.figure()
        plt.plot(range(1, epochs + 1), rmse_per_epoch, marker='o', linestyle='-', color='b')
        plt.xlabel("Epochs")
        plt.ylabel("RMSE")
        plt.title(f"Best RMSE vs. Epochs\nTrajectory: {trajectory_id}, a={a}, b={b}")
        plt.grid()
        plt.savefig(plot_path)
        plt.close()
        log_message(f"Saved RMSE plot to {plot_path}")
        
        
        if best_rmse <= rmse_threshold:
            break
        else:
            log_message(f"Best RMSE ({best_rmse:.5f}) exceeded threshold ({rmse_threshold}). Retrying with updated hyperparameters...")
            learning_rate *= 1.25
            dropout_rate *= 1.15
            hidden_neurons = min(hidden_neurons + 20, 150)
            stacking_count += 1
            attempt += 1



print("Starting training process...")
for trajectory_folder in sorted(os.listdir(DATA_DIR)):
    trajectory_path = os.path.join(DATA_DIR, trajectory_folder)
    if os.path.isdir(trajectory_path):
        for file in sorted(os.listdir(trajectory_path)):
            if file.startswith("combined_") and file.endswith(".csv"):
                a, b = parse_filename(file)
                if a is None or b is None:
                    continue
                trajectory_id = trajectory_folder
                data_file = os.path.join(trajectory_path, file)
                if not check_duplicate(trajectory_id, a, b):
                    log_message(f"Starting training for Trajectory: {trajectory_id}, a={a}, b={b}")
                    train_model(trajectory_id, data_file, a, b)
                else:
                    log_message(f"Skipping duplicate experiment for Trajectory: {trajectory_id}, a={a}, b={b}")
