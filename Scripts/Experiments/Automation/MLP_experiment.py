import torch
import os
import json
import pandas as pd
import time
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Architecture_Codes'))) 
from MLP import MLPNetwork

# Load configuration
CONFIG_PATH = "MLP.json"
RESULTS_PATH = "../../Experiments/Results_New/MLP/Results_MLP.csv"
CHECKPOINT_DIR = "../../Experiments/Results_New/MLP/Checkpoints_MLP/"
PLOTS_DIR = "../../Experiments/Results_New/MLP/Plots/"
LOG_FILE = "experiment_log_MLP.txt"
DATA_DIR = "../../../Data"

print("[INFO] Loading configuration... (Checking JSON file existence)")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
print("[INFO] Configuration loaded successfully. Validating hyperparameters...")

# Load hyperparameters from config
learning_rate = config["learning_rate"]
dropout_rate = config["dropout_rate"]
hidden_neurons = config["hidden_neurons"]
stacking_count = config["stacking_count"]
epochs = config["epochs"]
regularization = config["regularization"]
rmse_threshold = config.get("Threshold")
number_of_trials = config["Trials"]
number_of_layers = config["number_of_layers"]

print(f"[INFO] Ensuring checkpoint directory exists: {CHECKPOINT_DIR}")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f"[INFO] Ensuring plots directory exists: {PLOTS_DIR}")
os.makedirs(PLOTS_DIR, exist_ok=True)

def log_message(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(LOG_FILE, "a", encoding="utf-8") as log:  # Use UTF-8 encoding
        log.write(f"[{timestamp}] {message}\n")
    print(message)

def parse_filename(file_path):
    match_folder = re.search(r'Trajectory(\d+)', file_path)
    if not match_folder:
        return None, None, None

    trajectory_id = int(match_folder.group(1))

    match_file = re.search(r'combined_(\d+)_(\d+)\.csv', os.path.basename(file_path))
    if not match_file:
        return None, None, None

    a, b = int(match_file.group(1)), int(match_file.group(2))

    return trajectory_id, a, b

def check_duplicate(trajectory_id, a, b):
    print("[INFO] Checking for existing results file...")
    if os.path.exists(RESULTS_PATH):
        df = pd.read_csv(RESULTS_PATH)
        match = df[
            (df["Learning Rate"] == learning_rate) & (df["Dropout Rate"] == dropout_rate) &
            (df["Trajectory Id"] == trajectory_id) & (df["Number of Hidden Neurons"] == hidden_neurons) &
            (df["Epochs"] == epochs) & (df["Stacking Count"] == stacking_count) &
            (df["6()"] == a) & (df["4()"] == b)
        ]
        if not match.empty:
            log_message("[WARNING] Duplicate experiment found. Skipping training.")
            return True
    return False

def train_model(trajectory_id, data_file, a, b):
    global learning_rate, dropout_rate, hidden_neurons, stacking_count, epochs, number_of_layers
    log_message(f"[INFO] Loading data for Trajectory: {trajectory_id}, a={a}, b={b}")

    try:
        data = pd.read_csv(data_file)
    except Exception as e:
        log_message(f"[ERROR] Error loading data: {e}")
        return
    
    num_inputs = (6 * a) + (4 * b)
    input_data = data.iloc[:-1, :num_inputs].values
    target_data = data.iloc[1:, -3:].values

    attempt = 0
    best_rmse = float("inf")

    while attempt < number_of_trials:
        MLP = MLPNetwork(
            learning_rate=learning_rate,
            number_of_input_neurons=num_inputs,
            number_of_hidden_neurons=hidden_neurons,
            number_of_layers = number_of_layers,
            number_of_output_neurons=3,
            dropout_rate=dropout_rate
        ).to("cuda")

        optimizer = torch.optim.Adam(MLP.parameters(), lr=learning_rate, weight_decay=regularization)
        loss_fn = torch.nn.MSELoss()

        checkpoint_path = os.path.join(
            CHECKPOINT_DIR, 
            f"MLP_lr{learning_rate}_drop{dropout_rate}_{trajectory_id}_hid{hidden_neurons}_ep{epochs}_stack{stacking_count}_a{a}_b{b}.pth"
        )

        print("[INFO] Model initialized. Starting training...")
        MLP.train()
        start_time = time.time()
        rmse_per_epoch = []

        for epoch in range(epochs):
            epoch_loss = 0
            for i in tqdm(range(len(input_data)), desc=f"Epoch {epoch + 1}/{epochs}"):
                x = torch.tensor(input_data[i], dtype=torch.float32, device="cuda")
                y = torch.tensor(target_data[i], dtype=torch.float32, device="cuda")
                optimizer.zero_grad()
                output = MLP.forward(x)
                loss = loss_fn(output, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(MLP.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_rmse = (epoch_loss / len(input_data)) ** 0.5
            rmse_per_epoch.append(avg_rmse)
            best_rmse = min(best_rmse, avg_rmse)
            log_message(f"[INFO] Epoch {epoch + 1}: RMSE = {avg_rmse:.5f}")
            torch.save(MLP.state_dict(), checkpoint_path)

        total_time = time.time() - start_time
        plot_path = os.path.join(PLOTS_DIR, f"MLP_lr{learning_rate}_drop{dropout_rate}_{trajectory_id}_hid{hidden_neurons}_ep{epochs}_stack{stacking_count}_a{a}_b{b}.png")
        plt.figure()
        plt.plot(range(1, epochs + 1), rmse_per_epoch, marker='o', linestyle='-', color='b')
        plt.xlabel("Epochs")
        plt.ylabel("RMSE")
        plt.title(f"MLP RMSE - Trajectory {trajectory_id}, a={a}, b={b}, Attempt {attempt}")
        plt.grid()
        plt.savefig(plot_path)
        plt.close()
        log_message(f"[INFO] Saved RMSE plot to {plot_path}")
        
        results_data = pd.DataFrame([{
            "Trajectory ID": trajectory_id,
            "6()": a,
            "4()": b,
            "Learning Rate": learning_rate,
            "Dropout Rate": dropout_rate,
            "Number of Hidden Neurons": hidden_neurons,
            "Epochs": epochs,
            "Stacking Count": stacking_count,
            "Time Taken": round(total_time, 2),
            "Best RMSE" : best_rmse
        }])

        if os.path.exists(RESULTS_PATH):
            results_data.to_csv(RESULTS_PATH, mode='a', header=False, index=False)
        else:
            results_data.to_csv(RESULTS_PATH, index=False)

        log_message(f"[INFO] Results saved to {RESULTS_PATH}")

        if best_rmse <= rmse_threshold:
            log_message(f"[SUCCESS] RMSE {best_rmse:.5f} is within threshold!")
            break
        else:
            log_message(f"[WARNING] RMSE {best_rmse:.5f} exceeds threshold. Retrying with new hyperparameters...")
            
            # ✅ Update hyperparameters for next attempt
            learning_rate *= 1.1
            dropout_rate *= 1.05
            hidden_neurons = min(hidden_neurons+20, 150)
            stacking_count += 1
            epochs += 5
            attempt += 1

if __name__ == "__main__":
    for traj_folder in os.listdir(DATA_DIR):
        traj_path = os.path.join(DATA_DIR, traj_folder)
        if not os.path.isdir(traj_path):
            continue
        for file in os.listdir(traj_path):
            file_path = os.path.join(traj_path, file)
            trajectory_id, a, b = parse_filename(file_path)
            if trajectory_id is not None and a is not None and b is not None:
                train_model(trajectory_id, file_path, a, b)