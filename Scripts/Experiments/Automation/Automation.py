import json
import os
import subprocess
import pandas as pd
import torch
from concurrent.futures import ProcessPoolExecutor

# Paths
config_paths = {
    "LSTM": "LSTM_config.json",
    "FAN": "FAN_config.json",
    "MNN": "MNN_config.json"
}
experiment_scripts = {
    "LSTM": "LSTM_experiment.py",
    "FAN": "FAN_experiment.py",
    "MNN": "MNN_experiment.py"
}
results_path = "Results.csv"
checkpoint_dir = "Checkpoints"

# Load JSON config for each model
def load_config(model_name):
    with open(config_paths[model_name], "r") as f:
        return json.load(f)

# Check if a model needs retraining based on RMSE threshold
def needs_retraining(results_df, model_name, trajectory_id, a, b, rmse_threshold=0.05):
    model_results = results_df[(results_df["Model Type"] == model_name) & 
                               (results_df["Trajectory ID"] == trajectory_id) & 
                               (results_df["6()"] == a) & 
                               (results_df["4()"] == b)]
    
    if model_results.empty:
        return True  # No previous training exists
    
    best_rmse = model_results["Best RMSE Loss"].min()
    return best_rmse > rmse_threshold

# Retrain model with modified hyperparameters
def update_hyperparameters(model_name):
    config = load_config(model_name)
    config["hidden_neurons"] = int(config["hidden_neurons"] * 1.5)
    config["learning_rate"] *= 1.5
    config["stacking_count"] += 1
    config["epochs"] += 10  # More epochs for retraining
    
    with open(config_paths[model_name], "w") as f:
        json.dump(config, f, indent=4)

# Run training for a model
def run_model(model_name):
    print(f"Starting training for {model_name}...")
    subprocess.run(["python", experiment_scripts[model_name]])

# Run models in parallel
def run_all_models():
    with ProcessPoolExecutor() as executor:
        executor.map(run_model, ["LSTM", "FAN", "MNN"])

# Main execution flow
if __name__ == "__main__":
    # Load previous results
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
    else:
        results_df = pd.DataFrame(columns=["Model Type", "Learning Rate", "Trajectory ID", "No. of Hidden Neurons", "Loss Function", "Regularization", "6()", "4()", "No. of Epochs", "Stacking Count", "Time Taken", "Best RMSE Loss"])

    # Run models in parallel
    run_all_models()

    # Check RMSE for retraining
    for model in ["LSTM", "FAN", "MNN"]:
        for index, row in results_df.iterrows():
            trajectory_id, a, b = row["Trajectory ID"], row["6()"], row["4()"]
            if needs_retraining(results_df, model, trajectory_id, a, b):
                print(f"Retraining {model} for Trajectory {trajectory_id}...")
                update_hyperparameters(model)
                run_model(model)

    print("All training and retraining completed.")
