import os
import re
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from MNN import MemoryNeuralNetwork

# --------- Helper Functions for Parsing Architecture Parameters ---------

def parse_folder_name(folder_path):
    folder_name = os.path.basename(folder_path)
    print(f"[DEBUG] Parsing folder name: {folder_name}")
    hn_match = re.search(r"hn(\d+)", folder_name)
    bfw_match = re.search(r"bfw(\d+)", folder_name)
    if not hn_match:
        print(f"[ERROR] 'hn' not found in folder name: {folder_name}")
        return None, None
    if not bfw_match:
        print(f"[ERROR] 'bfw' not found in folder name: {folder_name}")
        return None, None
    hidden_neurons = int(hn_match.group(1))
    beam_fill_window = int(bfw_match.group(1))
    print(f"[INFO] Extracted hidden_neurons={hidden_neurons}, beam_fill_window={beam_fill_window} from folder name")
    return hidden_neurons, beam_fill_window

def parse_checkpoint_name(filename):
    print(f"[DEBUG] Parsing checkpoint file name: {filename}")
    npbi_match = re.search(r"npbi(\d+)", filename)
    niu_match = re.search(r"niu(\d+)", filename)
    nl_match = re.search(r"nl(\d+)", filename)
    if not npbi_match:
        print(f"[ERROR] 'npbi' not found in checkpoint file name: {filename}")
        return None, None, None
    if not niu_match:
        print(f"[ERROR] 'niu' not found in checkpoint file name: {filename}")
        return None, None, None
    if not nl_match:
        print(f"[ERROR] 'nl' not found in checkpoint file name: {filename}")
        return None, None, None
    num_past_beam_instances = int(npbi_match.group(1))
    num_imu_instances = int(niu_match.group(1))
    num_layers = int(nl_match.group(1))
    print(f"[INFO] Extracted npbi={num_past_beam_instances}, niu={num_imu_instances}, nl={num_layers} from checkpoint file name")
    return num_past_beam_instances, num_imu_instances, num_layers

def compute_input_size(num_past_beam_instances, num_imu_instances):
    num_inputs = 4 + (num_past_beam_instances * 4) + (num_imu_instances * 6)
    print(f"[INFO] Computed input size: 4 + ({num_past_beam_instances}*4) + ({num_imu_instances}*6) = {num_inputs}")
    return num_inputs

# --------- Data Processing and Plotting Functions ---------

def load_csv_files(traj_path):
    print(f"[INFO] Loading CSV files from trajectory folder: {traj_path}")
    try:
        beams_gt = pd.read_csv(os.path.join(traj_path, "beams_gt.csv"))
        velocity_gt = pd.read_csv(os.path.join(traj_path, "velocity_gt.csv"))
        print(f"[INFO] Loaded beams_gt ({len(beams_gt)} rows) and velocity_gt ({len(velocity_gt)} rows)")
    except Exception as e:
        print(f"[ERROR] Failed to load beams_gt or velocity_gt: {e}")
        raise e

    try:
        imu_files = [f for f in os.listdir(traj_path) if f.startswith("IMU_") and f.endswith(".csv")]
        if not imu_files:
            raise FileNotFoundError("No IMU file found")
        imu = pd.read_csv(os.path.join(traj_path, imu_files[0]))
        print(f"[INFO] Loaded IMU file: {imu_files[0]} with {len(imu)} rows")
    except Exception as e:
        print(f"[ERROR] Failed to load IMU file: {e}")
        raise e

    beams_gt.sort_values("Time", inplace=True)
    velocity_gt.sort_values("Time", inplace=True)
    imu.sort_values("Time [s]", inplace=True)
    beams_gt.reset_index(drop=True, inplace=True)
    velocity_gt.reset_index(drop=True, inplace=True)
    imu.reset_index(drop=True, inplace=True)
    return beams_gt, beams_gt.copy(), imu, velocity_gt

def fill_missing_beams(beams_df, beam_fill_window, beam_cols=["b1", "b2", "b3", "b4"]):
    print(f"[INFO] Filling missing beams with beam_fill_window={beam_fill_window}")
    filled = beams_df.copy()
    for i in range(beam_fill_window, len(filled)):
        for col in beam_cols:
            if pd.isna(filled.loc[i, col]):
                window = filled.loc[i - beam_fill_window:i - 1, col]
                if window.isna().all():
                    filled.loc[i, col] = filled[col].ffill().iloc[i - 1]
                    print(f"[WARN] All previous values missing for {col} at row {i}, using last valid value")
                else:
                    filled.loc[i, col] = window.mean()
                    print(f"[INFO] Filled missing {col} at row {i} with moving average")
    return filled, beam_fill_window

def construct_input_target(filled_beams, velocity_gt, imu, t, num_past_beam_instances, num_imu_instances):
    try:
        current_beams = filled_beams.loc[t, ["b1", "b2", "b3", "b4"]].values.astype(float)
        print(f"[DEBUG] Current beams at row {t}: {current_beams}")
    except Exception as e:
        print(f"[ERROR] Failed to get current beams at row {t}: {e}")
        raise e

    past_beams = []
    for i in range(1, num_past_beam_instances + 1):
        try:
            past_row = filled_beams.loc[t - i, ["b1", "b2", "b3", "b4"]].values.astype(float)
            past_beams.extend(past_row)
        except Exception as e:
            print(f"[ERROR] Failed to get past beams at row {t-i}: {e}")
            raise e

    imu_cols = ['ACC X [m/s^2]', 'ACC Y [m/s^2]', 'ACC Z [m/s^2]',
                'GYRO X [rad/s]', 'GYRO Y [rad/s]', 'GYRO Z [rad/s]']
    past_imu = []
    for i in range(num_imu_instances):
        try:
            imu_row = imu.loc[t - i, imu_cols].values.astype(float)
            past_imu.extend(imu_row)
        except Exception as e:
            print(f"[ERROR] Failed to get IMU data at row {t-i}: {e}")
            raise e

    input_vector = np.concatenate([current_beams, np.array(past_beams), np.array(past_imu)])
    try:
        target_vector = velocity_gt.loc[t, ["V North", "V East", "V Down"]].values.astype(float)
    except Exception as e:
        print(f"[ERROR] Failed to get target velocities at row {t}: {e}")
        raise e
    print(f"[DEBUG] Constructed input vector (length={len(input_vector)}) and target vector at row {t}")
    return input_vector, target_vector

def plot_velocity_predictions(predictions, traj, title_suffix=""):
    print(f"[INFO] Plotting Prediction vs Ground Truth for trajectory: {traj}")
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    samples = [pred["Sample"] for pred in predictions]
    components = ["V North", "V East", "V Down"]
    for i, comp in enumerate(components):
        try:
            pred_vals = [pred[f"Pred_{comp}"] for pred in predictions]
        except KeyError as ke:
            print(f"[ERROR] Key error while retrieving prediction values for {comp}: {ke}")
            raise
        try:
            gt_vals = [pred[f"GT_{comp}"] for pred in predictions]
        except KeyError as ke:
            print(f"[ERROR] Key error while retrieving ground truth values for {comp}: {ke}")
            raise

        axes[i].plot(samples, gt_vals, label=f"Ground Truth {comp}",linestyle='-')
        axes[i].plot(samples, pred_vals, label=f"Predicted {comp}",linestyle='-')
        axes[i].set_ylabel(comp)
        axes[i].legend(loc="upper right")
        axes[i].grid(True)
        print(f"[DEBUG] {comp} - First 5 GT: {gt_vals[:5]}, Pred: {pred_vals[:5]}")
    axes[-1].set_xlabel("Sample Index")
    fig.suptitle(f"Predicted vs GT for {traj} {title_suffix}")
    return fig

def plot_pred_error(predictions, traj, title_suffix=""):
    print(f"[INFO] Plotting Squared Error for trajectory: {traj}")
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    samples = [pred["Sample"] for pred in predictions]
    components = ["V North", "V East", "V Down"]
    for i, comp in enumerate(components):
        try:
            pred_vals = [pred[f"Pred_{comp}"] for pred in predictions]
        except KeyError as ke:
            print(f"[ERROR] Key error while retrieving prediction values for {comp}: {ke}")
            raise
        try:
            gt_vals = [pred[f"GT_{comp}"] for pred in predictions]
        except KeyError as ke:
            print(f"[ERROR] Key error while retrieving ground truth values for {comp}: {ke}")
            raise

        sq_error = [(p - g) ** 2 for p, g in zip(pred_vals, gt_vals)]
        axes[i].plot(samples, sq_error, label=f"Squared Error {comp}", marker='o')
        axes[i].set_ylabel("Squared Error")
        axes[i].legend(loc="upper right")
        axes[i].grid(True)
        print(f"[DEBUG] {comp} - First 5 Squared Errors: {sq_error[:5]}")
    axes[-1].set_xlabel("Sample Index")
    fig.suptitle(f"Squared Error (Pred vs GT) for {traj} {title_suffix}")
    return fig

# --------- Inference Functions ---------

def scan_all_checkpoints(base_folder):
    print(f"[INFO] Scanning for checkpoints under base folder: {base_folder}")
    checkpoint_files = []
    for root, dirs, files in os.walk(base_folder):
        if os.path.basename(root).lower() == "checkpoints":
            parent_folder = os.path.dirname(root)  # Full experiment folder path
            print(f"[DEBUG] Found Checkpoints folder under: {parent_folder}")
            for file in files:
                if file.endswith(".pth"):
                    full_path = os.path.join(root, file)
                    print(f"[DEBUG] Found checkpoint file: {file} in {root}")
                    checkpoint_files.append((parent_folder, file, full_path))
    print(f"[INFO] Total checkpoints found: {len(checkpoint_files)}")
    return checkpoint_files

def run_inference_for_checkpoint(model, checkpoint_path, parent_folder, ckpt_filename, trajectory, config):
    # Construct a unique identifier using the full parent folder name
    checkpoint_id = f"{os.path.basename(parent_folder)}_{os.path.splitext(ckpt_filename)[0]}"
    print(f"[INFO] Running inference for checkpoint: {checkpoint_id} on trajectory: {trajectory}")
    
    traj_path = os.path.join("..", "..", "Data", trajectory)
    if not os.path.exists(traj_path):
        print(f"[ERROR] Trajectory path does not exist: {traj_path}")
        return None
    
    try:
        beams_gt, beams_training, imu, velocity_gt = load_csv_files(traj_path)
    except Exception as e:
        print(f"[ERROR] Failed to load CSV files for trajectory {trajectory}: {e}")
        return None

    # Extract architecture parameters from the full parent folder name
    hidden_neurons, beam_fill_window = parse_folder_name(parent_folder)
    if hidden_neurons is None or beam_fill_window is None:
        print("[ERROR] Could not determine architecture parameters from folder name.")
        return None

    try:
        filled_beams, _ = fill_missing_beams(beams_training, beam_fill_window)
    except Exception as e:
        print(f"[ERROR] Failed during beam filling for trajectory {trajectory}: {e}")
        return None

    num_past_beam_instances, num_imu_instances, num_layers = parse_checkpoint_name(ckpt_filename)
    if None in (num_past_beam_instances, num_imu_instances, num_layers):
        print(f"[ERROR] Could not extract history parameters from checkpoint file name: {ckpt_filename}")
        return None

    num_inputs = compute_input_size(num_past_beam_instances, num_imu_instances)
    print(f"[INFO] Model Input Size: {num_inputs}")
    
    # Determine the starting index based on history requirements
    min_history = max(beam_fill_window, num_past_beam_instances, num_imu_instances - 1)
    inputs, targets, predictions = [], [], []
    for t in range(min_history, min(len(filled_beams), len(velocity_gt))):
        try:
            inp, tar = construct_input_target(filled_beams, velocity_gt, imu, t, num_past_beam_instances, num_imu_instances)
            inputs.append(inp)
            targets.append(tar)
        except Exception as e:
            print(f"[WARN] Skipping sample at index {t} due to error: {e}")
            continue

    if len(inputs) == 0:
        print(f"[ERROR] Not enough data for trajectory {trajectory} after processing.")
        return None

    inputs = np.array(inputs)
    targets = np.array(targets)
    num_samples = len(inputs)
    print(f"[INFO] Constructed {num_samples} input samples for trajectory {trajectory}")

    try:
        print(f"[INFO] Loading model weights from checkpoint: {checkpoint_path}")
        # Use weights_only=True to avoid potential security issues in future releases
        state = torch.load(checkpoint_path, map_location=model.device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        print(f"[INFO] Successfully loaded weights for checkpoint: {checkpoint_id}")
    except Exception as e:
        print(f"[ERROR] Failed to load model weights from {checkpoint_path}: {e}")
        return None

    for i in range(num_samples):
        try:
            x = torch.tensor(inputs[i], dtype=torch.float32, device=model.device).unsqueeze(0)
            with torch.no_grad():
                y_pred = model(x).squeeze().view(-1)
            predictions.append({
                "Sample": i,
                "Pred_V North": y_pred[0].item(),
                "Pred_V East": y_pred[1].item(),
                "Pred_V Down": y_pred[2].item(),
                "GT_V North": targets[i][0],
                "GT_V East": targets[i][1],
                "GT_V Down": targets[i][2]
            })
        except Exception as e:
            print(f"[ERROR] Inference error at sample {i}: {e}")
            continue

    output_dir = os.path.join(config["output_folder"], trajectory)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Saving outputs to directory: {output_dir}")

    try:
        pred_plot = plot_velocity_predictions(predictions, trajectory, title_suffix="(Inference)")
        pred_plot_filename = os.path.join(output_dir, f"{checkpoint_id}_pred_vs_gt.png")
        pred_plot.savefig(pred_plot_filename)
        plt.close('all')
        print(f"[INFO] Saved Prediction vs GT plot: {pred_plot_filename}")
    except Exception as e:
        print(f"[ERROR] Failed to save Prediction vs GT plot: {e}")

    try:
        error_plot = plot_pred_error(predictions, trajectory, title_suffix="(Inference)")
        error_plot_filename = os.path.join(output_dir, f"{checkpoint_id}_squared_error.png")
        error_plot.savefig(error_plot_filename)
        plt.close('all')
        print(f"[INFO] Saved Squared Error plot: {error_plot_filename}")
    except Exception as e:
        print(f"[ERROR] Failed to save Squared Error plot: {e}")

    # Compute test summary metrics (RMSE for each velocity component)
    squared_errors = np.zeros(3)
    for pred in predictions:
        error = np.array([
            pred["Pred_V North"] - pred["GT_V North"],
            pred["Pred_V East"] - pred["GT_V East"],
            pred["Pred_V Down"] - pred["GT_V Down"]
        ]) ** 2
        squared_errors += error
    test_rmse = np.sqrt(squared_errors / num_samples)
    summary = {
        "Trajectory": trajectory,
        "NumSamples": num_samples,
        "Test_RMSE_V North": test_rmse[0],
        "Test_RMSE_V East": test_rmse[1],
        "Test_RMSE_V Down": test_rmse[2],
        "AvgTest_RMSE": float(np.mean(test_rmse)),
        "TrainedOn": checkpoint_id,
        "RemovalMethod": "extracted_from_checkpoint"
    }
    print(f"[INFO] Test summary for trajectory {trajectory} with checkpoint {checkpoint_id}: {summary}")
    return summary

def main():
    print("[INFO] Starting inference process...")
    try:
        with open("Inference.json", "r") as f:
            config = json.load(f)
        print(f"[INFO] Loaded inference configuration from Inference.json")
    except Exception as e:
        print(f"[ERROR] Failed to load Inference.json: {e}")
        return

    checkpoints = scan_all_checkpoints(config["checkpoints_base_folder"])
    if not checkpoints:
        print("[ERROR] No checkpoint files found. Exiting.")
        return

    all_results = []
    for parent_folder, ckpt_filename, ckpt_path in checkpoints:
        print(f"[INFO] Processing checkpoint file: {ckpt_filename} from folder: {parent_folder}")
        hidden_neurons, folder_bfw = parse_folder_name(parent_folder)
        if hidden_neurons is None or folder_bfw is None:
            print(f"[ERROR] Skipping checkpoint from folder {parent_folder} due to parsing error.")
            continue
        npbi, niu, num_layers = parse_checkpoint_name(ckpt_filename)
        if None in (npbi, niu, num_layers):
            print(f"[ERROR] Skipping checkpoint file {ckpt_filename} due to parsing error.")
            continue
        num_inputs = compute_input_size(npbi, niu)
        print(f"[INFO] Creating model with num_inputs={num_inputs}, hidden_neurons={hidden_neurons}")
        
        model = MemoryNeuralNetwork(
            number_of_input_neurons=num_inputs,
            number_of_hidden_neurons=hidden_neurons,
            number_of_output_neurons=3,
            dropout_rate=0.0,  # Inference doesn't need dropout
            learning_rate=0.001,
            learning_rate_2=0.0005,
            lipschitz_constant=0.0
        )
        
        checkpoint_id = f"{os.path.basename(parent_folder)}_{os.path.splitext(ckpt_filename)[0]}"
        print(f"[INFO] Unique checkpoint identifier: {checkpoint_id}")
        
        for traj in config["inference_trajectories"]:
            print(f"[INFO] Running inference on trajectory: {traj} using checkpoint: {checkpoint_id}")
            summary = run_inference_for_checkpoint(model, ckpt_path, parent_folder, ckpt_filename, traj, config)
            if summary is not None:
                all_results.append(summary)
            else:
                print(f"[WARN] Inference for trajectory {traj} with checkpoint {checkpoint_id} failed or returned no results.")
    
    # Save aggregated test summary CSV
    output_dir = config["output_folder"]
    os.makedirs(output_dir, exist_ok=True)
    summary_csv = os.path.join(output_dir, "inference_summary.csv")
    try:
        pd.DataFrame(all_results).to_csv(summary_csv, index=False)
        print(f"[INFO] Inference complete. Test summary saved to: {summary_csv}")
    except Exception as e:
        print(f"[ERROR] Failed to save test summary CSV: {e}")

if __name__ == "__main__":
    main()
