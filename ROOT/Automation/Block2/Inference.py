import os
import sys
import json
import re
import time
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from MNN import MemoryNeuralNetwork

DATA_DIR = "../../Data"
VALID_BEAM_MIN = -1.1
VALID_BEAM_MAX = 1.5
REQUIRED_VELOCITY_COLS = ["V North", "V East", "V Down"]
START_FILL_INDEX = 10

def load_csv_files(traj_path):
    beams_df = pd.read_csv(os.path.join(traj_path, "beams_gt.csv"), na_values=[''])
    imu_files = [f for f in os.listdir(traj_path) if f.startswith("IMU_") and f.endswith(".csv")]
    if not imu_files:
        raise ValueError(f"No IMU file found in {traj_path}")
    imu_df = pd.read_csv(os.path.join(traj_path, imu_files[0]))
    beams_df['Time'] = beams_df['Time'].astype(str)
    imu_df['Time'] = imu_df['Time [s]'].astype(str)
    common_times = set(beams_df['Time']) & set(imu_df['Time'])
    if not common_times:
        raise ValueError("No common timestamps found between beams and IMU data.")
    beams_df = beams_df[beams_df['Time'].isin(common_times)].sort_values("Time").reset_index(drop=True)
    imu_df = imu_df[imu_df['Time'].isin(common_times)].sort_values("Time").reset_index(drop=True)
    return beams_df, imu_df

def remove_invalid_beam_values(beams_df, beam_cols=["b1", "b2", "b3", "b4"]):
    beams_df = beams_df.copy()
    for col in beam_cols:
        beams_df.loc[(beams_df[col] < VALID_BEAM_MIN) | (beams_df[col] > VALID_BEAM_MAX), col] = np.nan
    return beams_df

def compute_switch_value(beams_df, beam_cols=["b1", "b2", "b3", "b4"]):
    beams_df["SWITCH VALUE"] = beams_df[beam_cols].isna().sum(axis=1).apply(lambda x: 0 if x >= 2 else 1)
    return beams_df

def fill_missing_beams(beams_df, beam_fill_window, beam_cols=["b1", "b2", "b3", "b4"]):
    filled = beams_df.copy()
    for i in range(beam_fill_window + 3, len(filled)):
        for col in beam_cols:
            if pd.isna(filled.loc[i, col]):
                window = filled.loc[i - beam_fill_window:i - 1, col]
                if window.isna().all():
                    ffill_value = filled[col].ffill().iloc[i - 1]
                    filled.loc[i, col] = ffill_value
                else:
                    filled.loc[i, col] = window.mean()
                if pd.isna(filled.loc[i, col]):
                    print(f"[ERROR] Still NaN after filling '{col}' at index {i}. Exiting.")
                    sys.exit(1)
    return filled

def construct_input_sample(filled_beams, imu_df, t, num_past_beam_instances, num_imu_instances):
    current_beams = filled_beams.loc[t, ["b1", "b2", "b3", "b4"]].values.astype(float)
    if np.isnan(current_beams).any():
        sys.exit(1)
    past_beams = []
    for i in range(1, num_past_beam_instances + 1):
        past_row = filled_beams.loc[t - i, ["b1", "b2", "b3", "b4"]].values.astype(float)
        if np.isnan(past_row).any():
            sys.exit(1)
        past_beams.extend(past_row)
    imu_cols = ['ACC X [m/s^2]', 'ACC Y [m/s^2]', 'ACC Z [m/s^2]',
                'GYRO X [rad/s]', 'GYRO Y [rad/s]', 'GYRO Z [rad/s]']
    past_imu = []
    for i in range(num_imu_instances - 1, -1, -1):
        imu_values = imu_df.loc[t - i, imu_cols].values.astype(float)
        if np.isnan(imu_values).any():
            sys.exit(1)
        past_imu.extend(imu_values)
    input_vector = np.concatenate([current_beams, np.array(past_beams), np.array(past_imu)])
    if np.isnan(input_vector).any():
        sys.exit(1)
    return input_vector

def find_first_valid_index(filled_beams, imu_df, num_past_beam_instances, num_imu_instances):
    start = max(num_past_beam_instances, num_imu_instances - 1)
    for t in range(start, len(filled_beams)):
        try:
            _ = construct_input_sample(filled_beams, imu_df, t, num_past_beam_instances, num_imu_instances)
            return t
        except Exception:
            continue
    return None

def run_inference(trajectory, config, checkpoint_path):
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error

    output_csv = trajectory + "_inference_results.csv"
    traj_path = os.path.join(DATA_DIR, trajectory)
    print(f"[INFO] Running inference on trajectory: {trajectory}")
    beams_df, imu_df = load_csv_files(traj_path)
    beams_df = remove_invalid_beam_values(beams_df)
    beams_df = compute_switch_value(beams_df)
    beam_fill_window = config.get("beam_fill_window", 5)
    filled_beams = fill_missing_beams(beams_df, beam_fill_window)
    filled_beams = filled_beams.iloc[START_FILL_INDEX:].reset_index(drop=True)
    imu_df = imu_df.iloc[START_FILL_INDEX:].reset_index(drop=True)

    first_valid = find_first_valid_index(filled_beams, imu_df,
                                         config["num_past_beam_instances"],
                                         config["num_imu_instances"])
    if first_valid is None:
        raise ValueError("No valid starting index found in trajectory " + trajectory)

    start_t = max(config["num_past_beam_instances"], config["num_imu_instances"] - 1)
    inputs, valid_indices = [], []
    for t in range(start_t, len(filled_beams)):
        sample = construct_input_sample(filled_beams, imu_df, t,
                                        config["num_past_beam_instances"],
                                        config["num_imu_instances"])
        inputs.append(sample)
        valid_indices.append(t)

    inputs = np.array(inputs)
    input_size = inputs.shape[1]
    model = MemoryNeuralNetwork(number_of_input_neurons=input_size,
                                number_of_hidden_neurons=config["hidden_neurons"],
                                number_of_output_neurons=3,
                                dropout_rate=config["dropout_rate"],
                                learning_rate=config["learning_rate"],
                                learning_rate_2=config["learning_rate_2"],
                                lipschitz_constant=config["lipschitz_constant"])
    model.load_state_dict(torch.load(checkpoint_path, map_location=model.device))
    model.eval()

    predictions = []
    for idx, sample in enumerate(inputs):
        sample_time = filled_beams.loc[valid_indices[idx], "Time"]
        switch_value = filled_beams.loc[valid_indices[idx], "SWITCH VALUE"]
        x = torch.tensor(sample, dtype=torch.float32, device=model.device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            y_pred = model(x).squeeze().view(-1)
            pred_velocities = y_pred.cpu().numpy().tolist()
            predictions.append({
                "Time": sample_time,
                "V_X": pred_velocities[0],
                "V Y": pred_velocities[1],
                "V_Z": pred_velocities[2],
                "SWITCH VALUE": switch_value
            })

    output_path = os.path.join(os.getcwd(), output_csv)
    pd.DataFrame(predictions).to_csv(output_path, index=False)
    print(f"[INFO] Inference complete. Predictions saved to {output_path}")

    # --- Plot: Predicted vs Actual Velocities (3 subplots) ---
    velocity_gt_path = os.path.join(traj_path, "velocity_gt.csv")
    velocity_df = pd.read_csv(velocity_gt_path).iloc[START_FILL_INDEX:].reset_index(drop=True)
    velocity_df = velocity_df.loc[valid_indices].reset_index(drop=True)
    pred_df = pd.DataFrame(predictions).reset_index(drop=True)

    gt_vn = velocity_df["V North"].to_numpy()
    gt_ve = velocity_df["V East"].to_numpy()
    gt_vd = velocity_df["V Down"].to_numpy()
    pr_vn = pred_df["V_X"].to_numpy()
    pr_ve = pred_df["V Y"].to_numpy()
    pr_vd = pred_df["V_Z"].to_numpy()
    switch_vals = pred_df["SWITCH VALUE"].to_numpy()
    x_vals = np.arange(len(gt_vn))

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for i, (gt, pr, label, color) in enumerate(zip(
        [gt_vn, gt_ve, gt_vd],
        [pr_vn, pr_ve, pr_vd],
        ["V North", "V East", "V Down"],
        ['blue', 'green', 'red']
    )):
        axs[i].plot(x_vals, gt, label=f"GT {label}", color="blue", linestyle='-')
        axs[i].plot(x_vals, pr, label=f"Pred {label}", color="magenta")
        for j in x_vals:
            if switch_vals[j] == 0:
                axs[i].axvline(x=j, color='gray', linestyle=':', alpha=0.3)
                axs[i].text(j, max(gt.max(), pr.max()), "SKIPPED", rotation=90, fontsize=7,
                            ha='center', va='bottom', color='gray')
        axs[i].set_ylabel(f"{label}")
        axs[i].legend()
        axs[i].grid(True)

    axs[2].set_xlabel("Sample Index")

    valid_mask = switch_vals == 1
    if np.any(valid_mask):
        rmse_vn = np.sqrt(mean_squared_error(gt_vn[valid_mask], pr_vn[valid_mask]))
        rmse_ve = np.sqrt(mean_squared_error(gt_ve[valid_mask], pr_ve[valid_mask]))
        rmse_vd = np.sqrt(mean_squared_error(gt_vd[valid_mask], pr_vd[valid_mask]))
        rmse_text = f"RMSE V_N: {rmse_vn:.4f} | V_E: {rmse_ve:.4f} | V_D: {rmse_vd:.4f}"
    else:
        rmse_text = "RMSE SKIPPED — All SWITCH VALUE = 0"

    plt.suptitle(f"Predicted vs Ground Truth Velocities\n{rmse_text}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(trajectory + "_inference_velocity_plot.png")
    print(f"[INFO] Saved 3-subplot velocity plot to '{trajectory}_inference_velocity_plot.png'")
        # --- Plot: RMSE per timestep (3 subplots) ---
    rmse_vn_per_step, rmse_ve_per_step, rmse_vd_per_step = [], [], []
    for i in x_vals:
        if switch_vals[i] == 1:
            rmse_vn_per_step.append((pr_vn[i] - gt_vn[i])**2)
            rmse_ve_per_step.append((pr_ve[i] - gt_ve[i])**2)
            rmse_vd_per_step.append((pr_vd[i] - gt_vd[i])**2)
        else:
            rmse_vn_per_step.append(np.nan)
            rmse_ve_per_step.append(np.nan)
            rmse_vd_per_step.append(np.nan)

    rmse_vn_per_step = np.sqrt(np.array(rmse_vn_per_step))
    rmse_ve_per_step = np.sqrt(np.array(rmse_ve_per_step))
    rmse_vd_per_step = np.sqrt(np.array(rmse_vd_per_step))

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for i, (rmse_vals, label, color) in enumerate(zip(
        [rmse_vn_per_step, rmse_ve_per_step, rmse_vd_per_step],
        ["V North", "V East", "V Down"],
        ['blue', 'green', 'red']
    )):
        axs[i].plot(x_vals, rmse_vals, label=f"RMSE {label}", color="red")
        for j in x_vals:
            if switch_vals[j] == 0:
                axs[i].axvline(x=j, color='gray', linestyle=':', alpha=0.3)
                axs[i].text(j, np.nanmax(rmse_vals), "SKIPPED", rotation=90, fontsize=7,
                            ha='center', va='bottom', color='gray')
        axs[i].set_ylabel(f"{label}")
        axs[i].legend()
        axs[i].grid(True)

    axs[2].set_xlabel("Sample Index")
    plt.suptitle("Per-Timestep RMSE for Velocity Components\n(SWITCH = 0 marked as SKIPPED)", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(trajectory + "_inference_rmse_plot.png")
    print(f"[INFO] Saved 3-subplot RMSE plot to '{trajectory}_inference_rmse_plot.png'")


def main():
    with open("MNN.json", "r") as f:
        config = json.load(f)
    trajectory = "Trajectory9"
    checkpoint_path = "Checkpoints/final_model_checkpoint.pth"
    run_inference(trajectory, config, checkpoint_path)

if __name__ == "__main__":
    main()
