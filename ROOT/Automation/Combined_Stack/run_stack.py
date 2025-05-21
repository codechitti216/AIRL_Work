import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MNN_Stack import BeamVelocityStack

def plot_predictions(predicted_beams, predicted_velocities, ground_truth_beams, ground_truth_velocities, traj_name, save_dir):
    """Plot both beam and velocity predictions"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot beam predictions
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for i, beam in enumerate(['b1', 'b2', 'b3', 'b4']):
        axes[i].plot(ground_truth_beams[:, i], label=f'Ground Truth {beam}', color='blue')
        axes[i].plot(predicted_beams[:, i], label=f'Predicted {beam}', color='red')
        axes[i].set_ylabel(f'{beam} Value')
        axes[i].legend()
        axes[i].grid(True)
    axes[-1].set_xlabel('Sample Index')
    plt.suptitle(f'Beam Predictions for {traj_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{traj_name}_beam_predictions.png'))
    plt.close()

    # Plot velocity predictions
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    components = ['V North', 'V East', 'V Down']
    for i, comp in enumerate(components):
        axes[i].plot(ground_truth_velocities[:, i], label=f'Ground Truth {comp}', color='blue')
        axes[i].plot(predicted_velocities[:, i], label=f'Predicted {comp}', color='red')
        axes[i].set_ylabel(f'{comp} (m/s)')
        axes[i].legend()
        axes[i].grid(True)
    axes[-1].set_xlabel('Sample Index')
    plt.suptitle(f'Velocity Predictions for {traj_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{traj_name}_velocity_predictions.png'))
    plt.close()

def calculate_metrics(predicted, ground_truth):
    """Calculate RMSE for predictions"""
    mse = np.mean((predicted - ground_truth) ** 2, axis=0)
    rmse = np.sqrt(mse)
    return rmse

def main():
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Create results directory
    results_dir = os.path.join('Results', 'Stacked_Model')
    os.makedirs(results_dir, exist_ok=True)

    # Initialize the stacked model
    stack = BeamVelocityStack(config)

    # Process each training trajectory
    for traj_pair in config['training_trajectories']:
        traj_name, traj_epochs = traj_pair
        print(f"\nProcessing trajectory: {traj_name}")

        # Load and preprocess data
        traj_path = os.path.join('../../Data', traj_name)
        beams_df, imu_df, velocity_df = stack.load_csv_files(traj_path)

        # Train beam prediction model
        print("\nTraining beam prediction model...")
        stack.train_beam_model(beams_df, imu_df, traj_name, traj_epochs)

        # Get beam predictions for velocity model training
        print("\nGenerating beam predictions for velocity model training...")
        predicted_beams, _ = stack.predict(beams_df, imu_df)

        # Train velocity prediction model
        print("\nTraining velocity prediction model...")
        stack.train_velocity_model(predicted_beams, imu_df, velocity_df, traj_name, traj_epochs)

        # Save models
        model_save_dir = os.path.join(results_dir, traj_name)
        stack.save_models(model_save_dir)
        print(f"\nModels saved to {model_save_dir}")

        # Evaluate on training data
        print("\nEvaluating on training data...")
        final_predicted_beams, final_predicted_velocities = stack.predict(beams_df, imu_df)

        # Calculate metrics
        beam_rmse = calculate_metrics(final_predicted_beams, beams_df[['b1', 'b2', 'b3', 'b4']].values)
        velocity_rmse = calculate_metrics(final_predicted_velocities, velocity_df[['V North', 'V East', 'V Down']].values)

        print("\nBeam Prediction RMSE:")
        for i, beam in enumerate(['b1', 'b2', 'b3', 'b4']):
            print(f"{beam}: {beam_rmse[i]:.4f}")

        print("\nVelocity Prediction RMSE:")
        for i, comp in enumerate(['V North', 'V East', 'V Down']):
            print(f"{comp}: {velocity_rmse[i]:.4f}")

        # Plot predictions
        plot_predictions(
            final_predicted_beams,
            final_predicted_velocities,
            beams_df[['b1', 'b2', 'b3', 'b4']].values,
            velocity_df[['V North', 'V East', 'V Down']].values,
            traj_name,
            os.path.join(results_dir, 'plots')
        )

        # Save predictions to trajectory-specific files
        beam_predictions = pd.DataFrame(final_predicted_beams, columns=['b1', 'b2', 'b3', 'b4'])
        beam_predictions['Time'] = beams_df['Time'].iloc[start_t:start_t + len(final_predicted_beams)].values
        beam_predictions = beam_predictions[['Time', 'b1', 'b2', 'b3', 'b4']]
        beam_predictions.to_csv(os.path.join(results_dir, f'{traj_name}_predicted_beams.csv'), index=False)

        velocity_predictions = pd.DataFrame(final_predicted_velocities, columns=['V North', 'V East', 'V Down'])
        velocity_predictions['Time'] = beams_df['Time'].iloc[start_t:start_t + len(final_predicted_velocities)].values
        velocity_predictions = velocity_predictions[['Time', 'V North', 'V East', 'V Down']]
        velocity_predictions.to_csv(os.path.join(results_dir, f'{traj_name}_predicted_velocities.csv'), index=False)

if __name__ == "__main__":
    main() 