import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from MNN_Stack import BeamVelocityStack

def plot_predictions(beams_df, predicted_beams, velocity_df, predicted_velocities, save_dir, traj_name):
    """Plot both beam and velocity predictions"""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot beam predictions
    time = beams_df['Time'].values
    start_t = len(time) - len(predicted_beams)
    time_pred = time[start_t:]
    for i in range(4):
        ax1.plot(time, beams_df[f'b{i+1}'].values, label=f'Beam {i+1} (True)', alpha=0.7)
        ax1.plot(time_pred, predicted_beams[:, i], '--', label=f'Beam {i+1} (Predicted)', alpha=0.7)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Beam Values')
    ax1.set_title(f'Beam Predictions - {traj_name}')
    ax1.legend()
    ax1.grid(True)
    
    # Plot velocity predictions
    velocity_components = ['V North', 'V East', 'V Down']
    vtime = velocity_df['Time'].values
    vstart_t = len(vtime) - len(predicted_velocities)
    vtime_pred = vtime[vstart_t:]
    for i, comp in enumerate(velocity_components):
        ax2.plot(vtime, velocity_df[comp].values, label=f'{comp} (True)', alpha=0.7)
        ax2.plot(vtime_pred, predicted_velocities[:, i], '--', label=f'{comp} (Predicted)', alpha=0.7)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title(f'Velocity Predictions - {traj_name}')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'predictions_{traj_name}.png'))
    plt.close()

def calculate_metrics(true_values, predicted_values):
    """Calculate RMSE for predictions"""
    mse = np.mean((true_values - predicted_values) ** 2, axis=0)
    rmse = np.sqrt(mse)
    return rmse

def main():
    # Load configuration
    with open('FinalStack/config.json', 'r') as f:
        config = json.load(f)
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'results_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Create subdirectories for each component
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Initialize stacked model once for all trajectories
    print("\n[INFO] Initializing model stack for sequential training across all trajectories")
    stack = BeamVelocityStack(config)
    model_save_dir = os.path.join(results_dir, 'trained_models')
    
    # Process each training trajectory
    for traj_info in config['training_trajectories']:
        traj_name, traj_epochs = traj_info
        print(f"\nProcessing training trajectory: {traj_name}")
        
        # Create trajectory-specific directory
        traj_dir = os.path.join(results_dir, traj_name)
        os.makedirs(traj_dir, exist_ok=True)
        
        # Load data using the stack's load_csv_files method
        traj_path = os.path.join('FinalStack', 'data', traj_name)
        try:
            beams_df, imu_df, velocity_df = stack.load_csv_files(traj_path)
        except Exception as e:
            print(f"Error loading data for {traj_name}: {e}")
            continue
        
        # Train beam model (continues from previous trajectory if exists)
        try:
            stack.beam_model = stack.train_beam_model(beams_df, imu_df, traj_name, traj_epochs)
        except Exception as e:
            print(f"Error training beam model for {traj_name}: {e}")
            continue
        
        # Generate predicted beams for velocity model training
        try:
            predicted_beams = stack.predict_beams_only(beams_df, imu_df)
        except Exception as e:
            print(f"Error generating beam predictions for {traj_name}: {e}")
            continue
        
        # Train velocity model (continues from previous trajectory if exists)
        try:
            stack.velocity_model = stack.train_velocity_model(predicted_beams, imu_df, velocity_df, traj_name, traj_epochs)
        except Exception as e:
            print(f"Error training velocity model for {traj_name}: {e}")
            continue
        
        # Save models after each trajectory (in case of interruption)
        try:
            stack.save_models(model_save_dir)
        except Exception as e:
            print(f"Error saving models for {traj_name}: {e}")
            continue
        
        # Evaluate on training data
        try:
            predicted_beams, predicted_velocities = stack.predict(beams_df, imu_df, traj_dir)
        except Exception as e:
            print(f"Error evaluating models for {traj_name}: {e}")
            continue
        
        # Calculate metrics
        start_t = max(config['num_past_beam_instances'], config['num_imu_instances'] - 1)
        beam_rmse = calculate_metrics(beams_df[['b1', 'b2', 'b3', 'b4']].values[start_t:], predicted_beams)
        velocity_rmse = calculate_metrics(velocity_df[['V North', 'V East', 'V Down']].values[start_t:start_t + len(predicted_velocities)], predicted_velocities)
        
        # Save metrics
        metrics = {
            'beam_rmse': beam_rmse.tolist(),
            'velocity_rmse': velocity_rmse.tolist(),
            'model_config': config
        }
        with open(os.path.join(traj_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Plot predictions
        plot_predictions(beams_df, predicted_beams, velocity_df, predicted_velocities, plots_dir, traj_name)
        
        print(f"Results saved in {traj_dir}")
    
    # Process testing trajectories
    if 'testing_trajectories' in config:
        print("\nProcessing testing trajectories...")
        for traj_name in config['testing_trajectories']:
            print(f"\nProcessing testing trajectory: {traj_name}")
            
            # Create trajectory-specific directory
            traj_dir = os.path.join(results_dir, f"test_{traj_name}")
            os.makedirs(traj_dir, exist_ok=True)
            
            # Load data
            traj_path = os.path.join('FinalStack', 'Data', traj_name)
            try:
                beams_df, imu_df, velocity_df = stack.load_csv_files(traj_path)
            except Exception as e:
                print(f"Error loading data for {traj_name}: {e}")
                continue
            
            # Create a new stack and load trained models
            test_stack = BeamVelocityStack(config)
            try:
                test_stack.load_models(model_save_dir)
            except Exception as e:
                print(f"Error loading trained models for {traj_name}: {e}")
                continue
            
            # Generate predictions
            try:
                predicted_beams, predicted_velocities = test_stack.predict(beams_df, imu_df, traj_dir)
            except Exception as e:
                print(f"Error generating predictions for {traj_name}: {e}")
                continue
            
            # Calculate metrics
            start_t = max(config['num_past_beam_instances'], config['num_imu_instances'] - 1)
            beam_rmse = calculate_metrics(beams_df[['b1', 'b2', 'b3', 'b4']].values[start_t:], predicted_beams)
            velocity_rmse = calculate_metrics(velocity_df[['V North', 'V East', 'V Down']].values[start_t:start_t + len(predicted_velocities)], predicted_velocities)
            
            # Save metrics
            metrics = {
                'beam_rmse': beam_rmse.tolist(),
                'velocity_rmse': velocity_rmse.tolist(),
                'model_config': config
            }
            with open(os.path.join(traj_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=4)
            
            # Plot predictions
            plot_predictions(beams_df, predicted_beams, velocity_df, predicted_velocities, plots_dir, f"test_{traj_name}")
            
            print(f"Test results saved in {traj_dir}")

if __name__ == '__main__':
    main() 