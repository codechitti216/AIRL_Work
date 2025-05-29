import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import signal
import sys
import traceback
from datetime import datetime
from tqdm import tqdm
from MNN_Stack import BeamVelocityStack, REQUIRED_VELOCITY_COLS
from MNN import MemoryNeuralNetwork

# Global variables for graceful exit
global stack, results_dir
stack = None
results_dir = None

# Signal handler for graceful exit
def signal_handler(sig, frame):
    print("\n[INFO] Interrupt received, saving current state before exit...")
    
    if stack is not None and results_dir is not None:
        try:
            # Save models if they exist
            model_save_dir = os.path.join(results_dir, 'trained_models')
            os.makedirs(model_save_dir, exist_ok=True)
            
            if stack.beam_model is not None and stack.velocity_model is not None:
                stack.save_models(model_save_dir)
                print(f"[INFO] Successfully saved models to {model_save_dir}")
            else:
                print("[WARNING] Models not fully initialized, could not save")
                
        except Exception as e:
            print(f"[ERROR] Failed to save models during interrupt: {e}")
            traceback.print_exc()
    else:
        print("[WARNING] Stack or results directory not initialized, nothing to save")
        
    print("[INFO] Exiting gracefully...")
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

def calculate_metrics(original, predicted):
    """Calculate RMSE for each component"""
    # Calculate RMSE for each component
    n_components = original.shape[1]
    rmse_values = []
    
    for i in range(n_components):
        diff = original[:, i] - predicted[:, i]
        mse = np.mean(diff ** 2)
        rmse = np.sqrt(mse)
        rmse_values.append(rmse)
    
    return rmse_values


def save_predictions_and_metrics(beam_predictions, velocity_predictions, beam_rmse, velocity_rmse, 
                               beam_dir, velocity_dir, hybrid_beams):
    """Save all prediction files and metrics to the specified directories"""
    # Load original beam data from files
    original_beams = pd.read_csv(os.path.join(beam_dir, 'original_beams.csv'))
    filled_beams = pd.read_csv(os.path.join(beam_dir, 'filled_beams.csv'))
    original_velocities = pd.read_csv(os.path.join(velocity_dir, 'original_velocities.csv'))
    
    # Calculate offsets
    beam_start = len(original_beams) - len(beam_predictions)
    vel_start = len(original_velocities) - len(velocity_predictions)
    
    # Save beam predictions
    pred_beam_df = pd.DataFrame()
    pred_beam_df['Time'] = original_beams['Time'].values[beam_start:]
    for i, col in enumerate(['b1', 'b2', 'b3', 'b4']):
        pred_beam_df[col] = beam_predictions[:, i]
    pred_beam_df.to_csv(os.path.join(beam_dir, 'predicted_beams.csv'), index=False)
    
    # Save hybrid beams (original where available, predicted where missing)
    hybrid_beams.to_csv(os.path.join(beam_dir, 'hybrid_beams.csv'), index=False)
    
    # Save beam metrics
    beam_metrics = pd.DataFrame()
    beam_metrics['Beam'] = ['b1', 'b2', 'b3', 'b4']
    beam_metrics['RMSE_vs_Original'] = beam_rmse
    
    # Calculate RMSE between predicted and moving averages (filled)
    filled_start = filled_beams.iloc[beam_start:].reset_index(drop=True)
    filled_array = filled_start[['b1', 'b2', 'b3', 'b4']].values
    filled_rmse = calculate_metrics(filled_array, beam_predictions)
    beam_metrics['RMSE_vs_MovingAvg'] = filled_rmse
    
    beam_metrics.to_csv(os.path.join(beam_dir, 'beam_metrics.csv'), index=False)
    
    # Save velocity files
    hybrid_beams.to_csv(os.path.join(velocity_dir, 'hybrid_beams.csv'), index=False)
    
    # Save predicted velocities
    pred_vel_df = pd.DataFrame()
    pred_vel_df['Time'] = original_velocities['Time'].values[vel_start:]
    for i, col in enumerate(REQUIRED_VELOCITY_COLS):
        pred_vel_df[col] = velocity_predictions[:, i]
    pred_vel_df.to_csv(os.path.join(velocity_dir, 'predicted_velocities.csv'), index=False)
    
    # Save velocity metrics
    vel_metrics = pd.DataFrame()
    vel_metrics['Component'] = REQUIRED_VELOCITY_COLS
    vel_metrics['RMSE'] = velocity_rmse
    vel_metrics.to_csv(os.path.join(velocity_dir, 'velocity_metrics.csv'), index=False)


def plot_beam_predictions(original_beams, predicted_beams, save_dir, traj_name):
    """Plot beam predictions"""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    time = original_beams['Time'].values
    start_t = len(time) - len(predicted_beams)
    time_pred = time[start_t:]
    
    for i in range(4):
        plt.plot(time, original_beams[f'b{i+1}'].values, label=f'Beam {i+1} (True)', alpha=0.7)
        plt.plot(time_pred, predicted_beams[:, i], '--', label=f'Beam {i+1} (Predicted)', alpha=0.7)
    
    plt.xlabel('Time')
    plt.ylabel('Beam Values')
    plt.title(f'Beam Predictions - {traj_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'beam_predictions_{traj_name}.png'))
    plt.close()

def plot_velocity_predictions(velocity_df, predicted_velocities, save_dir, traj_name):
    """Plot velocity predictions"""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    velocity_components = REQUIRED_VELOCITY_COLS
    vtime = velocity_df['Time'].values
    vstart_t = len(vtime) - len(predicted_velocities)
    vtime_pred = vtime[vstart_t:]
    
    for i, comp in enumerate(velocity_components):
        plt.plot(vtime, velocity_df[comp].values, label=f'{comp} (True)', alpha=0.7)
        plt.plot(vtime_pred, predicted_velocities[:, i], '--', label=f'{comp} (Predicted)', alpha=0.7)
    
    plt.xlabel('Time')
    plt.ylabel('Velocity (m/s)')
    plt.title(f'Velocity Predictions - {traj_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'velocity_predictions_{traj_name}.png'))
    plt.close()
def plot_rmse(beam_rmse, velocity_rmse, save_dir, traj_name):
    """Plot RMSE for both beam and velocity predictions"""
    # Create plots directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Beam RMSE plot
    plt.figure(figsize=(10, 6))
    beam_labels = ['b1', 'b2', 'b3', 'b4']
    plt.bar(beam_labels, beam_rmse)
    plt.ylabel('RMSE')
    plt.title(f'Beam Prediction RMSE - {traj_name}')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, f'beam_rmse_{traj_name}.png'))
    plt.close()
    
    # Velocity RMSE plot
    plt.figure(figsize=(10, 6))
    vel_labels = REQUIRED_VELOCITY_COLS
    plt.bar(vel_labels, velocity_rmse)
    plt.ylabel('RMSE')
    plt.title(f'Velocity Prediction RMSE - {traj_name}')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, f'velocity_rmse_{traj_name}.png'))
    plt.close()


def plot_original_vs_predictions(original_data, predictions, save_dir, traj_name, data_type):
    """Generic function to plot original vs predicted values
    
    Args:
        original_data: DataFrame containing original data
        predictions: numpy array of predictions
        save_dir: directory to save plots
        traj_name: trajectory name
        data_type: 'beam' or 'velocity'
    """
    # Create plots directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # Determine columns and labels based on data type
    if data_type.lower() == 'beam':
        columns = ['b1', 'b2', 'b3', 'b4']
        y_label = 'Beam Values'
        title = f'Beam Predictions - {traj_name}'
        filename = f'beam_predictions_{traj_name}.png'
    elif data_type.lower() == 'velocity':
        columns = REQUIRED_VELOCITY_COLS
        y_label = 'Velocity (m/s)'
        title = f'Velocity Predictions - {traj_name}'
        filename = f'velocity_predictions_{traj_name}.png'
    else:
        raise ValueError(f"Unknown data_type: {data_type}. Must be 'beam' or 'velocity'.")
    
    # Get time values and calculate start offset for predictions
    time = original_data['Time'].values
    start_t = len(time) - len(predictions)
    time_pred = time[start_t:]
    
    # Plot each component
    for i, col in enumerate(columns):
        plt.subplot(len(columns), 1, i+1)
        plt.plot(time, original_data[col].values, label=f'{col} (Original)', alpha=0.7)
        plt.plot(time_pred, predictions[:, i], '--', label=f'{col} (Predicted)', alpha=0.7)
        plt.ylabel(col)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        
        # Only show x-label for the bottom subplot
        if i == len(columns) - 1:
            plt.xlabel('Time')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

def save_trajectory_data(traj_name, traj_dir, original_beams, missed_beams, filled_beams, 
                        predicted_beams, beam_rmse, hybrid_beams, original_velocities, 
                        predicted_velocities, velocity_rmse):
    """Save all data files for a trajectory"""
    # Create directory structure
    beams_dir = os.path.join(traj_dir, 'beams')
    velocities_dir = os.path.join(traj_dir, 'velocities')
    plots_dir = os.path.join(traj_dir, 'plots')
    
    os.makedirs(beams_dir, exist_ok=True)
    os.makedirs(velocities_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save beam files
    original_beams.to_csv(os.path.join(beams_dir, 'original_beams.csv'), index=False)
    missed_beams.to_csv(os.path.join(beams_dir, 'missed_beams.csv'), index=False)
    filled_beams.to_csv(os.path.join(beams_dir, 'filled_beams.csv'), index=False)
    
    # Calculate start indices for beam and velocity data
    beam_start = len(original_beams) - len(predicted_beams)
    vel_start = len(original_velocities) - len(predicted_velocities)
    
    print(f"[DEBUG] Beam start: {beam_start}, Velocity start: {vel_start}")
    print(f"[DEBUG] Original beams: {len(original_beams)}, Predicted beams: {len(predicted_beams)}")
    print(f"[DEBUG] Original velocities: {len(original_velocities)}, Predicted velocities: {len(predicted_velocities)}")
    
    # Save predicted beams (convert numpy array to DataFrame)
    pred_beams_df = pd.DataFrame()
    pred_beams_df['Time'] = original_beams['Time'].values[beam_start:]
    for i in range(4):
        pred_beams_df[f'b{i+1}'] = predicted_beams[:, i]
    pred_beams_df.to_csv(os.path.join(beams_dir, 'predicted_beams.csv'), index=False)
    
    # Save beam metrics
    beam_metrics = pd.DataFrame()
    beam_metrics['Beam'] = ['b1', 'b2', 'b3', 'b4']
    beam_metrics['RMSE_vs_Original'] = beam_rmse
    
    # Calculate RMSE between predicted and moving averages (filled)
    filled_start = filled_beams.iloc[beam_start:].reset_index(drop=True)
    filled_array = filled_start[['b1', 'b2', 'b3', 'b4']].values
    filled_rmse = calculate_metrics(filled_array, predicted_beams)
    beam_metrics['RMSE_vs_MovingAvg'] = filled_rmse
    
    beam_metrics.to_csv(os.path.join(beams_dir, 'beam_metrics.csv'), index=False)
    
    # Save velocity files
    hybrid_beams.to_csv(os.path.join(velocities_dir, 'hybrid_beams.csv'), index=False)
    original_velocities.to_csv(os.path.join(velocities_dir, 'original_velocities.csv'), index=False)
    
    # Save predicted velocities
    pred_vel_df = pd.DataFrame()
    pred_vel_df['Time'] = original_velocities['Time'].values[vel_start:]
    for i, col in enumerate(REQUIRED_VELOCITY_COLS):
        pred_vel_df[col] = predicted_velocities[:, i]
    pred_vel_df.to_csv(os.path.join(velocities_dir, 'predicted_velocities.csv'), index=False)
    
    # Save velocity metrics
    vel_metrics = pd.DataFrame()
    vel_metrics['Component'] = REQUIRED_VELOCITY_COLS
    vel_metrics['RMSE'] = velocity_rmse
    vel_metrics.to_csv(os.path.join(velocities_dir, 'velocity_metrics.csv'), index=False)
    
    # Create plots
    plot_beam_predictions(original_beams, predicted_beams, plots_dir, traj_name)
    plot_velocity_predictions(original_velocities, predicted_velocities, plots_dir, traj_name)
    plot_rmse(beam_rmse, velocity_rmse, plots_dir, traj_name)

def process_trajectory(traj_config, stack, results_dir, is_training=True):
    """Process a single trajectory for training or testing"""
    # Get trajectory name and epochs
    if isinstance(traj_config, list):
        traj_name, traj_epochs = traj_config
    else:
        traj_name = traj_config
        traj_epochs = 1  # Default for testing
    
    print(f"\n[INFO] Processing trajectory: {traj_name}")
    
    # Create trajectory-specific directory
    traj_dir = os.path.join(results_dir, traj_name)
    os.makedirs(traj_dir, exist_ok=True)
    
    beam_predictions, velocity_predictions, hybrid_beams = None, None, None
    
    # Load data
    traj_path = os.path.join('Data_XYZ_change', traj_name)
    try:
        # Use enhanced load method to get missed beams as well
        beams_df, imu_df, velocity_df, original_beams, missed_beams = stack.load_csv_files_enhanced(traj_path)
        print(f"[INFO] Loaded and synchronized data: beams={beams_df.shape}, velocity={velocity_df.shape}, imu={imu_df.shape}")
    except Exception as e:
        print(f"[ERROR] Failed loading data for {traj_name}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Try to save the loaded data first, in case training fails
    try:
        # Save original data files - these don't depend on model training
        beam_dir = os.path.join(traj_dir, 'beams')
        velocity_dir = os.path.join(traj_dir, 'velocities')
        os.makedirs(beam_dir, exist_ok=True)
        os.makedirs(velocity_dir, exist_ok=True)
        
        original_beams.to_csv(os.path.join(beam_dir, 'original_beams.csv'), index=False)
        missed_beams.to_csv(os.path.join(beam_dir, 'missed_beams.csv'), index=False)
        beams_df.to_csv(os.path.join(beam_dir, 'filled_beams.csv'), index=False)
        velocity_df.to_csv(os.path.join(velocity_dir, 'original_velocities.csv'), index=False)
        print(f"[INFO] Saved original data files for {traj_name}")
    except Exception as e:
        print(f"[WARNING] Could not save original data files: {e}")
    
    if is_training:
        print(f"[WARNING] Using process_trajectory for training is deprecated.")
        print(f"[WARNING] Please use the two-phase training approach with stack.train() instead.")
        # Fallback to direct prediction
        try:
            beam_predictions, velocity_predictions, hybrid_beams = stack.predict_enhanced(traj_name)
        except Exception as e:
            print(f"[ERROR] Failed training on {traj_name}: {e}")
            import traceback
            traceback.print_exc()
            # Try to save the model after each trajectory, even if incomplete
            try:
                # Save partial models if they exist
                model_save_dir = os.path.join(results_dir, 'trained_models')
                if stack.beam_model is not None and stack.velocity_model is not None:
                    stack.save_models(model_save_dir)
                    print(f"[INFO] Partial models saved after error")
            except Exception as save_error:
                print(f"[ERROR] Could not save partial models: {save_error}")
            return False
    else:
        # Testing mode - just run predictions
        try:
            print(f"[INFO] Running predictions for trajectory: {traj_name}")
            beam_predictions, velocity_predictions, hybrid_beams = stack.predict_enhanced(traj_name)
        except Exception as e:
            print(f"[ERROR] Failed generating predictions for {traj_name}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Calculate metrics - ensure arrays are the same size
    start_t = stack.config['num_past_beam_instances']
    
    # For beam metrics
    # Get the actual start offset in the original data
    beam_offset = len(original_beams) - len(beam_predictions)
    original_beams_subset = original_beams.iloc[beam_offset:].reset_index(drop=True)
    beam_array = original_beams_subset[['b1', 'b2', 'b3', 'b4']].values
    
    print(f"[INFO] Original beams shape: {beam_array.shape}, Predicted beams shape: {beam_predictions.shape}")
    beam_rmse = calculate_metrics(beam_array, beam_predictions)
    
    # For velocity metrics
    vel_offset = len(velocity_df) - len(velocity_predictions)
    velocity_df_subset = velocity_df.iloc[vel_offset:].reset_index(drop=True)
    velocity_array = velocity_df_subset[REQUIRED_VELOCITY_COLS].values
    
    print(f"[INFO] Original velocity shape: {velocity_array.shape}, Predicted velocity shape: {velocity_predictions.shape}")
    velocity_rmse = calculate_metrics(velocity_array, velocity_predictions)
    
    # Save trajectory data and generate plots
    try:
        save_trajectory_data(
            traj_name=traj_name,
            traj_dir=traj_dir,
            original_beams=original_beams,
            missed_beams=missed_beams,
            filled_beams=beams_df,  # This is the filled beam data with moving averages
            predicted_beams=beam_predictions,
            beam_rmse=beam_rmse,
            hybrid_beams=hybrid_beams,
            original_velocities=velocity_df,
            predicted_velocities=velocity_predictions,
            velocity_rmse=velocity_rmse
        )
        print(f"[INFO] All data files and plots saved for {traj_name}")
    except Exception as e:
        print(f"[ERROR] Failed to save some data files for {traj_name}: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to save models before returning, even if data saving failed
        if is_training:
            try:
                model_save_dir = os.path.join(results_dir, 'trained_models')
                if stack.beam_model is not None and stack.velocity_model is not None:
                    stack.save_models(model_save_dir)
                    print(f"[INFO] Models saved despite data saving failure")
            except Exception as save_error:
                print(f"[ERROR] Could not save models: {save_error}")
        return False
    return True

def main():
    # Make variables global so signal handler can access them
    global stack, results_dir
    
    # Load configuration
    config_path = 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'Results/results_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Create trained_models directory
    model_save_dir = os.path.join(results_dir, 'trained_models')
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Initialize stacked model
    stack = BeamVelocityStack(config)
    
    # Get training and testing trajectories
    training_trajectories = config['training_trajectories']
    testing_trajectories = config['testing_trajectories']
    
    try:
        # Train the model using two-phase approach
        print("\n[INFO] Starting two-phase training...")
        stack.train(training_trajectories)
        
        # Save trained models
        if stack.beam_model is not None and stack.velocity_model is not None:
            stack.save_models(model_save_dir)
            print(f"[INFO] Saved trained models to {model_save_dir}")
        
        # Process testing trajectories
        if testing_trajectories:
            print("\n[INFO] Processing testing trajectories...")
            for traj_name in testing_trajectories:
                process_trajectory(traj_name, stack, results_dir, is_training=False)
                
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Try to save models one last time
        try:
            if stack.beam_model is not None and stack.velocity_model is not None:
                stack.save_models(model_save_dir)
                print(f"[INFO] Saved models before exit")
        except Exception as e:
            print(f"[ERROR] Failed to save models before exit: {e}")

if __name__ == '__main__':
    main()