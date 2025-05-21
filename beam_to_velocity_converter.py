import os
import numpy as np
import pandas as pd
from pathlib import Path

def create_beam_matrix():
    """Create the beam transformation matrix A."""
    b_vectors = [
        [np.cos((45 + i * 90) * np.pi / 180) * np.sin(20 * np.pi / 180),
         np.sin((45 + i * 90) * np.pi / 180) * np.sin(20 * np.pi / 180),
         np.cos(20 * np.pi / 180)]
        for i in range(4)
    ]
    return np.array(b_vectors).reshape((4, 3))

def convert_beams_to_velocities(beams_df):
    """Convert beam measurements to velocities using pseudo-inverse."""
    # Create beam transformation matrix
    A = create_beam_matrix()
    
    # Calculate pseudo-inverse of A
    A_pinv = np.linalg.pinv(A)
    
    # Extract beam measurements
    beams = beams_df[['b1', 'b2', 'b3', 'b4']].values
    
    # Convert beams to velocities using the correct matrix multiplication
    velocities = np.dot(beams, A_pinv.T)
    
    # Create DataFrame with velocities
    velocity_df = pd.DataFrame(velocities, columns=['V North', 'V East', 'V Down'])
    velocity_df.insert(0, 'Time', beams_df['Time'].values)
    
    return velocity_df

def process_results_folders():
    """Process all results folders and their test trajectories."""
    # Get current directory
    current_dir = Path('.')
    
    # Find all results folders
    results_folders = [f for f in current_dir.iterdir() if f.is_dir() and f.name.startswith('results_')]
    
    for results_folder in results_folders:
        print(f"\nProcessing {results_folder.name}...")
        
        # Find all test trajectory folders
        test_folders = [f for f in results_folder.iterdir() 
                       if f.is_dir() and f.name.startswith('test_Trajectory')]
        
        for test_folder in test_folders:
            print(f"  Processing {test_folder.name}...")
            
            # Path to beam predictions
            beams_path = test_folder / 'beam_predictions' / 'predicted_beams.csv'
            
            if not beams_path.exists():
                print(f"    Warning: No beam predictions found in {beams_path}")
                continue
            
            try:
                # Read beam predictions
                beams_df = pd.read_csv(beams_path)
                
                # Convert to velocities
                velocity_df = convert_beams_to_velocities(beams_df)
                
                # Save velocities
                output_path = test_folder / 'beam_predictions' / 'predicted_velocities.csv'
                output_path.parent.mkdir(exist_ok=True)  # Create directory if it doesn't exist
                velocity_df.to_csv(output_path, index=False)
                
                print(f"    Successfully created {output_path}")
                
            except Exception as e:
                print(f"    Error processing {test_folder.name}: {str(e)}")

if __name__ == "__main__":
    process_results_folders() 