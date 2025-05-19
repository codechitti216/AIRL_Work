import pandas as pd
import matplotlib.pyplot as plt

# Replace these with your actual CSV file paths
csv_file1 = '../Data/Trajectory1/velocity_gt.csv'
csv_file2 = '../Data/Trajectory6/velocity_gt.csv'

# Read the CSV files
df1 = pd.read_csv(csv_file1)
df2 = pd.read_csv(csv_file2)

# Read the CSV files
df1 = pd.read_csv(csv_file1)
df2 = pd.read_csv(csv_file2)

# Convert the "Time" column to a NumPy array
if 'Time' in df1.columns and 'Time' in df2.columns:
    time1 = df1['Time'].to_numpy()
    time2 = df2['Time'].to_numpy()
else:
    time1 = df1.index.to_numpy()
    time2 = df2.index.to_numpy()

# List of velocity columns to plot
velocity_columns = ['V North', 'V East', 'V Down']
colors = ['blue', 'red']  # Colors for file1 and file2 respectively

# Create a figure with 3 subplots (one for each velocity component)
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), sharex=True)

for i, col in enumerate(velocity_columns):
    ax = axes[i]
    # Plot from first CSV file (convert Series to numpy arrays)
    ax.plot(time1, df1[col].to_numpy(), label=f'{csv_file1} - {col}', color=colors[0])
    # Plot from second CSV file
    ax.plot(time2, df2[col].to_numpy(), label=f'{csv_file2} - {col}', color=colors[1], linestyle='--')
    
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title(f'Comparison of {col}')
    ax.legend()
    ax.grid(True)

axes[-1].set_xlabel('Time')

plt.tight_layout()
plt.show()