
import pandas as pd
import matplotlib.pyplot as plt

# Replace these with your actual CSV file paths
csv_file1 = '../Data/Trajectory4/beams_gt.csv'
csv_file2 = '../Data/Trajectory4/beams_gt.csv'


# Read the CSV files
df1 = pd.read_csv(csv_file1)
df2 = pd.read_csv(csv_file2)

# Convert the "Time" column to NumPy arrays (or use the index if missing)
if 'Time' in df1.columns and 'Time' in df2.columns:
    time1 = df1['Time'].to_numpy()
    time2 = df2['Time'].to_numpy()
else:
    time1 = df1.index.to_numpy()
    time2 = df2.index.to_numpy()

# Define the beam columns we wish to plot
beam_columns = ['b1', 'b2', 'b3', 'b4']
colors = ['blue', 'red']  # Blue for first file, red for second file

# Create a figure with 4 subplots (one for each beam)
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 12), sharex=True)

for i, col in enumerate(beam_columns):
    ax = axes[i]
    # Plot beam values from the first CSV file
    ax.plot(time1, df1[col].to_numpy(), label=f'{csv_file1} - {col}', color=colors[0])
    # Plot beam values from the second CSV file
    ax.plot(time2, df2[col].to_numpy(), label=f'{csv_file2} - {col}', color=colors[1], linestyle='--')
    
    ax.set_ylabel('Beam Value')
    ax.set_title(f'Comparison of {col}')
    ax.legend()
    ax.grid(True)

axes[-1].set_xlabel('Time')
plt.tight_layout()
plt.show()