import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define the Kalman Filter class
class KalmanFilter:
    def __init__(self, state_size, measurement_size):
        # Initialize state vector and measurement vector sizes
        self.state_size = state_size
        self.measurement_size = measurement_size
        
        # Initialize state estimate (x) and covariance (P)
        self.x = np.zeros(state_size)  # State vector
        self.P = np.eye(state_size)    # Covariance matrix (initial uncertainty)
        
        # Process noise covariance (Q) and measurement noise covariance (R)
        self.Q = np.eye(state_size) * 0.1  # Process noise (small for bias estimation)
        self.R = np.eye(measurement_size) * 1.0  # Measurement noise (based on sensor error)

    def predict(self, F):
        """
        Prediction step
        """
        # Prediction of the state: x' = F * x
        self.x = np.dot(F, self.x)
        
        # Prediction of the error covariance: P' = F * P * F^T + Q
        self.P = np.dot(np.dot(F, self.P), F.T) + self.Q

    def update(self, z, H):
        """
        Update step
        """
        # Compute the residual (difference between the measured and predicted values)
        y = z - np.dot(H, self.x)  # y = z - H * x
        
        # Compute the Kalman Gain: K = P * H^T * (H * P * H^T + R)^-1
        S = np.dot(np.dot(H, self.P), H.T) + self.R
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))
        
        # Update the state estimate: x = x' + K * y
        self.x = self.x + np.dot(K, y)
        
        # Update the error covariance: P = (I - K * H) * P
        I = np.eye(self.state_size)
        self.P = np.dot(I - np.dot(K, H), self.P)

    def get_state(self):
        return self.x

# Define the dimensions of the state and measurement vectors
state_size = 15  # 3 velocities, 3 accelerations, 3 angular velocities, 3 biases for acc, 3 for gyro, 3 for DVL
measurement_size = 9  # 3 accelerometer, 3 gyroscope, 3 DVL measurements

# Initialize the Kalman Filter
kf = KalmanFilter(state_size, measurement_size)

# Define the state transition matrix F
F = np.eye(state_size)  # Assume constant motion with no change in bias

# Define the measurement matrix H for sensor bias estimation
H = np.zeros((measurement_size, state_size))
H[0, 6] = 1  # Accelerometer x bias
H[1, 7] = 1  # Accelerometer y bias
H[2, 8] = 1  # Accelerometer z bias
H[3, 9] = 1  # Gyroscope x bias
H[4, 10] = 1  # Gyroscope y bias
H[5, 11] = 1  # Gyroscope z bias
H[6, 12] = 1  # DVL x bias
H[7, 13] = 1  # DVL y bias
H[8, 14] = 1  # DVL z bias

# Load the data from CSV file
data = pd.read_csv('../Data/Data_Cleaned.csv')  # Replace with your actual CSV file path

# Extract sensor data
z_acc = data[['ACC X [m/s^2]', 'ACC Y [m/s^2]', 'ACC Z [m/s^2]']].values
z_gyro = data[['GYRO X [rad/s]', 'GYRO Y [rad/s]', 'GYRO Z [rad/s]']].values
z_dvl = data[['DVL X [m/s]', 'DVL Y [m/s]', 'DVL Z [m/s]']].values

# Combine sensor data into a single measurement vector (z)
z = np.hstack((z_acc, z_gyro, z_dvl))

# Store the results for plotting
estimated_biasses = []
corrected_acc = []
corrected_gyro = []
corrected_dvl = []

# Kalman filter loop: Predict and Update steps for each time step in the data
for i in range(len(data)):  # Loop over the rows in the CSV data
    # Prediction step
    kf.predict(F)
    
    # Update step with the sensor measurements for the current iteration
    current_z = z[i, :]  # Current measurement vector
    kf.update(current_z, H)
    
    # Get the current state estimate (including biases)
    estimated_state = kf.get_state()
    
    # Store the estimated biases (accelerometer, gyroscope, DVL)
    estimated_biasses.append(estimated_state[6:15])  # We store only the biases (indices 6-14)
    
    # Correct the sensor values using the estimated biases
    corrected_acc.append(current_z[0:3] - estimated_state[6:9])  # Correct accelerometer data
    corrected_gyro.append(current_z[3:6] - estimated_state[9:12])  # Correct gyroscope data
    corrected_dvl.append(current_z[6:9] - estimated_state[12:15])  # Correct DVL data

# Convert to numpy arrays for easier manipulation
corrected_acc = np.array(corrected_acc)
corrected_gyro = np.array(corrected_gyro)
corrected_dvl = np.array(corrected_dvl)

# Plotting the corrected values vs sensor values
fig, axs = plt.subplots(3, 1, figsize=(10, 8))

# Plot accelerometer
axs[0].plot(z_acc[:, 0], label="Accelerometer X - Sensor")
axs[0].plot(corrected_acc[:, 0], label="Accelerometer X - Corrected")
axs[0].plot(z_acc[:, 1], label="Accelerometer Y - Sensor")
axs[0].plot(corrected_acc[:, 1], label="Accelerometer Y - Corrected")
axs[0].plot(z_acc[:, 2], label="Accelerometer Z - Sensor")
axs[0].plot(corrected_acc[:, 2], label="Accelerometer Z - Corrected")
axs[0].set_title("Accelerometer Measurements: Sensor vs Corrected")
axs[0].set_xlabel("Iterations")
axs[0].set_ylabel("Acceleration (m/s^2)")
axs[0].legend()

# Plot gyroscope
axs[1].plot(z_gyro[:, 0], label="Gyroscope X - Sensor")
axs[1].plot(corrected_gyro[:, 0], label="Gyroscope X - Corrected")
axs[1].plot(z_gyro[:, 1], label="Gyroscope Y - Sensor")
axs[1].plot(corrected_gyro[:, 1], label="Gyroscope Y - Corrected")
axs[1].plot(z_gyro[:, 2], label="Gyroscope Z - Sensor")
axs[1].plot(corrected_gyro[:, 2], label="Gyroscope Z - Corrected")
axs[1].set_title("Gyroscope Measurements: Sensor vs Corrected")
axs[1].set_xlabel("Iterations")
axs[1].set_ylabel("Angular Velocity (rad/s)")
axs[1].legend()

# Plot DVL
axs[2].plot(z_dvl[:, 0], label="DVL X - Sensor")
axs[2].plot(corrected_dvl[:, 0], label="DVL X - Corrected")
axs[2].plot(z_dvl[:, 1], label="DVL Y - Sensor")
axs[2].plot(corrected_dvl[:, 1], label="DVL Y - Corrected")
axs[2].plot(z_dvl[:, 2], label="DVL Z - Sensor")
axs[2].plot(corrected_dvl[:, 2], label="DVL Z - Corrected")
axs[2].set_title("DVL Measurements: Sensor vs Corrected")
axs[2].set_xlabel("Iterations")
axs[2].set_ylabel("Velocity (m/s)")
axs[2].legend()

plt.tight_layout()
plt.show()