import numpy as np
import matplotlib.pyplot as plt

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
        Update step (Corrects the state estimate using measurements)
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
        """
        Return the current state estimate
        """
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

# Simulated sensor data (real measurements would come from sensors)
z_acc = np.array([0.1, 0.2, 0.3])  # Accelerometer measurements (with noise)
z_gyro = np.array([0.05, -0.02, 0.1])  # Gyroscope measurements (with noise)
z_dvl = np.array([1.2, 0.1, -0.5])  # DVL velocity measurements (with noise)

# Measurement vector combining all sensor measurements
z = np.concatenate([z_acc, z_gyro, z_dvl])

# Store the results for plotting
estimated_biasses = []

# Kalman filter loop: Predict and Update steps for 100 iterations
for _ in range(100):  # Loop for 100 iterations
    # Prediction step
    kf.predict(F)
    
    # Update step with the sensor measurements
    kf.update(z, H)
    
    # Get the current state estimate (including biases)
    estimated_state = kf.get_state()
    
    # Store estimated biases (accelerometer, gyroscope, DVL)
    estimated_biasses.append(estimated_state[6:15])  # We store only the biases (indices 6-14)

# Convert the results into a NumPy array for easier manipulation
estimated_biasses = np.array(estimated_biasses)

# Plotting the estimated biases over 100 iterations
fig, axs = plt.subplots(3, 1, figsize=(10, 8))

# Plot accelerometer biases
axs[0].plot(estimated_biasses[:, 0], label="Accelerometer Bias (X)")
axs[0].plot(estimated_biasses[:, 1], label="Accelerometer Bias (Y)")
axs[0].plot(estimated_biasses[:, 2], label="Accelerometer Bias (Z)")
axs[0].set_title("Accelerometer Biases over Time")
axs[0].set_xlabel("Iterations")
axs[0].set_ylabel("Bias Value")
axs[0].legend()

# Plot gyroscope biases
axs[1].plot(estimated_biasses[:, 3], label="Gyroscope Bias (X)")
axs[1].plot(estimated_biasses[:, 4], label="Gyroscope Bias (Y)")
axs[1].plot(estimated_biasses[:, 5], label="Gyroscope Bias (Z)")
axs[1].set_title("Gyroscope Biases over Time")
axs[1].set_xlabel("Iterations")
axs[1].set_ylabel("Bias Value")
axs[1].legend()

# Plot DVL biases
axs[2].plot(estimated_biasses[:, 6], label="DVL Bias (X)")
axs[2].plot(estimated_biasses[:, 7], label="DVL Bias (Y)")
axs[2].plot(estimated_biasses[:, 8], label="DVL Bias (Z)")
axs[2].set_title("DVL Biases over Time")
axs[2].set_xlabel("Iterations")
axs[2].set_ylabel("Bias Value")
axs[2].legend()

plt.tight_layout()
plt.show()
