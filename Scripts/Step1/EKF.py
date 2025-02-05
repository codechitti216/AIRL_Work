import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ExtendedKalmanFilter:
    def __init__(self, dt):
        self.dt = dt  # Time step
        
        # State vector [px, py, pz, vx, vy, vz, ax, ay, az, wx, wy, wz, bpx, bpy, bpz, bvx, bvy, bvz, bax, bay, baz, bwx, bwy, bwz]
        self.X = np.zeros((24, 1))
        
        # Covariance matrix
        self.P = np.eye(24) * 0.1
        
        # Process noise covariance (IMU noise model with bias estimation)
        self.Q = np.diag([
            0.001, 0.001, 0.001,   # Position noise
            0.01, 0.01, 0.01,      # Velocity noise
            0.0001, 0.0001, 0.0001,  # Acceleration noise
            0.0001, 0.0001, 0.0001,  # Angular velocity noise
            0.00001, 0.00001, 0.00001,  # Position bias noise
            0.00001, 0.00001, 0.00001,  # Velocity bias noise
            0.00001, 0.00001, 0.00001,  # Acceleration bias noise
            0.00001, 0.00001, 0.00001   # Angular velocity bias noise
        ])
        
        # Measurement noise covariance
        self.R = np.eye(12) * 0.05  # Noise in position, velocity, acceleration, angular velocity
        
        # Measurement model (observes position, velocity, acceleration, angular velocity)
        self.H = np.zeros((12, 24))
        self.H[:12, :12] = np.eye(12)  # Direct observation of position, velocity, acceleration, angular velocity
    
    def f(self, X):
        """ Nonlinear process model for state propagation """
        X_new = np.copy(X)
        
        # Position update: x' = x + v*dt
        X_new[:3] += X[3:6] * self.dt
        
        # Velocity update: v' = v + a*dt
        X_new[3:6] += (X[6:9] - X[15:18]) * self.dt  # Correcting for acceleration bias
        
        # Angular velocity update: Correcting for gyro bias
        X_new[9:12] = X[9:12] - X[21:24]
        
        # Bias drift (modeled as slow random walk)
        X_new[12:] += np.random.normal(0, 0.0001, (12, 1)) * self.dt
        
        return X_new
    
    def predict(self):
        """ EKF Prediction Step """
        F = np.eye(24)
        F[:3, 3:6] = np.eye(3) * self.dt  # Position-velocity coupling
        F[3:6, 6:9] = np.eye(3) * self.dt  # Velocity-acceleration coupling
        
        self.X = self.f(self.X)
        self.P = F @ self.P @ F.T + self.Q
    
    def update(self, Z):
        """ EKF Measurement Update """
        y = Z - self.H @ self.X  # Innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman Gain
        
        self.X += K @ y
        self.P = (np.eye(24) - K @ self.H) @ self.P

def process_data(file_path):
    data = pd.read_csv(file_path)
    ekf = ExtendedKalmanFilter(dt=0.01)
    
    sensor_data, predicted_data, estimated_data = [], [], []
    
    for _, row in data.iterrows():
        Z = np.array([row['Longitude [rad]'], row['Latitude [rad]'], row['Altitude [m]'],
                      row['V North [m/s]'], row['V East [m/s]'], row['V Down [m/s]'],
                      row['ACC X [m/s^2]'], row['ACC Y [m/s^2]'], row['ACC Z [m/s^2]'],
                      row['GYRO X [rad/s]'], row['GYRO Y [rad/s]'], row['GYRO Z [rad/s]']]).reshape(-1, 1)
        
        ekf.predict()
        predicted_data.append(ekf.X[:12].flatten())
        ekf.update(Z)
        
        sensor_data.append(Z.flatten())
        estimated_data.append(ekf.X[:12].flatten())
    
    return np.array(sensor_data), np.array(predicted_data), np.array(estimated_data)

def plot_results(sensor_data, predicted_data, estimated_data):
    labels = ['X', 'Y', 'Z']
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    
    for i in range(3):  
        for j in range(4):  
            axes[i, j].plot(sensor_data[:, i + j*3], label='Sensor Data', linestyle='dotted', alpha=0.7)
            axes[i, j].plot(predicted_data[:, i + j*3], label='Predicted Data', linestyle='--', alpha=0.7)
            axes[i, j].plot(estimated_data[:, i + j*3], label='Estimated Data', linestyle='-', alpha=0.7)
            axes[i, j].set_title(f"{labels[i]} - Measurement {j+1}")
            axes[i, j].legend()
    
    plt.tight_layout()
    plt.show()
def main():
    file_path = '../../Data/Trajectory1/combined.csv'  # Update this with the actual file path
    sensor_data, predicted_data, estimated_data = process_data(file_path)
    plot_results(sensor_data, predicted_data, estimated_data)

if __name__ == "__main__":
    main()
