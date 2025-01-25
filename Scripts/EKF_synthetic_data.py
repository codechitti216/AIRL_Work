import numpy as np
import matplotlib.pyplot as plt

def f(state, dt):
    x, y, z, vx, vy, vz, ax, ay, az, bx, by, bz, bvx, bvy, bvz, bax, bay, baz = state
    new_x = x + vx * dt + 0.5 * ax * dt**2 + bx
    new_y = y + vy * dt + 0.5 * ay * dt**2 + by
    new_z = z + vz * dt + 0.5 * az * dt**2 + bz
    new_vx = vx + ax * dt + bvx
    new_vy = vy + ay * dt + bvy
    new_vz = vz + az * dt + bvz
    new_ax = ax + bax
    new_ay = ay + bay
    new_az = az + baz
    return np.array([new_x, new_y, new_z, new_vx, new_vy, new_vz,
                     new_ax, new_ay, new_az, bx, by, bz, bvx, bvy, bvz, bax, bay, baz])

def h(state):
    x, y, z, vx, vy, vz, ax, ay, az, bx, by, bz, bvx, bvy, bvz, bax, bay, baz = state
    z_x = x + bx
    z_y = y + by
    z_z = z + bz
    z_vx = vx + bvx
    z_vy = vy + bvy
    z_vz = vz + bvz
    z_ax = ax + bax
    z_ay = ay + bay
    z_az = az + baz
    return np.array([z_x, z_y, z_z, z_vx, z_vy, z_vz, z_ax, z_ay, z_az])

def compute_jacobians(state, dt):
    F = np.eye(18)
    F[0, 3] = dt  
    F[0, 6] = 0.5 * dt**2  
    F[1, 4] = dt  
    F[1, 7] = 0.5 * dt**2  
    F[2, 5] = dt  
    F[2, 8] = 0.5 * dt**2  
    F[3, 6] = dt  
    F[4, 7] = dt  
    F[5, 8] = dt  

    H = np.zeros((9, 18))
    H[0, 0] = 1; H[0, 9] = 1  
    H[1, 1] = 1; H[1, 10] = 1  
    H[2, 2] = 1; H[2, 11] = 1  
    H[3, 3] = 1; H[3, 12] = 1  
    H[4, 4] = 1; H[4, 13] = 1  
    H[5, 5] = 1; H[5, 14] = 1  
    H[6, 6] = 1; H[6, 15] = 1  
    H[7, 7] = 1; H[7, 16] = 1  
    H[8, 8] = 1; H[8, 17] = 1  

    return F, H

def ekf(z, initial_state, P_0, dt, Q, R, num_steps):
    state = np.array(initial_state)
    P = np.array(P_0)
    estimates = []
    predictions = []
    for k in range(num_steps):
        F, H = compute_jacobians(state, dt)
        state_pred = f(state, dt)
        P_pred = F @ P @ F.T + Q
        z_pred = h(state_pred)
        y = z[:, k] - z_pred  
        S = H @ P_pred @ H.T + R  
        K = P_pred @ H.T @ np.linalg.inv(S)  
        state = state_pred + K @ y
        P = (np.eye(len(P)) - K @ H) @ P_pred
        predictions.append(state_pred)
        estimates.append(state)
    return np.array(estimates), np.array(predictions)

if __name__ == "__main__":
    x_0 = [0] * 18  
    P_0 = np.eye(18) * 1e-3  
    Q = np.eye(18) * 1e-5  
    R = np.eye(9) * 1e-1  
    dt = 0.1  
    num_steps = 50  

    true_measurements = np.array([
        [i * dt, i * dt * 0.5, 0.1 * np.sin(i * dt), 
         1, 0.5, 0.2, 
         0.01, 0.02, 0.03] for i in range(num_steps)
    ]).T
    noisy_measurements = true_measurements + np.random.normal(0, 0.1, true_measurements.shape)

    estimated_states, predicted_states = ekf(noisy_measurements, x_0, P_0, dt, Q, R, num_steps)

    time = np.arange(num_steps) * dt
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))
    labels = ["Position (X, Y, Z)", "Velocity (Vx, Vy, Vz)", "Acceleration (Ax, Ay, Az)"]
    for i in range(3):
        for j in range(3):
            idx = i * 3 + j
            axs[i, j].plot(time, true_measurements[idx], label="True", color="black", linestyle="dotted")
            axs[i, j].plot(time, noisy_measurements[idx], label="Noisy Measurements", color="red", alpha=0.6)
            axs[i, j].plot(time, estimated_states[:, idx], label="EKF Estimate", color="blue")
            axs[i, j].plot(time, predicted_states[:, idx], label="EKF Prediction", color="green", linestyle="dashed")
            axs[i, j].set_title(labels[i] + f" (Axis {['X', 'Y', 'Z'][j]})")
            axs[i, j].set_xlabel("Time (s)")
            axs[i, j].legend()
            axs[i, j].grid()
    plt.tight_layout()
    plt.show()