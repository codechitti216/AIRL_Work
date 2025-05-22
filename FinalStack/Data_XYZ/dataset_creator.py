import os
import pandas as pd
import numpy as np

data_folder = "../Data_XYZ"

for traj_folder in os.listdir(data_folder):
    traj_path = os.path.join(data_folder, traj_folder)
    if not os.path.isdir(traj_path):
        continue

    dvl_data = pd.read_csv(os.path.join(traj_path, f"DVL_t{traj_folder[1:]}.csv"))
    gt_data = pd.read_csv(os.path.join(traj_path, f"GT_t{traj_folder[1:]}.csv"))
    imu_data = pd.read_csv(os.path.join(traj_path, f"IMU_t{traj_folder[1:]}.csv"))
    
    dvl_data['Time [s]'] = dvl_data['Time [s]'].round(2)
    imu_data['Time [s]'] = imu_data['Time [s]'].round(2)
    gt_data['Time [s]'] = gt_data['Time [s]'].round(2)

    # Only keep common timestamps
    common_timestamps = set(dvl_data['Time [s]']).intersection(imu_data['Time [s]']).intersection(gt_data['Time [s]'])
    dvl_data = dvl_data[dvl_data['Time [s]'].isin(common_timestamps)]
    gt_data = gt_data[gt_data['Time [s]'].isin(common_timestamps)]

    dvl_data = dvl_data.sort_values(by='Time [s]').reset_index(drop=True)
    gt_data = gt_data.sort_values(by='Time [s]').reset_index(drop=True)

    assert np.array_equal(dvl_data['Time [s]'].values, gt_data['Time [s]'].values), "Timestamps do not align!"

    b_vectors = [
        [np.cos((45 + i * 90) * np.pi / 180) * np.sin(20 * np.pi / 180),
         np.sin((45 + i * 90) * np.pi / 180) * np.sin(20 * np.pi / 180),
         np.cos(20 * np.pi / 180)]
        for i in range(4)
    ]
    A = np.array(b_vectors).reshape((4, 3))
    V = dvl_data[['DVL X [m/s]', 'DVL Y [m/s]', 'DVL Z [m/s]']].to_numpy().T
    beams = np.matmul(A, V).T

    beams_df = pd.DataFrame(beams, columns=['b1', 'b2', 'b3', 'b4'])
    beams_df.insert(0, 'Time', dvl_data['Time [s]'].values)
    beams_df.to_csv(os.path.join(traj_path, 'beams_gt.csv'), index=False)

    # Save velocity_gt.csv (ground truth velocities)
    velocity_df = gt_data[['Time [s]', 'V North [m/s]', 'V East [m/s]', 'V Down [m/s]']].copy()
    velocity_df.columns = ['Time', 'V North', 'V East', 'V Down']
    velocity_df.to_csv(os.path.join(traj_path, 'velocity_gt.csv'), index=False)

    print(f"Ground truth files created in {traj_folder}")
