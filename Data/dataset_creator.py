import os
import pandas as pd
import numpy as np


num_past_imu = 1
num_past_dvl = 5


data_folder = "../Data"


for traj_folder in os.listdir(data_folder):
    traj_path = os.path.join(data_folder, traj_folder)
    if not os.path.isdir(traj_path):
        continue

    
    dvl_data = pd.read_csv(os.path.join(traj_path, f"DVL_{traj_folder}.csv"))
    gt_data = pd.read_csv(os.path.join(traj_path, f"GT_{traj_folder}.csv"))
    imu_data = pd.read_csv(os.path.join(traj_path, f"IMU_{traj_folder}.csv"))

    
    dvl_data['Time [s]'] = dvl_data['Time [s]'].round(2)
    imu_data['Time [s]'] = imu_data['Time [s]'].round(2)
    gt_data['Time [s]'] = gt_data['Time [s]'].round(2)

    
    common_timestamps = set(dvl_data['Time [s]']).intersection(imu_data['Time [s]']).intersection(gt_data['Time [s]'])
    dvl_data = dvl_data[dvl_data['Time [s]'].isin(common_timestamps)]
    imu_data = imu_data[imu_data['Time [s]'].isin(common_timestamps)]
    gt_data = gt_data[gt_data['Time [s]'].isin(common_timestamps)]

    
    dvl_data = dvl_data.sort_values(by='Time [s]').reset_index(drop=True)
    imu_data = imu_data.sort_values(by='Time [s]').reset_index(drop=True)
    gt_data = gt_data.sort_values(by='Time [s]').reset_index(drop=True)

    
    assert np.array_equal(dvl_data['Time [s]'].values, imu_data['Time [s]'].values), "Timestamps do not align between DVL and IMU!"
    assert np.array_equal(dvl_data['Time [s]'].values, gt_data['Time [s]'].values), "Timestamps do not align between DVL and GT!"

    
    b_vectors = [
        [np.cos((45 + i * 90) * np.pi / 180) * np.sin(20 * np.pi / 180),
         np.sin((45 + i * 90) * np.pi / 180) * np.sin(20 * np.pi / 180),
         np.cos(20 * np.pi / 180)]
        for i in range(4)
    ]
    A = np.array(b_vectors).reshape((4, 3))

    
    V = dvl_data[['DVL X [m/s]', 'DVL Y [m/s]', 'DVL Z [m/s]']].to_numpy().T
    beams = np.matmul(A, V).T
    dvl_data[['b1', 'b2', 'b3', 'b4']] = beams

    
    samples = []
    for idx in range(max(num_past_imu, num_past_dvl), len(dvl_data)):
        time_value = dvl_data.iloc[idx]['Time [s]']

        
        imu_values = []
        for t_offset in range(num_past_imu):
            imu_values.extend(imu_data.iloc[idx - t_offset][[
                'ACC X [m/s^2]', 'ACC Y [m/s^2]', 'ACC Z [m/s^2]',
                'GYRO X [rad/s]', 'GYRO Y [rad/s]', 'GYRO Z [rad/s]'
            ]].values)

        
        dvl_values = []
        for t_offset in range(num_past_dvl):
            dvl_values.extend(dvl_data.iloc[idx - t_offset][['b1', 'b2', 'b3', 'b4']].values)

        
        velocity_values = gt_data.iloc[idx][['V North [m/s]', 'V East [m/s]', 'V Down [m/s]']].values

        
        sample = np.concatenate([[time_value], imu_values, dvl_values, velocity_values])
        samples.append(sample)

    
    columns = ['Time']
    for i in range(num_past_imu):
        columns.extend([f'ACC X_{i}', f'ACC Y_{i}', f'ACC Z_{i}', f'GYRO X_{i}', f'GYRO Y_{i}', f'GYRO Z_{i}'])
    for i in range(num_past_dvl):
        columns.extend([f'DVL{i}_1', f'DVL{i}_2', f'DVL{i}_3', f'DVL{i}_4'])
    columns.extend(['V North', 'V East', 'V Down'])

    
    new_dataset = pd.DataFrame(samples, columns=columns)
    output_filename = f"combined_{num_past_imu}_{num_past_dvl}.csv"
    new_dataset.to_csv(os.path.join(traj_path, output_filename), index=False)

    print(f"New dataset created: {traj_folder}/{output_filename}")
