import pandas as pd


dvl_df = pd.read_csv("../Data/Trajectory1/DVL_trajectory1.csv")
gt_df = pd.read_csv("../Data/Trajectory1/GT_trajectory1.csv")
imu_df = pd.read_csv("../Data/Trajectory1/IMU_trajectory1.csv")


dvl_df["Time [s]"] = dvl_df["Time [s]"].round(2)
gt_df["Time [s]"] = gt_df["Time [s]"].round(2)
imu_df["Time [s]"] = imu_df["Time [s]"].round(2)


merged_df = pd.merge(gt_df, dvl_df, on="Time [s]", how="inner")
merged_df = pd.merge(merged_df, imu_df, on="Time [s]", how="inner")


merged_df.to_csv("../Data/Trajectory1/combined.csv", index=False)
