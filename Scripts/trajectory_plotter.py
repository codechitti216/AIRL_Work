import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

df = pd.read_csv('../Data/Trajectory1/GT_trajectory1.csv')
longitude = df['Longitude [rad]'].values  
latitude = df['Latitude [rad]'].values    
altitude = df['Altitude [m]'].values       
x = (altitude) * np.cos(latitude) * np.cos(longitude)
y = (altitude) * np.cos(latitude) * np.sin(longitude)
z = (altitude) * np.sin(latitude)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
ax.plot(x, y, z, marker='o', color='b', linestyle='-', markersize=5)
ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')
ax.set_zlabel('Z (meters)')
ax.set_title('3D Trajectory on Earth')
plt.show()
