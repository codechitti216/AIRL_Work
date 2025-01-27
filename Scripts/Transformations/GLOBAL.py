import numpy as np

# Constants
EARTH_RADIUS = 6378137.0  # Semi-major axis (meters)
ECCENTRICITY_SQUARED = 0.00669437999014  # Square of Earth's eccentricity
EARTH_ANGULAR_VELOCITY = np.array([0, 0, 7.292115e-5])  # Earth's angular velocity (rad/s)

lat = np.radians(90)  
lon = np.radians(60)  
alt = 500e3           

vx, vy, vz = 100, 20, 300  
ax, ay, az = 0, 0, -9.8     
wx, wy, wz = 0.01, 0.01, 0.01  
theta = np.radians(190)

# Rotation Matrices
def rotation_matrix(theta):
    return np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

def rotation_matrix_dot(theta):
    return np.array([
        [-np.sin(theta), np.cos(theta), 0],
        [-np.cos(theta), -np.sin(theta), 0],
        [0, 0, 0]
    ])

def rotation_matrix_dot_dot(theta):
    return np.array([
        [-np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), -np.cos(theta), 0],
        [0, 0, 0]
    ])

# Geodetic to ECEF (accounts for Earth's eccentricity)
def lla_to_xyz(lat, lon, alt):   
    a = EARTH_RADIUS  # Semi-major axis
    e2 = ECCENTRICITY_SQUARED  # Earth's eccentricity squared

    lat_rad = lat
    lon_rad = lon

    # Prime vertical radius of curvature
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad) ** 2)

    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - e2) + alt) * np.sin(lat_rad)
    return np.array([x, y, z])

# ECEF to Geodetic (accounts for Earth's eccentricity)
def xyz_to_lla(x, y, z):
    a = EARTH_RADIUS  # Semi-major axis
    e2 = ECCENTRICITY_SQUARED  # Earth's eccentricity squared
    b = np.sqrt(a**2 * (1 - e2))  # Semi-minor axis

    p = np.sqrt(x**2 + y**2)  # Distance from the Z-axis
    lon = np.arctan2(y, x)

    # Iterative calculation for latitude
    lat = np.arctan2(z, p * (1 - e2))
    for _ in range(5):  # Iterative refinement
        N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        lat = np.arctan2(z + e2 * N * np.sin(lat), p)

    # Altitude calculation
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N

    return lat, lon, alt

# Global transformations (ECEF <-> ECI)
def ecef_to_eci(lat, lon, alt, vx, vy, vz, ax, ay, az, wx, wy, wz, theta):
    X_ecef = lla_to_xyz(lat, lon, alt)   
    V_ecef = np.array([vx, vy, vz])
    A_ecef = np.array([ax, ay, az]) 
    W_ecef = np.array([wx, wy, wz])   
    R = rotation_matrix(-theta)
    X_eci = np.dot(R, X_ecef)
    print("Position ECEF:", X_ecef, "|||", "Position ECI:", X_eci)
    V_eci = np.dot(R, V_ecef) + np.cross(EARTH_ANGULAR_VELOCITY, np.dot(R, X_eci))
    print("Velocity ECEF:", V_ecef, "|||", "Velocity ECI:", V_eci)
    W_eci = np.dot(R, W_ecef) + EARTH_ANGULAR_VELOCITY
    print("Angular Velocity ECEF:", W_ecef, "|||", "Angular Velocity ECI:", W_eci)
    A_eci = np.dot(R, A_ecef) - 2 * np.dot(R, np.dot(rotation_matrix_dot(theta), V_eci)) - np.dot(R, np.dot(rotation_matrix_dot_dot(theta), X_eci))
    print("Acceleration ECEF:", A_ecef, "|||", "Acceleration ECI:", A_eci)
    lat_eci, lon_eci, alt_eci = xyz_to_lla(X_eci[0], X_eci[1], X_eci[2])
    return [lat_eci, lon_eci, alt_eci, V_eci[0], V_eci[1], V_eci[2], A_eci[0], A_eci[1], A_eci[2], W_eci[0], W_eci[1], W_eci[2], theta]

def eci_to_ecef(lat, lon, alt, vx, vy, vz, ax, ay, az, wx, wy, wz, theta):
    X_eci = lla_to_xyz(lat, lon, alt)   
    V_eci = np.array([vx, vy, vz])
    A_eci = np.array([ax, ay, az]) 
    W_eci = np.array([wx, wy, wz])   
    R = rotation_matrix(theta)
    X_ecef = np.dot(R, X_eci)
    print("Position ECI:", X_eci, "|||", "Position ECEF:", X_ecef)
    V_ecef = np.dot(R, V_eci) - np.cross(EARTH_ANGULAR_VELOCITY, np.dot(R, X_eci))
    print("Velocity ECI:", V_eci, "|||", "Velocity ECEF:", V_ecef)
    W_ecef = np.dot(R, W_eci) - EARTH_ANGULAR_VELOCITY
    print("Angular Velocity ECI:", W_eci, "|||", "Angular Velocity ECEF:", W_ecef)
    A_ecef = np.dot(R, A_eci) + 2 * np.dot(rotation_matrix_dot(theta), V_eci) + np.dot(rotation_matrix_dot_dot(theta), X_eci)
    print("Acceleration ECI:", A_eci, "|||", "Acceleration ECEF:", A_ecef)
    lat_ecef, lon_ecef, alt_ecef = xyz_to_lla(X_ecef[0], X_ecef[1], X_ecef[2])
    return [lat_ecef, lon_ecef, alt_ecef, V_ecef[0], V_ecef[1], V_ecef[2], A_ecef[0], A_ecef[1], A_ecef[2], W_ecef[0], W_ecef[1], W_ecef[2], theta]

print("ECI TO ECEF")
print("_" * 50)
result = eci_to_ecef(lat, lon, alt, vx, vy, vz, ax, ay, az, wx, wy, wz, theta)
print("_" * 50)
print("ECEF TO ECI")
result = ecef_to_eci(result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7], result[8], result[9], result[10], result[11], result[12])
