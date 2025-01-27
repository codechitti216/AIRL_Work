import numpy as np

EARTH_RADIUS = 6378137.0
EARTH_ANGULAR_VELOCITY = np.array([0, 0, 7.292115e-5])  
def ned_to_enu(ned_vector):
    transform_matrix = np.array([
        [0, 1, 0],  
        [1, 0, 0],  
        [0, 0, -1]  
    ])
    return transform_matrix @ ned_vector
def ned_to_seu(ned_vector):
    transform_matrix = np.array([
        [0, 1, 0],  
        [-1, 0, 0], 
        [0, 0, -1]  
    ])
    return transform_matrix @ ned_vector
def rotate_vector(vector, theta):
    rotation_matrix = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    return rotation_matrix @ vector
def eci_to_ecef(vector, theta):
    return rotate_vector(vector, -theta)
def ecef_to_eci(vector, theta):
    return rotate_vector(vector, theta)
def ned_to_ecef(lat, lon, alt, ned_vector):
    R_ned_to_ecef = np.array([
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
        [-np.sin(lon), np.cos(lon), 0],
        [-np.cos(lat) * np.cos(lon), -np.cos(lat) * np.sin(lon), -np.sin(lat)]
    ])
    return R_ned_to_ecef.T @ ned_vector
def ecef_to_ned(lat, lon, alt, ecef_vector):
    R_ecef_to_ned = np.array([
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
        [-np.sin(lon), np.cos(lon), 0],
        [-np.cos(lat) * np.cos(lon), -np.cos(lat) * np.sin(lon), -np.sin(lat)]
    ])
    return R_ecef_to_ned @ ecef_vector
def enu_to_ned(enu_vector):
    transform_matrix = np.array([
        [0, 1, 0],  
        [1, 0, 0],  
        [0, 0, -1]  
    ])
    return transform_matrix @ enu_vector
def seu_to_ned(seu_vector):
    transform_matrix = np.array([
        [0, 1, 0],  
        [-1, 0, 0], 
        [0, 0, -1]  
    ])
    return transform_matrix @ seu_vector
def convert_frame(lat, lon, alt, vx, vy, vz, ax, ay, az, wx, wy, wz, theta, input_frame, output_frame):
    velocity = np.array([vx, vy, vz])
    acceleration = np.array([ax, ay, az])
    angular_velocity = np.array([wx, wy, wz])
    if input_frame == output_frame:
        return velocity, acceleration, angular_velocity, theta
    if input_frame == "ECI" and output_frame == "ECEF":
        velocity = eci_to_ecef(velocity, theta)
        acceleration = eci_to_ecef(acceleration, theta)
        angular_velocity = eci_to_ecef(angular_velocity, theta)
    elif input_frame == "ECEF" and output_frame == "ECI":
        velocity = ecef_to_eci(velocity, theta)
        acceleration = ecef_to_eci(acceleration, theta)
        angular_velocity = ecef_to_eci(angular_velocity, theta)
    elif input_frame == "NED" and output_frame == "ECEF":
        velocity = ned_to_ecef(lat, lon, alt, velocity)
        acceleration = ned_to_ecef(lat, lon, alt, acceleration)
        angular_velocity = ned_to_ecef(lat, lon, alt, angular_velocity)
    elif input_frame == "ECEF" and output_frame == "NED":
        velocity = ecef_to_ned(lat, lon, alt, velocity)
        acceleration = ecef_to_ned(lat, lon, alt, acceleration)
        angular_velocity = ecef_to_ned(lat, lon, alt, angular_velocity)
    elif input_frame == "NED" and output_frame == "ENU":
        velocity = ned_to_enu(velocity)
        acceleration = ned_to_enu(acceleration)
        angular_velocity = ned_to_enu(angular_velocity)
    elif input_frame == "ENU" and output_frame == "NED":
        velocity = enu_to_ned(velocity)
        acceleration = enu_to_ned(acceleration)
        angular_velocity = enu_to_ned(angular_velocity)
    elif input_frame == "NED" and output_frame == "SEU":
        velocity = ned_to_seu(velocity)
        acceleration = ned_to_seu(acceleration)
        angular_velocity = ned_to_seu(angular_velocity)
    elif input_frame == "SEU" and output_frame == "NED":
        velocity = seu_to_ned(velocity)
        acceleration = seu_to_ned(acceleration)
        angular_velocity = seu_to_ned(angular_velocity)
    else:
        raise ValueError(f"Unsupported conversion: {input_frame} to {output_frame}")
    return velocity, acceleration, angular_velocity, theta
if __name__ == "__main__":
    lat, lon, alt = np.radians(37.7749), np.radians(-122.4194), 30  
    vx, vy, vz = 100, 200, -50
    ax, ay, az = 0, 0, 9.8
    wx, wy, wz = 0.01, 0.01, 0.01
    theta = np.radians(45)    
    velocity, acceleration, angular_velocity, theta = convert_frame(
        lat, lon, alt, vx, vy, vz, ax, ay, az, wx, wy, wz, theta, "NED", "ENU"
    )
    print("Converted to ENU:")
    print("Velocity:", velocity)
    print("Acceleration:", acceleration)
    print("Angular Velocity:", angular_velocity)
    velocity, acceleration, angular_velocity, theta = convert_frame(
        lat, lon, alt, vx, vy, vz, ax, ay, az, wx, wy, wz, theta, "ENU", "NED"
    )
    print("\nConverted to ECI:")
    print("Velocity:", velocity)
    print("Acceleration:", acceleration)
    print("Angular Velocity:", angular_velocity)