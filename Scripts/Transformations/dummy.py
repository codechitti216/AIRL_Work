import numpy as np

# Constants for Earth's dimensions (WGS84)
a = 6378137  # Semi-major axis (meters)
e2 = 0.00669437999014  # Eccentricity squared
omega_e = 7.292115e-5  # Earth's angular velocity in rad/s

def ecef_to_ned_velocity(lat_ref, lon_ref, alt_ref, ecef_vel, ecef_pos):
    """
    Convert velocity from ECEF to NED.
    """
    # Convert reference latitude and longitude to radians
    lat_ref_rad = np.radians(lat_ref)
    lon_ref_rad = np.radians(lon_ref)

    # Compute the prime vertical radius of curvature
    N = a / np.sqrt(1 - e2 * np.sin(lat_ref_rad) ** 2)

    # Reference ECEF coordinates
    x_ref = (N + alt_ref) * np.cos(lat_ref_rad) * np.cos(lon_ref_rad)
    y_ref = (N + alt_ref) * np.cos(lat_ref_rad) * np.sin(lon_ref_rad)
    z_ref = ((1 - e2) * N + alt_ref) * np.sin(lat_ref_rad)

    # Compute the rotation matrix from ECEF to NED
    rotation_matrix_ecef_to_ned = np.array([
        [-np.sin(lat_ref_rad) * np.cos(lon_ref_rad), -np.sin(lat_ref_rad) * np.sin(lon_ref_rad), np.cos(lat_ref_rad)],
        [-np.sin(lon_ref_rad), np.cos(lon_ref_rad), 0],
        [-np.cos(lat_ref_rad) * np.cos(lon_ref_rad), -np.cos(lat_ref_rad) * np.sin(lon_ref_rad), -np.sin(lat_ref_rad)]
    ])

    # Earth's rotational velocity at the reference point
    omega_ref = omega_e * np.array([-y_ref, x_ref, 0]) / np.linalg.norm([x_ref, y_ref])
    reference_velocity = np.cross(omega_ref, ecef_pos)

    # Convert velocity from ECEF to NED
    ned_vel = np.dot(rotation_matrix_ecef_to_ned, ecef_vel - reference_velocity)

    return ned_vel


def ned_to_ecef_velocity(lat_ref, lon_ref, alt_ref, ned_vel, ecef_pos):
    """
    Convert velocity from NED to ECEF.
    """
    # Convert latitude and longitude to radians
    lat_ref_rad = np.radians(lat_ref)
    lon_ref_rad = np.radians(lon_ref)

    # Compute the prime vertical radius of curvature
    N = a / np.sqrt(1 - e2 * np.sin(lat_ref_rad) ** 2)

    # Reference ECEF coordinates
    x_ref = (N + alt_ref) * np.cos(lat_ref_rad) * np.cos(lon_ref_rad)
    y_ref = (N + alt_ref) * np.cos(lat_ref_rad) * np.sin(lon_ref_rad)
    z_ref = ((1 - e2) * N + alt_ref) * np.sin(lat_ref_rad)

    # Compute the rotation matrix from NED to ECEF (transpose of ECEF to NED)
    rotation_matrix_ecef_to_ned = np.array([
        [-np.sin(lat_ref_rad) * np.cos(lon_ref_rad), -np.sin(lat_ref_rad) * np.sin(lon_ref_rad), np.cos(lat_ref_rad)],
        [-np.sin(lon_ref_rad), np.cos(lon_ref_rad), 0],
        [-np.cos(lat_ref_rad) * np.cos(lon_ref_rad), -np.cos(lat_ref_rad) * np.sin(lon_ref_rad), -np.sin(lat_ref_rad)]
    ])
    rotation_matrix_ned_to_ecef = rotation_matrix_ecef_to_ned.T

    # Earth's rotational velocity at the reference point
    omega_ref = omega_e * np.array([-y_ref, x_ref, 0]) / np.linalg.norm([x_ref, y_ref])
    reference_velocity = np.cross(omega_ref, ecef_pos)

    # Convert velocity from NED to ECEF
    ecef_vel = np.dot(rotation_matrix_ned_to_ecef, ned_vel) + reference_velocity

    return ecef_vel


# Example usage
lat_ref = 37.7749  # Latitude of the reference point (San Francisco)
lon_ref = -122.4194  # Longitude of the reference point (San Francisco)
alt_ref = 30  # Altitude of the reference point (30 meters)

# Example ECEF velocity input
ecef_vel = np.array([10, 5, -2])
ecef_pos = np.array([-2706187.55929115, -4261079.50629063, 3885743.86684897])

# Convert ECEF to NED velocity
ned_vel = ecef_to_ned_velocity(lat_ref, lon_ref, alt_ref, ecef_vel, ecef_pos)
print(f"NED velocity: {ned_vel}")

# Convert NED back to ECEF velocity
ecef_vel_back = ned_to_ecef_velocity(lat_ref, lon_ref, alt_ref, ned_vel, ecef_pos)
print(f"ECEF velocity (back): {ecef_vel_back}")
