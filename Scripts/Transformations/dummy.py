import numpy as np

def ned_to_enu(ned_vector):
    """Convert a vector from NED to ENU coordinate frame."""
    transform_matrix = np.array([
        [0, 1, 0],  
        [1, 0, 0],  
        [0, 0, -1]  
    ])
    return transform_matrix @ ned_vector

def ned_to_seu(ned_vector):
    """Convert a vector from NED to SEU coordinate frame."""
    transform_matrix = np.array([
        [0, 1, 0],  
        [-1, 0, 0], 
        [0, 0, -1]  
    ])
    return transform_matrix @ ned_vector

def ned_to_body(ned_vector, rotation_matrix):
    """Convert a vector from NED to Body Frame using a rotation matrix."""
    return rotation_matrix @ ned_vector

def update_lat_lon_alt(lat, lon, alt, ned_displacement):
    """Update latitude, longitude, and altitude from NED displacement."""
    R_EARTH = 6378137.0
    north, east, down = ned_displacement
    up = -down

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    delta_lat = north / R_EARTH
    delta_lon = east / (R_EARTH * np.cos(lat_rad))

    lat_new = lat + np.degrees(delta_lat)
    lon_new = lon + np.degrees(delta_lon)
    alt_new = alt + up

    return lat_new, lon_new, alt_new

def rotate_vector(vector, theta):
    """Rotate a vector using the angle theta between ECI and ECEF frames."""
    rotation_matrix = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    return rotation_matrix @ vector


def run_tests():
    """Run validation tests."""
    
    ned_vector = np.array([100, 200, -50])
    expected_enu = np.array([200, 100, 50])
    assert np.allclose(ned_to_enu(ned_vector), expected_enu), "NED to ENU failed"

    
    expected_seu = np.array([200, -100, 50])
    assert np.allclose(ned_to_seu(ned_vector), expected_seu), "NED to SEU failed"

    
    lat, lon, alt = 37.7749, -122.4194, 30
    expected_lat, expected_lon, expected_alt = 37.775798, -122.417127, 80
    new_lat, new_lon, new_alt = update_lat_lon_alt(lat, lon, alt, ned_vector)
    assert np.isclose(new_lat, expected_lat, atol=1e-6), "Latitude update failed"
    assert np.isclose(new_lon, expected_lon, atol=1e-6), "Longitude update failed"
    assert np.isclose(new_alt, expected_alt), "Altitude update failed"

    
    theta = np.radians(45)
    rotated = rotate_vector(ned_vector, theta)
    expected_rotated = np.array([212.132034, 70.710678, -50])
    assert np.allclose(rotated, expected_rotated, atol=1e-6), "Vector rotation failed"

    print("All tests passed!")

if __name__ == "__main__":
    
    ned_vector = np.array([100, 200, -50])

    enu_vector = ned_to_enu(ned_vector)
    print("ENU Vector:", enu_vector)

    seu_vector = ned_to_seu(ned_vector)
    print("SEU Vector:", seu_vector)

    rotation_matrix = np.eye(3)
    body_vector = ned_to_body(ned_vector, rotation_matrix)
    print("Body Frame Vector:", body_vector)

    lat, lon, alt = 37.7749, -122.4194, 30
    new_lat, new_lon, new_alt = update_lat_lon_alt(lat, lon, alt, ned_vector)
    print("Updated Lat, Lon, Alt:", new_lat, new_lon, new_alt)

    theta = np.radians(45)
    rotated_vector = rotate_vector(ned_vector, theta)
    print("Rotated Vector:", rotated_vector)

    
    run_tests()