import numpy as np


def generate_random_geodetic():
    lat = np.random.uniform(-90, 90)  
    lon = np.random.uniform(-180, 180)  
    alt = np.random.uniform(0, 10000)  
    print("TRUE ASSUMPTION")
    print(lat,lon,alt)
    print("-"*30)
    
    return lat, lon, alt


def geodetic_to_ecef(lat, lon, alt):
    a = 6378137.0  
    e2 = 0.00669437999014  

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    N = a / np.sqrt(1 - e2 * np.sin(lat_rad) ** 2)
    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - e2) + alt) * np.sin(lat_rad)
    return np.array([x, y, z])


def ecef_to_geodetic(x, y, z):
    a = 6378137.0  
    e2 = 0.00669437999014  
    b = np.sqrt(a**2 * (1 - e2))  
    ep = np.sqrt((a**2 - b**2) / b**2)
    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * a, p * b)

    lon = np.arctan2(y, x)
    lat = np.arctan2(z + ep**2 * b * np.sin(theta)**3, p - e2 * a * np.cos(theta)**3)
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    alt = p / np.cos(lat) - N

    lat = np.degrees(lat)
    lon = np.degrees(lon)
    return lat, lon, alt


def ecef_to_ned(ecef, reference_ecef, lat_ref, lon_ref):
    lat_ref_rad = np.radians(lat_ref)
    lon_ref_rad = np.radians(lon_ref)

    R_ecef_to_ned = np.array([
        [-np.sin(lat_ref_rad) * np.cos(lon_ref_rad), -np.sin(lat_ref_rad) * np.sin(lon_ref_rad),  np.cos(lat_ref_rad)],
        [-np.sin(lon_ref_rad),                       np.cos(lon_ref_rad),                        0],
        [-np.cos(lat_ref_rad) * np.cos(lon_ref_rad), -np.cos(lat_ref_rad) * np.sin(lon_ref_rad), -np.sin(lat_ref_rad)]
    ])

    relative_ecef = ecef - reference_ecef
    ned = R_ecef_to_ned @ relative_ecef
    return ned, R_ecef_to_ned


def ned_to_ecef(ned, reference_ecef, lat_ref, lon_ref):
    lat_ref_rad = np.radians(lat_ref)
    lon_ref_rad = np.radians(lon_ref)

    R_ned_to_ecef = np.array([
        [-np.sin(lat_ref_rad) * np.cos(lon_ref_rad), -np.sin(lon_ref_rad), -np.cos(lat_ref_rad) * np.cos(lon_ref_rad)],
        [-np.sin(lat_ref_rad) * np.sin(lon_ref_rad),  np.cos(lon_ref_rad), -np.cos(lat_ref_rad) * np.sin(lon_ref_rad)],
        [ np.cos(lat_ref_rad),                       0,                   -np.sin(lat_ref_rad)]
    ])

    ecef = R_ned_to_ecef @ ned + reference_ecef
    return ecef

def main():
    
    lat, lon, alt = generate_random_geodetic()
    print(f"Random Geodetic Position: Latitude={lat:.6f}, Longitude={lon:.6f}, Altitude={alt:.2f} meters")

    
    ecef_position = geodetic_to_ecef(lat, lon, alt)
    print(f"ECEF Position: {ecef_position}")

    
    lat_ref, lon_ref, alt_ref = 0.0, 0.0, 0.0
    reference_ecef = geodetic_to_ecef(lat_ref, lon_ref, alt_ref)

    
    ned_position, R_ecef_to_ned = ecef_to_ned(ecef_position, reference_ecef, lat_ref, lon_ref)
    print(f"NED Position: {ned_position}")

    
    ecef_position_back = ned_to_ecef(ned_position, reference_ecef, lat_ref, lon_ref)
    print(f"Back to ECEF Position: {ecef_position_back}")

    
    lat_back, lon_back, alt_back = ecef_to_geodetic(*ecef_position_back)
    print(f"Back to Geodetic Position: Latitude={lat_back:.6f}, Longitude={lon_back:.6f}, Altitude={alt_back:.2f} meters")

    
    position_diff = np.linalg.norm(ecef_position - ecef_position_back)
    lat_diff = abs(lat - lat_back)
    lon_diff = abs(lon - lon_back)
    alt_diff = abs(alt - alt_back)

    print(f"Position Difference (ECEF): {position_diff:.6f}")
    print(f"Latitude Difference: {lat_diff:.6f}")
    print(f"Longitude Difference: {lon_diff:.6f}")
    print(f"Altitude Difference: {alt_diff:.6f}")

    
    if lat_diff < 1e-6 and lon_diff < 1e-6 and alt_diff < 1e-3:
        print("Condition satisfied: Geodetic consistency verified.")
    else:
        print("Condition not satisfied: Differences found in geodetic coordinates.")


if __name__ == "__main__":
    main()
