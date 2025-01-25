import numpy as np

EARTH_ANGULAR_VELOCITY = np.array([0, 0, 7.292115e-5])  
EARTH_RADIUS = 6378137  

def skew_symmetric_matrix(omega):
    return np.array([
        [0, -omega[2], omega[1]],
        [omega[2], 0, -omega[0]],
        [-omega[1], omega[0], 0]
    ])

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
    
    
def lla_to_xyz(lat, lon, alt):   
    R = EARTH_RADIUS + alt
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    return np.array([x, y, z])

def xyz_to_lla(x, y, z):
    lon = np.arctan2(y, x)
    r_xy = np.sqrt(x**2 + y**2)  
    lat = np.arctan2(z, r_xy)
    r = np.sqrt(x**2 + y**2 + z**2)  
    alt = r - EARTH_RADIUS  
    return lat, lon, alt

def eci_to_ecef(lat, lon, alt, vx, vy, vz, ax, ay, az, wx, wy, wz, theta):
    X_eci = lla_to_xyz(lat, lon, alt)   
    V_eci = np.array([vx, vy, vz])
    A_eci = np.array([ax, ay, az]) 
    W_eci = np.array([wx, wy, wz])   
    R = rotation_matrix(theta)
    X_ecef = np.dot(R, X_eci)
    print("position ECI: ",X_eci, "|||","position : ECEF",X_ecef)
    V_ecef = np.dot(R, V_eci) - np.cross(EARTH_ANGULAR_VELOCITY, np.dot(R, X_eci))
    print("velocity ECI: ",V_eci, "|||","position : ECEF",V_ecef)
    W_ecef = np.dot(R, W_eci) - EARTH_ANGULAR_VELOCITY
    print("Angular velocity ECI: ",W_eci, "|||","position : ECEF",W_ecef)
    A_ecef = np.dot(R, A_eci) + 2 * np.dot(rotation_matrix_dot(theta),V_eci) + np.dot(rotation_matrix_dot_dot(theta),X_eci)
    print("Acceleration ECI: ",A_eci, "|||","position : ECEF",A_ecef)
    lat_ecef, lon_ecef, alt_ecef = xyz_to_lla(X_ecef[0], X_ecef[1], X_ecef[2])
    return [lat_ecef, lon_ecef, alt_ecef,V_ecef[0], V_ecef[1], V_ecef[2],A_ecef[0], A_ecef[1],A_ecef[2],W_ecef[0],W_ecef[1],W_ecef[2],theta]
    
def ecef_to_eci(lat, lon, alt, vx, vy, vz, ax, ay, az, wx, wy, wz, theta):
    X_ecef = lla_to_xyz(lat, lon, alt)   
    V_ecef = np.array([vx, vy, vz])
    A_ecef = np.array([ax, ay, az]) 
    W_ecef = np.array([wx, wy, wz])   
    R = rotation_matrix(-theta)
    X_eci = np.dot(R, X_ecef)
    print("position : ECEF",X_ecef, "|||","position ECI: ",X_eci)
    V_eci = np.dot(R, V_ecef) + np.cross(EARTH_ANGULAR_VELOCITY, np.dot(R, X_eci))
    print("velocity ECEF: ",V_ecef, "|||","velocity ECI: ",V_eci)
    W_eci = np.dot(R, W_ecef) + EARTH_ANGULAR_VELOCITY
    print("Angular velocity ECEF: ",W_ecef, "|||","Angular velocity ECI: ",W_eci)
    A_eci = np.dot(R, A_ecef) - 2*np.dot(R,np.dot(rotation_matrix_dot(theta),V_eci)) - np.dot(R,np.dot(rotation_matrix_dot_dot(theta),X_eci))
    print("Acceleration ECEF: ",A_ecef, "|||","position ECI: ",A_eci)
    lat_eci, lon_eci, alt_eci = xyz_to_lla(X_eci[0], X_eci[1], X_eci[2])
    return [lat_eci, lon_eci, alt_eci,V_eci[0], V_eci[1], V_eci[2],A_eci[0], A_eci[1],A_eci[2],W_eci[0],W_eci[1],W_eci[2],theta]
    
    
def ned_ecef_transformation(lat,lon):
    return np.array([
    [-np.sin(lat)*np.cos(lon), -np.sin(lon), -np.cos(lat)*np.cos(lon)],
    [-np.sin(lat)*np.sin(lon), np.cos(lon), -np.cos(lat)*np.sin(lon)],
    [np.cos(lat), 0, -np.sin(lat)]
])
    
def ecef_to_ned(lat, lon, alt, vx, vy, vz, ax, ay, az, wx, wy, wz, theta):
    X_ecef = lla_to_xyz(lat, lon, alt)
    V_ecef = np.array([vx, vy, vz])
    A_ecef = np.array([ax, ay, az])
    W_ecef = np.array([wx, wy, wz])
    R_ecef_to_ned = ned_ecef_transformation(lat,lon)   
    X_ned = np.dot(R_ecef_to_ned, X_ecef)
    print("Position (ECEF):", X_ecef, "||| Position (NED):", X_ned)
    V_ned = np.dot(R_ecef_to_ned, V_ecef)
    print("Velocity (ECEF):", V_ecef, "||| Velocity (NED):", V_ned)
    W_ned = np.dot(R_ecef_to_ned, W_ecef)
    print("Angular Velocity (ECEF):", W_ecef, "||| Angular Velocity (NED):", W_ned)
    A_ned = np.dot(R_ecef_to_ned, A_ecef)
    print("Acceleration (ECEF):", A_ecef, "||| Acceleration (NED):", A_ned)
    lat_ned, lon_ned, alt_ned = xyz_to_lla(X_ned[0], X_ned[1], X_ned[2])
    return [lat_ned, lon_ned, alt_ned,V_ned[0], V_ned[1], V_ned[2],A_ned[0], A_ned[1],A_ned[2],W_ned[0],W_ned[1],W_ned[2],theta]


def ned_to_ecef(lat, lon, alt, vx, vy, vz, ax, ay, az, wx, wy, wz, theta):
    X_ned = lla_to_xyz(lat, lon, alt)
    V_ned = np.array([vx, vy, vz])  
    A_ned = np.array([ax, ay, az])  
    W_ned = np.array([wx, wy, wz])  
    R_ned_to_ecef = ned_ecef_transformation(lat,lon).T
    X_ecef = np.dot(R_ned_to_ecef, X_ned)
    print("position NED:", X_ned, "||| position ECEF:", X_ecef)

    
    V_ecef = np.dot(R_ned_to_ecef, V_ned)
    print("velocity NED:", V_ned, "||| velocity ECEF:", V_ecef)

    
    W_ecef = np.dot(R_ned_to_ecef, W_ned)
    print("Angular velocity NED:", W_ned, "||| Angular velocity ECEF:", W_ecef)

    
    A_ecef = np.dot(R_ned_to_ecef, A_ned)
    print("Acceleration NED:", A_ned, "||| Acceleration ECEF:", A_ecef)

    
    lat_ecef, lon_ecef, alt_ecef = xyz_to_lla(X_ecef[0], X_ecef[1], X_ecef[2])
    return [
        lat_ecef, lon_ecef, alt_ecef,
        V_ecef[0], V_ecef[1], V_ecef[2],
        A_ecef[0], A_ecef[1], A_ecef[2],
        W_ecef[0], W_ecef[1], W_ecef[2],
        theta
    ]


    


if __name__ == "__main__":
    lat = np.radians(90)  
    lon = np.radians(60)  
    alt = 500e3           
    
    vx, vy, vz = 100, 200, 300  
    ax, ay, az = 0, 0, -9.8     
    wx, wy, wz = 0.01, 0.01, 0.01  
    theta = np.radians(190)
    
    print("ECI TO ECEF")
    print("_"*50)
    result = eci_to_ecef(lat, lon, alt, vx, vy, vz, ax, ay, az, wx, wy, wz, theta)
    print("_"*50)
    print("ECEF TO ECI")
    result = ecef_to_eci(result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7],result[8],result[9],result[10],result[11],result[12])
    
    print("ECEF TO NED")
    result = ecef_to_ned(lat, lon, alt, vx, vy, vz, ax, ay, az, wx, wy, wz, theta)
    print("_"*50)
    print("NED TO ECEF")
    result = ned_to_ecef(result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7],result[8],result[9],result[10],result[11],result[12])


    
    # print(np.matmul(ned_ecef_transormation(lat,lon),ned_ecef_transormation(lat,lon).T))