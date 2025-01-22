def julian_date(year, month, day, hour, minute, second):
    JD = 367*year - int((7*(year + int((month + 9) / 12))) / 4) + int((275*month) / 9) + day + 1721013.5 + (hour + minute / 60 + second / 3600) / 24
    return JD
def gst_to_theta(JD):
    t_J = (JD - 2451545.0) / 36525    
    GMST = 280.46061837 + 360.98564736629 * t_J  
    GST_degrees = GMST % 360     
    theta = GST_degrees * (3.14159265358979 / 180)  
    return theta
def compute_theta(year, month, day, hour, minute, second):  
    JD = julian_date(year, month, day, hour, minute, second)    
    theta = gst_to_theta(JD)
    return theta
year, month, day, hour, minute, second = 2025, 1, 21, 12, 0, 0
theta = compute_theta(year, month, day, hour, minute, second)
print(f"Theta at {year}-{month}-{day} {hour}:{minute}:{second} UTC: {theta} radians")
import numpy as np

def lat_lon_to_eci(lat_rad, lon_rad, alt):
    
    a = 6378137.0  
    e = 0.081819190842622  

    N = a / np.sqrt(1 - e**2 * np.sin(lat_rad)**2)
    X_eci = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    Y_eci = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    Z_eci = ((1 - e**2) * N + alt) * np.sin(lat_rad)
    return np.array([X_eci, Y_eci, Z_eci])
def eci_to_ecef(X_eci, Y_eci, Z_eci, theta):
    R = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    ecef_coords = R.dot(np.array([X_eci, Y_eci, Z_eci]))
    return ecef_coords
def compute_theta(year, month, day, hour, minute, second):  
    JD = julian_date(year, month, day, hour, minute, second)    
    theta = gst_to_theta(JD)
    return theta
lat_rad = 37.7749 * (np.pi / 180)  
lon_rad = -122.4194 * (np.pi / 180)  
alt = 10  
year, month, day, hour, minute, second = 2025, 1, 21, 12, 0, 0
theta = compute_theta(year, month, day, hour, minute, second)
X_eci, Y_eci, Z_eci = lat_lon_to_eci(lat_rad, lon_rad, alt)
ecef_coords = eci_to_ecef(X_eci, Y_eci, Z_eci, theta)
print(f"ECI coordinates: {X_eci}, {Y_eci}, {Z_eci}")
print(f"ECEF coordinates: {ecef_coords[0]}, {ecef_coords[1]}, {ecef_coords[2]}")
