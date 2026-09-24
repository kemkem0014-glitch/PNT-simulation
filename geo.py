import numpy as np

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    s = np.sin(lat)
    c = np.cos(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * s * s)

    x = (n + alt_m) * c * np.cos(lon)
    y = (n + alt_m) * c * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + alt_m) * s
    return np.array([x, y, z], dtype=float)


def elevation_deg(receiver_ecef: np.ndarray, satellite_ecef: np.ndarray) -> float:
    los = satellite_ecef - receiver_ecef
    los_norm = np.linalg.norm(los)
    if los_norm == 0.0:
        return -90.0

    up = receiver_ecef / np.linalg.norm(receiver_ecef)
    sin_el = np.clip(np.dot(los / los_norm, up), -1.0, 1.0)
    return float(np.rad2deg(np.arcsin(sin_el)))
