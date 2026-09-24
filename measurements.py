import numpy as np

C_MPS = 299_792_458.0


def predict_doppler_hz(
    receiver_position: np.ndarray,
    receiver_velocity: np.ndarray,
    clock_drift_sps: float,
    satellite_position: np.ndarray,
    satellite_velocity: np.ndarray,
    carrier_hz: float,
) -> float:
    delta = satellite_position - receiver_position
    rho = np.linalg.norm(delta)
    u = delta / rho

    range_rate = float(np.dot(u, satellite_velocity - receiver_velocity))

    # Sign convention:
    # positive geometric range-rate means satellite and receiver separate,
    # therefore the received carrier shifts downward.
    return -(carrier_hz / C_MPS) * range_rate + carrier_hz * clock_drift_sps


def simulate_doppler_hz(
    rng: np.random.Generator,
    receiver_position: np.ndarray,
    receiver_velocity: np.ndarray,
    clock_drift_sps: float,
    satellite_position: np.ndarray,
    satellite_velocity: np.ndarray,
    carrier_hz: float,
    sigma_hz: float,
) -> float:
    ideal = predict_doppler_hz(
        receiver_position,
        receiver_velocity,
        clock_drift_sps,
        satellite_position,
        satellite_velocity,
        carrier_hz,
    )
    return float(ideal + rng.normal(0.0, sigma_hz))


def simulate_timing_s(
    rng: np.random.Generator,
    clock_bias_s: float,
    sigma_s: float,
) -> float:
    return float(clock_bias_s + rng.normal(0.0, sigma_s))
