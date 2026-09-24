from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Simulation
    dt_s: float = 1.0
    duration_s: float = 1200.0
    random_seed: int = 42

    # Receiver truth location: Tokyo-area example
    receiver_lat_deg: float = 35.6762
    receiver_lon_deg: float = 139.6503
    receiver_alt_m: float = 40.0

    # Synthetic LEO constellation.
    # 216 satellites is used in Ver.1 so that the simplified geometry normally
    # leaves several satellites above the elevation mask. This is not intended
    # to reproduce a specific operational constellation.
    leo_satellites: int = 216
    leo_altitude_m: float = 550_000.0
    leo_inclination_deg: float = 53.0
    elevation_mask_deg: float = 10.0

    # Approximate communications carrier used for Doppler simulation
    carrier_hz: float = 12.0e9
    doppler_sigma_hz: float = 3.0

    # Iridium-like timing observation model.
    # This is a research parameter, not a claim of service specification.
    iridium_time_sigma_ns: float = 50.0
    iridium_update_interval_s: float = 1.0

    # EKF initial errors immediately after GNSS loss
    initial_position_sigma_m: float = 20.0
    initial_velocity_sigma_mps: float = 0.2
    initial_clock_bias_ns: float = 200.0
    initial_clock_drift_ppb: float = 5.0

    # EKF process-noise tuning
    acceleration_process_sigma_mps2: float = 0.02
    clock_bias_rw_sigma_ns_sqrt_s: float = 5.0
    clock_drift_rw_sigma_ppb_sqrt_s: float = 0.05


CFG = Config()
