from dataclasses import dataclass

import numpy as np

from config import Config
from ekf import HybridPntEkf
from geo import elevation_deg, geodetic_to_ecef
from measurements import simulate_doppler_hz, simulate_timing_s
from orbit import SyntheticLEOConstellation


@dataclass(frozen=True)
class Scenario:
    name: str
    use_timing: bool
    use_doppler: bool


SCENARIOS = (
    Scenario("A_holdover", False, False),
    Scenario("B_iridium_time", True, False),
    Scenario("C_leo_doppler", False, True),
    Scenario("D_hybrid", True, True),
)


def _rngs(cfg: Config):
    """
    Use independent random streams so every scenario shares the same:
    - initial state error
    - receiver clock truth
    - timing measurement noise
    - Doppler measurement noise

    This makes A/B/C/D a paired comparison instead of four unrelated trials.
    """
    seed = cfg.random_seed
    return {
        "init": np.random.default_rng(seed + 1),
        "truth": np.random.default_rng(seed + 2),
        "timing": np.random.default_rng(seed + 3),
        "doppler": np.random.default_rng(seed + 4),
    }


def _build_filter(
    cfg: Config,
    truth_position: np.ndarray,
    rng_init: np.random.Generator,
):
    x0 = np.zeros(8)
    x0[0:3] = truth_position + rng_init.normal(
        0.0,
        cfg.initial_position_sigma_m,
        size=3,
    )
    x0[3:6] = rng_init.normal(
        0.0,
        cfg.initial_velocity_sigma_mps,
        size=3,
    )
    x0[6] = cfg.initial_clock_bias_ns * 1e-9
    x0[7] = cfg.initial_clock_drift_ppb * 1e-9

    P0 = np.diag(
        [
            cfg.initial_position_sigma_m**2,
            cfg.initial_position_sigma_m**2,
            cfg.initial_position_sigma_m**2,
            cfg.initial_velocity_sigma_mps**2,
            cfg.initial_velocity_sigma_mps**2,
            cfg.initial_velocity_sigma_mps**2,
            (cfg.initial_clock_bias_ns * 1e-9) ** 2,
            (cfg.initial_clock_drift_ppb * 1e-9) ** 2,
        ]
    )

    return HybridPntEkf(
        x0=x0,
        p0=P0,
        dt_s=cfg.dt_s,
        acceleration_process_sigma_mps2=cfg.acceleration_process_sigma_mps2,
        clock_bias_rw_sigma_s_sqrt_s=cfg.clock_bias_rw_sigma_ns_sqrt_s * 1e-9,
        clock_drift_rw_sigma_sps_sqrt_s=cfg.clock_drift_rw_sigma_ppb_sqrt_s * 1e-9,
    )


def run_scenario(cfg: Config, scenario: Scenario):
    rng = _rngs(cfg)

    truth_position = geodetic_to_ecef(
        cfg.receiver_lat_deg,
        cfg.receiver_lon_deg,
        cfg.receiver_alt_m,
    )
    truth_velocity = np.zeros(3)

    constellation = SyntheticLEOConstellation(
        cfg.leo_satellites,
        cfg.leo_altitude_m,
        cfg.leo_inclination_deg,
    )
    ekf = _build_filter(cfg, truth_position, rng["init"])

    times = np.arange(0.0, cfg.duration_s + cfg.dt_s, cfg.dt_s)

    # Truth clock starts at the post-GNSS-loss state, then wanders.
    truth_clock_bias_s = cfg.initial_clock_bias_ns * 1e-9
    truth_clock_drift_sps = cfg.initial_clock_drift_ppb * 1e-9

    position_error_m = []
    clock_error_ns = []
    clock_drift_error_ppb = []
    visible_count = []

    timing_interval_steps = max(
        1,
        int(round(cfg.iridium_update_interval_s / cfg.dt_s)),
    )

    for k, t_s in enumerate(times):
        if k > 0:
            # Truth receiver oscillator random walk.
            truth_clock_drift_sps += rng["truth"].normal(
                0.0,
                cfg.clock_drift_rw_sigma_ppb_sqrt_s
                * 1e-9
                * np.sqrt(cfg.dt_s),
            )
            truth_clock_bias_s += truth_clock_drift_sps * cfg.dt_s
            truth_clock_bias_s += rng["truth"].normal(
                0.0,
                cfg.clock_bias_rw_sigma_ns_sqrt_s
                * 1e-9
                * np.sqrt(cfg.dt_s),
            )

            ekf.predict()

        satellites = constellation.states_ecef(t_s)
        visible = [
            sat
            for sat in satellites
            if elevation_deg(truth_position, sat["position"])
            >= cfg.elevation_mask_deg
        ]
        visible_count.append(len(visible))

        if scenario.use_timing and (k % timing_interval_steps == 0):
            z_time = simulate_timing_s(
                rng["timing"],
                truth_clock_bias_s,
                cfg.iridium_time_sigma_ns * 1e-9,
            )
            ekf.update_timing(
                z_time,
                cfg.iridium_time_sigma_ns * 1e-9,
            )

        if scenario.use_doppler:
            for sat in visible:
                z_doppler = simulate_doppler_hz(
                    rng["doppler"],
                    truth_position,
                    truth_velocity,
                    truth_clock_drift_sps,
                    sat["position"],
                    sat["velocity"],
                    cfg.carrier_hz,
                    cfg.doppler_sigma_hz,
                )
                ekf.update_doppler(
                    z_doppler,
                    sat["position"],
                    sat["velocity"],
                    cfg.carrier_hz,
                    cfg.doppler_sigma_hz,
                )

        position_error_m.append(
            float(np.linalg.norm(ekf.x[0:3] - truth_position))
        )
        clock_error_ns.append(
            float((ekf.x[6] - truth_clock_bias_s) * 1e9)
        )
        clock_drift_error_ppb.append(
            float((ekf.x[7] - truth_clock_drift_sps) * 1e9)
        )

    return {
        "scenario": scenario.name,
        "time_s": times,
        "position_error_m": np.array(position_error_m),
        "clock_error_ns": np.array(clock_error_ns),
        "clock_drift_error_ppb": np.array(clock_drift_error_ppb),
        "visible_satellites": np.array(visible_count),
    }


def run_all(cfg: Config):
    return {
        scenario.name: run_scenario(cfg, scenario)
        for scenario in SCENARIOS
    }
