import numpy as np

MU_EARTH = 3.986004418e14
OMEGA_EARTH = 7.2921150e-5
R_EARTH = 6378137.0


def _r1(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _r3(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


class SyntheticLEOConstellation:
    def __init__(self, count: int, altitude_m: float, inclination_deg: float):
        self.count = count
        self.radius = R_EARTH + altitude_m
        self.inclination = np.deg2rad(inclination_deg)
        self.mean_motion = np.sqrt(MU_EARTH / self.radius**3)

        # Spread satellites over several orbital planes.
        planes = max(3, int(np.ceil(np.sqrt(count))))
        sats = []
        for k in range(count):
            plane = k % planes
            slot = k // planes
            raan = 2.0 * np.pi * plane / planes
            phase = 2.0 * np.pi * (slot / max(1, int(np.ceil(count / planes))))
            phase += (plane % 2) * np.pi / max(1, count)
            sats.append((raan, phase))
        self.geometry = sats

    def states_ecef(self, t_s: float):
        states = []
        earth_rotation = _r3(-OMEGA_EARTH * t_s)

        for sat_id, (raan, phase) in enumerate(self.geometry):
            u = self.mean_motion * t_s + phase

            r_orb = np.array(
                [self.radius * np.cos(u), self.radius * np.sin(u), 0.0],
                dtype=float,
            )
            v_orb = np.array(
                [
                    -self.radius * self.mean_motion * np.sin(u),
                    self.radius * self.mean_motion * np.cos(u),
                    0.0,
                ],
                dtype=float,
            )

            orbital_to_eci = _r3(raan) @ _r1(self.inclination)
            r_eci = orbital_to_eci @ r_orb
            v_eci = orbital_to_eci @ v_orb

            omega_cross_r = np.cross(np.array([0.0, 0.0, OMEGA_EARTH]), r_eci)
            r_ecef = earth_rotation @ r_eci
            v_ecef = earth_rotation @ (v_eci - omega_cross_r)

            states.append(
                {
                    "id": sat_id,
                    "position": r_ecef,
                    "velocity": v_ecef,
                }
            )

        return states
