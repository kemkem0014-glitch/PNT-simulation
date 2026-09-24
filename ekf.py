import numpy as np

from measurements import predict_doppler_hz


class HybridPntEkf:
    """
    State:
        [x, y, z, vx, vy, vz, clock_bias_s, clock_drift_sps]
    """

    def __init__(
        self,
        x0: np.ndarray,
        p0: np.ndarray,
        dt_s: float,
        acceleration_process_sigma_mps2: float,
        clock_bias_rw_sigma_s_sqrt_s: float,
        clock_drift_rw_sigma_sps_sqrt_s: float,
    ):
        self.x = x0.astype(float).copy()
        self.P = p0.astype(float).copy()
        self.dt = float(dt_s)
        self.sigma_a = float(acceleration_process_sigma_mps2)
        self.sigma_cb = float(clock_bias_rw_sigma_s_sqrt_s)
        self.sigma_cd = float(clock_drift_rw_sigma_sps_sqrt_s)

    def predict(self):
        dt = self.dt

        F = np.eye(8)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        F[6, 7] = dt

        self.x = F @ self.x

        q = np.zeros((8, 8))

        # Constant-velocity process noise.
        qa = self.sigma_a**2
        block = qa * np.array(
            [[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]]
        )
        for axis in range(3):
            i = axis
            j = axis + 3
            q[np.ix_([i, j], [i, j])] = block

        # Simple clock random walks.
        q[6, 6] = (self.sigma_cb**2) * dt
        q[7, 7] = (self.sigma_cd**2) * dt

        self.P = F @ self.P @ F.T + q

    def _update(self, residual: np.ndarray, H: np.ndarray, R: np.ndarray):
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ residual

        # Joseph form for better numerical stability.
        I = np.eye(len(self.x))
        A = I - K @ H
        self.P = A @ self.P @ A.T + K @ R @ K.T

    def update_timing(self, measured_clock_bias_s: float, sigma_s: float):
        H = np.zeros((1, 8))
        H[0, 6] = 1.0

        predicted = self.x[6]
        residual = np.array([measured_clock_bias_s - predicted])
        R = np.array([[sigma_s**2]])

        self._update(residual, H, R)

    def update_doppler(
        self,
        measured_doppler_hz: float,
        satellite_position: np.ndarray,
        satellite_velocity: np.ndarray,
        carrier_hz: float,
        sigma_hz: float,
    ):
        def h(x):
            return predict_doppler_hz(
                x[0:3],
                x[3:6],
                x[7],
                satellite_position,
                satellite_velocity,
                carrier_hz,
            )

        predicted = h(self.x)

        # Numerical Jacobian keeps Ver.1 readable and reduces derivation mistakes.
        H = np.zeros((1, 8))
        steps = np.array([1.0, 1.0, 1.0, 0.01, 0.01, 0.01, 1e-9, 1e-11])

        for i in range(8):
            if i == 6:
                # Pure Doppler does not depend on absolute clock bias in this model.
                H[0, i] = 0.0
                continue
            dx = np.zeros(8)
            dx[i] = steps[i]
            H[0, i] = (h(self.x + dx) - h(self.x - dx)) / (2.0 * steps[i])

        residual = np.array([measured_doppler_hz - predicted])
        R = np.array([[sigma_hz**2]])

        self._update(residual, H, R)
