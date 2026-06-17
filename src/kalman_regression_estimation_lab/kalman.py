"""Linear discrete-time Kalman filter implementations."""

from __future__ import annotations

import numpy as np


class KalmanFilter:
    """Generic linear Kalman filter with predict-update recursion."""

    def __init__(
        self,
        a: np.ndarray,
        h: np.ndarray,
        q: np.ndarray,
        r: np.ndarray,
        x0: np.ndarray,
        p0: np.ndarray,
        b: np.ndarray | None = None,
    ) -> None:
        self.a = a
        self.h = h
        self.q = q
        self.r = r
        self.b = b

        self.x = x0.astype(float).copy()
        self.p = p0.astype(float).copy()

    def predict(self, u: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Propagate state mean and covariance forward one step."""
        if self.b is not None and u is not None:
            self.x = self.a @ self.x + self.b @ u
        else:
            self.x = self.a @ self.x

        self.p = self.a @ self.p @ self.a.T + self.q
        return self.x.copy(), self.p.copy()

    def update(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply measurement update using Joseph covariance form for stability."""
        innovation = z - self.h @ self.x
        s = self.h @ self.p @ self.h.T + self.r
        k = self.p @ self.h.T @ np.linalg.inv(s)

        self.x = self.x + k @ innovation
        i = np.eye(self.p.shape[0])
        self.p = (i - k @ self.h) @ self.p @ (i - k @ self.h).T + k @ self.r @ k.T

        return self.x.copy(), self.p.copy(), k.copy()


def make_cv_kalman_1d(
    dt: float,
    process_accel_std: float,
    measurement_std: float,
    x0: float,
    v0: float,
    p0_pos: float = 20.0,
    p0_vel: float = 20.0,
) -> KalmanFilter:
    """Factory for constant-velocity 1D tracker with position measurements."""
    a = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
    h = np.array([[1.0, 0.0]], dtype=float)
    g = np.array([[0.5 * dt * dt], [dt]], dtype=float)
    q = (process_accel_std**2) * (g @ g.T)
    r = np.array([[measurement_std**2]], dtype=float)
    x_init = np.array([x0, v0], dtype=float)
    p_init = np.diag([p0_pos, p0_vel]).astype(float)
    return KalmanFilter(a=a, h=h, q=q, r=r, x0=x_init, p0=p_init)


def run_kalman_1d(
    z: np.ndarray,
    dt: float,
    process_accel_std: float,
    measurement_std: float,
    x0: float,
    v0: float,
) -> dict[str, np.ndarray]:
    """Run 1D Kalman filtering over a stream of position measurements."""
    kf = make_cv_kalman_1d(
        dt=dt,
        process_accel_std=process_accel_std,
        measurement_std=measurement_std,
        x0=x0,
        v0=v0,
    )

    n = len(z)
    pred = np.zeros((n, 2), dtype=float)
    est = np.zeros((n, 2), dtype=float)
    gain = np.zeros((n, 2), dtype=float)

    for k in range(n):
        x_pred, _ = kf.predict()
        x_est, _, k_mat = kf.update(np.array([z[k]], dtype=float))

        pred[k] = x_pred
        est[k] = x_est
        gain[k] = k_mat[:, 0]

    return {
        "pred": pred,
        "est": est,
        "gain": gain,
    }


def run_kalman_2d_position_tracking(
    measurements: np.ndarray,
    dt: float,
    process_accel_std: float,
    measurement_std: float,
) -> dict[str, np.ndarray]:
    """Run 2D constant-velocity Kalman tracking using x/y position measurements."""
    a = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    h = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    g = np.array(
        [
            [0.5 * dt * dt, 0.0],
            [0.0, 0.5 * dt * dt],
            [dt, 0.0],
            [0.0, dt],
        ],
        dtype=float,
    )
    q = (process_accel_std**2) * (g @ g.T)
    r = np.diag([measurement_std**2, measurement_std**2]).astype(float)

    x0 = np.array([measurements[0, 0], measurements[0, 1], 0.0, 0.0], dtype=float)
    p0 = np.diag([30.0, 30.0, 10.0, 10.0]).astype(float)

    kf = KalmanFilter(a=a, h=h, q=q, r=r, x0=x0, p0=p0)

    n = len(measurements)
    est = np.zeros((n, 4), dtype=float)
    pred = np.zeros((n, 4), dtype=float)

    for k in range(n):
        x_pred, _ = kf.predict()
        x_est, _, _ = kf.update(measurements[k])
        pred[k] = x_pred
        est[k] = x_est

    return {
        "pred": pred,
        "est": est,
    }
