"""Nonlinear state-estimation extensions: EKF and UKF for bearing-range tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def _normalize_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _cv_state_transition(x: np.ndarray, dt: float) -> np.ndarray:
    """Constant-velocity 2D state transition for x=[px, py, vx, vy]."""
    return np.array([
        x[0] + x[2] * dt,
        x[1] + x[3] * dt,
        x[2],
        x[3],
    ], dtype=float)


def _cv_state_jacobian(_: np.ndarray, dt: float) -> np.ndarray:
    """Jacobian of constant-velocity state transition."""
    return np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _bearing_range_measurement(x: np.ndarray) -> np.ndarray:
    """Nonlinear measurement function: z=[range, bearing]."""
    r = np.sqrt(max(x[0] ** 2 + x[1] ** 2, 1e-12))
    b = np.arctan2(x[1], x[0])
    return np.array([r, b], dtype=float)


def _bearing_range_jacobian(x: np.ndarray) -> np.ndarray:
    """Jacobian of z=[range, bearing] wrt x=[px, py, vx, vy]."""
    px, py = float(x[0]), float(x[1])
    r2 = max(px * px + py * py, 1e-12)
    r = np.sqrt(r2)
    return np.array(
        [
            [px / r, py / r, 0.0, 0.0],
            [-py / r2, px / r2, 0.0, 0.0],
        ],
        dtype=float,
    )


def _build_cv_process_noise(dt: float, process_accel_std: float) -> np.ndarray:
    """Build 2D constant-velocity process noise covariance from accel std."""
    g = np.array(
        [
            [0.5 * dt * dt, 0.0],
            [0.0, 0.5 * dt * dt],
            [dt, 0.0],
            [0.0, dt],
        ],
        dtype=float,
    )
    return (process_accel_std**2) * (g @ g.T)


def _nearest_spd(matrix: np.ndarray, floor: float = 1e-9) -> np.ndarray:
    """Project a symmetric matrix to positive definite form for numerical stability."""
    sym = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals_clipped = np.clip(eigvals, floor, None)
    spd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    return 0.5 * (spd + spd.T)


@dataclass
class ExtendedKalmanFilter:
    """Generic EKF with user-provided nonlinear functions and Jacobians."""

    f: Callable[[np.ndarray], np.ndarray]
    h: Callable[[np.ndarray], np.ndarray]
    jac_f: Callable[[np.ndarray], np.ndarray]
    jac_h: Callable[[np.ndarray], np.ndarray]
    q: np.ndarray
    r: np.ndarray
    x: np.ndarray
    p: np.ndarray

    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        """EKF predict step with local linearization of dynamics."""
        f_j = self.jac_f(self.x)
        self.x = self.f(self.x)
        self.p = f_j @ self.p @ f_j.T + self.q
        return self.x.copy(), self.p.copy()

    def update(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """EKF update step with local linearization of measurement model."""
        h_j = self.jac_h(self.x)
        innovation = z - self.h(self.x)
        innovation[1] = _normalize_angle(float(innovation[1]))

        s = h_j @ self.p @ h_j.T + self.r
        k = self.p @ h_j.T @ np.linalg.inv(s)

        self.x = self.x + k @ innovation
        i = np.eye(self.p.shape[0])
        self.p = (i - k @ h_j) @ self.p @ (i - k @ h_j).T + k @ self.r @ k.T
        return self.x.copy(), self.p.copy(), k.copy()


@dataclass
class UnscentedKalmanFilter:
    """Scaled UKF for nonlinear systems using deterministic sigma points."""

    f: Callable[[np.ndarray], np.ndarray]
    h: Callable[[np.ndarray], np.ndarray]
    q: np.ndarray
    r: np.ndarray
    x: np.ndarray
    p: np.ndarray
    alpha: float = 0.3
    beta: float = 2.0
    kappa: float = 0.0

    def _weights(self) -> tuple[np.ndarray, np.ndarray, float]:
        n = self.x.shape[0]
        lam = self.alpha * self.alpha * (n + self.kappa) - n
        wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)), dtype=float)
        wc = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)), dtype=float)
        wm[0] = lam / (n + lam)
        wc[0] = wm[0] + (1.0 - self.alpha * self.alpha + self.beta)
        return wm, wc, lam

    def _sigma_points(self) -> np.ndarray:
        n = self.x.shape[0]
        _, _, lam = self._weights()
        p_sym = _nearest_spd(self.p)
        chol = np.linalg.cholesky((n + lam) * p_sym)

        chi = np.zeros((2 * n + 1, n), dtype=float)
        chi[0] = self.x
        for i in range(n):
            chi[i + 1] = self.x + chol[:, i]
            chi[n + i + 1] = self.x - chol[:, i]
        return chi

    def predict(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """UKF predict step using propagated sigma points."""
        wm, wc, _ = self._weights()
        chi = self._sigma_points()
        chi_pred = np.array([self.f(c) for c in chi], dtype=float)

        x_pred = np.sum(wm[:, None] * chi_pred, axis=0)
        p_pred = self.q.copy()
        for i in range(chi_pred.shape[0]):
            dx = (chi_pred[i] - x_pred).reshape(-1, 1)
            p_pred = p_pred + wc[i] * (dx @ dx.T)

        self.x = x_pred
        self.p = _nearest_spd(p_pred)
        return self.x.copy(), self.p.copy(), chi_pred

    def update(self, z: np.ndarray, chi_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """UKF update step with nonlinear measurement transform."""
        wm, wc, _ = self._weights()
        z_sigma = np.array([self.h(c) for c in chi_pred], dtype=float)

        # Weighted mean with circular handling on bearing component.
        z_mean = np.zeros(2, dtype=float)
        z_mean[0] = float(np.sum(wm * z_sigma[:, 0]))
        sin_mean = np.sum(wm * np.sin(z_sigma[:, 1]))
        cos_mean = np.sum(wm * np.cos(z_sigma[:, 1]))
        z_mean[1] = float(np.arctan2(sin_mean, cos_mean))

        s = self.r.copy()
        pxz = np.zeros((self.x.shape[0], z.shape[0]), dtype=float)

        for i in range(z_sigma.shape[0]):
            dz = z_sigma[i] - z_mean
            dz[1] = _normalize_angle(float(dz[1]))
            dx = chi_pred[i] - self.x

            s = s + wc[i] * np.outer(dz, dz)
            pxz = pxz + wc[i] * np.outer(dx, dz)

        k = pxz @ np.linalg.inv(s)
        innovation = z - z_mean
        innovation[1] = _normalize_angle(float(innovation[1]))

        self.x = self.x + k @ innovation
        self.p = _nearest_spd(self.p - k @ s @ k.T)
        return self.x.copy(), self.p.copy(), k.copy()


def run_ekf_2d_bearing_tracking(
    measurements_bearing_range: np.ndarray,
    dt: float,
    process_accel_std: float,
    range_std: float,
    bearing_std_rad: float,
) -> dict[str, np.ndarray]:
    """Run EKF on a 2D constant-velocity model with [range, bearing] observations."""
    r0, b0 = float(measurements_bearing_range[0, 0]), float(measurements_bearing_range[0, 1])
    x0 = np.array([r0 * np.cos(b0), r0 * np.sin(b0), 0.0, 0.0], dtype=float)
    p0 = np.diag([50.0, 50.0, 20.0, 20.0]).astype(float)

    q = _build_cv_process_noise(dt=dt, process_accel_std=process_accel_std)
    r = np.diag([range_std**2, bearing_std_rad**2]).astype(float)

    ekf = ExtendedKalmanFilter(
        f=lambda x: _cv_state_transition(x, dt=dt),
        h=_bearing_range_measurement,
        jac_f=lambda x: _cv_state_jacobian(x, dt=dt),
        jac_h=_bearing_range_jacobian,
        q=q,
        r=r,
        x=x0,
        p=p0,
    )

    n = len(measurements_bearing_range)
    pred = np.zeros((n, 4), dtype=float)
    est = np.zeros((n, 4), dtype=float)

    for k in range(n):
        x_pred, _ = ekf.predict()
        x_est, _, _ = ekf.update(measurements_bearing_range[k])
        pred[k] = x_pred
        est[k] = x_est

    return {"pred": pred, "est": est}


def run_ukf_2d_bearing_tracking(
    measurements_bearing_range: np.ndarray,
    dt: float,
    process_accel_std: float,
    range_std: float,
    bearing_std_rad: float,
    alpha: float = 0.3,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> dict[str, np.ndarray]:
    """Run UKF on a 2D constant-velocity model with [range, bearing] observations."""
    r0, b0 = float(measurements_bearing_range[0, 0]), float(measurements_bearing_range[0, 1])
    x0 = np.array([r0 * np.cos(b0), r0 * np.sin(b0), 0.0, 0.0], dtype=float)
    p0 = np.diag([50.0, 50.0, 20.0, 20.0]).astype(float)

    q = _build_cv_process_noise(dt=dt, process_accel_std=process_accel_std)
    r = np.diag([range_std**2, bearing_std_rad**2]).astype(float)

    ukf = UnscentedKalmanFilter(
        f=lambda x: _cv_state_transition(x, dt=dt),
        h=_bearing_range_measurement,
        q=q,
        r=r,
        x=x0,
        p=p0,
        alpha=alpha,
        beta=beta,
        kappa=kappa,
    )

    n = len(measurements_bearing_range)
    pred = np.zeros((n, 4), dtype=float)
    est = np.zeros((n, 4), dtype=float)

    for k in range(n):
        x_pred, _, chi_pred = ukf.predict()
        x_est, _, _ = ukf.update(measurements_bearing_range[k], chi_pred)
        pred[k] = x_pred
        est[k] = x_est

    return {"pred": pred, "est": est}
