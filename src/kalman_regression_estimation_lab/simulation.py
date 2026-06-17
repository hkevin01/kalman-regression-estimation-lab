"""Simulation helpers for static and dynamic estimation examples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OneDScenario:
    """Container for 1D motion simulation outputs."""

    t: np.ndarray
    true_position: np.ndarray
    true_velocity: np.ndarray
    measured_position: np.ndarray


def simulate_static_line(
    n_points: int = 120,
    slope: float = 1.6,
    intercept: float = -2.0,
    noise_std: float = 2.0,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate static line data with additive Gaussian noise."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 20.0, n_points)
    y_true = slope * x + intercept
    y_obs = y_true + rng.normal(0.0, noise_std, size=n_points)
    return x, y_true, y_obs


def simulate_1d_motion(
    n_steps: int = 180,
    dt: float = 1.0,
    x0: float = 0.0,
    v0: float = 1.0,
    process_accel_std: float = 0.2,
    measurement_std: float = 2.0,
    maneuver_step: int | None = 90,
    maneuver_accel: float = 0.8,
    seed: int = 5,
) -> OneDScenario:
    """Simulate 1D constant-velocity motion with optional maneuver and noisy measurements."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps, dtype=float) * dt

    true_pos = np.zeros(n_steps, dtype=float)
    true_vel = np.zeros(n_steps, dtype=float)
    true_pos[0] = x0
    true_vel[0] = v0

    for k in range(1, n_steps):
        accel = rng.normal(0.0, process_accel_std)
        if maneuver_step is not None and k >= maneuver_step:
            accel += maneuver_accel
        true_pos[k] = true_pos[k - 1] + true_vel[k - 1] * dt + 0.5 * accel * dt * dt
        true_vel[k] = true_vel[k - 1] + accel * dt

    measured_pos = true_pos + rng.normal(0.0, measurement_std, size=n_steps)

    return OneDScenario(
        t=t,
        true_position=true_pos,
        true_velocity=true_vel,
        measured_position=measured_pos,
    )


def simulate_2d_gnc(
    n_steps: int = 140,
    dt: float = 1.0,
    process_accel_std: float = 0.25,
    measurement_std: float = 3.0,
    seed: int = 21,
) -> dict[str, np.ndarray]:
    """Simulate 2D position/velocity dynamics used in a GNC-style tracker."""
    rng = np.random.default_rng(seed)

    true_state = np.zeros((n_steps, 4), dtype=float)
    true_state[0] = np.array([0.0, 0.0, 1.3, 0.8], dtype=float)

    for k in range(1, n_steps):
        if k < n_steps // 3:
            maneuver = np.array([0.02, 0.00])
        elif k < 2 * n_steps // 3:
            maneuver = np.array([0.00, 0.04])
        else:
            maneuver = np.array([-0.03, 0.00])

        accel_noise = rng.normal(0.0, process_accel_std, size=2)
        a = maneuver + accel_noise

        x, y, vx, vy = true_state[k - 1]
        true_state[k, 0] = x + vx * dt + 0.5 * a[0] * dt * dt
        true_state[k, 1] = y + vy * dt + 0.5 * a[1] * dt * dt
        true_state[k, 2] = vx + a[0] * dt
        true_state[k, 3] = vy + a[1] * dt

    measurements = true_state[:, :2] + rng.normal(0.0, measurement_std, size=(n_steps, 2))

    return {
        "true_state": true_state,
        "measurements": measurements,
        "dt": np.array([dt]),
    }
