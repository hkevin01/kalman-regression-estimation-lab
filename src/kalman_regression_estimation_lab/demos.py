"""High-level educational demos for notebook and app use."""

from __future__ import annotations

import numpy as np

from .kalman import run_kalman_1d, run_kalman_2d_position_tracking
from .nonlinear import run_ekf_2d_bearing_tracking, run_ukf_2d_bearing_tracking
from .regression import fit_static_linear_regression, fit_time_regression, recursive_least_squares
from .simulation import simulate_1d_motion, simulate_2d_bearing_sensor, simulate_2d_gnc, simulate_static_line


def module_a_static_vs_dynamic(
    static_noise_std: float = 2.0,
    process_accel_std: float = 0.2,
    measurement_std: float = 2.0,
    seed: int = 7,
) -> dict[str, dict[str, np.ndarray | float]]:
    """Module A - compare static line fit vs dynamic state estimation."""
    x, y_true, y_obs = simulate_static_line(noise_std=static_noise_std, seed=seed)
    static_fit = fit_static_linear_regression(x, y_obs)

    scenario = simulate_1d_motion(
        process_accel_std=process_accel_std,
        measurement_std=measurement_std,
        seed=seed,
    )
    kf = run_kalman_1d(
        z=scenario.measured_position,
        dt=float(scenario.t[1] - scenario.t[0]),
        process_accel_std=process_accel_std,
        measurement_std=measurement_std,
        x0=float(scenario.measured_position[0]),
        v0=0.0,
    )

    return {
        "static": {
            "x": x,
            "y_true": y_true,
            "y_obs": y_obs,
            **static_fit,
        },
        "dynamic": {
            "t": scenario.t,
            "true_position": scenario.true_position,
            "true_velocity": scenario.true_velocity,
            "measured_position": scenario.measured_position,
            "kf_pred": kf["pred"],
            "kf_est": kf["est"],
            "kf_gain": kf["gain"],
        },
    }


def module_b_same_data_different_lenses(
    process_accel_std: float = 0.3,
    measurement_std: float = 2.5,
    maneuver_step: int = 85,
    maneuver_accel: float = 1.0,
    seed: int = 9,
) -> dict[str, np.ndarray]:
    """Module B - same noisy stream viewed by regression and Kalman filtering."""
    scenario = simulate_1d_motion(
        process_accel_std=process_accel_std,
        measurement_std=measurement_std,
        maneuver_step=maneuver_step,
        maneuver_accel=maneuver_accel,
        seed=seed,
    )

    reg_pred = fit_time_regression(scenario.t, scenario.measured_position)
    rls = recursive_least_squares(scenario.t, scenario.measured_position)

    kf = run_kalman_1d(
        z=scenario.measured_position,
        dt=float(scenario.t[1] - scenario.t[0]),
        process_accel_std=process_accel_std,
        measurement_std=measurement_std,
        x0=float(scenario.measured_position[0]),
        v0=0.0,
    )

    return {
        "t": scenario.t,
        "true_position": scenario.true_position,
        "measured_position": scenario.measured_position,
        "batch_regression": reg_pred,
        "online_regression": rls["pred"],
        "kalman_prediction": kf["pred"][:, 0],
        "kalman_estimate": kf["est"][:, 0],
        "kalman_gain_pos": kf["gain"][:, 0],
    }


def module_c_ai_contexts() -> dict[str, list[str]]:
    """Module C - practical context mapping for estimator selection."""
    return {
        "kalman_good_for": [
            "Tracking and sensor fusion in drones, robots, AR/VR, and autonomous systems",
            "Sequential prediction and nowcasting with streaming measurements",
            "Simulation and control in GNC and model-based reinforcement learning",
            "Interpretable uncertainty-aware state estimation",
        ],
        "regression_or_nn_enough_for": [
            "Static tabular prediction with no explicit state dynamics",
            "Offline batch modeling where online recursion is unnecessary",
            "Feature-to-target mapping when physical process models are unavailable",
        ],
        "selection_rule_of_thumb": [
            "Use regression for static global trends",
            "Use Kalman for time-varying latent state tracking with known dynamics",
            "Use neural models for complex nonlinear mappings when data is abundant",
        ],
    }


def module_d_gnc_monte_carlo(
    n_runs: int = 40,
    process_accel_std: float = 0.25,
    measurement_std: float = 3.0,
    seed: int = 21,
) -> dict[str, np.ndarray]:
    """GNC-style 2D tracking with Monte Carlo measurement clouds and Kalman overlay."""
    base = simulate_2d_gnc(
        process_accel_std=process_accel_std,
        measurement_std=measurement_std,
        seed=seed,
    )
    true_state = base["true_state"]
    dt = float(base["dt"][0])

    kf = run_kalman_2d_position_tracking(
        measurements=base["measurements"],
        dt=dt,
        process_accel_std=process_accel_std,
        measurement_std=measurement_std,
    )

    rng = np.random.default_rng(seed + 100)
    mc_measurements = np.zeros((n_runs, len(true_state), 2), dtype=float)
    for i in range(n_runs):
        mc_measurements[i] = true_state[:, :2] + rng.normal(
            0.0, measurement_std, size=(len(true_state), 2)
        )

    return {
        "true_state": true_state,
        "measurements": base["measurements"],
        "kalman_estimate": kf["est"],
        "mc_measurements": mc_measurements,
    }


def module_e_nonlinear_ekf_ukf(
    n_steps: int = 140,
    dt: float = 1.0,
    process_accel_std: float = 0.25,
    range_std: float = 2.5,
    bearing_std_rad: float = 0.03,
    seed: int = 77,
) -> dict[str, np.ndarray]:
    """Module E - nonlinear bearing/range tracking using EKF and UKF."""
    sim = simulate_2d_bearing_sensor(
        n_steps=n_steps,
        dt=dt,
        process_accel_std=process_accel_std,
        range_std=range_std,
        bearing_std_rad=bearing_std_rad,
        seed=seed,
    )

    ekf = run_ekf_2d_bearing_tracking(
        measurements_bearing_range=sim["measurements_bearing_range"],
        dt=dt,
        process_accel_std=process_accel_std,
        range_std=range_std,
        bearing_std_rad=bearing_std_rad,
    )
    ukf = run_ukf_2d_bearing_tracking(
        measurements_bearing_range=sim["measurements_bearing_range"],
        dt=dt,
        process_accel_std=process_accel_std,
        range_std=range_std,
        bearing_std_rad=bearing_std_rad,
    )

    return {
        "true_state": sim["true_state"],
        "measurements_bearing_range": sim["measurements_bearing_range"],
        "measurements_cartesian_naive": sim["measurements_cartesian_naive"],
        "ekf_estimate": ekf["est"],
        "ukf_estimate": ukf["est"],
        "ekf_prediction": ekf["pred"],
        "ukf_prediction": ukf["pred"],
    }
