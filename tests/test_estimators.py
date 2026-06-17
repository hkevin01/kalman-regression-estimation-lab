"""Basic estimator behavior tests for educational examples."""

from __future__ import annotations

import numpy as np

from kalman_regression_estimation_lab.demos import module_b_same_data_different_lenses
from kalman_regression_estimation_lab.kalman import run_kalman_1d


def test_kalman_output_shapes() -> None:
    z = np.linspace(0.0, 10.0, 25)
    out = run_kalman_1d(
        z=z,
        dt=1.0,
        process_accel_std=0.2,
        measurement_std=1.0,
        x0=0.0,
        v0=0.0,
    )
    assert out["pred"].shape == (25, 2)
    assert out["est"].shape == (25, 2)
    assert out["gain"].shape == (25, 2)


def test_kalman_beats_batch_regression_on_maneuver_rmse() -> None:
    data = module_b_same_data_different_lenses(
        process_accel_std=0.3,
        measurement_std=2.5,
        maneuver_step=70,
        maneuver_accel=1.2,
        seed=42,
    )
    true_pos = data["true_position"]
    rmse_batch = float(np.sqrt(np.mean((data["batch_regression"] - true_pos) ** 2)))
    rmse_kf = float(np.sqrt(np.mean((data["kalman_estimate"] - true_pos) ** 2)))
    assert rmse_kf < rmse_batch
