"""
File: test_estimators.py
Purpose: Correctness and regression tests for all estimators and simulation helpers.
         Every test verifies a numerical property against a known-good reference
         so changes to core algorithms cannot silently break results.
"""

from __future__ import annotations

import numpy as np
import pytest

from kalman_regression_estimation_lab.demos import (
    module_a_static_vs_dynamic,
    module_b_same_data_different_lenses,
    module_c_ai_contexts,
    module_d_gnc_monte_carlo,
)
from kalman_regression_estimation_lab.kalman import (
    KalmanFilter,
    make_cv_kalman_1d,
    run_kalman_1d,
    run_kalman_2d_position_tracking,
)
from kalman_regression_estimation_lab.regression import (
    fit_static_linear_regression,
    fit_time_regression,
    recursive_least_squares,
)
from kalman_regression_estimation_lab.simulation import (
    simulate_1d_motion,
    simulate_2d_gnc,
    simulate_static_line,
)


# ---------------------------------------------------------------------------
# Kalman filter - output shapes
# ---------------------------------------------------------------------------

def test_kalman_1d_output_shapes() -> None:
    """run_kalman_1d must return arrays of exactly the right shapes."""
    z = np.linspace(0.0, 10.0, 25)
    out = run_kalman_1d(z=z, dt=1.0, process_accel_std=0.2,
                        measurement_std=1.0, x0=0.0, v0=0.0)
    assert out["pred"].shape == (25, 2)
    assert out["est"].shape == (25, 2)
    assert out["gain"].shape == (25, 2)


def test_kalman_2d_output_shapes() -> None:
    """run_kalman_2d_position_tracking must return (n,4) arrays."""
    rng = np.random.default_rng(0)
    meas = rng.normal(0.0, 1.0, size=(50, 2))
    out = run_kalman_2d_position_tracking(meas, dt=1.0,
                                          process_accel_std=0.2,
                                          measurement_std=1.0)
    assert out["pred"].shape == (50, 4)
    assert out["est"].shape == (50, 4)


def test_kalman_predict_increases_covariance() -> None:
    """Predict step must increase (or equal) the trace of P - uncertainty grows without measurement."""
    kf = make_cv_kalman_1d(dt=1.0, process_accel_std=0.5,
                           measurement_std=1.0, x0=0.0, v0=0.0)
    p_before = np.trace(kf.p)
    kf.predict()
    p_after = np.trace(kf.p)
    assert p_after >= p_before


def test_kalman_update_decreases_covariance() -> None:
    """Update step must reduce the trace of P - measurement adds information."""
    kf = make_cv_kalman_1d(dt=1.0, process_accel_std=0.5,
                           measurement_std=1.0, x0=0.0, v0=0.0)
    kf.predict()
    p_before = np.trace(kf.p)
    kf.update(np.array([0.5]))
    p_after = np.trace(kf.p)
    assert p_after < p_before


def test_kalman_covariance_stays_symmetric() -> None:
    """P must remain symmetric after many predict-update cycles."""
    kf = make_cv_kalman_1d(dt=1.0, process_accel_std=0.3,
                           measurement_std=2.0, x0=0.0, v0=1.0)
    rng = np.random.default_rng(42)
    for _ in range(50):
        kf.predict()
        z = rng.normal(0.0, 2.0, size=(1,))
        kf.update(z)
    assert np.allclose(kf.p, kf.p.T, atol=1e-10), "Covariance matrix became asymmetric"


def test_kalman_covariance_stays_positive_semidefinite() -> None:
    """P must remain positive semi-definite (all eigenvalues >= 0) after 100 steps."""
    kf = make_cv_kalman_1d(dt=1.0, process_accel_std=0.3,
                           measurement_std=2.0, x0=0.0, v0=1.0)
    rng = np.random.default_rng(7)
    for _ in range(100):
        kf.predict()
        kf.update(rng.normal(0.0, 2.0, size=(1,)))
    eigvals = np.linalg.eigvalsh(kf.p)
    assert np.all(eigvals >= -1e-10), f"P has negative eigenvalue: {eigvals.min()}"


def test_kalman_gain_between_zero_and_one() -> None:
    """Kalman gain must stay in [0, 1] for a well-conditioned 1D problem."""
    z = np.linspace(0.0, 10.0, 60)
    out = run_kalman_1d(z=z, dt=1.0, process_accel_std=0.2,
                        measurement_std=1.5, x0=0.0, v0=0.5)
    assert np.all(out["gain"] >= -1e-9)
    assert np.all(out["gain"] <= 1.0 + 1e-9)


def test_kalman_beats_batch_regression_on_maneuver_rmse() -> None:
    """Kalman must outperform batch OLS when there is a mid-sequence maneuver."""
    data = module_b_same_data_different_lenses(
        process_accel_std=0.3, measurement_std=2.5,
        maneuver_step=70, maneuver_accel=1.2, seed=42,
    )
    true_pos = data["true_position"]
    rmse_batch = float(np.sqrt(np.mean((data["batch_regression"] - true_pos) ** 2)))
    rmse_kf    = float(np.sqrt(np.mean((data["kalman_estimate"]   - true_pos) ** 2)))
    assert rmse_kf < rmse_batch, (
        f"Kalman RMSE {rmse_kf:.3f} should be less than OLS RMSE {rmse_batch:.3f}"
    )


# ---------------------------------------------------------------------------
# OLS regression
# ---------------------------------------------------------------------------

def test_ols_recovers_known_slope_and_intercept() -> None:
    """OLS must recover exact slope and intercept from noiseless data."""
    x = np.linspace(0.0, 10.0, 200)
    y = 3.7 * x - 1.4   # exact line, zero noise
    result = fit_static_linear_regression(x, y)
    assert abs(result["slope"] - 3.7) < 1e-8
    assert abs(result["intercept"] - (-1.4)) < 1e-8


def test_ols_rmse_near_zero_on_noiseless_data() -> None:
    """RMSE must be essentially zero when data has no noise."""
    x = np.linspace(0.0, 5.0, 100)
    y = 2.0 * x + 0.5
    result = fit_static_linear_regression(x, y)
    assert result["rmse"] < 1e-10


def test_ols_ci_bands_contain_true_line() -> None:
    """95% CI bands must bracket the true line on a well-posed problem."""
    x, y_true, y_obs = simulate_static_line(n_points=200, noise_std=1.5, seed=0)
    result = fit_static_linear_regression(x, y_obs)
    # At least 90% of true-line points should be inside the CI
    inside = (y_true >= result["ci_low"]) & (y_true <= result["ci_high"])
    coverage = float(inside.mean())
    assert coverage >= 0.90, f"CI coverage only {coverage:.2%}"


def test_ols_output_keys() -> None:
    """fit_static_linear_regression must return all expected keys."""
    x = np.arange(20, dtype=float)
    y = x * 2.0 + rng_global().normal(0, 1, 20)
    result = fit_static_linear_regression(x, y)
    for key in ("slope", "intercept", "pred", "residuals", "rmse", "ci_low", "ci_high"):
        assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# RLS regression
# ---------------------------------------------------------------------------

def test_rls_converges_to_ols_with_lambda_one() -> None:
    """RLS with lambda=1 over enough samples must converge close to OLS solution."""
    rng = np.random.default_rng(3)
    t = np.linspace(0.0, 20.0, 500)
    y = 1.8 * t + 2.5 + rng.normal(0, 0.5, 500)

    rls = recursive_least_squares(t, y, forgetting_factor=1.0)
    ols = fit_static_linear_regression(t, y)

    # Final RLS pred vs OLS pred should be close
    rls_final_pred = rls["pred"][-50:].mean()
    ols_final_pred = ols["pred"][-50:].mean()
    assert abs(rls_final_pred - ols_final_pred) < 0.5


def test_rls_output_shapes() -> None:
    """recursive_least_squares must return arrays of the correct shape."""
    t = np.arange(80, dtype=float)
    y = 2.0 * t + rng_global().normal(0, 1, 80)
    result = recursive_least_squares(t, y)
    assert result["pred"].shape == (80,)
    assert result["coefficients"].shape == (80, 2)


def test_rls_forgetting_factor_adapts_faster() -> None:
    """Lower forgetting factor must adapt to a regime shift faster than lambda=1."""
    rng = np.random.default_rng(11)
    t = np.linspace(0.0, 10.0, 200)
    # First 100 points: slope 1, then slope 5
    y = np.where(t < 5.0, 1.0 * t, 5.0 * t - 20.0) + rng.normal(0, 0.3, 200)

    rls_slow = recursive_least_squares(t, y, forgetting_factor=1.00)
    rls_fast = recursive_least_squares(t, y, forgetting_factor=0.95)

    # After the shift, fast RLS should track better
    true_after = np.where(t >= 5.0, 5.0 * t - 20.0, np.nan)
    mask = t >= 7.0
    rmse_slow = float(np.sqrt(np.mean((rls_slow["pred"][mask] - true_after[mask]) ** 2)))
    rmse_fast = float(np.sqrt(np.mean((rls_fast["pred"][mask] - true_after[mask]) ** 2)))
    assert rmse_fast < rmse_slow


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def test_simulate_static_line_shape() -> None:
    x, y_true, y_obs = simulate_static_line(n_points=80)
    assert x.shape == (80,)
    assert y_true.shape == (80,)
    assert y_obs.shape == (80,)


def test_simulate_1d_motion_shape() -> None:
    scenario = simulate_1d_motion(n_steps=120)
    assert scenario.t.shape == (120,)
    assert scenario.true_position.shape == (120,)
    assert scenario.true_velocity.shape == (120,)
    assert scenario.measured_position.shape == (120,)


def test_simulate_2d_gnc_shape() -> None:
    out = simulate_2d_gnc(n_steps=100)
    assert out["true_state"].shape == (100, 4)
    assert out["measurements"].shape == (100, 2)


def test_simulate_1d_motion_is_seeded() -> None:
    """Same seed must produce identical output."""
    s1 = simulate_1d_motion(seed=42)
    s2 = simulate_1d_motion(seed=42)
    np.testing.assert_array_equal(s1.true_position, s2.true_position)


# ---------------------------------------------------------------------------
# Demo orchestration
# ---------------------------------------------------------------------------

def test_module_a_output_structure() -> None:
    data = module_a_static_vs_dynamic()
    assert "static" in data and "dynamic" in data
    assert "slope" in data["static"]
    assert "kf_est" in data["dynamic"]


def test_module_b_output_keys() -> None:
    data = module_b_same_data_different_lenses()
    for key in ("t", "true_position", "batch_regression",
                "online_regression", "kalman_estimate", "kalman_gain_pos"):
        assert key in data, f"Missing key: {key}"


def test_module_c_returns_all_categories() -> None:
    ctx = module_c_ai_contexts()
    assert "kalman_good_for" in ctx
    assert "regression_or_nn_enough_for" in ctx
    assert "selection_rule_of_thumb" in ctx
    assert len(ctx["kalman_good_for"]) >= 1


def test_module_d_output_structure() -> None:
    data = module_d_gnc_monte_carlo(n_runs=5)
    assert "true_state" in data
    assert "kalman_estimate" in data
    assert "mc_measurements" in data
    assert data["mc_measurements"].shape[0] == 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rng_global() -> np.random.Generator:
    """Return a fixed-seed RNG for deterministic test data generation."""
    return np.random.default_rng(99)
