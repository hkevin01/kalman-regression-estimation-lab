"""Regression baselines for static and online comparisons."""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression


def fit_static_linear_regression(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray | float]:
    """Fit a one-dimensional linear model and compute residual diagnostics."""
    X = x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    y_hat = model.predict(X)

    residuals = y - y_hat
    mse = float(np.mean(residuals**2))
    rmse = float(np.sqrt(mse))

    n = len(x)
    dof = max(n - 2, 1)
    x_bar = float(np.mean(x))
    sxx = float(np.sum((x - x_bar) ** 2))
    sigma2 = float(np.sum(residuals**2) / dof)
    t_crit = float(stats.t.ppf(0.975, dof))

    se_mean = np.sqrt(sigma2 * (1.0 / n + ((x - x_bar) ** 2) / max(sxx, 1e-12)))
    ci_low = y_hat - t_crit * se_mean
    ci_high = y_hat + t_crit * se_mean

    return {
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "pred": y_hat,
        "residuals": residuals,
        "rmse": rmse,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def fit_time_regression(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Fit x(t) using batch linear regression over all samples."""
    X = t.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, x)
    return model.predict(X)


def recursive_least_squares(
    t: np.ndarray,
    y: np.ndarray,
    forgetting_factor: float = 0.98,
    delta: float = 1000.0,
) -> dict[str, np.ndarray]:
    """Online linear regression via recursive least squares (RLS)."""
    n = len(t)
    theta = np.zeros(2, dtype=float)
    p = np.eye(2, dtype=float) * delta

    coeff_history = np.zeros((n, 2), dtype=float)
    pred_history = np.zeros(n, dtype=float)

    for k in range(n):
        phi = np.array([1.0, t[k]], dtype=float)
        pred = float(phi @ theta)
        err = y[k] - pred

        gain_num = p @ phi
        gain_den = forgetting_factor + phi.T @ p @ phi
        gain = gain_num / gain_den

        theta = theta + gain * err
        p = (p - np.outer(gain, phi) @ p) / forgetting_factor

        pred_history[k] = pred
        coeff_history[k] = theta

    return {
        "pred": pred_history,
        "coefficients": coeff_history,
    }
