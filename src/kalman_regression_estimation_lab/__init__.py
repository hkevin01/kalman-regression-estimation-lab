"""Educational toolkit for comparing regression and Kalman filtering."""

from .demos import (
    module_a_static_vs_dynamic,
    module_b_same_data_different_lenses,
    module_c_ai_contexts,
    module_d_gnc_monte_carlo,
)
from .kalman import KalmanFilter, make_cv_kalman_1d, run_kalman_1d, run_kalman_2d_position_tracking
from .regression import fit_static_linear_regression, fit_time_regression, recursive_least_squares
from .simulation import simulate_1d_motion, simulate_2d_gnc, simulate_static_line

__all__ = [
    # demos
    "module_a_static_vs_dynamic",
    "module_b_same_data_different_lenses",
    "module_c_ai_contexts",
    "module_d_gnc_monte_carlo",
    # kalman
    "KalmanFilter",
    "make_cv_kalman_1d",
    "run_kalman_1d",
    "run_kalman_2d_position_tracking",
    # regression
    "fit_static_linear_regression",
    "fit_time_regression",
    "recursive_least_squares",
    # simulation
    "simulate_static_line",
    "simulate_1d_motion",
    "simulate_2d_gnc",
]
