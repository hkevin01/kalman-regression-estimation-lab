"""Educational toolkit for comparing regression and Kalman filtering."""

from .demos import (
    module_a_static_vs_dynamic,
    module_b_same_data_different_lenses,
    module_c_ai_contexts,
    module_d_gnc_monte_carlo,
    module_e_nonlinear_ekf_ukf,
)
from .kalman import KalmanFilter, make_cv_kalman_1d, run_kalman_1d, run_kalman_2d_position_tracking
from .nonlinear import run_ekf_2d_bearing_tracking, run_ukf_2d_bearing_tracking
from .regression import fit_static_linear_regression, fit_time_regression, recursive_least_squares
from .simulation import (
    bearing_range_to_cartesian,
    cartesian_to_bearing_range,
    simulate_1d_motion,
    simulate_2d_bearing_sensor,
    simulate_2d_gnc,
    simulate_static_line,
)

__all__ = [
    # demos
    "module_a_static_vs_dynamic",
    "module_b_same_data_different_lenses",
    "module_c_ai_contexts",
    "module_d_gnc_monte_carlo",
    "module_e_nonlinear_ekf_ukf",
    # kalman
    "KalmanFilter",
    "make_cv_kalman_1d",
    "run_kalman_1d",
    "run_kalman_2d_position_tracking",
    # nonlinear
    "run_ekf_2d_bearing_tracking",
    "run_ukf_2d_bearing_tracking",
    # regression
    "fit_static_linear_regression",
    "fit_time_regression",
    "recursive_least_squares",
    # simulation
    "simulate_static_line",
    "simulate_1d_motion",
    "simulate_2d_gnc",
    "simulate_2d_bearing_sensor",
    "cartesian_to_bearing_range",
    "bearing_range_to_cartesian",
]
