"""
File: plots.py
Purpose: Reusable Plotly figure builders for all four teaching modules.
         Keeps the Streamlit app and notebook free of figure construction logic.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Module A helpers
# ---------------------------------------------------------------------------

def fig_module_a_static(data: dict) -> go.Figure:
    """
    Purpose: Plot OLS regression fit over noisy static line data with CI bands.
    Inputs:  data dict from module_a_static_vs_dynamic()["static"]
    Outputs: Plotly Figure
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["x"], y=data["y_obs"],
        mode="markers", name="Observations",
        marker=dict(size=5, opacity=0.5, color="#7ec8e3"),
    ))
    fig.add_trace(go.Scatter(
        x=data["x"], y=data["y_true"],
        mode="lines", name="True line",
        line=dict(color="#ffffff", dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=data["x"], y=data["pred"],
        mode="lines", name="OLS fit",
        line=dict(color="#f4a261", width=2),
    ))
    # 95% CI band
    x_rev = data["x"][::-1].tolist()
    fig.add_trace(go.Scatter(
        x=list(data["x"]) + x_rev,
        y=list(data["ci_high"]) + list(data["ci_low"][::-1]),
        fill="toself", fillcolor="rgba(244,162,97,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% CI",
    ))
    fig.update_layout(
        title=f"Module A - OLS Fit  |  slope={data['slope']:.3f}  intercept={data['intercept']:.3f}  RMSE={data['rmse']:.3f}",
        xaxis_title="x", yaxis_title="y",
        template="plotly_dark", legend=dict(orientation="h", y=-0.25),
    )
    return fig


def fig_module_a_dynamic(data: dict) -> go.Figure:
    """
    Purpose: Plot Kalman position estimate and uncertainty envelope over 1D motion.
    Inputs:  data dict from module_a_static_vs_dynamic()["dynamic"]
    Outputs: Plotly Figure
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["t"], y=data["measured_position"],
        mode="markers", name="Noisy measurement",
        marker=dict(size=4, opacity=0.4, color="#7ec8e3"),
    ))
    fig.add_trace(go.Scatter(
        x=data["t"], y=data["true_position"],
        mode="lines", name="True position",
        line=dict(color="#ffffff", dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=data["t"], y=data["kf_est"][:, 0],
        mode="lines", name="Kalman estimate",
        line=dict(color="#f4a261", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=data["t"], y=data["kf_pred"][:, 0],
        mode="lines", name="Kalman prediction",
        line=dict(color="#e76f51", width=1, dash="dash"),
    ))
    fig.update_layout(
        title="Module A - Kalman 1D Position Tracking",
        xaxis_title="Time (s)", yaxis_title="Position",
        template="plotly_dark", legend=dict(orientation="h", y=-0.25),
    )
    return fig


def fig_module_a_velocity(data: dict) -> go.Figure:
    """
    Purpose: Show Kalman velocity estimate vs true velocity - quantity never directly measured.
    Inputs:  data dict from module_a_static_vs_dynamic()["dynamic"]
    Outputs: Plotly Figure
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["t"], y=data["true_velocity"],
        mode="lines", name="True velocity",
        line=dict(color="#ffffff", dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=data["t"], y=data["kf_est"][:, 1],
        mode="lines", name="Kalman velocity estimate",
        line=dict(color="#2a9d8f", width=2),
    ))
    fig.update_layout(
        title="Kalman Velocity Estimate (never directly measured)",
        xaxis_title="Time (s)", yaxis_title="Velocity",
        template="plotly_dark",
    )
    return fig


# ---------------------------------------------------------------------------
# Module B helpers
# ---------------------------------------------------------------------------

def fig_module_b_three_estimators(data: dict) -> go.Figure:
    """
    Purpose: Overlay OLS, RLS, and Kalman estimates on the same noisy trajectory.
    Inputs:  data dict from module_b_same_data_different_lenses()
    Outputs: Plotly Figure
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["t"], y=data["measured_position"],
        mode="markers", name="Measurements",
        marker=dict(size=4, opacity=0.35, color="#adb5bd"),
    ))
    fig.add_trace(go.Scatter(
        x=data["t"], y=data["true_position"],
        mode="lines", name="True position",
        line=dict(color="#ffffff", dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=data["t"], y=data["batch_regression"],
        mode="lines", name="Batch OLS",
        line=dict(color="#e63946", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=data["t"], y=data["online_regression"],
        mode="lines", name="Online RLS",
        line=dict(color="#f4a261", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=data["t"], y=data["kalman_estimate"],
        mode="lines", name="Kalman",
        line=dict(color="#2a9d8f", width=3),
    ))
    fig.update_layout(
        title="Module B - Same Data, Three Estimators",
        xaxis_title="Time (s)", yaxis_title="Position",
        template="plotly_dark", legend=dict(orientation="h", y=-0.25),
    )
    return fig


def fig_module_b_predict_vs_update(data: dict) -> go.Figure:
    """
    Purpose: Show the Kalman predict step vs posterior update to illuminate the two-step cycle.
    Inputs:  data dict from module_b_same_data_different_lenses()
    Outputs: Plotly Figure
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["t"], y=data["kalman_prediction"],
        mode="lines", name="Predict (prior)",
        line=dict(color="#e63946", dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=data["t"], y=data["kalman_estimate"],
        mode="lines", name="Update (posterior)",
        line=dict(color="#2a9d8f", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=data["t"], y=data["true_position"],
        mode="lines", name="True position",
        line=dict(color="#ffffff", dash="dot"),
    ))
    fig.update_layout(
        title="Kalman Predict vs Update",
        xaxis_title="Time (s)", yaxis_title="Position",
        template="plotly_dark",
    )
    return fig


def fig_module_b_kalman_gain(data: dict) -> go.Figure:
    """
    Purpose: Plot Kalman gain over time to show how filter confidence evolves.
    Inputs:  data dict with "t" and "kalman_gain_pos" arrays
    Outputs: Plotly Figure
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["t"], y=data["kalman_gain_pos"],
        mode="lines", name="Kalman gain (position)",
        line=dict(color="#e9c46a", width=2),
        fill="tozeroy", fillcolor="rgba(233,196,106,0.15)",
    ))
    fig.update_layout(
        title="Kalman Gain - How Much to Trust the Sensor",
        xaxis_title="Time (s)", yaxis_title="Gain K",
        template="plotly_dark",
        annotations=[dict(
            x=0.5, y=1.05, xref="paper", yref="paper", showarrow=False,
            text="K near 1 = trust sensor more | K near 0 = trust physics model more",
            font=dict(size=11, color="#adb5bd"),
        )],
    )
    return fig


def fig_module_b_innovation(data: dict) -> go.Figure:
    """
    Purpose: Show innovation sequence (measurement residual) - should be white noise if filter is consistent.
    Inputs:  data dict from module_b_same_data_different_lenses()
    Outputs: Plotly Figure
    """
    innovation = data["measured_position"] - data["kalman_prediction"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data["t"], y=innovation,
        name="Innovation z - H*x_pred",
        marker_color="#e76f51",
    ))
    fig.add_hline(y=0, line_color="white", line_dash="dash")
    fig.update_layout(
        title="Innovation Sequence (should look like white noise if filter is well-tuned)",
        xaxis_title="Time (s)", yaxis_title="Innovation",
        template="plotly_dark",
    )
    return fig


def fig_module_b_rmse_bar(data: dict) -> go.Figure:
    """
    Purpose: Bar chart comparing RMSE of all three estimators against true position.
    Inputs:  data dict from module_b_same_data_different_lenses()
    Outputs: Plotly Figure
    """
    true = data["true_position"]
    rmse_ols = float(np.sqrt(np.mean((data["batch_regression"] - true) ** 2)))
    rmse_rls = float(np.sqrt(np.mean((data["online_regression"] - true) ** 2)))
    rmse_kf  = float(np.sqrt(np.mean((data["kalman_estimate"]   - true) ** 2)))

    fig = go.Figure(go.Bar(
        x=["Batch OLS", "Online RLS", "Kalman"],
        y=[rmse_ols, rmse_rls, rmse_kf],
        marker_color=["#e63946", "#f4a261", "#2a9d8f"],
        text=[f"{v:.3f}" for v in [rmse_ols, rmse_rls, rmse_kf]],
        textposition="outside",
    ))
    fig.update_layout(
        title="RMSE vs True Position - Lower is Better",
        yaxis_title="RMSE", template="plotly_dark",
        yaxis=dict(range=[0, max(rmse_ols, rmse_rls, rmse_kf) * 1.3]),
    )
    return fig


# ---------------------------------------------------------------------------
# Module D helpers
# ---------------------------------------------------------------------------

def fig_module_d_2d_track(data: dict) -> go.Figure:
    """
    Purpose: 2D plot of true path, Monte Carlo measurement cloud, and Kalman filtered track.
    Inputs:  data dict from module_d_gnc_monte_carlo()
    Outputs: Plotly Figure
    """
    fig = go.Figure()

    # Monte Carlo cloud - all runs as faint scatter
    mc = data["mc_measurements"]
    n_runs = mc.shape[0]
    for i in range(min(n_runs, 40)):
        fig.add_trace(go.Scatter(
            x=mc[i, :, 0], y=mc[i, :, 1],
            mode="markers",
            marker=dict(size=2, opacity=0.12, color="#7ec8e3"),
            showlegend=(i == 0),
            name="MC measurements" if i == 0 else None,
        ))

    # Raw measurements
    fig.add_trace(go.Scatter(
        x=data["measurements"][:, 0], y=data["measurements"][:, 1],
        mode="markers", name="Single-run measurements",
        marker=dict(size=5, opacity=0.6, color="#f4a261"),
    ))

    # True path
    fig.add_trace(go.Scatter(
        x=data["true_state"][:, 0], y=data["true_state"][:, 1],
        mode="lines", name="True path",
        line=dict(color="#ffffff", dash="dot", width=2),
    ))

    # Kalman track
    fig.add_trace(go.Scatter(
        x=data["kalman_estimate"][:, 0], y=data["kalman_estimate"][:, 1],
        mode="lines", name="Kalman track",
        line=dict(color="#2a9d8f", width=3),
    ))

    fig.update_layout(
        title="Module D - GNC 2D Tracking with Monte Carlo Measurement Cloud",
        xaxis_title="X position", yaxis_title="Y position",
        template="plotly_dark",
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def fig_module_d_velocity(data: dict) -> go.Figure:
    """
    Purpose: Show Kalman-estimated vx and vy vs true velocities.
    Inputs:  data dict from module_d_gnc_monte_carlo()
    Outputs: Plotly Figure
    """
    n = data["true_state"].shape[0]
    t = np.arange(n, dtype=float)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["X Velocity", "Y Velocity"])

    fig.add_trace(go.Scatter(x=t, y=data["true_state"][:, 2],
                             name="True vx", line=dict(color="#ffffff", dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=data["kalman_estimate"][:, 2],
                             name="Kalman vx", line=dict(color="#2a9d8f", width=2)), row=1, col=1)

    fig.add_trace(go.Scatter(x=t, y=data["true_state"][:, 3],
                             name="True vy", line=dict(color="#f4a261", dash="dot")), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=data["kalman_estimate"][:, 3],
                             name="Kalman vy", line=dict(color="#e9c46a", width=2)), row=2, col=1)

    fig.update_layout(
        title="Module D - Velocity Estimates (never directly measured)",
        template="plotly_dark", height=450,
    )
    return fig


def fig_module_d_position_error(data: dict) -> go.Figure:
    """
    Purpose: Show position error (Kalman estimate - true) over time for 2D tracker.
    Inputs:  data dict from module_d_gnc_monte_carlo()
    Outputs: Plotly Figure
    """
    n = data["true_state"].shape[0]
    t = np.arange(n, dtype=float)
    err_x = data["kalman_estimate"][:, 0] - data["true_state"][:, 0]
    err_y = data["kalman_estimate"][:, 1] - data["true_state"][:, 1]
    err_norm = np.sqrt(err_x**2 + err_y**2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=err_norm, mode="lines",
                             name="Position error norm",
                             line=dict(color="#e63946", width=2),
                             fill="tozeroy", fillcolor="rgba(230,57,70,0.15)"))
    fig.update_layout(
        title="Module D - Kalman Position Error ||estimate - true||",
        xaxis_title="Timestep", yaxis_title="Error (units)",
        template="plotly_dark",
    )
    return fig
