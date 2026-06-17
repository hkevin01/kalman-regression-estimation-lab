"""Interactive teaching app: regression vs Kalman side-by-side."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from kalman_regression_estimation_lab.demos import module_b_same_data_different_lenses


def _line_fig(data: dict[str, np.ndarray]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=data["t"], y=data["true_position"], name="True position", mode="lines")
    )
    fig.add_trace(
        go.Scatter(
            x=data["t"],
            y=data["measured_position"],
            name="Noisy measurement",
            mode="markers",
            marker={"size": 5, "opacity": 0.45},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["t"],
            y=data["batch_regression"],
            name="Batch linear regression",
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["t"],
            y=data["online_regression"],
            name="Online regression (RLS)",
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["t"],
            y=data["kalman_estimate"],
            name="Kalman estimate",
            mode="lines",
            line={"width": 3},
        )
    )
    fig.update_layout(
        title="Same Data, Different Lenses",
        xaxis_title="Time",
        yaxis_title="Position",
        template="plotly_white",
        legend={"orientation": "h", "y": -0.25},
    )
    return fig


def _pred_update_fig(data: dict[str, np.ndarray]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["t"],
            y=data["kalman_prediction"],
            name="Kalman prediction",
            mode="lines",
            line={"dash": "dash"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["t"],
            y=data["kalman_estimate"],
            name="Kalman update (posterior)",
            mode="lines",
            line={"width": 3},
        )
    )
    fig.update_layout(
        title="Kalman Predict vs Update",
        xaxis_title="Time",
        yaxis_title="Position",
        template="plotly_white",
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="Kalman vs Regression Lab", layout="wide")

    st.title("Kalman Filter vs Linear Regression Lab")
    st.caption("Dynamic state estimation vs static fitting with controllable noise and maneuvers")

    with st.sidebar:
        st.header("Simulation Controls")
        process_accel_std = st.slider("Process noise (accel std)", 0.01, 1.50, 0.30, 0.01)
        measurement_std = st.slider("Measurement noise std", 0.1, 8.0, 2.5, 0.1)
        maneuver_step = st.slider("Maneuver start step", 10, 150, 85, 1)
        maneuver_accel = st.slider("Maneuver accel bias", -2.0, 2.0, 1.0, 0.05)
        seed = st.number_input("Random seed", min_value=0, max_value=9999, value=9)

    data = module_b_same_data_different_lenses(
        process_accel_std=float(process_accel_std),
        measurement_std=float(measurement_std),
        maneuver_step=int(maneuver_step),
        maneuver_accel=float(maneuver_accel),
        seed=int(seed),
    )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(_line_fig(data), use_container_width=True)
    with c2:
        st.plotly_chart(_pred_update_fig(data), use_container_width=True)

    st.subheader("Teaching Notes")
    st.markdown(
        "- Batch regression uses one global line and cannot re-interpret past structure quickly.\n"
        "- Online regression adapts coefficients but has no explicit state-transition physics.\n"
        "- Kalman filter carries state and uncertainty forward, then fuses each new sensor reading."
    )


if __name__ == "__main__":
    main()
