"""
File: streamlit_app.py
Purpose: Interactive 4-tab teaching app for all estimation modules.
         Tab A - static OLS vs Kalman 1D
         Tab B - same data three estimators with gain/innovation/RMSE panels
         Tab C - algorithm selection guide
         Tab D - GNC 2D Monte Carlo tracker
"""

from __future__ import annotations

import streamlit as st

from kalman_regression_estimation_lab.demos import (
    module_a_static_vs_dynamic,
    module_b_same_data_different_lenses,
    module_c_ai_contexts,
    module_d_gnc_monte_carlo,
)
from kalman_regression_estimation_lab.plots import (
    fig_module_a_dynamic,
    fig_module_a_static,
    fig_module_a_velocity,
    fig_module_b_innovation,
    fig_module_b_kalman_gain,
    fig_module_b_predict_vs_update,
    fig_module_b_rmse_bar,
    fig_module_b_three_estimators,
    fig_module_d_2d_track,
    fig_module_d_position_error,
    fig_module_d_velocity,
)


def _sidebar_a() -> dict:
    st.sidebar.header("Module A Controls")
    return dict(
        static_noise_std=st.sidebar.slider("OLS noise std", 0.1, 6.0, 2.0, 0.1),
        process_accel_std=st.sidebar.slider("Kalman process noise (accel std)", 0.01, 1.5, 0.2, 0.01),
        measurement_std=st.sidebar.slider("Kalman measurement noise std", 0.1, 8.0, 2.0, 0.1),
        seed=st.sidebar.number_input("Random seed", 0, 9999, 7),
    )


def _sidebar_b() -> dict:
    st.sidebar.header("Module B Controls")
    return dict(
        process_accel_std=st.sidebar.slider("Process noise (accel std)", 0.01, 1.5, 0.30, 0.01),
        measurement_std=st.sidebar.slider("Measurement noise std", 0.1, 8.0, 2.5, 0.1),
        maneuver_step=st.sidebar.slider("Maneuver start step", 10, 150, 85, 1),
        maneuver_accel=st.sidebar.slider("Maneuver accel bias", -2.0, 2.0, 1.0, 0.05),
        seed=st.sidebar.number_input("Random seed", 0, 9999, 9),
    )


def _sidebar_d() -> dict:
    st.sidebar.header("Module D Controls")
    return dict(
        n_runs=st.sidebar.slider("Monte Carlo runs", 5, 80, 40, 5),
        process_accel_std=st.sidebar.slider("Process noise (accel std)", 0.01, 1.5, 0.25, 0.01),
        measurement_std=st.sidebar.slider("Measurement noise std", 0.5, 10.0, 3.0, 0.1),
        seed=st.sidebar.number_input("Random seed", 0, 9999, 21),
    )


def tab_a() -> None:
    st.header("Module A - Static vs Dynamic Estimation")
    st.markdown(
        "**Left:** OLS fits one frozen line over all observations. "
        "**Right:** Kalman tracks position step-by-step using physics. "
        "The velocity panel (bottom right) shows a quantity the sensor never measured - "
        "Kalman infers it from the physics model."
    )

    params = _sidebar_a()
    data = module_a_static_vs_dynamic(**params)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_module_a_static(data["static"]), use_container_width=True)
    with col2:
        st.plotly_chart(fig_module_a_dynamic(data["dynamic"]), use_container_width=True)

    st.plotly_chart(fig_module_a_velocity(data["dynamic"]), use_container_width=True)

    with st.expander("What is happening here?"):
        st.markdown(
            "- OLS draws the single best-fit line over all 120 points at once. "
            "The result is frozen - it will never change regardless of new data.\n"
            "- Kalman processes one measurement per step. It predicts where the object "
            "should be (using `x_next = x + v*dt`), then corrects that prediction with "
            "the actual sensor reading.\n"
            "- The **velocity estimate** is the key demonstration: the sensor only records "
            "position. Kalman infers velocity by tracking how position changes over time "
            "relative to the physics model. OLS cannot do this."
        )


def tab_b() -> None:
    st.header("Module B - Same Data, Three Estimators")
    st.markdown(
        "All three estimators receive the exact same noisy position stream. "
        "A maneuver (sudden acceleration) happens mid-sequence. "
        "Watch how each estimator responds - or fails to respond."
    )

    params = _sidebar_b()
    data = module_b_same_data_different_lenses(**params)

    st.plotly_chart(fig_module_b_three_estimators(data), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_module_b_predict_vs_update(data), use_container_width=True)
    with col2:
        st.plotly_chart(fig_module_b_rmse_bar(data), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(fig_module_b_kalman_gain(data), use_container_width=True)
    with col4:
        st.plotly_chart(fig_module_b_innovation(data), use_container_width=True)

    with st.expander("Reading the Kalman Gain chart"):
        st.markdown(
            "- The Kalman gain `K` is the blending weight between the physics prediction "
            "and the sensor measurement.\n"
            "- `K = 1` means trust the sensor completely, ignore the model.\n"
            "- `K = 0` means trust the model completely, ignore the sensor.\n"
            "- At startup, `K` is high because uncertainty `P` is large. "
            "As the filter gains confidence, `K` settles to its steady-state value.\n"
            "- After a maneuver, `K` rises again because the model prediction is "
            "suddenly wrong and the innovation is large."
        )

    with st.expander("Reading the Innovation chart"):
        st.markdown(
            "- The **innovation** is `z_k - H * x_pred` - the gap between what the sensor "
            "saw and what the filter predicted.\n"
            "- If the filter is well-tuned, the innovation sequence should look like white "
            "noise (no trend, no autocorrelation).\n"
            "- A persistent bias or sudden spike at the maneuver step reveals model mismatch. "
            "This is the primary diagnostic for tuning `Q` and `R`."
        )


def tab_c() -> None:
    st.header("Module C - Algorithm Selection Guide")
    st.markdown(
        "Use this guide to decide which estimator is appropriate for your problem. "
        "The most common mistake is reaching for a neural network when a classical "
        "estimator would be more accurate, faster, and more interpretable."
    )

    ctx = module_c_ai_contexts()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Kalman filter is the right tool when...")
        for item in ctx["kalman_good_for"]:
            st.markdown(f"- {item}")

    with col2:
        st.subheader("Regression or neural net is enough when...")
        for item in ctx["regression_or_nn_enough_for"]:
            st.markdown(f"- {item}")

    st.divider()
    st.subheader("Selection rule of thumb")
    for rule in ctx["selection_rule_of_thumb"]:
        st.info(rule)

    st.divider()
    st.subheader("Decision table")
    st.markdown(
        "| Question | If YES | If NO |\n"
        "|---|---|---|\n"
        "| Do you know the equations of motion? | Kalman filter | Regression or neural net |\n"
        "| Is data arriving one sample at a time in real time? | RLS or Kalman | Batch OLS |\n"
        "| Do you need to estimate velocity from position? | Kalman only | Regression sufficient |\n"
        "| Does the relationship change slowly over time? | RLS with forgetting | Batch OLS |\n"
        "| Do you have millions of examples and no physics model? | Neural net | Kalman or OLS |\n"
        "| Do you need uncertainty bounds that update in real time? | Kalman | Regression CI |"
    )


def tab_d() -> None:
    st.header("Module D - GNC 2D Position / Velocity Tracking")
    st.markdown(
        "This module simulates a 2D constant-velocity target (drone, satellite, vehicle). "
        "The sensor reports only (x, y) position with noise. "
        "The Kalman filter tracks the full 4-state vector [x, y, vx, vy] and "
        "the Monte Carlo cloud shows how measurement noise scatters around the true path."
    )

    params = _sidebar_d()
    data = module_d_gnc_monte_carlo(**params)

    st.plotly_chart(fig_module_d_2d_track(data), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_module_d_velocity(data), use_container_width=True)
    with col2:
        st.plotly_chart(fig_module_d_position_error(data), use_container_width=True)

    with st.expander("What makes this a GNC problem?"):
        st.markdown(
            "**Guidance, Navigation, and Control (GNC)** is the engineering discipline "
            "responsible for knowing where a vehicle is (navigation), deciding where it "
            "should go (guidance), and making it go there (control). The Kalman filter "
            "was developed for exactly this - Rudolf Kalman's 1960 paper was motivated "
            "by Apollo program navigation.\n\n"
            "In this simulation:\n"
            "- The **true state** is [x, y, vx, vy] - 2D position and velocity\n"
            "- The **sensor** measures only x and y position, corrupted by noise\n"
            "- The **filter** recovers all four state components including velocity "
            "which was never directly observed\n"
            "- The **Monte Carlo cloud** shows what a real sensor array would produce "
            "across 40 independent noise realisations"
        )


def main() -> None:
    st.set_page_config(
        page_title="Kalman vs Regression Lab",
        page_icon="📡",
        layout="wide",
    )

    st.title("📡 Kalman Filter vs Regression - Estimation Lab")
    st.caption(
        "Four interactive modules teaching the deep connection between "
        "linear regression, recursive least squares, and the Kalman filter."
    )

    tab_labels = [
        "A - Static vs Dynamic",
        "B - Three Estimators",
        "C - Algorithm Guide",
        "D - GNC 2D Tracker",
    ]
    tab_a_ui, tab_b_ui, tab_c_ui, tab_d_ui = st.tabs(tab_labels)

    with tab_a_ui:
        tab_a()
    with tab_b_ui:
        tab_b()
    with tab_c_ui:
        tab_c()
    with tab_d_ui:
        tab_d()


if __name__ == "__main__":
    main()
