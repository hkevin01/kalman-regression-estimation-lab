# Kalman Filter As Dynamic Regression

Linear regression solves a static mapping:

$$
y = X\beta + \epsilon
$$

Kalman filtering solves a dynamic state-space model:

$$
x_{k+1} = A x_k + B u_k + w_k
$$

$$
y_k = H x_k + v_k
$$

Under Gaussian assumptions, both are least-squares estimators. The key difference is structure:

- Regression estimates fixed coefficients from a batch of samples.
- Kalman estimates latent state recursively and updates uncertainty each step.
- Kalman includes explicit process dynamics through $A$ and process noise through $Q$.

In this project, Kalman can be interpreted as sequential constrained regression where each update combines:

- Model prediction from previous state
- New measurement weighted by uncertainty (Kalman gain)

This makes it suitable for online tracking and sensor fusion.
