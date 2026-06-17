# When To Use Kalman vs Regression vs Neural Nets

## Use Kalman Filter When

- You have time-varying hidden state (position, velocity, attitude, drift).
- You know approximate dynamics and sensor structure.
- You need online updates with uncertainty.
- You are fusing multiple sensors in tracking, navigation, or control.

## Use Linear Regression When

- Problem is static mapping from features to target.
- Batch fitting is acceptable.
- Interpretability and simplicity matter more than temporal dynamics.

## Use Neural Nets When

- Dynamics or observation mappings are strongly nonlinear and data-rich.
- You prioritize expressive predictive power over strict interpretability.
- You can afford model training and validation overhead.

## Hybrid Pattern

In practical AI systems, neural networks can estimate difficult latent features, while Kalman filtering performs physically consistent temporal fusion and uncertainty propagation.
