# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- `.gitignore` for Python, Jupyter, Streamlit, and editor artefacts
- `.github/` folder with CI workflow, issue templates, and PR template
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CHANGELOG.md`
- Enhanced `README.md` with Mermaid diagrams, tables, GitHub Alerts, and ArXiv citations
- `plots.py` reusable plotting layer with Module A/B/D figure builders
- Streamlit app expanded to four tabs with gain, innovation, velocity, error, and RMSE diagnostics
- Public API exports in `__init__.py` for demos, Kalman, regression, and simulation utilities
- Comprehensive estimator test suite expanded to 23 tests
- README roadmap/status section documenting implemented milestones and remaining next steps

---

## [0.1.0] - 2026-06-17

### Added
- `KalmanFilter` class with predict-update recursion using Joseph-form covariance update
- `make_cv_kalman_1d` and `run_kalman_1d` factory and runner for 1D constant-velocity tracking
- `run_kalman_2d_position_tracking` for 4-state (x, y, vx, vy) 2D tracker
- `fit_static_linear_regression` with OLS, residual diagnostics, and 95% CI bands
- `recursive_least_squares` with configurable forgetting factor
- `simulate_static_line`, `simulate_1d_motion`, `simulate_2d_gnc` data generators
- `demos.py` orchestration for Modules A, B, C, D
- Streamlit interactive app (`app/streamlit_app.py`)
- Teaching notebook (`notebooks/kalman_vs_regression.ipynb`)
- Baseline tests (`tests/test_estimators.py`)
- `pyproject.toml` with setuptools build backend
