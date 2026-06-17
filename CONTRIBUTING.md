# Contributing

Thank you for your interest in contributing to the Kalman Regression Estimation Lab. This project is primarily educational, so contributions that improve clarity, correctness, and teaching value are the most welcome.

## Types of Contributions

- **Bug fixes** - incorrect numerical results, broken imports, test failures
- **New teaching scenarios** - additional modules or sub-experiments that illustrate estimation concepts
- **Documentation improvements** - clearer explanations, better docstrings, updated README sections
- **New estimator comparisons** - Extended Kalman Filter (EKF), Unscented Kalman Filter (UKF), particle filter
- **Test coverage** - additional edge cases, numerical sanity checks

## Development Setup

```bash
git clone <repo-url>
cd kalman-regression-estimation-lab
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pip install black ruff pytest
```

## Code Style

- **Formatter:** `black` - run `black src/ tests/ app/` before committing
- **Linter:** `ruff` - run `ruff check src/ tests/ app/` before committing
- **Type hints:** use them for all public function signatures
- **Docstrings:** one-line summary + parameter descriptions for all public functions

## Branching and Commits

- Branch from `master` with a descriptive name: `fix/rls-gain`, `feat/ukf-module`, `docs/module-b-explanation`
- Commit messages should be concise and in the imperative: `Add UKF implementation`, `Fix RLS forgetting factor edge case`
- One logical change per commit

## Running Tests

```bash
pytest -q
```

All tests must pass before opening a PR. New code should include new tests.

## Pull Request Process

1. Open a PR against `master` using the provided template.
2. Describe what changed and why.
3. Ensure CI (tests + lint) passes.
4. A maintainer will review and merge or request changes.

## Numerical Correctness Standards

Since this project is about estimation algorithms, correctness matters more than style. Any change to `kalman.py` or `regression.py` must include at least one test that verifies the numerical output against a known-good reference (analytical solution, published example, or scipy equivalent).

## Code of Conduct

All contributors are expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
