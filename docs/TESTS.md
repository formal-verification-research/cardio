# TESTS.md — Cardio Test Results

This document records the expected and observed results of running Cardio's test
suites. It is intended to make test outcomes reproducible for contributors and
reviewers.

## Environment

- Rust toolchain with `cargo`.
- Python virtual environment at repository root (`.venv`).
- Python packages: `numpy`, `scipy`, `pytest`, `stormpy`.
- Build Cardio into the venv with `maturin develop`.

## Build

```text
maturin develop
```

Expected: successful build/install of the `cardio` extension module.

## Rust unit tests

Command:

```text
LIBRARY_PATH=/usr/lib/python3.13/config-3.13-x86_64-linux-gnu:$LIBRARY_PATH cargo test
```

Expected/actual:

- `41 passed; 0 failed; 0 ignored`.
- Covers matrix builder, uniformization row sums, Fox-Glynn, labels, checker
  absorption, property parsing, model import, and related helpers.

## Python tests

Command:

```text
source .venv/bin/activate
python -m pytest tests/ -q
```

Expected/actual:

- `119 passed, 10 warnings` (warnings are NumPy `matrix` deprecation warnings
  from existing tests; no failures).
- Breakdown:
  - `tests/poisson.py`: 67 passed (Fox-Glynn vs. SciPy).
  - `tests/test_cardio_simple.py`: 2 passed.
  - `tests/test_intervals.py`: 2 passed.
  - `tests/test_models.py`: 3 passed.
  - `tests/test_random_models.py`: 3 passed.
  - `tests/test_differential_random.py`: 100 passed (randomized Cardio vs. Storm).
  - `tests/test_edge_cases.py`: 10 passed (edge cases, Cardio vs. Storm).

Tolerance for differential comparisons: absolute difference `< 1e-5` (or
`< 1e-6` for the simpler tests).

## Toy model end-to-end

Command:

```text
source .venv/bin/activate
python tests/toy.py
```

Expected/actual:

- State count: `9722`.
- Max exit rate: `6.2`.
- Cardio: `[4.8091177643074e-12, 0.999999999999999]`, ~`6.11 s`.
- Storm: `[4.809117764307402e-12, 0.9999999999999989]`, ~`0.11 s`.
- Absorbing-state probability: `0.9999999999951894`.
- Bounds agree within tolerance.

## Notes

- Cardio is substantially slower than Storm on large models (~60x on the toy
  model), which is expected per the audit.
- The randomized differential tests use fixed seeds, so failures are
  reproducible.
