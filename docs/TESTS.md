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

- `222 passed, 10 warnings` (warnings are NumPy `matrix` deprecation warnings
  from existing tests; no failures).
- Breakdown:
  - `tests/poisson.py`: 67 passed (Fox-Glynn vs. SciPy).
  - `tests/test_cardio_simple.py`: 2 passed.
  - `tests/test_intervals.py`: 2 passed.
  - `tests/test_models.py`: 3 passed.
  - `tests/test_random_models.py`: 3 passed.
  - `tests/test_differential_random.py`: 200 passed (randomized Cardio vs. Storm).
  - `tests/test_edge_cases.py`: 13 passed (edge cases, intervals, next, Cardio vs. Storm).

## Test inventory (sample)

| Test | Description | Expected vs. actual |
|---|---|---|
| `test_simple_absorption` | Single transition rate 1 to absorbing | lower ~0, upper ~0.6321 (Storm/closed form) |
| `test_window_matches_storm` | Window property on 2-state CTMC | Cardio == Storm |
| `test_lower_matches_storm` | Lower-bounded property on 2-state CTMC | Cardio == Storm |
| `test_absorbing_target_immediate` | Initial state already satisfying | Cardio == Storm |
| `test_two_state_chain` | Incrementing counter to threshold | Cardio == Storm |
| `test_birth_death_small` | Birth/death with no satisfying target | Cardio == Storm |
| `test_random_walk_1d` | Random walk with state-dependent rates | Cardio == Storm |
| `test_random_two_transitions` | Two independent incrementing transitions | Cardio == Storm |
| `test_random_absorbing_heavy` | Low rates, no satisfying target | Cardio == Storm |
| `test_var_bound_zero` | Zero variable bound (immediate absorption) | Cardio == Storm |
| `test_initial_state_satisfying` | Initial state satisfies target | Cardio == Storm |
| `test_all_rates_zero` | All transition rates zero (deadlocks) | Cardio == Storm |
| `test_deadlock_state` | State with no enabled transitions | Cardio == Storm |
| `test_duplicate_transitions_same_target` | Duplicate edges accumulate rates | Cardio == Storm |
| `test_needed_constraints` | Guarded transitions via `needed` | Cardio == Storm |
| `test_self_loop_transition` | Explicit self-loop plus progress | Cardio == Storm |
| `test_multi_dim_birth_death` | 2D birth/death to threshold | Cardio == Storm |
| `test_large_time_bound` | Long time horizon (t=50) | Cardio == Storm |
| `test_small_time_bound` | Short time horizon (t=1e-3) | Cardio == Storm |
| `test_window_simple` | Window property on absorption chain | Cardio == Storm |
| `test_lower_simple` | Lower-bounded property on absorption chain | Cardio == Storm |
| `test_next_simple` | One-step probability to satisfying | Cardio == 1.0 |
| `test_random_differential (x200)` | Random CTMCs, sizes up to ~6^3 states | Cardio == Storm (<1e-5) |
| Fox-Glynn tests (67) | PMF/structure vs SciPy | All pass |

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
