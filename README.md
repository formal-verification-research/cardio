# Cardio: an efficient probabilistic model checker written in Rust

Cardio is a probabilistic model checker for continuous-time Markov chains (CTMCs)
and (planned) discrete-time chains, written in Rust with Python bindings for
testing and experimentation. It focuses on bounded-until CSL properties via
uniformization and Fox-Glynn Poisson truncation.

## Features

- Sparse CTMC construction and uniformization
- Fox-Glynn Poisson truncation helper (`cardio.util.fg_find`)
- Quantitative reachability bounds (lower/upper) for bounded-until properties
- Python bindings for rapid prototyping and differential testing against Storm

## Prerequisites

- Rust toolchain (stable is sufficient for normal builds)
- Python 3.8+ with a virtual environment
- `maturin`, `numpy`, `scipy`, `pytest`, and (optionally) `stormpy`

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install maturin numpy scipy pytest stormpy
maturin develop
```

This compiles the Rust core and installs the `cardio` Python module into the venv.

## Quick start (Python)

```python
import cardio, math

rf = cardio.QuantitativeReachabilityFinder()
# state 0 is absorbing, state 1 is initial
rf.insert(1, 0, 1.0)          # rate 1 from state 1 to absorbing state 0
rf.set_state_satisfying(0)     # state 0 satisfies the target

lower, upper = rf.build_matrix_and_get_bounds(1.0)
print(lower, upper)           # ~0.0, ~0.6321
```

## Building and testing

Rust unit tests:

```bash
cargo test
```

Python tests:

```bash
python -m pytest tests/poisson.py
python -m pytest tests/test_cardio_simple.py
```

End-to-end model comparison against Storm:

```bash
python tests/toy.py                # both engines
python tests/toy.py --bypass_cardio # Storm only
python tests/toy.py --bypass_storm  # Cardio only
```

## Formal verification (optional)

Verification tooling is **not** required to build or run Cardio.

Kani (bounded model checking):

```bash
cargo kani
```

Verus (deductive verification):

```bash
/home/landon/Downloads/verus/verus verification/verus/proofs.rs
```

Creusot (requires `nightly-2026-08-03` + Why3):

```bash
RUSTUP_TOOLCHAIN=nightly-2026-08-03 cargo-creusot
```

See `VERIFICATION.md` for the full verification inventory and remaining work.

## API overview

- `cardio.QuantitativeReachabilityFinder`
  - `insert(row, col, entry)`: add a transition rate.
  - `get_value(row, col)`: query a stored rate.
  - `set_state_satisfying(state_index)`: mark a state as satisfying.
  - `build_matrix_and_get_bounds(time_bound)`: returns `(lower, upper)`.
- `cardio.util.fg_find(lambda, epsilon)`: Fox-Glynn helper returning
  `(left, right, total_weight, weights)`.

## Development notes

- State index `0` is absorbing; state index `1` is initial.
- CTMCs are uniformized; Fox-Glynn weights drive bounded-time computation.
- Rust formatting follows `rustfmt.toml` (tabs, Unix newlines).
- Production-readiness gaps are tracked in `AUDIT.md`.

## Troubleshooting

- **Linker cannot find `libpython`**: set
  `LIBRARY_PATH=/usr/lib/python3.13/config-3.13-x86_64-linux-gnu:$LIBRARY_PATH`.
- **`stormpy` missing**: the toy test still runs with `--bypass_storm`.
- **Kani/Verus not installed**: skip; they are optional.

---

Cardio is licensed under the GNU General Public License v3.
