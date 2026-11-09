# Verification Inventory

This document tracks the formal and empirical verification status of Cardio.
It is intended to be updated as proofs and tests are added.

## Legend

- **Verified**: has a machine-checked proof or a passing automated test suite
  covering the listed property.
- **Partial**: some properties verified, others remain.
- **Unverified**: no proof or test coverage yet.
- **Not applicable**: component is experimental/unused.

## Rust unit tests (`cargo test`)

| Component | File | Status | Notes |
|---|---|---|---|
| Matrix builder (insert/get/overwrite) | `src/matrix.rs` | Verified | Deterministic + randomized tests |
| Uniformization row sums | `src/matrix.rs` | Verified | `rows_sum_to_one`, `random_ctmc_rows_sum_to_one` |
| Fox-Glynn structure/PMF accuracy | `src/poisson.rs` | Verified | Compared against Poisson PMF |
| Labels indexing | `src/labels.rs` | Partial | Basic round-trip tests |
| Until probability (simple absorption) | `src/checker.rs` | Verified | Matches closed form `1 - e^{-t}` |
| Property parsing/construction | `src/property.rs` | Partial | Construction + parse smoke tests |

Run with:
```bash
LIBRARY_PATH=/usr/lib/python3.13/config-3.13-x86_64-linux-gnu:$LIBRARY_PATH cargo test
```

## Python tests (`pytest`)

| Component | File | Status | Notes |
|---|---|---|---|
| Fox-Glynn vs. SciPy | `tests/poisson.py` | Verified | 68 tests |
| Simple absorption bound | `tests/test_cardio_simple.py` | Verified | Cardio vs. closed form |
| End-to-end toy model vs. Storm | `tests/toy.py` | Verified | Bounds agree with Storm |

Run with:
```bash
source .venv/bin/activate
python -m pytest tests/poisson.py tests/test_cardio_simple.py -q
python tests/toy.py
```

## Kani proofs

| Harness | File | Status | Notes |
|---|---|---|---|
| `check_addition_checked` | `src/lib.rs` | Verified | Overflow-safe addition |
| `evaluate_value_minus_no_panic` | `src/property.rs` | Verified | Subtraction correctness |
| `evaluate_value_plus_no_panic` | `src/property.rs` | Verified | Addition correctness |
| `evaluate_value_multiply_no_panic` | `src/property.rs` | Verified | Multiplication correctness |
| Labels bitvec proofs | `src/labels.rs` | Not attempted | BitVec-heavy; Kani does not terminate in reasonable time |

Run a single harness with:
```bash
cargo kani --harness <name>
```

## Verus proofs

| Proof | File | Status | Notes |
|---|---|---|---|
| `eval_minus` | `verification/verus/proofs.rs` | Verified | Checked subtraction |
| `checked_add_spec` | `verification/verus/proofs.rs` | Verified | Checked addition |
| `checked_mul_spec` | `verification/verus/proofs.rs` | Verified | Checked multiplication |

Run with:
```bash
/home/landon/Downloads/verus/verus verification/verus/proofs.rs
```

## Remaining work

- Prove matrix indexing and uniformization arithmetic (Kani/Verus).
- Prove Fox-Glynn truncation bounds (currently validated empirically vs. SciPy).
- Prove checker transient/until algorithms (currently validated empirically vs. Storm).
- Add randomized differential tests for `TimeBoundWindow` and `TimeBoundedLower`.
- Remove or implement `unimplemented!()` paths in `matrix.rs` (`to_inf_matrix` for `ExplicitSprsMatBuilder`) and `property.rs`.

## Why some components are not verifiable

- **BitVec-heavy code**: Kani struggles to unwind the internal loops of `bitvec` allocations, so
  proofs involving `Labels` do not terminate in reasonable time. Refactoring these structures to
  use plain `Vec<bool>` or small fixed-size arrays would make them Kani-friendly.
- **Floating point**: Verus does not reason about floating-point rounding; Fox-Glynn and
  uniformization are therefore validated empirically rather than proved.
- **External solvers**: Storm comparisons require the `stormpy` Python package.
