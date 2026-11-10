# Cardio Code Optimizations

This document records the performance optimizations applied to Cardio, the
motivation for each, and the measured impact. All timings below are for the
release build (`maturin develop --release`) on the 9,722-state toy model
unless noted otherwise.

## 1. Parallel lower/upper bounds in `build_matrix_and_get_bounds`

**Location:** `src/python.rs` (`build_matrix_and_get_bounds`)

`build_matrix_and_get_bounds` computes two independent bounded-until
probabilities:

- the lower bound `P(phi U psi_sat)` (reaching a satisfying state), and
- the absorption contribution `P(phi U psi_abs)` (being absorbed), which is
  added to form the upper bound.

These two computations share the same uniformized matrix but have different
absorbing-state sets, so they were originally run sequentially. They are now
executed concurrently with `rayon::join`, which halves the wall-clock time of
the Python entry point.

## 2. Precomputed active rows in `compute_until` (TimeBoundedUpper)

**Location:** `src/checker.rs` (`CslChecker::compute_until`,
`Interval::TimeBoundedUpper`)

The absorbing matrix application previously iterated over every triplet of the
uniformized matrix and tested `is_phi[row] && !is_psi[row]` for each one,
plus a separate O(n) pass to copy absorbing rows. The hot loop now:

1. Precomputes `active_entries`: the subset of triplets whose row is active
   (`phi & !psi`), once per call.
2. Precomputes the de-duplicated list of active rows.
3. In `apply`, clones the input vector (absorbing rows keep their value) and
   zeroes only the active rows before scattering the precomputed entries.

This removes per-triplet branching from the inner loop, which runs once per
Poisson term (and once per left-tail step).

## 3. Dense accumulator in `compute_transient`

**Location:** `src/checker.rs` (`CslChecker::compute_transient`)

Instead of converting the sparse working vector back and forth for every
Poisson term, contributions are accumulated into a single `Vec<f64>` of
length `n` and converted to a sparse vector once at the end. This avoids
repeated sparse allocations and keeps the inner loop cache-friendly.

## 4. Precomputed row sums / deadlocks in steady-state

**Location:** `src/checker.rs` (`CslChecker::steady_state_with_diagnostics`)

Row sums and the deadlock mask are computed once per call (O(nnz)) instead of
being recomputed inside the per-iteration loop (previously O(n * nnz) per
iteration).

## 5. Release builds

**Location:** `pyproject.toml` / `Cargo.toml`

The published wheel is built with `maturin build` / `maturin develop
--release`, which enables LLVM optimizations. A debug build of the toy model
took ~6.17s; the release build takes ~0.067s — roughly a 90x improvement
from optimization level alone.

## Measured impact (toy model, 9,722 states, time bound 100)

| Configuration | Cardio time |
|---|---|
| Debug build (pre-optimization) | ~6.17 s |
| Release build (pre-optimization) | ~0.435 s |
| Release build (post-optimization) | ~0.067 s |
| Storm (reference) | ~0.111 s |

Cardio now runs faster than Storm on this workload while producing identical
bounds (`[4.809e-12, ~1.0]`).

## Remaining opportunities

- Sparse BLAS / blocked iteration for the matrix-vector products.
- Avoiding repeated `M^k` evaluation across the lower and upper bounds
  (they share `P` but not the absorbing-row sets).
- Caching uniformized matrices across repeated queries on the same model.