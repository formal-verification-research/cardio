# Cardio Code Audit Report

## Summary
Cardio is a Rust CTMC model checker with Python bindings used for testing and
differential validation against Storm. Core uniformization, Fox-Glynn, bounded-until
checking, and Python bindings are functional and agree with Storm on the toy model.
This document tracks production-readiness work; items marked [Resolved] have been
addressed in the codebase.

## Findings

### Correctness
- [Resolved] Uniformization row sums verified (Rust tests).
- [Resolved] Python bindings now return `PyResult` with input validation.
- [Resolved] `CheckContext` no longer uses `RwLock`; model context is owned directly,
  eliminating lock-poisoning panics.
- [Resolved] Labels `state_has_label`/`state_has_labels` no longer panic on out-of-range
  inputs; `add_label`/`add_label_to_state` return `Result`.
- Fox-Glynn PMF validated against SciPy (Python tests, 68 tests).
- Toy model bounds agree with Storm.
- Floating-point and BitVec-heavy code remain empirically validated only.

### Performance
- [Improved] Dense accumulator in `compute_transient` reduces conversions.
- [Improved] `steady_state` now precomputes row sums/deadlocks once per call instead
  of O(n*nnz) per iteration.
- Matrix-vector products in `compute_transient` are O(nnz) but repeated per Poisson
  term; for 9k-state models Cardio is ~60x slower than Storm. Consider caching,
  sparse BLAS, or blocked iteration.

### Robustness / API
- [Improved] `OptimalSprsMatBuilder::insert_row` now updates `state_count`, and
  uniformization uses `state_count.max(data.len())`.
- [Improved] `to_emb_dtmc` handles empty/zero-rate models without panicking.
- [Resolved] Parser integer/float parsing hardened to avoid panics on overflow.
- [Resolved] Labels lazily initializes state labelling instead of `unimplemented!()`.
- [Resolved] `OptimalSprsMatBuilder` row count now matches state count.
- [Improved] `AtomicProposition::parse` now implemented for arithmetic expressions
  and comparison operators.
- Public APIs still largely return values rather than `Result`; invalid inputs may
  panic. Continue migrating to `Result`-based error handling.

### Verification
- Kani: 6 harnesses verified.
- Verus: 5 proofs verified.
- Creusot: toolchain available; contracts not yet on production code.
- Floating-point and BitVec-heavy code remain empirically validated only.

### Missing features (production readiness)
- [Improved] PCTL-style operators `Next`, `Globally`, and weak until now have checker
  implementations (`compute_next`, `compute_globally`, `compute_weak_until`).
- Reward model checking integration.
- [Improved] Steady-state now exposes `steady_state_with_diagnostics` (iterations,
  residual) and uses precomputed row sums.
- [Improved] Basic edge-list model import implemented (`SparseModel::from_edge_list`);
  full PRISM/JANI import remains future work.
- Better error reporting instead of panics/log.

## Recommendations
1. Continue introducing `Result`-based error handling in public APIs.
2. Add benchmarks and optimize sparse operations.
3. Expand differential tests (random CTMCs vs Storm).
4. Complete verification of pure helpers; document float assumptions.
5. Provide CI for Kani/Verus/Creusot (optional).
