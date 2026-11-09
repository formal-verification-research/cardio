# Cardio Code Audit Report

## Summary
Cardio is a research-grade CTMC model checker with Rust core and Python bindings.
Core uniformization and Fox-Glynn paths are functional and agree with Storm on the
toy model. Several areas need work before production use.

## Findings

### Correctness
- Uniformization row sums verified (Rust tests).
- Fox-Glynn PMF validated against SciPy (Python tests).
- Toy model bounds agree with Storm.
- `CheckContext` uses `RwLock` and many `.unwrap()` calls; poisoning would panic.

### Performance
- Matrix-vector products in `compute_transient` are O(nnz) but repeated per Poisson term; for 9k-state models Cardio is ~60x slower than Storm. Consider caching, sparse BLAS, or blocked iteration.
- `steady_state` recomputes row sums O(n*nnz) per iteration.

### Robustness / API
- Public APIs return values rather than `Result`; invalid inputs panic.
- Parser integer/float parsing hardened to avoid panics on overflow.
- Labels now lazily initializes state labelling instead of `unimplemented!()`.
- `OptimalSprsMatBuilder` row count now matches state count.

### Verification
- Kani: 6 harnesses verified.
- Verus: 5 proofs verified.
- Creusot: toolchain available; contracts not yet on production code.
- Floating-point and BitVec-heavy code remain empirically validated only.

### Missing features (production readiness)
- PCTL temporal operators beyond bounded until.
- Reward model checking integration.
- Steady-state analysis maturity.
- Model import (PRISM/JANI) not implemented.
- Better error reporting instead of panics/log.

## Recommendations
1. Introduce `Result`-based error handling in public APIs.
2. Add benchmarks and optimize sparse operations.
3. Expand differential tests (random CTMCs vs Storm).
4. Complete verification of pure helpers; document float assumptions.
5. Provide CI for Kani/Verus/Creusot (optional).
