//! Cardio: a probabilistic model checker for continuous-time Markov chains (CTMCs).
//!
//! This crate exposes sparse matrix builders, labelling utilities, CSL property
//! definitions, a CSL model checker, Fox-Glynn Poisson truncation helpers, and
//! Python bindings used for testing.
//!
//! # Building
//!
//! ```text
//! maturin develop
//! ```
//!
//! # Testing
//!
//! Rust unit tests:
//!
//! ```text
//! cargo test
//! ```
//!
//! Python tests:
//!
//! ```text
//! python -m pytest tests/poisson.py
//! python tests/toy.py
//! ```
//!
//! # Formal verification
//!
//! Optional Kani proofs are gated behind `cfg(kani)`. To run them, install Kani
//! (<https://github.com/model-checking/kani>) and invoke `cargo kani`.

pub mod checker;
pub mod labels;
pub mod matrix;
pub mod model;
pub mod parser;
pub mod poisson;
pub mod property;
pub mod python;
pub mod rewards;


#[cfg(kani)]
mod kani_simple {
	#[kani::proof]
	#[kani::unwind(1)]
	fn check_addition_checked() {
		let a: i32 = kani::any();
		let b: i32 = kani::any();
		// checked_add never panics and agrees with wrapping-free arithmetic when in range
		let r = a.checked_add(b);
		if let Some(s) = r {
			assert!(s as i64 == a as i64 + b as i64);
		}
	}
}
