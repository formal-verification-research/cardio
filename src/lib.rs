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

