use pyo3::prelude::*;

pub mod checker;
pub mod labels;
pub mod matrix;
pub mod model;
pub mod parser;
pub mod poisson;
pub mod property;
pub mod python;
pub mod rewards;

use python::*;

pub struct CheckContext {
	precision: f64,
}
