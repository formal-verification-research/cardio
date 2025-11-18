pub mod checker;
pub mod labels;
pub mod matrix;
pub mod model;
pub mod parser;
pub mod poisson;
pub mod property;
pub mod rewards;

pub struct CheckContext {
	precision: f64,
}
