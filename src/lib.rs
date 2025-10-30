pub mod matrix;
pub mod checker;

pub enum PropertyType {
	SteadyStateProbability,
	TransientProbability(f64, f64),
}

pub struct CheckContext {
	precision: f64,
	prop_type: PropertyType,
}
