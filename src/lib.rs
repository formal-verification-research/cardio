pub mod checker;
pub mod matrix;
pub mod rewards;

pub enum PropertyType {
	SteadyStateProbability,
	TransientProbability(f64, f64),
}

pub struct CheckContext {
	precision: f64,
	prop_type: PropertyType,
}
