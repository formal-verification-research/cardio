// Python bindings
#[pyo3::pymodule]
mod cardio {
	use crate::{
		checker::{CheckContext, CslChecker},
		labels::Labels,
		matrix::{OptimalSprsMatBuilder, SprsMatBuilder},
		property::*,
	};
	use pyo3::prelude::*;

	#[pyclass]
	pub struct QuantitativeReachabilityFinder {
		matrix_builder: OptimalSprsMatBuilder,
		labelling: Labels,
	}

	#[pymethods]
	impl QuantitativeReachabilityFinder {
		#[new]
		pub fn new() -> Self {
			Self {
				matrix_builder: OptimalSprsMatBuilder::new(),
				labelling: Labels::with_abs_and_sat(),
			}
		}

		pub fn insert(&mut self, row: usize, col: usize, entry: f64) {
			self.matrix_builder.insert(row, col, entry);
		}

		pub fn get_value(&mut self, row: usize, col: usize) -> Option<f64> {
			self.matrix_builder.get_value(row, col)
		}

		pub fn set_state_satisfying(&mut self, state_index: usize) {
			self.labelling.add_label_to_state(1, state_index)
		}

		pub fn build_matrix_and_get_bounds(&mut self, time_bound: f64) -> (f64, f64) {
			let model_context = self.matrix_builder.to_model_context(&self.labelling, false);
			let relevant_bitmask = self
				.labelling
				.create_label_bitmask(vec!["absorbing".to_string(), "satisfying".to_string()]);
			let state_count = model_context.state_count();
			let relevant_states = self
				.labelling
				.create_relevant(&relevant_bitmask, state_count);
			let mut check_context: CheckContext = CheckContext::initialize_with_abs(
				&model_context,
				time_bound,
				1e-99,
				relevant_states.clone(),
				relevant_states.clone(),
			);
			println!("Creating CSL checker.");
			let mut csl_checker: CslChecker = CslChecker::default();
			let interval = Interval::TimeBoundedUpper(time_bound);
			// let property = StateFormula::TransientQuery(
			// 	ProbabilityQueryType::SimpleQuery,
			// 	Box::new(PathFormula::Until(
			// 		Box::new(StateFormula::True),
			// 		interval,
			// 		Box::new(StateFormula::StringLabel("satisfying".to_string())),
			// 	)),
			// );
			// let (lower_bound, upper_bound) = property.create_bounds().unwrap();
			let distribution = csl_checker.compute_until(
				&mut check_context,
				interval,
				relevant_bitmask.clone(),
				relevant_bitmask.clone(),
			);

			let (mut lower_bound, mut upper_bound): (f64, f64) = (0.0, 0.0);
			for (state_id, probability) in distribution.iter() {
				// if the state has the absorbing or satisfying label, then it can go
				// to the upper_bound.
				if self.labelling.state_has_labels(state_id, &relevant_bitmask) {
					upper_bound += probability;
					// If it does NOT have the absorbing label, it can go to the lower
					// bound
					if !self.labelling.state_has_label(state_id, 0) {
						lower_bound += probability;
					}
				}
			}
			(lower_bound, upper_bound)
		}
	}
}
