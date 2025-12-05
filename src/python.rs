// Python bindings
#[pyo3::pymodule]
mod cardiopy {
	use crate::{
		checker::{CheckContext, CslChecker},
		labels::Labels,
		matrix::{OptimalSprsMatBuilder, SprsMatBuilder},
	};
	use pyo3::prelude::*;

	#[pyclass]
	pub struct QuantitativeReachabilityFinder {
		matrix_builder: OptimalSprsMatBuilder<f64>,
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

		pub fn build_matrix_and_get_bounds(&self, time_bound: f64) -> (f64, f64) {
			let model_context = self.matrix_builder.to_model_context(&self.labelling, false);
			let relevant_bitmask = self
				.labelling
				.create_label_bitmask(vec!["absorbing".to_string(), "satisfying".to_string()]);
			let state_count = model_context.state_count();
			let relevant_states = self
				.labelling
				.create_relevant(&relevant_bitmask, state_count);
			let mut check_context: CheckContext<f64> = CheckContext::initialize_with_abs(
				&model_context,
				time_bound,
				1e-99,
				relevant_states.clone(),
				relevant_states.clone(),
			);
			// let mut csl_checker: CslChecker<f64> = CslChecker::default();
			unimplemented!();
		}
	}
}
