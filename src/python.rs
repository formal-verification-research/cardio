// Python bindings
#[pyo3::pymodule]
mod cardio {
	use crate::{
		checker::{CheckContext, CslChecker},
		labels::Labels,
		matrix::{OptimalSprsMatBuilder, SprsMatBuilder},
		property::*,
	};
	use log::*;
	use pyo3::prelude::*;

	/// Builds a CTMC from explicit transitions and computes bounded-until reachability bounds.
	#[pyclass]
	pub struct QuantitativeReachabilityFinder {
		matrix_builder: OptimalSprsMatBuilder,
		labelling: Labels,
		satisfying_state_count: usize,
	}

	#[pymethods]
	impl QuantitativeReachabilityFinder {
		/// Create a new, empty reachability finder.
		#[new]
		pub fn new() -> Self {
			Self {
				matrix_builder: OptimalSprsMatBuilder::new(),
				labelling: Labels::with_abs_and_sat(),
				satisfying_state_count: 0,
			}
		}

		/// Insert a transition rate `entry` from `row` to `col`.
		pub fn insert(&mut self, row: usize, col: usize, entry: f64) -> pyo3::PyResult<()> {
			if !entry.is_finite() {
				return Err(pyo3::exceptions::PyValueError::new_err("entry must be finite"));
			}
			let _ = self.matrix_builder.insert(row, col, entry);
			Ok(())
		}

		/// Get the stored rate at (`row`, `col`), if any.
		pub fn get_value(&mut self, row: usize, col: usize) -> Option<f64> {
			self.matrix_builder.get_value(row, col)
		}

		/// Mark a state as satisfying the target predicate.
		pub fn set_state_satisfying(&mut self, state_index: usize) {
			self.satisfying_state_count += 1;
			let _ = self.labelling.add_label_to_state(1, state_index);
		}

		/// Build the uniformized matrix and return (lower_bound, upper_bound) reachability.
		pub fn build_matrix_and_get_bounds(&mut self, time_bound: f64) -> pyo3::PyResult<(f64, f64)> {
			let _ = env_logger::try_init();
			if time_bound <= 0.0 {
				return Err(pyo3::exceptions::PyValueError::new_err("time_bound must be positive"));
			}
			debug!("Satisfying state count: {}", self.satisfying_state_count);
			let model_context = self.matrix_builder.to_model_context(&self.labelling, false);
			let state_count = model_context.state_count();
			let deadlock_idxes = self.matrix_builder.get_deadlocks();
			let exit_rates = self.matrix_builder.row_sum_vec();

			// `phi` is `true` for every state, which is represented by an empty label
			// bitmask (no label requirements).
			let phi_mask = self.labelling.create_label_bitmask(vec![]);
			let psi_sat = self
				.labelling
				.create_label_bitmask(vec!["satisfying".to_string()]);
			let psi_abs = self
				.labelling
				.create_label_bitmask(vec!["absorbing".to_string()]);

			// Relevant states are those that satisfy phi (here, all states).
			let relevant_states = self.labelling.create_relevant(&phi_mask, state_count);

			info!("Creating CSL checker.");
			let csl_checker: CslChecker = CslChecker::default();
			let interval = Interval::TimeBoundedUpper(time_bound);

			// Lower bound: probability of reaching a satisfying state.
			let mut check_context: CheckContext = CheckContext::initialize_with_abs(
				&model_context,
				time_bound,
				1e-9,
				relevant_states.clone(),
				relevant_states.clone(),
				deadlock_idxes.clone(),
				exit_rates.clone(),
			);
			let dist_lower = csl_checker.compute_until(
				&mut check_context,
				interval,
				phi_mask.clone(),
				psi_sat.clone(),
			);
			let lower_bound = dist_lower.get(1).copied().unwrap_or(0.0);

			// Upper bound: also count paths that are absorbed before satisfying.
			let mut check_context: CheckContext = CheckContext::initialize_with_abs(
				&model_context,
				time_bound,
				1e-9,
				relevant_states.clone(),
				relevant_states.clone(),
				deadlock_idxes,
				exit_rates,
			);
			let dist_abs =
				csl_checker.compute_until(&mut check_context, interval, phi_mask, psi_abs);
			let absorption_prob = dist_abs.get(1).copied().unwrap_or(0.0);
			let upper_bound = lower_bound + absorption_prob;

			Ok((lower_bound, upper_bound))
		}

		/// Build the uniformized matrix and return the window probability
		/// P(phi U_[lower, upper] psi) for the initial state.
		pub fn build_matrix_and_get_window(
			&mut self,
			lower: f64,
			upper: f64,
		) -> pyo3::PyResult<f64> {
			if lower < 0.0 || upper < lower {
				return Err(pyo3::exceptions::PyValueError::new_err(
					"invalid window bounds",
				));
			}
			let model_context = self.matrix_builder.to_model_context(&self.labelling, false);
			let state_count = model_context.state_count();
			let exit_rates = self.matrix_builder.row_sum_vec();
			let phi_mask = self.labelling.create_label_bitmask(vec![]);
			let psi_sat = self
				.labelling
				.create_label_bitmask(vec!["satisfying".to_string()]);
			let relevant_states = self.labelling.create_relevant(&phi_mask, state_count);
			let deadlock_idxes = self.matrix_builder.get_deadlocks();
			let mut check_context = CheckContext::initialize_with_abs(
				&model_context,
				upper,
				1e-9,
				relevant_states.clone(),
				relevant_states.clone(),
				deadlock_idxes,
				exit_rates,
			);
			let checker = CslChecker::default();
			let interval = Interval::TimeBoundWindow(lower, upper);
			let dist = checker.compute_until(&mut check_context, interval, phi_mask, psi_sat);
			Ok(dist.get(1).copied().unwrap_or(0.0))
		}

		/// Build the uniformized matrix and return the lower-bounded probability
		/// P(phi U_[lower, oo) psi) for the initial state.
		pub fn build_matrix_and_get_lower(&mut self, lower: f64) -> pyo3::PyResult<f64> {
			if lower < 0.0 {
				return Err(pyo3::exceptions::PyValueError::new_err(
					"lower bound must be non-negative",
				));
			}
			let model_context = self.matrix_builder.to_model_context(&self.labelling, false);
			let state_count = model_context.state_count();
			let exit_rates = self.matrix_builder.row_sum_vec();
			let phi_mask = self.labelling.create_label_bitmask(vec![]);
			let psi_sat = self
				.labelling
				.create_label_bitmask(vec!["satisfying".to_string()]);
			let relevant_states = self.labelling.create_relevant(&phi_mask, state_count);
			let deadlock_idxes = self.matrix_builder.get_deadlocks();
			let mut check_context = CheckContext::initialize_with_abs(
				&model_context,
				lower.max(1.0),
				1e-9,
				relevant_states.clone(),
				relevant_states.clone(),
				deadlock_idxes,
				exit_rates,
			);
			let checker = CslChecker::default();
			let interval = Interval::TimeBoundedLower(lower);
			let dist = checker.compute_until(&mut check_context, interval, phi_mask, psi_sat);
			Ok(dist.get(1).copied().unwrap_or(0.0))
		}

		/// Build the uniformized matrix and return the one-step probability
		/// (in the uniformized chain) that the next state is satisfying.
		pub fn build_matrix_and_get_next(&mut self) -> pyo3::PyResult<f64> {
			let model_context = self.matrix_builder.to_model_context(&self.labelling, false);
			let state_count = model_context.state_count();
			let exit_rates = self.matrix_builder.row_sum_vec();
			let phi_mask = self.labelling.create_label_bitmask(vec![]);
			let psi_sat = self
				.labelling
				.create_label_bitmask(vec!["satisfying".to_string()]);
			let relevant_states = self.labelling.create_relevant(&phi_mask, state_count);
			let deadlock_idxes = self.matrix_builder.get_deadlocks();
			let mut check_context = CheckContext::initialize_with_abs(
				&model_context,
				1.0,
				1e-9,
				relevant_states.clone(),
				relevant_states.clone(),
				deadlock_idxes,
				exit_rates,
			);
			let checker = CslChecker::default();
			let dist = checker.compute_next(&mut check_context, psi_sat);
			Ok(dist.get(1).copied().unwrap_or(0.0))
		}
	}

	#[pyo3::pymodule]
	mod util {
		use crate::poisson::FoxGlynnBound;
		/// Fox-Glynn truncation helper. Returns (left, right, total_weight, weights).
		#[pyo3::pyfunction]
		pub fn fg_find(lambda: f64, epsilon: f64) -> pyo3::PyResult<(usize, usize, f64, Vec<f64>)> {
			if !lambda.is_finite() || lambda <= 0.0 {
				return Err(pyo3::exceptions::PyValueError::new_err("lambda must be positive and finite"));
			}
			if !epsilon.is_finite() || epsilon <= 0.0 {
				return Err(pyo3::exceptions::PyValueError::new_err("epsilon must be positive and finite"));
			}
			let bound = FoxGlynnBound::<f64>::fox_glynn(lambda, epsilon);
			Ok((bound.left, bound.right, bound.total_weight, bound.weights))
		}
	}
}
