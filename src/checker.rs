use std::ops::{Add, BitAnd, Not};
use std::sync::RwLock;

use crate::matrix::*;
use crate::poisson::FoxGlynnBound;
use crate::*;

use bitvec::prelude::*;
use log::*;
use num::traits::{Bounded, real::Real};
use sprs::{CsMat, CsVec, CsVecBase};

use self::property::Interval;

/// A CTMC transition matrix
pub trait CtmcTransMat {
	fn uniformize(rate_mat: Self) -> Self;
}

impl CtmcTransMat for sprs::CsMat<f64>
where
	f64: num::Num + Clone,
{
	fn uniformize(rate_mat: Self) -> Self {
		unimplemented!();
	}
}

#[derive(Clone, Debug)]
pub struct ExplicitModelContext
where
	f64: CheckableNumber,
{
	/// Whether the model is in discrete or continuous time
	discrete_time: bool,
	/// The state and transition labelling
	labels: labels::Labels,
	/// The uniformized DTMC if a CTMC or the probability matrix if it is a DTMC.
	uniformized_matrix: CsMat<f64>,
	/// The epoch time. If a DTMC, this should be one.
	epoch: f64,
}

impl ExplicitModelContext
where
	f64: CheckableNumber,
{
	pub fn new(
		discrete_time: bool,
		labels: &labels::Labels,
		uniformized_matrix: &CsMat<f64>,
		epoch: f64,
	) -> Self {
		Self {
			discrete_time,
			labels: (*labels).clone(),
			uniformized_matrix: uniformized_matrix.clone(),
			epoch,
		}
	}

	/// Returns the number of states in the explicit model
	pub fn state_count(&self) -> usize {
		self.uniformized_matrix.cols()
	}
}

/// A struct that contains the program context for a model checker.
pub struct CheckContext {
	/// The (current) probability distribution over states.
	/// TODO: should this be a Vec rather than a sparse vector?
	distribution: CsVec<f64>,
	/// The model to be checked, including the uniformization matrix as well as the labelling and
	/// the epoch
	model_context: RwLock<ExplicitModelContext>,
	/// The time bound to compute probabilities to.
	time_bound: f64,
	/// The numerical precision
	epsilon: f64,
	/// The states for which we perform model checking
	checked_values: BitVec,
	/// Exit rates for each row
	exit_rates: Vec<f64>,
	/// The value we add during the self multiplication
	add_vec: CsVec<f64>,
	/// The states for which precision is relevant
	relevant_states: BitVec,
	/// The precision to which we check
	precision: f64,
}

impl CheckContext {
	/// Creates a check context with an initial distribution, where the initial state index is 1,
	/// given a model context, time bound, and relevant states
	pub fn initialize_with_abs(
		model_context: &ExplicitModelContext,
		time_bound: f64,
		precision: f64,
		relevant_states: BitVec,
		checked_values: BitVec,
		exit_rates: Vec<f64>,
	) -> Self {
		let num_states = model_context.uniformized_matrix.cols();
		// The distribution starts with 100% of the probability at state 1, i.e., the initial state
		let distribution: CsVec<f64> = CsVec::new(num_states, vec![1], vec![1.0]);
		// epsilon and precision start at the same value, but epsilon is modified throughout model
		// checking, whereas precision remains the same.
		Self {
			distribution,
			model_context: RwLock::new((*model_context).clone()),
			time_bound,
			epsilon: precision,
			checked_values,
			exit_rates,
			add_vec: CsVec::empty(num_states), // TODO: I think this is where the error
			// lies. This should be the one-step vector
			relevant_states,
			precision,
		}
	}

	pub fn build_one_step(&mut self, non_sat_states: &BitVec) {
		let epoch = self.model_context.read().unwrap().epoch;
		// Create a bit vector representing the states which are both non-satisfying AND
		// are relevant
		let ns_rel = non_sat_states.clone().bitand(&self.relevant_states);
		for (idx, val) in self.exit_rates.iter().enumerate() {
			if *val != 0.0 && *ns_rel.get(idx).as_deref().unwrap() {
				self.add_vec.append(idx, val / epoch);
			}
		}
	}

	/// If there are states for which the precision is relevant.
	pub fn has_relevant_states(&self) -> bool {
		!self.relevant_states.is_empty()
	}

	/// If any state in the distribution has a nonzero value
	pub fn has_reachable_states(&self) -> bool {
		self.distribution.nnz() > 0
	}

	/// Get the epoch from the model with the read lock only taken in the context of this function.
	pub fn epoch(&self) -> f64 {
		self.model_context.read().unwrap().epoch
	}

	/// Checks to see if we've reached the desired precision for all of the relevant states. This
	/// function also updates the epsilon value thus it takes a `&mut self`.
	pub fn precision_reached(&mut self, intermediate_result: &CsVec<f64>) -> bool {
		// First, check if any elements in the intermediate result are NaN. If so, we cannot
		// continue iteration.
		for (idx, val) in intermediate_result.iter() {
			if val.is_nan() {
				panic!("Got NaN value for probability at index {idx} of intermediate result!");
			} else if val.is_infinite() {
				panic!("Got infinite value for probability at index {idx} of intermediate result!");
			}
		}

		// The element for new_epsilon when the result is zero
		let zero_epsilon = self.epsilon * 0.1;
		// Iterate over all relevant state indecies, take the results and map them to a candidate
		// next epsilon. We take the minimum of these as our new epsilon. If our new epsilon is
		// lower than the old epsilon then we can terminate, otherwise, continue.
		let new_epsilon = intermediate_result
			.iter()
			.filter(|(idx, _)| {
				if let Some(val) = self.relevant_states.get(*idx).as_deref() {
					*val
				} else {
					false
				}
			})
			.map(|(_, &state_result)| state_result * self.precision)
			// We can't just use `min()` since floats do not implement Ord (only PartialOrd)
			.fold(
				zero_epsilon,
				|val, new_val| if val > new_val { val } else { new_val },
			);

		if new_epsilon < self.epsilon {
			self.epsilon = new_epsilon;
			true
		} else {
			false
		}
	}

	/// Scales the distribution so that its L1 norm is 1.
	pub fn scale_distribution(&mut self) {
		let l1 = self.distribution.l1_norm();
		self.distribution /= l1;
	}

	/// Tells whether the distribution sums to 1, as is needed for an unconditional probability
	/// distribution, but is not strictly necessary for, e.g., the second step in a phi U psi
	/// computation.
	pub fn is_valid_distribution(&self) -> bool {
		self.distribution.l1_norm() == 1.0
	}

	/// Updates the relevant_states vector with the states that have state labels whose indecies
	/// are marked with the `label_indecies` input parameter. For use in computing until formulae,
	/// the user should first zero out the non-satisfying states for phi in the distribution. Do
	/// not scale the distribution, since the distribution will only need to be re-scaled at the
	/// end in the inverse direction.
	pub fn update_relevant_states(&mut self, label_indecies: &BitVec) {
		let model = self.model_context.read().unwrap();
		assert!(label_indecies.len() == model.labels.label_count());
		let state_count = model.state_count();
		self.relevant_states = BitVec::with_capacity(state_count);
		// Initialize the bitvector to all zeros
		(0..state_count).for_each(|_| self.relevant_states.push(false));
		assert!(self.relevant_states.len() == state_count);
		let mut relevant_state_count = 0;
		for (idx, &val) in self.distribution.iter() {
			if val != 0.0 || model.labels.state_has_labels(idx, &label_indecies) {
				self.relevant_states.set(idx, true);
				relevant_state_count += 1;
			}
		}
		debug!("Relevant state count: {relevant_state_count}");
	}

	/// Creates a vector of states which satisfy phi and not psi
	pub fn create_add_vec(&mut self, phi: &BitVec, psi: &BitVec) {
		// let phi_not_psi = phi.clone().bitand(psi.clone().not());
		let model = self.model_context.get_mut().unwrap();
		assert!(phi.len() == model.labels.label_count() && psi.len() == phi.len());
		let state_count = model.state_count();
		self.add_vec = CsVec::empty(state_count);
		let mut pns: BitVec = BitVec::with_capacity(state_count);
		// Initialize the bitvector to all zeros
		(0..state_count).for_each(|_| pns.push(false));
		for (idx, &val) in self.distribution.iter() {
			if val != 0.0
				|| (model.labels.state_has_labels(idx, &phi)
					&& !model.labels.state_has_labels(idx, &psi))
			{
				pns.set(idx, true);
			}
		}
		self.build_one_step(&pns);
	}

	/// Zeroes any state index in the distribution that does not have the labels in the label
	/// bitmask provided. If the bitmask does not match the number of labels, this function will
	/// panic. This function does not re-scale the distribution so that its L1 norm is 1, and so
	/// does not preserve the property that the distribution is a valid probability distribution.
	/// However, this is useful in until formulae phi U psi.
	pub fn zero_unsatisfying_states(&mut self, label_indecies: &BitVec) {
		let model = self.model_context.read().unwrap();
		assert!(label_indecies.len() == model.labels.label_count());
		// We can just filter_map out the states into the new distribution who do not have all the
		// labels in the label_indecies bitvector. Then, just unzip them into the indecies and data
		// vectors needed to create a new sparse vector.
		let (new_idxes, new_data): (Vec<_>, Vec<_>) = self
			.distribution
			.iter()
			.filter_map(|(idx, &val)| {
				if model.labels.state_has_labels(idx, &label_indecies) {
					Some((idx, val.clone()))
				} else {
					None
				}
			})
			.unzip();
		self.distribution = CsVec::new(new_idxes.len(), new_idxes, new_data);
	}
}

/// A CSL or PCTL model checker.
pub struct CslChecker {
	qualitative: bool,
	use_mixed_poisson: bool,
}

impl CslChecker {
	pub fn new(qualitative: bool, use_mixed_poisson: bool) -> Self {
		Self {
			qualitative,
			use_mixed_poisson,
		}
	}

	/// Computes the transient probabilities for a given context and relevent values. The relevant
	/// values are the nonzero probabilities and the states who have the labels we care about.
	pub fn compute_transient(&self, context: &mut CheckContext) -> CsVec<f64> {
		// TODO: more graceful handling if cannot read
		let model = context.model_context.read().unwrap();
		let lambda = model.epoch * context.time_bound;
		// Return the initial distribution if no epochs pass.
		if lambda == 0.0 {
			return context.distribution.clone();
		}
		if context.epsilon <= 1e-20 {
			warn!("Warning: extremely low truncation error may lead to numerical instability.");
		}
		debug!(
			"About to compute Fox-Glynn bound with lambda {} and epsilon {}",
			lambda, context.epsilon
		);
		let mut fg_result = FoxGlynnBound::fox_glynn(lambda, context.epsilon);
		debug!(
			"Fox-Glynn bounds are as follows: ({}, {})",
			fg_result.left, fg_result.right
		);

		if self.use_mixed_poisson {
			let (mut left, mut right): (usize, usize) = (0, fg_result.weights.len() - 1);
			let (mut sum_left, mut sum_right) = (0.0, 0.0);
			while left <= right {
				if fg_result.weights[left] < fg_result.weights[right] {
					sum_left += fg_result.weights[left];
					fg_result.weights[left] = (fg_result.total_weight - sum_left) / context.epsilon;
					left += 1;
				} else {
					let right_weight = fg_result.weights[right];
					fg_result.weights[right] = sum_right / model.epoch;
					sum_right += right_weight;
					if right == 0 {
						// Avoid underflow
						break;
					} else {
						right -= 1;
					}
				}
			}
			// TODO: check for numerical instability
		}

		debug!(
			"Starting iterations. Matrix size: {} x {}",
			model.uniformized_matrix.rows(),
			model.uniformized_matrix.cols()
		);

		// Create the result vector
		let mut first_iteration = fg_result.left;
		let mut result = if first_iteration == 0 {
			first_iteration += 1;
			context.distribution.map(|elem| elem * fg_result.weights[0])
		// The initial result must be uniformized if we are in continuous time and using mixed
		// poisson probabilities.
		} else if self.use_mixed_poisson && !model.discrete_time {
			context.distribution.map(|&elem| elem / model.epoch)
		} else {
			CsVec::empty(context.distribution.dim())
		};

		// An optimization shamelessly stolen from storm: if we don't have to use mixed poisson
		// probabilities and our left fox-glynn result is > 1, we don't have to add anything and
		// can just multiply in place.
		if !self.use_mixed_poisson && fg_result.left > 1 {
			for _i in 0..fg_result.left - 1 {
				// We use this operation to take advantage of the MulAssign trait provided by the
				// CsVec<f64>I type in the sprs crate.
				result = &model.uniformized_matrix * &result;
				// Unfortunately, I don't believe that there is an optimizable version of AddAssign
				result = result + context.add_vec.clone();
			}
		} else if self.use_mixed_poisson {
			let epoch = context.epoch();
			// let add_scale = |a: f64, b: f64| a + b / epoch;
			for _idx in 1..first_iteration {
				// Multiply and then apply the scaling
				context.distribution = &model.uniformized_matrix * &context.distribution;
				result = &context.distribution + result.map(|elem| elem / epoch);
			}

			// scale values by total fox-glynn weight
			if fg_result.left > 0 {
				result.map_inplace(|val| *val * fg_result.total_weight);
			}
		}

		// In between the left and right fox glynn points, compute, scale and add results
		for idx in first_iteration..=fg_result.right {
			let weight = fg_result.weights[idx - fg_result.left];
			context.distribution =
				&model.uniformized_matrix * &context.distribution + &context.add_vec;
			context.distribution = &context.distribution + &result.map(|x| *x * weight);
		}

		// Scale the vector by total weight
		result.map_inplace(|val| *val / fg_result.total_weight);
		result
	}

	pub fn steady_state(&self, _context: &mut CheckContext) -> CsVec<f64> {
		unimplemented!();
	}

	/// This function computes until probabilities of the form Phi U Psi. It takes two parameters:
	/// 1. The template context, called `context`. This includes things like the precision, and the model, but
	/// things like the time bound may be altered. The user should not re-use the context after
	/// this function is called as it may modify it.
	/// 2. The time bound `bound`. If it is time-unbounded, then `self.steady_state()` will instead
	/// be called.
	pub fn compute_until(
		&self,
		context: &mut CheckContext,
		bound: Interval,
		phi_label_mask: BitVec,
		psi_label_mask: BitVec,
	) -> CsVec<f64> {
		let epoch = context.epoch();
		// Loop until we've reached the desired termination.
		let mut iteration_count = 0;
		let num_states = context.exit_rates.len();
		// TODO: figure out if there's a way to eliminate these .clone() calls
		loop {
			iteration_count += 1;
			// TODO: once everything is working, have this edited in-place
			let intermediate_result = match bound {
				// TODO: update this.
				Interval::TimeUnbounded => self.steady_state(context),
				Interval::TimeBoundedUpper(upper_bound) => {
					context.time_bound = upper_bound;
					context.create_add_vec(&phi_label_mask, &psi_label_mask);
					// Update relevant values based on the states which satisfy phi.
					context.update_relevant_states(&phi_label_mask);
					self.compute_transient(context)
				}
				Interval::StepBoundUpper(steps) => {
					// Here it works just like time bounded upper except rather than compute the number
					// of steps from the epochs we can just tell the checker the number of steps to
					// take, since it will be a DTMC.

					// Must be a DTMC and thus the epoch (the time in between steps) must be 1
					assert!(epoch == 1.0);
					// Update the context's bound with the number of steps.
					context.time_bound = steps as f64;
					context.create_add_vec(&phi_label_mask, &psi_label_mask);
					// Update relevant values based on the states which satisfy phi.
					context.update_relevant_states(&phi_label_mask);
					// Finally, compute transient probabilities
					self.compute_transient(context)
				}
				Interval::TimeBoundWindow(lower_bound, upper_bound) => {
					// Here there are two computations. For an interval of `Phi U [T,T'] Psi` we have
					// two probabilities:
					// (1) Stay in states |= Phi to time t, or
					// (2) Reaching a state |= Psi in time t' - t.
					// On pages 26-27 of *Stochastic Model Checking* (https://doi.org/10.1007/978-3-540-72522-0_6),
					// they note that if (2) is performed first, we can use it as an initial
					// distributiuon for computation (1).

					// First, we compute (2) given the initial distribution.
					context.time_bound = upper_bound - lower_bound;
					// TODO: one-step vector

					// Update relevant values based on the states which satisfy phi.
					context.update_relevant_states(&phi_label_mask);
					let distribution = self.compute_transient(context);
					// Now, compute (1) from (2).
					context.distribution = distribution;
					// Zero the distribution values for any state which does not satisfy psi
					context.zero_unsatisfying_states(&psi_label_mask);
					context.time_bound = lower_bound;
					// context.add_vec = CsVec::empty(num_states);
					context.create_add_vec(&phi_label_mask, &psi_label_mask);
					// Update relevant values based on the states which satisfy phi.
					context.update_relevant_states(&phi_label_mask);
					self.compute_transient(context)
				}
				Interval::TimeBoundedLower(lower_bound) => {
					// This works the same way as the time bounded window, except t' - t is still
					// unbounded and can be computed via the steady_state probability. Again, we
					// compute (2) first, but this time it's via steady state.

					// First, we compute (2) i.e., reaching a state that satisphies Psi in the
					// steady state (since t' - t is unbounded).
					let distribution = self.steady_state(context);
					// Update relevant values based on the states which satisfy psi.
					context.update_relevant_states(&psi_label_mask);
					// Like in the window time bound, now we compute (1) from (2)
					context.distribution = distribution;
					// Zero the distribution values for any state which does not satisfy psi
					context.zero_unsatisfying_states(&psi_label_mask);
					context.time_bound = lower_bound;
					// Update relevant values based on the states which satisfy phi.
					context.update_relevant_states(&phi_label_mask);
					self.compute_transient(context)
				}
			};
			if context.precision_reached(&intermediate_result) {
				debug!("Precision reached after {iteration_count} iterations");
				return intermediate_result;
			} else {
				debug!(
					"Precision not yet reached. Intermediate result: {:?}",
					intermediate_result
				);
			}
		}
	}

	/// Creates a "timeline" of distributions by the number of time epochs/steps, and the step
	/// size. The return value is a vector of distributions with their time-steps.
	pub fn distribution_timeline(
		&self,
		context: &mut CheckContext,
		num_epochs: usize,
		epoch_step: usize,
	) -> Vec<(f64, CsVec<f64>)> {
		let epoch = context.epoch();
		// Iteratively compute the intermediate distributions at the given granularity, collecting
		// the intermediate distributions into a vector and then returning them.
		(0..=num_epochs)
			.step_by(epoch_step)
			.map(|i| {
				context.time_bound = epoch * (i as f64);
				let distribution = self.compute_transient(context);
				context.distribution = distribution.clone();
				(context.time_bound, distribution)
			})
			.collect::<Vec<_>>()
	}

	/// A parallelized version of `distribution_timeline()`.
	pub fn distribution_timeline_concurrent(
		&self,
		context: &mut CheckContext,
		num_epochs: usize,
		epoch_step: usize,
		num_threads: usize,
	) -> Vec<(f64, CsVec<f64>)> {
		unimplemented!();
	}
}

impl Default for CslChecker {
	fn default() -> Self {
		Self::new(true, true)
	}
}

#[cfg(test)]
mod checker_tests {
	use super::*;

	#[test]
	fn some_test() {
		let mut checker = CslChecker::default();
		// TODO
	}
}
