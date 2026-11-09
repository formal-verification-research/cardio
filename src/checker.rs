use std::ops::BitAnd;
use std::sync::RwLock;

use crate::labels;
use crate::matrix::*;
use crate::poisson::FoxGlynnBound;
use crate::property::Interval;

use bitvec::prelude::*;
use log::*;

use sprs::{CsMat, CsVec};

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
	/// This is also the reciprocal of the uniformization rate if a CTMC
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

	pub fn reuniformize(&mut self, new_rate: f64, deadlock: &BitVec) {
		let old_rate = 1.0 / self.epoch;
		debug!("Reuniformizing {old_rate} -> {new_rate}");
		reuniformize_ma(&mut self.uniformized_matrix, old_rate, new_rate, deadlock);
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
		_checked_values: BitVec,
		_deadlock: BitVec,
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
			exit_rates,
			add_vec: CsVec::empty(num_states), // TODO: I think this is where the error
			// lies. This should be the one-step vector
			relevant_states,
			precision,
		}
	}

	pub fn build_one_step(&mut self, non_sat_states: &BitVec) {
		let epoch = self.model_context.read().unwrap().epoch;
		assert!(epoch != 0.0);
		debug!("The epoch for this model is {epoch}");
		// Create a bit vector representing the states which are both non-satisfying AND
		// are relevant
		let ns_rel = non_sat_states.clone().bitand(&self.relevant_states);
		for (idx, val) in self.exit_rates.iter().enumerate() {
			if *val != 0.0 && *ns_rel.get(idx).as_deref().unwrap() {
				// debug!("Adding val {val} / {epoch} to index {idx}");
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
			if val != 0.0 && model.labels.state_has_labels(idx, &label_indecies) {
				self.relevant_states.set(idx, true);
				relevant_state_count += 1;
			}
		}
		debug!("Relevant state count: {relevant_state_count}");
	}

	pub fn uniformization_rate(&self) -> f64 {
		self.relevant_states
			.iter_ones()
			.map(|idx| self.distribution[idx])
			.reduce(|a, b| {
				// Sanity checks we can maybe remove if everything works
				debug_assert!(a.is_finite() && b.is_finite());
				debug_assert!(!a.is_nan() && !b.is_nan());
				// Get the highest
				if a > b { a } else { b }
			})
			.unwrap() * 1.2
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
		for idx in 0..state_count {
			if model.labels.state_has_labels(idx, &phi) && !model.labels.state_has_labels(idx, &psi)
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
		let dim = model.state_count();
		self.distribution = CsVec::new(dim, new_idxes, new_data);
	}
}

/// A CSL or PCTL model checker.
pub struct CslChecker {
	use_mixed_poisson: bool,
}

impl CslChecker {
	pub fn new(use_mixed_poisson: bool) -> Self {
		Self { use_mixed_poisson }
	}

	/// Computes the transient probabilities for a given context and relevent values. The relevant
	/// values are the nonzero probabilities and the states who have the labels we care about.
	pub fn compute_transient(&self, context: &mut CheckContext) -> CsVec<f64> {
		let model = context.model_context.read().unwrap();
		let matrix = model.uniformized_matrix.clone();
		let epoch = model.epoch;
		let add_vec = context.add_vec.clone();
		drop(model);

		let lambda = epoch * context.time_bound;
		// Return the initial distribution if no time passes.
		if lambda == 0.0 {
			return context.distribution.clone();
		}
		if context.epsilon <= 1e-20 {
			warn!("Warning: extremely low truncation error may lead to numerical instability.");
		}

		let fg_result = FoxGlynnBound::fox_glynn(lambda, context.epsilon);
		let dim = context.distribution.dim();

		// Skip the (negligible) left tail by advancing the working vector.
		let mut v = context.distribution.clone();
		for _ in 0..fg_result.left {
			v = &matrix * &v + &add_vec;
		}

		// Accumulate Poisson-weighted contributions for k = left..=right.
		let mut result = CsVec::empty(dim);
		for k in 0..=(fg_result.right - fg_result.left) {
			let weight = fg_result.weights[k];
			let scaled = v.map(|elem| *elem * weight);
			// Add scaled into result.
			// CsVec does not implement AddAssign for scaled vectors directly in all
			// versions, so accumulate via a dense temporary.
			let mut tmp = vec![0.0f64; dim];
			for (idx, &val) in result.iter() {
				if idx < dim {
					tmp[idx] += val;
				}
			}
			for (idx, &val) in scaled.iter() {
				if idx < dim {
					tmp[idx] += val;
				}
			}
			let mut idxs = Vec::new();
			let mut vals = Vec::new();
			for (i, &x) in tmp.iter().enumerate() {
				if x != 0.0 {
					idxs.push(i);
					vals.push(x);
				}
			}
			result = CsVec::new(dim, idxs, vals);

			// Advance for the next k.
			v = &matrix * &v + &add_vec;
		}

		// Normalize by the total (truncated) Poisson weight.
		result.map_inplace(|val| *val / fg_result.total_weight);
		result
	}

	pub fn steady_state(&self, context: &mut CheckContext) -> CsVec<f64> {
		let model = context.model_context.read().unwrap();
		let matrix = model.uniformized_matrix.clone();
		drop(model);
		let n = matrix.cols();

		// Start from the current distribution if available, otherwise uniform.
		let mut dist = vec![0.0f64; n];
		if context.distribution.nnz() > 0 {
			for (idx, &val) in context.distribution.iter() {
				if idx < n {
					dist[idx] = val;
				}
			}
		} else {
			for x in &mut dist {
				*x = 1.0 / (n as f64);
			}
		}

		let tol = context.epsilon.max(1e-12);
		for _ in 0..10000 {
			let mut next = vec![0.0f64; n];
			// Forward multiplication: next[dest] += P[src,dest] * dist[src].
			for (val, (src, dst)) in matrix.iter() {
				next[dst] += *val * dist[src];
			}
			// Account for deadlock rows (missing rows) as self-loops.
			for i in 0..n {
				let mut row_sum = 0.0f64;
				for (val, (src, _dst)) in matrix.iter() {
					if src == i {
						row_sum += *val;
					}
				}
				if row_sum == 0.0 {
					next[i] += dist[i];
				}
			}
			let mut change = 0.0f64;
			for i in 0..n {
				change += (next[i] - dist[i]).abs();
				dist[i] = next[i];
			}
			if change < tol {
				break;
			}
		}

		let mut indices = Vec::new();
		let mut data = Vec::new();
		for (i, &x) in dist.iter().enumerate() {
			if x != 0.0 {
				indices.push(i);
				data.push(x);
			}
		}
		CsVec::new(n, indices, data)
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
		// Single pass: the precision loop is handled by the caller via epsilon.
		let iteration_count = 1;
		// TODO: figure out if there's a way to eliminate these .clone() calls
		let intermediate_result = match bound {
			// TODO: update this.
			Interval::TimeUnbounded => self.steady_state(context),
			Interval::TimeBoundedUpper(upper_bound) => {
				context.time_bound = upper_bound;
				let model = context.model_context.read().unwrap();
				let matrix = model.uniformized_matrix.clone();
				let epoch = model.epoch;
				let labels = model.labels.clone();
				drop(model);
				let n = matrix.rows();
				let lambda = epoch * upper_bound;
				let fg_result = FoxGlynnBound::fox_glynn(lambda, context.epsilon);

				// Determine phi/psi membership for each state.
				let mut is_phi = vec![false; n];
				let mut is_psi = vec![false; n];
				for i in 0..n {
					is_phi[i] = if phi_label_mask.count_ones() == 0 {
						true
					} else {
						labels.state_has_labels(i, &phi_label_mask)
					};
					is_psi[i] = labels.state_has_labels(i, &psi_label_mask);
				}

				// Target vector: psi states.
				let mut target = vec![0.0f64; n];
				for i in 0..n {
					if is_psi[i] {
						target[i] = 1.0;
					}
				}

				// Apply the absorbing (phi/psi) uniformized matrix to a vector.
				let apply = |v: &Vec<f64>| -> Vec<f64> {
					let mut res = vec![0.0f64; n];
					// Absorbing rows: psi or !phi -> identity.
					for i in 0..n {
						if is_psi[i] || !is_phi[i] {
							res[i] = v[i];
						}
					}
					// phi & !psi rows: original matrix row.
					for (val, (row, col)) in matrix.iter() {
						if is_phi[row] && !is_psi[row] {
							res[row] += *val * v[col];
						}
					}
					res
				};

				// Skip the (negligible) left tail.
				let mut v = target.clone();
				for _ in 0..fg_result.left {
					v = apply(&v);
				}

				// Accumulate Poisson-weighted contributions.
				let mut acc = vec![0.0f64; n];
				for k in 0..=(fg_result.right - fg_result.left) {
					let weight = fg_result.weights[k];
					for i in 0..n {
						acc[i] += weight * v[i];
					}
					v = apply(&v);
				}
				for x in &mut acc {
					*x /= fg_result.total_weight;
				}

				// Convert to sparse vector.
				let mut indices = Vec::new();
				let mut data = Vec::new();
				for (i, &x) in acc.iter().enumerate() {
					if x != 0.0 {
						indices.push(i);
						data.push(x);
					}
				}
				CsVec::new(n, indices, data)
			}
			Interval::StepBoundUpper(steps) => {
				assert!(epoch == 1.0);
				context.time_bound = steps as f64;
				// Fall through by computing directly:
				let _upper_bound = steps as f64;
				let model = context.model_context.read().unwrap();
				let matrix = model.uniformized_matrix.clone();
				let labels = model.labels.clone();
				drop(model);
				let n = matrix.rows();
				// For DTMC, treat as one step per unit; lambda not needed for direct powering.
				let mut is_phi = vec![false; n];
				let mut is_psi = vec![false; n];
				for i in 0..n {
					is_phi[i] = if phi_label_mask.count_ones() == 0 {
						true
					} else {
						labels.state_has_labels(i, &phi_label_mask)
					};
					is_psi[i] = labels.state_has_labels(i, &psi_label_mask);
				}
				let mut target = vec![0.0f64; n];
				for i in 0..n {
					if is_psi[i] {
						target[i] = 1.0;
					}
				}
				let apply = |v: &Vec<f64>| -> Vec<f64> {
					let mut res = vec![0.0f64; n];
					for i in 0..n {
						if is_psi[i] || !is_phi[i] {
							res[i] = v[i];
						}
					}
					for (val, (row, col)) in matrix.iter() {
						if is_phi[row] && !is_psi[row] {
							res[row] += *val * v[col];
						}
					}
					res
				};
				let mut v = target.clone();
				for _ in 0..steps {
					v = apply(&v);
				}
				let mut indices = Vec::new();
				let mut data = Vec::new();
				for (i, &x) in v.iter().enumerate() {
					if x != 0.0 {
						indices.push(i);
						data.push(x);
					}
				}
				CsVec::new(n, indices, data)
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
		debug!("Returning after {iteration_count} iterations");
		intermediate_result
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
		_num_threads: usize,
	) -> Vec<(f64, CsVec<f64>)> {
		// A simple sequential fallback. True parallel execution would require
		// independent contexts per thread; this keeps behaviour consistent with
		// `distribution_timeline` while preserving the API.
		self.distribution_timeline(context, num_epochs, epoch_step)
	}
}

impl Default for CslChecker {
	fn default() -> Self {
		Self::new(true)
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

#[cfg(test)]
mod until_tests {
	use super::*;
	use crate::labels::Labels;
	use crate::matrix::{OptimalSprsMatBuilder, SprsMatBuilder};

	#[test]
	fn simple_absorption_probability() {
		let mut mb = OptimalSprsMatBuilder::new();
		mb.insert(1, 0, 1.0);
		let mut labels = Labels::with_abs_and_sat();
		let mc = mb.to_model_context(&labels, false);
		let phi_mask = labels.create_label_bitmask(vec![]);
		let psi_mask = labels.create_label_bitmask(vec!["absorbing".to_string()]);
		let relevant = labels.create_relevant(&phi_mask, mc.state_count());
		let deadlock = mb.get_deadlocks();
		let exit_rates = mb.row_sum_vec();
		let mut ctx = CheckContext::initialize_with_abs(
			&mc,
			1.0,
			1e-9,
			relevant.clone(),
			relevant.clone(),
			deadlock,
			exit_rates,
		);
		let checker = CslChecker::default();
		let interval = Interval::TimeBoundedUpper(1.0);
		let dist = checker.compute_until(&mut ctx, interval, phi_mask, psi_mask);
		let p = dist.get(1).copied().unwrap_or(0.0);
		assert!((p - 0.6321205588285577).abs() < 1e-6, "got {}", p);
	}

	#[test]
	fn time_bound_window_smoke() {
		let mut mb = OptimalSprsMatBuilder::new();
		mb.insert(1, 0, 1.0);
		let mut labels = Labels::with_abs_and_sat();
		let mc = mb.to_model_context(&labels, false);
		let phi_mask = labels.create_label_bitmask(vec![]);
		let psi_mask = labels.create_label_bitmask(vec!["absorbing".to_string()]);
		let relevant = labels.create_relevant(&phi_mask, mc.state_count());
		let deadlock = mb.get_deadlocks();
		let exit_rates = mb.row_sum_vec();
		let mut ctx = CheckContext::initialize_with_abs(
			&mc,
			1.0,
			1e-9,
			relevant.clone(),
			relevant.clone(),
			deadlock,
			exit_rates,
		);
		let checker = CslChecker::default();
		let interval = Interval::TimeBoundWindow(0.2, 1.0);
		let dist = checker.compute_until(&mut ctx, interval, phi_mask, psi_mask);
		let p = dist.get(1).copied().unwrap_or(0.0);
		assert!(p.is_finite(), "got {}", p);
	}

	#[test]
	fn time_bounded_lower_smoke() {
		let mut mb = OptimalSprsMatBuilder::new();
		mb.insert(1, 0, 1.0);
		let mut labels = Labels::with_abs_and_sat();
		let mc = mb.to_model_context(&labels, false);
		let phi_mask = labels.create_label_bitmask(vec![]);
		let psi_mask = labels.create_label_bitmask(vec!["absorbing".to_string()]);
		let relevant = labels.create_relevant(&phi_mask, mc.state_count());
		let deadlock = mb.get_deadlocks();
		let exit_rates = mb.row_sum_vec();
		let mut ctx = CheckContext::initialize_with_abs(
			&mc,
			1.0,
			1e-9,
			relevant.clone(),
			relevant.clone(),
			deadlock,
			exit_rates,
		);
		let checker = CslChecker::default();
		let interval = Interval::TimeBoundedLower(0.5);
		let dist = checker.compute_until(&mut ctx, interval, phi_mask, psi_mask);
		let p = dist.get(1).copied().unwrap_or(0.0);
		assert!(p.is_finite(), "got {}", p);
	}
}
