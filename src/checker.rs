//! Floating-point assumptions: probabilities and rates are `f64`; comparisons use
//! exact equality only where values are structurally identical. Numerical results are
//! validated empirically against Storm/SciPy rather than formally proved.
use std::ops::BitAnd;
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
	#[allow(dead_code)]
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
		#[allow(dead_code)] discrete_time: bool,
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
		let _ = reuniformize_ma(&mut self.uniformized_matrix, old_rate, new_rate, deadlock);
	}
}

/// A struct that contains the program context for a model checker.
pub struct CheckContext {
	/// The (current) probability distribution over states.
	/// TODO: should this be a Vec rather than a sparse vector?
	distribution: CsVec<f64>,
	/// The model to be checked, including the uniformization matrix as well as the labelling and
	/// the epoch
	model_context: ExplicitModelContext,
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
			model_context: (*model_context).clone(),
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
		let epoch = self.model_context.epoch;
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
		self.model_context.epoch
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
		assert!(label_indecies.len() == self.model_context.labels.label_count());
		let state_count = self.model_context.state_count();
		let mut relevant = BitVec::with_capacity(state_count);
		(0..state_count).for_each(|_| relevant.push(false));
		assert!(relevant.len() == state_count);
		let mut relevant_state_count = 0;
		for (idx, &val) in self.distribution.iter() {
			if val != 0.0 && self.model_context.labels.state_has_labels(idx, label_indecies) {
				relevant.set(idx, true);
				relevant_state_count += 1;
			}
		}
		self.relevant_states = relevant;
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
		assert!(phi.len() == self.model_context.labels.label_count() && psi.len() == phi.len());
		let state_count = self.model_context.state_count();
		self.add_vec = CsVec::empty(state_count);
		let mut pns: BitVec = BitVec::with_capacity(state_count);
		(0..state_count).for_each(|_| pns.push(false));
		for idx in 0..state_count {
			if self.model_context.labels.state_has_labels(idx, phi)
				&& !self.model_context.labels.state_has_labels(idx, psi)
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
		assert!(label_indecies.len() == self.model_context.labels.label_count());
		let dim = self.model_context.state_count();
		let (new_idxes, new_data): (Vec<_>, Vec<_>) = self
			.distribution
			.iter()
			.filter_map(|(idx, &val)| {
				if self.model_context.labels.state_has_labels(idx, label_indecies) {
					Some((idx, val))
				} else {
					None
				}
			})
			.unzip();
		self.distribution = CsVec::new(dim, new_idxes, new_data);
	}
}

/// A CSL or PCTL model checker.
pub struct CslChecker {
	#[allow(dead_code)]
	use_mixed_poisson: bool,
}

impl CslChecker {
	pub fn new(use_mixed_poisson: bool) -> Self {
		Self { use_mixed_poisson }
	}

	/// Computes the transient probabilities for a given context and relevent values. The relevant
	/// values are the nonzero probabilities and the states who have the labels we care about.
	pub fn compute_transient(&self, context: &mut CheckContext) -> CsVec<f64> {
		let matrix = context.model_context.uniformized_matrix.clone();
		let epoch = context.model_context.epoch;
		let add_vec = context.add_vec.clone();

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
		// Use a single dense accumulator to avoid repeated sparse conversions.
		let mut acc = vec![0.0f64; dim];
		for k in 0..=(fg_result.right - fg_result.left) {
			let weight = fg_result.weights[k];
			for (idx, &val) in v.iter() {
				if idx < dim {
					acc[idx] += val * weight;
				}
			}

			// Advance for the next k.
			v = &matrix * &v + &add_vec;
		}

		// Normalize by the total (truncated) Poisson weight.
		for x in &mut acc {
			*x /= fg_result.total_weight;
		}
		let mut idxs = Vec::new();
		let mut vals = Vec::new();
		for (i, &x) in acc.iter().enumerate() {
			if x != 0.0 {
				idxs.push(i);
				vals.push(x);
			}
		}
		CsVec::new(dim, idxs, vals)
	}

	/// One-step (uniformized) probability that the next state satisfies `psi`.
	pub fn compute_next(&self, context: &mut CheckContext, psi_mask: BitVec) -> CsVec<f64> {
		let matrix = context.model_context.uniformized_matrix.clone();
		let labels = context.model_context.labels.clone();
		let n = matrix.rows();
		let mut is_psi = vec![false; n];
		for i in 0..n {
			is_psi[i] = if psi_mask.count_ones() == 0 {
				true
			} else {
				labels.state_has_labels(i, &psi_mask)
			};
		}
		let mut res = vec![0.0f64; n];
		for (val, (row, col)) in matrix.iter() {
			if col < n && is_psi[col] {
				res[row] += *val;
			}
		}
		let mut indices = Vec::new();
		let mut data = Vec::new();
		for (i, &x) in res.iter().enumerate() {
			if x != 0.0 {
				indices.push(i);
				data.push(x);
			}
		}
		CsVec::new(n, indices, data)
	}

	/// Probability that `phi` holds continuously over [0, time_bound].
	pub fn compute_globally(
		&self,
		context: &mut CheckContext,
		phi_mask: BitVec,
		time_bound: f64,
	) -> CsVec<f64> {
		context.time_bound = time_bound;
		let matrix = context.model_context.uniformized_matrix.clone();
		let epoch = context.model_context.epoch;
		let labels = context.model_context.labels.clone();
		let n = matrix.rows();
		let lambda = epoch * time_bound;
		if lambda == 0.0 {
			// At time 0, globally holds iff phi holds now.
			let mut indices = Vec::new();
			let mut data = Vec::new();
			for i in 0..n {
				let phi = if phi_mask.count_ones() == 0 {
					true
				} else {
					labels.state_has_labels(i, &phi_mask)
				};
				if phi {
					indices.push(i);
					data.push(1.0);
				}
			}
			return CsVec::new(n, indices, data);
		}
		let mut is_phi = vec![false; n];
		for i in 0..n {
			is_phi[i] = if phi_mask.count_ones() == 0 {
				true
			} else {
				labels.state_has_labels(i, &phi_mask)
			};
		}
		let fg_result = FoxGlynnBound::fox_glynn(lambda, context.epsilon);
		// target: phi states start with 1.
		let mut target = vec![0.0f64; n];
		for i in 0..n {
			if is_phi[i] {
				target[i] = 1.0;
			}
		}
		let apply = |v: &Vec<f64>| -> Vec<f64> {
			let mut res = vec![0.0f64; n];
			for (val, (row, col)) in matrix.iter() {
				if is_phi[row] && is_phi[col] {
					res[row] += *val * v[col];
				}
			}
			res
		};
		let mut v = target.clone();
		for _ in 0..fg_result.left {
			v = apply(&v);
		}
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

	/// Weak until: phi W psi = (phi U psi) + G phi.
	pub fn compute_weak_until(
		&self,
		context: &mut CheckContext,
		phi_mask: BitVec,
		psi_mask: BitVec,
		time_bound: f64,
	) -> CsVec<f64> {
		let until_res = self.compute_until(
			context,
			Interval::TimeBoundedUpper(time_bound),
			phi_mask.clone(),
			psi_mask.clone(),
		);
		let global_res = self.compute_globally(context, phi_mask, time_bound);
		let n = until_res.dim().max(global_res.dim());
		let mut res = vec![0.0f64; n];
		for (idx, &val) in until_res.iter() {
			if idx < n {
				res[idx] += val;
			}
		}
		for (idx, &val) in global_res.iter() {
			if idx < n {
				res[idx] += val;
				if res[idx] > 1.0 {
					res[idx] = 1.0;
				}
			}
		}
		let mut indices = Vec::new();
		let mut data = Vec::new();
		for (i, &x) in res.iter().enumerate() {
			if x != 0.0 {
				indices.push(i);
				data.push(x);
			}
		}
		CsVec::new(n, indices, data)
	}

	/// Expected cumulative reward up to `time_bound`, given state and transition rewards.
	/// Returns a vector indexed by starting state.
	pub fn compute_expected_reward(
		&self,
		context: &mut CheckContext,
		state_rewards: &sprs::CsVec<f64>,
		transition_rewards: &sprs::CsMat<f64>,
		time_bound: f64,
	) -> CsVec<f64> {
		let matrix = context.model_context.uniformized_matrix.clone();
		let epoch = context.model_context.epoch;
		let n = matrix.rows();
		context.time_bound = time_bound;

		// Build dense reward rate per state in the uniformized chain.
		let mut r = vec![0.0f64; n];
		for (idx, &val) in state_rewards.iter() {
			if idx < n {
				r[idx] += val;
			}
		}
		for i in 0..n {
			// transition rewards: epoch * sum_{j != i} P_ij * reward(i,j)
			let mut tran = 0.0f64;
			for j in 0..n {
				if i == j {
					continue;
				}
				let p = matrix.get(i, j).copied().unwrap_or(0.0);
				let rw = transition_rewards.get(i, j).copied().unwrap_or(0.0);
				tran += p * rw;
			}
			r[i] += epoch * tran;
		}

		let lambda = epoch * time_bound;
		if lambda <= 0.0 {
			return CsVec::new(n, Vec::new(), Vec::new());
		}
		let fg = FoxGlynnBound::fox_glynn(lambda, context.epsilon);
		let left = fg.left;
		let right = fg.right;
		// Normalized suffix probabilities P(N >= k+1).
		// For m in [left, right], normalized weight.
		let mut suffix = vec![0.0f64; right + 2];
		// suffix[m] = sum_{l=m}^{right} w_norm[l]
		let mut acc = 0.0f64;
		for m in (left..=right).rev() {
			let w = fg.weights[m - left] / fg.total_weight;
			acc += w;
			suffix[m] = acc;
		}
		// For k+1 <= left, suffix ~ 1.0; for k+1 > right, 0.

		let mut v = r.clone(); // P^0 * r
		let mut result = vec![0.0f64; n];
		for k in 0..=right {
			let prob_suffix = if k + 1 <= left {
				1.0f64
			} else if k + 1 > right {
				0.0f64
			} else {
				suffix[k + 1]
			};
			if prob_suffix > 0.0 {
				for i in 0..n {
					result[i] += prob_suffix * v[i];
				}
			}
			// advance v = P * v
			let mut next = vec![0.0f64; n];
			for (val, (row, col)) in matrix.iter() {
				if row < n && col < n {
					next[row] += *val * v[col];
				}
			}
			v = next;
		}
		for x in &mut result {
			*x /= epoch;
		}
		let mut indices = Vec::new();
		let mut data = Vec::new();
		for (i, &x) in result.iter().enumerate() {
			if x != 0.0 {
				indices.push(i);
				data.push(x);
			}
		}
		CsVec::new(n, indices, data)
	}

	pub fn steady_state(&self, context: &mut CheckContext) -> CsVec<f64> {
		let (dist, _iterations, _residual) = self.steady_state_with_diagnostics(context, 10000);
		dist
	}

	/// Steady-state computation with convergence diagnostics. Returns the
	/// distribution, the number of iterations performed, and the final L1
	/// residual.
	pub fn steady_state_with_diagnostics(
		&self,
		context: &mut CheckContext,
		max_iterations: usize,
	) -> (CsVec<f64>, usize, f64) {
		let matrix = context.model_context.uniformized_matrix.clone();
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
		// Precompute row sums once instead of O(n*nnz) per iteration.
		let mut row_sums = vec![0.0f64; n];
		for (val, (src, _dst)) in matrix.iter() {
			row_sums[src] += *val;
		}
		let deadlocks: Vec<bool> = row_sums.iter().map(|&s| s == 0.0).collect();
		let mut iterations = 0usize;
		let mut residual = 0.0f64;
		for _ in 0..max_iterations {
			iterations += 1;
			let mut next = vec![0.0f64; n];
			// Forward multiplication: next[dest] += P[src,dest] * dist[src].
			for (val, (src, dst)) in matrix.iter() {
				next[dst] += *val * dist[src];
			}
			// Account for deadlock rows (missing rows) as self-loops.
			for i in 0..n {
				if deadlocks[i] {
					next[i] += dist[i];
				}
			}
			let mut change = 0.0f64;
			for i in 0..n {
				change += (next[i] - dist[i]).abs();
				dist[i] = next[i];
			}
			residual = change;
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
		(CsVec::new(n, indices, data), iterations, residual)
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
				let matrix = context.model_context.uniformized_matrix.clone();
				let epoch = context.model_context.epoch;
				let labels = context.model_context.labels.clone();
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
				let matrix = context.model_context.uniformized_matrix.clone();
				let labels = context.model_context.labels.clone();
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
				// If phi holds in all psi states, an earlier hit still satisfies the
				// interval (phi remains true), so the result is just F(upper).
				// Otherwise the first hit must lie in [lower, upper].
				let phi_holds_in_psi = (0..phi_label_mask.len()).all(|i| {
					if psi_label_mask.get(i).as_deref().copied().unwrap_or(false) {
						phi_label_mask.get(i).as_deref().copied().unwrap_or(true)
					} else {
						true
					}
				}) || phi_label_mask.count_ones() == 0;
				if phi_holds_in_psi {
					self.compute_until(
						context,
						Interval::TimeBoundedUpper(upper_bound),
						phi_label_mask.clone(),
						psi_label_mask.clone(),
					)
				} else {
					let upper_res = self.compute_until(
						context,
						Interval::TimeBoundedUpper(upper_bound),
						phi_label_mask.clone(),
						psi_label_mask.clone(),
					);
					let lower_res = self.compute_until(
						context,
						Interval::TimeBoundedUpper(lower_bound),
						phi_label_mask.clone(),
						psi_label_mask.clone(),
					);
					let n = upper_res.dim().max(lower_res.dim());
					let mut res = vec![0.0f64; n];
					for (idx, &val) in upper_res.iter() {
						if idx < n {
							res[idx] += val;
						}
					}
					for (idx, &val) in lower_res.iter() {
						if idx < n {
							res[idx] -= val;
						}
					}
					let mut indices = Vec::new();
					let mut data = Vec::new();
					for (i, &x) in res.iter().enumerate() {
						if x != 0.0 {
							indices.push(i);
							data.push(x);
						}
					}
					CsVec::new(n, indices, data)
				}
			}
			Interval::TimeBoundedLower(lower_bound) => {
				let phi_holds_in_psi = phi_label_mask.count_ones() == 0
					|| (0..phi_label_mask.len()).all(|i| {
						if psi_label_mask.get(i).as_deref().copied().unwrap_or(false) {
							phi_label_mask.get(i).as_deref().copied().unwrap_or(true)
						} else {
							true
						}
					});
				if phi_holds_in_psi {
					self.compute_until(
						context,
						Interval::TimeBoundedUpper(1000.0),
						phi_label_mask.clone(),
						psi_label_mask.clone(),
					)
				} else {
					let unbounded_res = self.compute_until(
						context,
						Interval::TimeBoundedUpper(1000.0),
						phi_label_mask.clone(),
						psi_label_mask.clone(),
					);
					let lower_res = self.compute_until(
						context,
						Interval::TimeBoundedUpper(lower_bound),
						phi_label_mask.clone(),
						psi_label_mask.clone(),
					);
					let n = unbounded_res.dim().max(lower_res.dim());
					let mut res = vec![0.0f64; n];
					for (idx, &val) in unbounded_res.iter() {
						if idx < n {
							res[idx] += val;
						}
					}
					for (idx, &val) in lower_res.iter() {
						if idx < n {
							res[idx] -= val;
						}
					}
					let mut indices = Vec::new();
					let mut data = Vec::new();
					for (i, &x) in res.iter().enumerate() {
						if x != 0.0 {
							indices.push(i);
							data.push(x);
						}
					}
					CsVec::new(n, indices, data)
				}
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
	fn time_bound_window_absorption() {
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
		let expected = 1.0 - (-1.0f64).exp();
		assert!((p - expected).abs() < 1e-5, "window got {} expected {}", p, expected);
	}

	#[test]
	fn time_bounded_lower_absorption() {
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
		let expected = 1.0;
		assert!((p - expected).abs() < 1e-5, "lower got {} expected {}", p, expected);
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
