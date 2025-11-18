use crate::matrix::*;
use crate::poisson::FoxGlynnBound;
use crate::*;
use bitvec::prelude::*;
use num::traits::{real::Real, Bounded};
use sprs::{CsMat, CsVec};

use self::property::Interval;

/// A CTMC transition matrix
pub trait CtmcTransMat {
	fn uniformize(rate_mat: Self) -> Self;
}

impl<EntryType> CtmcTransMat for sprs::CsMat<EntryType>
where
	EntryType: num::Num + Clone,
{
	fn uniformize(rate_mat: Self) -> Self {
		unimplemented!();
	}
}

pub struct CheckContext<EntryType>
where
	EntryType: CheckableNumber + std::convert::From<f64>,
{
	discrete_time: bool,
	distribution: CsVec<EntryType>,
	uniformized_matrix: CsMat<EntryType>,
	time_bound: EntryType,
	epoch: EntryType,
	epsilon: EntryType,
	/// The states for which we perform model checking
	checked_values: BitVec,
	/// The value we add during the self multiplication
	add_vec: CsVec<EntryType>,
	/// The states for which precision is relevant
	relevant_states: BitVec,
	/// The precision to which we check
	precision: EntryType,
}

impl<EntryType> CheckContext<EntryType>
where
	EntryType: CheckableNumber + std::convert::From<f64>,
{
	/// If there are states for which the precision is relevant.
	pub fn has_relevant_states(&self) -> bool {
		!self.relevant_states.is_empty()
	}

	/// If any state in the distribution has a nonzero value
	pub fn has_reachable_states(&self) -> bool {
		self.distribution.nnz() > 0
	}

	/// Checks to see if we've reached the desired precision for all of the relevant states. This
	/// function also updates the epsilon value thus it takes a `&mut self`.
	pub fn precision_reached(&mut self, intermediate_result: &CsVec<EntryType>) -> bool {
		// The element for new_epsilon when the result is zero
		let zero_epsilon = self.epsilon * EntryType::from(0.1);
		// Iterate over all relevant state indecies and check if their precision
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
}

pub struct CslChecker<EntryType>
where
	EntryType: num::Num + Clone,
{
	qualitative: bool,
	use_mixed_poisson: bool,
	placeholder: EntryType,
}

impl<EntryType> CslChecker<EntryType>
where
	EntryType: CheckableNumber
		+ Bounded
		+ std::convert::From<f64>
		+ std::convert::From<usize>
		+ std::convert::From<isize>
		+ Real,
	usize: From<EntryType>,
	isize: From<EntryType>,
	f64: From<EntryType>,
{
	pub fn compute_transient(&self, context: &mut CheckContext<EntryType>) -> CsVec<EntryType> {
		let lambda = context.epoch * context.time_bound;
		// Return the initial distribution if no epochs pass.
		if lambda.is_zero() {
			return context.distribution.clone();
		}
		let mut fg_result = FoxGlynnBound::fox_glynn(lambda, context.epsilon);

		if self.use_mixed_poisson {
			let (mut left, mut right): (usize, usize) = (0, fg_result.weights.len() - 1);
			let (mut sum_left, mut sum_right) = (EntryType::zero(), EntryType::zero());
			while left <= right {
				if fg_result.weights[left] < fg_result.weights[right] {
					sum_left += fg_result.weights[left];
					fg_result.weights[left] = (fg_result.total_weight - sum_left) / context.epsilon;
					left += 1;
				} else {
					let right_weight = fg_result.weights[right];
					fg_result.weights[right] = sum_right / context.epoch;
					sum_right += right_weight;
					if right == 0 {
						break;
					} else {
						right -= 1;
					}
				}
			}
			// TODO: check for numerical instability
		}

		// Create the result vector
		let mut first_iteration = fg_result.left;
		let mut result = if first_iteration == 0 {
			first_iteration += 1;
			context.distribution.clone()
		// The initial result must be uniformized if we are in continuous time and using mixed
		// poisson probabilities.
		} else if self.use_mixed_poisson && !context.discrete_time {
			context.distribution.map(|&elem| elem / context.epoch)
		} else {
			CsVec::empty(context.distribution.dim())
		};

		// An optimization shamelessly stolen from storm: if we don't have to use mixed poisson
		// probabilities and our left fox-glynn result is > 1, we don't have to add anything and
		// can just multiply in place.
		if !self.use_mixed_poisson && fg_result.left > 1 {
			for i in 0..fg_result.left - 1 {
				// We use this operation to take advantage of the MulAssign trait provided by the
				// CsVecI type in the sprs crate.
				result = context.uniformized_matrix * result;
				// Unfortunately, I don't believe that there is an optimizable version of AddAssign
				result = result + context.add_vec;
			}
		} else if self.use_mixed_poisson {
			// If using mixed poisson probabilities we have to scale the vector by the
			// uniformization rate and add the values each iteration.
			for i in 0..fg_result.left - 1 {
				context.distribution = context.uniformized_matrix * context.distribution;
				context.distribution += result.map(|val| *val / context.epoch);
			}

			// scale values by total fox-glynn weight
			if fg_result.left > 0 {
				result.map_inplace(|val| *val * fg_result.total_weight);
			}
		}

		// In between the left and right fox glynn points, compute, scale and add results
		for idx in first_iteration..=fg_result.right {
			let weight = fg_result.weights[idx - fg_result.left];
			context.distribution *= context.uniformized_matrix;
			context.distribution += result.map(|x| *x * weight);
		}

		// Scale the vector by total weight
		result.map_inplace(|val| *val / fg_result.total_weight);
		result
	}

	pub fn steady_state(&self, context: &mut CheckContext<EntryType>) -> CsVec<EntryType> {
		unimplemented!();
	}

	pub fn compute_until(
		&self,
		context: &mut CheckContext<EntryType>,
		bound: Interval<EntryType>,
	) -> CsVec<EntryType> {
		loop {
			let intermediate_result = match bound {
				Interval::TimeUnbounded => self.steady_state(context),
				Interval::TimeBoundedUpper(upper_bound) => {
					unimplemented!();
				}
				Interval::StepBoundUpper(steps) => {
					// Here it works just like time bounded upper except rather than compute the number
					// of steps from the epochs we can just tell the checker the number of steps to
					// take, since it will be a DTMC.
					unimplemented!();
				}
				Interval::TimeBoundWindow(lower_bound, upper_bound) => {
					// Here there are two computations. For an interval of `Phi U [T,T'] Psi` we have
					// two probabilities:
					// (1) Stay in states |= Phi to time t, or
					// (2) Reaching a state |= Psi in time t' - t.
					// On pages 26-27 of *Stochastic Model Checking* (https://doi.org/10.1007/978-3-540-72522-0_6),
					// they note that if (2) is performed first, we can use it as an initial
					// distributiuon for
					unimplemented!();
				}
				Interval::TimeBoundedLower(lower_bound) => {
					unimplemented!();
				}
			};
			if context.precision_reached(&intermediate_result) {
				return intermediate_result;
			}
		}
	}
}
