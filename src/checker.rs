use crate::matrix::*;
use crate::poisson::FoxGlynnBound;
use crate::*;
use num::{
	Zero,
	traits::{Bounded, real::Real},
};
use sprs::{CsMat, CsVec};

use self::model::TimeBound;

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
	EntryType: CheckableNumber,
{
	distribution: CsVec<EntryType>,
	uniformized_matrix: CsMat<EntryType>,
	time_bound: EntryType,
	epoch: EntryType,
	epsilon: EntryType,
	/// The states for which we perform model checking
	checked_values: bitvector::BitVector,
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
		let mut lambda = context.epoch * context.time_bound;
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
		// let mut result = if first_iteration == 0 {
		// 	first_iteration += 1;
		// 	let res = context.distribution.clone();
		// 	res.
		// } else {
		//
		// }
		unimplemented!();
	}

	pub fn steady_state(&self, context: &mut CheckContext<EntryType>) -> CsVec<EntryType> {
		unimplemented!();
	}

	pub fn compute_until(
		&self,
		context: &mut CheckContext<EntryType>,
		bound: TimeBound<EntryType>,
	) -> CsVec<EntryType> {
		match bound {
			TimeBound::TimeUnbounded => self.steady_state(context),
			TimeBound::TimeBoundedUpper(upper_bound) => {
				unimplemented!();
			}
			TimeBound::TimeBoundWindow(lower_bound, upper_bound) => {
				unimplemented!();
			}
			TimeBound::TimeBoundedLower(lower_bound) => {
				unimplemented!();
			}
		}
	}
}
