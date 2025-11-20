// Poisson sum convergence calculation stuff

use std::f64::{self, consts::PI};

use num::{Bounded, traits::real::Real};

use crate::matrix;

/// A bound from a Fox-Glynn computation.
pub struct FoxGlynnBound<ValueType>
where
	ValueType: matrix::CheckableNumber + Bounded + Real,
{
	pub left: usize,
	pub right: usize,
	pub total_weight: ValueType,
	pub weights: Vec<ValueType>,
}

impl<ValueType> Default for FoxGlynnBound<ValueType>
where
	ValueType: matrix::CheckableNumber + Bounded + Real,
{
	fn default() -> Self {
		Self {
			left: 0,
			right: 0,
			total_weight: ValueType::zero(),
			weights: Vec::new(),
		}
	}
}

impl<ValueType> FoxGlynnBound<ValueType>
where
	ValueType: matrix::CheckableNumber + Bounded + Real,
{
	/// Finds the left and right truncation points for Fox-Glynn as described in
	/// https://doi.org/10.1145/42404.42409. During the implementation of this function, we heavily
	/// referenced the `numeric` module of the [Storm](https://github.com/moves-rwth/storm) model
	/// checker, and make optimizations similar to them.
	///
	/// Note that this is the `FINDER` subroutine described in Section 3 of the Fox-Glynn paper.
	fn fg_find(lambda: ValueType, epsilon: ValueType) -> Self {
		let one = ValueType::one();
		let p5 = ValueType::from_f64(0.5).unwrap();
		// Start by setting up constants and variables
		let (mut tau, omega) = (
			<ValueType as Bounded>::min_value(),
			<ValueType as Bounded>::max_value(),
		);
		let root2pi = ValueType::from_f64((2.0 * PI).sqrt()).unwrap();
		// Error bound only uses epsilon * root2pi
		let mut er2pi = epsilon * root2pi;

		// Create the left and right bounds, which may be negative. Initialize them to zero
		let (mut left, mut right): (isize, isize) = (0, 0);

		// Like the main `fox_glynn` method, we get the mid-point from the value of lambda
		let m = lambda.to_usize().unwrap();

		// Because we only use tau in underflow checks, we can log it first.
		let tlog = tau.log2();

		// First, compute the left truncation point
		if m < 25 {
			// The left truncation point is zero for lambda midpoint is < 25
			left = 0;

			// Warn underflow if lambda is below 25.
			if -lambda <= tlog {
				eprintln!("Fox-Glynn underflow."); // TODO: better error message
			}
		} else {
			// We actually have to look for the left truncation point iteratively if m >= 25

			// First, compute a couple constants that are needed.
			let b = (one + lambda.recip())
				* (lambda.recip() * ValueType::from_f64(0.125).unwrap()).exp();
			let root_lmbda = lambda.sqrt();
			let mut k: ValueType = ValueType::from_usize(4).unwrap();

			loop {
				// First, compute a candidate for `left`.
				left = m as isize - (k * root_lmbda + p5).ceil().to_isize().unwrap();

				// If the truncation point is negative, then make it zero and terminate the loop.
				if left.is_negative() {
					left = 0;
					break;
				}

				// It's a good thing we reference the Storm code in this implementation, since they
				// correctly point out that Fox-Glynn mixes up Phi and 1 - Phi in Propositions 2-4.
				let max_err = b * (-k.powi(2) / p5).exp() / k;

				// If the left-hand error is relatively small, loosen the requirements on the right
				// hand side and do not bound the left-hand side any farther.
				if max_err + max_err <= epsilon {
					er2pi -= max_err;
					break;
				}

				// Increment k
				k += one;
			}

			// If the loop has terminated, the left bound has been found.
		}

		// Now we just have to compute the right bound. However, first we must compute a couple
		// of constants and update the epsilon value.

		let mut k: ValueType = ValueType::from_i8(4).unwrap();
		// Fox-Glynn draws a line at lambda = 400. If below, then set lambda at 400, and if
		// not, use the higher value to compute the right bound.
		let (lambda_max, m_max): (ValueType, isize) = if m < 400 {
			let magic_const = ValueType::from_f64(0.662608824988162441697980).unwrap();
			er2pi *= magic_const;
			(ValueType::from_usize(400).unwrap(), 400)
		} else {
			let magic_const = ValueType::from_f64(0.664265347050632847802225).unwrap();
			// This allows us to prevent multiple casting.
			er2pi *= (one - (lambda + one).recip()) * magic_const;
			(lambda, m as isize)
		};

		// Like Storm, we terminate by the error, which provides more precise results but doesn't
		// include the stop condition in the Fox-Glynn paper. Again, we have an unterminating loop
		// with break statements.
		loop {
			// The magic constants in the above if-statement come from the fact that we don't have
			// to compute the extra multiplier factor here.
			if er2pi * k >= (-k.powi(2) * p5).exp() {
				break;
			}
			// Increment k
			k += one;
		}
		// Compute the right bound and determine if it's reliable.
		right = m_max
			+ (k * (lambda_max + lambda_max).sqrt() + p5)
				.ceil()
				.to_isize()
				.unwrap();
		let reliability_bound = m_max + ((lambda_max + ValueType::one()) * p5).to_isize().unwrap();
		if right > reliability_bound {
			eprintln!(
				"Right bound unreliable! ({0} > {1})",
				right, reliability_bound
			);
		}

		// The right bound has now been found, so initialize the weights.
		let weights_count = (right - left + 1) as usize;
		let mut res = Self {
			left: left as usize,
			right: right as usize,
			weights: Vec::with_capacity(weights_count),
			..Default::default()
		};
		// Although we've reserved the capacity, we actually have to make the vector the correct
		// size. We'll set the uninitialized values to zero...
		res.weights.resize(weights_count, ValueType::zero());
		// ...but we do have one slot we know the value for.
		res.weights[m - res.left] = omega
			/ (ValueType::from_usize(res.right - res.left).unwrap()
				* ValueType::from_f64(1.0e10).unwrap());

		// We have one more underflow check we have to perform. This underflow check will be
		// performed in f64 rather than valuetype since this is a numeric method.
		if m >= 25 {
			// We compare to tau - ln(W[m]) so change in-place
			tau -= res.weights[m - res.left].ln();

			let i = m as isize - res.left as isize;
			// Cast it to ValueType early
			let ir = i as f64;
			let lambda_f64 = lambda.to_f64().unwrap();
			// Another magic constant stolen from Storm. This one comes from the fact that
			// -1 - 1 / (12 * 25) - ln(2 * pi) * 0.5 is roughly equal to -1.922272.
			let magic_const = -1.922272;
			// Only do one cast to f64
			let lnc_m = magic_const - (m as f64).ln() * 0.5;

			let numeric_result = if left >= i {
				// Fox-Glynn proposition 6
				lnc_m - ir * (ir + 1.0) * (0.5 + (ir + ir + 1.0) / (6.0 * lambda_f64)) / lambda_f64
			} else if res.left != 0 {
				// Fox-Glynn Corollary 4 (iii)
				// Proposition 6 (ii)
				let num_res_alt = lnc_m + ir * (1.0 - ir / ((m + 1) as f64));
				num_res_alt.max(-lambda_f64)
			} else {
				// Proposition 6 (ii)
				-lambda_f64
			};

			let tau_f64 = tau.to_f64().unwrap();

			if numeric_result <= tau_f64 {
				eprintln!("Underflow in lambda >= 25!");
			}

			// Right truncation point underflow check
			if m >= 400 {
				// Proposition 5
				let i = res.right as isize - m as isize;
				let ir = i as f64;
				let numeric_result = lnc_m - ir * (ir + 1.0) / (2.0 * lambda_f64);
				if numeric_result <= tau_f64 {
					eprintln!("Underflow in lambda >= 400!");
				}
			}
		}
		// Return the result
		res
	}

	/// The publicly accessible Fox-Glynn function, which performs the convergence as described in
	/// the paper at [this DOI](https://doi.org/10.1145/42404.42409).
	pub fn fox_glynn(lambda: ValueType, epsilon: ValueType) -> Self {
		assert!(lambda.is_positive());
		// Start the mid point at the the current value of `lambda`.
		let m = lambda.to_usize().unwrap();

		let tau = <ValueType as Bounded>::min_value();
		let mut res = Self::fg_find(lambda, epsilon);
		let mut t = res.right - res.left;

		// The left side of the weights array is easy to fill in.
		for j in (1..=m - res.left).rev() {
			res.weights[j - 1] =
				ValueType::from_usize(j + res.left).unwrap() / lambda * res.weights[j];
		}

		// Now we fill in the right side of the array. If lambda < 400, we have a separate case
		// than if it's >= 400. The 400 number may seem like a magic number, but it is explained in
		// Section 3 of the Fox-Glynn paper. Specifically, in Corollary 1, the restrictions on
		// lambda naturally derive the mid-point being 400.
		if m >= 400 {
			// No danger of underflow, so just compute the weights
			for j in (m - res.left)..t {
				res.weights[j + 1] =
					lambda / ValueType::from_usize(j + 1 + res.left).unwrap() * res.weights[j]
			}
		} else {
			// Make sure we haven't underflowed
			if res.right <= 600 {
				eprintln!(
					"[Cardio: WARNING] Because {0} <= 600, underflow may occur.",
					res.right
				)
			}

			// Fill the rest of the array
			for j in (m - res.left)..t {
				let q = lambda / ValueType::from_usize(j + 1 + res.left).unwrap();
				if res.weights[j] > tau / q {
					res.weights[j + 1] = q * res.weights[j];
				} else {
					t = j;
					res.right = j + res.left;
					res.weights.resize(res.right - res.left, ValueType::zero());

					break;
				}
			}
		}

		// Compute normalization rate.
		res.total_weight = ValueType::zero();
		let mut j = 0; // We will compare this to t, which was set earlier

		while j < t {
			// We only have to add the minimal weight between indecies j and t to the total weight,
			// but we also have to increment j or decrement t, so we can't just use `min()`.
			if res.weights[j] <= res.weights[t] {
				res.total_weight += res.weights[j];
				j += 1;
			} else {
				res.total_weight += res.weights[t];
				t -= 1;
			}
		}
		// Get the last weight to add to the total weights
		res.total_weight += res.weights[j];

		res
	}
}
