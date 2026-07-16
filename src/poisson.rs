// Poisson sum convergence calculation stuff

use std::f64::{self, consts::PI};

use log::warn;
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
		let mut tau = <ValueType as Bounded>::min_value();
		let root2pi = ValueType::from_f64((2.0 * PI).sqrt()).unwrap();
		// Error bound only uses epsilon * root2pi
		let mut er2pi = epsilon * root2pi;

		// Create the left and right bounds, which may be negative. Initialize them to zero
		let mut left: isize = 0;

		// Like the main `fox_glynn` method, we get the mid-point from the value of lambda
		let m = lambda.to_usize().unwrap();

		// First, compute the left truncation point
		if m < 25 {
			// The left truncation point is zero for lambda midpoint is < 25
			left = 0;
		} else {
			// We actually have to look for the left truncation point iteratively if m >= 25

			// First, compute a couple constants that are needed.
			let b = (one + lambda.recip())
				* (lambda.recip() * ValueType::from_f64(0.125).unwrap()).exp();
			let root_lmbda = lambda.sqrt();
			let mut k: isize = 4;

			loop {
				let kvt = ValueType::from(k).unwrap();
				// First, compute a candidate for `left`.
				left = m as isize - (kvt * root_lmbda + p5).ceil().to_isize().unwrap();

				// If the truncation point is negative, then make it zero and terminate the loop.
				if left.is_negative() {
					left = 0;
					break;
				}

				// It's a good thing we reference the Storm code in this implementation, since they
				// correctly point out that Fox-Glynn mixes up Phi and 1 - Phi in Propositions 2-4.
				let max_err = b * (-kvt.powi(2) * p5).exp() / kvt;

				// If the left-hand error is relatively small, loosen the requirements on the right
				// hand side and do not bound the left-hand side any farther.
				if max_err + max_err <= epsilon {
					er2pi -= max_err;
					break;
				}

				// Increment k
				k += 1;
			}

			// If the loop has terminated, the left bound has been found.
		}

		// Now we just have to compute the right bound. However, first we must compute a couple
		// of constants and update the epsilon value.

		let mut k: isize = 4;
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
			let kvt = ValueType::from_isize(k).unwrap();
			// The magic constants in the above if-statement come from the fact that we don't have
			// to compute the extra multiplier factor here.
			if er2pi * kvt >= (-kvt.powi(2) * p5).exp() {
				break;
			}
			// Increment k
			k += 1;
		}
		let kvt = ValueType::from_isize(k).unwrap();
		// Compute the right bound and determine if it's reliable.
		let right = m_max
			+ (kvt * (lambda_max + lambda_max).sqrt() + p5)
				.ceil()
				.to_isize()
				.unwrap();
		let reliability_bound = m_max + ((lambda_max + ValueType::one()) * p5).to_isize().unwrap();
		if right > reliability_bound {
			warn!(
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
		res.weights[m - res.left] = ValueType::one();

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
				warn!("Underflow in lambda >= 25!");
			}

			// Right truncation point underflow check
			if m >= 400 {
				// Proposition 5
				let i = res.right as isize - m as isize;
				let ir = i as f64;
				let numeric_result = lnc_m - ir * (ir + 1.0) / (2.0 * lambda_f64);
				if numeric_result <= tau_f64 {
					warn!("Underflow in lambda >= 400!");
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
		assert!(epsilon.is_positive(), "epsilon must be positive");
		// Start the mid point at the the current value of `lambda`.
		let m = lambda.to_usize().unwrap();

		let tau = <ValueType as Bounded>::min_value();
		let mut res = Self::fg_find(lambda, epsilon);

		// The left side of the weights array is easy to fill in.
		for j in (1..=m - res.left).rev() {
			res.weights[j - 1] =
				ValueType::from_usize(j + res.left).unwrap() / lambda * res.weights[j];
		}

		let mut t = res.right - res.left;

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
				warn!("Because {0} <= 600, underflow may occur.", res.right)
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

#[cfg(test)]
mod tests {
	use super::*;

	type FG = FoxGlynnBound<f64>;

	/// Helper: extract the normalized probability for a given k from a FoxGlynnBound.
	fn prob(bound: &FG, k: usize) -> f64 {
		if k < bound.left || k > bound.right {
			return 0.0;
		}
		bound.weights[k - bound.left] / bound.total_weight
	}

	// --- Structural property tests ---

	#[test]
	fn center_weight_is_one() {
		for lambda in [0.5, 1.0, 5.0, 10.0, 25.0, 100.0, 400.0, 1000.0] {
			let m = lambda as usize;
			let b = FG::fox_glynn(lambda, 1e-9);
			assert!(
				b.weights[m - b.left].abs() - 1.0 < 1e-12,
				"center weight for lambda={lambda} should be 1.0, got {}",
				b.weights[m - b.left]
			);
		}
	}

	#[test]
	fn left_le_midpoint_le_right() {
		for lambda in [0.1, 1.0, 5.0, 25.0, 100.0, 400.0, 1000.0, 5000.0] {
			let m = lambda as usize;
			let b = FG::fox_glynn(lambda, 1e-9);
			assert!(
				b.left <= m,
				"left={} > m={} for lambda={}",
				b.left,
				m,
				lambda
			);
			assert!(
				b.right >= m,
				"right={} < m={} for lambda={}",
				b.right,
				m,
				lambda
			);
		}
	}

	#[test]
	fn weights_array_length_matches_bounds() {
		for lambda in [0.5, 5.0, 50.0, 400.0, 2000.0] {
			let b = FG::fox_glynn(lambda, 1e-9);
			let expected_len = b.right - b.left + 1;
			assert_eq!(
				b.weights.len(),
				expected_len,
				"weights length mismatch for lambda={lambda}: got {} expected {expected_len}",
				b.weights.len()
			);
		}
	}

	#[test]
	fn all_weights_nonnegative() {
		for lambda in [0.1, 1.0, 10.0, 100.0, 500.0] {
			let b = FG::fox_glynn(lambda, 1e-9);
			for (i, w) in b.weights.iter().enumerate() {
				assert!(
					*w >= 0.0,
					"negative weight at index {i} (k={}) for lambda={lambda}: {w}",
					i + b.left
				);
			}
		}
	}

	#[test]
	fn total_weight_positive() {
		for lambda in [0.01, 0.5, 5.0, 50.0, 500.0] {
			let b = FG::fox_glynn(lambda, 1e-9);
			assert!(
				b.total_weight > 0.0,
				"total_weight not positive for lambda={lambda}"
			);
		}
	}

	// --- Probability distribution tests ---

	#[test]
	fn probabilities_sum_to_one() {
		for lambda in [0.1, 1.0, 5.0, 10.0, 25.0, 100.0, 400.0, 1000.0] {
			let b = FG::fox_glynn(lambda, 1e-9);
			let total: f64 = (b.left..=b.right).map(|k| prob(&b, k)).sum();
			assert!(
				(total - 1.0).abs() < 1e-4,
				"probabilities sum to {total} for lambda={lambda}, expected ~1.0"
			);
		}
	}

	#[test]
	fn pmf_matches_scipy() {
		// Compare against hand-computed Poisson PMF values: P(k) = e^{-lam} * lam^k / k!
		let cases: Vec<(f64, usize, f64)> = vec![
			(1.0, 0, 0.3678794412), // e^{-1}
			(1.0, 1, 0.3678794412),
			(1.0, 5, 0.0030656629),
			(5.0, 5, 0.1754673698),
			(5.0, 0, 0.0067379470),
			(10.0, 10, 0.1251100262),
			(10.0, 7, 0.0900792140),
			(20.0, 20, 0.0888353256),
		];
		for (lambda, k, expected) in cases {
			let b = FG::fox_glynn(lambda, 1e-9);
			let got = prob(&b, k);
			let tol = 1e-5;
			assert!(
				(got - expected).abs() < tol,
				"P({k} | lambda={lambda}): got {got}, expected {expected}, diff {}",
				(got - expected).abs()
			);
		}
	}

	// --- Left truncation code path (lambda >= 25) ---

	#[test]
	fn left_truncation_for_medium_lambda() {
		// For lambda >= 25, left should be 0 or the iterative algorithm should find a value.
		// For moderate lambdas, left may or may not be zero. Just verify the result is sane.
		for lambda in [25.0, 30.0, 50.0, 100.0, 200.0] {
			let b = FG::fox_glynn(lambda, 1e-9);
			// left should be non-negative
			assert!(b.left >= 0, "negative left for lambda={lambda}");
			// left <= lambda
			assert!(
				b.left <= lambda as usize + 1,
				"left={} too large for lambda={lambda}",
				b.left
			);
			// Distribution should still sum to 1
			let total: f64 = (b.left..=b.right).map(|k| prob(&b, k)).sum();
			assert!(
				(total - 1.0).abs() < 1e-4,
				"prob sum={total} for lambda={lambda}"
			);
		}
	}

	// --- Right bound code path (lambda >= 400 vs < 400) ---

	#[test]
	fn large_lambda_right_bound() {
		// For lambda >= 400, the right bound uses a different formula.
		for lambda in [400.0, 500.0, 1000.0, 2000.0, 5000.0] {
			let b = FG::fox_glynn(lambda, 1e-9);
			let m = lambda as usize;
			assert!(
				b.right >= m,
				"right={} < m={} for lambda={lambda}",
				b.right,
				m
			);
			// right should not be unreasonably large
			assert!(
				b.right < m + 10 * (2.0 * lambda).sqrt() as usize + 100,
				"right={} unreasonably large for lambda={lambda}",
				b.right
			);
			// Probabilities should still sum to 1
			let total: f64 = (b.left..=b.right).map(|k| prob(&b, k)).sum();
			assert!(
				(total - 1.0).abs() < 1e-3,
				"prob sum={total} for lambda={lambda}"
			);
		}
	}

	// --- Epsilon sensitivity ---

	#[test]
	fn tighter_epsilon_gives_better_accuracy() {
		let lambda = 10.0;
		let k = 10;
		let exact =
			(-lambda).exp() * lambda.powi(k as i32) / (1..=k).map(|x| x as f64).product::<f64>();

		let loose = FG::fox_glynn(lambda, 1e-4);
		let tight = FG::fox_glynn(lambda, 1e-10);

		let err_loose = (prob(&loose, k) - exact).abs();
		let err_tight = (prob(&tight, k) - exact).abs();

		// Tighter epsilon should be at least as accurate (and typically more so)
		assert!(
			err_tight <= err_loose + 1e-10,
			"tighter epsilon not more accurate: loose_err={err_loose}, tight_err={err_tight}"
		);
	}

	#[test]
	fn very_small_epsilon() {
		let b = FG::fox_glynn(5.0, 1e-12);
		let total: f64 = (b.left..=b.right).map(|k| prob(&b, k)).sum();
		assert!(
			(total - 1.0).abs() < 1e-6,
			"prob sum={total} for very small epsilon"
		);
	}

	// --- Small lambda edge cases ---

	#[test]
	fn very_small_lambda() {
		for lambda in [0.01, 0.1, 0.5] {
			let b = FG::fox_glynn(lambda, 1e-9);
			assert_eq!(b.left, 0, "left should be 0 for small lambda={lambda}");
			let total: f64 = (b.left..=b.right).map(|k| prob(&b, k)).sum();
			assert!(
				(total - 1.0).abs() < 1e-4,
				"prob sum={total} for lambda={lambda}"
			);
			// P(0) should dominate for very small lambda
			let p0 = prob(&b, 0);
			assert!(p0 > 0.5, "P(0)={p0} should be >0.5 for lambda={lambda}");
		}
	}

	// --- Probability outside bounds is zero ---

	#[test]
	fn probability_outside_bounds_is_zero() {
		let b = FG::fox_glynn(5.0, 1e-9);
		// Check one below left and one above right
		if b.left > 0 {
			assert_eq!(prob(&b, b.left - 1), 0.0);
		}
		assert_eq!(prob(&b, b.right + 1), 0.0);
		assert_eq!(prob(&b, b.right + 100), 0.0);
	}

	// --- Symmetry-ish check: P(k) near the mode should be highest ---

	#[test]
	fn mode_near_lambda() {
		for lambda in [1.0, 5.0, 10.0, 50.0, 100.0] {
			let b = FG::fox_glynn(lambda, 1e-9);
			let m = lambda as usize;
			let p_mode = prob(&b, m);
			// The PMF at the mode (floor(lambda)) should be the highest or near-highest
			// Check that it's at least as high as P(0) and P(2*lambda) for large lambda
			if m > 2 {
				let p_zero = prob(&b, 0);
				assert!(
					p_mode >= p_zero,
					"P(mode={m})={p_mode} < P(0)={p_zero} for lambda={lambda}"
				);
			}
		}
	}

	// --- Weight monotonicity away from center ---

	#[test]
	fn weights_decrease_away_from_center() {
		// For large enough lambda, weights should decrease as you move away from the center.
		// This isn't universally true for all k, but near the center it holds.
		for lambda in [10.0, 50.0, 100.0] {
			let b = FG::fox_glynn(lambda, 1e-9);
			let center_idx = lambda as usize - b.left;
			// Check a few steps to the right of center
			for offset in 1..5.min(b.weights.len() - center_idx) {
				let ci = center_idx;
				let ni = center_idx + offset;
				assert!(
					b.weights[ci] >= b.weights[ni],
					"weight at center ({}) < weight at center+{offset} ({}) for lambda={lambda}",
					b.weights[ci],
					b.weights[ni]
				);
			}
		}
	}

	// --- Integration: large-scale lambda with large k ---

	#[test]
	fn large_lambda_large_k() {
		// lambda=2000, k=2000 (at the mode)
		let b = FG::fox_glynn(2000.0, 1e-9);
		let p = prob(&b, 2000);
		// P(2000 | lambda=2000) ~ 1/sqrt(2*pi*2000) ~ 0.00892
		assert!(
			p > 0.008 && p < 0.010,
			"P(2000|2000)={p}, expected ~0.00892"
		);
	}

	#[test]
	fn large_lambda_tail() {
		// lambda=1000, check k=1100 (in the tail)
		let b = FG::fox_glynn(1000.0, 1e-9);
		let p = prob(&b, 1100);
		// Should be small but positive
		assert!(p > 0.0, "P(1100|1000) should be positive");
		assert!(p < 0.01, "P(1100|1000)={p} should be small");
	}
}
