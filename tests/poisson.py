import pytest

import numpy as np
from scipy import stats

import cardio


class FoxGlynnBound:
	def __init__(self, left, right, total_weight, weights):
		self.left = left
		self.right = right
		self.total_weight = total_weight
		self.weights = weights

	@staticmethod
	def from_tuple(contents_tuple):
		left, right, total_weight, weights = contents_tuple
		return FoxGlynnBound(left, right, total_weight, weights)


def scipy_poisson_pmf(lambda_val, k):
	return stats.poisson.pmf(k, lambda_val)


def numpy_poisson_pmf(lambda_val, k):
	simulations = np.random.poisson(lambda_val, 100000)
	return np.mean(simulations == k)


def extract_fox_glynn_probability(fox_glynn_result, k):
	if k < fox_glynn_result.left or k > fox_glynn_result.right:
		return 0.0
	index = k - fox_glynn_result.left
	return fox_glynn_result.weights[index] / fox_glynn_result.total_weight


def make_bound(lambda_val, epsilon=1e-9):
	return FoxGlynnBound.from_tuple(cardio.util.fg_find(lambda_val, epsilon))


# ---------------------------------------------------------------------------
# Structural / invariant tests
# ---------------------------------------------------------------------------

class TestStructuralProperties:
	def test_left_le_midpoint_le_right(self):
		for lam in [0.1, 1.0, 5.0, 10.0, 25.0, 100.0, 400.0, 1000.0]:
			m = int(lam)
			b = make_bound(lam)
			assert b.left <= m, f"left={b.left} > m={m} for lambda={lam}"
			assert b.right >= m, f"right={b.right} < m={m} for lambda={lam}"

	def test_center_weight_is_one(self):
		for lam in [0.5, 1.0, 5.0, 10.0, 25.0, 100.0, 400.0, 1000.0]:
			m = int(lam)
			b = make_bound(lam)
			center_weight = b.weights[m - b.left]
			assert abs(center_weight - 1.0) < 1e-12, (
				f"center weight for lambda={lam} should be 1.0, got {center_weight}"
			)

	def test_weights_array_length_matches_bounds(self):
		for lam in [0.5, 5.0, 50.0, 400.0, 2000.0]:
			b = make_bound(lam)
			expected_len = b.right - b.left + 1
			assert len(b.weights) == expected_len, (
				f"weights length mismatch for lambda={lam}: "
				f"got {len(b.weights)}, expected {expected_len}"
			)

	def test_all_weights_nonnegative(self):
		for lam in [0.1, 1.0, 10.0, 100.0, 500.0]:
			b = make_bound(lam)
			for i, w in enumerate(b.weights):
				assert w >= 0.0, (
					f"negative weight at index {i} (k={i + b.left}) "
					f"for lambda={lam}: {w}"
				)

	def test_total_weight_positive(self):
		for lam in [0.01, 0.5, 5.0, 50.0, 500.0]:
			b = make_bound(lam)
			assert b.total_weight > 0.0, (
				f"total_weight not positive for lambda={lam}"
			)

	def test_weights_sum_equals_total_weight(self):
		for lam in [0.5, 5.0, 50.0, 400.0, 1000.0]:
			b = make_bound(lam)
			np.testing.assert_almost_equal(
				sum(b.weights),
				b.total_weight,
				decimal=5,
				err_msg=f"weights sum != total_weight for lambda={lam}"
			)


# ---------------------------------------------------------------------------
# Probability distribution tests
# ---------------------------------------------------------------------------

class TestProbabilityDistribution:
	@pytest.mark.parametrize("lam", [0.1, 1.0, 5.0, 10.0, 25.0, 100.0, 400.0, 1000.0])
	def test_probabilities_sum_to_one(self, lam):
		b = make_bound(lam)
		total = sum(extract_fox_glynn_probability(b, k) for k in range(b.left, b.right + 1))
		assert abs(total - 1.0) < 1e-4, f"prob sum={total} for lambda={lam}"

	@pytest.mark.parametrize("lam", [0.1, 1.0, 5.0, 10.0, 50.0, 200.0, 500.0])
	def test_probabilities_nonnegative(self, lam):
		b = make_bound(lam)
		for k in range(b.left, b.right + 1):
			p = extract_fox_glynn_probability(b, k)
			assert p >= 0.0, f"negative probability at k={k} for lambda={lam}"


# ---------------------------------------------------------------------------
# Accuracy tests against scipy
# ---------------------------------------------------------------------------

class TestAccuracy:
	@pytest.mark.parametrize("lam,k", [
		(1.0, 0),
		(1.0, 1),
		(1.0, 5),
		(2.5, 2),
		(5.0, 5),
		(5.0, 0),
		(10.0, 8),
		(10.0, 10),
		(10.0, 7),
		(20.0, 15),
		(20.0, 20),
		(50.0, 45),
		(50.0, 50),
		(100.0, 95),
		(100.0, 100),
	])
	def test_pmf_matches_scipy(self, lam, k):
		b = make_bound(lam)
		rust_val = extract_fox_glynn_probability(b, k)
		scipy_val = scipy_poisson_pmf(lam, k)
		np.testing.assert_almost_equal(
			rust_val, scipy_val, decimal=5,
			err_msg=f"mismatch for lambda={lam}, k={k}"
		)

	@pytest.mark.parametrize("lam,k", [
		(400.0, 400),
		(400.0, 380),
		(500.0, 500),
		(500.0, 480),
		(1000.0, 1000),
		(1000.0, 980),
		(1000.0, 1050),
		(2000.0, 2000),
		(2000.0, 1950),
		(5000.0, 5000),
	])
	def test_large_lambda_pmf_matches_scipy(self, lam, k):
		b = make_bound(lam)
		rust_val = extract_fox_glynn_probability(b, k)
		scipy_val = scipy_poisson_pmf(lam, k)
		np.testing.assert_almost_equal(
			rust_val, scipy_val, decimal=4,
			err_msg=f"mismatch for lambda={lam}, k={k}"
		)

	def test_simulated_matches_scipy(self):
		np.random.seed(42)
		for lam in [1.0, 5.0, 10.0, 25.0]:
			b = make_bound(lam)
			for k in [int(lam) - 1, int(lam), int(lam) + 1]:
				rust_val = extract_fox_glynn_probability(b, k)
				sim_val = numpy_poisson_pmf(lam, k)
				np.testing.assert_almost_equal(
					rust_val, sim_val, decimal=2,
					err_msg=f"simulation mismatch for lambda={lam}, k={k}"
				)


# ---------------------------------------------------------------------------
# Boundary condition tests
# ---------------------------------------------------------------------------

class TestBoundaryConditions:
	def test_probability_at_left_bound(self):
		b = make_bound(5.0)
		p = extract_fox_glynn_probability(b, b.left)
		assert p >= 0.0, "probability at left bound should be >= 0"
		assert p <= 1.0, "probability at left bound should be <= 1"

	def test_probability_at_right_bound(self):
		b = make_bound(5.0)
		p = extract_fox_glynn_probability(b, b.right)
		assert p >= 0.0
		assert p <= 1.0

	def test_probability_outside_bounds_is_zero(self):
		b = make_bound(5.0)
		assert extract_fox_glynn_probability(b, b.right + 1) == 0.0
		assert extract_fox_glynn_probability(b, b.right + 100) == 0.0
		if b.left > 0:
			assert extract_fox_glynn_probability(b, b.left - 1) == 0.0

	def test_k_zero_small_lambda(self):
		b = make_bound(0.5)
		p0 = extract_fox_glynn_probability(b, 0)
		scipy_p0 = scipy_poisson_pmf(0.5, 0)
		np.testing.assert_almost_equal(p0, scipy_p0, decimal=5)


# ---------------------------------------------------------------------------
# Epsilon sensitivity tests
# ---------------------------------------------------------------------------

class TestEpsilonSensitivity:
	def test_different_epsilons_all_sum_to_one(self):
		for eps in [1e-4, 1e-6, 1e-9, 1e-12]:
			b = make_bound(10.0, epsilon=eps)
			total = sum(
				extract_fox_glynn_probability(b, k)
				for k in range(b.left, b.right + 1)
			)
			assert abs(total - 1.0) < 1e-3, (
				f"prob sum={total} for epsilon={eps}"
			)

	def test_tighter_epsilon_more_accurate(self):
		lam = 10.0
		k = 10
		exact = scipy_poisson_pmf(lam, k)

		err_loose = abs(extract_fox_glynn_probability(make_bound(lam, 1e-4), k) - exact)
		err_tight = abs(extract_fox_glynn_probability(make_bound(lam, 1e-10), k) - exact)

		assert err_tight <= err_loose + 1e-10, (
			f"tighter epsilon not more accurate: loose={err_loose}, tight={err_tight}"
		)

	def test_loose_epsilon_still_reasonable(self):
		b = make_bound(5.0, epsilon=1e-3)
		p5 = extract_fox_glynn_probability(b, 5)
		scipy_p5 = scipy_poisson_pmf(5.0, 5)
		np.testing.assert_almost_equal(p5, scipy_p5, decimal=3)


# ---------------------------------------------------------------------------
# Lambda regime tests (exercises different code paths in poisson.rs)
# ---------------------------------------------------------------------------

class TestLambdaRegimes:
	"""Each regime triggers different branches in the Fox-Glynn implementation."""

	def test_very_small_lambda(self):
		for lam in [0.01, 0.1, 0.5]:
			b = make_bound(lam)
			assert b.left == 0, f"left should be 0 for small lambda={lam}"
			p0 = extract_fox_glynn_probability(b, 0)
			assert p0 > 0.5, f"P(0)={p0} should be >0.5 for lambda={lam}"

	def test_small_lambda(self):
		for lam in [1.0, 2.0, 5.0, 10.0, 20.0]:
			b = make_bound(lam)
			assert b.left == 0
			total = sum(
				extract_fox_glynn_probability(b, k)
				for k in range(b.left, b.right + 1)
			)
			np.testing.assert_almost_equal(total, 1.0, decimal=4)

	def test_medium_lambda(self):
		"""lambda in [25, 400) triggers iterative left truncation search."""
		for lam in [25.0, 50.0, 100.0, 200.0, 399.0]:
			b = make_bound(lam)
			assert b.left >= 0
			total = sum(
				extract_fox_glynn_probability(b, k)
				for k in range(b.left, b.right + 1)
			)
			np.testing.assert_almost_equal(total, 1.0, decimal=4)

	def test_large_lambda(self):
		"""lambda >= 400 triggers the large-lambda right bound path."""
		for lam in [400.0, 500.0, 1000.0, 2000.0, 5000.0]:
			b = make_bound(lam)
			assert b.right >= int(lam)
			total = sum(
				extract_fox_glynn_probability(b, k)
				for k in range(b.left, b.right + 1)
			)
			np.testing.assert_almost_equal(total, 1.0, decimal=3)

	def test_very_large_lambda(self):
		lam = 10000.0
		b = make_bound(lam)
		p = extract_fox_glynn_probability(b, 10000)
		assert p > 0.0
		total = sum(
			extract_fox_glynn_probability(b, k)
			for k in range(b.left, b.right + 1)
		)
		np.testing.assert_almost_equal(total, 1.0, decimal=3)


# ---------------------------------------------------------------------------
# Mode / shape tests
# ---------------------------------------------------------------------------

class TestDistributionShape:
	def test_mode_near_lambda(self):
		for lam in [1.0, 5.0, 10.0, 50.0, 100.0]:
			b = make_bound(lam)
			m = int(lam)
			p_mode = extract_fox_glynn_probability(b, m)
			# Mode PMF should dominate the tail at k=0
			p_zero = extract_fox_glynn_probability(b, 0)
			if m > 2:
				assert p_mode >= p_zero, (
					f"P(mode={m})={p_mode} < P(0)={p_zero} for lambda={lam}"
				)

	def test_pmf_at_mode_of_large_lambda(self):
		lam = 2000.0
		b = make_bound(lam)
		p = extract_fox_glynn_probability(b, 2000)
		# P(2000 | lambda=2000) ~ 1/sqrt(2*pi*2000) ~ 0.00892
		assert 0.008 < p < 0.010, f"P(2000|2000)={p}, expected ~0.00892"

	def test_tail_is_small(self):
		lam = 1000.0
		b = make_bound(lam)
		p = extract_fox_glynn_probability(b, 1100)
		assert p > 0.0, "tail probability should be positive"
		assert p < 0.01, f"tail probability {p} should be small"

	def test_weights_decrease_from_center(self):
		for lam in [10.0, 50.0, 100.0]:
			b = make_bound(lam)
			center_idx = int(lam) - b.left
			for offset in range(1, min(5, len(b.weights) - center_idx)):
				assert b.weights[center_idx] >= b.weights[center_idx + offset], (
					f"weight at center < weight at center+{offset} for lambda={lam}"
				)


# ---------------------------------------------------------------------------
# Cross-validation: check a range of k values match scipy
# ---------------------------------------------------------------------------

class TestCrossValidation:
	@pytest.mark.parametrize("lam", [5.0, 10.0, 50.0, 100.0])
	def test_all_k_in_range_match_scipy(self, lam):
		b = make_bound(lam)
		for k in range(b.left, b.right + 1):
			rust_val = extract_fox_glynn_probability(b, k)
			scipy_val = scipy_poisson_pmf(lam, k)
			tol = 1e-4 if lam < 200 else 1e-3
			assert abs(rust_val - scipy_val) < tol, (
				f"k={k}: rust={rust_val}, scipy={scipy_val}, "
				f"diff={abs(rust_val - scipy_val)} for lambda={lam}"
			)
