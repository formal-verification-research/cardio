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

	def from_tuple(contents_tuple):
		left, right, total_weights, weights = contents_tuple
		return FoxGlynnBound(left, right, total_weights, weights)

def scipy_poisson_pmf(lambda_val, k):
	return stats.poisson.pmf(k, lambda_val)

def numpy_poisson_pmf(lambda_val, k):
	# Multiple simulations to verify probability
	simulations = np.random.poisson(lambda_val, 10000)
	return np.mean(simulations == k)

def extract_fox_glynn_probability(fox_glynn_result, k):
	"""
	Extract the probability for a specific k from the Fox-Glynn bound result.
	
	Args:
		fox_glynn_result (FoxGlynnBound): The result from the Fox-Glynn computation
		k (int): The specific k value to extract the probability for
	
	Returns:
		float: The probability weight for the given k
	"""
	# Check if k is within the bounds
	if k < fox_glynn_result.left or k > fox_glynn_result.right:
		return 0.0
	
	# Calculate the index in the weights array
	index = k - fox_glynn_result.left
	
	# Return the normalized weight
	return fox_glynn_result.weights[index] / fox_glynn_result.total_weight


def test_fox_glynn_poisson_probabilities():
	# Test cases with various lambda values
	test_cases = [
		(1.0, 0),   # Low lambda, zero occurrences
		(2.5, 2),   # Moderate lambda, specific occurrence
		(10.0, 8),  # Higher lambda
		(20.0, 15)  # Very high lambda
	]

	for lambda_val, k in test_cases:
		# Get Rust Fox-Glynn result
		fox_glynn_result = FoxGlynnBound.from_tuple(cardio.util.fg_find(lambda_val, k))
		rust_result = extract_fox_glynn_probability(fox_glynn_result, k)
		
		# Compare with SciPy's direct PMF calculation
		scipy_result = stats.poisson.pmf(k, lambda_val)
		
		# Allow small numerical tolerance due to different computational methods
		np.testing.assert_almost_equal(
			rust_result, 
			scipy_result, 
			decimal=5,  # Adjust precision as needed
			err_msg=f"Mismatch for lambda={lambda_val}, k={k}"
		)

def test_fox_glynn_bound_properties():
	"""
	Validate the properties of the FoxGlynnBound result
	"""
	lambda_val = 5.0
	k = 6
	fox_glynn_result = FoxGlynnBound.from_tuple(cardio.util.fg_find(lambda_val, k))
	
	# Check basic properties
	assert fox_glynn_result.left <= fox_glynn_result.right, "Invalid left and right bounds"
	assert fox_glynn_result.total_weight > 0, "Total weight should be positive"
	assert len(fox_glynn_result.weights) > 0, "Weights list should not be empty"
	
	# Verify that weights sum approximately to total_weight
	np.testing.assert_almost_equal(
		sum(fox_glynn_result.weights), 
		fox_glynn_result.total_weight, 
		decimal=5,
		err_msg="Weights do not sum to total weight"
	)

def validate_probability_distribution(lambda_val, k=2.0):
	"""
	Comprehensive validation of the probability distribution
	"""
	fox_glynn_result = FoxGlynnBound.from_tuple(cardio.util.fg_find(lambda_val, k))
	
	# Calculate total probability
	total_prob = sum(
		extract_fox_glynn_probability(fox_glynn_result, k) 
		for k in range(fox_glynn_result.left, fox_glynn_result.right + 1)
	)
	
	# Total probability should be very close to 1
	np.testing.assert_almost_equal(
		total_prob, 
		1.0, 
		decimal=4,
		err_msg="Probabilities do not sum to 1"
	)

def test_multiple_lambda_values():
	"""
	Test Fox-Glynn computation across various lambda values
	"""
	lambda_values = [0.1, 1.0, 2.5, 5.0, 10.0, 20.0, 50.0]
	
	for lambda_val in lambda_values:
		
		# Run comprehensive validation
		validate_probability_distribution(lambda_val)

if __name__=="__main__":
	# test_fox_glynn_bound_properties()
	test_fox_glynn_poisson_probabilities()

