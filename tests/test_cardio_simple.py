import math

import cardio


def test_simple_absorption():
	rf = cardio.QuantitativeReachabilityFinder()
	rf.insert(1, 0, 1.0)
	lower, upper = rf.build_matrix_and_get_bounds(1.0)
	assert abs(lower - 0.0) < 1e-6
	assert abs(upper - (1.0 - math.exp(-1.0))) < 1e-6

