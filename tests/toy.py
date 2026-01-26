#!/usr/bin/env python3

from common import *

import numpy as np

transitions = [
	Transition(np.matrix([1.0, -2.0, 0, 0]).T, None, lambda state: 2.0 * state[0]),
	Transition(np.matrix([-1.0, 2.0, 0, 0]).T, None, lambda state: 1.3 * state[1]),
	Transition(np.matrix([-1.0, 2.0, 1.0, 0]).T, None, lambda state: 0.3 * state[2]),
	Transition(np.matrix([1.0, -2.0, 0, 1.0]).T, None, lambda state: 0.3 * state[2]),
	Transition(np.matrix([0, 0, 0, -1.0]).T, None, lambda state: 1.3 * state[3])
]

init_state = np.matrix([0,0,0,0], dtype="float64").T

sat_predicate = lambda state: state[3] >= 50

model = Model(init_state, transitions, 100, sat_predicate)

model.check_cardio_and_storm(100.0)
