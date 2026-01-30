#!/usr/bin/env python3

from common import *

import numpy as np

import sys

transitions = [
	Transition(np.matrix([1.0, -2.0, 0, 0]).T, None, lambda state: 2.0),
	Transition(np.matrix([1.0, 0.0, 0, 0]).T, None, lambda state: 1.0),
	Transition(np.matrix([-1.0, 2.0, 0, 0]).T, None, lambda state: 1.3),
	Transition(np.matrix([-1.0, 2.0, 1.0, 0]).T, None, lambda state: 0.3),
	Transition(np.matrix([1.0, -2.0, 0, 1.0]).T, None, lambda state: 0.3),
	Transition(np.matrix([0, 0, 0, -1.0]).T, None, lambda state: 1.3)
]

init_state = np.matrix([20, 0, 0, 0], dtype="float64").T


def sat_predicate(state): return state[3] >= 10


model = Model(init_state, transitions, 20, sat_predicate)

bypass_cardio = "--bypass_cardio" in sys.argv
bypass_storm = "--bypass_storm" in sys.argv

model.check_cardio_and_storm(100.0, bypass_cardio, bypass_storm)
