import math

import numpy as np

from common import Model, Transition

try:
    import stormpy
    HAS_STORM = True
except Exception:
    HAS_STORM = False

import pytest


def _assert(res):
    if not HAS_STORM:
        return
    c = res["cardio"]
    s = res["storm"]
    assert c is not None and s is not None
    assert abs(c[0] - s[0]) < 1e-5, (c, s)
    assert abs(c[1] - s[1]) < 1e-5, (c, s)


@pytest.mark.skipif(not HAS_STORM, reason="stormpy not available")
def test_var_bound_zero():
    transitions = [Transition(np.array([[1.0]]), None, lambda st: 2.0)]
    init = np.array([[0.0]])
    model = Model(init, transitions, 0, lambda st: False)
    res = model.check_cardio_and_storm(1.0, bypass_cardio=False, bypass_storm=False)
    _assert(res)


@pytest.mark.skipif(not HAS_STORM, reason="stormpy not available")
def test_initial_state_satisfying():
    transitions = [Transition(np.array([[1.0]]), None, lambda st: 1.0)]
    init = np.array([[0.0]])
    model = Model(init, transitions, 3, lambda st: True)
    res = model.check_cardio_and_storm(1.0, bypass_cardio=False, bypass_storm=False)
    _assert(res)


@pytest.mark.skipif(not HAS_STORM, reason="stormpy not available")
def test_all_rates_zero():
    transitions = [
        Transition(np.array([[1.0]]), None, lambda st: 0.0),
        Transition(np.array([[-1.0]]), None, lambda st: 0.0),
    ]
    init = np.array([[1.0]])
    model = Model(init, transitions, 3, lambda st: st[0, 0] >= 2)
    res = model.check_cardio_and_storm(1.0, bypass_cardio=False, bypass_storm=False)
    _assert(res)


@pytest.mark.skipif(not HAS_STORM, reason="stormpy not available")
def test_deadlock_state():
    transitions = [Transition(np.array([[1.0]]), None, lambda st: 1.0)]
    init = np.array([[2.0]])
    # bound 2: state 2 is deadlock (no valid update? from 2, +1 -> 3 >= bound, absorbing)
    model = Model(init, transitions, 2, lambda st: False)
    res = model.check_cardio_and_storm(1.0, bypass_cardio=False, bypass_storm=False)
    _assert(res)


@pytest.mark.skipif(not HAS_STORM, reason="stormpy not available")
def test_duplicate_transitions_same_target():
    transitions = [
        Transition(np.array([[1.0]]), None, lambda st: 1.5),
        Transition(np.array([[1.0]]), None, lambda st: 2.5),
    ]
    init = np.array([[0.0]])
    model = Model(init, transitions, 4, lambda st: st[0, 0] >= 3)
    res = model.check_cardio_and_storm(1.0, bypass_cardio=False, bypass_storm=False)
    _assert(res)


@pytest.mark.skipif(not HAS_STORM, reason="stormpy not available")
def test_needed_constraints():
    transitions = [
        Transition(np.array([[1.0]]), np.array([[2.0]]), lambda st: 1.0),
        Transition(np.array([[1.0]]), None, lambda st: 1.0),
    ]
    init = np.array([[0.0]])
    model = Model(init, transitions, 4, lambda st: st[0, 0] >= 3)
    res = model.check_cardio_and_storm(1.0, bypass_cardio=False, bypass_storm=False)
    _assert(res)


@pytest.mark.skipif(not HAS_STORM, reason="stormpy not available")
def test_self_loop_transition():
    transitions = [
        Transition(np.array([[0.0]]), None, lambda st: 1.0),
        Transition(np.array([[1.0]]), None, lambda st: 0.5),
    ]
    init = np.array([[0.0]])
    model = Model(init, transitions, 3, lambda st: st[0, 0] >= 2)
    res = model.check_cardio_and_storm(1.0, bypass_cardio=False, bypass_storm=False)
    _assert(res)


@pytest.mark.skipif(not HAS_STORM, reason="stormpy not available")
def test_multi_dim_birth_death():
    transitions = [
        Transition(np.array([[1.0, 0.0]]), None, lambda st: 1.0),
        Transition(np.array([[-1.0, 0.0]]), None, lambda st: 0.5),
        Transition(np.array([[0.0, 1.0]]), None, lambda st: 0.7),
        Transition(np.array([[0.0, -1.0]]), None, lambda st: 0.3),
    ]
    init = np.array([[1.0], [1.0]])
    model = Model(init, transitions, 4, lambda st: st[0, 0] + st[1, 0] >= 6)
    res = model.check_cardio_and_storm(2.0, bypass_cardio=False, bypass_storm=False)
    _assert(res)


@pytest.mark.skipif(not HAS_STORM, reason="stormpy not available")
def test_large_time_bound():
    transitions = [Transition(np.array([[1.0]]), None, lambda st: 1.0)]
    init = np.array([[0.0]])
    model = Model(init, transitions, 3, lambda st: st[0, 0] >= 2)
    res = model.check_cardio_and_storm(50.0, bypass_cardio=False, bypass_storm=False)
    _assert(res)


@pytest.mark.skipif(not HAS_STORM, reason="stormpy not available")
def test_small_time_bound():
    transitions = [Transition(np.array([[1.0]]), None, lambda st: 1.0)]
    init = np.array([[0.0]])
    model = Model(init, transitions, 3, lambda st: st[0, 0] >= 2)
    res = model.check_cardio_and_storm(1e-3, bypass_cardio=False, bypass_storm=False)
    _assert(res)
