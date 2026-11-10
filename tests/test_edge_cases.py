import math

import cardio

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


# Interval and next-step tests (simple rate-1 absorption chain)

@pytest.mark.skipif(not HAS_STORM, reason="stormpy not available")
def test_window_simple():
    import math
    import stormpy
    rf = cardio.QuantitativeReachabilityFinder()
    rf.insert(1, 0, 1.0)
    rf.set_state_satisfying(0)
    c = rf.build_matrix_and_get_window(0.2, 1.0)
    # Storm reference
    builder = stormpy.SparseMatrixBuilder(2, 2)
    builder.add_next_value(0, 0, 1.0)
    builder.add_next_value(1, 0, 1.0)
    m = builder.build()
    labels = stormpy.StateLabeling(2)
    labels.add_label("absorbing")
    labels.add_label("satisfying")
    labels.add_label("init")
    labels.add_label_to_state("absorbing", 0)
    labels.add_label_to_state("satisfying", 0)
    labels.add_label_to_state("init", 1)
    comps = stormpy.SparseModelComponents(m, labels, {}, rate_transitions=True)
    ctmc = stormpy.SparseCtmc(comps)
    prop = stormpy.parse_properties('P=? [ true U[0.2, 1.0] "satisfying" ]')[0]
    res = stormpy.check_model_sparse(ctmc, prop, only_initial_states=True)
    s = res.at(1)
    assert abs(c - s) < 1e-6, (c, s)


@pytest.mark.skipif(not HAS_STORM, reason="stormpy not available")
def test_lower_simple():
    import math
    import stormpy
    rf = cardio.QuantitativeReachabilityFinder()
    rf.insert(1, 0, 1.0)
    rf.set_state_satisfying(0)
    c = rf.build_matrix_and_get_lower(0.5)
    builder = stormpy.SparseMatrixBuilder(2, 2)
    builder.add_next_value(0, 0, 1.0)
    builder.add_next_value(1, 0, 1.0)
    m = builder.build()
    labels = stormpy.StateLabeling(2)
    labels.add_label("absorbing")
    labels.add_label("satisfying")
    labels.add_label("init")
    labels.add_label_to_state("absorbing", 0)
    labels.add_label_to_state("satisfying", 0)
    labels.add_label_to_state("init", 1)
    comps = stormpy.SparseModelComponents(m, labels, {}, rate_transitions=True)
    ctmc = stormpy.SparseCtmc(comps)
    prop = stormpy.parse_properties('P=? [ true U[0.5, 100] "satisfying" ]')[0]
    res = stormpy.check_model_sparse(ctmc, prop, only_initial_states=True)
    s = res.at(1)
    assert abs(c - s) < 1e-5, (c, s)


@pytest.mark.skipif(not HAS_STORM, reason="stormpy not available")
def test_next_simple():
    rf = cardio.QuantitativeReachabilityFinder()
    rf.insert(1, 0, 1.0)
    rf.set_state_satisfying(0)
    c = rf.build_matrix_and_get_next()
    assert abs(c - 1.0) < 1e-6, c
