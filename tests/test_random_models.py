import numpy as np
import random

from common import Model, Transition

try:
    import stormpy
    HAS_STORM = True
except Exception:
    HAS_STORM = False


def _assert_close(res):
    if not HAS_STORM:
        return
    c = res["cardio"]
    s = res["storm"]
    assert c is not None and s is not None
    assert abs(c[0] - s[0]) < 1e-5, (c, s)
    assert abs(c[1] - s[1]) < 1e-5, (c, s)


def test_random_walk_1d():
    random.seed(1)
    transitions = [
        Transition(np.array([[1.0]]), None, lambda st: 1.0 + random.random()),
        Transition(np.array([[-1.0]]), None, lambda st: 0.5 + random.random()),
    ]
    init = np.matrix([2.0]).T
    model = Model(init, transitions, 6, lambda st: st[0] >= 5)
    res = model.check_cardio_and_storm(3.0, bypass_cardio=False, bypass_storm=not HAS_STORM)
    _assert_close(res)


def test_random_two_transitions():
    transitions = [
        Transition(np.array([[1.0, 0.0]]), None, lambda st: 2.0),
        Transition(np.array([[0.0, 1.0]]), None, lambda st: 1.5),
        Transition(np.array([[-1.0, -1.0]]), None, lambda st: 0.7),
    ]
    init = np.matrix([0.0, 0.0]).T
    model = Model(init, transitions, 4, lambda st: int(st[0,0]) + int(st[1,0]) >= 6)
    res = model.check_cardio_and_storm(2.0, bypass_cardio=False, bypass_storm=not HAS_STORM)
    _assert_close(res)


def test_random_absorbing_heavy():
    transitions = [
        Transition(np.array([[1.0]]), None, lambda st: 0.3),
        Transition(np.array([[-1.0]]), None, lambda st: 0.2),
    ]
    init = np.matrix([1.0]).T
    model = Model(init, transitions, 3, lambda st: False)
    res = model.check_cardio_and_storm(1.0, bypass_cardio=False, bypass_storm=not HAS_STORM)
    _assert_close(res)
