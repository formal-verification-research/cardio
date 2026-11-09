import numpy as np

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
    assert abs(c[0] - s[0]) < 1e-6, (c, s)
    assert abs(c[1] - s[1]) < 1e-6, (c, s)



def test_absorbing_target_immediate():
    transitions = [Transition(np.matrix([0.0, 0.0]).T, None, lambda st: 1.0)]
    init = np.matrix([0.0, 0.0]).T
    model = Model(init, transitions, 0, lambda st: True)
    # initial state satisfying -> self-loop 1
    res = model.check_cardio_and_storm(1.0, bypass_cardio=False, bypass_storm=not HAS_STORM)
    if HAS_STORM:
        _assert_close(res)


def test_two_state_chain():
    # state (x) with x from 0..2; transition increments x with rate 1, else absorbing
    transitions = [Transition(np.matrix([1.0]).T, None, lambda st: 1.0)]
    init = np.matrix([0.0]).T
    model = Model(init, transitions, 2, lambda st: st[0] >= 2)
    res = model.check_cardio_and_storm(2.0, bypass_cardio=False, bypass_storm=not HAS_STORM)
    if HAS_STORM:
        _assert_close(res)


def test_birth_death_small():
    transitions = [
        Transition(np.matrix([1.0]).T, None, lambda st: 2.0),
        Transition(np.matrix([-1.0]).T, None, lambda st: 1.0),
    ]
    init = np.matrix([3.0]).T
    model = Model(init, transitions, 5, lambda st: False)
    # no satisfying -> lower 0; upper maybe absorption? Actually no absorbing target; satisfying none -> lower/upper 0? 
    res = model.check_cardio_and_storm(1.0, bypass_cardio=False, bypass_storm=not HAS_STORM)
    if HAS_STORM:
        _assert_close(res)
