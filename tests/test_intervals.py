import math

import cardio

try:
    import stormpy
    HAS_STORM = True
except Exception:
    HAS_STORM = False


def _storm_bounds(time_bound_kind, a, b=None):
    if not HAS_STORM:
        return None
    builder = stormpy.SparseMatrixBuilder(2, 2)
    # state 0 absorbing, state 1 initial with rate 1 to 0
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
    components = stormpy.SparseModelComponents(m, labels, {}, rate_transitions=True)
    ctmc = stormpy.SparseCtmc(components)
    if time_bound_kind == "window":
        prop = stormpy.parse_properties(f'P=? [ true U[{a}, {b}] "satisfying" ]')[0]
    elif time_bound_kind == "lower":
        prop = stormpy.parse_properties(f'P=? [ true U[{a}, 100] "satisfying" ]')[0]
    else:
        prop = stormpy.parse_properties(f'P=? [ true U[0, {a}] "satisfying" ]')[0]
    res = stormpy.check_model_sparse(ctmc, prop, only_initial_states=True)
    return res.at(1)


def test_window_matches_storm():
    rf = cardio.QuantitativeReachabilityFinder()
    rf.insert(1, 0, 1.0)
    rf.set_state_satisfying(0)
    # lower, upper
    c = rf.build_matrix_and_get_window(0.2, 1.0)
    if HAS_STORM:
        s = _storm_bounds("window", 0.2, 1.0)
        assert abs(c - s) < 1e-6, (c, s)
    else:
        expected = math.exp(-0.2) - math.exp(-1.0)
        assert abs(c - expected) < 1e-6, (c, expected)


def test_lower_matches_storm():
    rf = cardio.QuantitativeReachabilityFinder()
    rf.insert(1, 0, 1.0)
    rf.set_state_satisfying(0)
    c = rf.build_matrix_and_get_lower(0.5)
    if HAS_STORM:
        s = _storm_bounds("lower", 0.5)
        assert abs(c - s) < 1e-5, (c, s)
    else:
        expected = math.exp(-0.5)
        assert abs(c - expected) < 1e-6, (c, expected)
