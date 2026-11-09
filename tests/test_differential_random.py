import random

import numpy as np
import pytest

from common import Model, Transition

try:
    import stormpy
    HAS_STORM = True
except Exception:
    HAS_STORM = False

SEED = 20240517
N_CASES = 40


def _make_case(rng):
    dim = rng.randint(1, 3)
    bound = rng.randint(0, 6)
    ntrans = rng.randint(1, 4)
    transitions = []
    for _ in range(ntrans):
        upd = np.array([rng.randint(-2, 2) for _ in range(dim)], dtype=float).reshape(dim, 1)
        if rng.random() < 0.35:
            needed = np.array([rng.randint(0, 3) for _ in range(dim)], dtype=float).reshape(dim, 1)
        else:
            needed = None
        rate_val = rng.choice([0.0, 0.1, 0.5, 1.0, 2.0, 3.0])
        # deterministic rate (may depend on first coordinate but reproducibly)
        def rf(st, rv=rate_val):
            return rv
        transitions.append(Transition(upd, needed, rf))
    if bound == 0:
        init = np.zeros((dim, 1))
    else:
        init = np.array([float(rng.randint(0, bound - 1 if bound > 1 else 0)) for _ in range(dim)]).reshape(dim, 1)
    mode = rng.choice(["false", "true", "threshold", "sum", "mod2"])
    threshold = rng.randint(0, max(0, bound - 1)) if bound > 0 else 0

    def sat(st):
        if mode == "false":
            return False
        if mode == "true":
            return True
        if mode == "threshold":
            try:
                return bool(st[0, 0] >= threshold)
            except Exception:
                return False
        if mode == "sum":
            try:
                return bool(int(st.sum()) >= max(1, bound * dim // 2))
            except Exception:
                return False
        if mode == "mod2":
            try:
                return bool(int(st[0, 0]) % 2 == 0)
            except Exception:
                return False
        return False

    model = Model(init, transitions, bound, sat)
    return model, bound, mode


@pytest.mark.skipif(not HAS_STORM, reason="stormpy not available")
@pytest.mark.parametrize("case_idx", range(N_CASES))
def test_random_differential(case_idx):
    rng = random.Random(SEED + case_idx)
    model, bound, mode = _make_case(rng)
    tb = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0][case_idx % 6]
    res = model.check_cardio_and_storm(tb, bypass_cardio=False, bypass_storm=False)
    c = res["cardio"]
    s = res["storm"]
    assert c is not None and s is not None
    assert abs(c[0] - s[0]) < 1e-5, (case_idx, c, s)
    assert abs(c[1] - s[1]) < 1e-5, (case_idx, c, s)
