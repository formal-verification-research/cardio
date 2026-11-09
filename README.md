# Cardio: an efficient probabilistic model checker written in Rust.

Cardio aims to be a probabilistic model checking engine for deterministic Markov chains, i.e., CTMCs and DTMCs, written in Rust and eventually verified using [Kani](https://github.com/model-checking/kani) and/or [Verus](https://github.com/verus-lang/verus). Using Rust best-practice, Cardio wishes to be:

- modular
- efficient
- fast
- customizable

We would like Cardio to be competitive in speed to [Storm](https://github.com/moves-rwth/storm).

## Features

- [ ] CSL and PCTL parsing (in progress--mostly finished)
- [x] Rewards
- [ ] Transient analysis (in progress)
- [ ] Steady State Analysis (planned)

...and more.

## Building

Cardio uses [maturin](https://www.maturin.rs/) to build Python bindings.

```bash
python -m venv .venv
source .venv/bin/activate
pip install maturin numpy scipy pytest stormpy
maturin develop
```

## Testing

Rust unit tests:

```bash
cargo test
```

Python tests:

```bash
python -m pytest tests/poisson.py
python tests/toy.py
```

`tests/toy.py` compares Cardio against StormPy. Use `--bypass_cardio` or
`--bypass_storm` to isolate one engine.

## Formal verification

See `VERIFICATION.md` for the verification inventory and remaining work.

Optional Kani proofs are gated behind `cfg(kani)`. Install Kani and run:

```bash
cargo kani
```

## API overview

- `cardio.QuantitativeReachabilityFinder`: builds a CTMC from explicit transitions
  and computes bounded-until reachability bounds.
- `cardio.util.fg_find`: Fox-Glynn Poisson truncation helper.

## Development notes

- State index `0` is treated as absorbing and state index `1` as the initial state.
- The checker uniformizes CTMCs and uses Fox-Glynn weights for bounded-time
  computations.
- Rust formatting follows `rustfmt.toml` (tabs, Unix newlines).

---

Cardio is licensed under the GNU General Public License v3. It is open source software, free as in free speech, and always will be.

Cardio is developed at Utah State University in Logan, UT, USA.
