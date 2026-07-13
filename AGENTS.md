# AGENTS.md

## Repository Overview

This repo contains the source code for Cardio. Cardio is a work in progress probabilistic model checker for continuous-time Markov chains which primarily focuses on bounded until properties specified in the CSL (continuous stochastic logic) format.

Cardio is written in Rust and provides Python bindings via `pyo3`. The Python bindings are currently used for testing.

## Testing instructions

1. Tests will be done in Python. Ensure that a Python virtual environment is set up at the repository root. Then activate that virtual environment. Tests use `numpy`, `scipy`, `pytest` and `stormpy` (a different probabilistic model checking engine), so ensure that these are installed within the venv.
2. From within the project root, run `maturin develop` to compile cardio and install it in the venv.
3. Test files are located in `./tests`. Currently, the most interesting tests are `poisson.py` and `toy.py`. These are what we are trying to get to work.
    - `toy.py` constructs a model and attempts to test it in cardio and compares it with stormpy. To just get the result from storm (known to be correct) run with `--bypass_cardio`. To get just Cardio's results, run with `--bypass_storm`.

## General Repo Guidance

- Formatting:
    - With the exception of markdown, we prefer tabs over spaces. For rust code, we provide formatting rules in `rustfmt.toml`. Python formatting is less strict.

## Where things are in code

From within `src`, there are a few files that encapsulate different functionality:

1. `checker.rs`: The main file that does the heavy lifting. Performs iteration on the uniformized matrix.
2. `labels.rs`: The file that contains definitions for CTMC state and transition labels.
3. `lib.rs`: Root of the cardio library.
4. `matrix.rs`: Creates uniformized sparse matrix
5. `parser.rs`: Parses CSL and eventually other formats including model formats like PRISM and JANI.
6. `poisson.rs`: An implementation of Fox-Glynn
7. `property.rs`: Defines CSL properties
8. `python.rs`: Defines the Python bindings
9. `rewards.rs`: Defines state and transition rewards
