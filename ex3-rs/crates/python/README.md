# qsp-rs-py (Python bindings)

Python bindings for [qsp-rs](../..), built with [pyo3](https://pyo3.rs) and packaged with [maturin](https://maturin.rs).

## Install

In an activated virtual environment:

```bash
pip install maturin
cd crates/python
maturin develop --release
```

Always use `--release` — the debug build is dramatically slower for numerical work.

For a distributable wheel:

```bash
maturin build --release
# wheel lands in target/wheels/
```

## Usage

```python
import numpy as np
import qsp_rs as qsp

# Solve for phases
result = qsp.solve_poly_with_pattern(
    target_half_len=10,
    target_pattern="rand",
    parity="even",
    solver="bfgs",
    phase_map="none",
    seed=1,
    mode="hotstart,20,40",
)
result

# Evaluate the resulting QSP polynomial
xs = np.linspace(-1, 1, 1000)
ys = qsp.evaluate_poly(result.phases, xs)     # array in  -> complex128 array out
y0 = qsp.evaluate_poly(result.phases, 0.5)    # scalar in -> Python complex out

# Inspect the target the solver was matching
target = result.target
target.xs       # Chebyshev grid in [-1, 1]
target.ys       # complex target values at xs
target.thetas   # arccos(xs)
```

## API

### Functions

| Symbol | Description |
| --- | --- |
| `solve_poly_with_pattern(target_half_len, target_pattern, **kwargs) -> SolveResult` | Solve for QSP phases against a built-in target pattern (e.g. `"rand"`). |
| `solve_poly(ys, **kwargs) -> SolveResult` | Solve for QSP phases against an explicit `complex128` array of target values. |
| `evaluate_poly(phases, x) -> complex \| ndarray` | Evaluate the QSP polynomial. Scalar `float` `x` → Python `complex`; 1-D `float64` array `x` → `complex128` array. |
| `theta_k(k, n_half) -> float` | Returns `θ_k` for integer index `k` on a length-`n_half` grid. |
| `theta_k_continuous(k, n_half) -> float` | Continuous-`k` variant of `theta_k` (accepts a `float` `k`). |

### Common solve keyword arguments

Both `solve_poly` and `solve_poly_with_pattern` accept the following keyword-only arguments:

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `parity` | `str` | `"even"` | `"even"` or `"odd"`. |
| `solver` | `str` | `"bfgs"` | `"bfgs"` or `"lm"` (Levenberg–Marquardt). |
| `mode` | `str` | `"hotstart,20,60"` | Solve-mode string, e.g. `"hotstart,<start>,<end>"`. |
| `phase_map` | `str` | `"mirror-if-possible"` | Phase-mapping strategy (e.g. `"none"`, `"mirror-if-possible"`). |
| `init_perturb_mag` | `float` | `0.4` | Magnitude of the random perturbation applied to the initial guess. |
| `backend_mode` | `str` | `"auto"` | Compute-backend selection. |
| `seed` | `int \| None` | `None` | RNG seed; when set, the seeded solver entry point is used. |
| `bfgs_options` | `dict \| None` | `None` | Per-call overrides for BFGS: `max_iters`, `mem`, `tol_grad`. |
| `lm_options` | `dict \| None` | `None` | Per-call overrides for LM: `max_iters`, `initial_lambda`, `tol`. |

### Classes

#### `SolveResult` (frozen)

| Attribute | Type | Description |
| --- | --- | --- |
| `cost` | `float` | Final objective value. |
| `phases` | `ndarray[float64]` | Solved phase sequence. |
| `total_phase` | `float` | Sum of `\|phase\|` across the sequence. |
| `iterations` | `int` | Solver iteration count. |
| `termination` | `str` | One of `"converged"`, `"max_iters_reached"`, `"line_search_failed"`, `"diverged"`, `"other"`. |
| `elapsed_ms` | `float` | Wall time for the solve, in milliseconds. |
| `target` | `TargetPoly` | The target polynomial the solver matched against. |

`SolveResult.__repr__` summarises the key fields on one line.

#### `TargetPoly` (frozen)

| Attribute | Type | Description |
| --- | --- | --- |
| `n_half` | `int` | Half-length of the target grid. |
| `xs` | `ndarray[float64]` | Grid points in `[-1, 1]`. |
| `ys` | `ndarray[complex128]` | Target polynomial values at `xs`. |
| `ks` | `ndarray[uintp]` | Index array `0..n_half`. |
| `thetas` | `ndarray[float64]` | `arccos(xs)` — angle representation of the grid. |