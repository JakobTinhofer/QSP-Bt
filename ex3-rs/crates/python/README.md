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
import qsp_rs

# Solve for phases
result = qsp_rs.solve_poly(
    target_half_len=30,
    target_pattern="rand",
    parity="even",
    solver="bfgs",
    mode="hotstart,20,60",
    seed=42,                            # optional, for reproducibility
    bfgs_options={"max_iters": 100_000},
)
print(result)
# SolveResult(cost=1.2e-09, n_phases=60, iterations=412, termination='converged', elapsed_ms=87.3)

# Evaluate the resulting QSP polynomial
xs = np.linspace(-1, 1, 1000)
ys = qsp_rs.evaluate_poly(result.phases, xs)     # array in  -> array out
y0 = qsp_rs.evaluate_poly(result.phases, 0.5)    # scalar in -> scalar out
```

## API

| Symbol                                                                       | Description                                                                                                     |
| ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `solve_poly_with_pattern(target_half_len, pattern, **kwargs) -> SolveResult` | Solve for QSP phases.                                                                                           |
| `evaluate_poly(phases, x) -> complex \| ndarray`                             | Evaluate the QSP polynomial. Returns a scalar `complex` for scalar input, a `complex128` array for array input. |
| `SolveResult`                                                                | Result object with `cost`, `phases` (float64 ndarray), `iterations`, `termination`, `elapsed_ms`.               |
