# qsp-rs

A Rust implementation of numerical methods for **Quantum Signal Processing (QSP)** phase-angle synthesis.

Given a target polynomial $P$ — specified as a set of points $(x_k, y_k)$ with $|y_k| \le 1$ and a definite parity — this library solves the inverse problem: find phase angles $\phi_1, \ldots, \phi_n$ such that the QSP unitary built from those phases evaluates to a polynomial $\tilde P \approx P$ at the target points.

## Workspace layout

This repository is a Cargo workspace with three crates:

| Crate                        | Purpose                                                                                                                |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| [`qsp-rs-core`](crates/core) | Numerical engine. Pure Rust library, no frontend dependencies. Use this to embed QSP solvers in your own Rust project. |
| [`qsp-rs-cli`](crates/cli)   | Command-line interface for solving, benchmarking, and analysis.                                                        |
| [`qsp-rs-py`](crates/python) | Python bindings via [pyo3](https://pyo3.rs) and [maturin](https://maturin.rs).                                         |

## Features

- **Two optimizers:** L-BFGS (via `argmin`) and Levenberg–Marquardt (via `levenberg-marquardt`).
- **Solve strategies:** direct (`simple`), warm-started (`hotstart`), and progressive (`cascade`).
- **Phase symmetries:** optional mirror map exploits the symmetric phase factor representation to halve the parameter count for real-valued targets.
- **Backend:** SIMD-friendly CPU implementation with auto/single/multi-threaded gradient evaluation.

## Quick start

### CLI

```bash
cargo run --release -p qsp-rs-cli -- solve-poly -n 30 -t rand -p even
```

### Python

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin
cd crates/python && maturin develop --release
python -c "import qsp_rs; print(qsp_rs.solve_poly(target_half_len=30))"
```

See each crate's README for details.
