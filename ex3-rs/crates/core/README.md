# qsp-rs-core

The numerical engine for [qsp-rs](../..). Pure Rust library with no CLI or Python dependencies — use this if you want to embed QSP phase synthesis in your own Rust code.

## Modules

- **`target`** — target polynomial representations: `TargetPoly`, `TargetPattern`, `Parity`.
- **`compute`** — the `ComputeBackend` trait and the `CpuComputeBackend` implementation (handles loss, gradient, and Jacobian evaluation, with optional multithreading via `BackendMode`).
- **`solvers`** — the `Solver` trait, two implementations (`bfgs::BfgsOptions`, `lm::LmOptions`), `SolveMode` (Simple, Hotstart, Cascade), and `PhaseMap` (None, Mirror, MirrorIfPossible).

## Example

```rust
use qsp_rs_core::{
    compute::{cpu::{CpuComputeBackend, BackendMode}, ComputeBackend},
    solvers::{bfgs::BfgsOptions, PhaseMap, SolveMode, Solver},
    target::{Parity, TargetPattern, TargetPoly},
};

// Build a random even target of length 60 (mirrored from 30 half-points).
let target  = TargetPoly::from_pattern(&TargetPattern::Random, Parity::Even, 30);
let backend = CpuComputeBackend::new(target, BackendMode::Auto);

// Solve with L-BFGS, warm-starting at degree 20 and finishing at 60.
let outcome = BfgsOptions::default()
    .solve(
        &backend,
        SolveMode::Hotstart(20, 60),
        PhaseMap::MirrorIfPossible,
        0.4,                              // initial perturbation magnitude
    )
    .expect("solver failed");

println!("cost = {:e}, n_phases = {}", outcome.cost, outcome.phases.len());
```
