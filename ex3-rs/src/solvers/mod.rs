use ndarray::Array1;

use crate::{
    compute::ComputeBackend,
    solvers::strategies::{solve_cascade_seeded, solve_hotstart_seeded, solve_seeded},
};

pub mod bfgs;
pub mod lm;
mod strategies;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminationReason {
    Converged,
    MaxItersReached,
    LineSearchFailed,
    Diverged,
    Other,
}

#[derive(Debug, Clone)]
pub struct SolveOutcome {
    pub phases: Array1<f64>,
    pub cost: f64,
    pub iterations: u64,
    pub term_reason: TerminationReason,
}

#[derive(Debug, Clone, Copy)]
pub enum SolveMode {
    /// Direct solve at the given degree.
    Simple(usize),
    /// Solve at the first degree, then warm-start a solve at the second.
    /// Constraint: first < second.
    Hotstart(usize, usize),
    /// N cascading solves, gradually increasing degree from a small initial value
    /// up to the final degree, each warm-starting from the previous.
    Cascade(usize, usize),
}

pub trait Solver<T: ComputeBackend> {
    fn run(&self, backend: &T, xs: Array1<f64>) -> SolveOutcome;

    fn solve(&self, backend: &T, mode: SolveMode) -> Result<SolveOutcome, String> {
        self.solve_seeded(backend, mode, rand::random::<u64>())
    }

    fn solve_seeded(
        &self,
        backend: &T,
        mode: SolveMode,
        seed: u64,
    ) -> Result<SolveOutcome, String> {
        match mode {
            SolveMode::Simple(d) => Ok(solve_seeded::<T, Self>(&self, backend, d, seed)),
            SolveMode::Hotstart(s, d) => {
                solve_hotstart_seeded::<T, Self>(&self, backend, s, d, seed)
            }
            SolveMode::Cascade(n, d) => solve_cascade_seeded::<T, Self>(&self, backend, n, d, seed),
        }
    }
}
