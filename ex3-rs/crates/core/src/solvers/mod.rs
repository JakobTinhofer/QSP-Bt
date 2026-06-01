use crate::{
    compute::ComputeBackend,
    solvers::{
        configuration::{PhaseGenerator, PhaseMap, SolveMode},
        observe::{SolverContext, StageInfo},
        strategies::{solve, solve_cascade_seeded, solve_hotstart_seeded},
    },
};
use anyhow::Result;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod bfgs;
pub mod configuration;
pub mod lm;
pub mod observe;
mod strategies;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TerminationReason {
    Converged,
    MaxItersReached,
    LineSearchFailed,
    Diverged,
    Other,
}

#[derive(Error, Debug)]
#[error("invalid strategy parameters: {message}")]
pub struct StrategyError {
    pub message: String,
}

impl StrategyError {
    pub fn new(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
        }
    }
}

#[derive(Error, Debug)]
pub enum SolveError {
    #[error("solver cancelled")]
    Cancelled,
    #[error(transparent)]
    Other(#[from] anyhow::Error),
    #[error("Error with ")]
    StrategyError(#[from] StrategyError),
    #[error("internal solver error: ")]
    SolverError(&'static str),
}

pub type SolveResult<T = SolveOutcome> = Result<T, SolveError>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolveOutcome {
    #[serde(skip)]
    pub phases: Array1<f64>,
    pub cost: f64,
    pub iterations: u64,
    pub term_reason: TerminationReason,
    pub phase_mag_sum: f64,
}

impl SolveOutcome {
    pub fn new(
        phases: Array1<f64>,
        cost: f64,
        iterations: u64,
        term_reason: TerminationReason,
    ) -> Self {
        let phase_mag_sum = phases.iter().map(|p| p.abs()).sum();
        Self {
            phases,
            cost,
            iterations,
            term_reason,
            phase_mag_sum,
        }
    }
}

pub trait Solver<T: ComputeBackend>: Send + Sync {
    fn run(
        &self,
        backend: &T,
        ctx: &SolverContext,
        xs: Array1<f64>,
        map: PhaseMap,
        stage: StageInfo,
    ) -> SolveResult;

    fn solve(
        &self,
        backend: &T,
        ctx: &SolverContext,
        mode: SolveMode,
        map: PhaseMap,
        init: PhaseGenerator,
    ) -> SolveResult {
        let mut res = match mode {
            SolveMode::Simple(d) => Ok(solve::<T, Self>(&self, backend, ctx, d, map, init, None)?),
            SolveMode::Hotstart(s, d) => {
                solve_hotstart_seeded::<T, Self>(&self, backend, ctx, s, d, map, init)
            }
            SolveMode::Cascade(n, d) => {
                solve_cascade_seeded::<T, Self>(&self, backend, ctx, n, d, map, init)
            }
        }?;

        map.apply(&mut res.phases, backend.get_target())?;
        Ok(res)
    }
}
