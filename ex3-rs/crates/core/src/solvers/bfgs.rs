use crate::{
    compute::ComputeBackend,
    solvers::{
        PhaseMap, SolveError, SolveOutcome, SolveResult, Solver, TerminationReason,
        observe::{CancelToken, ProgressObserver, ProgressReport, SolverContext, StageInfo},
    },
};
use anyhow::{Context, Result};
use argmin::{
    core::{CostFunction, Executor, Gradient, KV, State, TerminationStatus, observers::Observe},
    solver::{linesearch::MoreThuenteLineSearch, quasinewton::LBFGS},
};
use argmin_math::Error;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::{cell::RefCell, sync::Arc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BfgsOptions {
    pub max_iters: u64,
    pub mem: usize,
    pub tol_grad: f64,
}

impl Default for BfgsOptions {
    fn default() -> Self {
        Self {
            max_iters: 500000,
            mem: 10,
            tol_grad: 1e-8,
        }
    }
}

struct QspProblem<'a, T: ComputeBackend> {
    backend: &'a T,
    map: PhaseMap,
    cache: RefCell<Option<(Array1<f64>, f64, Array1<f64>)>>,
    cancel: CancelToken,
}

impl<'a, T: ComputeBackend> QspProblem<'a, T> {
    fn cached(&self, p: &Array1<f64>) -> (f64, Array1<f64>) {
        {
            let c = self.cache.borrow();
            if let Some((cp, cost, grad)) = c.as_ref() {
                if cp == p {
                    return (*cost, grad.clone());
                }
            }
        }

        let mut p_mapped = p.clone();
        self.map
            .apply(&mut p_mapped, self.backend.get_target())
            .unwrap();

        let (cost, mut grad) = self.backend.evaluate_f_grad(&p_mapped.view());

        // in case we applied a map that changed our param size,
        // reduce it. If different maps are used at a later point we might
        // need to abstract this away into the map type.
        if grad.len() > p.len() {
            let mut new_grad = Array1::from_iter(grad.iter().take(p.len()).map(|p| *p));
            for i in 0..(grad.len() / 2 as usize) {
                new_grad[i] += grad[grad.len() - (i + 1)];
            }
            grad = new_grad;
        }
        *self.cache.borrow_mut() = Some((p_mapped, cost, grad.clone()));
        (cost, grad)
    }
}

impl<'a, T: ComputeBackend> CostFunction for QspProblem<'a, T> {
    type Param = Array1<f64>;
    type Output = f64;
    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        if self.cancel.is_cancelled() {
            return Err(anyhow::anyhow!("cancelled").into());
        }
        Ok(self.cached(p).0)
    }
}

impl<'a, T: ComputeBackend> Gradient for QspProblem<'a, T> {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;
    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
        Ok(self.cached(p).1)
    }
}

struct ArgminObserverBridge {
    observer: Arc<dyn ProgressObserver>,
    stage: StageInfo,
}

fn translate_argmin_termination(
    termination_status: TerminationStatus,
) -> Result<TerminationReason> {
    Ok(match termination_status {
        argmin::core::TerminationStatus::Terminated(termination_reason) => match termination_reason
        {
            argmin::core::TerminationReason::MaxItersReached => {
                super::TerminationReason::MaxItersReached
            }
            argmin::core::TerminationReason::TargetCostReached => {
                super::TerminationReason::Converged
            }
            argmin::core::TerminationReason::Interrupt => super::TerminationReason::Other,
            argmin::core::TerminationReason::SolverConverged => super::TerminationReason::Converged,
            argmin::core::TerminationReason::Timeout => super::TerminationReason::Other,
            argmin::core::TerminationReason::SolverExit(_) => super::TerminationReason::Other,
        },
        argmin::core::TerminationStatus::NotTerminated => {
            anyhow::bail!("Solver exited with status NotTerminated!")
        }
    })
}

impl<I> Observe<I> for ArgminObserverBridge
where
    I: State<Float = f64>,
{
    fn observe_iter(&mut self, state: &I, _kv: &KV) -> Result<(), Error> {
        self.observer.on_iter(ProgressReport {
            stage: self.stage,
            iter: state.get_iter(),
            cost: state.get_cost(),
        });
        Ok(())
    }
}

impl<B: ComputeBackend> Solver<B> for BfgsOptions {
    fn run(
        &self,
        backend: &B,
        ctx: &SolverContext,
        xs: ndarray::Array1<f64>,
        map: PhaseMap,
        stage: StageInfo,
    ) -> SolveResult {
        let problem = QspProblem {
            backend,
            map,
            cancel: ctx.cancel.clone(),
            cache: RefCell::new(None),
        };
        let linesearch: MoreThuenteLineSearch<Array1<f64>, Array1<f64>, f64> =
            MoreThuenteLineSearch::new();
        let solver: LBFGS<_, Array1<f64>, Array1<f64>, f64> =
            LBFGS::new(linesearch, self.mem).with_tolerance_grad(self.tol_grad)?;

        let observer_bridge = ArgminObserverBridge {
            observer: ctx.observer.clone(),
            stage,
        };

        let exec_result = Executor::new(problem, solver)
            .configure(|state| state.param(xs).max_iters(self.max_iters))
            .add_observer(
                observer_bridge,
                argmin::core::observers::ObserverMode::Always,
            )
            .run();

        if ctx.cancel.is_cancelled() {
            return Err(SolveError::Cancelled);
        }
        let res = exec_result.map_err(|e| SolveError::Other(e.into()))?;
        let final_param = res
            .state
            .best_param
            .clone()
            .context("Failed to clone output array.")?;

        let final_cost = res.state.best_cost;
        let iters = res.state.iter;

        Ok(SolveOutcome::new(
            final_param,
            final_cost,
            iters,
            translate_argmin_termination(res.state.termination_status)?,
        ))
    }
}
