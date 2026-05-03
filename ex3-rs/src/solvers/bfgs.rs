use crate::{
    compute::ComputeBackend,
    solvers::{SolveOutcome, Solver},
};
use argmin::{
    core::{CostFunction, Executor, Gradient},
    solver::{linesearch::MoreThuenteLineSearch, quasinewton::LBFGS},
};
use argmin_math::Error;
use clap::Args as ClapArgs;
use ndarray::Array1;
use std::cell::RefCell;

#[derive(ClapArgs, Debug, Clone)]
#[command(next_help_heading = "L-BFGS Options")]
pub struct BfgsOptions {
    #[arg(
        id = "bfgs_max_iters",
        long = "bfgs-max-iters",
        default_value = "500000"
    )]
    pub max_iters: u64,
    #[arg(id = "bfgs_mem", long = "bfgs-mem", default_value = "10")]
    pub mem: usize,
    #[arg(id = "bfgs_tol_grad", long = "bfgs-tol-grad", default_value = "1e-8")]
    pub tol_grad: f64,
}

struct QspProblem<'a, T: ComputeBackend> {
    backend: &'a T,
    cache: RefCell<Option<(Array1<f64>, f64, Array1<f64>)>>,
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

        let (cost, grad) = self.backend.evaluate_f_grad(p);
        *self.cache.borrow_mut() = Some((p.clone(), cost, grad.clone()));
        (cost, grad)
    }
}

impl<'a, T: ComputeBackend> CostFunction for QspProblem<'a, T> {
    type Param = Array1<f64>;
    type Output = f64;
    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
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

impl<B: ComputeBackend> Solver<B> for BfgsOptions {
    fn run(&self, backend: &B, xs: ndarray::Array1<f64>) -> SolveOutcome {
        let problem = QspProblem {
            backend,
            cache: RefCell::new(None),
        };
        let linesearch: MoreThuenteLineSearch<Array1<f64>, Array1<f64>, f64> =
            MoreThuenteLineSearch::new();
        let solver: LBFGS<_, Array1<f64>, Array1<f64>, f64> = LBFGS::new(linesearch, self.mem)
            .with_tolerance_grad(self.tol_grad)
            .expect("Error on creating LBGFS!");

        let res = Executor::new(problem, solver)
            .configure(|state| state.param(xs).max_iters(self.max_iters))
            .run()
            .expect("Solver raised an error!");

        let final_param = res.state.best_param.clone().unwrap();
        let final_cost = res.state.best_cost;
        let iters = res.state.iter;

        SolveOutcome {
            phases: final_param,
            cost: final_cost,
            iterations: iters,
            term_reason: super::TerminationReason::Other,
        }
    }
}
