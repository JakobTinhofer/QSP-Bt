use clap::Args as ClapArgs;

use crate::{compute::ComputeBackend, solvers::Solver};

#[derive(ClapArgs, Debug, Clone)]
#[command(next_help_heading = "Levenberg-Marquardt Options")]
pub struct LmOptions {
    #[arg(id = "lm_max_iters", long = "lm-max-iters", default_value = "500")]
    pub max_iters: u64,
    #[arg(
        id = "lm_initial_lambda",
        long = "lm-initial-lambda",
        default_value = "1e-4"
    )]
    pub initial_lambda: f64,
    #[arg(id = "lm_tol", long = "lm-tol", default_value = "1e-10")]
    pub tol: f64,
}

impl<T: ComputeBackend> Solver<T> for LmOptions {
    fn run(&self, backend: &T, xs: ndarray::Array1<f64>) -> super::SolveOutcome {
        todo!()
    }
}
