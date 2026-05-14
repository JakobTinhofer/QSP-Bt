use crate::{
    compute::ComputeBackend,
    solvers::{PhaseMap, SolveOutcome, Solver},
};
use anyhow::Result;
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{DMatrix, DVector, Dyn, Owned};
use ndarray::ArrayView1;
use serde::{Deserialize, Serialize};

struct QspLmProblem<'a, T: ComputeBackend> {
    p: DVector<f64>,
    r: DVector<f64>,
    j: DMatrix<f64>,
    backend: &'a T,
    map: PhaseMap,
}

impl<'a, T: ComputeBackend> QspLmProblem<'a, T> {
    fn new(backend: &'a T, init_xs: DVector<f64>, map: PhaseMap) -> Self {
        let (r, j) = Self::get_jac_res(backend, &init_xs, map);
        Self {
            backend,
            p: init_xs,
            r,
            j,
            map,
        }
    }

    fn get_jac_res(
        backend: &'a T,
        p: &DVector<f64>,
        map: PhaseMap,
    ) -> (DVector<f64>, DMatrix<f64>) {
        let mut v = ArrayView1::from(p.as_slice()).into_owned();
        map.apply(&mut v, backend.get_target())
            .map_err(|e| format!("Failed to apply map! Err: {e}"))
            .unwrap();
        let (r, jac) = backend.evaluate_res_jac(&v.view());
        let (r_vec, _) = r.into_raw_vec_and_offset();
        let (rows, cols) = jac.dim();
        let transposed = jac.t().as_standard_layout().into_owned();
        let (vec, _) = transposed.into_raw_vec_and_offset();
        (DVector::from_vec(r_vec), DMatrix::from_vec(rows, cols, vec))
    }
}
impl<'a, T: ComputeBackend> LeastSquaresProblem<f64, Dyn, Dyn> for QspLmProblem<'a, T> {
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, Dyn>;
    type ParameterStorage = Owned<f64, Dyn>;

    fn set_params(&mut self, x: &nalgebra::Vector<f64, Dyn, Self::ParameterStorage>) {
        self.p.copy_from(x);
        (self.r, self.j) = Self::get_jac_res(self.backend, &self.p, self.map);
    }

    fn params(&self) -> nalgebra::Vector<f64, Dyn, Self::ParameterStorage> {
        self.p.clone()
    }

    fn residuals(&self) -> Option<nalgebra::Vector<f64, Dyn, Self::ResidualStorage>> {
        Some(self.r.clone())
    }

    fn jacobian(&self) -> Option<nalgebra::Matrix<f64, Dyn, Dyn, Self::JacobianStorage>> {
        Some(self.j.clone())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LmOptions {
    pub max_iters: u64,
    pub initial_lambda: f64,
    pub tol: f64,
}

impl Default for LmOptions {
    fn default() -> Self {
        Self {
            max_iters: 500,
            initial_lambda: 1e-4,
            tol: 1e-10,
        }
    }
}

impl<T: ComputeBackend> Solver<T> for LmOptions {
    fn run(
        &self,
        backend: &T,
        xs: ndarray::Array1<f64>,
        map: PhaseMap,
    ) -> Result<super::SolveOutcome> {
        let (vec, _) = xs.into_raw_vec_and_offset();
        let problem = QspLmProblem::new(backend, DVector::from_vec(vec), map);
        let (_res, _rep) = LevenbergMarquardt::new().minimize(problem);
        Ok(SolveOutcome {
            term_reason: match _rep.termination {
                levenberg_marquardt::TerminationReason::Converged { ftol: _, xtol: _ } => {
                    super::TerminationReason::Converged
                }
                _ => super::TerminationReason::Other,
            },
            iterations: _rep.number_of_evaluations as u64,
            phases: ndarray::Array1::from_iter(_res.p.into_iter().map(|f| *f)),
            cost: _rep.objective_function,
        })
    }
}
