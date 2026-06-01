use ndarray::{Array1, Array2, ArrayView1};
use num_complex::Complex64;

use crate::{
    compute::{cpu::CpuComputeBackend, regularized::RidgeRegularizedBackend},
    target::TargetPoly,
};
pub mod cpu;
pub mod regularized;

pub trait ComputeBackend {
    fn evaluate_f_grad(&self, phases: &ArrayView1<f64>) -> (f64, Array1<f64>);
    fn evaluate_res_jac(&self, phases: &ArrayView1<f64>) -> (Array1<f64>, Array2<f64>);
    fn evaluate_f(&self, phases: &ArrayView1<f64>) -> f64;
    fn evaluate_poly(phases: &ArrayView1<f64>, xs: &ArrayView1<f64>) -> Array1<Complex64>;
    fn get_target(&self) -> &TargetPoly;
}

pub enum Backend {
    Plain(CpuComputeBackend),
    RidgeRegularized(RidgeRegularizedBackend<CpuComputeBackend>),
}

impl Backend {
    pub fn match_regularization(b: CpuComputeBackend, l: Option<f64>) -> Self {
        match l {
            Some(lambda) => Self::RidgeRegularized(RidgeRegularizedBackend::new(b, lambda)),
            None => Self::Plain(b),
        }
    }
}

impl ComputeBackend for Backend {
    fn evaluate_f(&self, p: &ArrayView1<f64>) -> f64 {
        match self {
            Backend::Plain(b) => b.evaluate_f(p),
            Backend::RidgeRegularized(b) => b.evaluate_f(p),
        }
    }
    fn evaluate_f_grad(&self, p: &ArrayView1<f64>) -> (f64, Array1<f64>) {
        match self {
            Backend::Plain(b) => b.evaluate_f_grad(p),
            Backend::RidgeRegularized(b) => b.evaluate_f_grad(p),
        }
    }
    fn evaluate_res_jac(&self, p: &ArrayView1<f64>) -> (Array1<f64>, Array2<f64>) {
        match self {
            Backend::Plain(b) => b.evaluate_res_jac(p),
            Backend::RidgeRegularized(b) => b.evaluate_res_jac(p),
        }
    }
    fn evaluate_poly(phases: &ArrayView1<f64>, xs: &ArrayView1<f64>) -> Array1<Complex64> {
        CpuComputeBackend::evaluate_poly(phases, xs)
    }
    fn get_target(&self) -> &TargetPoly {
        match self {
            Backend::Plain(b) => b.get_target(),
            Backend::RidgeRegularized(b) => b.get_target(),
        }
    }
}
