use ndarray::{Array1, Array2};
use num_complex::Complex64;

use crate::target::TargetPoly;
pub mod cpu;

pub trait ComputeBackend {
    fn evaluate_f_grad(&self, phases: &Array1<f64>) -> (f64, Array1<f64>);
    fn evaluate_res_jac(&self, phases: &Array1<f64>) -> (Array1<f64>, Array2<f64>);
    fn evaluate_f(&self, phases: &Array1<f64>) -> f64;
    fn evaluate_poly(&self, phases: &Array1<f64>, xs: &Array1<f64>) -> Array1<Complex64>;
    fn get_target(&self) -> &TargetPoly;
}
