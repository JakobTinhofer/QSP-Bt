use ndarray::{Array1, Array2, ArrayView1};
use num_complex::Complex64;

use crate::target::TargetPoly;
pub mod cpu;

pub trait ComputeBackend {
    fn evaluate_f_grad(&self, phases: &ArrayView1<f64>) -> (f64, Array1<f64>);
    fn evaluate_res_jac(&self, phases: &ArrayView1<f64>) -> (Array1<f64>, Array2<f64>);
    fn evaluate_f(&self, phases: &ArrayView1<f64>) -> f64;
    fn evaluate_poly(&self, phases: &ArrayView1<f64>, xs: &ArrayView1<f64>) -> Array1<Complex64>;
    fn get_target(&self) -> &TargetPoly;
}
