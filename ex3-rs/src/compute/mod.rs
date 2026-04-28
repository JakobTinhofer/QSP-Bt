use crate::solver::TargetPoly;
use ndarray::Array1;
use num_complex::Complex64;
pub mod cpu;

pub trait ComputeBackend {
    fn evaluate_both(&self, phases: &Array1<f64>) -> (f64, Array1<f64>);
    fn evaluate_f(&self, phases: &Array1<f64>) -> f64;
    fn evaluate_poly(&self, phases: &Array1<f64>, xs: &Array1<f64>) -> Array1<Complex64>;
    fn get_target(&self) -> &TargetPoly;
}
