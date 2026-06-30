use std::str::FromStr;

use ndarray::{Array1, Array2, ArrayView1};
use serde::{Deserialize, Serialize};

use crate::{
    compute::{regularized::RidgeRegularizedBackend, wx::WxBackend, wz::WzBackend},
    target::TargetPoly,
};

mod c2x2;
pub mod regularized;
pub mod wx;
pub mod wz;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendMode {
    SingleThread,
    MultiThread,
    Auto,
}

impl FromStr for BackendMode {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.trim().to_ascii_lowercase();
        match lower.as_str() {
            "single" | "single-thread" | "single_thread" | "s" => Ok(Self::SingleThread),
            "multi" | "multi-thread" | "multi_thread" | "m" => Ok(Self::MultiThread),
            "auto" | "a" => Ok(Self::Auto),
            _ => anyhow::bail!(format!("Could not convert {s} into BackendMode type!")),
        }
    }
}

pub trait ComputeBackend {
    fn evaluate_f_grad(&self, phases: &ArrayView1<f64>) -> (f64, Array1<f64>);
    fn evaluate_res_jac(&self, phases: &ArrayView1<f64>) -> (Array1<f64>, Array2<f64>);
    fn evaluate_f(&self, phases: &ArrayView1<f64>) -> f64;
    fn get_target(&self) -> &TargetPoly;
}

pub trait QspEvaluator {
    fn evaluate_poly(
        phases: &ArrayView1<f64>,
        xs: &ArrayView1<f64>,
    ) -> Array1<num_complex::Complex64>;
}

pub enum Backend {
    Wz(WzBackend),
    Wx(WxBackend),
    RidgeRegularizedWz(RidgeRegularizedBackend<WzBackend>),
    RidgeRegularizedWx(RidgeRegularizedBackend<WxBackend>),
}

impl ComputeBackend for Backend {
    fn evaluate_f(&self, p: &ArrayView1<f64>) -> f64 {
        match self {
            Backend::Wz(b) => b.evaluate_f(p),
            Backend::Wx(b) => b.evaluate_f(p),
            Backend::RidgeRegularizedWz(b) => b.evaluate_f(p),
            Backend::RidgeRegularizedWx(b) => b.evaluate_f(p),
        }
    }
    fn evaluate_f_grad(&self, p: &ArrayView1<f64>) -> (f64, Array1<f64>) {
        match self {
            Backend::Wz(b) => b.evaluate_f_grad(p),
            Backend::Wx(b) => b.evaluate_f_grad(p),
            Backend::RidgeRegularizedWz(b) => b.evaluate_f_grad(p),
            Backend::RidgeRegularizedWx(b) => b.evaluate_f_grad(p),
        }
    }
    fn evaluate_res_jac(&self, p: &ArrayView1<f64>) -> (Array1<f64>, Array2<f64>) {
        match self {
            Backend::Wz(b) => b.evaluate_res_jac(p),
            Backend::Wx(b) => b.evaluate_res_jac(p),
            Backend::RidgeRegularizedWz(b) => b.evaluate_res_jac(p),
            Backend::RidgeRegularizedWx(b) => b.evaluate_res_jac(p),
        }
    }

    fn get_target(&self) -> &TargetPoly {
        match self {
            Backend::Wz(b) => b.get_target(),
            Backend::Wx(b) => b.get_target(),
            Backend::RidgeRegularizedWz(b) => b.get_target(),
            Backend::RidgeRegularizedWx(b) => b.get_target(),
        }
    }
}
