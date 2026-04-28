pub mod c2x2;
pub mod qsp;
use ndarray::Array1;
use num_complex::{Complex64, ComplexFloat};

use crate::{
    compute::{
        ComputeBackend,
        cpu::{
            c2x2::C2x2,
            qsp::{qsp_poly, signal_operator, z_rotation},
        },
    },
    solver::TargetPoly,
};

pub struct CpuComputeBackend {
    target: TargetPoly,
}

impl CpuComputeBackend {
    pub fn new(p: TargetPoly) -> Self {
        Self { target: p }
    }
}

impl ComputeBackend for CpuComputeBackend {
    fn evaluate_both(&self, phases: &Array1<f64>) -> (f64, Array1<f64>) {
        let d = phases.len() - 1;
        let r_z: Vec<C2x2> = phases.iter().map(|p| z_rotation(*p)).collect();
        let mut loss = 0.;
        let mut g = Array1::zeros(d + 1);
        let mut left_side = vec![C2x2::empty(); d + 1];
        let mut right_side = vec![C2x2::empty(); d + 1];
        for (x, f) in self.target.points_iter() {
            let wx = signal_operator(*x);
            left_side[0] = r_z[0];
            right_side[d] = C2x2::eye();
            for k in 1..d + 1 {
                left_side[k] = left_side[k - 1] * wx * r_z[k];
                right_side[d - k] = wx * r_z[d - k + 1] * right_side[d - k + 1];
            }

            let u = left_side[d];
            let r = u.get(0, 0) - f;
            loss += r.norm_sqr();

            let pauli_z_i2 = C2x2::new([
                [Complex64::new(0., 0.5), (0.).into()],
                [(0.).into(), Complex64::new(0., -0.5).into()],
            ]);
            for k in 0..d + 1 {
                let m = if k == 0 {
                    pauli_z_i2 * u
                } else {
                    left_side[k - 1] * wx * pauli_z_i2 * r_z[k] * right_side[k]
                };
                g[k] += 2. * (r.conj() * m.get(0, 0)).re;
            }
        }
        (loss, g)
    }

    fn evaluate_f(&self, phases: &Array1<f64>) -> f64 {
        self.target
            .points_iter()
            .zip(
                qsp_poly(
                    phases.as_slice().unwrap(),
                    self.target.xs.as_slice().unwrap(),
                )
                .iter(),
            )
            .map(|((_, y), p)| (p.re - y.re).powf(2.) + (p.im - y.im).powf(2.))
            .sum()
    }

    fn get_target(&self) -> &TargetPoly {
        &self.target
    }

    fn evaluate_poly(&self, phases: &Array1<f64>, xs: &Array1<f64>) -> Array1<Complex64> {
        qsp_poly(phases.as_slice().unwrap(), xs.as_slice().unwrap())
    }
}
