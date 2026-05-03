pub mod c2x2;
pub mod qsp;
use clap::ValueEnum;
use ndarray::Array1;
use num_complex::{Complex64, ComplexFloat};

use crate::{
    compute::{
        ComputeBackend,
        cpu::{
            c2x2::C2x2,
            qsp::{qsp_poly, z_rotation},
        },
    },
    solver::TargetPoly,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum BackendMode {
    SingleThread,
    MultiThread,
    Auto,
}

pub struct CpuComputeBackend {
    target: TargetPoly,
    mode: BackendMode,
}

impl CpuComputeBackend {
    pub fn new(p: TargetPoly, mode: BackendMode) -> Self {
        Self { target: p, mode }
    }

    fn evaluate_both_st(&self, phases: &Array1<f64>) -> (f64, Array1<f64>) {
        let d = phases.len() - 1;
        let r_z: Vec<C2x2> = phases.iter().map(|p| z_rotation(*p)).collect();
        let mut loss = 0.;
        let mut g = Array1::zeros(d + 1);
        let mut left_side = vec![C2x2::empty(); d + 1];
        let mut right_side = vec![C2x2::empty(); d + 1];
        for (x, f) in self.target.points_iter() {
            let wx = qsp::signal_operator(*x);
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
}

impl ComputeBackend for CpuComputeBackend {
    fn evaluate_both(&self, phases: &Array1<f64>) -> (f64, Array1<f64>) {
        if self.mode == BackendMode::SingleThread
            || self.mode == BackendMode::Auto
                && (phases.len() <= 100 && self.target.xs.len() <= 100)
        {
            return self.evaluate_both_st(phases);
        }

        use rayon::prelude::*;

        let d = phases.len() - 1;
        let r_z: Vec<C2x2> = phases.iter().map(|p| z_rotation(*p)).collect();

        let half_i = Complex64::new(0., 0.5);
        let alphas: Vec<Complex64> = r_z.iter().map(|rz| half_i * rz.get(0, 0)).collect();
        let betas: Vec<Complex64> = r_z.iter().map(|rz| -half_i * rz.get(1, 1)).collect();

        let points: Vec<(f64, Complex64)> =
            self.target.points_iter().map(|(x, y)| (*x, *y)).collect();

        let (loss, g) = points
            .par_iter()
            .map(|(x, f)| {
                let s = (1.0 - x * x).max(0.0).sqrt();
                let x = *x;

                let mut left_side = vec![C2x2::empty(); d + 1];
                let mut right_side = vec![C2x2::empty(); d + 1];
                let mut m_pre = vec![C2x2::empty(); d + 1];

                for k in 0..d + 1 {
                    let a = r_z[k].get(0, 0);
                    let b = r_z[k].get(1, 1);
                    m_pre[k] = C2x2::new([
                        [
                            Complex64::new(x * a.re, x * a.im),
                            Complex64::new(-s * b.im, s * b.re),
                        ],
                        [
                            Complex64::new(-s * a.im, s * a.re),
                            Complex64::new(x * b.re, x * b.im),
                        ],
                    ]);
                }

                left_side[0] = r_z[0];
                right_side[d] = C2x2::eye();
                for k in 1..d + 1 {
                    left_side[k] = left_side[k - 1] * m_pre[k];
                    right_side[d - k] = m_pre[d - k + 1] * right_side[d - k + 1];
                }

                let u = left_side[d];
                let r = u.get(0, 0) - f;
                let loss_term = r.norm_sqr();

                let mut g_local = Array1::<f64>::zeros(d + 1);
                let m00 = half_i * u.get(0, 0);
                g_local[0] = 2.0 * (r.conj() * m00).re;

                let r_conj = r.conj();
                for k in 1..d + 1 {
                    let l00 = left_side[k - 1].get(0, 0);
                    let l01 = left_side[k - 1].get(0, 1);
                    let r00 = right_side[k].get(0, 0);
                    let r10 = right_side[k].get(1, 0);

                    let xl00 = Complex64::new(x * l00.re, x * l00.im);
                    let xl01 = Complex64::new(x * l01.re, x * l01.im);
                    let is_l00 = Complex64::new(-s * l00.im, s * l00.re);
                    let is_l01 = Complex64::new(-s * l01.im, s * l01.re);

                    let a_term = xl00 + is_l01;
                    let b_term = is_l00 + xl01;

                    let m00 = alphas[k] * r00 * a_term + betas[k] * r10 * b_term;
                    g_local[k] = 2.0 * (r_conj * m00).re;
                }

                (loss_term, g_local)
            })
            .reduce(
                || (0.0_f64, Array1::<f64>::zeros(d + 1)),
                |(la, ga), (lb, gb)| (la + lb, ga + gb),
            );

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::TargetPoly;
    use ndarray::Array1;
    use num_complex::Complex64;
    use rand::{RngExt, SeedableRng, rngs::StdRng};
    use std::f64::consts::PI;

    /// Build a small deterministic backend for testing.
    fn make_test_backend(i_count: usize) -> CpuComputeBackend {
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        let xs: Array1<f64> = (0..i_count)
            .map(|k| ((k as f64 + 0.5) / i_count as f64 * PI).cos())
            .collect();
        let ys: Array1<Complex64> = (0..i_count)
            .map(|_| Complex64::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)))
            .collect();
        let target = TargetPoly::from_parts(xs, ys);
        CpuComputeBackend::new(target, BackendMode::MultiThread)
    }

    fn random_phases(n: usize, seed: u64) -> Array1<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n).map(|_| rng.random_range(0.0..2.0 * PI)).collect()
    }

    /// Test 1: new implementation matches the reference implementation.
    #[test]
    fn evaluate_both_matches_reference() {
        let backend = make_test_backend(20);

        // Run for a few different phase vector sizes and seeds
        for &n in &[5_usize, 16, 50] {
            for seed in 0u64..5 {
                let phases = random_phases(n, seed);

                let (loss_new, grad_new) = backend.evaluate_both(&phases);
                let (loss_ref, grad_ref) = backend.evaluate_both_st(&phases);

                let loss_diff = (loss_new - loss_ref).abs();
                let loss_rel = loss_diff / loss_ref.abs().max(1e-300);
                assert!(
                    loss_rel < 1e-12,
                    "Loss mismatch at n={}, seed={}: new={:.16e}, ref={:.16e}, rel_err={:.2e}",
                    n,
                    seed,
                    loss_new,
                    loss_ref,
                    loss_rel
                );

                for (k, (gn, gr)) in grad_new.iter().zip(grad_ref.iter()).enumerate() {
                    let diff = (gn - gr).abs();
                    let scale = gn.abs().max(gr.abs()).max(1e-12);
                    let rel = diff / scale;
                    assert!(
                        rel < 1e-12,
                        "Grad mismatch at n={}, seed={}, k={}: new={:.16e}, ref={:.16e}, rel_err={:.2e}",
                        n,
                        seed,
                        k,
                        gn,
                        gr,
                        rel
                    );
                }
            }
        }
    }

    /// Test 2: gradient agrees with central finite differences of evaluate_f.
    /// This is independent of evaluate_both_reference and catches errors that
    /// might exist in both implementations.
    #[test]
    fn gradient_matches_finite_differences() {
        let backend = make_test_backend(15);

        for &n in &[4_usize, 12] {
            for seed in 0u64..3 {
                let phases = random_phases(n, seed);
                let (_, grad_analytic) = backend.evaluate_both(&phases);

                // Central differences: g[k] ≈ (f(phi + h*e_k) - f(phi - h*e_k)) / (2h)
                // h chosen as compromise between truncation (O(h^2)) and roundoff (O(eps/h)).
                let h = 1e-6;
                for k in 0..n {
                    let mut phi_plus = phases.clone();
                    let mut phi_minus = phases.clone();
                    phi_plus[k] += h;
                    phi_minus[k] -= h;

                    let f_plus = backend.evaluate_f(&phi_plus);
                    let f_minus = backend.evaluate_f(&phi_minus);
                    let g_numeric = (f_plus - f_minus) / (2.0 * h);

                    let diff = (grad_analytic[k] - g_numeric).abs();
                    let scale = grad_analytic[k].abs().max(g_numeric.abs()).max(1e-8);
                    let rel = diff / scale;

                    // Central differences with h=1e-6 give ~h^2 = 1e-12 truncation
                    // error in the math, plus ~eps/h = 1e-10 roundoff. So tolerance
                    // ~1e-7 is comfortable; tighter would risk false positives.
                    assert!(
                        rel < 1e-7,
                        "FD gradient mismatch at n={}, seed={}, k={}: analytic={:.10e}, numeric={:.10e}, rel_err={:.2e}",
                        n,
                        seed,
                        k,
                        grad_analytic[k],
                        g_numeric,
                        rel
                    );
                }
            }
        }
    }
}
