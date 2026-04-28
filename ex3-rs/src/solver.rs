use crate::compute::ComputeBackend;
use bfgs::bfgs;
use clap::ValueEnum;
use ndarray::{Array1, Axis, stack};
use num_complex::Complex64;
use rand::distr::{Distribution, Uniform};
use std::f64::consts::PI;

#[derive(Debug)]
pub struct TargetPoly {
    pub xs: Array1<f64>,
    pub ys: Array1<Complex64>,
    pub thetas: Array1<f64>,
}

impl TargetPoly {
    pub fn points_iter<'a>(&'a self) -> impl Iterator<Item = (&'a f64, &'a Complex64)> {
        self.xs.iter().zip(self.ys.iter())
    }

    pub fn from_parts(xs: Array1<f64>, ys: Array1<Complex64>) -> Self {
        let thetas = xs.mapv(|x| x.acos());
        Self { xs, ys, thetas }
    }

    pub fn new_forced_parity(target_y_half: Array1<Complex64>, parity: Parity) -> Self {
        let n_half = target_y_half.len();
        let mut s = Self {
            xs: Array1::zeros(2 * n_half),
            ys: Array1::zeros(2 * n_half),
            thetas: Array1::zeros(2 * n_half),
        };
        let parity_sign = match parity {
            Parity::Even => 1.,
            Parity::Odd => -1.,
        };
        for i in 0..n_half {
            let t = theta_k(i + 1, n_half);
            s.thetas[n_half + i] = t;
            s.thetas[n_half - i - 1] = PI - t;
            s.xs[n_half + i] = t.cos();
            s.xs[n_half - i - 1] = (PI - t).cos();
            s.ys[n_half + i] = target_y_half[i];
            s.ys[n_half - i - 1] = parity_sign * target_y_half[i];
        }
        s
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum Parity {
    Even,
    Odd,
}

fn theta_k(k: usize, n_half: usize) -> f64 {
    assert!(k > 0 && k <= n_half, "Range for k: 1..N_HALF");
    return (((k as f64) / ((n_half + 1) as f64)) * (PI / 2.).powf(2.)).sqrt();
}

pub fn solve<T: ComputeBackend>(backend: &T, degree: usize) -> (Array1<f64>, f64) {
    let t_dist = Uniform::new(0., 2. * PI).expect("Failed to create random distribution!");
    let mut rng = rand::rng();
    let init_phases = (0..degree + 1)
        .map(|_| t_dist.sample(&mut rng))
        .collect::<Array1<f64>>();
    let s = bfgs(
        init_phases,
        |p| backend.evaluate_f(p),
        |p| backend.evaluate_both(p).1,
    )
    .expect("Optimization failed!");
    let f_err = backend.evaluate_f(&s);
    (s, f_err)
}

pub fn solve_hotstart<T: ComputeBackend>(
    backend: &T,
    hotstart_degree: usize,
    main_degree: usize,
) -> Result<(Array1<f64>, f64), String> {
    let (initial, _) = solve(backend, hotstart_degree);
    let t_dist = Uniform::new(0., 2. * PI).expect("Failed to create random distribution!");
    let mut rng = rand::rng();
    let random_pertub = (0..main_degree - hotstart_degree)
        .map(|_| t_dist.sample(&mut rng))
        .collect::<Array1<f64>>();
    let padded = stack![Axis(0), initial, random_pertub];
    let s = bfgs(
        padded,
        |p| backend.evaluate_f(p),
        |p| backend.evaluate_both(p).1,
    )
    .expect("Optimize failed!");
    let f_err = backend.evaluate_f(&s);

    Ok((s, f_err))
}
