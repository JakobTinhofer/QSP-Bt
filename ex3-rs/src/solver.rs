use crate::{
    c2x2::C2x2,
    qsp::{qsp_poly, signal_operator, z_rotation},
};
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

pub fn objective(phases: &[f64], target: &TargetPoly) -> f64 {
    target
        .points_iter()
        .zip(qsp_poly(phases, &target.xs.as_slice().unwrap()))
        .map(|((_, y), p)| (p.re - y.re).powf(2.) + (p.im - y.im).powf(2.))
        .sum()
}

pub fn grad_objective(phases: &[f64], target: &TargetPoly) -> Array1<f64> {
    let d = phases.len() - 1;
    let r_z: Vec<C2x2> = phases.iter().map(|p| z_rotation(*p)).collect();
    let mut grad = Array1::zeros(d + 1);
    let mut left_side = vec![C2x2::empty(); d + 1];
    let mut right_side = vec![C2x2::empty(); d + 1];
    for (x, f) in target.points_iter() {
        let wx = signal_operator(*x);
        left_side[0] = r_z[0];
        right_side[d] = C2x2::eye();
        for k in 1..d + 1 {
            left_side[k] = left_side[k - 1] * wx * r_z[k];
            right_side[d - k] = wx * r_z[d - k + 1] * right_side[d - k + 1];
        }

        let u = left_side[d];
        let r = u.get(0, 0) - f;
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
            grad[k] += 2. * (r.conj() * m.get(0, 0)).re;
        }
    }
    grad
}

pub fn solve(target: &TargetPoly, degree: usize) -> (Array1<f64>, f64) {
    let t_dist = Uniform::new(0., 2. * PI).expect("Failed to create random distribution!");
    let mut rng = rand::rng();
    let init_phases = (0..degree + 1)
        .map(|_| t_dist.sample(&mut rng))
        .collect::<Array1<f64>>();
    let s = bfgs(
        init_phases,
        |p| objective(p.as_slice().unwrap(), target),
        |p| grad_objective(p.as_slice().unwrap(), target),
    )
    .expect("Optimization failed!");
    let f_err = objective(s.as_slice().unwrap(), target);
    (s, f_err)
}

pub fn solve_hotstart(
    target: &TargetPoly,
    hotstart_degree: usize,
    main_degree: usize,
    retry_tolerance: f64,
    max_iter: usize,
) -> Result<(Array1<f64>, f64), String> {
    let mut f_err = f64::MAX;
    let mut i = 0;
    let s = loop {
        let (initial, _) = solve(target, hotstart_degree);
        let t_dist = Uniform::new(0., 2. * PI).expect("Failed to create random distribution!");
        let mut rng = rand::rng();
        let random_pertub = (0..main_degree - hotstart_degree)
            .map(|_| t_dist.sample(&mut rng))
            .collect::<Array1<f64>>();
        let padded = stack![Axis(0), initial, random_pertub];
        let s = bfgs(
            padded,
            |p| objective(p.as_slice().unwrap(), target),
            |p| grad_objective(p.as_slice().unwrap(), target),
        )
        .expect("Optimize failed!");
        f_err = objective(s.as_slice().unwrap(), target);
        if f_err <= retry_tolerance {
            break s;
        } else if i >= max_iter {
            return Err(String::from("Maximum iterations reached"));
        } else {
            println!(
                "[SOLVER]: Iter {i}/{max_iter} has a remaining error of {f_err:e}, which is larger than target tolerance {retry_tolerance:e}."
            );
            i += 1;
        }
    };

    Ok((s, f_err))
}
