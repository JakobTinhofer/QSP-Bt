use clap::ValueEnum;
use ndarray::Array1;
use num_complex::Complex64;
use rand::distr::{Distribution, Uniform};

use std::{f64::consts::PI, f64::consts::TAU};

use crate::utils::parse_usize_gt_0;

#[derive(Debug)]
pub struct TargetPoly {
    pub xs: Array1<f64>,
    pub ys: Array1<Complex64>,
    pub thetas: Array1<f64>,
    pattern: Option<TargetPattern>,
}

#[derive(Clone, Debug)]
pub enum TargetPattern {
    RandomPeaks,
    RandomPhases,
    GeneralizedParity { r: usize, k: usize },
    DataRepeating(Array1<Complex64>),
}

impl TargetPattern {
    pub fn parse(str: &str) -> Result<TargetPattern, String> {
        let trimmed = str.trim();

        let parts: Vec<&str> = trimmed.split(",").collect();
        match parts.as_slice() {
            ["rand-peaks" | "rand"] => Ok(TargetPattern::RandomPeaks),
            ["rand-phases"] => Ok(TargetPattern::RandomPhases),
            ["gp", r, k] => Ok(TargetPattern::GeneralizedParity {
                r: parse_usize_gt_0(r, "gp")?,
                k: k.parse()
                    .map_err(|e| format!("Failed to parse k. Err: {e}"))?,
            }),
            [first, ..] if matches!(*first, "rand-peaks" | "rand-phases" | "gp") => {
                Err(format!("Wrong number of arguments for '{}'", first))
            }
            _ => Ok(TargetPattern::DataRepeating(
                trimmed
                    .split(',')
                    .map(|part| {
                        let cleaned = part.replace(char::is_whitespace, "");
                        cleaned
                            .parse::<Complex64>()
                            .map_err(|e| format!("'{}': {}", part.trim(), e))
                    })
                    .collect::<Result<_, _>>()?,
            )),
        }
    }

    pub fn get_yhalf(&self, n: usize) -> Array1<Complex64> {
        let mut rng = rand::rng();
        match self {
            TargetPattern::RandomPeaks => {
                let bool_dist = rand::distr::Bernoulli::new(0.5).unwrap();
                (0..n)
                    .map(|_| { if bool_dist.sample(&mut rng) { 1. } else { 0. } }.into())
                    .collect()
            }
            TargetPattern::RandomPhases => {
                let p_dist = Uniform::new(0., TAU).unwrap();
                (0..n)
                    .map(|_| {
                        let phi: f64 = p_dist.sample(&mut rng);
                        Complex64::from_polar(1.0, phi)
                    })
                    .collect()
            }
            TargetPattern::GeneralizedParity { r, k } => (0..n)
                .map(|m| {
                    if ((m - k) % r) == 0 {
                        Complex64::ONE
                    } else {
                        Complex64::ZERO
                    }
                })
                .collect(),
            // extend repeating pattern
            TargetPattern::DataRepeating(a) => (0..n).map(|i| a[i % a.len()]).collect(),
        }
    }

    pub fn all_real(&self) -> bool {
        match self {
            TargetPattern::RandomPeaks | TargetPattern::GeneralizedParity { r: _, k: _ } => true,
            TargetPattern::RandomPhases => false,
            TargetPattern::DataRepeating(a) => a.iter().all(|c| c.im.abs() <= 1e-8),
        }
    }
}

impl TargetPoly {
    pub fn all_real(&self) -> bool {
        if let Some(p) = &self.pattern {
            p.all_real()
        } else {
            self.ys.iter().all(|c| c.im.abs() <= 1e-8)
        }
    }

    pub fn points_iter<'a>(&'a self) -> impl Iterator<Item = (&'a f64, &'a Complex64)> {
        self.xs.iter().zip(self.ys.iter())
    }

    pub fn xs_ys(&self) -> (&[f64], &[Complex64]) {
        (
            self.xs.as_slice().expect("xs must be contiguous"),
            self.ys.as_slice().expect("ys must be contiguous"),
        )
    }

    pub fn from_parts(xs: Array1<f64>, ys: Array1<Complex64>) -> Self {
        let thetas = xs.mapv(|x| x.acos());
        Self {
            xs,
            ys,
            thetas,
            pattern: None,
        }
    }

    pub fn new_forced_parity(target_y_half: Array1<Complex64>, parity: Parity) -> Self {
        let n_half = target_y_half.len();
        let mut s = Self {
            xs: Array1::zeros(2 * n_half),
            ys: Array1::zeros(2 * n_half),
            thetas: Array1::zeros(2 * n_half),
            pattern: None,
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

    pub fn from_pattern(bp: &TargetPattern, p: Parity, n: usize) -> Self {
        let mut t = Self::new_forced_parity(bp.get_yhalf(n), p);
        t.pattern = Some(bp.clone());
        t
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
