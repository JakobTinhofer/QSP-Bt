use crate::compute::ComputeBackend;
use argmin::{
    core::{CostFunction, Executor, Gradient},
    solver::{linesearch::MoreThuenteLineSearch, quasinewton::LBFGS},
};
use argmin_math::Error;
use clap::ValueEnum;
use ndarray::{Array1, Axis, concatenate, s};
use num_complex::Complex64;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::{cell::RefCell, f64::consts::PI};

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

struct QspProblem<'a, T: ComputeBackend> {
    backend: &'a T,
    cache: RefCell<Option<(Array1<f64>, f64, Array1<f64>)>>,
}

impl<'a, T: ComputeBackend> QspProblem<'a, T> {
    fn cached(&self, p: &Array1<f64>) -> (f64, Array1<f64>) {
        {
            let c = self.cache.borrow();
            if let Some((cp, cost, grad)) = c.as_ref() {
                if cp == p {
                    return (*cost, grad.clone());
                }
            }
        } // borrow dropped before borrow_mut

        let (cost, grad) = self.backend.evaluate_both(p);
        *self.cache.borrow_mut() = Some((p.clone(), cost, grad.clone()));
        (cost, grad)
    }
}

impl<'a, T: ComputeBackend> CostFunction for QspProblem<'a, T> {
    type Param = Array1<f64>;
    type Output = f64;
    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        Ok(self.cached(p).0)
    }
}

impl<'a, T: ComputeBackend> Gradient for QspProblem<'a, T> {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;
    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
        Ok(self.cached(p).1)
    }
}

fn theta_k(k: usize, n_half: usize) -> f64 {
    assert!(k > 0 && k <= n_half, "Range for k: 1..N_HALF");
    return (((k as f64) / ((n_half + 1) as f64)) * (PI / 2.).powf(2.)).sqrt();
}

fn run_lbfgs<T: ComputeBackend>(
    backend: &T,
    init_phases: Array1<f64>,
    max_iters: u64,
    tol_grad: f64,
    lbfgs_mem: usize,
) -> Result<(Array1<f64>, f64, u64), Error> {
    let problem = QspProblem {
        backend,
        cache: RefCell::new(None),
    };
    let linesearch: MoreThuenteLineSearch<Array1<f64>, Array1<f64>, f64> =
        MoreThuenteLineSearch::new();
    let solver: LBFGS<_, Array1<f64>, Array1<f64>, f64> =
        LBFGS::new(linesearch, lbfgs_mem).with_tolerance_grad(tol_grad)?;

    let res = Executor::new(problem, solver)
        .configure(|state| state.param(init_phases).max_iters(max_iters))
        .run()?;

    let final_param = res.state.best_param.clone().unwrap();
    let final_cost = res.state.best_cost;
    let iters = res.state.iter;

    Ok((final_param, final_cost, iters))
}
#[derive(Debug, Clone, Copy)]
pub enum SolveMode {
    /// Direct solve at the given degree.
    Simple(usize),
    /// Solve at the first degree, then warm-start a solve at the second.
    /// Constraint: first < second.
    Hotstart(usize, usize),
    /// N cascading solves, gradually increasing degree from a small initial value
    /// up to the final degree, each warm-starting from the previous.
    Cascade(usize, usize),
}

pub fn solve_with_mode<T: ComputeBackend>(
    backend: &T,
    mode: &SolveMode,
    max_iters: u64,
    tol_grad: f64,
    lbfgs_mem: usize,
) -> Result<(Array1<f64>, f64), String> {
    let seed = rand::random::<u64>();
    match mode {
        SolveMode::Simple(d) => {
            let (sol, cost) = solve_seeded(backend, *d, seed, max_iters, tol_grad, lbfgs_mem);
            Ok((sol, cost))
        }
        SolveMode::Hotstart(s, d) => {
            solve_hotstart_seeded(backend, *s, *d, seed, max_iters, tol_grad, lbfgs_mem)
        }
        SolveMode::Cascade(n, d) => {
            solve_cascade_seeded(backend, *n, *d, seed, max_iters, tol_grad, lbfgs_mem)
        }
    }
}

pub fn solve<T: ComputeBackend>(
    backend: &T,
    degree: usize,
    max_iters: u64,
    tol_grad: f64,
    lbfgs_mem: usize,
) -> (Array1<f64>, f64) {
    solve_seeded(
        backend,
        degree,
        rand::random::<u64>(),
        max_iters,
        tol_grad,
        lbfgs_mem,
    )
}

pub fn solve_seeded<T: ComputeBackend>(
    backend: &T,
    degree: usize,
    seed: u64,
    max_iters: u64,
    tol_grad: f64,
    lbfgs_mem: usize,
) -> (Array1<f64>, f64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let init: Array1<f64> = (0..degree + 1)
        .map(|_| rng.random_range(0.0..2.0 * PI))
        .collect();

    let (sol, cost, iters) =
        run_lbfgs(backend, init, max_iters, tol_grad, lbfgs_mem).expect("Optimization failed!");
    eprintln!("[L-BFGS] {} iterations, cost {:e}", iters, cost);

    let f_err = backend.evaluate_f(&sol);
    (sol, f_err)
}

pub fn solve_hotstart<T: ComputeBackend>(
    backend: &T,
    hotstart_degree: usize,
    main_degree: usize,
    max_iters: u64,
    tol_grad: f64,
    lbfgs_mem: usize,
) -> Result<(Array1<f64>, f64), String> {
    solve_hotstart_seeded(
        backend,
        hotstart_degree,
        main_degree,
        rand::random::<u64>(),
        max_iters,
        tol_grad,
        lbfgs_mem,
    )
}

pub fn solve_cascade_seeded<T: ComputeBackend>(
    backend: &T,
    n_steps: usize,
    final_degree: usize,
    seed: u64,
    max_iters: u64,
    tol_grad: f64,
    lbfgs_mem: usize,
) -> Result<(Array1<f64>, f64), String> {
    if n_steps < 2 {
        return Err("cascade requires at least 2 steps".into());
    }
    if final_degree < 2 * n_steps {
        return Err(format!(
            "cascade: final_degree ({}) must be >= 2 * n_steps ({}) to keep parity-matched steps spaced by 2",
            final_degree,
            2 * n_steps
        ));
    }

    let target_parity = final_degree % 2;

    let mut degrees: Vec<usize> = (1..=n_steps)
        .map(|k| {
            let ideal = (final_degree * k) / n_steps;
            let snapped = if ideal % 2 == target_parity {
                ideal
            } else {
                ideal.saturating_sub(1)
            };
            let min_deg = if target_parity == 0 { 2 } else { 3 };
            snapped.max(min_deg)
        })
        .collect();

    *degrees.last_mut().unwrap() = final_degree;
    degrees.dedup();

    eprintln!(
        "[cascade] schedule (parity {}): {:?}",
        if target_parity == 0 { "even" } else { "odd" },
        degrees
    );

    let (mut current, mut last_cost) =
        solve_seeded(backend, degrees[0], seed, max_iters, tol_grad, lbfgs_mem);
    eprintln!(
        "[cascade] step 0: degree {} → cost {:e}",
        degrees[0], last_cost
    );

    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1));

    for (i, &d) in degrees.iter().enumerate().skip(1) {
        let extra = (d + 1) - current.len();
        if extra == 0 {
            continue;
        }
        let pertub: Array1<f64> = (0..extra).map(|_| rng.random_range(-0.01..0.01)).collect();
        let mut padded = Array1::zeros(current.len() + extra);
        padded.slice_mut(s![..current.len()]).assign(&current);
        padded.slice_mut(s![current.len()..]).assign(&pertub);

        let (sol, cost, iters) = run_lbfgs(backend, padded, max_iters, tol_grad, lbfgs_mem)
            .map_err(|e| format!("cascade step {} failed: {}", i, e))?;
        eprintln!(
            "[cascade] step {}: degree {} → cost {:e} ({} iters)",
            i, d, cost, iters
        );
        current = sol;
        last_cost = cost;
    }

    Ok((current, last_cost))
}

pub fn solve_hotstart_seeded<T: ComputeBackend>(
    backend: &T,
    hotstart_degree: usize,
    main_degree: usize,
    seed: u64,
    max_iters: u64,
    tol_grad: f64,
    lbfgs_mem: usize,
) -> Result<(Array1<f64>, f64), String> {
    let (initial, _) = solve_seeded(
        backend,
        hotstart_degree,
        seed,
        max_iters,
        tol_grad,
        lbfgs_mem,
    );

    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1));
    let pertub: Array1<f64> = (0..main_degree - hotstart_degree)
        .map(|_| rng.random_range(0.0..2.0 * PI))
        .collect();
    let padded = concatenate![Axis(0), initial, pertub];

    let (sol, cost, iters) = run_lbfgs(backend, padded, max_iters, tol_grad, lbfgs_mem)
        .map_err(|e| format!("L-BFGS failed: {}", e))?;
    eprintln!("[L-BFGS hotstart] {} iterations, cost {:e}", iters, cost);

    let f_err = backend.evaluate_f(&sol);
    Ok((sol, f_err))
}
