use crate::{
    compute::ComputeBackend,
    solvers::{PhaseMap, SolveOutcome, Solver},
};
use anyhow::Result;
use ndarray::{Array1, Axis, concatenate, s};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::f64::consts::PI;

pub fn solve_seeded<T: ComputeBackend, S: Solver<T> + ?Sized>(
    s: &S,
    backend: &T,
    degree: usize,
    map: PhaseMap,
    seed: u64,
    init_perturb: f64,
) -> Result<SolveOutcome> {
    let mut rng = StdRng::seed_from_u64(seed);
    let init: Array1<f64> = if init_perturb > 1e-8 {
        (0..degree + 1)
            .map(|_| rng.random_range(0.0..init_perturb))
            .collect()
    } else {
        Array1::zeros(degree + 1)
    };
    s.run(backend, init, map)
}

pub fn solve_cascade_seeded<T: ComputeBackend, S: Solver<T> + ?Sized>(
    s: &S,
    backend: &T,
    n_steps: usize,
    final_degree: usize,
    map: PhaseMap,
    seed: u64,
    init_perturb: f64,
) -> Result<SolveOutcome> {
    if n_steps < 2 {
        anyhow::bail!(String::from("Cascade requires at least 2 Steps!"));
    }
    if final_degree < 2 * n_steps {
        anyhow::bail!(format!(
            "cascade: final_degree ({}) must be >= 2 * n_steps ({}) to keep parity-matched steps spaced by 2",
            final_degree,
            2 * n_steps
        ));
    }

    // TODO: get real parity
    // this is very incorrect rn
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

    let SolveOutcome {
        mut phases,
        mut cost,
        mut iterations,
        mut term_reason,
    } = solve_seeded(s, backend, degrees[0], map, seed, init_perturb)?;

    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1));

    for &d in degrees.iter().skip(1) {
        let extra = (d + 1) - phases.len();
        if extra == 0 {
            continue;
        }
        let pertub: Array1<f64> = (0..extra).map(|_| rng.random_range(-0.01..0.01)).collect();
        let mut padded = Array1::zeros(phases.len() + extra);
        padded.slice_mut(s![..phases.len()]).assign(&phases);
        padded.slice_mut(s![phases.len()..]).assign(&pertub);

        SolveOutcome {
            phases,
            cost,
            iterations,
            term_reason,
        } = s.run(backend, padded, map)?;
    }
    Ok(SolveOutcome {
        phases,
        cost,
        iterations,
        term_reason,
    })
}

pub fn solve_hotstart_seeded<T: ComputeBackend, S: Solver<T> + ?Sized>(
    s: &S,
    backend: &T,
    hotstart_degree: usize,
    main_degree: usize,
    map: PhaseMap,
    seed: u64,
    init_perturb: f64,
) -> Result<SolveOutcome> {
    let SolveOutcome {
        phases,
        cost: _,
        iterations: _,
        term_reason: _,
    } = solve_seeded(s, backend, hotstart_degree, map, seed, init_perturb)?;

    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1));
    let pertub: Array1<f64> = (0..main_degree - hotstart_degree)
        .map(|_| rng.random_range(0.0..2.0 * PI))
        .collect();
    let padded = concatenate![Axis(0), phases, pertub];

    let res = s.run(backend, padded, map)?;

    Ok(res)
}
