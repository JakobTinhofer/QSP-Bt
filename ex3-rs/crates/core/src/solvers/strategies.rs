use crate::{
    compute::ComputeBackend,
    solvers::{
        PhaseMap, SolveOutcome, SolveResult, Solver, StrategyError,
        observe::{SolverContext, StageInfo},
    },
};
use ndarray::{Array1, Axis, concatenate, s};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::f64::consts::PI;

pub fn solve_seeded<T: ComputeBackend, S: Solver<T> + ?Sized>(
    s: &S,
    backend: &T,
    ctx: &SolverContext,
    degree: usize,
    map: PhaseMap,
    seed: u64,
    init_perturb: f64,
    stage: Option<StageInfo>,
) -> SolveResult {
    let mut rng = StdRng::seed_from_u64(seed);
    let init: Array1<f64> = if init_perturb > 1e-8 {
        (0..degree + 1)
            .map(|_| rng.random_range(0.0..init_perturb))
            .collect()
    } else {
        Array1::zeros(degree + 1)
    };
    let stage = stage.unwrap_or(StageInfo {
        current_stage: 0,
        current_degree: degree,
        total_stages: 1,
    });
    ctx.observer.on_new_stage(stage);
    let res = s.run(backend, ctx, init, map, stage)?;
    ctx.observer.on_end_stage(stage, &res);
    Ok(res)
}

pub fn solve_cascade_seeded<T: ComputeBackend, S: Solver<T> + ?Sized>(
    s: &S,
    backend: &T,
    ctx: &SolverContext,
    n_steps: usize,
    final_degree: usize,
    map: PhaseMap,
    seed: u64,
    init_perturb: f64,
) -> SolveResult {
    if n_steps < 2 {
        return Err(StrategyError::new(format!(
            "cascade requires at least 2 steps, got {n_steps}"
        ))
        .into());
    }
    if final_degree < 2 * n_steps {
        return Err(StrategyError::new(format!(
            "cascade: final_degree ({final_degree}) must be >= 2 * n_steps ({})",
            2 * n_steps
        ))
        .into());
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
        phase_mag_sum: _,
    } = solve_seeded(
        s,
        backend,
        ctx,
        degrees[0],
        map,
        seed,
        init_perturb,
        Some(StageInfo {
            current_stage: 0,
            total_stages: n_steps,
            current_degree: degrees[0],
        }),
    )?;

    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1));

    for (i, &d) in degrees.iter().enumerate().skip(1) {
        let extra = (d + 1) - phases.len();
        if extra == 0 {
            continue;
        }
        let pertub: Array1<f64> = (0..extra).map(|_| rng.random_range(-0.01..0.01)).collect();
        let mut padded = Array1::zeros(phases.len() + extra);

        padded.slice_mut(s![..phases.len()]).assign(&phases);
        padded.slice_mut(s![phases.len()..]).assign(&pertub);
        let stage = StageInfo {
            current_stage: i,
            total_stages: n_steps,
            current_degree: d,
        };
        ctx.observer.on_new_stage(stage);
        let res = s.run(backend, ctx, padded, map, stage)?;
        ctx.observer.on_end_stage(stage, &res);
        (phases, cost, iterations, term_reason) =
            (res.phases, res.cost, res.iterations, res.term_reason);
    }
    Ok(SolveOutcome::new(phases, cost, iterations, term_reason))
}

pub fn solve_hotstart_seeded<T: ComputeBackend, S: Solver<T> + ?Sized>(
    s: &S,
    backend: &T,
    ctx: &SolverContext,
    hotstart_degree: usize,
    main_degree: usize,
    map: PhaseMap,
    seed: u64,
    init_perturb: f64,
) -> SolveResult {
    let SolveOutcome {
        phases,
        cost: _,
        iterations: _,
        term_reason: _,
        phase_mag_sum: _,
    } = solve_seeded(
        s,
        backend,
        ctx,
        hotstart_degree,
        map,
        seed,
        init_perturb,
        Some(StageInfo {
            current_stage: 0,
            current_degree: hotstart_degree,
            total_stages: 2,
        }),
    )?;

    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1));
    let pertub: Array1<f64> = (0..main_degree - hotstart_degree)
        .map(|_| rng.random_range(0.0..2.0 * PI))
        .collect();
    let padded = concatenate![Axis(0), phases, pertub];
    let stage = StageInfo {
        current_stage: 1,
        total_stages: 2,
        current_degree: main_degree,
    };
    ctx.observer.on_new_stage(stage);
    let res = s.run(backend, ctx, padded, map, stage)?;
    ctx.observer.on_end_stage(stage, &res);
    Ok(res)
}
