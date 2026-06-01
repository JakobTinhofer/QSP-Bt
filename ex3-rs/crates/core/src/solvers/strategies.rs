use crate::{
    compute::ComputeBackend,
    solvers::{
        PhaseMap, SolveOutcome, SolveResult, Solver, StrategyError,
        configuration::PhaseGenerator,
        observe::{SolverContext, StageInfo},
    },
};
pub fn solve<T: ComputeBackend, S: Solver<T> + ?Sized>(
    s: &S,
    backend: &T,
    ctx: &SolverContext,
    degree: usize,
    map: PhaseMap,
    i: PhaseGenerator,
    stage: Option<StageInfo>,
) -> SolveResult {
    let init = i.get(degree + 1);
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
    init: PhaseGenerator,
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
    } = solve(
        s,
        backend,
        ctx,
        degrees[0],
        map,
        init.clone(),
        Some(StageInfo {
            current_stage: 0,
            total_stages: n_steps,
            current_degree: degrees[0],
        }),
    )?;

    for (i, &d) in degrees.iter().enumerate().skip(1) {
        init.resize(&mut phases, d + 1);

        let stage = StageInfo {
            current_stage: i,
            total_stages: n_steps,
            current_degree: d,
        };
        ctx.observer.on_new_stage(stage);
        let res = s.run(backend, ctx, phases, map, stage)?;
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
    init: PhaseGenerator,
) -> SolveResult {
    let SolveOutcome {
        mut phases,
        cost: _,
        iterations: _,
        term_reason: _,
        phase_mag_sum: _,
    } = solve(
        s,
        backend,
        ctx,
        hotstart_degree,
        map,
        init.clone(),
        Some(StageInfo {
            current_stage: 0,
            current_degree: hotstart_degree,
            total_stages: 2,
        }),
    )?;

    init.resize(&mut phases, main_degree + 1);
    let stage = StageInfo {
        current_stage: 1,
        total_stages: 2,
        current_degree: main_degree,
    };
    ctx.observer.on_new_stage(stage);
    let res = s.run(backend, ctx, phases, map, stage)?;
    ctx.observer.on_end_stage(stage, &res);
    Ok(res)
}
