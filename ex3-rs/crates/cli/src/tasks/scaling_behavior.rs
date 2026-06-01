use crate::cli::{BLUE, GREEN, PhaseMapArg, RESET, YELLOW};
use clap::Args;
use qsp_rs_core::{
    compute::{
        Backend,
        cpu::{BackendMode, CpuComputeBackend},
    },
    solvers::{configuration::PhaseMap, observe::SolverContext},
    target::{Parity, TargetPoly},
};
use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::tasks::TaskTrait;

#[derive(Clone, Args, Serialize, Deserialize)]
pub struct ScalingBehaviorTask {
    #[arg(short = 's', default_value = "5")]
    length_step: usize,
    #[arg(short = 'e', default_value = "1e-6")]
    max_error: f64,
    #[arg(short = 'n', default_value = "1")]
    avg_n: usize,
    #[arg(long, default_value = "1")]
    start_from_n: usize,
    #[arg(short = 'R', default_value = "5.5")]
    starting_phase_ratio: f64,
    #[arg(long = "steps", default_value = "-1")]
    run_steps: i32,
    #[arg(long)]
    dont_force_parity: bool,
}

impl TaskTrait for ScalingBehaviorTask {
    fn execute(&self, cfg: crate::cli::ProgramConfig) -> anyhow::Result<()> {
        let t = cfg.target;
        let b = BackendMode::from(cfg.backend_mode);
        let s = cfg.solver;
        let p = Parity::from(t.parity);

        let mut current_target_length = self.start_from_n;
        let mut step = 0;

        while self.run_steps < 0 || step < self.run_steps {
            let mut lower_edge = 1;
            let mut upper_edge =
                (current_target_length as f64 * self.starting_phase_ratio) as usize;

            let mut last_vals: Option<(usize, usize, f64, f64, f64, f64)> = None;
            let mut last_low: Option<(usize, usize, f64, f64, f64, f64)> = None;
            while lower_edge < upper_edge {
                let d = (upper_edge - lower_edge) / 2 as usize + lower_edge;
                let parity_adjusted_d = if !self.dont_force_parity {
                    match (s.strategy.phase_map, t.target_pattern.all_real()) {
                        (PhaseMapArg::Mirror, _) | (PhaseMapArg::MirrorIfPossible, true) => d,
                        _ => match (d % 2, p) {
                            (0, Parity::Odd) | (1, Parity::Even) => d + 1,
                            _ => d,
                        },
                    }
                } else {
                    d
                };

                let target = TargetPoly::from_pattern(&t.target_pattern, p, current_target_length)?;
                let backend = Backend::match_regularization(
                    CpuComputeBackend::new(target, b),
                    s.strategy.regularization_lambda,
                );
                let mode = s.strategy.mode.rescale(parity_adjusted_d);
                let mut nr_of_phases = 0;
                let (mut avg_cost, mut avg_rt, mut avg_iter, mut avg_tot_phase) =
                    (0.0, 0.0, 0.0, 0.0);

                for i in 0..self.avg_n {
                    let now = Instant::now();
                    let res = s.get_solver::<Backend>().solve(
                        &backend,
                        &SolverContext::default(),
                        mode,
                        PhaseMap::from(s.strategy.phase_map),
                        s.strategy.phase_init.clone(),
                    )?;
                    avg_cost += res.cost / self.avg_n as f64;
                    let elapsed = now.elapsed();
                    avg_rt += elapsed.as_millis() as f64 / self.avg_n as f64;
                    avg_iter += res.iterations as f64 / self.avg_n as f64;
                    avg_tot_phase += res.phase_mag_sum / self.avg_n as f64;
                    nr_of_phases = res.phases.len();

                    let digits = (self.avg_n / 10) as usize + 1;
                    eprintln!(
                        "[{BLUE}i{RESET}] ({:>digits$}/{:>digits$}) n={current_target_length} d={d}: cost={:e} tot_phase:{} iter={} rt={elapsed:?}",
                        i + 1,
                        self.avg_n,
                        res.cost,
                        res.phase_mag_sum,
                        res.iterations
                    );
                }
                last_vals = Some((
                    parity_adjusted_d,
                    nr_of_phases,
                    avg_cost,
                    avg_rt,
                    avg_iter,
                    avg_tot_phase,
                ));
                if avg_cost < self.max_error {
                    upper_edge = d;
                    last_low = last_vals.clone();
                } else {
                    lower_edge = d + 1;
                }
            }

            if let Some((
                last_d,
                last_nr_phases,
                last_avg_cost,
                last_avg_rt,
                last_avg_iter,
                last_tot_phase,
            )) = last_low
            {
                eprintln!(
                    "[{GREEN}+{RESET}] Reached error limit. Last: d={last_d} cost={last_avg_cost:e} phases_n={last_nr_phases} tot_phase={last_tot_phase} rt={last_avg_rt}ms iter={last_avg_iter}"
                );
                println!(
                    "{current_target_length} {last_d} {last_nr_phases} {last_tot_phase} {last_avg_cost} {last_avg_rt} {last_avg_iter} 1",
                );
            } else if let Some((d, nr_phases, avg_cost, avg_rt, avg_iter, tot_phase)) = last_vals {
                eprintln!(
                    "[{YELLOW}!{RESET}] First iteration step already above error limit! Consider raising the tolerance...Current: d={d} cost={avg_cost:e} rt={avg_rt}ms iter={avg_iter}"
                );
                println!(
                    "{current_target_length} {d} {nr_phases} {tot_phase} {avg_cost} {avg_rt} {avg_iter} 0",
                );
            } else {
                anyhow::bail!("No evaluations took place!");
            }

            step += 1;
            current_target_length += self.length_step;
        }
        Ok(())
    }
}
