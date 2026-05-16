use std::time::Instant;

use crate::cli::{GREEN, RESET};
use crate::{cli::ProgramConfig, tasks::TaskTrait};
use anyhow::Result;
use clap::Args;
use qsp_rs_core::compute::cpu::{BackendMode, CpuComputeBackend};
use qsp_rs_core::solvers::observe::SolverContext;
use qsp_rs_core::solvers::{PhaseMap, SolveOutcome};
use qsp_rs_core::target::{Parity, TargetPoly};
use serde::{Deserialize, Serialize};

#[derive(Args, Serialize, Deserialize)]
pub struct PlotRuntimesTask {
    /// Cutoff runtime in seconds
    #[arg(short = 'r', long, default_value = "180")]
    max_runtime: usize,

    /// How large to make the steps between different tries
    #[arg(short = 's', long, default_value = "5")]
    target_len_step: usize,

    /// How many phase parameters to use per point of the target
    #[arg(short = 'R', long, default_value = "4")]
    ratio_phases_to_target: f64,

    /// Average over n runs
    #[arg(short = 'n', long, default_value = "3")]
    avg_n: usize,

    /// When providing a ratio, will ensure that
    /// the phases have the correct parity for the
    /// target polynomial
    #[arg(long)]
    force_degree_parity: bool,
}

impl TaskTrait for PlotRuntimesTask {
    fn execute(&self, cfg: ProgramConfig) -> Result<()> {
        let t = cfg.target;
        let b = BackendMode::from(cfg.backend_mode);
        let s = cfg.solver;
        let p = Parity::from(t.parity);

        let mut current_target_len = self.target_len_step;
        let mut current_n = 0;
        let mut running_avg_rt = 0.;
        let mut running_avg_e = 0.;
        loop {
            let mut current_degree =
                ((current_target_len as f64) * self.ratio_phases_to_target) as usize;
            if self.force_degree_parity {
                match (current_degree % 2, p) {
                    (0, Parity::Odd) | (1, Parity::Even) => current_degree += 1,
                    _ => (),
                }
            }
            let target = TargetPoly::from_pattern(&t.target_pattern, p, current_target_len)?;
            eprintln!(
                "Target length: 2 * {}, Degree: {}",
                target.xs.len() / 2 as usize,
                current_degree
            );

            let backend = CpuComputeBackend::new(target, b);

            let mode = s.strategy.mode.rescale(current_degree);

            let start = Instant::now();
            let SolveOutcome {
                phases: _,
                cost: f_err,
                iterations: _,
                term_reason: _,
                phase_mag_sum: _,
            } = s.get_solver::<CpuComputeBackend>().solve(
                &backend,
                &SolverContext::default(),
                mode,
                PhaseMap::from(s.strategy.phase_map),
                s.strategy.init_perturb_mag,
            )?;

            let elapsed = start.elapsed();
            let elapsed_s = elapsed.as_secs_f64();
            running_avg_rt += elapsed_s / (self.avg_n as f64);
            running_avg_e += f_err / (self.avg_n as f64);
            eprintln!(
                "[{GREEN}+{RESET}] Done in {:?}. Current avg: {}s, err: {}",
                elapsed,
                running_avg_rt * (self.avg_n as f64 / (current_n + 1) as f64),
                running_avg_e * (self.avg_n as f64 / (current_n + 1) as f64)
            );
            current_n += 1;

            if current_n % self.avg_n == 0 {
                println!("{current_target_len} {} {f_err}", running_avg_rt);
                if running_avg_rt >= self.max_runtime as f64 {
                    break;
                } else {
                    current_target_len += self.target_len_step;
                    current_n = 0;
                    running_avg_e = 0.;
                    running_avg_rt = 0.;
                }
            }
        }

        Ok(())
    }
}
