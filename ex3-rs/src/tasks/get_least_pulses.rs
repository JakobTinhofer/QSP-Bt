use crate::{
    cli::ProgramConfig, compute::cpu::CpuComputeBackend, solvers::SolveOutcome, target::TargetPoly,
    tasks::TaskTrait,
};
use anyhow::Result;
use clap::Args;
use serde::{Deserialize, Serialize};

#[derive(Args, Serialize, Deserialize)]
pub struct GetLeastPulsesTask {
    /// Choose the degree startpoint.
    #[arg(short = 'd')]
    start_d: usize,

    /// How much to reduce the pulse size by every step.
    #[arg(short = 's', long, default_value = "1")]
    reduce_step_d: usize,

    /// Length of the target
    #[arg(short = 'n', long)]
    target_len: usize,

    /// Stop runnig once the error goes above this value.
    #[arg(short = 'e', long, default_value = "1e-6")]
    max_error: f64,
}

impl TaskTrait for GetLeastPulsesTask {
    fn execute(&self, cfg: ProgramConfig) -> Result<()> {
        let t = cfg.target;
        let b = cfg.backend_mode;
        let s = cfg.solver;

        let backend = CpuComputeBackend::new(
            TargetPoly::from_pattern(&t.target_pattern, t.parity, self.target_len),
            b,
        );

        for current_d in (1..(self.start_d + 1)).rev().step_by(self.reduce_step_d) {
            let SolveOutcome {
                phases: _,
                cost: f_err,
                iterations: _,
                term_reason: _,
            } = s
                .get_solver::<CpuComputeBackend>()
                .solve(
                    &backend,
                    s.strategy.mode.rescale(current_d),
                    s.strategy.phase_map,
                    s.strategy.init_perturb_mag,
                )
                .expect("Solver failed!");
            println!("{current_d} {f_err}");
            if f_err >= self.max_error {
                eprintln!("Done! Final error: {f_err:?}");
                break;
            }
        }
        Ok(())
    }
}
