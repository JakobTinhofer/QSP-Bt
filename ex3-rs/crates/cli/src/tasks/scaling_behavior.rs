use crate::cli::{BLUE, GREEN, RESET, YELLOW};
use clap::Args;
use qsp_rs_core::{
    compute::cpu::{BackendMode, CpuComputeBackend},
    solvers::PhaseMap,
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
    #[arg(short = 'd', default_value = "1")]
    degree_reduction_step: usize,
    #[arg(long = "steps", default_value = "-1")]
    run_steps: i32,
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
            let mut d = (current_target_length as f64 * self.starting_phase_ratio) as usize;
            let mut last_vals: Option<(usize, usize, f64, f64, f64)> = None;
            // TODO: Replace sequential scan with binary search

            loop {
                let target = TargetPoly::from_pattern(&t.target_pattern, p, current_target_length);
                let backend = CpuComputeBackend::new(target, b);
                let mode = s.strategy.mode.rescale(d);

                let mut nr_of_phases = 0;
                let (mut avg_cost, mut avg_rt, mut avg_iter) = (0.0, 0.0, 0.0);

                for i in 0..self.avg_n {
                    let now = Instant::now();
                    let res = s.get_solver::<CpuComputeBackend>().solve(
                        &backend,
                        mode,
                        PhaseMap::from(s.strategy.phase_map),
                        s.strategy.init_perturb_mag,
                    )?;
                    avg_cost += res.cost / self.avg_n as f64;
                    let elapsed = now.elapsed();
                    avg_rt += elapsed.as_millis() as f64 / self.avg_n as f64;
                    avg_iter += res.iterations as f64 / self.avg_n as f64;
                    nr_of_phases = res.phases.len();
                    //eprintln!("[{RED}*{RESET}] CONFIG: {mode:?} {t:?}");
                    let digits = (self.avg_n / 10) as usize + 1;
                    eprintln!(
                        "[{BLUE}i{RESET}] ({:>digits$}/{:>digits$}) n={current_target_length} d={d}: cost={:e} iter={} rt={elapsed:?}",
                        i + 1,
                        self.avg_n,
                        res.cost,
                        res.iterations
                    );
                }

                if avg_cost > self.max_error {
                    if let Some((
                        last_d,
                        last_nr_phases,
                        last_avg_cost,
                        last_avg_rt,
                        last_avg_iter,
                    )) = last_vals
                    {
                        eprintln!(
                            "[{GREEN}+{RESET}] Reached error limit. Last: d={last_d} cost={last_avg_cost:e} phases_n={last_nr_phases} rt={last_avg_rt}ms iter={last_avg_iter}"
                        );
                        println!(
                            "{current_target_length} {last_d} {last_nr_phases} {last_avg_cost} {last_avg_rt} {last_avg_iter} 1",
                        );
                    } else {
                        eprintln!(
                            "[{YELLOW}!{RESET}] First iteration step already above error limit! Consider raising the tolerance...Current: d={d} cost={avg_cost:e} rt={avg_rt}ms iter={avg_iter}"
                        );
                        println!("{current_target_length} {d} {avg_cost} {avg_rt} {avg_iter} 0",);
                    }
                    eprintln!("\n\n");
                    break;
                }

                if d <= 0 {
                    anyhow::bail!("Never got reached error limit, even with d <= 0!");
                }
                last_vals = Some((d, nr_of_phases, avg_cost, avg_rt, avg_iter));
                d -= self.degree_reduction_step;
            }
            step += 1;
            current_target_length += self.length_step;
        }
        Ok(())
    }
}
