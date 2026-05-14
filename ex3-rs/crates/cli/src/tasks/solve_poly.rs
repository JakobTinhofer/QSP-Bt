use crate::cli::{BLUE, GREEN, RESET, format_array, format_array_real};

use crate::data::datafile::{DataFileHeader, DataFileType};
use crate::{cli::ProgramConfig, tasks::TaskTrait};
use anyhow::Result;
use clap::Args;
use ndarray::Array1;
use qsp_rs_core::compute::ComputeBackend;
use qsp_rs_core::compute::cpu::{BackendMode, CpuComputeBackend};
use qsp_rs_core::solvers::{PhaseMap, SolveOutcome};
use qsp_rs_core::target::{Parity, TargetPoly};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Clone, Args, Serialize, Deserialize)]
pub struct SolvePolyTask {
    /// How long to make the target. Since the target is mirrored afterwars, this is half of the final length.
    #[arg(short = 'n', long)]
    target_half_len: usize,

    #[serde(skip)]
    /// Path for the data output
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,
}

impl TaskTrait for SolvePolyTask {
    fn execute(&self, cfg: ProgramConfig) -> Result<()> {
        let backend = CpuComputeBackend::new(
            TargetPoly::from_pattern(
                &cfg.target.target_pattern,
                Parity::from(cfg.target.parity),
                self.target_half_len,
            ),
            BackendMode::from(cfg.backend_mode),
        );

        println!(
            "[{BLUE}i{RESET}] Running with config={:?} and multithreading={:?}. Target:",
            cfg.solver.strategy, cfg.backend_mode
        );

        println!("x:\n{}", format_array_real(&backend.get_target().xs, 3));
        println!("y:\n{}", format_array(&backend.get_target().ys, 3));

        let start = Instant::now();
        let outcome: SolveOutcome = cfg
            .solver
            .get_solver::<CpuComputeBackend>()
            .solve(
                &backend,
                cfg.solver.strategy.mode,
                PhaseMap::from(cfg.solver.strategy.phase_map),
                cfg.solver.strategy.init_perturb_mag,
            )
            .expect("Solver failed!");

        let elapsed = start.elapsed();

        println!(
            "[{GREEN}+{RESET}] Finished solving! Elapsed: {:?}. Final loss: {:e}. Resulting phases: \n{}",
            elapsed,
            outcome.cost,
            format_array_real(&outcome.phases, 5)
        );

        if let Some(path) = &self.output {
            let header = DataFileHeader::new(
                DataFileType::SolveData {
                    result: outcome.clone(),
                    options: (*self).clone(),
                },
                cfg,
                elapsed,
            );
            let mut file = header.create_file(path)?;
            let t = backend.get_target();

            writeln!(file, "# Result phases (i,phi_i)")?;
            for (i, p) in outcome.phases.iter().enumerate() {
                writeln!(file, "{i} {p}")?;
            }

            write!(file, "\n\n")?;

            writeln!(file, "# Polynomial target points (x,theta,Re[y],Im[y])")?;
            t.points_iter()
                .zip(t.thetas.iter())
                .try_for_each(|((x, y), theta)| {
                    writeln!(file, "{} {} {} {}", x, theta, y.re, y.im,)
                })?;

            write!(file, "\n\n")?;

            writeln!(
                file,
                "# Plottable polynomial evaluation in x \\in (-1,1) with ~2k points (x,theta,Re[P], Im[P])"
            )?;
            let xs = Array1::linspace(-1., 1., 2000);
            let px = backend.evaluate_poly(&outcome.phases.view(), &xs.view());
            for (x, p) in xs.iter().zip(px.iter()) {
                writeln!(file, "{} {} {} {}", x, x.acos(), p.re, p.im,)?;
            }
            println!("Wrote drawing data to '{}'", path.to_string_lossy());
        }

        Ok(())
    }
}
