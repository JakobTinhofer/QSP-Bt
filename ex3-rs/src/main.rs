use clap::Parser;
use ex3_rs::cli::{Args, BLUE, GREEN, RESET, Task, format_array, format_array_real};
use ex3_rs::compute::ComputeBackend;
use ex3_rs::compute::cpu::CpuComputeBackend;
use ex3_rs::solvers::{SolveMode, SolveOutcome};
use ex3_rs::target::{Parity, TargetPoly};
use ndarray::Array1;
use num_complex::Complex64;
use rand::distr::Distribution;
use std::time::Instant;
use std::{fs::File, io::Write};

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    match args.task {
        Task::SolvePoly {
            target_y,
            parity,
            output,
            drawable,
        } => {
            let backend = CpuComputeBackend::new(
                TargetPoly::new_forced_parity(target_y, parity),
                args.backend_mode,
            );

            println!(
                "[{BLUE}i{RESET}] Running with mode={:?} and multithreading={:?}. Target:",
                args.mode, args.backend_mode
            );
            println!("x:\n{}", format_array_real(&backend.get_target().xs, 3));
            println!("y:\n{}", format_array(&backend.get_target().ys, 3));

            let start = Instant::now();
            let SolveOutcome {
                phases: sol,
                cost: f_err,
                iterations: _,
                term_reason: _,
            } = args
                .solver
                .get_solver::<CpuComputeBackend>()
                .solve(&backend, args.mode)
                .expect("Solver failed!");

            let elapsed = start.elapsed();
            let sol_str = sol
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            println!(
                "[{GREEN}+{RESET}] Finished solving! Elapsed: {:?}. Final loss: {f_err:e}. Resulting phases: \n{}",
                elapsed,
                format_array_real(&sol, 5)
            );

            if let Some(path) = &output {
                let mut file = File::create(path)?;
                write!(file, "{}", sol_str)?;
            }

            if let Some(path) = &drawable {
                let mut file = File::create(path)?;

                backend.get_target().points_iter().for_each(|(x, y)| {
                    writeln!(file, "{} {} {}", x, y.re, y.im).expect("Failed to write to file!")
                });

                write!(file, "\n\n")?;

                let xs = Array1::linspace(-1., 1., 600);

                let px = backend.evaluate_poly(&sol, &xs);
                for (x, p) in xs.iter().zip(px.iter()) {
                    writeln!(file, "{} {} {}", x, p.re, p.im)?;
                }
                println!("Wrote drawing data to '{}'", path.to_string_lossy());
            }

            Ok(())
        }

        Task::PlotRuntimes {
            max_runtime,
            target_len_step,
            ratio_phases_to_target,
            avg_n,
        } => {
            let mut current_target_len = target_len_step;
            let bool_dist = rand::distr::Bernoulli::new(0.5).unwrap();
            let mut rng = rand::rng();
            let mut current_n = 0;
            let mut running_avg_rt = 0.;
            let mut running_avg_e = 0.;
            loop {
                let numbers: Array1<Complex64> = (0..current_target_len)
                    .map(|_| { if bool_dist.sample(&mut rng) { 1. } else { 0. } }.into())
                    .collect();
                let current_degree =
                    ((current_target_len as f64) * ratio_phases_to_target) as usize;
                eprintln!(
                    "Target length: {}, Degree: {}",
                    numbers.len(),
                    current_degree
                );
                let target = TargetPoly::new_forced_parity(numbers, Parity::Even);
                let backend = CpuComputeBackend::new(target, args.backend_mode);

                let mode = match args.mode {
                    SolveMode::Simple(_) => SolveMode::Simple(current_degree),
                    SolveMode::Hotstart(d1, d2) => SolveMode::Hotstart(
                        ((d1 as f64) / (d2 as f64) * (current_degree as f64)) as usize,
                        current_degree as usize,
                    ),
                    SolveMode::Cascade(s, d) => SolveMode::Cascade(
                        ((s as f64) / (d as f64) * (current_degree as f64)) as usize,
                        ((current_target_len as f64) * ratio_phases_to_target) as usize,
                    ),
                };

                let start = Instant::now();
                let SolveOutcome {
                    phases: _,
                    cost: f_err,
                    iterations: _,
                    term_reason: _,
                } = args
                    .solver
                    .get_solver::<CpuComputeBackend>()
                    .solve(&backend, mode)
                    .expect("Solver failed!");

                let elapsed = start.elapsed();
                let elapsed_s = elapsed.as_secs_f64();
                running_avg_rt += elapsed_s / (avg_n as f64);
                running_avg_e += f_err / (avg_n as f64);
                eprintln!(
                    "[{GREEN}+{RESET}] Done in {:?}. Current avg: {}s, err: {}",
                    elapsed,
                    running_avg_rt * (avg_n as f64 / (current_n + 1) as f64),
                    running_avg_e * (avg_n as f64 / (current_n + 1) as f64)
                );
                current_n += 1;

                if current_n % avg_n == 0 {
                    println!("{current_target_len} {} {f_err}", running_avg_rt);
                    if running_avg_rt >= max_runtime as f64 {
                        break;
                    } else {
                        current_target_len += target_len_step;
                        current_n = 0;
                        running_avg_e = 0.;
                        running_avg_rt = 0.;
                    }
                }
            }

            Ok(())
        }
    }
}
