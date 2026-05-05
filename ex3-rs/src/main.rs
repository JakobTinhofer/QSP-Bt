use clap::Parser;
use ex3_rs::cli::{
    Args, BLUE, GREEN, RESET, SolverConfig, TargetConfig, Task, format_array, format_array_real,
};
use ex3_rs::compute::ComputeBackend;
use ex3_rs::compute::cpu::{BackendMode, CpuComputeBackend};
use ex3_rs::solvers::SolveOutcome;
use ex3_rs::target::{Parity, TargetPoly};
use ndarray::Array1;
use std::path::PathBuf;
use std::time::Instant;
use std::{fs::File, io::Write};

fn do_solve(
    output: Option<PathBuf>,
    drawable: Option<PathBuf>,
    target_half_len: usize,
    b: BackendMode,
    s: SolverConfig,
    t: TargetConfig,
) -> std::io::Result<()> {
    let backend = CpuComputeBackend::new(
        TargetPoly::from_pattern(&t.target_pattern, t.parity, target_half_len),
        b,
    );

    println!(
        "[{BLUE}i{RESET}] Running with mode={:?} and multithreading={:?}. Target:",
        s.mode, b
    );
    println!("x:\n{}", format_array_real(&backend.get_target().xs, 3));
    println!("y:\n{}", format_array(&backend.get_target().ys, 3));

    let start = Instant::now();
    let SolveOutcome {
        phases: sol,
        cost: f_err,
        iterations: _,
        term_reason: _,
    } = s
        .get_solver::<CpuComputeBackend>()
        .solve(&backend, s.mode, s.phase_map)
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

        let px = backend.evaluate_poly(&sol.view(), &xs.view());
        for (x, p) in xs.iter().zip(px.iter()) {
            writeln!(file, "{} {} {}", x, p.re, p.im)?;
        }
        println!("Wrote drawing data to '{}'", path.to_string_lossy());
    }

    Ok(())
}

fn do_plot_runtimes(
    max_runtime: usize,
    target_len_step: usize,
    ratio_phases_to_target: f64,
    avg_n: usize,
    force_degree_parity: bool,
    b: BackendMode,
    s: SolverConfig,
    t: TargetConfig,
) -> std::io::Result<()> {
    let mut current_target_len = target_len_step;
    let mut current_n = 0;
    let mut running_avg_rt = 0.;
    let mut running_avg_e = 0.;
    loop {
        let mut current_degree = ((current_target_len as f64) * ratio_phases_to_target) as usize;
        if force_degree_parity {
            match (current_degree % 2, t.parity) {
                (0, Parity::Odd) | (1, Parity::Even) => current_degree += 1,
                _ => (),
            }
        }
        let target = TargetPoly::from_pattern(&t.target_pattern, t.parity, current_target_len);
        eprintln!(
            "Target length: 2 * {}, Degree: {}",
            target.xs.len() / 2 as usize,
            current_degree
        );

        let backend = CpuComputeBackend::new(target, b);

        let mode = s.mode.rescale(current_degree);

        let start = Instant::now();
        let SolveOutcome {
            phases: _,
            cost: f_err,
            iterations: _,
            term_reason: _,
        } = s
            .get_solver::<CpuComputeBackend>()
            .solve(&backend, mode, s.phase_map)
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

fn main() -> std::io::Result<()> {
    let args = Args::parse();

    match args {
        Args {
            task:
                Task::PlotRuntimes {
                    max_runtime,
                    target_len_step,
                    ratio_phases_to_target,
                    avg_n,
                    force_degree_parity,
                },
            backend_mode,
            solver,
            target,
        } => do_plot_runtimes(
            max_runtime,
            target_len_step,
            ratio_phases_to_target,
            avg_n,
            force_degree_parity,
            backend_mode,
            solver,
            target,
        ),

        Args {
            task:
                Task::SolvePoly {
                    output,
                    drawable,
                    target_half_len,
                },
            backend_mode,
            solver,
            target,
        } => do_solve(
            output,
            drawable,
            target_half_len,
            backend_mode,
            solver,
            target,
        ),
    }
}
