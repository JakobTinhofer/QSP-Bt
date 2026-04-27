use crate::cli::{Args, BLUE, GREEN, RESET, format_array, format_array_real};
use crate::solver::Parity;
use clap::Parser;
use ndarray::Array1;
use num_complex::{Complex64, ComplexFloat};
use std::time::Instant;
use std::{fs::File, io::Write};

use crate::{
    qsp::qsp_poly,
    solver::{TargetPoly, solve_hotstart},
};

mod c2x2;
mod cli;
mod qsp;
mod solver;

fn main() -> std::io::Result<()> {
    let mut args = Args::parse();

    if args.target_y[0].abs() <= 0.8 {
        args.target_y[0] = (1.).into();
    }
    let t_len = args.target_y.len();
    let goal_last: Complex64 = match args.parity {
        Parity::Even => 1.,
        Parity::Odd => 0.,
    }
    .into();
    if (args.target_y[t_len - 1].abs() - goal_last.abs()).abs() > 0.2 {
        args.target_y[t_len - 1] = goal_last;
    }

    let target = TargetPoly::new_forced_parity(args.target_y, args.parity);

    println!(
        "[{BLUE}i{RESET}] Running with degree={} and hotstart_degree={}. Target:",
        args.degree, args.hotstart
    );
    println!("x:\n{}", format_array_real(&target.xs, 3));
    println!("y:\n{}", format_array(&target.ys, 3));

    let start = Instant::now();
    let (sol, f_err) = solve_hotstart(
        &target,
        args.hotstart,
        args.degree,
        args.tolerance,
        args.maxiter,
    ).expect("Did not converge. Try increasing tolerance or max iter. Some polynomials might not converge at all.");
    let elapsed = start.elapsed();
    let sol_str = sol
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    println!(
        "[{GREEN}+{RESET}] Finished solving! Elapsed: {:?} ms. Final loss: {f_err:e}. Resulting phases: \n{}",
        elapsed,
        format_array_real(&sol, 5)
    );

    if let Some(path) = &args.output {
        let mut file = File::create(path)?;
        write!(file, "{}", sol_str)?;
    }

    if let Some(path) = &args.drawable {
        let mut file = File::create(path)?;

        target.points_iter().for_each(|(x, y)| {
            writeln!(file, "{} {} {}", x, y.re, y.im).expect("Failed to write to file!")
        });

        write!(file, "\n\n")?;

        let xs = Array1::linspace(-1., 1., 600);
        let px = qsp_poly(&sol.as_slice().unwrap(), &xs.as_slice().unwrap());
        for (x, p) in xs.iter().zip(px) {
            writeln!(file, "{} {} {}", x, p.re, p.im)?;
        }
        println!("Wrote drawing data to '{}'", path.to_string_lossy());
    }

    Ok(())
}
