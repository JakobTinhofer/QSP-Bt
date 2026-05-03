use clap::{Parser, Subcommand};
use ndarray::Array1;
use num_complex::Complex64;
use rand::distr::{Distribution, Uniform};
use std::f64::consts::TAU;
use std::path::PathBuf;

use crate::compute::cpu::BackendMode;
use crate::solver::{Parity, SolveMode};

#[derive(Subcommand)]
pub enum Task {
    SolvePoly {
        /// complex numbers with mod <= 1 seperated by commas (eg "0.5+0.3i, -0.2-0.7i, 0.1i") or
        /// "rand,n": initializes with n points that are either 1 or 0 (which are then mirrored, see parity)
        /// "rand-phase,n" initializes with n points that are like exp(iφ) with random φ ∈ [0, 2π)
        #[arg(value_parser = parse_target)]
        target_y: Array1<Complex64>,

        /// parity ("even" or "odd")
        #[arg(short = 'p', long, value_enum, default_value_t = Parity::Even)]
        parity: Parity,

        /// Path for the solution output
        #[arg(short = 'o', long)]
        output: Option<PathBuf>,

        /// Path for outputing the data formated to be drawn in gnuplot
        #[arg(short = 'D', long)]
        drawable: Option<PathBuf>,
    },
    PlotRuntimes {
        /// Cutoff runtime in seconds
        #[arg(short = 'r', long, default_value = "180")]
        max_runtime: usize,

        /// How large to make the steps between different tries
        #[arg(short = 'l', long, default_value = "5")]
        target_len_step: usize,

        /// How many phase parameters to use per point of the target
        #[arg(short = 'R', long, default_value = "4")]
        ratio_phases_to_target: f64,

        #[arg(short = 'n', long, default_value = "3")]
        avg_n: usize,
    },
}

#[derive(Parser)]
#[command(about = "Fit a QSP polynomial to the given sequence of target points.")]
pub struct Args {
    ///which task to execute. Will solve given poly be default.
    #[command(subcommand)]
    pub task: Task,

    /// Enable/disable multithreading for gradient, lossfunction evaluation. Auto: will do single threading for small d & short sequences. (both <= 100)
    #[arg(short = 'm', long, value_enum, default_value_t = BackendMode::Auto)]
    pub backend_mode: BackendMode,

    /// Solve mode: "simple,D" — direct solve at degree D
    ///             "hotstart,S,D" — solve at degree S, then continue at degree D
    ///             "cascade,N,D" — N cascading steps up to degree D
    /// if running PlotRuntimes task, this will be interpreted as a ratio and scaled accordingly
    #[arg(short = 'M', long, value_parser = parse_solve_mode, default_value = "hotstart,20,60")]
    pub mode: SolveMode,

    /// Any L-BFGS call will only go up to this number of iters. Will break when this is reached and display a debug message.
    #[arg(short = 'i', long, default_value = "500000")]
    pub lbfgs_max_iters: u64,

    /// If the local change is smaller than this tolerance, break
    #[arg(short = 't', long, default_value = "1e-8")]
    pub tol_grad: f64,

    /// Memory for L-BFGS
    #[arg(long, default_value = "10")]
    pub lbfgs_mem: usize,
}

pub fn parse_solve_mode(s: &str) -> Result<SolveMode, String> {
    let trimmed = s.trim();

    if let Some(rest) = trimmed.strip_prefix("simple,") {
        let d = parse_positive(rest.trim())?;
        return Ok(SolveMode::Simple(d));
    }

    if let Some(rest) = trimmed.strip_prefix("hotstart,") {
        let parts: Vec<&str> = rest.split(',').collect();
        if parts.len() != 2 {
            return Err(format!(
                "hotstart: expected 'hotstart,S,D' but got '{}'",
                trimmed
            ));
        }
        let s_deg = parse_positive(parts[0].trim())?;
        let d_deg = parse_positive(parts[1].trim())?;
        if s_deg >= d_deg {
            return Err(format!(
                "hotstart: hotstart degree ({}) must be < final degree ({})",
                s_deg, d_deg
            ));
        }
        return Ok(SolveMode::Hotstart(s_deg, d_deg));
    }

    if let Some(rest) = trimmed.strip_prefix("cascade,") {
        let parts: Vec<&str> = rest.split(',').collect();
        if parts.len() != 2 {
            return Err(format!(
                "cascade: expected 'cascade,N,D' but got '{}'",
                trimmed
            ));
        }
        let n_steps = parse_positive(parts[0].trim())?;
        let d_deg = parse_positive(parts[1].trim())?;
        if n_steps < 2 {
            return Err("cascade: N must be >= 2".into());
        }
        return Ok(SolveMode::Cascade(n_steps, d_deg));
    }

    Err(format!(
        "Unknown solve mode '{}'. Expected one of: simple,D | hotstart,S,D | cascade,N,D",
        trimmed
    ))
}

pub fn parse_positive(s: &str) -> Result<usize, String> {
    let n: usize = s
        .parse()
        .map_err(|_| format!("'{s}' ist keine gültige Ganzzahl"))?;
    if n == 0 {
        Err("Wert muss > 0 sein".into())
    } else {
        Ok(n)
    }
}

pub fn parse_count(s: &str, mode: &str) -> Result<usize, String> {
    let n: usize = s
        .trim()
        .parse()
        .map_err(|_| format!("{mode}: '{s}' ist keine gültige Anzahl"))?;
    if n == 0 {
        Err(format!("{mode}: Anzahl muss > 0 sein"))
    } else {
        Ok(n)
    }
}

pub fn parse_target(s: &str) -> Result<Array1<Complex64>, String> {
    let trimmed = s.trim();
    let mut rng = rand::rng();

    if let Some(rest) = trimmed.strip_prefix("rand-phase,") {
        let n = parse_count(rest, "rand-phase")?;
        let p_dist = Uniform::new(0., TAU).unwrap();
        let numbers: Array1<Complex64> = (0..n)
            .map(|_| {
                let phi: f64 = p_dist.sample(&mut rng);
                Complex64::from_polar(1.0, phi)
            })
            .collect();
        return Ok(numbers);
    }

    if let Some(rest) = trimmed.strip_prefix("rand,") {
        let n = parse_count(rest, "rand")?;
        let bool_dist = rand::distr::Bernoulli::new(0.5).unwrap();
        let numbers: Array1<Complex64> = (0..n)
            .map(|_| { if bool_dist.sample(&mut rng) { 1. } else { 0. } }.into())
            .collect();
        return Ok(numbers);
    }

    let numbers: Array1<Complex64> = trimmed
        .split(',')
        .map(|part| {
            let cleaned = part.replace(char::is_whitespace, "");
            cleaned
                .parse::<Complex64>()
                .map_err(|e| format!("'{}': {}", part.trim(), e))
        })
        .collect::<Result<_, _>>()?;

    for (i, z) in numbers.iter().enumerate() {
        if z.norm() > 1.0 {
            return Err(format!("Zahl {i} ({z}) hat Betrag {:.4} > 1", z.norm()));
        }
    }

    Ok(numbers)
}

pub const RED: &str = "\x1b[31m";
pub const GREEN: &str = "\x1b[32m";
pub const DIM: &str = "\x1b[2m";
pub const YELLOW: &str = "\x1b[33m";
pub const BLUE: &str = "\x1b[34m";
pub const CYAN: &str = "\x1b[36m";
pub const BOLD: &str = "\x1b[1m";
pub const RESET: &str = "\x1b[0m";

pub fn trim_zeros(value: f64, precision: usize) -> String {
    let s = format!("{:.prec$}", value, prec = precision);
    if s.contains('.') {
        let trimmed = s.trim_end_matches('0').trim_end_matches('.');
        trimmed.to_string()
    } else {
        s
    }
}

pub fn format_complex_polar(z: &Complex64, precision: usize) -> String {
    let r = z.norm();
    let phi = z.arg();
    let i = format!("{YELLOW}{BOLD}i{RESET}");

    let eps = 10f64.powi(-(precision as i32));
    let r_str = trim_zeros(r, precision);

    if phi.abs() < eps {
        // φ ≈ 0 → kein Exponentialteil
        r_str
    } else {
        format!("{r_str} · exp({i}·{})", trim_zeros(phi, precision),)
    }
}

/// Formatiert ein ganzes Array, mehrzeilig oder einzeilig je nach Länge
pub fn format_array(arr: &Array1<Complex64>, precision: usize) -> String {
    let formatted: Vec<String> = arr
        .iter()
        .map(|z| format_complex_polar(z, precision))
        .collect();

    if arr.len() <= 6 {
        format!("[ {} ]", formatted.join(", "))
    } else {
        let mut s = String::from("[\n");
        for (i, item) in formatted.iter().enumerate() {
            s.push_str(&format!("  {DIM}{i:3}:{RESET} {item}\n"));
        }
        s.push(']');
        s
    }
}

fn format_real_compact(value: f64, precision: usize) -> String {
    if value >= 0.0 {
        format!(" {}", trim_zeros(value, precision))
    } else {
        format!(
            "{GREEN}{BOLD}-{RESET}{}",
            trim_zeros(value.abs(), precision)
        )
    }
}

/// Formatiert ein ganzes Array, mehrzeilig oder einzeilig je nach Länge
pub fn format_array_real(arr: &Array1<f64>, precision: usize) -> String {
    let formatted: Vec<String> = arr
        .iter()
        .map(|x| format_real_compact(*x, precision))
        .collect();

    if arr.len() <= 8 {
        format!("[{} ]", formatted.join(","))
    } else {
        let mut s = String::from("[\n");
        for (i, item) in formatted.iter().enumerate() {
            s.push_str(&format!("  {DIM}{i:3}:{RESET} {item}\n"));
        }
        s.push(']');
        s
    }
}
