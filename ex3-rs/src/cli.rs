use clap::{Args as ClapArgs, Parser, Subcommand, ValueEnum};
use ndarray::Array1;
use num_complex::Complex64;
use std::path::PathBuf;

use crate::compute::ComputeBackend;
use crate::compute::cpu::BackendMode;
use crate::solvers::bfgs::BfgsOptions;
use crate::solvers::lm::LmOptions;
use crate::solvers::{PhaseMap, SolveMode, Solver};
use crate::target::{Parity, TargetPattern};

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum SolverKind {
    /// Limited-memory BFGS (default; good for smooth, well-conditioned problems)
    Bfgs,
    /// Levenberg-Marquardt (best for nonlinear least-squares with ill-conditioning)
    Lm, /*
        /// Gauss-Newton (LM without damping; faster when it works, can diverge)
        Gn,
         */
}
/*
#[derive(ClapArgs, Debug, Clone)]
#[command(next_help_heading = "Gauss-Newton Options")]
pub struct GnOptions {
    #[arg(id = "gn_max_iters", long = "gn-max-iters", default_value = "200")]
    pub max_iters: u64,
    #[arg(id = "gn_tol", long = "gn-tol", default_value = "1e-10")]
    pub tol: f64,
} */

#[derive(ClapArgs, Debug, Clone)]
pub struct SolverConfig {
    /// Which optimizer to use
    #[arg(short = 's', long = "solver", value_enum, default_value_t = SolverKind::Bfgs)]
    pub kind: SolverKind,

    #[command(flatten)]
    pub bfgs: BfgsOptions,

    #[command(flatten)]
    pub lm: LmOptions,

    /// Solve mode: "simple,D" — direct solve at degree D
    ///             "hotstart,S,D" — solve at degree S, then continue at degree D
    ///             "cascade,N,D" — N cascading steps up to degree D
    /// if running PlotRuntimes task, this will be interpreted as a ratio and scaled accordingly
    #[arg(short = 'M', long, value_parser = SolveMode::parse, default_value = "hotstart,20,60")]
    pub mode: SolveMode,

    /// Maps the phases before the QSP unitary is constructed.
    /// Use mirror for polynomial targets that are real.
    /// Mirror will double the effective parameter count, the
    /// parameter count given by --mode is the number of parameters
    /// used by the optimizer, but (if mirrored) only half the amount of phases used to construct
    /// the qsp unitary.
    #[arg(short = 'P', long, value_enum, default_value_t = PhaseMap::MirrorIfPossible)]
    pub phase_map: PhaseMap,
}

impl SolverConfig {
    pub fn get_solver<T: ComputeBackend>(&self) -> Box<dyn Solver<T>> {
        match self.kind {
            SolverKind::Bfgs => Box::new(self.bfgs.clone()),
            SolverKind::Lm => Box::new(self.lm.clone()),
        }
    }
}

#[derive(Subcommand)]
pub enum Task {
    SolvePoly {
        /// How long to make the target. Since the target is mirrored afterwars, this is half of the final length.
        #[arg(short = 'n', long, default_value_t = 100)]
        target_half_len: usize,

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

        #[arg(long)]
        force_degree_parity: bool,
    },
}
#[derive(ClapArgs, Debug, Clone)]
#[command(next_help_heading = "Target configuration")]
pub struct TargetConfig {
    /// complex numbers with mod <= 1 seperated by commas (eg "0.5+0.3i, -0.2-0.7i, 0.1i") that will be repeated up to target length
    /// "rand": random points that are either 1 or 0 (which are then mirrored, see parity)
    /// "rand-phase" random points that are like exp(iφ) with random φ ∈ [0, 2π)
    /// "gp,r,k" generalized parity target with r,k
    #[arg(short='t', long, value_parser = TargetPattern::parse, default_value = "rand")]
    pub target_pattern: TargetPattern,

    /// parity ("even" or "odd")
    #[arg(short = 'p', long, value_enum, default_value_t = Parity::Even)]
    pub parity: Parity,
}

#[derive(Parser)]
#[command(about = "Fit a QSP polynomial to the given sequence of target points.")]
pub struct Args {
    #[command(subcommand)]
    pub task: Task,

    /// Enable/disable multithreading for gradient, lossfunction evaluation. Auto: will do single threading for small d & short sequences. (both <= 100)
    #[arg(short = 'm', long, value_enum, default_value_t = BackendMode::Auto)]
    pub backend_mode: BackendMode,

    /// Solver selection and configuration. Use --solver to pick;
    /// pass --bfgs-*, --lm-*, or --gn-* flags as appropriate.
    #[command(flatten)]
    pub solver: SolverConfig,

    #[command(flatten)]
    pub target: TargetConfig,
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
