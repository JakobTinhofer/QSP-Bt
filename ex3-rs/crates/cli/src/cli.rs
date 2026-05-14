use crate::tasks::TaskType;
use clap::{Args as ClapArgs, Parser, ValueEnum};
use ndarray::Array1;
use num_complex::Complex64;
use qsp_rs_core::{
    compute::{ComputeBackend, cpu::BackendMode},
    solvers::{PhaseMap, SolveMode, Solver, bfgs::BfgsOptions, lm::LmOptions},
    target::{Parity, TargetPattern},
};
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
pub enum SolverKind {
    /// Limited-memory BFGS (default; good for smooth, well-conditioned problems)
    Bfgs,
    /// Levenberg-Marquardt (best for nonlinear least-squares with ill-conditioning)
    Lm,
}

#[derive(ClapArgs, Debug, Clone, Serialize, Deserialize)]
#[command(next_help_heading = "General solver options")]
pub struct SolverStrategy {
    /// Solve mode: "simple,D" — direct solve at degree D
    ///             "hotstart,S,D" — solve at degree S, then continue at degree D
    ///             "cascade,N,D" — N cascading steps up to degree D
    /// if running PlotRuntimes task, this will be interpreted as a ratio and scaled accordingly
    #[arg(short = 'M', long, value_parser = SolveMode::parse, default_value = "hotstart,20,60")]
    #[serde(flatten)]
    pub mode: SolveMode,

    /// Maps the phases before the QSP unitary is constructed.
    /// Use mirror for polynomial targets that are real.
    /// Mirror will double the effective parameter count, the
    /// parameter count given by --mode is the number of parameters
    /// used by the optimizer, but (if mirrored) only half the amount of phases used to construct
    /// the qsp unitary.
    #[arg(short = 'P', long, value_enum, default_value_t = PhaseMap::MirrorIfPossible)]
    pub phase_map: PhaseMap,

    /// The max. magnitude of the first guess for the random phases
    /// to initialize the solver. Choose 0 for mirrored phases to get faster
    /// convergance and lower total phase values.
    #[arg(short = 'i', long, default_value = ".4")]
    pub init_perturb_mag: f64,
}

/// What gets serialized to the file: only the relevant solver's params.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SolverConfig {
    Bfgs(BfgsOptions),
    Lm(LmOptions),
}

impl SolverArgs {
    /// Build the serializable form, picking only the active solver's options.
    pub fn to_config(&self) -> SolverConfig {
        match self.kind {
            SolverKind::Bfgs => SolverConfig::Bfgs(self.bfgs.clone()),
            SolverKind::Lm => SolverConfig::Lm(self.lm.clone()),
        }
    }
}

#[derive(ClapArgs, Debug, Clone)]
pub struct SolverArgs {
    /// Which optimizer to use
    #[arg(short = 'S', long = "solver", value_enum, default_value_t = SolverKind::Bfgs)]
    pub kind: SolverKind,

    #[command(flatten)]
    pub bfgs: BfgsOptions,

    #[command(flatten)]
    pub lm: LmOptions,

    #[command(flatten)]
    pub strategy: SolverStrategy,
}

impl SolverArgs {
    pub fn get_solver<T: ComputeBackend>(&self) -> Box<dyn Solver<T>> {
        match self.kind {
            SolverKind::Bfgs => Box::new(self.bfgs.clone()),
            SolverKind::Lm => Box::new(self.lm.clone()),
        }
    }
}

#[derive(ClapArgs, Debug, Clone, Serialize, Deserialize)]
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

#[derive(ClapArgs, Debug, Clone)]
pub struct ProgramConfig {
    /// Enable/disable multithreading for gradient, lossfunction evaluation. Auto: will do single threading for small d & short sequences. (both <= 100)
    #[arg(short = 'm', long, value_enum, default_value_t = BackendMode::Auto)]
    pub backend_mode: BackendMode,

    /// Solver selection and configuration. Use --solver to pick;
    /// pass --bfgs-*, --lm-*, or --gn-* flags as appropriate.
    #[command(flatten)]
    pub solver: SolverArgs,

    #[command(flatten)]
    pub target: TargetConfig,

    #[arg(short = 'l', long)]
    pub label: Option<String>,
}

#[derive(Parser)]
#[command(about = "Fit a QSP polynomial to the given sequence of target points.")]
pub struct Args {
    #[command(subcommand)]
    pub task: TaskType,

    #[command(flatten)]
    pub config: ProgramConfig,
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
