use std::str::FromStr;

use crate::tasks::TaskType;
use clap::{Args as ClapArgs, Parser, ValueEnum};
use ndarray::Array1;
use qsp_rs_core::{
    compute::{
        Backend, BackendMode, ComputeBackend, QspEvaluator, regularized::RidgeRegularizedBackend,
        wx::WxBackend, wz::WzBackend,
    },
    solvers::{
        Solver,
        bfgs::BfgsOptions,
        configuration::{PhaseGenerator, PhaseMap, SolveMode},
        lm::LmOptions,
    },
    target::{Parity, TargetDistribution, TargetPattern, TargetPoly},
};
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
pub enum SolverKind {
    /// Limited-memory BFGS (default; good for smooth, well-conditioned problems)
    Bfgs,
    /// Levenberg-Marquardt (best for nonlinear least-squares with ill-conditioning)
    Lm,
}

#[derive(clap::ValueEnum, Clone, Copy, Serialize, Deserialize, Debug)]
pub enum PhaseMapArg {
    None,
    Mirror,
    MirrorIfPossible,
}

#[derive(clap::ValueEnum, Clone, Copy, Serialize, Deserialize, Debug)]
pub enum BackendConvention {
    Wx,
    Wz,
}

impl BackendConvention {
    pub fn evaluate_poly(
        &self,
        phases: &ndarray::prelude::ArrayView1<f64>,
        xs: &ndarray::prelude::ArrayView1<f64>,
    ) -> Array1<num_complex::Complex64> {
        match self {
            BackendConvention::Wx => WxBackend::evaluate_poly(phases, xs),
            BackendConvention::Wz => WzBackend::evaluate_poly(phases, xs),
        }
    }
}

impl From<PhaseMapArg> for PhaseMap {
    fn from(a: PhaseMapArg) -> Self {
        match a {
            PhaseMapArg::None => PhaseMap::None,
            PhaseMapArg::Mirror => PhaseMap::Mirror,
            PhaseMapArg::MirrorIfPossible => PhaseMap::MirrorIfPossible,
        }
    }
}

#[derive(ClapArgs, Debug, Clone, Serialize, Deserialize)]
#[command(next_help_heading = "General solver options")]
pub struct SolverStrategy {
    /// Solve mode: "simple,D" — direct solve at degree D
    ///             "hotstart,S,D" — solve at degree S, then continue at degree D
    ///             "cascade,N,D" — N cascading steps up to degree D
    /// if running PlotRuntimes task, this will be interpreted as a ratio and scaled accordingly
    #[arg(short = 'M', long, value_parser = SolveMode::from_str, default_value = "hotstart,20,60")]
    #[serde(flatten)]
    pub mode: SolveMode,

    /// Maps the phases before the QSP unitary is constructed.
    /// Use mirror for polynomial targets that are real.
    /// Mirror will double the effective parameter count, the
    /// parameter count given by --mode is the number of parameters
    /// used by the optimizer, but (if mirrored) only half the amount of phases used to construct
    /// the qsp unitary.
    #[arg(short = 'P', long, value_enum, default_value_t = PhaseMapArg::MirrorIfPossible)]
    pub phase_map: PhaseMapArg,

    /// The max. magnitude of the first guess for the random phases
    /// to initialize the solver. Choose 0 for mirrored phases to get faster
    /// convergance and lower total phase values.
    #[arg(short = 'i', long, value_parser = PhaseGenerator::from_str, default_value = "0")]
    pub phase_init: PhaseGenerator,

    /// If set, enables regularization (ridge/tikhonov) with the given weight.
    /// Choose a magnitude similar to the error you want to achieve.
    #[arg(short = 'r', long)]
    pub regularization_lambda: Option<f64>,

    #[arg(long, value_enum, default_value_t = BackendConvention::Wx)]
    pub backend_convention: BackendConvention,
}

pub fn backend_from_convention_and_lambda(
    conv: BackendConvention,
    t: TargetPoly,
    m: BackendMode,
    l: Option<f64>,
) -> Backend {
    match conv {
        BackendConvention::Wz => {
            let b = WzBackend::new(t, m);
            match l {
                Some(lambda) => {
                    Backend::RidgeRegularizedWz(RidgeRegularizedBackend::new(b, lambda))
                }
                None => Backend::Wz(b),
            }
        }
        BackendConvention::Wx => {
            let b = WxBackend::new(t, m);
            match l {
                Some(lambda) => {
                    Backend::RidgeRegularizedWx(RidgeRegularizedBackend::new(b, lambda))
                }
                None => Backend::Wx(b),
            }
        }
    }
}

/// What gets serialized to the file: only the relevant solver's params.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SolverConfig {
    Bfgs(BfgsArgs),
    Lm(LmArgs),
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

#[derive(clap::Args, Debug, Clone, Serialize, Deserialize)]
#[command(next_help_heading = "L-BFGS Options")]
pub struct BfgsArgs {
    #[arg(long = "bfgs-max-iters", default_value_t = BfgsOptions::default().max_iters)]
    pub max_iters: u64,
    #[arg(long = "bfgs-mem", default_value_t = BfgsOptions::default().mem)]
    pub mem: usize,
    #[arg(long = "bfgs-tol-grad", default_value_t = BfgsOptions::default().tol_grad)]
    pub tol_grad: f64,
}

impl From<BfgsArgs> for BfgsOptions {
    fn from(a: BfgsArgs) -> Self {
        Self {
            max_iters: a.max_iters,
            mem: a.mem,
            tol_grad: a.tol_grad,
        }
    }
}

#[derive(ClapArgs, Debug, Clone, Serialize, Deserialize)]
#[command(next_help_heading = "Levenberg-Marquardt Options")]
pub struct LmArgs {
    #[arg(id = "lm_max_iters", long = "lm-max-iters", default_value = "500")]
    pub max_iters: u64,
    #[arg(
        id = "lm_initial_lambda",
        long = "lm-initial-lambda",
        default_value = "1e-4"
    )]
    pub initial_lambda: f64,
    #[arg(id = "lm_tol", long = "lm-tol", default_value = "1e-10")]
    pub tol: f64,
}

impl From<LmArgs> for LmOptions {
    fn from(a: LmArgs) -> Self {
        Self {
            max_iters: a.max_iters,
            initial_lambda: a.initial_lambda,
            tol: a.tol,
        }
    }
}

#[derive(ClapArgs, Debug, Clone)]
pub struct SolverArgs {
    /// Which optimizer to use
    #[arg(short = 'S', long = "solver", value_enum, default_value_t = SolverKind::Bfgs)]
    pub kind: SolverKind,

    #[command(flatten)]
    pub bfgs: BfgsArgs,

    #[command(flatten)]
    pub lm: LmArgs,

    #[command(flatten)]
    pub strategy: SolverStrategy,
}

impl SolverArgs {
    pub fn get_solver<T: ComputeBackend>(&self) -> Box<dyn Solver<T>> {
        match self.kind {
            SolverKind::Bfgs => Box::new(BfgsOptions::from(self.bfgs.clone())),
            SolverKind::Lm => Box::new(LmOptions::from(self.lm.clone())),
        }
    }
}

#[derive(clap::ValueEnum, Clone, Copy, Serialize, Deserialize, Debug)]
pub enum ParityArg {
    Even,
    Odd,
}

impl From<ParityArg> for Parity {
    fn from(a: ParityArg) -> Self {
        match a {
            ParityArg::Even => Self::Even,
            ParityArg::Odd => Self::Odd,
        }
    }
}

#[derive(clap::ValueEnum, Clone, Copy, Serialize, Deserialize, Debug)]
pub enum TargetDistributionArg {
    /// This corresponds to a physical JC system
    Sqrt,
    /// This corresponds to the 2nd order pertubation system outlined in ref[1]
    Equidistant,
}

impl From<TargetDistributionArg> for TargetDistribution {
    fn from(value: TargetDistributionArg) -> Self {
        match value {
            TargetDistributionArg::Equidistant => Self::Equidistant,
            TargetDistributionArg::Sqrt => Self::Sqrt,
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
    #[arg(short='t', long, value_parser = TargetPattern::from_str, default_value = "rand")]
    pub target_pattern: TargetPattern,

    /// parity ("even" or "odd")
    #[arg(short = 'p', long, value_enum, default_value_t = ParityArg::Even)]
    pub parity: ParityArg,

    /// How to space the target points
    #[arg(long, value_enum, default_value_t = TargetDistributionArg::Sqrt)]
    pub distribution: TargetDistributionArg,
}

#[derive(clap::ValueEnum, Clone, Copy, Serialize, Deserialize, Debug)]
pub enum BackendModeArg {
    Auto,
    SingleThread,
    MultiThread,
}

impl From<BackendModeArg> for BackendMode {
    fn from(a: BackendModeArg) -> Self {
        match a {
            BackendModeArg::SingleThread => Self::SingleThread,
            BackendModeArg::MultiThread => Self::MultiThread,
            BackendModeArg::Auto => Self::Auto,
        }
    }
}

#[derive(ClapArgs, Debug, Clone)]
pub struct ProgramConfig {
    /// Enable/disable multithreading for gradient, lossfunction evaluation. Auto: will do single threading for small d & short sequences. (both <= 100)
    #[arg(short = 'm', long, value_enum, default_value_t = BackendModeArg::Auto)]
    pub backend_mode: BackendModeArg,

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

pub const GREEN: &str = "\x1b[32m";
pub const DIM: &str = "\x1b[2m";
pub const YELLOW: &str = "\x1b[33m";
pub const BLUE: &str = "\x1b[34m";
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
