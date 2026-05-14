use anyhow::Result;
use clap::Subcommand;

use crate::{
    cli::ProgramConfig,
    tasks::{
        get_least_pulses::GetLeastPulsesTask, plot_runtimes::PlotRuntimesTask,
        solve_poly::SolvePolyTask,
    },
};

pub trait TaskTrait {
    fn execute(&self, cfg: ProgramConfig) -> Result<()>;
}

pub mod get_least_pulses;
pub mod plot_runtimes;
pub mod solve_poly;

#[derive(Subcommand)]
pub enum TaskType {
    SolvePoly(SolvePolyTask),
    PlotRuntimes(PlotRuntimesTask),
    GetLeastPulses(GetLeastPulsesTask),
}

impl TaskTrait for TaskType {
    fn execute(&self, cfg: ProgramConfig) -> Result<()> {
        match self {
            TaskType::SolvePoly(a) => a.execute(cfg),
            TaskType::PlotRuntimes(a) => a.execute(cfg),
            TaskType::GetLeastPulses(a) => a.execute(cfg),
        }
    }
}
