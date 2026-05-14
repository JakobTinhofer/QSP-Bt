use std::{fs::File, io::Write, path::PathBuf, time::Duration};

use anyhow::Result;
use serde::{Deserialize, Serialize};

const DATA_GUARD: &str = "--- DATA ---";

use crate::{
    cli::{ProgramConfig, SolverConfig, SolverStrategy, TargetConfig},
    compute::cpu::BackendMode,
    solvers::SolveOutcome,
    tasks::{plot_runtimes::PlotRuntimesTask, solve_poly::SolvePolyTask},
    utils::format_duration,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProgramConfigSerialized {
    pub backend_mode: BackendMode,
    pub solver: SolverConfig,
    pub strategy: SolverStrategy,
    pub target: TargetConfig,
}

impl From<&ProgramConfig> for ProgramConfigSerialized {
    fn from(c: &ProgramConfig) -> Self {
        Self {
            backend_mode: c.backend_mode,
            solver: c.solver.to_config(),
            strategy: c.solver.strategy.clone(),
            target: c.target.clone(),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct DataFileHeader {
    label: Option<String>,

    #[serde(flatten)]
    type_data: DataFileType,

    provenance: Provenance,
    config: ProgramConfigSerialized,
    runtime: String,
}

impl DataFileHeader {
    pub fn new(t: DataFileType, c: ProgramConfig, rt: Duration) -> Self {
        Self {
            label: c.label.clone(),
            type_data: t,
            provenance: Provenance::capture(),
            config: ProgramConfigSerialized::from(&c),
            runtime: format_duration(rt),
        }
    }

    pub fn create_file(&self, p: &PathBuf) -> Result<File> {
        let mut f = File::create_new(p)?;
        writeln!(f, "{}", toml::to_string(self)?)?;
        writeln!(f, "{}", DATA_GUARD)?;
        Ok(f)
    }
}

#[derive(Serialize, Deserialize, Default, PartialEq)]
#[serde(default)]
pub struct Provenance {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command_line: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_hash: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub hostname: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
}

fn build_git_id() -> Option<String> {
    let sha = option_env!("VERGEN_GIT_SHA")?;
    let dirty = option_env!("VERGEN_GIT_DIRTY") == Some("true");
    Some(if dirty {
        format!("{sha}-dirty")
    } else {
        sha.to_string()
    })
}

impl Provenance {
    pub fn capture() -> Self {
        Self {
            timestamp: Some(chrono::Utc::now().to_rfc3339()),
            command_line: Some(
                std::env::args()
                    .map(|a| shlex::try_quote(&a).map(|c| c.into_owned()).unwrap_or(a))
                    .collect::<Vec<_>>()
                    .join(" "),
            ),
            git_hash: build_git_id(),
            hostname: hostname::get().ok().and_then(|h| h.into_string().ok()),
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DataFileType {
    RuntimeData {
        #[serde(flatten)]
        options: PlotRuntimesTask,
    },
    SolveData {
        result: SolveOutcome,
        #[serde(flatten)]
        options: SolvePolyTask,
    },
}
