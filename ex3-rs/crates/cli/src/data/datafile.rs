use std::{fs::File, io::Write, path::PathBuf, time::Duration};

use anyhow::Result;
use qsp_rs_core::{compute::BackendMode, solvers::SolveOutcome};
use serde::{Deserialize, Serialize};

const DATA_GUARD: &str = "--- DATA ---";

use crate::{
    cli::{ProgramConfig, SolverConfig, SolverStrategy, TargetConfig},
    tasks::{plot_runtimes::PlotRuntimesTask, solve_poly::SolvePolyTask},
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
            backend_mode: BackendMode::from(c.backend_mode),
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

pub fn format_duration(d: Duration) -> String {
    let total_secs = d.as_secs();

    // Sub-second: show as ms or smaller
    if total_secs == 0 {
        let ms = d.as_secs_f64() * 1000.0;
        if ms >= 1.0 {
            // Trim trailing zeros: 1.500ms → 1.5ms, 60.000ms → 60ms
            return format!("{}ms", trim_float(ms, 3));
        }
        let us = d.as_secs_f64() * 1_000_000.0;
        if us >= 1.0 {
            return format!("{}µs", trim_float(us, 1));
        }
        return format!("{}ns", d.as_nanos());
    }

    // 1s to <60s: show with millisecond precision
    if total_secs < 60 {
        let secs = d.as_secs_f64();
        return format!("{}s", trim_float(secs, 3));
    }

    // ≥1 minute: integer seconds, structured units
    let secs = total_secs % 60;
    let mins = (total_secs / 60) % 60;
    let hours = (total_secs / 3600) % 24;
    let days = total_secs / 86400;

    if days > 0 {
        format!("{days}d{hours:02}h{mins:02}m{secs:02}s")
    } else if hours > 0 {
        format!("{hours}h{mins:02}m{secs:02}s")
    } else {
        format!("{mins}m{secs:02}s")
    }
}

/// Format a float with up to `decimals` places, trimming trailing zeros.
/// 1.500 (3) → "1.5", 60.000 (3) → "60", 1.234 (3) → "1.234"
fn trim_float(value: f64, decimals: usize) -> String {
    let s = format!("{value:.*}", decimals);
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}
