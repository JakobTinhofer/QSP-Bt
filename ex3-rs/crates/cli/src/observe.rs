use std::time::Duration;

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use parking_lot::Mutex;
use qsp_rs_core::solvers::{
    SolveOutcome, TerminationReason,
    observe::{ProgressObserver, ProgressReport, StageInfo},
};

pub struct CliObserver {
    mp: MultiProgress,
    bars: Mutex<Vec<ProgressBar>>,
    spinner_style: ProgressStyle,
    finished_style: ProgressStyle,
}

impl CliObserver {
    pub fn new() -> Self {
        Self {
            mp: MultiProgress::new(),
            bars: Mutex::new(Vec::new()),
            spinner_style: ProgressStyle::with_template(
                "{prefix:.bold.cyan} {spinner:.cyan} {wide_msg}",
            )
            .unwrap()
            .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏ "),
            finished_style: ProgressStyle::with_template("{prefix:.bold.dim} {wide_msg}").unwrap(),
        }
    }

    pub fn println(&self, msg: impl AsRef<str>) -> std::io::Result<()> {
        self.mp.println(msg)
    }
}

impl ProgressObserver for CliObserver {
    fn on_new_stage(&self, stage: StageInfo) {
        let bar = self.mp.add(ProgressBar::new_spinner());
        bar.set_style(self.spinner_style.clone());
        bar.set_prefix(format!(
            "[{}/{}] d={:>3}",
            stage.current_stage + 1,
            stage.total_stages,
            stage.current_degree,
        ));
        bar.set_message("starting…");
        // Keep the spinner animating even between iter events.
        bar.enable_steady_tick(Duration::from_millis(80));
        self.bars.lock().push(bar);
    }
    fn on_iter(&self, u: ProgressReport) {
        if let Some(bar) = self.bars.lock().get(u.stage.current_stage) {
            bar.set_message(format!("iter {:>7}  cost={:.3e}", u.iter, u.cost));
        }
    }
    fn on_end_stage(&self, stage: StageInfo, out: &SolveOutcome) {
        let bars = self.bars.lock();
        let Some(bar) = bars.get(stage.current_stage) else {
            return;
        };

        bar.disable_steady_tick();
        bar.set_style(self.finished_style.clone());

        let (mark, label) = match out.term_reason {
            TerminationReason::Converged => ("✓", "converged"),
            TerminationReason::MaxItersReached => ("⏱", "max iters"),
            TerminationReason::LineSearchFailed => ("⚠", "line search"),
            TerminationReason::Diverged => ("✗", "diverged"),
            TerminationReason::Other => ("?", "other"),
        };
        bar.finish_with_message(format!(
            "{} {:<11} cost={:.3e}  iters={}",
            mark, label, out.cost, out.iterations,
        ));
    }
}
