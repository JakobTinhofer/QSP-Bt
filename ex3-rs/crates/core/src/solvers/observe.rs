use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[derive(Debug, Clone, Copy)]
pub struct StageInfo {
    pub current_stage: usize,
    pub total_stages: usize,
    pub current_degree: usize,
}

pub struct ProgressReport {
    pub stage: StageInfo,
    pub iter: u64,
    pub cost: f64,
}

pub trait ProgressObserver: Send + Sync {
    fn on_new_stage(&self, _stage: StageInfo) {}
    fn on_iter(&self, _update: ProgressReport) {}
    fn on_end_stage(&self, _stage: StageInfo, _outcome: &super::SolveOutcome) {}
}

pub struct NoopObserver;
impl ProgressObserver for NoopObserver {}

#[derive(Clone, Debug, Default)]
pub struct CancelToken {
    flag: Arc<AtomicBool>,
}

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Relaxed);
    }
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }
}

pub struct SolverContext {
    pub cancel: CancelToken,
    pub observer: Arc<dyn ProgressObserver>,
}

impl Default for SolverContext {
    fn default() -> Self {
        Self {
            cancel: CancelToken::new(),
            observer: Arc::new(NoopObserver),
        }
    }
}

impl SolverContext {
    pub fn new(cancel: CancelToken, observer: Arc<dyn ProgressObserver>) -> Self {
        Self { cancel, observer }
    }
}
