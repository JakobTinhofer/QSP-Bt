use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use pyo3::types::{PyAnyMethods, PyDict};
use pyo3::{Py, PyAny, Python};
use qsp_rs_core::solvers::observe::{CancelToken, ProgressObserver, ProgressReport, StageInfo};

#[derive(Default)]
pub(crate) struct ProgressSnapshot {
    iter: AtomicU64,
    cost_bits: AtomicU64,
    stage_index: AtomicUsize,
    stage_total: AtomicUsize,
    stage_degree: AtomicUsize,
    generation: AtomicU64,
}

pub(crate) struct SnapshotValues {
    pub iter: u64,
    pub cost: f64,
    pub stage_index: usize,
    pub stage_total: usize,
    pub stage_degree: usize,
    pub generation: u64,
}

impl ProgressSnapshot {
    pub fn read(&self) -> SnapshotValues {
        SnapshotValues {
            iter: self.iter.load(Ordering::Relaxed),
            cost: f64::from_bits(self.cost_bits.load(Ordering::Relaxed)),
            stage_index: self.stage_index.load(Ordering::Relaxed),
            stage_total: self.stage_total.load(Ordering::Relaxed),
            stage_degree: self.stage_degree.load(Ordering::Relaxed),
            generation: self.generation.load(Ordering::Relaxed),
        }
    }
}

pub(crate) struct PyObserver {
    snap: Arc<ProgressSnapshot>,
    cancel: CancelToken,
    callback: Option<Py<PyAny>>,
    last_py_call: Mutex<Instant>,
    py_interval: Duration,
    last_seen_gen: AtomicU64,
}

impl PyObserver {
    pub fn new(cancel: CancelToken, callback: Option<Py<PyAny>>, py_interval: Duration) -> Self {
        Self {
            snap: Arc::new(ProgressSnapshot::default()),
            cancel,
            callback,
            last_py_call: Mutex::new(Instant::now() - py_interval),
            py_interval,
            last_seen_gen: AtomicU64::new(0),
        }
    }
}

impl ProgressObserver for PyObserver {
    fn on_new_stage(&self, stage: StageInfo) {
        self.snap.iter.store(0, Ordering::Relaxed);
        self.snap
            .cost_bits
            .store(f64::NAN.to_bits(), Ordering::Relaxed);
        self.snap
            .stage_total
            .store(stage.total_stages, Ordering::Relaxed);
        self.snap
            .stage_degree
            .store(stage.current_degree, Ordering::Relaxed);
        self.snap
            .stage_index
            .store(stage.current_stage, Ordering::Relaxed);
        self.snap.generation.fetch_add(1, Ordering::Relaxed);
    }

    fn on_iter(&self, u: ProgressReport) {
        // Hot path: just stash the latest values.
        self.snap.iter.store(u.iter, Ordering::Relaxed);
        self.snap
            .cost_bits
            .store(u.cost.to_bits(), Ordering::Relaxed);

        // Throttled: every py_interval ms, reacquire the GIL on the main thread
        // to (1) run pending signal handlers, (2) dispatch to the Python callback.
        let now = Instant::now();
        {
            let mut last = self.last_py_call.lock().unwrap();
            if now.duration_since(*last) < self.py_interval {
                return;
            }
            *last = now;
        }

        Python::with_gil(|py| {
            // check_signals works here because we're on the main thread.
            if py.check_signals().is_err() {
                self.cancel.cancel();
                return;
            }
            let Some(cb) = self.callback.as_ref() else {
                return;
            };

            let snap = self.snap.read();
            let last_gen = self.last_seen_gen.load(Ordering::Relaxed);
            let new_stage = snap.generation != last_gen;
            if new_stage {
                self.last_seen_gen.store(snap.generation, Ordering::Relaxed);
            }

            let dict = PyDict::new(py);
            let _ = dict.set_item("stage_index", snap.stage_index);
            let _ = dict.set_item("stage_total", snap.stage_total);
            let _ = dict.set_item("stage_degree", snap.stage_degree);
            let _ = dict.set_item("iter", snap.iter);
            let _ = dict.set_item("cost", snap.cost);
            let _ = dict.set_item("new_stage", new_stage);
            if let Err(e) = cb.call1(py, (dict,)) {
                if e.is_instance_of::<pyo3::exceptions::PyKeyboardInterrupt>(py) {
                    self.cancel.cancel();
                } else {
                    e.print(py);
                }
            }
        });
    }
}
