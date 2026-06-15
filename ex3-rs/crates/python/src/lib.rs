use std::panic::{AssertUnwindSafe, catch_unwind};
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ndarray::arr1;
use numpy::{Complex64, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use qsp_rs_core::compute::regularized::RidgeRegularizedBackend;
use qsp_rs_core::compute::{Backend, ComputeBackend};
use qsp_rs_core::solvers::configuration::{PhaseGenerator, PhaseMap, SolveMode};
use qsp_rs_core::solvers::observe::{CancelToken, SolverContext};
use qsp_rs_core::target::TargetDistribution;
use qsp_rs_core::{
    compute::cpu::{BackendMode, CpuComputeBackend},
    solvers::{SolveOutcome, Solver},
    target::{Parity, TargetPattern, TargetPoly},
};

use crate::helpers::{build_bfgs, build_lm, phase_gen_from_pyobj, termination_str, vectorize};
use crate::progress::PyObserver;
use crate::types::{PySolveResult, PyTargetPoly};

mod helpers;
mod progress;
mod types;

fn __solve(
    py: Python<'_>,
    target: TargetPoly,
    solver: &str,
    mode: &str,
    phase_map: &str,
    phase_init: PhaseGenerator,
    backend_mode: &str,
    regularize: Option<f64>,
    seed: Option<u64>,
    bfgs_options: Option<&Bound<'_, PyDict>>,
    lm_options: Option<&Bound<'_, PyDict>>,
    progress: Option<&Bound<'_, PyAny>>,
    progress_interval_ms: u64,
) -> PyResult<PySolveResult> {
    let phase_map_p: PhaseMap = phase_map.parse()?;
    let backend_md: BackendMode = backend_mode.parse()?;
    let solve_mode = SolveMode::from_str(mode)?;

    let pytarget = PyTargetPoly {
        xs: target.xs.clone(),
        ys: target.ys.clone(),
        n_half: target.ys.len() / 2,
    };

    let backend = match regularize {
        Some(lambda) => Backend::RidgeRegularized(RidgeRegularizedBackend::new(
            CpuComputeBackend::new(target, backend_md),
            lambda,
        )),
        None => Backend::Plain(CpuComputeBackend::new(target, backend_md)),
    };

    let solver_box: Box<dyn Solver<Backend>> = match solver.to_ascii_lowercase().as_str() {
        "bfgs" => Box::new(build_bfgs(bfgs_options)?),
        "lm" => Box::new(build_lm(lm_options)?),
        other => return Err(PyValueError::new_err(format!("unknown solver: {other:?}"))),
    };
    let cancel = CancelToken::new();
    let callback: Option<Py<PyAny>> = progress.map(|p| p.clone().unbind());
    let observer = Arc::new(PyObserver::new(
        cancel.clone(),
        callback,
        Duration::from_millis(progress_interval_ms.max(10)),
    ));
    let ctx = SolverContext::new(cancel, observer);
    let start = Instant::now();
    // Release GIL while code is running
    let outcome: anyhow::Result<SolveOutcome> = py.allow_threads(|| {
        catch_unwind(AssertUnwindSafe(|| match seed {
            Some(_) => solver_box.solve(&backend, &ctx, solve_mode, phase_map_p, phase_init),
            None => solver_box.solve(&backend, &ctx, solve_mode, phase_map_p, phase_init),
        }))
        .map(|res| res.map_err(|e| e.into()))
        .unwrap_or_else(|panic_payload| {
            let msg = panic_payload
                .downcast_ref::<&'static str>()
                .map(|s| s.to_string())
                .or_else(|| panic_payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "<non-string panic payload>".to_string());
            Err(anyhow::anyhow!("solver panicked: {msg}"))
        })
    });

    let outcome = outcome.map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;

    Ok(PySolveResult {
        cost: outcome.cost,
        phases: outcome.phases,
        iterations: outcome.iterations,
        termination: termination_str(outcome.term_reason).to_string(),
        target: pytarget,
        total_phase: outcome.phase_mag_sum,
        elapsed_ms,
    })
}

#[pyfunction]
#[pyo3(signature = (
    ys,
    *,
    parity           = "even",
    target_dist      = "sqrt",
    solver           = "bfgs",
    mode             = "hotstart,20,60",
    phase_map        = "mirror-if-possible",
    init             = PhaseGenerator::Random { magnitude: 0.4, seed: None },
    backend_mode     = "auto",
    regularize       = None,
    seed             = None,
    bfgs_options     = None,
    lm_options       = None,
    progress = None, progress_interval_ms = 100,
))]
#[allow(clippy::too_many_arguments)]
fn solve_poly(
    py: Python<'_>,
    ys: &Bound<'_, PyAny>,
    parity: &str,
    target_dist: &str,
    solver: &str,
    mode: &str,
    phase_map: &str,
    #[pyo3(from_py_with = "phase_gen_from_pyobj")] init: PhaseGenerator,
    backend_mode: &str,
    regularize: Option<f64>,
    seed: Option<u64>,
    bfgs_options: Option<&Bound<'_, PyDict>>,
    lm_options: Option<&Bound<'_, PyDict>>,
    progress: Option<&Bound<'_, PyAny>>,
    progress_interval_ms: u64,
) -> PyResult<PySolveResult> {
    let np = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", np.getattr("complex128")?)?;
    let ys_any = np.call_method("asarray", (ys,), Some(&kwargs))?;
    let ys: PyReadonlyArray1<'_, Complex64> = ys_any.extract()?;

    let parity: Parity = parity.parse()?;
    let ys_array = ys.as_array().to_owned();
    let target = TargetPoly::new_forced_parity(
        ys_array,
        parity,
        TargetDistribution::from_str(target_dist)?,
    )?;

    __solve(
        py,
        target,
        solver,
        mode,
        phase_map,
        init,
        backend_mode,
        regularize,
        seed,
        bfgs_options,
        lm_options,
        progress,
        progress_interval_ms,
    )
}

#[pyfunction]
#[pyo3(signature=(
    k,
    n_half,
    dist="sqrt"
)
)]
fn theta_k<'py>(
    py: Python<'py>,
    k: &Bound<'py, PyAny>,
    n_half: usize,
    dist: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let d = TargetDistribution::from_str(dist)?;
    vectorize(py, k, |v| d.theta_m(v, n_half))
}

#[pyfunction]
#[pyo3(signature=(k,n_half, dist="sqrt"))]
fn theta_k_continuous<'py>(
    py: Python<'py>,
    k: &Bound<'py, PyAny>,
    n_half: usize,
    dist: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let d = TargetDistribution::from_str(dist)?;
    vectorize(py, k, |v| d.theta_m_continuous(v, n_half))
}

#[pyfunction]
#[pyo3(signature = (
    target_half_len,
    target_pattern,
    *,
    parity           = "even",
    target_dist      = "sqrt",
    solver           = "bfgs",
    mode             = "hotstart,20,60",
    phase_map        = "mirror-if-possible",
    init             = PhaseGenerator::Random { magnitude: 0.4, seed: None },
    backend_mode     = "auto",
    regularize       = None,
    seed             = None,
    bfgs_options     = None,
    lm_options       = None,
    progress = None, progress_interval_ms = 100,
))]
#[allow(clippy::too_many_arguments)]
fn solve_poly_with_pattern(
    py: Python<'_>,
    target_half_len: usize,
    target_pattern: &str,
    parity: &str,
    target_dist: &str,
    solver: &str,
    mode: &str,
    phase_map: &str,
    #[pyo3(from_py_with = "phase_gen_from_pyobj")] init: PhaseGenerator,
    backend_mode: &str,
    regularize: Option<f64>,
    seed: Option<u64>,
    bfgs_options: Option<&Bound<'_, PyDict>>,
    lm_options: Option<&Bound<'_, PyDict>>,
    progress: Option<&Bound<'_, PyAny>>,
    progress_interval_ms: u64,
) -> PyResult<PySolveResult> {
    let pattern: TargetPattern = target_pattern.parse()?;
    let parity: Parity = parity.parse()?;
    let target = TargetPoly::from_pattern(
        &pattern,
        parity,
        target_half_len,
        TargetDistribution::from_str(target_dist)?,
    )?;

    __solve(
        py,
        target,
        solver,
        mode,
        phase_map,
        init,
        backend_mode,
        regularize,
        seed,
        bfgs_options,
        lm_options,
        progress,
        progress_interval_ms,
    )
}

#[pyfunction]
#[pyo3(signature = (phases, x, *))]
fn evaluate_poly<'py>(
    py: Python<'py>,
    phases: PyReadonlyArray1<'_, f64>,
    x: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let phases_view = phases.as_array();

    vectorize::<_, _, _, anyhow::Error>(py, x, |v| {
        let a = arr1(&[v]);
        Ok(CpuComputeBackend::evaluate_poly(&phases_view, &a.view())[0])
    })
}

#[pymodule]
fn qsp_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySolveResult>()?;
    m.add_class::<PyTargetPoly>()?;
    m.add_function(wrap_pyfunction!(solve_poly_with_pattern, m)?)?;
    m.add_function(wrap_pyfunction!(solve_poly, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_poly, m)?)?;
    m.add_function(wrap_pyfunction!(theta_k, m)?)?;
    m.add_function(wrap_pyfunction!(theta_k_continuous, m)?)?;
    Ok(())
}
