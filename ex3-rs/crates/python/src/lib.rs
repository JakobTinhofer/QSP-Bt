use std::panic::{AssertUnwindSafe, catch_unwind};
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::bail;
use ndarray::arr1;
use numpy::{Complex64, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use qsp_rs_core::compute::regularized::RidgeRegularizedBackend;
use qsp_rs_core::compute::wx::WxBackend;
use qsp_rs_core::compute::wz::WzBackend;
use qsp_rs_core::compute::{Backend, BackendMode, QspEvaluator};
use qsp_rs_core::solvers::configuration::{PhaseGenerator, PhaseMap, SolveMode};
use qsp_rs_core::solvers::observe::{CancelToken, SolverContext};
use qsp_rs_core::target::TargetDistribution;
use qsp_rs_core::{
    solvers::{SolveOutcome, Solver},
    target::{Parity, TargetPattern, TargetPoly},
};

use crate::helpers::{build_bfgs, build_lm, phase_gen_from_pyobj, termination_str, vectorize};
use crate::progress::PyObserver;
use crate::types::{PySolveResult, PyTargetPoly};

mod helpers;
mod progress;
mod types;

// Add alongside your other core imports:
//     use qsp_rs_core::target::Parity;
// (SolveMode, PhaseMap, Backend, Solver, SolveOutcome, PhaseGenerator, etc. are already imported by __solve)

fn __solve(
    py: Python<'_>,
    target: TargetPoly,
    solver: &str,
    mode: &str,
    phase_map: &str,
    phase_init: PhaseGenerator,
    backend_mode: &str,
    backend_conv: &str,
    regularize: Option<f64>,
    seed: Option<u64>,
    bfgs_options: Option<&Bound<'_, PyDict>>,
    lm_options: Option<&Bound<'_, PyDict>>,
    progress: Option<&Bound<'_, PyAny>>,
    progress_interval_ms: u64,
    rescale_to_err: Option<f64>, // NEW — Some(err): shrink to the smallest degree reaching `err`
) -> PyResult<PySolveResult> {
    let phase_map_p: PhaseMap = phase_map.parse()?;
    let backend_md: BackendMode = backend_mode.parse()?;
    let solve_mode = SolveMode::from_str(mode)?;
    let pytarget = PyTargetPoly {
        xs: target.xs.clone(),
        ys: target.ys.clone(),
        n_half: target.ys.len() / 2,
    };

    // Read parity info BEFORE `target` is moved into the backend below. NEW
    let target_parity = target.get_parity();
    let target_all_real = target.all_real();

    let backend = match (regularize, &*backend_conv.trim().to_lowercase()) {
        (Some(lambda), "wz") => Backend::RidgeRegularizedWz(RidgeRegularizedBackend::new(
            WzBackend::new(target, backend_md),
            lambda,
        )),
        (None, "wz") => Backend::Wz(WzBackend::new(target, backend_md)),
        (Some(lambda), "wx") => Backend::RidgeRegularizedWx(RidgeRegularizedBackend::new(
            WxBackend::new(target, backend_md),
            lambda,
        )),
        (None, "wx") => Backend::Wx(WxBackend::new(target, backend_md)),
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unknown backend convention: {backend_conv}. May only be wx,wz"
            )));
        }
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

    // GIL released for the whole run; the match picks single-solve vs. rescale search. NEW branch
    let outcome: anyhow::Result<SolveOutcome> = py.allow_threads(|| {
        catch_unwind(AssertUnwindSafe(|| match rescale_to_err {
            Some(target_err) => solve_rescaled(
                &backend,
                &*solver_box,
                &ctx,
                solve_mode,
                phase_map_p,
                &phase_init,
                target_parity,
                target_all_real,
                target_err,
            ),
            None => Ok(match seed {
                Some(_) => solver_box.solve(&backend, &ctx, solve_mode, phase_map_p, phase_init),
                None => solver_box.solve(&backend, &ctx, solve_mode, phase_map_p, phase_init),
            }?),
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

// ── rescale: smallest degree that reaches the requested error ───────────────
//
// Mirrors the CLI's ScalingBehaviorTask binary search, with two differences:
//   * the ceiling is the degree already in `base_mode` — we only shrink, never
//     grow — so a too-small mode is reported as an error instead of silently grown;
//   * no avg_n averaging: one solve per candidate degree.
//
// Like the CLI, this assumes cost-vs-degree is monotone enough for a binary
// search. It isn't guaranteed, so the result is "a small working degree", not
// provably "the smallest".
fn solve_rescaled(
    backend: &Backend,
    solver: &dyn Solver<Backend>,
    ctx: &SolverContext,
    base_mode: SolveMode,
    phase_map: PhaseMap,
    phase_init: &PhaseGenerator,
    parity: Option<Parity>,
    all_real: bool,
    target_err: f64,
) -> anyhow::Result<SolveOutcome> {
    if !(target_err.is_finite() && target_err > 0.0) {
        anyhow::bail!("rescale_to_err must be a finite positive number, got {target_err}");
    }

    let ceiling = mode_max_degree(base_mode);

    let mut lower = 1usize; // inclusive
    let mut upper = ceiling + 1; // exclusive — the +1 is what lets the search actually test `ceiling`
    let mut best: Option<SolveOutcome> = None; // smallest degree that cleared the bar
    let mut last_cost: Option<f64> = None; // most recent failing cost, for the error message

    while lower < upper {
        let mid = lower + (upper - lower) / 2;
        // Solve at the parity-corrected degree, but move the search bounds on the
        // raw `mid` — exactly as the CLI does.
        let d = adjust_for_parity(mid, phase_map, parity, all_real);
        let outcome = solver.solve(
            backend,
            ctx,
            base_mode.rescale(d),
            phase_map.clone(),
            phase_init.clone(),
        )?;

        if outcome.cost < target_err {
            upper = mid; // good enough → look smaller
            best = Some(outcome);
        } else {
            lower = mid + 1; // too small → grow
            last_cost = Some(outcome.cost);
        }
    }

    best.ok_or_else(|| {
        let reached = match last_cost {
            Some(c) => format!(" (best cost reached: {c:e})"),
            None => String::new(),
        };
        anyhow::anyhow!(
            "rescale_to_err: the degree in the given mode (max degree {ceiling}) is too small to \
             reach error {target_err:e}{reached}. Increase the mode's degree and retry."
        )
    })
}

/// Largest degree a mode will ever solve at — the binary-search ceiling.
/// max() sidesteps the start/final field order in Hotstart/Cascade.
fn mode_max_degree(mode: SolveMode) -> usize {
    match mode {
        SolveMode::Simple(d) => d,
        SolveMode::Hotstart(a, b) | SolveMode::Cascade(a, b) => a.max(b),
    }
}

/// QSP phase-sequence parity bump, mirroring the CLI. Mirror maps are exempt;
/// otherwise an odd-parity target wants an odd degree (and vice versa), so bump
/// by one when they disagree. Unknown target parity → no bump.
fn adjust_for_parity(
    d: usize,
    phase_map: PhaseMap,
    parity: Option<Parity>,
    all_real: bool,
) -> usize {
    match (phase_map, all_real) {
        (PhaseMap::Mirror, _) | (PhaseMap::MirrorIfPossible, true) => d,
        _ => match (d % 2, parity) {
            (0, Some(Parity::Odd)) | (1, Some(Parity::Even)) => d + 1,
            _ => d,
        },
    }
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
    rescale_to_err   = None,
    backend_conv     = "wx",
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
    rescale_to_err: Option<f64>,
    backend_conv: &str,
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
        backend_conv,
        regularize,
        seed,
        bfgs_options,
        lm_options,
        progress,
        progress_interval_ms,
        rescale_to_err,
    )
}

#[pyfunction]
#[pyo3(signature=(
    k,
    n_half,
    dist="sqrt"
)
)]
fn theta_m<'py>(
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
fn theta_m_continuous<'py>(
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
    rescale_to_err   = None,
    backend_conv     = "wx",
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
    rescale_to_err: Option<f64>,
    backend_conv: &str,
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
        backend_conv,
        regularize,
        seed,
        bfgs_options,
        lm_options,
        progress,
        progress_interval_ms,
        rescale_to_err,
    )
}

#[pyfunction]
#[pyo3(signature = (phases, x, backend_conv = "wx", *))]
fn evaluate_poly<'py>(
    py: Python<'py>,
    phases: PyReadonlyArray1<'_, f64>,
    x: &Bound<'py, PyAny>,
    backend_conv: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let phases_view = phases.as_array();

    vectorize::<_, _, _, anyhow::Error>(py, x, |v| {
        let a = arr1(&[v]);
        match &*backend_conv.trim().to_lowercase() {
            "wx" => Ok(WxBackend::evaluate_poly(&phases_view, &a.view())[0]),
            "wz" => Ok(WzBackend::evaluate_poly(&phases_view, &a.view())[0]),
            _ => bail!("Unknown backend convention: {backend_conv}. May only be wx,wz"),
        }
    })
}

#[pymodule]
fn qsp_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySolveResult>()?;
    m.add_class::<PyTargetPoly>()?;
    m.add_function(wrap_pyfunction!(solve_poly_with_pattern, m)?)?;
    m.add_function(wrap_pyfunction!(solve_poly, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_poly, m)?)?;
    m.add_function(wrap_pyfunction!(theta_m, m)?)?;
    m.add_function(wrap_pyfunction!(theta_m_continuous, m)?)?;
    Ok(())
}
