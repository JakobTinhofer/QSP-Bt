use std::panic::{AssertUnwindSafe, catch_unwind};
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ndarray::{Array1, arr1};
use numpy::{Complex64, IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyComplex, PyDict};

use qsp_rs_core::compute::regularized::RidgeRegularizedBackend;
use qsp_rs_core::compute::{Backend, ComputeBackend};
use qsp_rs_core::solvers::configuration::{PhaseGenerator, PhaseMap, SolveMode};
use qsp_rs_core::solvers::observe::{CancelToken, SolverContext};
use qsp_rs_core::target::{theta_k as theta_k_core, theta_k_continuous as theta_k_cont_core};
use qsp_rs_core::{
    compute::cpu::{BackendMode, CpuComputeBackend},
    solvers::{SolveOutcome, Solver, TerminationReason, bfgs::BfgsOptions, lm::LmOptions},
    target::{Parity, TargetPattern, TargetPoly},
};

use crate::progress::PyObserver;

mod progress;

#[pyclass(name = "TargetPoly", module = "qsp_rs", frozen)]
#[derive(Debug, Clone)]
pub struct PyTargetPoly {
    xs: Array1<f64>,
    ys: Array1<Complex64>,
    #[pyo3(get)]
    n_half: usize,
}

#[pymethods]
impl PyTargetPoly {
    #[getter]
    fn xs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.xs.clone().into_pyarray(py)
    }

    #[getter]
    fn ys<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Complex64>> {
        self.ys.clone().into_pyarray(py)
    }

    #[getter]
    fn ks<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> {
        PyArray1::from_iter(py, 0..self.n_half)
    }

    #[getter]
    fn thetas<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_iter(py, self.xs.iter().map(|x| x.acos()))
    }
}

#[pyclass(name = "SolveResult", module = "qsp_rs", frozen)]
pub struct PySolveResult {
    #[pyo3(get)]
    cost: f64,
    phases: Array1<f64>,
    #[pyo3(get)]
    iterations: u64,
    #[pyo3(get)]
    termination: String,
    #[pyo3(get)]
    elapsed_ms: f64,
    #[pyo3(get)]
    target: PyTargetPoly,
    #[pyo3(get)]
    total_phase: f64,
}

#[pymethods]
impl PySolveResult {
    #[getter]
    fn phases<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.phases.clone().into_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "SolveResult(cost={:.3e}, n_phases={}, total_phase={}, iterations={}, termination='{}', elapsed_ms={:.1})",
            self.cost,
            self.phases.len(),
            self.total_phase,
            self.iterations,
            self.termination,
            self.elapsed_ms,
        )
    }
}

fn termination_str(t: TerminationReason) -> &'static str {
    match t {
        TerminationReason::Converged => "converged",
        TerminationReason::MaxItersReached => "max_iters_reached",
        TerminationReason::LineSearchFailed => "line_search_failed",
        TerminationReason::Diverged => "diverged",
        TerminationReason::Other => "other",
    }
}

fn override_field<T>(dict: &Bound<'_, PyDict>, key: &str, slot: &mut T) -> PyResult<()>
where
    T: for<'py> FromPyObject<'py>,
{
    if let Some(v) = dict.get_item(key)? {
        *slot = v
            .extract::<T>()
            .map_err(|e| PyValueError::new_err(format!("option {key}: {e}")))?;
    }
    Ok(())
}

fn build_bfgs(overrides: Option<&Bound<'_, PyDict>>) -> PyResult<BfgsOptions> {
    let mut o = BfgsOptions::default();
    if let Some(d) = overrides {
        override_field(d, "max_iters", &mut o.max_iters)?;
        override_field(d, "mem", &mut o.mem)?;
        override_field(d, "tol_grad", &mut o.tol_grad)?;
    }
    Ok(o)
}

fn build_lm(overrides: Option<&Bound<'_, PyDict>>) -> PyResult<LmOptions> {
    let mut o = LmOptions::default();
    if let Some(d) = overrides {
        override_field(d, "max_iters", &mut o.max_iters)?;
        override_field(d, "initial_lambda", &mut o.initial_lambda)?;
        override_field(d, "tol", &mut o.tol)?;
    }
    Ok(o)
}

fn phase_gen_from_pyobj(obj: &Bound<'_, PyAny>) -> PyResult<PhaseGenerator> {
    // 1) string
    if let Ok(s) = obj.extract::<String>() {
        return s
            .parse::<PhaseGenerator>()
            .map_err(|e| PyTypeError::new_err(format!("invalid PhaseGenerator string: {e}")));
    }
    // 2) real numpy float64 array
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<f64>>() {
        return Ok(PhaseGenerator::Fixed(arr.as_array().to_owned()));
    }
    // 3) plain Python list/tuple of numbers -> Fixed
    if let Ok(v) = obj.extract::<Vec<f64>>() {
        return Ok(PhaseGenerator::Fixed(Array1::from_vec(v)));
    }
    Err(PyTypeError::new_err(
        "expected a str, a 1-D float64 numpy array, or a sequence of floats",
    ))
}

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
    let target = TargetPoly::new_forced_parity(ys_array, parity)?;

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
#[pyo3(signature=(k,n_half))]
fn theta_k(k: usize, n_half: usize) -> PyResult<f64> {
    Ok(theta_k_core(k, n_half)?)
}

#[pyfunction]
#[pyo3(signature=(k,n_half))]
fn theta_k_continuous(k: f64, n_half: usize) -> PyResult<f64> {
    Ok(theta_k_cont_core(k, n_half)?)
}

#[pyfunction]
#[pyo3(signature = (
    target_half_len,
    target_pattern,
    *,
    parity           = "even",
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
    let target = TargetPoly::from_pattern(&pattern, parity, target_half_len)?;

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

    // check if argument is scalar (single x) or a numpy array of xs (vectorized)
    if x.hasattr("__array__")? {
        if let Ok(arr) = x.extract::<PyReadonlyArray1<'_, f64>>() {
            return Ok(
                CpuComputeBackend::evaluate_poly(&phases_view, &arr.as_array())
                    .into_pyarray(py)
                    .into_any(),
            );
        }
    } else {
        if let Ok(scalar_x) = x.extract::<f64>() {
            let xs = arr1(&[scalar_x]);
            let ys = CpuComputeBackend::evaluate_poly(&phases_view, &xs.view());
            return Ok(PyComplex::from_doubles(py, ys[0].re, ys[0].im).into_any());
        }
    }

    Err(PyTypeError::new_err(
        "x must be a float or a 1-D numpy float64 array",
    ))
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
