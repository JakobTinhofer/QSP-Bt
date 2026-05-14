use std::panic::{AssertUnwindSafe, catch_unwind};

use ndarray::Array1;
use numpy::{Complex64, IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use qsp_rs_core::{
    compute::{
        ComputeBackend,
        cpu::{BackendMode, CpuComputeBackend},
    },
    solvers::{
        PhaseMap, SolveMode, SolveOutcome, Solver, TerminationReason, bfgs::BfgsOptions,
        lm::LmOptions,
    },
    target::{Parity, TargetPattern, TargetPoly},
};

#[pyclass(name = "SolveResult", module = "qsp", frozen)]
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
}

#[pymethods]
impl PySolveResult {
    #[getter]
    fn phases<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.phases.clone().into_pyarray_bound(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "SolveResult(cost={:.3e}, n_phases={}, iterations={}, termination='{}', elapsed_ms={:.1})",
            self.cost,
            self.phases.len(),
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

fn __solve(
    py: Python<'_>,
    target: TargetPoly,
    solver: &str,
    mode: &str,
    phase_map: &str,
    init_perturb_mag: f64,
    backend_mode: &str,
    seed: Option<u64>,
    bfgs_options: Option<&Bound<'_, PyDict>>,
    lm_options: Option<&Bound<'_, PyDict>>,
) -> PyResult<PySolveResult> {
    let phase_map_p: PhaseMap = phase_map.parse()?;
    let backend_md: BackendMode = backend_mode.parse()?;

    let solve_mode = SolveMode::parse(mode)?;
    let backend = CpuComputeBackend::new(target, backend_md);

    let solver_box: Box<dyn Solver<CpuComputeBackend>> = match solver.to_ascii_lowercase().as_str()
    {
        "bfgs" => Box::new(build_bfgs(bfgs_options)?),
        "lm" => Box::new(build_lm(lm_options)?),
        other => return Err(PyValueError::new_err(format!("unknown solver: {other:?}"))),
    };

    let start = std::time::Instant::now();
    // Release GIL while code is running
    let outcome: anyhow::Result<SolveOutcome> = py.allow_threads(|| {
        catch_unwind(AssertUnwindSafe(|| match seed {
            Some(s) => {
                solver_box.solve_seeded(&backend, solve_mode, phase_map_p, s, init_perturb_mag)
            }
            None => solver_box.solve(&backend, solve_mode, phase_map_p, init_perturb_mag),
        }))
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
    init_perturb_mag = 0.4,
    backend_mode     = "auto",
    seed             = None,
    bfgs_options     = None,
    lm_options       = None,
))]
#[allow(clippy::too_many_arguments)]
fn solve_poly(
    py: Python<'_>,
    ys: PyReadonlyArray1<'_, Complex64>,
    parity: &str,
    solver: &str,
    mode: &str,
    phase_map: &str,
    init_perturb_mag: f64,
    backend_mode: &str,
    seed: Option<u64>,
    bfgs_options: Option<&Bound<'_, PyDict>>,
    lm_options: Option<&Bound<'_, PyDict>>,
) -> PyResult<PySolveResult> {
    let parity: Parity = parity.parse()?;
    let ys_array = ys.as_array().to_owned();
    let target = TargetPoly::new_forced_parity(ys_array, parity);

    __solve(
        py,
        target,
        solver,
        mode,
        phase_map,
        init_perturb_mag,
        backend_mode,
        seed,
        bfgs_options,
        lm_options,
    )
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
    init_perturb_mag = 0.4,
    backend_mode     = "auto",
    seed             = None,
    bfgs_options     = None,
    lm_options       = None,
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
    init_perturb_mag: f64,
    backend_mode: &str,
    seed: Option<u64>,
    bfgs_options: Option<&Bound<'_, PyDict>>,
    lm_options: Option<&Bound<'_, PyDict>>,
) -> PyResult<PySolveResult> {
    let pattern: TargetPattern = target_pattern.parse()?;
    let parity: Parity = parity.parse()?;
    let target = TargetPoly::from_pattern(&pattern, parity, target_half_len);

    __solve(
        py,
        target,
        solver,
        mode,
        phase_map,
        init_perturb_mag,
        backend_mode,
        seed,
        bfgs_options,
        lm_options,
    )
}

#[pymodule]
fn qsp_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySolveResult>()?;
    m.add_function(wrap_pyfunction!(solve_poly_with_pattern, m)?)?;
    m.add_function(wrap_pyfunction!(solve_poly, m)?)?;
    Ok(())
}
