use ndarray::{Array1, ArrayD};
use numpy::{Element, IntoPyArray, PyArrayLikeDyn, PyReadonlyArray1, TypeMustMatch};
use pyo3::{
    Bound, FromPyObject, IntoPyObject, IntoPyObjectExt, PyAny, PyErr, PyResult, Python,
    exceptions::{PyTypeError, PyValueError},
    types::{PyAnyMethods, PyDict},
};
use qsp_rs_core::solvers::{
    TerminationReason, bfgs::BfgsOptions, configuration::PhaseGenerator, lm::LmOptions,
};

pub fn vectorize<'py, T, U, F, E>(
    py: Python<'py>,
    arg: &Bound<'py, PyAny>,
    f: F,
) -> PyResult<Bound<'py, PyAny>>
where
    T: Element + Copy + for<'a> FromPyObject<'a> + IntoPyObject<'py> + 'py,
    U: Element + IntoPyObject<'py>,
    F: Fn(T) -> Result<U, E>,
    E: Into<PyErr>,
{
    // 1) Scalar path: try to read a single value of type T.
    if let Ok(scalar) = arg.extract::<T>() {
        return f(scalar).map_err(Into::into)?.into_bound_py_any(py);
    }

    // 2) Array path: coerce to an N-dim numpy view, map, hand back a numpy array.
    let arr: PyArrayLikeDyn<'py, T, TypeMustMatch> = arg.extract()?;
    let view = arr.as_array();
    let data: Vec<U> = view
        .iter()
        .map(|&x| f(x))
        .collect::<Result<Vec<U>, E>>() // short-circuits on first Err
        .map_err(Into::into)?;
    let out = ArrayD::from_shape_vec(view.raw_dim(), data)
        .expect("element count is unchanged, so the shape always matches");
    Ok(out.into_pyarray(py).into_any())
}

pub fn termination_str(t: TerminationReason) -> &'static str {
    match t {
        TerminationReason::Converged => "converged",
        TerminationReason::MaxItersReached => "max_iters_reached",
        TerminationReason::LineSearchFailed => "line_search_failed",
        TerminationReason::Diverged => "diverged",
        TerminationReason::Other => "other",
    }
}

pub fn override_field<T>(dict: &Bound<'_, PyDict>, key: &str, slot: &mut T) -> PyResult<()>
where
    T: for<'py> FromPyObject<'py>,
{
    let v = dict.get_item(key)?;
    *slot = v
        .extract::<T>()
        .map_err(|e| PyValueError::new_err(format!("option {key}: {e}")))?;

    Ok(())
}

pub fn build_bfgs(overrides: Option<&Bound<'_, PyDict>>) -> PyResult<BfgsOptions> {
    let mut o = BfgsOptions::default();
    if let Some(d) = overrides {
        override_field(d, "max_iters", &mut o.max_iters)?;
        override_field(d, "mem", &mut o.mem)?;
        override_field(d, "tol_grad", &mut o.tol_grad)?;
    }
    Ok(o)
}

pub fn build_lm(overrides: Option<&Bound<'_, PyDict>>) -> PyResult<LmOptions> {
    let mut o = LmOptions::default();
    if let Some(d) = overrides {
        override_field(d, "max_iters", &mut o.max_iters)?;
        override_field(d, "initial_lambda", &mut o.initial_lambda)?;
        override_field(d, "tol", &mut o.tol)?;
    }
    Ok(o)
}

pub fn phase_gen_from_pyobj(obj: &Bound<'_, PyAny>) -> PyResult<PhaseGenerator> {
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
