use ndarray::Array1;
use numpy::{Complex64, IntoPyArray, PyArray1};
use pyo3::{Bound, Python, pyclass, pymethods};

#[pyclass(name = "TargetPoly", module = "qsp_rs", frozen)]
#[derive(Debug, Clone)]
pub struct PyTargetPoly {
    pub xs: Array1<f64>,
    pub ys: Array1<Complex64>,
    #[pyo3(get)]
    pub n_half: usize,
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
    pub cost: f64,
    pub phases: Array1<f64>,
    #[pyo3(get)]
    pub iterations: u64,
    #[pyo3(get)]
    pub termination: String,
    #[pyo3(get)]
    pub elapsed_ms: f64,
    #[pyo3(get)]
    pub target: PyTargetPoly,
    #[pyo3(get)]
    pub total_phase: f64,
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
