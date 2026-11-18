/*!
# The sorts in Python

Each sort is a Python class of the same name, which the Python module of
`reactor` exports: `Real(shape, degree=0)`, `Int(shape)`, `Bool(shape)`,
`Word(width, shape)` and `Zero()`, where `shape` is `[rows, cols]`.

A sort's constructor takes its fields in order, and `__match_args__` lists
them, so that `match` takes a sort apart as it is built, as in
`case Real([m, n]):` or `case Word(w, [m, n]):`. Its fields are also read
as attributes, as `sort.shape`. Sorts are immutable and hashable, compare
by value, and print as in Rust (`str`) or as their constructor (`repr`).

[`Any`] converts from any of these classes, and into the class of the
sort it holds.
*/

use super::{Any, Bool, Int, Real, Word, Zero};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

#[pymethods]
impl Real {
    #[new]
    #[pyo3(signature = (shape, degree = 0))]
    fn py_new(shape: [usize; 2], degree: u8) -> Self {
        Real::with_degree(shape, degree)
    }

    #[getter(shape)]
    fn py_shape(&self) -> [usize; 2] {
        self.shape
    }

    #[getter(degree)]
    fn py_degree(&self) -> u8 {
        self.degree
    }

    #[classattr]
    fn __match_args__(py: Python<'_>) -> PyResult<Bound<'_, PyTuple>> {
        PyTuple::new(py, ["shape", "degree"])
    }

    fn __repr__(&self) -> String {
        let [m, n] = self.shape;
        match self.degree {
            0 => format!("Real([{m}, {n}])"),
            d => format!("Real([{m}, {n}], {d})"),
        }
    }
}

#[pymethods]
impl Int {
    #[new]
    fn py_new(shape: [usize; 2]) -> Self {
        Int::new(shape)
    }

    #[getter(shape)]
    fn py_shape(&self) -> [usize; 2] {
        self.shape
    }

    #[classattr]
    fn __match_args__(py: Python<'_>) -> PyResult<Bound<'_, PyTuple>> {
        PyTuple::new(py, ["shape"])
    }

    fn __repr__(&self) -> String {
        let [m, n] = self.shape;
        format!("Int([{m}, {n}])")
    }
}

#[pymethods]
impl Bool {
    #[new]
    fn py_new(shape: [usize; 2]) -> Self {
        Bool::new(shape)
    }

    #[getter(shape)]
    fn py_shape(&self) -> [usize; 2] {
        self.shape
    }

    #[classattr]
    fn __match_args__(py: Python<'_>) -> PyResult<Bound<'_, PyTuple>> {
        PyTuple::new(py, ["shape"])
    }

    fn __repr__(&self) -> String {
        let [m, n] = self.shape;
        format!("Bool([{m}, {n}])")
    }
}

#[pymethods]
impl Word {
    /// Raises `ValueError` if `width` is zero.
    #[new]
    fn py_new(width: usize, shape: [usize; 2]) -> PyResult<Self> {
        Word::new(width, shape).map_err(PyValueError::new_err)
    }

    #[getter(width)]
    fn py_width(&self) -> usize {
        self.width
    }

    #[getter(shape)]
    fn py_shape(&self) -> [usize; 2] {
        self.shape
    }

    #[classattr]
    fn __match_args__(py: Python<'_>) -> PyResult<Bound<'_, PyTuple>> {
        PyTuple::new(py, ["width", "shape"])
    }

    fn __repr__(&self) -> String {
        let [m, n] = self.shape;
        format!("Word({}, [{m}, {n}])", self.width)
    }
}

#[pymethods]
impl Zero {
    #[new]
    fn py_new() -> Self {
        Zero
    }

    #[classattr]
    fn __match_args__(py: Python<'_>) -> PyResult<Bound<'_, PyTuple>> {
        PyTuple::new(py, Vec::<&str>::new())
    }

    fn __repr__(&self) -> String {
        "Zero()".to_string()
    }
}

/// The sorts' classes, to add to a Python module.
pub fn add_classes(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Real>()?;
    module.add_class::<Int>()?;
    module.add_class::<Bool>()?;
    module.add_class::<Word>()?;
    module.add_class::<Zero>()?;
    Ok(())
}

impl<'py> FromPyObject<'py> for Any {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(s) = obj.downcast::<Real>() {
            return Ok((*s.get()).into());
        }
        if let Ok(s) = obj.downcast::<Int>() {
            return Ok((*s.get()).into());
        }
        if let Ok(s) = obj.downcast::<Bool>() {
            return Ok((*s.get()).into());
        }
        if let Ok(s) = obj.downcast::<Word>() {
            return Ok((*s.get()).into());
        }
        if let Ok(s) = obj.downcast::<Zero>() {
            return Ok((*s.get()).into());
        }
        Err(PyTypeError::new_err(format!(
            "expected a sort (Real, Int, Bool, Word or Zero), got {}",
            obj.get_type()
        )))
    }
}

impl<'py> IntoPyObject<'py> for Any {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        match self {
            Any::Real(s) => Bound::new(py, s).map(Bound::into_any),
            Any::Int(s) => Bound::new(py, s).map(Bound::into_any),
            Any::Bool(s) => Bound::new(py, s).map(Bound::into_any),
            Any::Word(s) => Bound::new(py, s).map(Bound::into_any),
            Any::Zero(s) => Bound::new(py, s).map(Bound::into_any),
        }
    }
}
