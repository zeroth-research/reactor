//! The tensors in Python, with the `pyo3` feature: a tensor converts from
//! and into its Python object, a `torch.Tensor`, or, for a [`Builtin`], a
//! list of rows.

#[cfg(feature = "torch")]
use super::Torch;
use super::{Builtin, Error, Tensor, Value};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyInt, PyList, PyTuple};

impl From<Error> for PyErr {
    /// A `TypeError` for elements of another kind, and a `ValueError` for
    /// tensors of another shape.
    fn from(error: Error) -> Self {
        match error {
            Error::Kind { .. } => PyTypeError::new_err(error.to_string()),
            Error::NotAMatrix(_) | Error::Affine { .. } => PyValueError::new_err(error.to_string()),
        }
    }
}

/// The Python objects that convert into a tensor, for the errors.
fn tensors() -> String {
    let backends: Vec<&str> = [(cfg!(feature = "torch"), "a torch.Tensor")]
        .into_iter()
        .filter_map(|(enabled, object)| enabled.then_some(object))
        .collect();
    if backends.is_empty() {
        "a matrix: a list of rows".into()
    } else {
        format!("a matrix: {} or a list of rows", backends.join(", "))
    }
}

impl<'py> FromPyObject<'py> for Tensor {
    /// Fails unless `ob` is a matrix: a `torch.Tensor`, with the `torch`
    /// feature, which the tensor shares, or a list of rows of numbers or
    /// booleans, which it copies into a [`Builtin`].
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        #[cfg(feature = "torch")]
        if let Some(t) = Torch::from_python(ob)? {
            return Ok(Tensor::Torch(t));
        }
        if ob.is_instance_of::<PyList>() || ob.is_instance_of::<PyTuple>() {
            return Ok(Tensor::Builtin(ob.extract()?));
        }
        Err(PyTypeError::new_err(format!(
            "expected {}, got {}",
            tensors(),
            ob.get_type()
        )))
    }
}

impl<'py> IntoPyObject<'py> for Tensor {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    /// The Python object of the tensor: the one it shares for a torch
    /// tensor, and a list of rows for a [`Builtin`].
    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        match self {
            Tensor::Builtin(t) => t.into_bound_py_any(py),
            #[cfg(feature = "torch")]
            Tensor::Torch(t) => t.into_pyobject(py),
        }
    }
}

impl<'py> FromPyObject<'py> for Builtin {
    /// Fails unless `ob` is a sequence of rows of equal length. The elements
    /// are booleans if all are, integers if all are, and floating-point
    /// numbers otherwise.
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let rows: Vec<Vec<Bound<'py, PyAny>>> = ob.extract()?;
        let cols = rows.first().map_or(0, Vec::len);
        if rows.iter().any(|row| row.len() != cols) {
            return Err(PyValueError::new_err("expected rows of equal length"));
        }
        let shape = [rows.len(), cols];
        let elements: Vec<_> = rows.into_iter().flatten().collect();
        let all =
            |is: fn(&Bound<'py, PyAny>) -> bool| !elements.is_empty() && elements.iter().all(is);
        if all(|e| e.is_instance_of::<PyBool>()) {
            Ok(Builtin::bools(shape, values(&elements)?))
        } else if all(|e| e.is_instance_of::<PyInt>()) {
            Ok(Builtin::ints(shape, values(&elements)?))
        } else {
            Ok(Builtin::floats(shape, values(&elements)?))
        }
    }
}

/// The Python objects `elements` as values of `T`.
fn values<'py, T: FromPyObject<'py>>(elements: &[Bound<'py, PyAny>]) -> PyResult<Vec<T>> {
    elements.iter().map(|e| e.extract()).collect()
}

impl<'py> IntoPyObject<'py> for Builtin {
    type Target = PyList;
    type Output = Bound<'py, PyList>;
    type Error = PyErr;

    /// The list of the rows of the matrix, each a list of its elements.
    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        let rows = self.rows().map(|row| {
            let elements = row.map(|value| match value {
                Value::Bool(v) => v.into_bound_py_any(py),
                Value::Int(v) => v.into_bound_py_any(py),
                Value::Float(v) => v.into_bound_py_any(py),
            });
            PyList::new(py, elements.collect::<PyResult<Vec<_>>>()?)
        });
        PyList::new(py, rows.collect::<PyResult<Vec<_>>>()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Kind;

    #[test]
    fn lists_of_rows_are_builtin_matrices_of_their_kind() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let extract = |code: &str| -> PyResult<Tensor> {
                py.eval(&std::ffi::CString::new(code).unwrap(), None, None)?
                    .extract()
            };
            let t = extract("[[1.5, 2], [3, 4]]").unwrap();
            assert_eq!((t.shape(), t.kind()), ([2, 2], Kind::Float));
            assert_eq!(extract("[[1, 2, 3]]").unwrap().kind(), Kind::Int);
            assert_eq!(extract("((True,), (False,))").unwrap().kind(), Kind::Bool);
            assert!(
                extract("[[1], [2, 3]]")
                    .unwrap_err()
                    .is_instance_of::<PyValueError>(py)
            );
            assert!(extract("42").unwrap_err().is_instance_of::<PyTypeError>(py));

            let back = t.into_pyobject(py).unwrap();
            assert_eq!(back.str().unwrap().to_string(), "[[1.5, 2.0], [3.0, 4.0]]");
        });
    }
}
