//! The tensors in Python, with the `pyo3` feature: a tensor converts from
//! and into its Python object, a `torch.Tensor`, a `jax.Array`, or, for a
//! [`VecTensor`], a list of rows.

#[cfg(feature = "jax")]
use super::JaxTensor;
#[cfg(feature = "torch")]
use super::TorchTensor;
use super::vec::Rows;
use super::{Error, Tensor, VecTensor};
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
            Error::NotAMatrix(_) | Error::Rows { .. } | Error::Affine { .. } => {
                PyValueError::new_err(error.to_string())
            }
        }
    }
}

/// The Python objects that convert into a tensor, for the errors.
fn tensors() -> String {
    let backends: Vec<&str> = [
        (cfg!(feature = "torch"), "a torch.Tensor"),
        (cfg!(feature = "jax"), "a jax.Array"),
    ]
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
    /// Fails unless `ob` is a matrix: a `torch.Tensor` or a `jax.Array`,
    /// with their features, which the tensor shares, or a list of rows of
    /// numbers or booleans, which it copies into a [`VecTensor`].
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        #[cfg(feature = "torch")]
        if let Some(t) = TorchTensor::from_python(ob)? {
            return Ok(Tensor::Torch(t));
        }
        #[cfg(feature = "jax")]
        if JaxTensor::is_array(ob)? {
            return Ok(Tensor::Jax(JaxTensor::new(ob)?));
        }
        if ob.is_instance_of::<PyList>() || ob.is_instance_of::<PyTuple>() {
            return Ok(Tensor::Vec(ob.extract()?));
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
    /// tensor or a JAX array, and a list of rows for a [`VecTensor`].
    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        match self {
            Tensor::Vec(t) => t.into_pyobject(py),
            #[cfg(feature = "torch")]
            Tensor::Torch(t) => t.into_pyobject(py),
            #[cfg(feature = "jax")]
            Tensor::Jax(t) => t.into_pyobject(py),
        }
    }
}

impl<'py> FromPyObject<'py> for VecTensor {
    /// Fails unless `ob` is a sequence of rows of equal length. The elements
    /// are booleans if all are, integers if all are, and floating-point
    /// numbers otherwise.
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let rows: Vec<Vec<Bound<'py, PyAny>>> = ob.extract()?;
        let shape = [rows.len(), rows.first().map_or(0, Vec::len)];
        let all = |is: fn(&Bound<'py, PyAny>) -> bool| {
            rows.iter().flatten().next().is_some() && rows.iter().flatten().all(is)
        };
        let tensor = if all(|e| e.is_instance_of::<PyBool>()) {
            VecTensor::bools(shape, values(&rows)?)
        } else if all(|e| e.is_instance_of::<PyInt>()) {
            VecTensor::ints(shape, values(&rows)?)
        } else {
            VecTensor::floats(shape, values(&rows)?)
        };
        Ok(tensor?)
    }
}

/// The rows of Python objects `rows` as rows of values of `T`.
fn values<'py, T: FromPyObject<'py>>(rows: &[Vec<Bound<'py, PyAny>>]) -> PyResult<Vec<Vec<T>>> {
    rows.iter()
        .map(|row| row.iter().map(|e| e.extract()).collect())
        .collect()
}

impl<'py> IntoPyObject<'py> for VecTensor {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    /// The list of the rows of the matrix, each a list of its elements.
    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        match self.rows {
            Rows::Bool(rows) => rows.into_bound_py_any(py),
            Rows::Int(rows) => rows.into_bound_py_any(py),
            Rows::Float(rows) => rows.into_bound_py_any(py),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Kind;

    #[test]
    fn lists_of_rows_are_vec_tensors_of_their_kind() {
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
