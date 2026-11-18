//! The Python class of the constants of propositional logic.

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3_tch::PyTensor;

/// A constant boolean matrix: a `torch.Tensor` of boolean elements.
#[pyclass(frozen, name = "Const", module = "reactor.op.bool")]
pub struct Const(tch::Tensor);

// SAFETY: the tensor is private to this module, which only reads its
// shape and element kind and takes shallow clones of it; libtorch makes
// these thread-safe, counting the references to a tensor atomically.
unsafe impl Sync for Const {}

#[pymethods]
impl Const {
    /// Fails when `tensor` is not a `torch.Tensor` of boolean elements.
    #[new]
    fn new(tensor: PyTensor) -> PyResult<Self> {
        if tensor.kind() != tch::Kind::Bool {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "expected a tensor of boolean elements, got {:?}",
                tensor.kind()
            )));
        }
        Ok(Const(tensor.0))
    }

    #[getter]
    fn tensor(&self) -> PyTensor {
        PyTensor(self.0.shallow_clone())
    }

    #[classattr]
    fn __match_args__(py: Python<'_>) -> PyResult<Bound<'_, PyTuple>> {
        PyTuple::new(py, ["tensor"])
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!("bool.Const({})", super::repr(py, &self.0)?))
    }

    fn __str__(&self) -> String {
        crate::op::bool::Const(super::display(&self.0)).to_string()
    }
}

impl<'py, T: From<tch::Tensor>> FromPyObject<'py> for crate::op::bool::Const<T> {
    /// Fails when `ob` is not a [`Const`].
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let constant = ob.downcast::<Const>()?.get();
        Ok(crate::op::bool::Const(constant.0.shallow_clone().into()))
    }
}

impl<'py, T: Into<tch::Tensor>> IntoPyObject<'py> for crate::op::bool::Const<T> {
    type Target = Const;
    type Output = Bound<'py, Const>;
    type Error = PyErr;

    /// The [`Const`] sharing the tensor of `self`, which is not checked
    /// again.
    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        Bound::new(py, Const(self.0.into()))
    }
}
