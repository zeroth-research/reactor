//! The Python class of the constants of propositional logic.

use pyo3::prelude::*;
use pyo3_tch::PyTensor;

/// A constant boolean matrix: a `torch.Tensor` of boolean elements.
#[pyclass(frozen, name = "Const", module = "bool")]
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
}

impl<'py> FromPyObject<'py> for crate::tensor::bool::Const<tch::Tensor> {
    /// Fails when `ob` is not a [`Const`].
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let constant = ob.downcast::<Const>()?.get();
        Ok(crate::tensor::bool::Const(constant.0.shallow_clone()))
    }
}
