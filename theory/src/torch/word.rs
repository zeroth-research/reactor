//! The Python class of the constants of fixed-size bit-vectors.

use pyo3::prelude::*;
use pyo3_tch::PyTensor;

/// A constant matrix of bit-vectors of a width: a `torch.Tensor` of
/// integer elements, each holding a bit-vector in its low bits.
#[pyclass(frozen, name = "Const", module = "word")]
pub struct Const(tch::Tensor, usize);

// SAFETY: the tensor is private to this module, which only reads its
// shape and element kind and takes shallow clones of it; libtorch makes
// these thread-safe, counting the references to a tensor atomically.
unsafe impl Sync for Const {}

#[pymethods]
impl Const {
    /// Fails when `tensor` is not a `torch.Tensor` of integer elements.
    #[new]
    fn new(tensor: PyTensor, width: usize) -> PyResult<Self> {
        if !super::is_integer(tensor.kind()) {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "expected a tensor of integer elements, got {:?}",
                tensor.kind()
            )));
        }
        Ok(Const(tensor.0, width))
    }

    #[getter]
    fn tensor(&self) -> PyTensor {
        PyTensor(self.0.shallow_clone())
    }

    #[getter]
    fn width(&self) -> usize {
        self.1
    }
}

impl<'py> FromPyObject<'py> for crate::tensor::word::Const<tch::Tensor> {
    /// Fails when `ob` is not a [`Const`].
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let constant = ob.downcast::<Const>()?.get();
        Ok(crate::tensor::word::Const(
            constant.0.shallow_clone(),
            constant.1,
        ))
    }
}
