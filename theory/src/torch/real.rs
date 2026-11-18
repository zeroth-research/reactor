//! The Python classes of the operations of real arithmetic carrying tensors.

use pyo3::prelude::*;
use pyo3_tch::PyTensor;

/// The affine map `x ↦ A·x + B` of real matrices: torch tensors of
/// floating-point elements.
#[pyclass(frozen, name = "Linear", module = "real")]
pub struct Linear(tch::Tensor, tch::Tensor);

// SAFETY: the tensors are private to this module, which only reads their
// shape and element kind and takes shallow clones of them; libtorch makes
// these thread-safe, counting the references to a tensor atomically.
unsafe impl Sync for Linear {}

#[pymethods]
impl Linear {
    /// Fails when `a` and `b` are not `torch.Tensor`s of floating-point
    /// elements, of shapes `[m, n]` and `[m, k]`.
    #[new]
    fn new(a: PyTensor, b: PyTensor) -> PyResult<Self> {
        if !a.is_floating_point() || !b.is_floating_point() {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "expected tensors of floating-point elements, got {:?} and {:?}",
                a.kind(),
                b.kind()
            )));
        }
        super::check_affine(&a, &b)?;
        Ok(Linear(a.0, b.0))
    }

    #[getter]
    fn a(&self) -> PyTensor {
        PyTensor(self.0.shallow_clone())
    }

    #[getter]
    fn b(&self) -> PyTensor {
        PyTensor(self.1.shallow_clone())
    }
}

impl<'py> FromPyObject<'py> for crate::op::real::Linear<tch::Tensor> {
    /// Fails when `ob` is not a [`Linear`].
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let linear = ob.downcast::<Linear>()?.get();
        Ok(crate::op::real::Linear(
            linear.0.shallow_clone(),
            linear.1.shallow_clone(),
        ))
    }
}

/// A constant real matrix: a `torch.Tensor` of floating-point elements.
#[pyclass(frozen, name = "Const", module = "real")]
pub struct Const(tch::Tensor);

// SAFETY: the tensor is private to this module, which only reads its
// shape and element kind and takes shallow clones of it; libtorch makes
// these thread-safe, counting the references to a tensor atomically.
unsafe impl Sync for Const {}

#[pymethods]
impl Const {
    /// Fails when `tensor` is not a `torch.Tensor` of floating-point
    /// elements.
    #[new]
    fn new(tensor: PyTensor) -> PyResult<Self> {
        if !tensor.is_floating_point() {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "expected a tensor of floating-point elements, got {:?}",
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

impl<'py> FromPyObject<'py> for crate::op::real::Const<tch::Tensor> {
    /// Fails when `ob` is not a [`Const`].
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let constant = ob.downcast::<Const>()?.get();
        Ok(crate::op::real::Const(constant.0.shallow_clone()))
    }
}
