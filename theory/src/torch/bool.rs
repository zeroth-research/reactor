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

/// A clone shares the tensor of the original: `tch::Tensor` is not `Clone`,
/// so this takes a shallow clone of it, as libtorch counts its references.
impl Clone for crate::tensor::bool::Const<tch::Tensor> {
    fn clone(&self) -> Self {
        crate::tensor::bool::Const(self.0.shallow_clone())
    }
}

/// Converts an operation of the theory into its Python class: the
/// operations carrying tensors into the classes of this module, holding the
/// same tensors, and the others into their own classes.
impl<'py> IntoPyObject<'py> for crate::tensor::bool::Prop<tch::Tensor> {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        use crate::tensor::bool::Prop;

        match self {
            Prop::Const(op) => Const(op.0).into_pyobject(py).map(Bound::into_any),
            Prop::And(op) => op.into_pyobject(py).map(Bound::into_any),
            Prop::Or(op) => op.into_pyobject(py).map(Bound::into_any),
            Prop::Xor(op) => op.into_pyobject(py).map(Bound::into_any),
            Prop::Not(op) => op.into_pyobject(py).map(Bound::into_any),
        }
    }
}
