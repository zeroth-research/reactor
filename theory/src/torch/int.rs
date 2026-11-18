//! The Python classes of the operations of integer arithmetic carrying
//! tensors.

use pyo3::prelude::*;
use pyo3_tch::PyTensor;

/// The affine map `x ↦ A·x + B` of integer matrices: torch tensors of
/// integer elements.
#[pyclass(frozen, name = "Linear", module = "int")]
pub struct Linear(tch::Tensor, tch::Tensor);

// SAFETY: the tensors are private to this module, which only reads their
// shape and element kind and takes shallow clones of them; libtorch makes
// these thread-safe, counting the references to a tensor atomically.
unsafe impl Sync for Linear {}

#[pymethods]
impl Linear {
    /// Fails when `a` and `b` are not `torch.Tensor`s of integer
    /// elements, of shapes `[m, n]` and `[m, k]`.
    #[new]
    fn new(a: PyTensor, b: PyTensor) -> Result<Self, crate::Error> {
        if !super::is_integer(a.kind()) || !super::is_integer(b.kind()) {
            return Err(crate::Error(format!(
                "expected tensors of integer elements, got {:?} and {:?}",
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

    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("a", "b")
    }
}

impl<'py> FromPyObject<'py> for crate::tensor::int::Linear<tch::Tensor> {
    /// Fails when `ob` is not an [`Linear`].
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let linear = ob.downcast::<Linear>()?.get();
        Ok(crate::tensor::int::Linear(
            linear.0.shallow_clone(),
            linear.1.shallow_clone(),
        ))
    }
}

/// A constant integer matrix: a `torch.Tensor` of integer elements.
#[pyclass(frozen, name = "Const", module = "int")]
pub struct Const(tch::Tensor);

// SAFETY: the tensor is private to this module, which only reads its
// shape and element kind and takes shallow clones of it; libtorch makes
// these thread-safe, counting the references to a tensor atomically.
unsafe impl Sync for Const {}

#[pymethods]
impl Const {
    /// Fails when `tensor` is not a `torch.Tensor` of integer elements.
    #[new]
    fn new(tensor: PyTensor) -> Result<Self, crate::Error> {
        if !super::is_integer(tensor.kind()) {
            return Err(crate::Error(format!(
                "expected a tensor of integer elements, got {:?}",
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
    fn __match_args__() -> (&'static str,) {
        ("tensor",)
    }
}

impl<'py> FromPyObject<'py> for crate::tensor::int::Const<tch::Tensor> {
    /// Fails when `ob` is not an [`Const`].
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let constant = ob.downcast::<Const>()?.get();
        Ok(crate::tensor::int::Const(constant.0.shallow_clone()))
    }
}

/// A clone shares the tensor of the original: `tch::Tensor` is not `Clone`,
/// so this takes a shallow clone of it, as libtorch counts its references.
impl Clone for crate::tensor::int::Const<tch::Tensor> {
    fn clone(&self) -> Self {
        crate::tensor::int::Const(self.0.shallow_clone())
    }
}

/// A clone shares the tensor of the original: `tch::Tensor` is not `Clone`,
/// so this takes a shallow clone of it, as libtorch counts its references.
impl Clone for crate::tensor::int::Linear<tch::Tensor> {
    fn clone(&self) -> Self {
        crate::tensor::int::Linear(self.0.shallow_clone(), self.1.shallow_clone())
    }
}

/// Converts an operation of the theory into its Python class: the
/// operations carrying tensors into the classes of this module, holding the
/// same tensors, and the others into their own classes.
impl<'py> IntoPyObject<'py> for crate::tensor::int::NIA<tch::Tensor> {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        use crate::tensor::int::NIA;

        match self {
            NIA::Const(op) => Const(op.0).into_pyobject(py).map(Bound::into_any),
            NIA::Add(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Sub(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Mul(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Div(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Mod(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Linear(op) => Linear(op.0, op.1).into_pyobject(py).map(Bound::into_any),
            NIA::ReLU(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Argmax(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Min(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Max(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Transpose(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::MatMul(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Lt(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Ite(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Le(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Gt(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Ge(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Eq(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Ne(op) => op.into_pyobject(py).map(Bound::into_any),
        }
    }
}
