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
    fn new(a: PyTensor, b: PyTensor) -> Result<Self, crate::Error> {
        if !a.is_floating_point() || !b.is_floating_point() {
            return Err(crate::Error(format!(
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

    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("a", "b")
    }
}

impl<'py> FromPyObject<'py> for crate::tensor::real::Linear<tch::Tensor> {
    /// Fails when `ob` is not a [`Linear`].
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let linear = ob.downcast::<Linear>()?.get();
        Ok(crate::tensor::real::Linear::new(
            linear.0.shallow_clone(),
            linear.1.shallow_clone(),
        )?)
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
    fn new(tensor: PyTensor) -> Result<Self, crate::Error> {
        if !tensor.is_floating_point() {
            return Err(crate::Error(format!(
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

    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("tensor",)
    }
}

impl<'py> FromPyObject<'py> for crate::tensor::real::Const<tch::Tensor> {
    /// Fails when `ob` is not a [`Const`].
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let constant = ob.downcast::<Const>()?.get();
        Ok(crate::tensor::real::Const::new(constant.0.shallow_clone()))
    }
}

/// A clone shares the tensor of the original: `tch::Tensor` is not `Clone`,
/// so this takes a shallow clone of it, as libtorch counts its references.
impl Clone for crate::tensor::real::Const<tch::Tensor> {
    fn clone(&self) -> Self {
        crate::tensor::real::Const::new(self.tensor().shallow_clone())
    }
}

/// A clone shares the tensor of the original: `tch::Tensor` is not `Clone`,
/// so this takes a shallow clone of it, as libtorch counts its references.
impl Clone for crate::tensor::real::Linear<tch::Tensor> {
    fn clone(&self) -> Self {
        crate::tensor::real::Linear::new(self.a().shallow_clone(), self.b().shallow_clone())
            .expect("a clone has the shapes of the original")
    }
}

/// Converts an operation of the theory into its Python class: the
/// operations carrying tensors into the classes of this module, holding the
/// same tensors, and the others into their own classes.
impl<'py> IntoPyObject<'py> for crate::tensor::real::NRA<tch::Tensor> {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        use crate::tensor::real::NRA;

        match self {
            NRA::Const(op) => Const(op.tensor().shallow_clone())
                .into_pyobject(py)
                .map(Bound::into_any),
            NRA::Add(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Sub(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Mul(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Div(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Linear(op) => Linear(op.a().shallow_clone(), op.b().shallow_clone())
                .into_pyobject(py)
                .map(Bound::into_any),
            NRA::ReLU(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Argmax(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Min(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Max(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Transpose(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::MatMul(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Lt(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Ite(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Le(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Gt(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Ge(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Eq(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Ne(op) => op.into_pyobject(py).map(Bound::into_any),
        }
    }
}
