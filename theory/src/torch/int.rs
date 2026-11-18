//! The Python classes of the operations of integer arithmetic carrying
//! tensors.

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3_tch::PyTensor;

/// The affine map `x ↦ A·x + B` of integer matrices: torch tensors of
/// integer elements.
#[pyclass(frozen, name = "Linear", module = "reactor.op.int")]
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
    fn new(a: PyTensor, b: PyTensor) -> PyResult<Self> {
        if !super::is_integer(a.kind()) || !super::is_integer(b.kind()) {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
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
    fn __match_args__(py: Python<'_>) -> PyResult<Bound<'_, PyTuple>> {
        PyTuple::new(py, ["a", "b"])
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "int.Linear({}, {})",
            super::repr(py, &self.0)?,
            super::repr(py, &self.1)?
        ))
    }

    fn __str__(&self) -> String {
        crate::op::int::Linear(super::display(&self.0), super::display(&self.1)).to_string()
    }
}

impl<'py, T: From<tch::Tensor>> FromPyObject<'py> for crate::op::int::Linear<T> {
    /// Fails when `ob` is not an [`Linear`].
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let linear = ob.downcast::<Linear>()?.get();
        Ok(crate::op::int::Linear(
            linear.0.shallow_clone().into(),
            linear.1.shallow_clone().into(),
        ))
    }
}

impl<'py, T: Into<tch::Tensor>> IntoPyObject<'py> for crate::op::int::Linear<T> {
    type Target = Linear;
    type Output = Bound<'py, Linear>;
    type Error = PyErr;

    /// The [`Linear`] sharing the tensors of `self`, which are not checked
    /// again.
    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        Bound::new(py, Linear(self.0.into(), self.1.into()))
    }
}

/// A constant integer matrix: a `torch.Tensor` of integer elements.
#[pyclass(frozen, name = "Const", module = "reactor.op.int")]
pub struct Const(tch::Tensor);

// SAFETY: the tensor is private to this module, which only reads its
// shape and element kind and takes shallow clones of it; libtorch makes
// these thread-safe, counting the references to a tensor atomically.
unsafe impl Sync for Const {}

#[pymethods]
impl Const {
    /// Fails when `tensor` is not a `torch.Tensor` of integer elements.
    #[new]
    fn new(tensor: PyTensor) -> PyResult<Self> {
        if !super::is_integer(tensor.kind()) {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
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
    fn __match_args__(py: Python<'_>) -> PyResult<Bound<'_, PyTuple>> {
        PyTuple::new(py, ["tensor"])
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!("int.Const({})", super::repr(py, &self.0)?))
    }

    fn __str__(&self) -> String {
        crate::op::int::Const(super::display(&self.0)).to_string()
    }
}

impl<'py, T: From<tch::Tensor>> FromPyObject<'py> for crate::op::int::Const<T> {
    /// Fails when `ob` is not an [`Const`].
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let constant = ob.downcast::<Const>()?.get();
        Ok(crate::op::int::Const(constant.0.shallow_clone().into()))
    }
}

impl<'py, T: Into<tch::Tensor>> IntoPyObject<'py> for crate::op::int::Const<T> {
    type Target = Const;
    type Output = Bound<'py, Const>;
    type Error = PyErr;

    /// The [`Const`] sharing the tensor of `self`, which is not checked
    /// again.
    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        Bound::new(py, Const(self.0.into()))
    }
}
