//! The Python classes of the operations of real arithmetic carrying tensors.

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3_tch::PyTensor;

/// The affine map `x ↦ A·x + B` of real matrices: torch tensors of
/// floating-point elements.
#[pyclass(frozen, name = "Linear", module = "reactor.op.real")]
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

    #[classattr]
    fn __match_args__(py: Python<'_>) -> PyResult<Bound<'_, PyTuple>> {
        PyTuple::new(py, ["a", "b"])
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "real.Linear({}, {})",
            super::repr(py, &self.0)?,
            super::repr(py, &self.1)?
        ))
    }

    fn __str__(&self) -> String {
        crate::op::real::Linear(super::display(&self.0), super::display(&self.1)).to_string()
    }
}

impl<'py, T: From<tch::Tensor>> FromPyObject<'py> for crate::op::real::Linear<T> {
    /// Fails when `ob` is not a [`Linear`].
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let linear = ob.downcast::<Linear>()?.get();
        Ok(crate::op::real::Linear(
            linear.0.shallow_clone().into(),
            linear.1.shallow_clone().into(),
        ))
    }
}

impl<'py, T: Into<tch::Tensor>> IntoPyObject<'py> for crate::op::real::Linear<T> {
    type Target = Linear;
    type Output = Bound<'py, Linear>;
    type Error = PyErr;

    /// The [`Linear`] sharing the tensors of `self`, which are not checked
    /// again.
    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        Bound::new(py, Linear(self.0.into(), self.1.into()))
    }
}

/// The affine map `x ↦ A·x + B` on the rates of change of degree `d` of
/// real matrices: torch tensors of floating-point elements, as for
/// [`Linear`].
#[pyclass(frozen, name = "TLinear", module = "reactor.op.real")]
pub struct TLinear(tch::Tensor, tch::Tensor, u8);

// SAFETY: the tensors are private to this module, which only reads their
// shape and element kind and takes shallow clones of them; libtorch makes
// these thread-safe, counting the references to a tensor atomically.
unsafe impl Sync for TLinear {}

#[pymethods]
impl TLinear {
    /// Fails when `a` and `b` are not `torch.Tensor`s of floating-point
    /// elements, of shapes `[m, n]` and `[m, k]`.
    #[new]
    fn new(a: PyTensor, b: PyTensor, d: u8) -> PyResult<Self> {
        let Linear(a, b) = Linear::new(a, b)?;
        Ok(TLinear(a, b, d))
    }

    #[getter]
    fn a(&self) -> PyTensor {
        PyTensor(self.0.shallow_clone())
    }

    #[getter]
    fn b(&self) -> PyTensor {
        PyTensor(self.1.shallow_clone())
    }

    #[getter]
    fn d(&self) -> u8 {
        self.2
    }

    #[classattr]
    fn __match_args__(py: Python<'_>) -> PyResult<Bound<'_, PyTuple>> {
        PyTuple::new(py, ["a", "b", "d"])
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "real.TLinear({}, {}, {})",
            super::repr(py, &self.0)?,
            super::repr(py, &self.1)?,
            self.2
        ))
    }

    fn __str__(&self) -> String {
        crate::op::real::TLinear(super::display(&self.0), super::display(&self.1), self.2)
            .to_string()
    }
}

impl<'py, T: From<tch::Tensor>> FromPyObject<'py> for crate::op::real::TLinear<T> {
    /// Fails when `ob` is not a [`TLinear`].
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let linear = ob.downcast::<TLinear>()?.get();
        Ok(crate::op::real::TLinear(
            linear.0.shallow_clone().into(),
            linear.1.shallow_clone().into(),
            linear.2,
        ))
    }
}

impl<'py, T: Into<tch::Tensor>> IntoPyObject<'py> for crate::op::real::TLinear<T> {
    type Target = TLinear;
    type Output = Bound<'py, TLinear>;
    type Error = PyErr;

    /// The [`TLinear`] sharing the tensors of `self`, which are not checked
    /// again.
    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        Bound::new(py, TLinear(self.0.into(), self.1.into(), self.2))
    }
}

/// A constant real matrix: a `torch.Tensor` of floating-point elements.
#[pyclass(frozen, name = "Const", module = "reactor.op.real")]
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

    #[classattr]
    fn __match_args__(py: Python<'_>) -> PyResult<Bound<'_, PyTuple>> {
        PyTuple::new(py, ["tensor"])
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!("real.Const({})", super::repr(py, &self.0)?))
    }

    fn __str__(&self) -> String {
        crate::op::real::Const(super::display(&self.0)).to_string()
    }
}

impl<'py, T: From<tch::Tensor>> FromPyObject<'py> for crate::op::real::Const<T> {
    /// Fails when `ob` is not a [`Const`].
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let constant = ob.downcast::<Const>()?.get();
        Ok(crate::op::real::Const(constant.0.shallow_clone().into()))
    }
}

impl<'py, T: Into<tch::Tensor>> IntoPyObject<'py> for crate::op::real::Const<T> {
    type Target = Const;
    type Output = Bound<'py, Const>;
    type Error = PyErr;

    /// The [`Const`] sharing the tensor of `self`, which is not checked
    /// again.
    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        Bound::new(py, Const(self.0.into()))
    }
}
