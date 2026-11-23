//! The Python classes of the operations of real arithmetic carrying tensors.

use pyo3::prelude::*;
use pyo3_tch::PyTensor;

/// The affine map `x ↦ A·x + B` of real matrices: torch tensors of
/// floating-point elements.
#[pyclass(frozen, name = "Linear", module = "real")]
pub struct Linear(tch::Tensor, tch::Tensor, u8);

// SAFETY: the tensors are private to this module, which only reads their
// shape and element kind and takes shallow clones of them; libtorch makes
// these thread-safe, counting the references to a tensor atomically.
unsafe impl Sync for Linear {}

#[pymethods]
impl Linear {
    /// The map of real matrices of the given `degree`, which `B` is of;
    /// fails when `a` and `b` are not `torch.Tensor`s of floating-point
    /// elements, of shapes `[m, n]` and `[m, k]`.
    #[new]
    #[pyo3(signature = (a, b, /, degree = 0))]
    fn new(a: PyTensor, b: PyTensor, degree: u8) -> Result<Self, crate::Error> {
        if !a.is_floating_point() || !b.is_floating_point() {
            return Err(crate::Error(format!(
                "expected tensors of floating-point elements, got {:?} and {:?}",
                a.kind(),
                b.kind()
            )));
        }
        super::check_affine(&a, &b)?;
        Ok(Linear(a.0, b.0, degree))
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
    fn degree(&self) -> u8 {
        self.2
    }

    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("a", "b")
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for crate::tensor::real::Linear<tch::Tensor> {
    type Error = PyErr;

    /// Fails when `ob` is not a [`Linear`].
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let linear = ob.cast::<Linear>()?.get();
        Ok(crate::tensor::real::Linear::with_degree(
            linear.0.shallow_clone(),
            linear.1.shallow_clone(),
            linear.2,
        )?)
    }
}

/// A constant real matrix: a `torch.Tensor` of floating-point or integer
/// elements, kept as given, since integers are reals too.
#[pyclass(frozen, name = "Const", module = "real")]
pub struct Const(tch::Tensor, u8);

// SAFETY: the tensor is private to this module, which only reads its
// shape and element kind and takes shallow clones of it; libtorch makes
// these thread-safe, counting the references to a tensor atomically.
unsafe impl Sync for Const {}

#[pymethods]
impl Const {
    /// The constant of the given `degree`; fails when `tensor` is not a
    /// `torch.Tensor` of two dimensions and floating-point or integer
    /// elements.
    #[new]
    #[pyo3(signature = (tensor, /, degree = 0))]
    fn new(tensor: PyTensor, degree: u8) -> Result<Self, crate::Error> {
        super::check_matrix(&tensor)?;
        if !tensor.is_floating_point() && !super::is_integer(tensor.kind()) {
            return Err(crate::Error(format!(
                "expected a tensor of floating-point or integer elements, got {:?}",
                tensor.kind()
            )));
        }
        Ok(Const(tensor.0, degree))
    }

    #[getter]
    fn tensor(&self) -> PyTensor {
        PyTensor(self.0.shallow_clone())
    }

    #[getter]
    fn degree(&self) -> u8 {
        self.1
    }

    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("tensor",)
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for crate::tensor::real::Const<tch::Tensor> {
    type Error = PyErr;

    /// Fails when `ob` is not a [`Const`].
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let constant = ob.cast::<Const>()?.get();
        Ok(crate::tensor::real::Const::with_degree(
            constant.0.shallow_clone(),
            constant.1,
        ))
    }
}

/// A clone shares the tensor of the original: `tch::Tensor` is not `Clone`,
/// so this takes a shallow clone of it, as libtorch counts its references.
impl Clone for crate::tensor::real::Const<tch::Tensor> {
    fn clone(&self) -> Self {
        crate::tensor::real::Const::with_degree(self.tensor().shallow_clone(), self.degree())
    }
}

/// The pointwise product `x ↦ C ⊙ x` with a constant real matrix `C`: a
/// `torch.Tensor` of floating-point elements.
#[pyclass(frozen, name = "Scale", module = "real")]
pub struct Scale(tch::Tensor, u8);

// SAFETY: the tensor is private to this module, which only reads its
// shape and element kind and takes shallow clones of it; libtorch makes
// these thread-safe, counting the references to a tensor atomically.
unsafe impl Sync for Scale {}

#[pymethods]
impl Scale {
    /// The product of real matrices of the given `degree`; fails when
    /// `tensor` is not a `torch.Tensor` of two dimensions and floating-point
    /// elements.
    #[new]
    #[pyo3(signature = (tensor, /, degree = 0))]
    fn new(tensor: PyTensor, degree: u8) -> Result<Self, crate::Error> {
        super::check_matrix(&tensor)?;
        if !tensor.is_floating_point() {
            return Err(crate::Error(format!(
                "expected a tensor of floating-point elements, got {:?}",
                tensor.kind()
            )));
        }
        Ok(Scale(tensor.0, degree))
    }

    #[getter]
    fn tensor(&self) -> PyTensor {
        PyTensor(self.0.shallow_clone())
    }

    #[getter]
    fn degree(&self) -> u8 {
        self.1
    }

    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("tensor",)
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for crate::tensor::real::Scale<tch::Tensor> {
    type Error = PyErr;

    /// Fails when `ob` is not a [`Scale`].
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let scale = ob.cast::<Scale>()?.get();
        Ok(crate::tensor::real::Scale::with_degree(
            scale.0.shallow_clone(),
            scale.1,
        ))
    }
}

/// A clone shares the tensor of the original: `tch::Tensor` is not `Clone`,
/// so this takes a shallow clone of it, as libtorch counts its references.
impl Clone for crate::tensor::real::Scale<tch::Tensor> {
    fn clone(&self) -> Self {
        crate::tensor::real::Scale::with_degree(self.tensor().shallow_clone(), self.degree())
    }
}

/// A clone shares the tensor of the original: `tch::Tensor` is not `Clone`,
/// so this takes a shallow clone of it, as libtorch counts its references.
impl Clone for crate::tensor::real::Linear<tch::Tensor> {
    fn clone(&self) -> Self {
        crate::tensor::real::Linear::with_degree(
            self.a().shallow_clone(),
            self.b().shallow_clone(),
            self.degree(),
        )
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
            NRA::Const(op) => Const(op.tensor().shallow_clone(), op.degree())
                .into_pyobject(py)
                .map(Bound::into_any),
            NRA::Scale(op) => Scale(op.tensor().shallow_clone(), op.degree())
                .into_pyobject(py)
                .map(Bound::into_any),
            NRA::Add(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Sub(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Mul(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Div(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Linear(op) => Linear(op.a().shallow_clone(), op.b().shallow_clone(), op.degree())
                .into_pyobject(py)
                .map(Bound::into_any),
            NRA::ReLU(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Abs(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Neg(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Argmax(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Min(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Max(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Transpose(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Get(op) => op.into_pyobject(py).map(Bound::into_any),
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
