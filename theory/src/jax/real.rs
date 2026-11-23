//! The Python classes of the operations of real arithmetic carrying tensors.

use super::Array;
use pyo3::prelude::*;

/// The affine map `x ↦ A·x + B` of real matrices: JAX arrays of floating-point
/// elements.
#[pyclass(frozen, name = "Linear", module = "real")]
pub struct Linear(Array, Array, u8);

#[pymethods]
impl Linear {
    /// The map of real matrices of the given `degree`, which `B` is of;
    /// fails when `a` and `b` are not JAX arrays of floating-point elements,
    /// of shapes `[m, n]` and `[m, k]`.
    #[new]
    #[pyo3(signature = (a, b, /, degree = 0))]
    fn new(a: Array, b: Array, degree: u8) -> Result<Self, crate::Error> {
        super::check_elements([&a, &b], Array::is_floating_point, "floating-point")?;
        super::check_affine(&a, &b)?;
        Ok(Linear(a, b, degree))
    }

    #[getter]
    fn a(&self) -> Array {
        self.0.clone()
    }

    #[getter]
    fn b(&self) -> Array {
        self.1.clone()
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

impl<'a, 'py> FromPyObject<'a, 'py> for crate::tensor::real::Linear<Array> {
    type Error = PyErr;

    /// Fails when `ob` is not a [`Linear`].
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let linear = ob.cast::<Linear>()?.get();
        Ok(crate::tensor::real::Linear::with_degree(
            linear.0.clone(),
            linear.1.clone(),
            linear.2,
        )?)
    }
}

/// A constant real matrix: a JAX array of floating-point or integer
/// elements, kept as given, since integers are reals too.
#[pyclass(frozen, name = "Const", module = "real")]
pub struct Const(Array, u8);

#[pymethods]
impl Const {
    /// The constant of the given `degree`; fails when `tensor` is not a JAX
    /// array of floating-point or integer elements.
    #[new]
    #[pyo3(signature = (tensor, /, degree = 0))]
    fn new(tensor: Array, degree: u8) -> Result<Self, crate::Error> {
        super::check_elements(
            [&tensor],
            |a| a.is_floating_point() || a.is_integer(),
            "floating-point or integer",
        )?;
        Ok(Const(tensor, degree))
    }

    #[getter]
    fn tensor(&self) -> Array {
        self.0.clone()
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

impl<'a, 'py> FromPyObject<'a, 'py> for crate::tensor::real::Const<Array> {
    type Error = PyErr;

    /// Fails when `ob` is not a [`Const`].
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let constant = ob.cast::<Const>()?.get();
        Ok(crate::tensor::real::Const::with_degree(
            constant.0.clone(),
            constant.1,
        ))
    }
}

/// The pointwise product `x ↦ C ⊙ x` with a constant real matrix `C`: a JAX
/// array of floating-point elements.
#[pyclass(frozen, name = "Scale", module = "real")]
pub struct Scale(Array, u8);

#[pymethods]
impl Scale {
    /// The product of real matrices of the given `degree`; fails when
    /// `tensor` is not a JAX array of floating-point elements.
    #[new]
    #[pyo3(signature = (tensor, /, degree = 0))]
    fn new(tensor: Array, degree: u8) -> Result<Self, crate::Error> {
        super::check_elements([&tensor], Array::is_floating_point, "floating-point")?;
        Ok(Scale(tensor, degree))
    }

    #[getter]
    fn tensor(&self) -> Array {
        self.0.clone()
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

impl<'a, 'py> FromPyObject<'a, 'py> for crate::tensor::real::Scale<Array> {
    type Error = PyErr;

    /// Fails when `ob` is not a [`Scale`].
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let scale = ob.cast::<Scale>()?.get();
        Ok(crate::tensor::real::Scale::with_degree(
            scale.0.clone(),
            scale.1,
        ))
    }
}

// A clone shares the arrays of the original, as a clone of an `Array`
// does; the operations derive no `Clone`, which would require `T: Clone`.

impl Clone for crate::tensor::real::Const<Array> {
    fn clone(&self) -> Self {
        crate::tensor::real::Const::with_degree(self.tensor().clone(), self.degree())
    }
}

impl Clone for crate::tensor::real::Scale<Array> {
    fn clone(&self) -> Self {
        crate::tensor::real::Scale::with_degree(self.tensor().clone(), self.degree())
    }
}

impl Clone for crate::tensor::real::Linear<Array> {
    fn clone(&self) -> Self {
        crate::tensor::real::Linear::with_degree(self.a().clone(), self.b().clone(), self.degree())
            .expect("a clone has the shapes of the original")
    }
}

/// Converts an operation of the theory into its Python class: the
/// operations carrying tensors into the classes of this module, holding the
/// same tensors, and the others into their own classes.
impl<'py> IntoPyObject<'py> for crate::tensor::real::NRA<Array> {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        use crate::tensor::real::NRA;

        match self {
            NRA::Const(op) => Const(op.tensor().clone(), op.degree())
                .into_pyobject(py)
                .map(Bound::into_any),
            NRA::Scale(op) => Scale(op.tensor().clone(), op.degree())
                .into_pyobject(py)
                .map(Bound::into_any),
            NRA::Add(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Sub(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Mul(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Div(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Linear(op) => Linear(op.a().clone(), op.b().clone(), op.degree())
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
