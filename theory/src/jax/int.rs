//! The Python classes of the operations of integer arithmetic carrying
//! tensors.

use super::Array;
use pyo3::prelude::*;

/// The affine map `x ↦ A·x + B` of integer matrices: JAX arrays of
/// integer elements.
#[pyclass(frozen, name = "Linear", module = "int")]
pub struct Linear(Array, Array);

#[pymethods]
impl Linear {
    /// Fails when `a` and `b` are not JAX arrays of integer
    /// elements, of shapes `[m, n]` and `[m, k]`.
    #[new]
    fn new(a: Array, b: Array) -> Result<Self, crate::Error> {
        super::check_elements([&a, &b], Array::is_integer, "integer")?;
        super::check_affine(&a, &b)?;
        Ok(Linear(a, b))
    }

    #[getter]
    fn a(&self) -> Array {
        self.0.clone()
    }

    #[getter]
    fn b(&self) -> Array {
        self.1.clone()
    }

    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("a", "b")
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for crate::tensor::int::Linear<Array> {
    type Error = PyErr;

    /// Fails when `ob` is not an [`Linear`].
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let linear = ob.cast::<Linear>()?.get();
        Ok(crate::tensor::int::Linear::new(
            linear.0.clone(),
            linear.1.clone(),
        )?)
    }
}

/// A constant integer matrix: a JAX array of integer elements.
#[pyclass(frozen, name = "Const", module = "int")]
pub struct Const(Array);

#[pymethods]
impl Const {
    /// Fails when `tensor` is not a JAX array of integer elements.
    #[new]
    fn new(tensor: Array) -> Result<Self, crate::Error> {
        super::check_elements([&tensor], Array::is_integer, "integer")?;
        Ok(Const(tensor))
    }

    #[getter]
    fn tensor(&self) -> Array {
        self.0.clone()
    }

    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("tensor",)
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for crate::tensor::int::Const<Array> {
    type Error = PyErr;

    /// Fails when `ob` is not an [`Const`].
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let constant = ob.cast::<Const>()?.get();
        Ok(crate::tensor::int::Const::new(constant.0.clone()))
    }
}

/// The pointwise product `x ↦ C ⊙ x` with a constant integer matrix `C`: a
/// JAX array of integer elements.
#[pyclass(frozen, name = "Scale", module = "int")]
pub struct Scale(Array);

#[pymethods]
impl Scale {
    /// Fails when `tensor` is not a JAX array of integer elements.
    #[new]
    fn new(tensor: Array) -> Result<Self, crate::Error> {
        super::check_elements([&tensor], Array::is_integer, "integer")?;
        Ok(Scale(tensor))
    }

    #[getter]
    fn tensor(&self) -> Array {
        self.0.clone()
    }

    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("tensor",)
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for crate::tensor::int::Scale<Array> {
    type Error = PyErr;

    /// Fails when `ob` is not a [`Scale`].
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let scale = ob.cast::<Scale>()?.get();
        Ok(crate::tensor::int::Scale::new(scale.0.clone()))
    }
}

/// A clone shares the array of the original, as a clone of an `Array`
/// does; the operations derive no `Clone`, which would require `T: Clone`.
impl Clone for crate::tensor::int::Scale<Array> {
    fn clone(&self) -> Self {
        crate::tensor::int::Scale::new(self.tensor().clone())
    }
}

/// A clone shares the array of the original, as a clone of an `Array`
/// does; the operations derive no `Clone`, which would require `T: Clone`.
impl Clone for crate::tensor::int::Const<Array> {
    fn clone(&self) -> Self {
        crate::tensor::int::Const::new(self.tensor().clone())
    }
}

/// A clone shares the array of the original, as a clone of an `Array`
/// does; the operations derive no `Clone`, which would require `T: Clone`.
impl Clone for crate::tensor::int::Linear<Array> {
    fn clone(&self) -> Self {
        crate::tensor::int::Linear::new(self.a().clone(), self.b().clone())
            .expect("a clone has the shapes of the original")
    }
}

/// Converts an operation of the theory into its Python class: the
/// operations carrying tensors into the classes of this module, holding the
/// same tensors, and the others into their own classes.
impl<'py> IntoPyObject<'py> for crate::tensor::int::NIA<Array> {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        use crate::tensor::int::NIA;

        match self {
            NIA::Const(op) => Const(op.tensor().clone())
                .into_pyobject(py)
                .map(Bound::into_any),
            NIA::Scale(op) => Scale(op.tensor().clone())
                .into_pyobject(py)
                .map(Bound::into_any),
            NIA::Add(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Sub(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Mul(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Div(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Mod(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Linear(op) => Linear(op.a().clone(), op.b().clone())
                .into_pyobject(py)
                .map(Bound::into_any),
            NIA::ReLU(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Neg(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Abs(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Argmax(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Min(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Max(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Transpose(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Get(op) => op.into_pyobject(py).map(Bound::into_any),
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
