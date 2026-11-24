//! The Python class of the constants of propositional logic.

use super::Array;
use pyo3::prelude::*;

/// A constant boolean matrix: a JAX array of boolean elements.
#[pyclass(frozen, name = "Const", module = "bool")]
pub struct Const(Array);

#[pymethods]
impl Const {
    /// Fails when `tensor` is not a JAX array of boolean elements.
    #[new]
    fn new(tensor: Array) -> Result<Self, crate::Error> {
        super::check_elements([&tensor], Array::is_bool, "boolean")?;
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

impl<'a, 'py> FromPyObject<'a, 'py> for crate::tensor::bool::Const<Array> {
    type Error = PyErr;

    /// Fails when `ob` is not a [`Const`].
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let constant = ob.cast::<Const>()?.get();
        Ok(crate::tensor::bool::Const::new(constant.0.clone()))
    }
}

/// A clone shares the array of the original, as a clone of an `Array`
/// does; the operations derive no `Clone`, which would require `T: Clone`.
impl Clone for crate::tensor::bool::Const<Array> {
    fn clone(&self) -> Self {
        crate::tensor::bool::Const::new(self.tensor().clone())
    }
}

/// Converts an operation of the theory into its Python class: the
/// operations carrying tensors into the classes of this module, holding the
/// same tensors, and the others into their own classes.
impl<'py> IntoPyObject<'py> for crate::tensor::bool::Prop<Array> {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        use crate::tensor::bool::Prop;

        match self {
            Prop::Const(op) => Const(op.tensor().clone())
                .into_pyobject(py)
                .map(Bound::into_any),
            Prop::And(op) => op.into_pyobject(py).map(Bound::into_any),
            Prop::Or(op) => op.into_pyobject(py).map(Bound::into_any),
            Prop::Xor(op) => op.into_pyobject(py).map(Bound::into_any),
            Prop::Not(op) => op.into_pyobject(py).map(Bound::into_any),
            Prop::Eq(op) => op.into_pyobject(py).map(Bound::into_any),
            Prop::Ne(op) => op.into_pyobject(py).map(Bound::into_any),
            Prop::All(op) => op.into_pyobject(py).map(Bound::into_any),
            Prop::Ite(op) => op.into_pyobject(py).map(Bound::into_any),
            Prop::Implies(op) => op.into_pyobject(py).map(Bound::into_any),
            Prop::Get(op) => op.into_pyobject(py).map(Bound::into_any),
        }
    }
}
