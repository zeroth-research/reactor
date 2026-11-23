//! The Python class of the constants of propositional logic.

use super::Tensor;
use pyo3::prelude::*;

/// A constant boolean matrix: a tensor of boolean elements.
#[pyclass(frozen, name = "Const", module = "bool")]
pub struct Const(Tensor);

// SAFETY: the tensor is private to this module, which only reads its
// shape and element kind and clones it; libtorch makes these thread-safe
// for a torch tensor, counting its references atomically, and a JAX array
// is immutable, behind an `Arc`.
unsafe impl Sync for Const {}

#[pymethods]
impl Const {
    /// Fails when `tensor` is not a tensor of boolean elements.
    #[new]
    fn new(tensor: Tensor) -> Result<Self, crate::Error> {
        super::check_elements([&tensor], Tensor::is_bool, "boolean")?;
        Ok(Const(tensor))
    }

    #[getter]
    fn tensor(&self) -> Tensor {
        self.0.clone()
    }

    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("tensor",)
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for crate::tensor::bool::Const<Tensor> {
    type Error = PyErr;

    /// Fails when `ob` is not a [`Const`].
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let constant = ob.cast::<Const>()?.get();
        Ok(crate::tensor::bool::Const::new(constant.0.clone()))
    }
}

/// A clone shares the tensor of the original, as a clone of a `Tensor`
/// does; the operations derive no `Clone`, which would require `T: Clone`.
impl Clone for crate::tensor::bool::Const<Tensor> {
    fn clone(&self) -> Self {
        crate::tensor::bool::Const::new(self.tensor().clone())
    }
}

/// Converts an operation of the theory into its Python class: the
/// operations carrying tensors into the classes of this module, holding the
/// same tensors, and the others into their own classes.
impl<'py> IntoPyObject<'py> for crate::tensor::bool::Prop<Tensor> {
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
