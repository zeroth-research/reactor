//! The Python class of the constants of propositional logic.

use crate::tensor::sort::Bool;
use pyo3::prelude::*;
use pyo3_tch::PyTensor;

/// A constant boolean matrix: a `torch.Tensor` of boolean elements.
#[pyclass(frozen, name = "Const", module = "reactor.tensor.bool")]
pub struct Const(tch::Tensor);

// SAFETY: the tensor is private to this module, which only reads its
// shape and element kind and takes shallow clones of it; libtorch makes
// these thread-safe, counting the references to a tensor atomically.
unsafe impl Sync for Const {}

#[pymethods]
impl Const {
    /// The constant held in `tensor`, writing `cod`. Fails when `tensor` is
    /// not a `torch.Tensor` of two dimensions and boolean elements, or `cod`
    /// not of its shape.
    #[new]
    #[pyo3(signature = (tensor, /, cod))]
    fn new(tensor: PyTensor, cod: Bool) -> Result<Self, crate::Error> {
        super::check_matrix(&tensor)?;
        if tensor.kind() != tch::Kind::Bool {
            return Err(crate::Error(format!(
                "expected a tensor of boolean elements, got {:?}",
                tensor.kind()
            )));
        }
        let op = crate::tensor::bool::Const::with_cod(tensor.0, [cod])?;
        Ok(Const(op.tensor().shallow_clone()))
    }

    #[getter]
    fn tensor(&self) -> PyTensor {
        PyTensor(self.0.shallow_clone())
    }

    /// The tensors it carries, then its domain, when it reads anything, and
    /// its codomain.
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("tensor", "cod")
    }

    /// The sort it writes.
    #[getter]
    fn cod<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let op: crate::tensor::bool::Const<tch::Tensor> = slf.extract()?;
        crate::tensor::py_sorts(slf.py(), base::boundary::Boundary::cod(&op))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for crate::tensor::bool::Const<tch::Tensor> {
    type Error = PyErr;

    /// Fails when `ob` is not a [`Const`].
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let constant = ob.cast::<Const>()?.get();
        Ok(crate::tensor::bool::Const::new(constant.0.shallow_clone()))
    }
}

/// A clone shares the tensor of the original: `tch::Tensor` is not `Clone`,
/// so this takes a shallow clone of it, as libtorch counts its references.
impl Clone for crate::tensor::bool::Const<tch::Tensor> {
    fn clone(&self) -> Self {
        crate::tensor::bool::Const::new(self.tensor().shallow_clone())
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
            Prop::Const(op) => Const(op.tensor().shallow_clone())
                .into_pyobject(py)
                .map(Bound::into_any),
            Prop::And(op) => op.into_pyobject(py).map(Bound::into_any),
            Prop::Or(op) => op.into_pyobject(py).map(Bound::into_any),
            Prop::Xor(op) => op.into_pyobject(py).map(Bound::into_any),
            Prop::Not(op) => op.into_pyobject(py).map(Bound::into_any),
            Prop::Eq(op) => op.into_pyobject(py).map(Bound::into_any),
            Prop::Ne(op) => op.into_pyobject(py).map(Bound::into_any),
            Prop::Ite(op) => op.into_pyobject(py).map(Bound::into_any),
        }
    }
}
