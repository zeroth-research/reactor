//! The Python class of the constants of propositional logic.

use crate::bool::Bool;
use base::boundary::Mono;
use pyo3::prelude::*;
use pyo3_tch::PyTensor;

/// A constant boolean matrix: a `torch.Tensor` of boolean elements.
#[pyclass(frozen, name = "Truth", module = "reactor.bool")]
pub struct Truth(super::Tensor);

// SAFETY: the tensor is private to this module, which only reads its
// shape and element kind and takes shallow clones of it; libtorch makes
// these thread-safe, counting the references to a tensor atomically.
unsafe impl Sync for Truth {}

#[pymethods]
impl Truth {
    /// The constant held in `tensor`, writing `cod`, which it determines and may be
    /// omitted. Fails when `tensor` is not a `torch.Tensor` of two dimensions and
    /// boolean elements, or `cod` not of its shape.
    #[new]
    #[pyo3(signature = (tensor, /, cod = None))]
    fn new(tensor: PyTensor, cod: Option<Bool>) -> Result<Self, crate::Error> {
        let tensor = super::Tensor(tensor.0);
        super::check_matrix(&tensor)?;
        if tensor.kind() != tch::Kind::Bool {
            return Err(crate::Error(format!(
                "expected a tensor of boolean elements, got {:?}",
                tensor.kind()
            )));
        }
        let op = match cod {
            Some(cod) => crate::bool::Truth::with_cod(tensor, [cod])?,
            None => crate::bool::Truth::new(tensor)?,
        };
        Ok(Truth(op.tensor().clone()))
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

    /// The sorts it reads, as a tuple: none.
    #[getter]
    fn dom(slf: &Bound<'_, Self>) -> PyResult<Mono<Bool>> {
        let op: crate::bool::Truth<super::Tensor> = slf.extract()?;
        Ok(op.dom())
    }

    /// The sorts it writes, as a tuple.
    #[getter]
    fn cod(slf: &Bound<'_, Self>) -> PyResult<Mono<Bool>> {
        let op: crate::bool::Truth<super::Tensor> = slf.extract()?;
        Ok(op.cod())
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for crate::bool::Truth<super::Tensor> {
    type Error = PyErr;

    /// Fails when `ob` is not a [`Truth`].
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let constant = ob.cast::<Truth>()?.get();
        let shape = crate::Matrix::shape(&constant.0);
        Ok(crate::bool::Truth::with_cod(
            constant.0.clone(),
            [crate::bool::Bool::new_unchecked(shape)],
        )?)
    }
}

/// Converts an operation of the theory into its Python class: the
/// operations carrying tensors into the classes of this module, holding the
/// same tensors, and the others into their own classes.
impl<'py> IntoPyObject<'py> for crate::bool::Prop<super::Tensor> {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        use crate::bool::Prop;

        match self {
            Prop::Truth(op) => Truth(op.tensor().clone())
                .into_pyobject(py)
                .map(Bound::into_any),
            Prop::And(op) => op.into_pyobject(py).map(Bound::into_any),
            Prop::Or(op) => op.into_pyobject(py).map(Bound::into_any),
            Prop::Xor(op) => op.into_pyobject(py).map(Bound::into_any),
            Prop::Not(op) => op.into_pyobject(py).map(Bound::into_any),
            Prop::Eq(op) => op.into_pyobject(py).map(Bound::into_any),
            Prop::Ne(op) => op.into_pyobject(py).map(Bound::into_any),
            Prop::Ite(op) => op.into_pyobject(py).map(Bound::into_any),
            Prop::BroadcastIte(op) => op.into_pyobject(py).map(Bound::into_any),
            Prop::ReduceAnd(op) => op.into_pyobject(py).map(Bound::into_any),
            Prop::ReduceOr(op) => op.into_pyobject(py).map(Bound::into_any),
        }
    }
}
