//! The Python class of the constants of fixed-size bit-vectors.

use super::Tensor;
use pyo3::prelude::*;

/// A constant matrix of bit-vectors of a width: a tensor of
/// integer elements, each holding a bit-vector in its low bits.
#[pyclass(frozen, name = "Const", module = "word")]
pub struct Const(Tensor, usize);

// SAFETY: the tensor is private to this module, which only reads its
// shape and element kind and clones it; libtorch makes these thread-safe
// for a torch tensor, counting its references atomically, and a JAX array
// is immutable, behind an `Arc`.
unsafe impl Sync for Const {}

#[pymethods]
impl Const {
    /// Fails when `tensor` is not a tensor of integer elements, or `width`
    /// is zero.
    #[new]
    fn new(tensor: Tensor, width: usize) -> Result<Self, crate::Error> {
        crate::tensor::sort::check_width(width)?;
        super::check_elements([&tensor], Tensor::is_integer, "integer")?;
        Ok(Const(tensor, width))
    }

    #[getter]
    fn tensor(&self) -> Tensor {
        self.0.clone()
    }

    #[getter]
    fn width(&self) -> usize {
        self.1
    }

    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("tensor", "width")
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for crate::tensor::word::Const<Tensor> {
    type Error = PyErr;

    /// Fails when `ob` is not a [`Const`].
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let constant = ob.cast::<Const>()?.get();
        Ok(crate::tensor::word::Const::new(
            constant.0.clone(),
            constant.1,
        )?)
    }
}

/// A clone shares the tensor of the original, as a clone of a `Tensor`
/// does; the operations derive no `Clone`, which would require `T: Clone`.
impl Clone for crate::tensor::word::Const<Tensor> {
    fn clone(&self) -> Self {
        crate::tensor::word::Const::new(self.tensor().clone(), self.width())
            .expect("a clone has the width of the original")
    }
}

/// Converts an operation of the theory into its Python class: the
/// operations carrying tensors into the classes of this module, holding the
/// same tensors, and the others into their own classes.
impl<'py> IntoPyObject<'py> for crate::tensor::word::BV<Tensor> {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        use crate::tensor::word::BV;

        match self {
            BV::Const(op) => Const(op.tensor().clone(), op.width())
                .into_pyobject(py)
                .map(Bound::into_any),
            BV::Concat(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::Extract(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::Repeat(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::Not(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::And(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::Or(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::Xor(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::Nand(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::Nor(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::Xnor(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::Neg(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::Get(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::Ite(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::Add(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::Sub(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::Mul(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::Shl(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::RotateLeft(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::RotateRight(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::Eq(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::Ne(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::SDiv(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::Abs(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::SRem(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::SMod(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::AShr(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::SignExtend(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::SLt(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::SLe(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::SGt(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::SGe(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::UDiv(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::URem(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::LShr(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::ZeroExtend(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::ULt(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::ULe(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::UGt(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::UGe(op) => op.into_pyobject(py).map(Bound::into_any),
        }
    }
}
