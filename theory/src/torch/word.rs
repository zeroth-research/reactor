//! The Python class of the constants of fixed-size bit-vectors.

use crate::word::Word;
use pyo3::prelude::*;
use pyo3_tch::PyTensor;

/// A constant matrix of bit-vectors of a width: a `torch.Tensor` of
/// integer elements, each holding a bit-vector in its low bits.
#[pyclass(frozen, name = "Num", module = "reactor.word")]
pub struct Num(tch::Tensor, usize);

// SAFETY: the tensor is private to this module, which only reads its
// shape and element kind and takes shallow clones of it; libtorch makes
// these thread-safe, counting the references to a tensor atomically.
unsafe impl Sync for Num {}

#[pymethods]
impl Num {
    /// The constant held in `tensor`, writing `cod`. Fails when `tensor` is
    /// not a `torch.Tensor` of two dimensions and integer elements, or `cod`
    /// not of its shape.
    #[new]
    #[pyo3(signature = (tensor, /, cod))]
    fn new(tensor: PyTensor, cod: Word) -> Result<Self, crate::Error> {
        super::check_matrix(&tensor)?;
        if !super::is_integer(tensor.kind()) {
            return Err(crate::Error(format!(
                "expected a tensor of integer elements, got {:?}",
                tensor.kind()
            )));
        }
        let op = crate::word::Num::with_cod(tensor.0, [cod])?;
        Ok(Num(op.tensor().shallow_clone(), op.width()))
    }

    #[getter]
    fn tensor(&self) -> PyTensor {
        PyTensor(self.0.shallow_clone())
    }

    #[getter]
    fn width(&self) -> usize {
        self.1
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
        let op: crate::word::Num<tch::Tensor> = slf.extract()?;
        crate::py_sorts(slf.py(), base::boundary::Boundary::cod(&op))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for crate::word::Num<tch::Tensor> {
    type Error = PyErr;

    /// Fails when `ob` is not a [`Num`].
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let constant = ob.cast::<Num>()?.get();
        let cod = crate::word::Word::new(constant.1, crate::Matrix::shape(&constant.0))?;
        Ok(crate::word::Num::with_cod(
            constant.0.shallow_clone(),
            [cod],
        )?)
    }
}

/// A clone shares the tensor of the original: `tch::Tensor` is not `Clone`,
/// so this takes a shallow clone of it, as libtorch counts its references.
impl Clone for crate::word::Num<tch::Tensor> {
    fn clone(&self) -> Self {
        let cod =
            crate::word::Word::new_unchecked(self.width(), crate::Matrix::shape(self.tensor()));
        crate::word::Num::with_cod(self.tensor().shallow_clone(), [cod])
            .expect("a clone has the sort of the original")
    }
}

/// Converts an operation of the theory into its Python class: the
/// operations carrying tensors into the classes of this module, holding the
/// same tensors, and the others into their own classes.
impl<'py> IntoPyObject<'py> for crate::word::BV<tch::Tensor> {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        use crate::word::BV;

        match self {
            BV::Num(op) => Num(op.tensor().shallow_clone(), op.width())
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
            BV::Ite(op) => op.into_pyobject(py).map(Bound::into_any),
            BV::BroadcastIte(op) => op.into_pyobject(py).map(Bound::into_any),
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
