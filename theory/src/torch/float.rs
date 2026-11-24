//! The Python class of the constants of floating-point numbers.

use crate::tensor::sort::{Float, Floats};
use base::boundary::Boundary;
use pyo3::prelude::*;
use pyo3_tch::PyTensor;

/// The format `(exponent, significand)` of the floats of the element kind
/// `kind`, when torch has floats of that format.
fn format(kind: tch::Kind) -> Option<(usize, usize)> {
    match kind {
        tch::Kind::Half => Some((5, 11)),
        tch::Kind::BFloat16 => Some((8, 8)),
        tch::Kind::Float => Some((8, 24)),
        tch::Kind::Double => Some((11, 53)),
        _ => None,
    }
}

/// The floats that the constant `op` writes.
fn sort(op: &crate::tensor::float::Const<tch::Tensor>) -> Float {
    match super::sole(op.cod()) {
        Floats::Float(sort) => sort,
        sort => unreachable!("a float constant writes floats, not {sort}"),
    }
}

/// A constant matrix of floats: a `torch.Tensor` of floating-point elements,
/// of one of the formats torch has, `Float16`, `BFloat16`, `Float32` and
/// `Float64`, which its sort must be.
#[pyclass(frozen, name = "Const", module = "reactor.tensor.float")]
pub struct Const(tch::Tensor, Float);

// SAFETY: the tensor is private to this module, which only reads its
// shape and element kind and takes shallow clones of it; libtorch makes
// these thread-safe, counting the references to a tensor atomically.
unsafe impl Sync for Const {}

/// Fails unless the elements of `tensor` are floats of the format of `cod`.
fn check_format(tensor: &tch::Tensor, cod: Float) -> Result<(), crate::Error> {
    if format(tensor.kind()) != Some((cod.exponent(), cod.significand())) {
        return Err(crate::Error(format!(
            "expected a tensor of floats of the format of {cod}, got {:?}",
            tensor.kind()
        )));
    }
    Ok(())
}

#[pymethods]
impl Const {
    /// The constant held in `tensor`, writing `cod`, which its element kind and
    /// shape determine and may be omitted. Fails when `tensor` is not a
    /// `torch.Tensor` of two dimensions and floating-point elements of a format
    /// torch has, or `cod` not of its format and shape.
    #[new]
    #[pyo3(signature = (tensor, /, cod = None))]
    fn new(tensor: PyTensor, cod: Option<Float>) -> Result<Self, crate::Error> {
        super::check_matrix(&tensor)?;
        let cod = match cod {
            Some(cod) => cod,
            None => {
                let Some((exponent, significand)) = format(tensor.kind()) else {
                    return Err(crate::Error(format!(
                        "expected a tensor of floating-point elements, got {:?}",
                        tensor.kind()
                    )));
                };
                Float::new(exponent, significand, crate::Tensor::shape(&tensor.0))?
            }
        };
        check_format(&tensor, cod)?;
        let op = crate::tensor::float::Const::with_cod(tensor.0, [cod])?;
        Ok(Const(op.tensor().shallow_clone(), cod))
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
    fn cod(&self) -> Float {
        self.1
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for crate::tensor::float::Const<tch::Tensor> {
    type Error = PyErr;

    /// Fails when `ob` is not a [`Const`].
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let constant = ob.cast::<Const>()?.get();
        Ok(crate::tensor::float::Const::with_cod(
            constant.0.shallow_clone(),
            [constant.1],
        )?)
    }
}

/// A clone shares the tensor of the original: `tch::Tensor` is not `Clone`,
/// so this takes a shallow clone of it, as libtorch counts its references.
impl Clone for crate::tensor::float::Const<tch::Tensor> {
    fn clone(&self) -> Self {
        crate::tensor::float::Const::with_cod(self.tensor().shallow_clone(), [sort(self)])
            .expect("a clone has the sort of the original")
    }
}

/// Converts an operation of the theory into its Python class: the constants
/// into the class of this module, holding the same tensor, and the others
/// into their own classes.
impl<'py> IntoPyObject<'py> for crate::tensor::float::FP<tch::Tensor> {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        use crate::tensor::float::FP;

        match self {
            FP::Const(op) => Const(op.tensor().shallow_clone(), sort(&op))
                .into_pyobject(py)
                .map(Bound::into_any),
            FP::Ite(op) => op.into_pyobject(py).map(Bound::into_any),
            FP::Convert(op) => op.into_pyobject(py).map(Bound::into_any),
            FP::PlusInfinity(op) => op.into_pyobject(py).map(Bound::into_any),
            FP::MinusInfinity(op) => op.into_pyobject(py).map(Bound::into_any),
            FP::PlusZero(op) => op.into_pyobject(py).map(Bound::into_any),
            FP::MinusZero(op) => op.into_pyobject(py).map(Bound::into_any),
            FP::NaN(op) => op.into_pyobject(py).map(Bound::into_any),
            FP::Abs(op) => op.into_pyobject(py).map(Bound::into_any),
            FP::Neg(op) => op.into_pyobject(py).map(Bound::into_any),
            FP::Sqrt(op) => op.into_pyobject(py).map(Bound::into_any),
            FP::RoundToIntegral(op) => op.into_pyobject(py).map(Bound::into_any),
            FP::Add(op) => op.into_pyobject(py).map(Bound::into_any),
            FP::Sub(op) => op.into_pyobject(py).map(Bound::into_any),
            FP::Mul(op) => op.into_pyobject(py).map(Bound::into_any),
            FP::Div(op) => op.into_pyobject(py).map(Bound::into_any),
            FP::Fma(op) => op.into_pyobject(py).map(Bound::into_any),
            FP::Rem(op) => op.into_pyobject(py).map(Bound::into_any),
            FP::Min(op) => op.into_pyobject(py).map(Bound::into_any),
            FP::Max(op) => op.into_pyobject(py).map(Bound::into_any),
            FP::Le(op) => op.into_pyobject(py).map(Bound::into_any),
            FP::Lt(op) => op.into_pyobject(py).map(Bound::into_any),
            FP::Ge(op) => op.into_pyobject(py).map(Bound::into_any),
            FP::Gt(op) => op.into_pyobject(py).map(Bound::into_any),
            FP::Eq(op) => op.into_pyobject(py).map(Bound::into_any),
            FP::IsNormal(op) => op.into_pyobject(py).map(Bound::into_any),
            FP::IsSubnormal(op) => op.into_pyobject(py).map(Bound::into_any),
            FP::IsZero(op) => op.into_pyobject(py).map(Bound::into_any),
            FP::IsInfinite(op) => op.into_pyobject(py).map(Bound::into_any),
            FP::IsNaN(op) => op.into_pyobject(py).map(Bound::into_any),
            FP::IsNegative(op) => op.into_pyobject(py).map(Bound::into_any),
            FP::IsPositive(op) => op.into_pyobject(py).map(Bound::into_any),
        }
    }
}
