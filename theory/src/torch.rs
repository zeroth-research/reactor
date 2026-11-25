/*!
# Torch

The torch backing of the operations of [`crate::matrix`]: [`tch::Tensor`]
implements [`crate::Matrix`], so that the operations carrying tensors hold
torch tensors directly, as `matrix::real::Const<tch::Tensor>`. The Python
`torch.Tensor` objects convert to and from them through
[`pyo3_tch::PyTensor`], sharing their storage.

The constants are Python classes of their own, one per theory, as a tensor
alone does not tell the theory of its constant: [`real::Const`],
[`int::Const`], [`bool::Const`] and [`word::Const`]. Each holds, in a
private field, a [`tch::Tensor`] of the element kind of its theory, checked
on construction, and [`word::Const`] its width too. A [`tch::Tensor`] is not
`Sync`, which a Python class must be, so each class implements `Sync`
itself, as only its own module touches the tensor. The constants of
[`crate::matrix`] extract from the class of their theory, sharing its tensor.
So do the pointwise products with a constant, [`real::Scale`] and
[`int::Scale`], whose class holds the constant in the same way.

The linear maps are Python classes too, [`real::Linear`] and
[`int::Linear`], each holding, in a private field, the [`tch::Tensor`] `A`
of its theory, together with the sort it reads, whose columns `A` does not
determine. So are the affine maps, [`real::Affine`] and
[`int::Affine`], each holding, in private fields, the [`tch::Tensor`]s `A`
and `B` of its theory, of shapes `[m, n]` and `[m, k]`, checked on
construction, and implementing `Sync` in the same way. The linear and
affine maps of [`crate::matrix::real`] and [`crate::matrix::int`] extract
from the class of their theory, sharing its tensors.
*/

pub mod bool;
pub mod int;
pub mod real;
pub mod word;

use pyo3::prelude::*;

impl crate::Matrix for tch::Tensor {
    /// The `[rows, cols]` shape of the tensor, which must be a matrix.
    fn shape(&self) -> [usize; 2] {
        let size = self.size();
        let [rows, cols] = size[..] else {
            panic!("expected a matrix, got a tensor of shape {size:?}")
        };
        [rows as usize, cols as usize]
    }
}

/// Whether `kind` is an integer element kind.
fn is_integer(kind: tch::Kind) -> bool {
    matches!(
        kind,
        tch::Kind::Uint8 | tch::Kind::Int8 | tch::Kind::Int16 | tch::Kind::Int | tch::Kind::Int64
    )
}

/// Fails unless `tensor` is a matrix, a tensor of two dimensions: the
/// constants are built from matrices only, so that [`crate::Matrix::shape`]
/// never panics on their tensors.
fn check_matrix(tensor: &tch::Tensor) -> Result<(), crate::Error> {
    if tensor.dim() != 2 {
        return Err(crate::Error(format!(
            "expected a matrix, got a tensor of shape {:?}",
            tensor.size()
        )));
    }
    Ok(())
}

/// The sort of a boundary holding exactly one: that of the operations
/// reading or writing a single matrix.
fn sole<S>(sorts: base::boundary::Poly<S>) -> S {
    sorts
        .into_iter()
        .flatten()
        .next()
        .expect("a boundary of one sort")
}

/// Fails unless `a` and `b` are the matrices of an affine map `x ↦ A·x + B`:
/// `A` of shape `[m, n]` and `B` of shape `[m, k]`.
fn check_affine(a: &tch::Tensor, b: &tch::Tensor) -> Result<(), crate::Error> {
    let (a, b) = (a.size(), b.size());
    if a.len() != 2 || b.len() != 2 {
        return Err(crate::Error(format!(
            "expected A of shape [m, n] and B of shape [m, k], got {a:?} and {b:?}"
        )));
    }
    let shape = |s: &[i64]| [s[0] as usize, s[1] as usize];
    crate::matrix::check_affine(shape(&a), shape(&b))
}

/// Converts an operation of any theory into its Python class, through the
/// conversion of its theory; the structural operations are classes of their
/// own.
impl<'py> IntoPyObject<'py> for crate::matrix::Any<tch::Tensor> {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        use crate::matrix::Any;

        match self {
            Any::Real(op) => op.into_pyobject(py),
            Any::Int(op) => op.into_pyobject(py),
            Any::Bool(op) => op.into_pyobject(py),
            Any::Word(op) => op.into_pyobject(py),
            Any::Havoc(op) => op.into_pyobject(py).map(Bound::into_any),
            Any::Skip(op) => op.into_pyobject(py).map(Bound::into_any),
            Any::Zero(op) => op.into_pyobject(py).map(Bound::into_any),
        }
    }
}
