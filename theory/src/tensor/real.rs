/*!
# Real arithmetic

Defines one type per operation on the sort [`Real`] of real matrices. Every operation reads and writes real matrices, and carries enough
information to compute its [`Boundary`]:

- [`Const`] — a constant real matrix literal.
- [`Add`], [`Sub`] — pointwise addition and subtraction.
- [`Mul`] — pointwise multiplication `x ⊙ y`.
- [`Div`] — pointwise division `x / y`.
- [`Linear`]`(A, B)` — the affine map `x ↦ A·x + B`, with `A` and `B`
  constant real matrices of compatible shapes.
- [`ReLU`] — the shape-preserving rectified-linear map `x ↦ max(x, 0)`.
- [`Min`], [`Max`] — pointwise minimum and maximum.
- [`Transpose`] — matrix transposition.
- [`MatMul`] — the matrix product `x·y`.
- [`Argmax`] — the index of a maximal entry, counted row by row, as a
  `[1, 1]` matrix.
- [`Ite`] — the pointwise choice `if c then x else y`, by a boolean matrix `c`.
- [`Lt`], [`Le`], [`Gt`], [`Ge`], [`Eq`](struct@Eq), [`Ne`] — pointwise
  comparisons, which write boolean matrices of the same shape.

The comparisons are those of SMT-LIB2's `Reals` theory, `<`, `<=`, `>` and
`>=`, together with `=` and `distinct` of the Core theory, lifted pointwise
to matrices. As they write booleans, their boundary is in [`Sort`], the
sorts of the theory, rather than in [`Real`].
*/

use super::sort::{Bool, Real};
use crate::Tensor;
use base::boundary::{Boundary, Poly};
use base::mono;
use derive_more::From;
use pyo3::prelude::*;
use std::fmt;
use subenum::subenum;

/// The sorts of the theory, as in the signature of SMT-LIB2's `Reals`:
/// real matrices, and the booleans of the Core theory, which comparisons
/// write, together with [`Zero`](crate::tensor::sort::Zero). It is the subset
/// [`tensor::sort::Reals`](crate::tensor::sort::Reals) of the sorts of all theories.
pub use super::sort::Reals as Sort;

/// A constant real matrix literal, held in a tensor of type `T`.
#[derive(Debug)]
pub struct Const<T>(pub T);

/// Pointwise addition `x + y` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[pyo3::pyclass(frozen, eq, str)]
pub struct Add(pub [usize; 2]);

/// Pointwise subtraction `x - y` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[pyo3::pyclass(frozen, eq, str)]
pub struct Sub(pub [usize; 2]);

/// Pointwise multiplication `x ⊙ y` of `[m, n]` matrices: the Hadamard
/// product, not the matrix product [`MatMul`].
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[pyo3::pyclass(frozen, eq, str)]
pub struct Mul(pub [usize; 2]);

/// Pointwise division `x / y` of `[m, n]` matrices: the Hadamard division
/// `x ⊘ y`. As in SMT-LIB2, `x / 0` is unspecified.
///
/// SMT-LIB2: `(/ x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[pyo3::pyclass(frozen, eq, str)]
pub struct Div(pub [usize; 2]);

/// The affine map `x ↦ A·x + B`, where `A` and `B` are constants.
#[derive(Debug)]
pub struct Linear<T>(pub T, pub T);

/// The rectified-linear map `x ↦ max(x, 0)`, applied pointwise to
/// `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[pyo3::pyclass(frozen, eq, str)]
pub struct ReLU(pub [usize; 2]);

/// The index of a maximal entry of an `[m, n]` real matrix, counting its
/// entries row by row, written as a `[1, 1]` matrix of the same sort.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[pyo3::pyclass(frozen, eq, str)]
pub struct Argmax(pub [usize; 2]);

/// Pointwise minimum `min(x, y)` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[pyo3::pyclass(frozen, eq, str)]
pub struct Min(pub [usize; 2]);

/// Pointwise maximum `max(x, y)` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[pyo3::pyclass(frozen, eq, str)]
pub struct Max(pub [usize; 2]);

/// Transposition `xᵀ` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[pyo3::pyclass(frozen, eq, str)]
pub struct Transpose(pub [usize; 2]);

/// The matrix product `x·y` of an `[m, n]` matrix `x` and an `[n, k]`
/// matrix `y`, with the sizes given as `MatMul([m, n], [n, k])`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[pyo3::pyclass(frozen, eq, str)]
pub struct MatMul(pub [usize; 2], pub [usize; 2]);

/// Pointwise comparison `x < y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(< x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[pyo3::pyclass(frozen, eq, str)]
pub struct Lt(pub [usize; 2]);

/// Pointwise choice `if c then x else y` between `[m, n]` real matrices `x` and `y`,
/// by the boolean matrix `c`, read first.
///
/// SMT-LIB2: `(ite c x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[pyo3::pyclass(frozen, eq, str)]
pub struct Ite(pub [usize; 2]);

/// Pointwise comparison `x ≤ y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(<= x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[pyo3::pyclass(frozen, eq, str)]
pub struct Le(pub [usize; 2]);

/// Pointwise comparison `x > y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(> x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[pyo3::pyclass(frozen, eq, str)]
pub struct Gt(pub [usize; 2]);

/// Pointwise comparison `x ≥ y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(>= x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[pyo3::pyclass(frozen, eq, str)]
pub struct Ge(pub [usize; 2]);

/// Pointwise comparison `x = y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(= x y)`, of the Core theory.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[pyo3::pyclass(frozen, eq, str)]
pub struct Eq(pub [usize; 2]);

/// Pointwise comparison `x ≠ y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(distinct x y)`, of the Core theory.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[pyo3::pyclass(frozen, eq, str)]
pub struct Ne(pub [usize; 2]);

/// The operations of the theory of real arithmetic, SMT-LIB2's `Reals`,
/// with their boundary in [`Sort`].
///
/// `subenum` defines [`LRA`], the linear ones: all but `Mul`, `Div` and
/// `MatMul`, which multiply or divide two variables. It also generates the
/// casts between them: `From` up to `NRA`, and `TryFrom` down from it.
#[subenum(LRA)]
#[derive(Debug, From)]
pub enum NRA<T> {
    #[subenum(LRA)]
    Const(Const<T>),
    #[subenum(LRA)]
    Add(Add),
    #[subenum(LRA)]
    Sub(Sub),
    Mul(Mul),
    Div(Div),
    #[subenum(LRA)]
    Linear(Linear<T>),
    #[subenum(LRA)]
    ReLU(ReLU),
    #[subenum(LRA)]
    Argmax(Argmax),
    #[subenum(LRA)]
    Min(Min),
    #[subenum(LRA)]
    Max(Max),
    #[subenum(LRA)]
    Transpose(Transpose),
    MatMul(MatMul),
    #[subenum(LRA)]
    Lt(Lt),
    #[subenum(LRA)]
    Ite(Ite),
    #[subenum(LRA)]
    Le(Le),
    #[subenum(LRA)]
    Gt(Gt),
    #[subenum(LRA)]
    Ge(Ge),
    #[subenum(LRA)]
    Eq(Eq),
    #[subenum(LRA)]
    Ne(Ne),
}

impl<T> Clone for NRA<T>
where
    Const<T>: Clone,
    Linear<T>: Clone,
{
    fn clone(&self) -> Self {
        match self {
            NRA::Const(op) => NRA::Const(op.clone()),
            NRA::Add(op) => NRA::Add(*op),
            NRA::Sub(op) => NRA::Sub(*op),
            NRA::Mul(op) => NRA::Mul(*op),
            NRA::Div(op) => NRA::Div(*op),
            NRA::Linear(op) => NRA::Linear(op.clone()),
            NRA::ReLU(op) => NRA::ReLU(*op),
            NRA::Argmax(op) => NRA::Argmax(*op),
            NRA::Min(op) => NRA::Min(*op),
            NRA::Max(op) => NRA::Max(*op),
            NRA::Transpose(op) => NRA::Transpose(*op),
            NRA::MatMul(op) => NRA::MatMul(*op),
            NRA::Lt(op) => NRA::Lt(*op),
            NRA::Ite(op) => NRA::Ite(*op),
            NRA::Le(op) => NRA::Le(*op),
            NRA::Gt(op) => NRA::Gt(*op),
            NRA::Ge(op) => NRA::Ge(*op),
            NRA::Eq(op) => NRA::Eq(*op),
            NRA::Ne(op) => NRA::Ne(*op),
        }
    }
}

impl<T: fmt::Display> fmt::Display for NRA<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NRA::Const(op) => op.fmt(f),
            NRA::Add(op) => op.fmt(f),
            NRA::Sub(op) => op.fmt(f),
            NRA::Mul(op) => op.fmt(f),
            NRA::Div(op) => op.fmt(f),
            NRA::Linear(op) => op.fmt(f),
            NRA::ReLU(op) => op.fmt(f),
            NRA::Argmax(op) => op.fmt(f),
            NRA::Min(op) => op.fmt(f),
            NRA::Max(op) => op.fmt(f),
            NRA::Transpose(op) => op.fmt(f),
            NRA::MatMul(op) => op.fmt(f),
            NRA::Lt(op) => op.fmt(f),
            NRA::Ite(op) => op.fmt(f),
            NRA::Le(op) => op.fmt(f),
            NRA::Gt(op) => op.fmt(f),
            NRA::Ge(op) => op.fmt(f),
            NRA::Eq(op) => op.fmt(f),
            NRA::Ne(op) => op.fmt(f),
        }
    }
}

impl<T> Clone for LRA<T>
where
    Const<T>: Clone,
    Linear<T>: Clone,
{
    fn clone(&self) -> Self {
        match self {
            LRA::Const(op) => LRA::Const(op.clone()),
            LRA::Add(op) => LRA::Add(*op),
            LRA::Sub(op) => LRA::Sub(*op),
            LRA::Linear(op) => LRA::Linear(op.clone()),
            LRA::ReLU(op) => LRA::ReLU(*op),
            LRA::Argmax(op) => LRA::Argmax(*op),
            LRA::Min(op) => LRA::Min(*op),
            LRA::Max(op) => LRA::Max(*op),
            LRA::Transpose(op) => LRA::Transpose(*op),
            LRA::Lt(op) => LRA::Lt(*op),
            LRA::Ite(op) => LRA::Ite(*op),
            LRA::Le(op) => LRA::Le(*op),
            LRA::Gt(op) => LRA::Gt(*op),
            LRA::Ge(op) => LRA::Ge(*op),
            LRA::Eq(op) => LRA::Eq(*op),
            LRA::Ne(op) => LRA::Ne(*op),
        }
    }
}

impl<T: fmt::Display> fmt::Display for LRA<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LRA::Const(op) => op.fmt(f),
            LRA::Add(op) => op.fmt(f),
            LRA::Sub(op) => op.fmt(f),
            LRA::Linear(op) => op.fmt(f),
            LRA::ReLU(op) => op.fmt(f),
            LRA::Argmax(op) => op.fmt(f),
            LRA::Min(op) => op.fmt(f),
            LRA::Max(op) => op.fmt(f),
            LRA::Transpose(op) => op.fmt(f),
            LRA::Lt(op) => op.fmt(f),
            LRA::Ite(op) => op.fmt(f),
            LRA::Le(op) => op.fmt(f),
            LRA::Gt(op) => op.fmt(f),
            LRA::Ge(op) => op.fmt(f),
            LRA::Eq(op) => op.fmt(f),
            LRA::Ne(op) => op.fmt(f),
        }
    }
}

impl<T: Tensor> Boundary<Sort> for NRA<T> {
    fn dom(&self) -> Poly<Sort> {
        match self {
            NRA::Const(op) => op.dom().map(Sort::from),
            NRA::Add(op) => op.dom().map(Sort::from),
            NRA::Sub(op) => op.dom().map(Sort::from),
            NRA::Mul(op) => op.dom().map(Sort::from),
            NRA::Div(op) => op.dom().map(Sort::from),
            NRA::Linear(op) => op.dom().map(Sort::from),
            NRA::ReLU(op) => op.dom().map(Sort::from),
            NRA::Argmax(op) => op.dom().map(Sort::from),
            NRA::Min(op) => op.dom().map(Sort::from),
            NRA::Max(op) => op.dom().map(Sort::from),
            NRA::Transpose(op) => op.dom().map(Sort::from),
            NRA::MatMul(op) => op.dom().map(Sort::from),
            NRA::Lt(op) => op.dom(),
            NRA::Ite(op) => op.dom(),
            NRA::Le(op) => op.dom(),
            NRA::Gt(op) => op.dom(),
            NRA::Ge(op) => op.dom(),
            NRA::Eq(op) => op.dom(),
            NRA::Ne(op) => op.dom(),
        }
    }

    fn cod(&self) -> Poly<Sort> {
        match self {
            NRA::Const(op) => op.cod().map(Sort::from),
            NRA::Add(op) => op.cod().map(Sort::from),
            NRA::Sub(op) => op.cod().map(Sort::from),
            NRA::Mul(op) => op.cod().map(Sort::from),
            NRA::Div(op) => op.cod().map(Sort::from),
            NRA::Linear(op) => op.cod().map(Sort::from),
            NRA::ReLU(op) => op.cod().map(Sort::from),
            NRA::Argmax(op) => op.cod().map(Sort::from),
            NRA::Min(op) => op.cod().map(Sort::from),
            NRA::Max(op) => op.cod().map(Sort::from),
            NRA::Transpose(op) => op.cod().map(Sort::from),
            NRA::MatMul(op) => op.cod().map(Sort::from),
            NRA::Lt(op) => op.cod(),
            NRA::Ite(op) => op.cod(),
            NRA::Le(op) => op.cod(),
            NRA::Gt(op) => op.cod(),
            NRA::Ge(op) => op.cod(),
            NRA::Eq(op) => op.cod(),
            NRA::Ne(op) => op.cod(),
        }
    }
}
impl<T: Tensor> Boundary<Sort> for LRA<T> {
    fn dom(&self) -> Poly<Sort> {
        match self {
            LRA::Const(op) => op.dom().map(Sort::from),
            LRA::Add(op) => op.dom().map(Sort::from),
            LRA::Sub(op) => op.dom().map(Sort::from),
            LRA::Linear(op) => op.dom().map(Sort::from),
            LRA::ReLU(op) => op.dom().map(Sort::from),
            LRA::Argmax(op) => op.dom().map(Sort::from),
            LRA::Min(op) => op.dom().map(Sort::from),
            LRA::Max(op) => op.dom().map(Sort::from),
            LRA::Transpose(op) => op.dom().map(Sort::from),
            LRA::Lt(op) => op.dom(),
            LRA::Ite(op) => op.dom(),
            LRA::Le(op) => op.dom(),
            LRA::Gt(op) => op.dom(),
            LRA::Ge(op) => op.dom(),
            LRA::Eq(op) => op.dom(),
            LRA::Ne(op) => op.dom(),
        }
    }

    fn cod(&self) -> Poly<Sort> {
        match self {
            LRA::Const(op) => op.cod().map(Sort::from),
            LRA::Add(op) => op.cod().map(Sort::from),
            LRA::Sub(op) => op.cod().map(Sort::from),
            LRA::Linear(op) => op.cod().map(Sort::from),
            LRA::ReLU(op) => op.cod().map(Sort::from),
            LRA::Argmax(op) => op.cod().map(Sort::from),
            LRA::Min(op) => op.cod().map(Sort::from),
            LRA::Max(op) => op.cod().map(Sort::from),
            LRA::Transpose(op) => op.cod().map(Sort::from),
            LRA::Lt(op) => op.cod(),
            LRA::Ite(op) => op.cod(),
            LRA::Le(op) => op.cod(),
            LRA::Gt(op) => op.cod(),
            LRA::Ge(op) => op.cod(),
            LRA::Eq(op) => op.cod(),
            LRA::Ne(op) => op.cod(),
        }
    }
}

impl<T: Tensor> Boundary<Real> for Const<T> {
    fn dom(&self) -> Poly<Real> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new(self.0.shape())].into()
    }
}

impl Boundary<Real> for Add {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new(self.0); 2].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new(self.0)].into()
    }
}

impl Boundary<Real> for Sub {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new(self.0); 2].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new(self.0)].into()
    }
}

impl Boundary<Real> for Mul {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new(self.0); 2].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new(self.0)].into()
    }
}

impl Boundary<Real> for Div {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new(self.0); 2].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new(self.0)].into()
    }
}

/// With `A` of shape `[m, n]` and `B` of shape `[m, k]`, the map takes `x`
/// of shape `[n, k]` to a matrix of shape `[m, k]`.
impl<T: Tensor> Boundary<Real> for Linear<T> {
    fn dom(&self) -> Poly<Real> {
        let [_, n] = self.0.shape();
        let [_, k] = self.1.shape();
        mono![Real::new([n, k])].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new(self.1.shape())].into()
    }
}

impl Boundary<Real> for ReLU {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new(self.0)].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new(self.0)].into()
    }
}

impl Boundary<Real> for Min {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new(self.0); 2].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new(self.0)].into()
    }
}

impl Boundary<Real> for Max {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new(self.0); 2].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new(self.0)].into()
    }
}

/// Takes a matrix of shape `[m, n]` to one of shape `[n, m]`.
impl Boundary<Real> for Transpose {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new(self.0)].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new([self.0[1], self.0[0]])].into()
    }
}

/// Takes matrices of shapes `[m, n]` and `[n, k]` to one of shape `[m, k]`.
impl Boundary<Real> for MatMul {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new(self.0), Real::new(self.1)].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new([self.0[0], self.1[1]])].into()
    }
}

impl Boundary<Sort> for Lt {
    fn dom(&self) -> Poly<Sort> {
        mono![Real::new(self.0).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new(self.0).into()].into()
    }
}

impl Boundary<Sort> for Le {
    fn dom(&self) -> Poly<Sort> {
        mono![Real::new(self.0).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new(self.0).into()].into()
    }
}

impl Boundary<Sort> for Gt {
    fn dom(&self) -> Poly<Sort> {
        mono![Real::new(self.0).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new(self.0).into()].into()
    }
}

impl Boundary<Sort> for Ge {
    fn dom(&self) -> Poly<Sort> {
        mono![Real::new(self.0).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new(self.0).into()].into()
    }
}

impl Boundary<Sort> for Eq {
    fn dom(&self) -> Poly<Sort> {
        mono![Real::new(self.0).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new(self.0).into()].into()
    }
}

impl Boundary<Sort> for Ne {
    fn dom(&self) -> Poly<Sort> {
        mono![Real::new(self.0).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new(self.0).into()].into()
    }
}

impl<T: fmt::Display> fmt::Display for Const<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<T: fmt::Display> fmt::Display for Linear<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Linear({}, {})", self.0, self.1)
    }
}

impl fmt::Display for Add {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Add")
    }
}

impl fmt::Display for Sub {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sub")
    }
}

impl fmt::Display for Mul {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Mul")
    }
}

impl fmt::Display for Div {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Div")
    }
}

impl fmt::Display for ReLU {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ReLU")
    }
}

impl Boundary<Real> for Argmax {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new(self.0)].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new([1, 1])].into()
    }
}

impl fmt::Display for Argmax {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Argmax")
    }
}

impl fmt::Display for Min {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Min")
    }
}

impl fmt::Display for Max {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Max")
    }
}

impl fmt::Display for Transpose {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Transpose")
    }
}

impl fmt::Display for MatMul {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MatMul")
    }
}

impl fmt::Display for Lt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Lt")
    }
}

impl Boundary<Sort> for Ite {
    fn dom(&self) -> Poly<Sort> {
        mono![
            Bool::new(self.0).into(),
            Real::new(self.0).into(),
            Real::new(self.0).into()
        ]
        .into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Real::new(self.0).into()].into()
    }
}

impl fmt::Display for Ite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ite")
    }
}

impl fmt::Display for Le {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Le")
    }
}

impl fmt::Display for Gt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Gt")
    }
}

impl fmt::Display for Ge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ge")
    }
}

impl fmt::Display for Eq {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Eq")
    }
}

impl fmt::Display for Ne {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ne")
    }
}

/// Extracts an operation of real arithmetic from a Python object, trying
/// each operation in turn: those carrying no data through their own
/// classes, and [`Const`] and [`Linear`] through the extraction that the
/// tensor type `T` provides.
impl<'py, T> pyo3::FromPyObject<'py> for NRA<T>
where
    Const<T>: pyo3::FromPyObject<'py>,
    Linear<T>: pyo3::FromPyObject<'py>,
{
    fn extract_bound(ob: &pyo3::Bound<'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use pyo3::types::PyAnyMethods;

        if let Ok(op) = ob.extract::<Add>() {
            return Ok(NRA::Add(op));
        }
        if let Ok(op) = ob.extract::<Sub>() {
            return Ok(NRA::Sub(op));
        }
        if let Ok(op) = ob.extract::<Mul>() {
            return Ok(NRA::Mul(op));
        }
        if let Ok(op) = ob.extract::<Div>() {
            return Ok(NRA::Div(op));
        }
        if let Ok(op) = ob.extract::<ReLU>() {
            return Ok(NRA::ReLU(op));
        }
        if let Ok(op) = ob.extract::<Argmax>() {
            return Ok(NRA::Argmax(op));
        }
        if let Ok(op) = ob.extract::<Min>() {
            return Ok(NRA::Min(op));
        }
        if let Ok(op) = ob.extract::<Max>() {
            return Ok(NRA::Max(op));
        }
        if let Ok(op) = ob.extract::<Transpose>() {
            return Ok(NRA::Transpose(op));
        }
        if let Ok(op) = ob.extract::<MatMul>() {
            return Ok(NRA::MatMul(op));
        }
        if let Ok(op) = ob.extract::<Lt>() {
            return Ok(NRA::Lt(op));
        }
        if let Ok(op) = ob.extract::<Ite>() {
            return Ok(NRA::Ite(op));
        }
        if let Ok(op) = ob.extract::<Le>() {
            return Ok(NRA::Le(op));
        }
        if let Ok(op) = ob.extract::<Gt>() {
            return Ok(NRA::Gt(op));
        }
        if let Ok(op) = ob.extract::<Ge>() {
            return Ok(NRA::Ge(op));
        }
        if let Ok(op) = ob.extract::<Eq>() {
            return Ok(NRA::Eq(op));
        }
        if let Ok(op) = ob.extract::<Ne>() {
            return Ok(NRA::Ne(op));
        }
        if let Ok(op) = ob.extract::<Const<T>>() {
            return Ok(NRA::Const(op));
        }
        if let Ok(op) = ob.extract::<Linear<T>>() {
            return Ok(NRA::Linear(op));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "expected an operation of real arithmetic, got {}",
            ob.get_type()
        )))
    }
}

// The Python constructors of the operations carrying no tensor: each takes
// its sizes positionally, in the order of its fields, the shape as a list.
#[pymethods]
impl Add {
    #[new]
    #[pyo3(signature = (shape, /))]
    fn new(shape: [usize; 2]) -> Self {
        Add(shape)
    }

    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[getter]
    fn shape(&self) -> [usize; 2] {
        self.0
    }
}

#[pymethods]
impl Sub {
    #[new]
    #[pyo3(signature = (shape, /))]
    fn new(shape: [usize; 2]) -> Self {
        Sub(shape)
    }

    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[getter]
    fn shape(&self) -> [usize; 2] {
        self.0
    }
}

#[pymethods]
impl Mul {
    #[new]
    #[pyo3(signature = (shape, /))]
    fn new(shape: [usize; 2]) -> Self {
        Mul(shape)
    }

    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[getter]
    fn shape(&self) -> [usize; 2] {
        self.0
    }
}

#[pymethods]
impl Div {
    #[new]
    #[pyo3(signature = (shape, /))]
    fn new(shape: [usize; 2]) -> Self {
        Div(shape)
    }

    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[getter]
    fn shape(&self) -> [usize; 2] {
        self.0
    }
}

#[pymethods]
impl ReLU {
    #[new]
    #[pyo3(signature = (shape, /))]
    fn new(shape: [usize; 2]) -> Self {
        ReLU(shape)
    }

    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[getter]
    fn shape(&self) -> [usize; 2] {
        self.0
    }
}

#[pymethods]
impl Argmax {
    #[new]
    #[pyo3(signature = (shape, /))]
    fn new(shape: [usize; 2]) -> Self {
        Argmax(shape)
    }

    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[getter]
    fn shape(&self) -> [usize; 2] {
        self.0
    }
}

#[pymethods]
impl Min {
    #[new]
    #[pyo3(signature = (shape, /))]
    fn new(shape: [usize; 2]) -> Self {
        Min(shape)
    }

    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[getter]
    fn shape(&self) -> [usize; 2] {
        self.0
    }
}

#[pymethods]
impl Max {
    #[new]
    #[pyo3(signature = (shape, /))]
    fn new(shape: [usize; 2]) -> Self {
        Max(shape)
    }

    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[getter]
    fn shape(&self) -> [usize; 2] {
        self.0
    }
}

#[pymethods]
impl Transpose {
    #[new]
    #[pyo3(signature = (shape, /))]
    fn new(shape: [usize; 2]) -> Self {
        Transpose(shape)
    }

    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[getter]
    fn shape(&self) -> [usize; 2] {
        self.0
    }
}

#[pymethods]
impl MatMul {
    /// Fails unless the columns of `lhs` are the rows of `rhs`.
    #[new]
    #[pyo3(signature = (lhs, rhs, /))]
    fn new(lhs: [usize; 2], rhs: [usize; 2]) -> Result<Self, crate::Error> {
        if lhs[1] != rhs[0] {
            return Err(crate::Error(format!(
                "cannot multiply a {lhs:?} matrix by a {rhs:?} matrix"
            )));
        }
        Ok(MatMul(lhs, rhs))
    }

    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("lhs", "rhs")
    }

    #[getter]
    fn lhs(&self) -> [usize; 2] {
        self.0
    }

    #[getter]
    fn rhs(&self) -> [usize; 2] {
        self.1
    }
}

#[pymethods]
impl Lt {
    #[new]
    #[pyo3(signature = (shape, /))]
    fn new(shape: [usize; 2]) -> Self {
        Lt(shape)
    }

    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[getter]
    fn shape(&self) -> [usize; 2] {
        self.0
    }
}

#[pymethods]
impl Ite {
    #[new]
    #[pyo3(signature = (shape, /))]
    fn new(shape: [usize; 2]) -> Self {
        Ite(shape)
    }

    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[getter]
    fn shape(&self) -> [usize; 2] {
        self.0
    }
}

#[pymethods]
impl Le {
    #[new]
    #[pyo3(signature = (shape, /))]
    fn new(shape: [usize; 2]) -> Self {
        Le(shape)
    }

    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[getter]
    fn shape(&self) -> [usize; 2] {
        self.0
    }
}

#[pymethods]
impl Gt {
    #[new]
    #[pyo3(signature = (shape, /))]
    fn new(shape: [usize; 2]) -> Self {
        Gt(shape)
    }

    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[getter]
    fn shape(&self) -> [usize; 2] {
        self.0
    }
}

#[pymethods]
impl Ge {
    #[new]
    #[pyo3(signature = (shape, /))]
    fn new(shape: [usize; 2]) -> Self {
        Ge(shape)
    }

    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[getter]
    fn shape(&self) -> [usize; 2] {
        self.0
    }
}

#[pymethods]
impl Eq {
    #[new]
    #[pyo3(signature = (shape, /))]
    fn new(shape: [usize; 2]) -> Self {
        Eq(shape)
    }

    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[getter]
    fn shape(&self) -> [usize; 2] {
        self.0
    }
}

#[pymethods]
impl Ne {
    #[new]
    #[pyo3(signature = (shape, /))]
    fn new(shape: [usize; 2]) -> Self {
        Ne(shape)
    }

    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[getter]
    fn shape(&self) -> [usize; 2] {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(s: Real) -> Poly<Real> {
        mono![s].into()
    }

    fn two(s: Real) -> Poly<Real> {
        mono![s, s].into()
    }

    #[test]
    fn pointwise_boundaries() {
        let r = Real::new([2, 3]);
        assert_eq!(Add([2, 3]).dom(), two(r));
        assert_eq!(Add([2, 3]).cod(), one(r));
        assert_eq!(Sub([2, 3]).dom(), two(r));
        assert_eq!(Mul([2, 3]).dom(), two(r));
        assert_eq!(Mul([2, 3]).cod(), one(r));
        assert_eq!(Div([2, 3]).dom(), two(r));
        assert_eq!(Div([2, 3]).cod(), one(r));
        assert_eq!(Min([2, 3]).dom(), two(r));
        assert_eq!(Max([2, 3]).cod(), one(r));
        assert_eq!(ReLU([2, 3]).dom(), one(r));
        assert_eq!(ReLU([2, 3]).cod(), one(r));
    }

    #[test]
    fn transpose_swaps_shape() {
        let t = Transpose([2, 3]);
        assert_eq!(t.dom(), one(Real::new([2, 3])));
        assert_eq!(t.cod(), one(Real::new([3, 2])));
    }

    #[test]
    fn matmul_contracts_inner_size() {
        let p = MatMul([2, 3], [3, 4]);
        let dom: Poly<Real> = mono![Real::new([2, 3]), Real::new([3, 4])].into();
        assert_eq!(p.dom(), dom);
        assert_eq!(p.cod(), one(Real::new([2, 4])));
    }

    #[test]
    fn ops_display_their_name() {
        assert_eq!(Add([1, 1]).to_string(), "Add");
        assert_eq!(Mul([1, 1]).to_string(), "Mul");
        assert_eq!(Div([1, 1]).to_string(), "Div");
        assert_eq!(ReLU([1, 1]).to_string(), "ReLU");
        assert_eq!(Transpose([1, 1]).to_string(), "Transpose");
    }
    #[test]
    fn comparisons_write_booleans() {
        let x: Sort = Real::new([2, 3]).into();
        let b: Sort = Bool::new([2, 3]).into();
        let dom: Poly<Sort> = mono![x, x].into();
        let cod: Poly<Sort> = mono![b].into();
        for (d, c) in [
            (Lt([2, 3]).dom(), Lt([2, 3]).cod()),
            (Le([2, 3]).dom(), Le([2, 3]).cod()),
            (Gt([2, 3]).dom(), Gt([2, 3]).cod()),
            (Ge([2, 3]).dom(), Ge([2, 3]).cod()),
            (Eq([2, 3]).dom(), Eq([2, 3]).cod()),
            (Ne([2, 3]).dom(), Ne([2, 3]).cod()),
        ] {
            assert_eq!(d, dom);
            assert_eq!(c, cod);
        }
        assert_eq!(Le([1, 1]).to_string(), "Le");
        assert_eq!(Ne([1, 1]).to_string(), "Ne");
    }
}
