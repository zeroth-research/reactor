/*!
# Integer arithmetic

Defines one type per operation on the sort [`Int`] of integer matrices. Every
operation reads and writes integer matrices, and carries enough
information to compute its [`Boundary`]:

- [`Const`] — a constant integer matrix literal.
- [`Add`], [`Sub`] — pointwise addition and subtraction.
- [`Mul`] — pointwise multiplication `x ⊙ y`.
- [`Div`], [`Mod`] — pointwise Euclidean division and remainder, as
  SMT-LIB2's `div` and `mod`.
- [`Linear`]`(A, B)` — the affine map `x ↦ A·x + B`, with `A` and `B`
  constant integer matrices of compatible shapes.
- [`ReLU`] — the shape-preserving rectified-linear map `x ↦ max(x, 0)`.
- [`Min`], [`Max`] — pointwise minimum and maximum.
- [`Transpose`] — matrix transposition.
- [`MatMul`] — the matrix product `x·y`.
- [`Argmax`] — the index of a maximal entry, counted row by row, as a
  `[1, 1]` matrix.
- [`Ite`] — the pointwise choice `if c then x else y`, by a boolean matrix `c`.
- [`Lt`], [`Le`], [`Gt`], [`Ge`], [`Eq`](struct@Eq), [`Ne`] — pointwise
  comparisons, which write boolean matrices of the same shape.

The comparisons are those of SMT-LIB2's `Ints` theory, `<`, `<=`, `>` and
`>=`, together with `=` and `distinct` of the Core theory, lifted pointwise
to matrices. As they write booleans, their boundary is in [`Sort`], the
sorts of the theory, rather than in [`Int`].
*/

use super::sort::{Bool, Int};
use crate::Tensor;
use base::boundary::{Boundary, Poly};
use base::mono;
use derive_more::From;
use pyo3::prelude::*;
use std::fmt;
use subenum::subenum;

/// The sorts of the theory, as in the signature of SMT-LIB2's `Ints`:
/// int matrices, and the booleans of the Core theory, which comparisons
/// write, together with [`Zero`](crate::tensor::sort::Zero). It is the subset
/// [`tensor::sort::Ints`](crate::tensor::sort::Ints) of the sorts of all theories.
pub use super::sort::Ints as Sort;

/// A constant integer matrix literal, held in a tensor of type `T`.
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

/// Pointwise Euclidean division `x div y` of `[m, n]` matrices: the
/// quotient `q` such that `x = q·y + r` with `0 ≤ r < |y|`. It rounds down
/// for positive `y` and up for negative `y`, unlike Rust's `/`, which
/// rounds towards zero. As in SMT-LIB2, `x div 0` is unspecified.
///
/// SMT-LIB2: `(div x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[pyo3::pyclass(frozen, eq, str)]
pub struct Div(pub [usize; 2]);

/// Pointwise Euclidean remainder `x mod y` of `[m, n]` matrices: the `r`
/// such that `x = (x div y)·y + r` with `0 ≤ r < |y|`, so never negative,
/// unlike Rust's `%`. As in SMT-LIB2, `x mod 0` is unspecified.
///
/// SMT-LIB2: `(mod x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[pyo3::pyclass(frozen, eq, str)]
pub struct Mod(pub [usize; 2]);

/// The affine map `x ↦ A·x + B`, where `A` and `B` are constants.
#[derive(Debug)]
pub struct Linear<T>(pub T, pub T);

/// The rectified-linear map `x ↦ max(x, 0)`, applied pointwise to
/// `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[pyo3::pyclass(frozen, eq, str)]
pub struct ReLU(pub [usize; 2]);

/// The index of a maximal entry of an `[m, n]` integer matrix, counting its
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

/// Pointwise choice `if c then x else y` between `[m, n]` integer matrices `x` and `y`,
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

/// The operations of the theory of integer arithmetic, SMT-LIB2's `Ints`,
/// with their boundary in [`Sort`].
///
/// `subenum` defines [`LIA`], the linear ones: all but `Mul`, `Div`, `Mod`
/// and `MatMul`, which multiply or divide two variables. It also generates
/// the casts between them: `From` up to `NIA`, and `TryFrom` down from it.
#[subenum(LIA)]
#[derive(Debug, From)]
pub enum NIA<T> {
    #[subenum(LIA)]
    Const(Const<T>),
    #[subenum(LIA)]
    Add(Add),
    #[subenum(LIA)]
    Sub(Sub),
    Mul(Mul),
    Div(Div),
    Mod(Mod),
    #[subenum(LIA)]
    Linear(Linear<T>),
    #[subenum(LIA)]
    ReLU(ReLU),
    #[subenum(LIA)]
    Argmax(Argmax),
    #[subenum(LIA)]
    Min(Min),
    #[subenum(LIA)]
    Max(Max),
    #[subenum(LIA)]
    Transpose(Transpose),
    MatMul(MatMul),
    #[subenum(LIA)]
    Lt(Lt),
    #[subenum(LIA)]
    Ite(Ite),
    #[subenum(LIA)]
    Le(Le),
    #[subenum(LIA)]
    Gt(Gt),
    #[subenum(LIA)]
    Ge(Ge),
    #[subenum(LIA)]
    Eq(Eq),
    #[subenum(LIA)]
    Ne(Ne),
}

impl<T> Clone for NIA<T>
where
    Const<T>: Clone,
    Linear<T>: Clone,
{
    fn clone(&self) -> Self {
        match self {
            NIA::Const(op) => NIA::Const(op.clone()),
            NIA::Add(op) => NIA::Add(*op),
            NIA::Sub(op) => NIA::Sub(*op),
            NIA::Mul(op) => NIA::Mul(*op),
            NIA::Div(op) => NIA::Div(*op),
            NIA::Mod(op) => NIA::Mod(*op),
            NIA::Linear(op) => NIA::Linear(op.clone()),
            NIA::ReLU(op) => NIA::ReLU(*op),
            NIA::Argmax(op) => NIA::Argmax(*op),
            NIA::Min(op) => NIA::Min(*op),
            NIA::Max(op) => NIA::Max(*op),
            NIA::Transpose(op) => NIA::Transpose(*op),
            NIA::MatMul(op) => NIA::MatMul(*op),
            NIA::Lt(op) => NIA::Lt(*op),
            NIA::Ite(op) => NIA::Ite(*op),
            NIA::Le(op) => NIA::Le(*op),
            NIA::Gt(op) => NIA::Gt(*op),
            NIA::Ge(op) => NIA::Ge(*op),
            NIA::Eq(op) => NIA::Eq(*op),
            NIA::Ne(op) => NIA::Ne(*op),
        }
    }
}

impl<T: fmt::Display> fmt::Display for NIA<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NIA::Const(op) => op.fmt(f),
            NIA::Add(op) => op.fmt(f),
            NIA::Sub(op) => op.fmt(f),
            NIA::Mul(op) => op.fmt(f),
            NIA::Div(op) => op.fmt(f),
            NIA::Mod(op) => op.fmt(f),
            NIA::Linear(op) => op.fmt(f),
            NIA::ReLU(op) => op.fmt(f),
            NIA::Argmax(op) => op.fmt(f),
            NIA::Min(op) => op.fmt(f),
            NIA::Max(op) => op.fmt(f),
            NIA::Transpose(op) => op.fmt(f),
            NIA::MatMul(op) => op.fmt(f),
            NIA::Lt(op) => op.fmt(f),
            NIA::Ite(op) => op.fmt(f),
            NIA::Le(op) => op.fmt(f),
            NIA::Gt(op) => op.fmt(f),
            NIA::Ge(op) => op.fmt(f),
            NIA::Eq(op) => op.fmt(f),
            NIA::Ne(op) => op.fmt(f),
        }
    }
}

impl<T> Clone for LIA<T>
where
    Const<T>: Clone,
    Linear<T>: Clone,
{
    fn clone(&self) -> Self {
        match self {
            LIA::Const(op) => LIA::Const(op.clone()),
            LIA::Add(op) => LIA::Add(*op),
            LIA::Sub(op) => LIA::Sub(*op),
            LIA::Linear(op) => LIA::Linear(op.clone()),
            LIA::ReLU(op) => LIA::ReLU(*op),
            LIA::Argmax(op) => LIA::Argmax(*op),
            LIA::Min(op) => LIA::Min(*op),
            LIA::Max(op) => LIA::Max(*op),
            LIA::Transpose(op) => LIA::Transpose(*op),
            LIA::Lt(op) => LIA::Lt(*op),
            LIA::Ite(op) => LIA::Ite(*op),
            LIA::Le(op) => LIA::Le(*op),
            LIA::Gt(op) => LIA::Gt(*op),
            LIA::Ge(op) => LIA::Ge(*op),
            LIA::Eq(op) => LIA::Eq(*op),
            LIA::Ne(op) => LIA::Ne(*op),
        }
    }
}

impl<T: fmt::Display> fmt::Display for LIA<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LIA::Const(op) => op.fmt(f),
            LIA::Add(op) => op.fmt(f),
            LIA::Sub(op) => op.fmt(f),
            LIA::Linear(op) => op.fmt(f),
            LIA::ReLU(op) => op.fmt(f),
            LIA::Argmax(op) => op.fmt(f),
            LIA::Min(op) => op.fmt(f),
            LIA::Max(op) => op.fmt(f),
            LIA::Transpose(op) => op.fmt(f),
            LIA::Lt(op) => op.fmt(f),
            LIA::Ite(op) => op.fmt(f),
            LIA::Le(op) => op.fmt(f),
            LIA::Gt(op) => op.fmt(f),
            LIA::Ge(op) => op.fmt(f),
            LIA::Eq(op) => op.fmt(f),
            LIA::Ne(op) => op.fmt(f),
        }
    }
}

impl<T: Tensor> Boundary<Sort> for NIA<T> {
    fn dom(&self) -> Poly<Sort> {
        match self {
            NIA::Const(op) => op.dom().map(Sort::from),
            NIA::Add(op) => op.dom().map(Sort::from),
            NIA::Sub(op) => op.dom().map(Sort::from),
            NIA::Mul(op) => op.dom().map(Sort::from),
            NIA::Div(op) => op.dom().map(Sort::from),
            NIA::Mod(op) => op.dom().map(Sort::from),
            NIA::Linear(op) => op.dom().map(Sort::from),
            NIA::ReLU(op) => op.dom().map(Sort::from),
            NIA::Argmax(op) => op.dom().map(Sort::from),
            NIA::Min(op) => op.dom().map(Sort::from),
            NIA::Max(op) => op.dom().map(Sort::from),
            NIA::Transpose(op) => op.dom().map(Sort::from),
            NIA::MatMul(op) => op.dom().map(Sort::from),
            NIA::Lt(op) => op.dom(),
            NIA::Ite(op) => op.dom(),
            NIA::Le(op) => op.dom(),
            NIA::Gt(op) => op.dom(),
            NIA::Ge(op) => op.dom(),
            NIA::Eq(op) => op.dom(),
            NIA::Ne(op) => op.dom(),
        }
    }

    fn cod(&self) -> Poly<Sort> {
        match self {
            NIA::Const(op) => op.cod().map(Sort::from),
            NIA::Add(op) => op.cod().map(Sort::from),
            NIA::Sub(op) => op.cod().map(Sort::from),
            NIA::Mul(op) => op.cod().map(Sort::from),
            NIA::Div(op) => op.cod().map(Sort::from),
            NIA::Mod(op) => op.cod().map(Sort::from),
            NIA::Linear(op) => op.cod().map(Sort::from),
            NIA::ReLU(op) => op.cod().map(Sort::from),
            NIA::Argmax(op) => op.cod().map(Sort::from),
            NIA::Min(op) => op.cod().map(Sort::from),
            NIA::Max(op) => op.cod().map(Sort::from),
            NIA::Transpose(op) => op.cod().map(Sort::from),
            NIA::MatMul(op) => op.cod().map(Sort::from),
            NIA::Lt(op) => op.cod(),
            NIA::Ite(op) => op.cod(),
            NIA::Le(op) => op.cod(),
            NIA::Gt(op) => op.cod(),
            NIA::Ge(op) => op.cod(),
            NIA::Eq(op) => op.cod(),
            NIA::Ne(op) => op.cod(),
        }
    }
}
impl<T: Tensor> Boundary<Sort> for LIA<T> {
    fn dom(&self) -> Poly<Sort> {
        match self {
            LIA::Const(op) => op.dom().map(Sort::from),
            LIA::Add(op) => op.dom().map(Sort::from),
            LIA::Sub(op) => op.dom().map(Sort::from),
            LIA::Linear(op) => op.dom().map(Sort::from),
            LIA::ReLU(op) => op.dom().map(Sort::from),
            LIA::Argmax(op) => op.dom().map(Sort::from),
            LIA::Min(op) => op.dom().map(Sort::from),
            LIA::Max(op) => op.dom().map(Sort::from),
            LIA::Transpose(op) => op.dom().map(Sort::from),
            LIA::Lt(op) => op.dom(),
            LIA::Ite(op) => op.dom(),
            LIA::Le(op) => op.dom(),
            LIA::Gt(op) => op.dom(),
            LIA::Ge(op) => op.dom(),
            LIA::Eq(op) => op.dom(),
            LIA::Ne(op) => op.dom(),
        }
    }

    fn cod(&self) -> Poly<Sort> {
        match self {
            LIA::Const(op) => op.cod().map(Sort::from),
            LIA::Add(op) => op.cod().map(Sort::from),
            LIA::Sub(op) => op.cod().map(Sort::from),
            LIA::Linear(op) => op.cod().map(Sort::from),
            LIA::ReLU(op) => op.cod().map(Sort::from),
            LIA::Argmax(op) => op.cod().map(Sort::from),
            LIA::Min(op) => op.cod().map(Sort::from),
            LIA::Max(op) => op.cod().map(Sort::from),
            LIA::Transpose(op) => op.cod().map(Sort::from),
            LIA::Lt(op) => op.cod(),
            LIA::Ite(op) => op.cod(),
            LIA::Le(op) => op.cod(),
            LIA::Gt(op) => op.cod(),
            LIA::Ge(op) => op.cod(),
            LIA::Eq(op) => op.cod(),
            LIA::Ne(op) => op.cod(),
        }
    }
}

impl<T: Tensor> Boundary<Int> for Const<T> {
    fn dom(&self) -> Poly<Int> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new(self.0.shape())].into()
    }
}

impl Boundary<Int> for Add {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new(self.0); 2].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new(self.0)].into()
    }
}

impl Boundary<Int> for Sub {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new(self.0); 2].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new(self.0)].into()
    }
}

impl Boundary<Int> for Mul {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new(self.0); 2].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new(self.0)].into()
    }
}

impl Boundary<Int> for Div {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new(self.0); 2].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new(self.0)].into()
    }
}

impl Boundary<Int> for Mod {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new(self.0); 2].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new(self.0)].into()
    }
}

/// With `A` of shape `[m, n]` and `B` of shape `[m, k]`, the map takes `x`
/// of shape `[n, k]` to a matrix of shape `[m, k]`.
impl<T: Tensor> Boundary<Int> for Linear<T> {
    fn dom(&self) -> Poly<Int> {
        let [_, n] = self.0.shape();
        let [_, k] = self.1.shape();
        mono![Int::new([n, k])].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new(self.1.shape())].into()
    }
}

impl Boundary<Int> for ReLU {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new(self.0)].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new(self.0)].into()
    }
}

impl Boundary<Int> for Min {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new(self.0); 2].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new(self.0)].into()
    }
}

impl Boundary<Int> for Max {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new(self.0); 2].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new(self.0)].into()
    }
}

/// Takes a matrix of shape `[m, n]` to one of shape `[n, m]`.
impl Boundary<Int> for Transpose {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new(self.0)].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new([self.0[1], self.0[0]])].into()
    }
}

/// Takes matrices of shapes `[m, n]` and `[n, k]` to one of shape `[m, k]`.
impl Boundary<Int> for MatMul {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new(self.0), Int::new(self.1)].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new([self.0[0], self.1[1]])].into()
    }
}

impl Boundary<Sort> for Lt {
    fn dom(&self) -> Poly<Sort> {
        mono![Int::new(self.0).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new(self.0).into()].into()
    }
}

impl Boundary<Sort> for Le {
    fn dom(&self) -> Poly<Sort> {
        mono![Int::new(self.0).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new(self.0).into()].into()
    }
}

impl Boundary<Sort> for Gt {
    fn dom(&self) -> Poly<Sort> {
        mono![Int::new(self.0).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new(self.0).into()].into()
    }
}

impl Boundary<Sort> for Ge {
    fn dom(&self) -> Poly<Sort> {
        mono![Int::new(self.0).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new(self.0).into()].into()
    }
}

impl Boundary<Sort> for Eq {
    fn dom(&self) -> Poly<Sort> {
        mono![Int::new(self.0).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new(self.0).into()].into()
    }
}

impl Boundary<Sort> for Ne {
    fn dom(&self) -> Poly<Sort> {
        mono![Int::new(self.0).into(); 2].into()
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

impl fmt::Display for Mod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Mod")
    }
}

impl fmt::Display for ReLU {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ReLU")
    }
}

impl Boundary<Int> for Argmax {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new(self.0)].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new([1, 1])].into()
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
            Int::new(self.0).into(),
            Int::new(self.0).into()
        ]
        .into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Int::new(self.0).into()].into()
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

/// Extracts an operation of integer arithmetic from a Python object, trying
/// each operation in turn: those carrying no data through their own
/// classes, and [`Const`] and [`Linear`] through the extraction that the
/// tensor type `T` provides.
impl<'py, T> pyo3::FromPyObject<'py> for NIA<T>
where
    Const<T>: pyo3::FromPyObject<'py>,
    Linear<T>: pyo3::FromPyObject<'py>,
{
    fn extract_bound(ob: &pyo3::Bound<'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use pyo3::types::PyAnyMethods;

        if let Ok(op) = ob.extract::<Add>() {
            return Ok(NIA::Add(op));
        }
        if let Ok(op) = ob.extract::<Sub>() {
            return Ok(NIA::Sub(op));
        }
        if let Ok(op) = ob.extract::<Mul>() {
            return Ok(NIA::Mul(op));
        }
        if let Ok(op) = ob.extract::<Div>() {
            return Ok(NIA::Div(op));
        }
        if let Ok(op) = ob.extract::<Mod>() {
            return Ok(NIA::Mod(op));
        }
        if let Ok(op) = ob.extract::<ReLU>() {
            return Ok(NIA::ReLU(op));
        }
        if let Ok(op) = ob.extract::<Argmax>() {
            return Ok(NIA::Argmax(op));
        }
        if let Ok(op) = ob.extract::<Min>() {
            return Ok(NIA::Min(op));
        }
        if let Ok(op) = ob.extract::<Max>() {
            return Ok(NIA::Max(op));
        }
        if let Ok(op) = ob.extract::<Transpose>() {
            return Ok(NIA::Transpose(op));
        }
        if let Ok(op) = ob.extract::<MatMul>() {
            return Ok(NIA::MatMul(op));
        }
        if let Ok(op) = ob.extract::<Lt>() {
            return Ok(NIA::Lt(op));
        }
        if let Ok(op) = ob.extract::<Ite>() {
            return Ok(NIA::Ite(op));
        }
        if let Ok(op) = ob.extract::<Le>() {
            return Ok(NIA::Le(op));
        }
        if let Ok(op) = ob.extract::<Gt>() {
            return Ok(NIA::Gt(op));
        }
        if let Ok(op) = ob.extract::<Ge>() {
            return Ok(NIA::Ge(op));
        }
        if let Ok(op) = ob.extract::<Eq>() {
            return Ok(NIA::Eq(op));
        }
        if let Ok(op) = ob.extract::<Ne>() {
            return Ok(NIA::Ne(op));
        }
        if let Ok(op) = ob.extract::<Const<T>>() {
            return Ok(NIA::Const(op));
        }
        if let Ok(op) = ob.extract::<Linear<T>>() {
            return Ok(NIA::Linear(op));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "expected an operation of integer arithmetic, got {}",
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
impl Mod {
    #[new]
    #[pyo3(signature = (shape, /))]
    fn new(shape: [usize; 2]) -> Self {
        Mod(shape)
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

    fn one(s: Int) -> Poly<Int> {
        mono![s].into()
    }

    fn two(s: Int) -> Poly<Int> {
        mono![s, s].into()
    }

    #[test]
    fn pointwise_boundaries() {
        let r = Int::new([2, 3]);
        assert_eq!(Add([2, 3]).dom(), two(r));
        assert_eq!(Add([2, 3]).cod(), one(r));
        assert_eq!(Sub([2, 3]).dom(), two(r));
        assert_eq!(Mul([2, 3]).dom(), two(r));
        assert_eq!(Mul([2, 3]).cod(), one(r));
        assert_eq!(Div([2, 3]).dom(), two(r));
        assert_eq!(Div([2, 3]).cod(), one(r));
        assert_eq!(Mod([2, 3]).dom(), two(r));
        assert_eq!(Mod([2, 3]).cod(), one(r));
        assert_eq!(Min([2, 3]).dom(), two(r));
        assert_eq!(Max([2, 3]).cod(), one(r));
        assert_eq!(ReLU([2, 3]).dom(), one(r));
        assert_eq!(ReLU([2, 3]).cod(), one(r));
    }

    #[test]
    fn transpose_swaps_shape() {
        let t = Transpose([2, 3]);
        assert_eq!(t.dom(), one(Int::new([2, 3])));
        assert_eq!(t.cod(), one(Int::new([3, 2])));
    }

    #[test]
    fn matmul_contracts_inner_size() {
        let p = MatMul([2, 3], [3, 4]);
        let dom: Poly<Int> = mono![Int::new([2, 3]), Int::new([3, 4])].into();
        assert_eq!(p.dom(), dom);
        assert_eq!(p.cod(), one(Int::new([2, 4])));
    }

    #[test]
    fn ops_display_their_name() {
        assert_eq!(Add([1, 1]).to_string(), "Add");
        assert_eq!(Mul([1, 1]).to_string(), "Mul");
        assert_eq!(Div([1, 1]).to_string(), "Div");
        assert_eq!(Mod([1, 1]).to_string(), "Mod");
        assert_eq!(ReLU([1, 1]).to_string(), "ReLU");
        assert_eq!(Transpose([1, 1]).to_string(), "Transpose");
    }
    #[test]
    fn comparisons_write_booleans() {
        let x: Sort = Int::new([2, 3]).into();
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
