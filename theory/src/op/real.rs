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
- [`Lt`], [`Le`], [`Gt`], [`Ge`], [`Eq`](struct@Eq), [`Ne`] — pointwise
  comparisons, which write boolean matrices of the same shape.
- [`Id`] — the identity.
- [`Ite`] — if-then-else, on a single boolean.
- [`Argmax`] — the index of a largest entry.
- [`Havoc`] — an arbitrary matrix, chosen nondeterministically.
- [`Uninterpreted`] — a named value of unknown meaning.
- [`Zero`] — the zero matrix, of rates of change of a given degree.
- [`TAdd`], [`TSub`], [`TLinear`], [`TTranspose`], [`TIte`], [`TId`] —
  [`Add`], [`Sub`], [`Linear`], [`Transpose`], [`Ite`] and [`Id`] on the
  rates of change of a given degree.

The comparisons are those of SMT-LIB2's `Reals` theory, `<`, `<=`, `>` and
`>=`, together with `=` and `distinct` of the Core theory, lifted pointwise
to matrices. As they write booleans, their boundary is in [`Sort`], the
sorts of the theory, rather than in [`Real`]. So is the boundary of
[`Ite`], which reads a boolean.
*/

use crate::Tensor;
use crate::sort::{Bool, Real};
use base::boundary::{Boundary, Poly};
use base::mono;
use derive_more::{Display, From};
use std::fmt;
use subenum::subenum;

/// The sorts of the theory, as in the signature of SMT-LIB2's `Reals`:
/// real matrices, and the booleans of the Core theory, which comparisons
/// write, together with [`Zero`](crate::sort::Zero). It is the subset
/// [`sort::Reals`](crate::sort::Reals) of the sorts of all theories.
pub use crate::sort::Reals as Sort;

/// A constant real matrix literal, held in a tensor of type `T`.
#[derive(Clone, Debug)]
pub struct Const<T>(pub T);

/// Pointwise addition `x + y` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, eq, str, module = "reactor.op.real")
)]
pub struct Add(pub usize, pub usize);

/// Pointwise subtraction `x - y` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, eq, str, module = "reactor.op.real")
)]
pub struct Sub(pub usize, pub usize);

/// Pointwise multiplication `x ⊙ y` of `[m, n]` matrices: the Hadamard
/// product, not the matrix product [`MatMul`].
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, eq, str, module = "reactor.op.real")
)]
pub struct Mul(pub usize, pub usize);

/// Pointwise division `x / y` of `[m, n]` matrices: the Hadamard division
/// `x ⊘ y`. As in SMT-LIB2, `x / 0` is unspecified.
///
/// SMT-LIB2: `(/ x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, eq, str, module = "reactor.op.real")
)]
pub struct Div(pub usize, pub usize);

/// The affine map `x ↦ A·x + B`, where `A` and `B` are constants.
#[derive(Clone, Debug)]
pub struct Linear<T>(pub T, pub T);

/// The rectified-linear map `x ↦ max(x, 0)`, applied pointwise to
/// `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, eq, str, module = "reactor.op.real")
)]
pub struct ReLU(pub usize, pub usize);

/// Pointwise minimum `min(x, y)` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, eq, str, module = "reactor.op.real")
)]
pub struct Min(pub usize, pub usize);

/// Pointwise maximum `max(x, y)` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, eq, str, module = "reactor.op.real")
)]
pub struct Max(pub usize, pub usize);

/// Transposition `xᵀ` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, eq, str, module = "reactor.op.real")
)]
pub struct Transpose(pub usize, pub usize);

/// The matrix product `x·y` of an `[m, n]` matrix `x` and an `[n, k]`
/// matrix `y`, with the sizes given as `MatMul(m, n, k)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, eq, str, module = "reactor.op.real")
)]
pub struct MatMul(pub usize, pub usize, pub usize);

/// Pointwise comparison `x < y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(< x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, eq, str, module = "reactor.op.real")
)]
pub struct Lt(pub usize, pub usize);

/// Pointwise comparison `x ≤ y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(<= x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, eq, str, module = "reactor.op.real")
)]
pub struct Le(pub usize, pub usize);

/// Pointwise comparison `x > y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(> x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, eq, str, module = "reactor.op.real")
)]
pub struct Gt(pub usize, pub usize);

/// Pointwise comparison `x ≥ y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(>= x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, eq, str, module = "reactor.op.real")
)]
pub struct Ge(pub usize, pub usize);

/// Pointwise comparison `x = y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(= x y)`, of the Core theory.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, eq, str, module = "reactor.op.real")
)]
pub struct Eq(pub usize, pub usize);

/// Pointwise comparison `x ≠ y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(distinct x y)`, of the Core theory.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, eq, str, module = "reactor.op.real")
)]
pub struct Ne(pub usize, pub usize);

/// The identity `x ↦ x` on `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, eq, str, module = "reactor.op.real")
)]
pub struct Id(pub usize, pub usize);

/// If-then-else `if c then x else y` of `[m, n]` matrices, on a single
/// boolean `c`, which picks either matrix whole.
///
/// SMT-LIB2: `(ite c x y)`, of the Core theory, with the same `c` for every
/// entry.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, eq, str, module = "reactor.op.real")
)]
pub struct Ite(pub usize, pub usize);

/// The index of the first largest entry of `[m, n]` matrices, counted from
/// `0` in row-major order, as a `[1, 1]` real matrix.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, eq, str, module = "reactor.op.real")
)]
pub struct Argmax(pub usize, pub usize);

/// An arbitrary real matrix of shape `[m, n]`, chosen nondeterministically.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, eq, str, module = "reactor.op.real")
)]
pub struct Havoc(pub usize, pub usize);

/// A real matrix of shape `[m, n]` of unknown meaning, named by the given
/// symbol, as `Uninterpreted(symbol, m, n)`: e.g., the result of a call that
/// the frontend cannot translate.
#[derive(Clone, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, eq, str, module = "reactor.op.real")
)]
pub struct Uninterpreted(pub String, pub usize, pub usize);

/// The zero `[m, n]` matrix of the rates of change of degree `d`, as
/// `Zero(m, n, d)`: with `d = 1`, the derivative of a constant. With
/// `d = 0`, it is the zero matrix of plain reals.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, eq, str, module = "reactor.op.real")
)]
pub struct Zero(pub usize, pub usize, pub u8);

/// Pointwise addition `x + y` of `[m, n]` matrices of the rates of change of
/// degree `d`, as `TAdd(m, n, d)`. Addition is linear, so `TAdd(m, n, d + 1)`
/// is the derivative of `TAdd(m, n, d)`, and `TAdd(m, n, 0)` is [`Add`].
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, eq, str, module = "reactor.op.real")
)]
pub struct TAdd(pub usize, pub usize, pub u8);

/// Pointwise subtraction `x - y` of `[m, n]` matrices of the rates of
/// change of degree `d`, as `TSub(m, n, d)`. Subtraction is linear, so
/// `TSub(m, n, d + 1)` is the derivative of `TSub(m, n, d)`, and
/// `TSub(m, n, 0)` is [`Sub`].
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, eq, str, module = "reactor.op.real")
)]
pub struct TSub(pub usize, pub usize, pub u8);

/// The affine map `x ↦ A·x + B` on the rates of change of degree `d`, as
/// `TLinear(A, B, d)`, where `A` and `B` are constants; `TLinear(A, B, 0)`
/// is [`Linear`]`(A, B)`. It adds `B`, as `Linear` did on rates in the old
/// LRA, so the derivative of `TLinear(A, B, d)` is `TLinear(A, 0, d + 1)`.
#[derive(Clone, Debug)]
pub struct TLinear<T>(pub T, pub T, pub u8);

/// Transposition `xᵀ` of `[m, n]` matrices of the rates of change of degree
/// `d`, as `TTranspose(m, n, d)`. Transposition is linear, so
/// `TTranspose(m, n, d + 1)` is the derivative of `TTranspose(m, n, d)`,
/// and `TTranspose(m, n, 0)` is [`Transpose`].
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, eq, str, module = "reactor.op.real")
)]
pub struct TTranspose(pub usize, pub usize, pub u8);

/// If-then-else `if c then x else y` of `[m, n]` matrices of the rates of
/// change of degree `d`, on a single (plain) boolean `c`, as
/// `TIte(m, n, d)`; `TIte(m, n, 0)` is [`Ite`].
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, eq, str, module = "reactor.op.real")
)]
pub struct TIte(pub usize, pub usize, pub u8);

/// The identity `x ↦ x` on `[m, n]` matrices of the rates of change of
/// degree `d`, as `TId(m, n, d)`; `TId(m, n, 0)` is [`Id`].
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, eq, str, module = "reactor.op.real")
)]
pub struct TId(pub usize, pub usize, pub u8);

/// The operations of the theory of real arithmetic, SMT-LIB2's `Reals`,
/// with their boundary in [`Sort`].
///
/// `subenum` defines [`LRA`], the linear ones: all but `Mul`, `Div` and
/// `MatMul`, which multiply or divide two variables. It also generates the
/// casts between them: `From` up to `NRA`, and `TryFrom` down from it.
#[subenum(LRA)]
#[derive(Clone, Debug, From, Display)]
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
    Min(Min),
    #[subenum(LRA)]
    Max(Max),
    #[subenum(LRA)]
    Transpose(Transpose),
    MatMul(MatMul),
    #[subenum(LRA)]
    Lt(Lt),
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
    #[subenum(LRA)]
    Id(Id),
    #[subenum(LRA)]
    Ite(Ite),
    #[subenum(LRA)]
    Argmax(Argmax),
    #[subenum(LRA)]
    Havoc(Havoc),
    #[subenum(LRA)]
    Uninterpreted(Uninterpreted),
    #[subenum(LRA)]
    Zero(Zero),
    #[subenum(LRA)]
    TAdd(TAdd),
    #[subenum(LRA)]
    TSub(TSub),
    #[subenum(LRA)]
    TLinear(TLinear<T>),
    #[subenum(LRA)]
    TTranspose(TTranspose),
    #[subenum(LRA)]
    TIte(TIte),
    #[subenum(LRA)]
    TId(TId),
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
            NRA::Min(op) => op.dom().map(Sort::from),
            NRA::Max(op) => op.dom().map(Sort::from),
            NRA::Transpose(op) => op.dom().map(Sort::from),
            NRA::MatMul(op) => op.dom().map(Sort::from),
            NRA::Lt(op) => op.dom(),
            NRA::Le(op) => op.dom(),
            NRA::Gt(op) => op.dom(),
            NRA::Ge(op) => op.dom(),
            NRA::Eq(op) => op.dom(),
            NRA::Ne(op) => op.dom(),
            NRA::Id(op) => op.dom().map(Sort::from),
            NRA::Ite(op) => op.dom(),
            NRA::Argmax(op) => op.dom().map(Sort::from),
            NRA::Havoc(op) => op.dom().map(Sort::from),
            NRA::Uninterpreted(op) => op.dom().map(Sort::from),
            NRA::Zero(op) => op.dom().map(Sort::from),
            NRA::TAdd(op) => op.dom().map(Sort::from),
            NRA::TSub(op) => op.dom().map(Sort::from),
            NRA::TLinear(op) => op.dom().map(Sort::from),
            NRA::TTranspose(op) => op.dom().map(Sort::from),
            NRA::TIte(op) => op.dom(),
            NRA::TId(op) => op.dom().map(Sort::from),
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
            NRA::Min(op) => op.cod().map(Sort::from),
            NRA::Max(op) => op.cod().map(Sort::from),
            NRA::Transpose(op) => op.cod().map(Sort::from),
            NRA::MatMul(op) => op.cod().map(Sort::from),
            NRA::Lt(op) => op.cod(),
            NRA::Le(op) => op.cod(),
            NRA::Gt(op) => op.cod(),
            NRA::Ge(op) => op.cod(),
            NRA::Eq(op) => op.cod(),
            NRA::Ne(op) => op.cod(),
            NRA::Id(op) => op.cod().map(Sort::from),
            NRA::Ite(op) => op.cod(),
            NRA::Argmax(op) => op.cod().map(Sort::from),
            NRA::Havoc(op) => op.cod().map(Sort::from),
            NRA::Uninterpreted(op) => op.cod().map(Sort::from),
            NRA::Zero(op) => op.cod().map(Sort::from),
            NRA::TAdd(op) => op.cod().map(Sort::from),
            NRA::TSub(op) => op.cod().map(Sort::from),
            NRA::TLinear(op) => op.cod().map(Sort::from),
            NRA::TTranspose(op) => op.cod().map(Sort::from),
            NRA::TIte(op) => op.cod(),
            NRA::TId(op) => op.cod().map(Sort::from),
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
            LRA::Min(op) => op.dom().map(Sort::from),
            LRA::Max(op) => op.dom().map(Sort::from),
            LRA::Transpose(op) => op.dom().map(Sort::from),
            LRA::Lt(op) => op.dom(),
            LRA::Le(op) => op.dom(),
            LRA::Gt(op) => op.dom(),
            LRA::Ge(op) => op.dom(),
            LRA::Eq(op) => op.dom(),
            LRA::Ne(op) => op.dom(),
            LRA::Id(op) => op.dom().map(Sort::from),
            LRA::Ite(op) => op.dom(),
            LRA::Argmax(op) => op.dom().map(Sort::from),
            LRA::Havoc(op) => op.dom().map(Sort::from),
            LRA::Uninterpreted(op) => op.dom().map(Sort::from),
            LRA::Zero(op) => op.dom().map(Sort::from),
            LRA::TAdd(op) => op.dom().map(Sort::from),
            LRA::TSub(op) => op.dom().map(Sort::from),
            LRA::TLinear(op) => op.dom().map(Sort::from),
            LRA::TTranspose(op) => op.dom().map(Sort::from),
            LRA::TIte(op) => op.dom(),
            LRA::TId(op) => op.dom().map(Sort::from),
        }
    }

    fn cod(&self) -> Poly<Sort> {
        match self {
            LRA::Const(op) => op.cod().map(Sort::from),
            LRA::Add(op) => op.cod().map(Sort::from),
            LRA::Sub(op) => op.cod().map(Sort::from),
            LRA::Linear(op) => op.cod().map(Sort::from),
            LRA::ReLU(op) => op.cod().map(Sort::from),
            LRA::Min(op) => op.cod().map(Sort::from),
            LRA::Max(op) => op.cod().map(Sort::from),
            LRA::Transpose(op) => op.cod().map(Sort::from),
            LRA::Lt(op) => op.cod(),
            LRA::Le(op) => op.cod(),
            LRA::Gt(op) => op.cod(),
            LRA::Ge(op) => op.cod(),
            LRA::Eq(op) => op.cod(),
            LRA::Ne(op) => op.cod(),
            LRA::Id(op) => op.cod().map(Sort::from),
            LRA::Ite(op) => op.cod(),
            LRA::Argmax(op) => op.cod().map(Sort::from),
            LRA::Havoc(op) => op.cod().map(Sort::from),
            LRA::Uninterpreted(op) => op.cod().map(Sort::from),
            LRA::Zero(op) => op.cod().map(Sort::from),
            LRA::TAdd(op) => op.cod().map(Sort::from),
            LRA::TSub(op) => op.cod().map(Sort::from),
            LRA::TLinear(op) => op.cod().map(Sort::from),
            LRA::TTranspose(op) => op.cod().map(Sort::from),
            LRA::TIte(op) => op.cod(),
            LRA::TId(op) => op.cod().map(Sort::from),
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
        mono![Real::new([self.0, self.1]); 2].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1])].into()
    }
}

impl Boundary<Real> for Sub {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1]); 2].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1])].into()
    }
}

impl Boundary<Real> for Mul {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1]); 2].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1])].into()
    }
}

impl Boundary<Real> for Div {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1]); 2].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1])].into()
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
        mono![Real::new([self.0, self.1])].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1])].into()
    }
}

impl Boundary<Real> for Min {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1]); 2].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1])].into()
    }
}

impl Boundary<Real> for Max {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1]); 2].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1])].into()
    }
}

/// Takes a matrix of shape `[m, n]` to one of shape `[n, m]`.
impl Boundary<Real> for Transpose {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1])].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new([self.1, self.0])].into()
    }
}

/// Takes matrices of shapes `[m, n]` and `[n, k]` to one of shape `[m, k]`.
impl Boundary<Real> for MatMul {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1]), Real::new([self.1, self.2])].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.2])].into()
    }
}

impl Boundary<Sort> for Lt {
    fn dom(&self) -> Poly<Sort> {
        mono![Real::new([self.0, self.1]).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new([self.0, self.1]).into()].into()
    }
}

impl Boundary<Sort> for Le {
    fn dom(&self) -> Poly<Sort> {
        mono![Real::new([self.0, self.1]).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new([self.0, self.1]).into()].into()
    }
}

impl Boundary<Sort> for Gt {
    fn dom(&self) -> Poly<Sort> {
        mono![Real::new([self.0, self.1]).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new([self.0, self.1]).into()].into()
    }
}

impl Boundary<Sort> for Ge {
    fn dom(&self) -> Poly<Sort> {
        mono![Real::new([self.0, self.1]).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new([self.0, self.1]).into()].into()
    }
}

impl Boundary<Sort> for Eq {
    fn dom(&self) -> Poly<Sort> {
        mono![Real::new([self.0, self.1]).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new([self.0, self.1]).into()].into()
    }
}

impl Boundary<Sort> for Ne {
    fn dom(&self) -> Poly<Sort> {
        mono![Real::new([self.0, self.1]).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new([self.0, self.1]).into()].into()
    }
}

impl Boundary<Real> for Id {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1])].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1])].into()
    }
}

/// Takes a boolean of shape `[1, 1]` and two matrices of shape `[m, n]` to
/// a matrix of shape `[m, n]`.
impl Boundary<Sort> for Ite {
    fn dom(&self) -> Poly<Sort> {
        let x: Sort = Real::new([self.0, self.1]).into();
        mono![Bool::new([1, 1]).into(), x, x].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Real::new([self.0, self.1]).into()].into()
    }
}

/// Takes a matrix of shape `[m, n]` to one of shape `[1, 1]`.
impl Boundary<Real> for Argmax {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1])].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new([1, 1])].into()
    }
}

impl Boundary<Real> for Havoc {
    fn dom(&self) -> Poly<Real> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1])].into()
    }
}

impl Boundary<Real> for Uninterpreted {
    fn dom(&self) -> Poly<Real> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new([self.1, self.2])].into()
    }
}

impl Boundary<Real> for Zero {
    fn dom(&self) -> Poly<Real> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::with_degree([self.0, self.1], self.2)].into()
    }
}

impl Boundary<Real> for TAdd {
    fn dom(&self) -> Poly<Real> {
        mono![Real::with_degree([self.0, self.1], self.2); 2].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::with_degree([self.0, self.1], self.2)].into()
    }
}

impl Boundary<Real> for TSub {
    fn dom(&self) -> Poly<Real> {
        mono![Real::with_degree([self.0, self.1], self.2); 2].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::with_degree([self.0, self.1], self.2)].into()
    }
}

/// As [`Linear`], with `A` of shape `[m, n]` and `B` of shape `[m, k]`, the
/// map takes `x` of shape `[n, k]` to a matrix of shape `[m, k]`, here of
/// the rates of change of degree `d`.
impl<T: Tensor> Boundary<Real> for TLinear<T> {
    fn dom(&self) -> Poly<Real> {
        let [_, n] = self.0.shape();
        let [_, k] = self.1.shape();
        mono![Real::with_degree([n, k], self.2)].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::with_degree(self.1.shape(), self.2)].into()
    }
}

/// Takes a matrix of shape `[m, n]` to one of shape `[n, m]`, both of the
/// rates of change of degree `d`.
impl Boundary<Real> for TTranspose {
    fn dom(&self) -> Poly<Real> {
        mono![Real::with_degree([self.0, self.1], self.2)].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::with_degree([self.1, self.0], self.2)].into()
    }
}

/// Takes a (plain) boolean of shape `[1, 1]` and two matrices of shape
/// `[m, n]` of the rates of change of degree `d` to such a matrix.
impl Boundary<Sort> for TIte {
    fn dom(&self) -> Poly<Sort> {
        let x: Sort = Real::with_degree([self.0, self.1], self.2).into();
        mono![Bool::new([1, 1]).into(), x, x].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Real::with_degree([self.0, self.1], self.2).into()].into()
    }
}

impl Boundary<Real> for TId {
    fn dom(&self) -> Poly<Real> {
        mono![Real::with_degree([self.0, self.1], self.2)].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::with_degree([self.0, self.1], self.2)].into()
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

impl fmt::Display for Id {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Id")
    }
}

impl fmt::Display for Ite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ite")
    }
}

impl fmt::Display for Argmax {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Argmax")
    }
}

impl fmt::Display for Havoc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Havoc")
    }
}

impl fmt::Display for Uninterpreted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Uninterpreted({})", self.0)
    }
}

impl fmt::Display for Zero {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Zero")
    }
}

impl<T: fmt::Display> fmt::Display for TLinear<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TLinear({}, {})", self.0, self.1)
    }
}

impl fmt::Display for TAdd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TAdd")
    }
}

impl fmt::Display for TSub {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TSub")
    }
}

impl fmt::Display for TTranspose {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TTranspose")
    }
}

impl fmt::Display for TIte {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TIte")
    }
}

impl fmt::Display for TId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TId")
    }
}

/// Extracts an operation of real arithmetic from a Python object, trying
/// each operation in turn: those carrying no data through their own
/// classes, and [`Const`], [`Linear`] and [`TLinear`] through the
/// extraction that the tensor type `T` provides.
#[cfg(feature = "pyo3")]
impl<'py, T> pyo3::FromPyObject<'py> for NRA<T>
where
    Const<T>: pyo3::FromPyObject<'py>,
    Linear<T>: pyo3::FromPyObject<'py>,
    TLinear<T>: pyo3::FromPyObject<'py>,
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
        if let Ok(op) = ob.extract::<Id>() {
            return Ok(NRA::Id(op));
        }
        if let Ok(op) = ob.extract::<Ite>() {
            return Ok(NRA::Ite(op));
        }
        if let Ok(op) = ob.extract::<Argmax>() {
            return Ok(NRA::Argmax(op));
        }
        if let Ok(op) = ob.extract::<Havoc>() {
            return Ok(NRA::Havoc(op));
        }
        if let Ok(op) = ob.extract::<Uninterpreted>() {
            return Ok(NRA::Uninterpreted(op));
        }
        if let Ok(op) = ob.extract::<Zero>() {
            return Ok(NRA::Zero(op));
        }
        if let Ok(op) = ob.extract::<TAdd>() {
            return Ok(NRA::TAdd(op));
        }
        if let Ok(op) = ob.extract::<TSub>() {
            return Ok(NRA::TSub(op));
        }
        if let Ok(op) = ob.extract::<TTranspose>() {
            return Ok(NRA::TTranspose(op));
        }
        if let Ok(op) = ob.extract::<TIte>() {
            return Ok(NRA::TIte(op));
        }
        if let Ok(op) = ob.extract::<TId>() {
            return Ok(NRA::TId(op));
        }
        if let Ok(op) = ob.extract::<Const<T>>() {
            return Ok(NRA::Const(op));
        }
        if let Ok(op) = ob.extract::<Linear<T>>() {
            return Ok(NRA::Linear(op));
        }
        if let Ok(op) = ob.extract::<TLinear<T>>() {
            return Ok(NRA::TLinear(op));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "expected an operation of real arithmetic, got {}",
            ob.get_type()
        )))
    }
}

/// Converts an operation of real arithmetic into a Python object: those
/// carrying no data into their own classes, and [`Const`], [`Linear`] and
/// [`TLinear`] through the conversion that the tensor type `T` provides.
#[cfg(feature = "pyo3")]
impl<'py, T> pyo3::IntoPyObject<'py> for NRA<T>
where
    Const<T>: pyo3::IntoPyObject<'py>,
    Linear<T>: pyo3::IntoPyObject<'py>,
    TLinear<T>: pyo3::IntoPyObject<'py>,
{
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, pyo3::PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> pyo3::PyResult<Self::Output> {
        use pyo3::IntoPyObjectExt;

        match self {
            NRA::Const(op) => op.into_bound_py_any(py),
            NRA::Add(op) => op.into_bound_py_any(py),
            NRA::Sub(op) => op.into_bound_py_any(py),
            NRA::Mul(op) => op.into_bound_py_any(py),
            NRA::Div(op) => op.into_bound_py_any(py),
            NRA::Linear(op) => op.into_bound_py_any(py),
            NRA::ReLU(op) => op.into_bound_py_any(py),
            NRA::Min(op) => op.into_bound_py_any(py),
            NRA::Max(op) => op.into_bound_py_any(py),
            NRA::Transpose(op) => op.into_bound_py_any(py),
            NRA::MatMul(op) => op.into_bound_py_any(py),
            NRA::Lt(op) => op.into_bound_py_any(py),
            NRA::Le(op) => op.into_bound_py_any(py),
            NRA::Gt(op) => op.into_bound_py_any(py),
            NRA::Ge(op) => op.into_bound_py_any(py),
            NRA::Eq(op) => op.into_bound_py_any(py),
            NRA::Ne(op) => op.into_bound_py_any(py),
            NRA::Id(op) => op.into_bound_py_any(py),
            NRA::Ite(op) => op.into_bound_py_any(py),
            NRA::Argmax(op) => op.into_bound_py_any(py),
            NRA::Havoc(op) => op.into_bound_py_any(py),
            NRA::Uninterpreted(op) => op.into_bound_py_any(py),
            NRA::Zero(op) => op.into_bound_py_any(py),
            NRA::TAdd(op) => op.into_bound_py_any(py),
            NRA::TSub(op) => op.into_bound_py_any(py),
            NRA::TLinear(op) => op.into_bound_py_any(py),
            NRA::TTranspose(op) => op.into_bound_py_any(py),
            NRA::TIte(op) => op.into_bound_py_any(py),
            NRA::TId(op) => op.into_bound_py_any(py),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base::differentiable::T;

    fn one(s: Real) -> Poly<Real> {
        mono![s].into()
    }

    fn two(s: Real) -> Poly<Real> {
        mono![s, s].into()
    }

    #[test]
    fn pointwise_boundaries() {
        let r = Real::new([2, 3]);
        assert_eq!(Add(2, 3).dom(), two(r));
        assert_eq!(Add(2, 3).cod(), one(r));
        assert_eq!(Sub(2, 3).dom(), two(r));
        assert_eq!(Mul(2, 3).dom(), two(r));
        assert_eq!(Mul(2, 3).cod(), one(r));
        assert_eq!(Div(2, 3).dom(), two(r));
        assert_eq!(Div(2, 3).cod(), one(r));
        assert_eq!(Min(2, 3).dom(), two(r));
        assert_eq!(Max(2, 3).cod(), one(r));
        assert_eq!(ReLU(2, 3).dom(), one(r));
        assert_eq!(ReLU(2, 3).cod(), one(r));
    }

    #[test]
    fn transpose_swaps_shape() {
        let t = Transpose(2, 3);
        assert_eq!(t.dom(), one(Real::new([2, 3])));
        assert_eq!(t.cod(), one(Real::new([3, 2])));
    }

    #[test]
    fn matmul_contracts_inner_size() {
        let p = MatMul(2, 3, 4);
        let dom: Poly<Real> = mono![Real::new([2, 3]), Real::new([3, 4])].into();
        assert_eq!(p.dom(), dom);
        assert_eq!(p.cod(), one(Real::new([2, 4])));
    }

    #[test]
    fn id_keeps_sort() {
        let x = Real::new([2, 3]);
        assert_eq!(Id(2, 3).dom(), one(x));
        assert_eq!(Id(2, 3).cod(), one(x));
    }

    #[test]
    fn ite_reads_a_single_boolean() {
        let c: Sort = Bool::new([1, 1]).into();
        let x: Sort = Real::new([2, 3]).into();
        let dom: Poly<Sort> = mono![c, x, x].into();
        let cod: Poly<Sort> = mono![x].into();
        assert_eq!(Ite(2, 3).dom(), dom);
        assert_eq!(Ite(2, 3).cod(), cod);
    }

    #[test]
    fn argmax_writes_a_single_entry() {
        assert_eq!(Argmax(2, 3).dom(), one(Real::new([2, 3])));
        assert_eq!(Argmax(2, 3).cod(), one(Real::new([1, 1])));
    }

    #[test]
    fn havoc_reads_nothing() {
        assert_eq!(Havoc(2, 3).dom(), Poly::unit());
        assert_eq!(Havoc(2, 3).cod(), one(Real::new([2, 3])));
    }

    #[test]
    fn uninterpreted_reads_nothing() {
        let u = Uninterpreted("f".to_string(), 2, 3);
        assert_eq!(u.dom(), Poly::unit());
        assert_eq!(u.cod(), one(Real::new([2, 3])));
    }

    #[test]
    fn zero_writes_rates_of_its_degree() {
        let r = Real::new([2, 3]);
        assert_eq!(Zero(2, 3, 1).dom(), Poly::unit());
        assert_eq!(Zero(2, 3, 1).cod(), one(T(r)));
        assert_eq!(Zero(2, 3, 2).cod(), one(T(T(r))));
        assert_eq!(Zero(2, 3, 0).cod(), one(r));
    }

    #[test]
    fn tangent_ops_act_on_rates_of_their_degree() {
        let r = Real::new([2, 3]);
        assert_eq!(TAdd(2, 3, 1).dom(), two(T(r)));
        assert_eq!(TAdd(2, 3, 1).cod(), one(T(r)));
        assert_eq!(TSub(2, 3, 2).dom(), two(T(T(r))));
        assert_eq!(TSub(2, 3, 2).cod(), one(T(T(r))));
        assert_eq!(TId(2, 3, 1).dom(), one(T(r)));
        assert_eq!(TId(2, 3, 1).cod(), one(T(r)));
        assert_eq!(TTranspose(2, 3, 1).dom(), one(T(r)));
        assert_eq!(TTranspose(2, 3, 1).cod(), one(T(Real::new([3, 2]))));

        let c: Sort = Bool::new([1, 1]).into();
        let x: Sort = T(r).into();
        let dom: Poly<Sort> = mono![c, x, x].into();
        let cod: Poly<Sort> = mono![x].into();
        assert_eq!(TIte(2, 3, 1).dom(), dom);
        assert_eq!(TIte(2, 3, 1).cod(), cod);
    }

    #[test]
    fn tangent_ops_of_degree_zero_are_the_plain_ones() {
        assert_eq!(TAdd(2, 3, 0).dom(), Add(2, 3).dom());
        assert_eq!(TSub(2, 3, 0).cod(), Sub(2, 3).cod());
        assert_eq!(TId(2, 3, 0).dom(), Id(2, 3).dom());
        assert_eq!(TTranspose(2, 3, 0).cod(), Transpose(2, 3).cod());
        assert_eq!(TIte(2, 3, 0).dom(), Ite(2, 3).dom());
        assert_eq!(TIte(2, 3, 0).cod(), Ite(2, 3).cod());
    }

    #[cfg(feature = "torch")]
    #[test]
    fn tlinear_is_linear_on_rates() {
        use pyo3_tch::tch;
        let a = || -> crate::PyTensor { tch::Tensor::zeros([4, 2], tch::kind::FLOAT_CPU).into() };
        let b = || -> crate::PyTensor { tch::Tensor::zeros([4, 3], tch::kind::FLOAT_CPU).into() };
        let t = TLinear(a(), b(), 1);
        assert_eq!(t.dom(), one(T(Real::new([2, 3]))));
        assert_eq!(t.cod(), one(T(Real::new([4, 3]))));
        let t = TLinear(a(), b(), 0);
        assert_eq!(t.dom(), Linear(a(), b()).dom());
        assert_eq!(t.cod(), Linear(a(), b()).cod());
    }

    #[test]
    fn ops_display_their_name() {
        assert_eq!(Add(1, 1).to_string(), "Add");
        assert_eq!(Mul(1, 1).to_string(), "Mul");
        assert_eq!(Div(1, 1).to_string(), "Div");
        assert_eq!(ReLU(1, 1).to_string(), "ReLU");
        assert_eq!(Transpose(1, 1).to_string(), "Transpose");
        assert_eq!(Id(1, 1).to_string(), "Id");
        assert_eq!(Ite(1, 1).to_string(), "Ite");
        assert_eq!(Argmax(1, 1).to_string(), "Argmax");
        assert_eq!(Havoc(1, 1).to_string(), "Havoc");
        assert_eq!(
            Uninterpreted("f".to_string(), 1, 1).to_string(),
            "Uninterpreted(f)"
        );
        assert_eq!(Zero(1, 1, 1).to_string(), "Zero");
        assert_eq!(TAdd(1, 1, 1).to_string(), "TAdd");
    }
    #[test]
    fn comparisons_write_booleans() {
        let x: Sort = Real::new([2, 3]).into();
        let b: Sort = Bool::new([2, 3]).into();
        let dom: Poly<Sort> = mono![x, x].into();
        let cod: Poly<Sort> = mono![b].into();
        for (d, c) in [
            (Lt(2, 3).dom(), Lt(2, 3).cod()),
            (Le(2, 3).dom(), Le(2, 3).cod()),
            (Gt(2, 3).dom(), Gt(2, 3).cod()),
            (Ge(2, 3).dom(), Ge(2, 3).cod()),
            (Eq(2, 3).dom(), Eq(2, 3).cod()),
            (Ne(2, 3).dom(), Ne(2, 3).cod()),
        ] {
            assert_eq!(d, dom);
            assert_eq!(c, cod);
        }
        assert_eq!(Le(1, 1).to_string(), "Le");
        assert_eq!(Ne(1, 1).to_string(), "Ne");
    }
}
