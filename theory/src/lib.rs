pub mod matrix;
pub mod scalar;
#[cfg(feature = "torch")]
pub mod torch;
pub mod zero;

pub use matrix::Matrix;

use base::boundary::{Boundary, Poly};
use base::differentiable::Differentiable;
use derive_more::From;
use std::fmt;

/// The operations of all theories: an operation of the matrix theories, with
/// its data held in matrices of type `T`, or one of the scalar theories; or
/// one of the structural operations [`Havoc`], [`Skip`] and [`Zero`], which
/// belong to no theory and take any sort. Their boundary is in the sorts of
/// all theories, [`Any`].
#[derive(Debug, From)]
pub enum Op<T> {
    Matrix(matrix::Op<T>),
    Scalar(scalar::Op),
    Havoc(Havoc),
    Skip(Skip),
    Zero(Zero),
}

/// The sorts of all theories: a sort of the matrix theories, a sort of the
/// scalar theories, or [`Real0`](zero::Real0), which both share.
///
/// The conversion from the matrix sorts sends their `Real0` to
/// [`Any::Zero`], so that, built through it, [`Any::Matrix`] holds no
/// `Real0`; the scalar sorts hold none.
#[derive(Clone, PartialEq, Debug, Eq)]
pub enum Any {
    Matrix(crate::matrix::Any),
    Scalar(crate::scalar::Any),
    Zero(zero::Real0),
}

impl From<crate::matrix::Any> for Any {
    fn from(s: crate::matrix::Any) -> Self {
        match s {
            crate::matrix::Any::Zero(z) => Any::Zero(z),
            s => Any::Matrix(s),
        }
    }
}

impl From<crate::scalar::Any> for Any {
    fn from(s: crate::scalar::Any) -> Self {
        Any::Scalar(s)
    }
}

impl From<crate::matrix::real::Real> for Any {
    fn from(s: crate::matrix::real::Real) -> Self {
        crate::matrix::Any::from(s).into()
    }
}

impl From<crate::matrix::int::Int> for Any {
    fn from(s: crate::matrix::int::Int) -> Self {
        crate::matrix::Any::from(s).into()
    }
}

impl From<crate::matrix::bool::Bool> for Any {
    fn from(s: crate::matrix::bool::Bool) -> Self {
        crate::matrix::Any::from(s).into()
    }
}

impl From<crate::matrix::word::Word> for Any {
    fn from(s: crate::matrix::word::Word) -> Self {
        crate::matrix::Any::from(s).into()
    }
}

impl From<crate::scalar::Enum> for Any {
    fn from(s: crate::scalar::Enum) -> Self {
        crate::scalar::Any::from(s).into()
    }
}

impl From<crate::scalar::Event> for Any {
    fn from(s: crate::scalar::Event) -> Self {
        crate::scalar::Any::from(s).into()
    }
}

impl From<zero::Real0> for Any {
    fn from(z: zero::Real0) -> Self {
        Any::Zero(z)
    }
}

impl fmt::Display for Any {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Any::Matrix(s) => s.fmt(f),
            Any::Scalar(s) => s.fmt(f),
            Any::Zero(z) => z.fmt(f),
        }
    }
}

/// Each sort's tangent: that of a matrix sort, taken within the matrix
/// theories, a real of one degree higher or [`Real0`](zero::Real0),
/// and [`Real0`](zero::Real0) for the scalar sorts, all discrete.
impl Differentiable for Any {
    type Tangent = Any;

    fn tangent(self) -> Any {
        match self {
            Any::Matrix(s) => s.tangent().into(),
            Any::Scalar(_) => Any::Zero(zero::Real0),
            Any::Zero(z) => z.tangent().into(),
        }
    }
}

/// Writes an arbitrary value of the given sort, of any theory, reading
/// nothing: the nondeterministic choice of a value.
#[derive(Clone, PartialEq, Debug, Eq)]
pub struct Havoc {
    sort: Any,
}

/// Writes the value it reads, of the given sort, of any theory: the identity.
#[derive(Clone, PartialEq, Debug, Eq)]
pub struct Skip {
    sort: Any,
}

/// Writes the zero rate of change of a value, reading nothing: a value of the
/// given sort, the tangent sort of that value, of any theory. Of a real of
/// degree `d`, the zero matrix of degree `d`; of [`zero::Real0`], the
/// tangent of every discrete sort, its only value. Only the tangent sorts are rates of
/// change, so only for them does it mean anything; it is built for any sort,
/// so that asking for it never fails.
#[derive(Clone, PartialEq, Debug, Eq)]
pub struct Zero {
    sort: Any,
}

impl<T> Clone for Op<T>
where
    matrix::Op<T>: Clone,
{
    fn clone(&self) -> Self {
        match self {
            Op::Matrix(op) => Op::Matrix(op.clone()),
            Op::Scalar(op) => Op::Scalar(op.clone()),
            Op::Havoc(op) => Op::Havoc(op.clone()),
            Op::Skip(op) => Op::Skip(op.clone()),
            Op::Zero(op) => Op::Zero(op.clone()),
        }
    }
}

impl Boundary<Any> for Havoc {
    fn dom(&self) -> Poly<Any> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Any> {
        base::mono![self.sort.clone()].into()
    }
}

impl Boundary<Any> for Skip {
    fn dom(&self) -> Poly<Any> {
        base::mono![self.sort.clone()].into()
    }

    fn cod(&self) -> Poly<Any> {
        base::mono![self.sort.clone()].into()
    }
}

impl Boundary<Any> for Zero {
    fn dom(&self) -> Poly<Any> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Any> {
        base::mono![self.sort.clone()].into()
    }
}

/// The boundary of each operation, those of each kind lifted into the sorts of
/// all theories.
impl<T: Matrix> Boundary<Any> for Op<T> {
    fn dom(&self) -> Poly<Any> {
        match self {
            Op::Matrix(op) => op.dom().map(Any::from),
            Op::Scalar(op) => op.dom(),
            Op::Havoc(op) => op.dom(),
            Op::Skip(op) => op.dom(),
            Op::Zero(op) => op.dom(),
        }
    }

    fn cod(&self) -> Poly<Any> {
        match self {
            Op::Matrix(op) => op.cod().map(Any::from),
            Op::Scalar(op) => op.cod(),
            Op::Havoc(op) => op.cod(),
            Op::Skip(op) => op.cod(),
            Op::Zero(op) => op.cod(),
        }
    }
}

/// The nondeterministic choice of a value of any sort.
impl<T> Combinatorial<Any> for Op<T> {
    fn havoc(range: &Any) -> Self {
        Op::Havoc(Havoc {
            sort: range.clone(),
        })
    }
}

/// The identity of any sort.
impl<T> Sequential<Any> for Op<T> {
    fn skip(range: &Any) -> Self {
        Op::Skip(Skip {
            sort: range.clone(),
        })
    }
}

/// The zero rate of change of any sort, meaningful for the tangent sorts.
impl<T> Differential<Any> for Op<T> {
    fn zero(range: &Any) -> Self {
        Op::Zero(Zero {
            sort: range.clone(),
        })
    }
}

impl fmt::Display for Havoc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Havoc")
    }
}

impl fmt::Display for Skip {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Skip")
    }
}

impl fmt::Display for Zero {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Zero")
    }
}

impl<T: fmt::Display> fmt::Display for Op<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::Matrix(op) => op.fmt(f),
            Op::Scalar(op) => op.fmt(f),
            Op::Havoc(op) => op.fmt(f),
            Op::Skip(op) => op.fmt(f),
            Op::Zero(op) => op.fmt(f),
        }
    }
}

/// The failure of a partial constructor, with its message: raised in Python
/// as a `ValueError`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

#[cfg(feature = "pyo3")]
impl From<Error> for pyo3::PyErr {
    fn from(e: Error) -> Self {
        pyo3::exceptions::PyValueError::new_err(e.0)
    }
}

pub trait Combinatorial<S> {
    // Returns the unary itype that chooses an element within the range
    fn havoc(range: &S) -> Self;
}

pub trait Sequential<S> {
    // Returns the unary itype that copies a value within the range
    fn skip(range: &S) -> Self;
}

pub trait Differential<S> {
    // Returns the unary itype that indicates zero derivative
    fn zero(range: &S) -> Self;
}
