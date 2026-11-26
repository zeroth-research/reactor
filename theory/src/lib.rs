pub mod matrix;
pub mod scalar;
pub mod sort;
#[cfg(feature = "torch")]
pub mod torch;
pub mod zero;

pub use matrix::Matrix;

use base::boundary::{Boundary, Poly};
use derive_more::From;
use std::fmt;

/// The operations of all theories: an operation of the matrix theories, with
/// its data held in matrices of type `T`, or one of the scalar theories; or
/// one of the structural operations [`Havoc`], [`Skip`] and [`Zero`], which
/// belong to no theory and take any sort. Their boundary is in the sorts of
/// all theories, [`sort::Any`].
#[derive(Debug, From)]
pub enum Any<T> {
    Matrix(matrix::Any<T>),
    Scalar(scalar::Any),
    Havoc(Havoc),
    Skip(Skip),
    Zero(Zero),
}

/// Writes an arbitrary value of the given sort, of any theory, reading
/// nothing: the nondeterministic choice of a value.
#[derive(Clone, PartialEq, Debug, Eq)]
pub struct Havoc {
    sort: sort::Any,
}

/// Writes the value it reads, of the given sort, of any theory: the identity.
#[derive(Clone, PartialEq, Debug, Eq)]
pub struct Skip {
    sort: sort::Any,
}

/// Writes the zero rate of change of a value, reading nothing: a value of the
/// given sort, the tangent sort of that value, of any theory. Of a real of
/// degree `d`, the zero matrix of degree `d`; of [`sort::Zero`], the tangent of
/// every discrete sort, its only value. Only the tangent sorts are rates of
/// change, so only for them does it mean anything; it is built for any sort,
/// so that asking for it never fails.
#[derive(Clone, PartialEq, Debug, Eq)]
pub struct Zero {
    sort: sort::Any,
}

impl<T> Clone for Any<T>
where
    matrix::Any<T>: Clone,
{
    fn clone(&self) -> Self {
        match self {
            Any::Matrix(op) => Any::Matrix(op.clone()),
            Any::Scalar(op) => Any::Scalar(op.clone()),
            Any::Havoc(op) => Any::Havoc(op.clone()),
            Any::Skip(op) => Any::Skip(op.clone()),
            Any::Zero(op) => Any::Zero(op.clone()),
        }
    }
}

impl Boundary<sort::Any> for Havoc {
    fn dom(&self) -> Poly<sort::Any> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<sort::Any> {
        base::mono![self.sort.clone()].into()
    }
}

impl Boundary<sort::Any> for Skip {
    fn dom(&self) -> Poly<sort::Any> {
        base::mono![self.sort.clone()].into()
    }

    fn cod(&self) -> Poly<sort::Any> {
        base::mono![self.sort.clone()].into()
    }
}

impl Boundary<sort::Any> for Zero {
    fn dom(&self) -> Poly<sort::Any> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<sort::Any> {
        base::mono![self.sort.clone()].into()
    }
}

/// The boundary of each operation, those of each kind lifted into the sorts of
/// all theories.
impl<T: Matrix> Boundary<sort::Any> for Any<T> {
    fn dom(&self) -> Poly<sort::Any> {
        match self {
            Any::Matrix(op) => op.dom().map(sort::Any::from),
            Any::Scalar(op) => op.dom(),
            Any::Havoc(op) => op.dom(),
            Any::Skip(op) => op.dom(),
            Any::Zero(op) => op.dom(),
        }
    }

    fn cod(&self) -> Poly<sort::Any> {
        match self {
            Any::Matrix(op) => op.cod().map(sort::Any::from),
            Any::Scalar(op) => op.cod(),
            Any::Havoc(op) => op.cod(),
            Any::Skip(op) => op.cod(),
            Any::Zero(op) => op.cod(),
        }
    }
}

/// The nondeterministic choice of a value of any sort.
impl<T> Combinatorial<sort::Any> for Any<T> {
    fn havoc(range: &sort::Any) -> Self {
        Any::Havoc(Havoc {
            sort: range.clone(),
        })
    }
}

/// The identity of any sort.
impl<T> Sequential<sort::Any> for Any<T> {
    fn skip(range: &sort::Any) -> Self {
        Any::Skip(Skip {
            sort: range.clone(),
        })
    }
}

/// The zero rate of change of any sort, meaningful for the tangent sorts.
impl<T> Differential<sort::Any> for Any<T> {
    fn zero(range: &sort::Any) -> Self {
        Any::Zero(Zero {
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

impl<T: fmt::Display> fmt::Display for Any<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Any::Matrix(op) => op.fmt(f),
            Any::Scalar(op) => op.fmt(f),
            Any::Havoc(op) => op.fmt(f),
            Any::Skip(op) => op.fmt(f),
            Any::Zero(op) => op.fmt(f),
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
