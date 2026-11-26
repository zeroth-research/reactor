/*!
# The sorts shared by all theories

[`Zero`] is the zero vector space `ℝ⁰`: it has exactly one value, the vector
of no coordinates. It is the tangent of every discrete sort, of booleans,
integers, words and enumerations, scalar or matrix alike: a discrete set is a
manifold of dimension 0, whose tangent space at every point is `ℝ⁰`, whatever
the shape of its values, as a matrix of discrete values is a discrete set
too. So it has no shape, and the matrix theories and the scalar theories
share it. It is its own tangent.

The zero rate of change of a real is not the value of [`Zero`], but the zero
matrix of the degree of the rate of change, written by
[`matrix::real::Zero`](crate::matrix::real::Zero).

[`Any`] holds the sorts of all theories, matrix and scalar, with [`Zero`],
which both share, once.
*/

use base::differentiable::Differentiable;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use std::fmt;

/// The zero space `ℝ⁰`, with exactly one value: the trivial tangent of the
/// discrete sorts. Its own tangent is itself.
///
/// Python names it `sort.ZeroSort`, to tell it apart from `Zero`, the
/// operation writing its only value, [`zero::Zero`](crate::zero::Zero).
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash, Default)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, name = "ZeroSort", module = "reactor.sort")
)]
pub struct Zero;

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Zero {
    #[cfg_attr(feature = "pyo3", new)]
    pub fn new() -> Self {
        Zero
    }

    /// The tangent sort `T(self)`: the sort of the rate of change of its
    /// values.
    #[cfg(feature = "pyo3")]
    #[pyo3(name = "tangent")]
    fn tangent(&self) -> Zero {
        Differentiable::tangent(self)
    }

    /// Python writes it by its Python name, `ZeroSort()`, as it builds it.
    #[cfg(feature = "pyo3")]
    fn __str__(&self) -> &'static str {
        "ZeroSort()"
    }
}

impl fmt::Display for Zero {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Zero()")
    }
}

impl Differentiable for Zero {
    type Tangent = Zero;

    fn tangent(self) -> Zero {
        Zero
    }
}

/// The sorts of all theories: a sort of the matrix theories, a sort of the
/// scalar theories, or [`Zero`], which both share.
///
/// The conversion from the matrix sorts sends their `Zero` to [`Any::Zero`],
/// so that, built through it, [`Any::Matrix`] holds no `Zero`; the scalar
/// sorts hold none.
#[derive(Clone, PartialEq, Debug, Eq)]
pub enum Any {
    Matrix(crate::matrix::sort::Any),
    Scalar(crate::scalar::sort::Any),
    Zero(Zero),
}

impl From<crate::matrix::sort::Any> for Any {
    fn from(s: crate::matrix::sort::Any) -> Self {
        match s {
            crate::matrix::sort::Any::Zero(z) => Any::Zero(z),
            s => Any::Matrix(s),
        }
    }
}

impl From<crate::scalar::sort::Any> for Any {
    fn from(s: crate::scalar::sort::Any) -> Self {
        Any::Scalar(s)
    }
}

impl From<crate::matrix::sort::Real> for Any {
    fn from(s: crate::matrix::sort::Real) -> Self {
        crate::matrix::sort::Any::from(s).into()
    }
}

impl From<crate::matrix::sort::Int> for Any {
    fn from(s: crate::matrix::sort::Int) -> Self {
        crate::matrix::sort::Any::from(s).into()
    }
}

impl From<crate::matrix::sort::Bool> for Any {
    fn from(s: crate::matrix::sort::Bool) -> Self {
        crate::matrix::sort::Any::from(s).into()
    }
}

impl From<crate::matrix::sort::Word> for Any {
    fn from(s: crate::matrix::sort::Word) -> Self {
        crate::matrix::sort::Any::from(s).into()
    }
}

impl From<crate::scalar::sort::Enum> for Any {
    fn from(s: crate::scalar::sort::Enum) -> Self {
        crate::scalar::sort::Any::from(s).into()
    }
}

impl From<crate::scalar::sort::Event> for Any {
    fn from(s: crate::scalar::sort::Event) -> Self {
        crate::scalar::sort::Any::from(s).into()
    }
}

impl From<Zero> for Any {
    fn from(z: Zero) -> Self {
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
/// theories, a real of one degree higher or [`Zero`], and [`Zero`] for the
/// scalar sorts, all discrete.
impl Differentiable for Any {
    type Tangent = Any;

    fn tangent(self) -> Any {
        match self {
            Any::Matrix(s) => s.tangent().into(),
            Any::Scalar(_) => Any::Zero(Zero),
            Any::Zero(z) => z.tangent().into(),
        }
    }
}
