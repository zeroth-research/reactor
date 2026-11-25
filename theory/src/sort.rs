/*!
# The sorts shared by all theories

[`Zero`] is the zero vector space `ℝ⁰`: it has exactly one value, the vector
of no coordinates. It is the tangent of every discrete sort, of booleans,
integers, words and enumerations, scalar or matrix alike: a discrete set is a
manifold of dimension 0, whose tangent space at every point is `ℝ⁰`, whatever
the shape of its values, as a matrix of discrete values is a discrete set
too. So it has no shape, and the tensor theories and the scalar theories
share it. It is its own tangent.

The zero rate of change of a real is not the value of [`Zero`], but the zero
matrix of the degree of the rate of change, written by
[`tensor::real::Zero`](crate::tensor::real::Zero).
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
