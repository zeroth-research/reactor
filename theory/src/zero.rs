/*!
# The zero operation

[`Zero`], the operation writing the only value of the sort [`Real0`],
shared by all theories: the rate of change of a value that does not change.

## The zero sort

[`Real0`] is `ℝ⁰`, the zero vector space: it has exactly one value, the
vector of no coordinates. It is the tangent of every discrete sort, of booleans,
integers, words and enumerations, scalar or matrix alike: a discrete set is a
manifold of dimension 0, whose tangent space at every point is `ℝ⁰`, whatever
the shape of its values, as a matrix of discrete values is a discrete set
too. So it has no shape, and the matrix theories and the scalar theories
share it. It is its own tangent.

It is named after `ℝ⁰`, the real space of dimension 0, not after a degree:
it is not a [`Real`](crate::real::Real) of degree 0, which is a plain
real. Nor is any `Real` the space `ℝ⁰`: the matrices of every sort have at
least one row and one column, as a matrix of no entries, `ℝ^{0×n}` or
`ℝ^{n×0}`, is `ℝ⁰` again, and `Real0` is its only sort, with neither a shape
nor a degree, so that it is its own tangent.

The zero rate of change of a real is not the value of [`Real0`], but the
zero matrix of the degree of the rate of change, written by
[`real::Zero`](crate::real::Zero).
*/

use base::boundary::{Boundary, Poly};
use base::differentiable::Differentiable;
use derive_more::Display;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use std::fmt;

/// `ℝ⁰`, the zero vector space, with exactly one value: the trivial tangent
/// of the discrete sorts. Its own tangent is itself. Not a
/// [`Real`](crate::real::Real) of degree 0.
///
/// Python finds it in the namespace `zero`, as `zero.Real0`, with
/// [`Zero`], the operation writing its only value.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash, Default)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.zero")
)]
pub struct Real0;

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Real0 {
    #[cfg_attr(feature = "pyo3", new)]
    pub fn new() -> Self {
        Real0
    }

    /// The tangent sort `T(self)`: the sort of the rate of change of its
    /// values.
    #[cfg(feature = "pyo3")]
    #[pyo3(name = "tangent")]
    fn tangent(&self) -> Real0 {
        Differentiable::tangent(self)
    }
}

impl fmt::Display for Real0 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Real0()")
    }
}

impl Differentiable for Real0 {
    type Tangent = Real0;

    fn tangent(self) -> Real0 {
        Real0
    }
}

/// Writes the only value of the sort [`Real0`], reading nothing: the
/// rate of change of a value that does not change.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Default, Display)]
#[display("Zero")]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.zero")
)]
pub struct Zero;

/// Writes the only value of the sort [`Real0`].
impl Boundary<Real0> for Zero {
    fn dom(&self) -> Poly<Real0> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Real0> {
        base::mono![Real0].into()
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Zero {
    /// Python builds it from nothing: its codomain, the singleton sort, carries no
    /// information.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = ())]
    fn py_new() -> Self {
        Zero
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("cod",)
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_boundary(py, Boundary::cod(self))
    }
}
