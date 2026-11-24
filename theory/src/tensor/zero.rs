//! The structural operation [`Zero`]: the rate of change of a value that does
//! not change.

use super::sort;
use base::boundary::{Boundary, Poly};
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use std::fmt;

/// Writes the only value of the singleton sort [`sort::Zero`], reading
/// nothing: the rate of change of a value that does not change.
///
/// The zero rate of change of a real matrix is not this, but the zero
/// matrix of the degree of the rate of change, [`real::Zero`](super::real::Zero).
#[derive(Clone, Copy, PartialEq, Debug, Eq, Default)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.zero")
)]
pub struct Zero;

/// Writes the only value of the singleton sort [`sort::Zero`].
impl Boundary<sort::Zero> for Zero {
    fn dom(&self) -> Poly<sort::Zero> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<sort::Zero> {
        base::mono![sort::Zero].into()
    }
}

impl fmt::Display for Zero {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Zero")
    }
}

impl Zero {
    /// The operation writing the only value of the sort of `cod`.
    pub fn with_cod(cod: [sort::Zero; 1]) -> Result<Self, crate::Error> {
        let [_] = cod;
        Ok(Zero)
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
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }
}
