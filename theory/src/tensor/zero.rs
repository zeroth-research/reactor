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

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Zero {
    #[cfg_attr(feature = "pyo3", new)]
    pub fn new() -> Self {
        Zero
    }
}
