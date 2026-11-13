use crate::bv::BV;
use crate::lia::LIA;
use crate::lra::LRA;
use crate::{bv, lia, lra};
use derive_more::From;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use std::fmt;
use subenum::subenum;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "pyo3", pyclass(frozen, eq, str))]
pub enum Sort {
    Bool([usize; 2]),
    Real([usize; 2]),
    Int([usize; 2]),
    BitVec(usize, [usize; 2]),
}

impl fmt::Display for Sort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sort::Bool(shape) => {
                write!(f, "Bool([{},{}])", shape[0], shape[1])
            }
            Sort::Real(shape) => {
                write!(f, "Real([{},{}])", shape[0], shape[1])
            }
            Sort::Int(shape) => {
                write!(f, "Int([{},{}])", shape[0], shape[1])
            }
            Sort::BitVec(bw, shape) => {
                write!(f, "Bv{}([{},{}])", bw, shape[0], shape[1])
            }
        }
    }
}

impl From<bv::Sort> for Sort {
    fn from(value: bv::Sort) -> Self {
        match value {
            bv::Sort::BV(bw, shape) => Sort::BitVec(bw, shape),
        }
    }
}

impl From<lia::Sort> for Sort {
    fn from(value: lia::Sort) -> Self {
        match value {
            lia::Sort::Int(shape) => Sort::Int(shape),
            lia::Sort::Bool(shape) => Sort::Bool(shape),
        }
    }
}

impl From<lra::Sort> for Sort {
    fn from(value: lra::Sort) -> Self {
        match value {
            lra::Sort::Bool(shape) => Sort::Bool(shape),
            lra::Sort::Real(shape) => Sort::Real(shape),
        }
    }
}

impl TryFrom<Sort> for bv::Sort {
    type Error = String;

    fn try_from(value: Sort) -> Result<Self, Self::Error> {
        match value {
            Sort::BitVec(bw, shape) => Ok(bv::Sort::BV(bw, shape)),
            _ => Err("invalid cast".to_string()),
        }
    }
}

impl TryFrom<Sort> for lia::Sort {
    type Error = String;

    fn try_from(value: Sort) -> Result<Self, Self::Error> {
        match value {
            Sort::Bool(shape) => Ok(lia::Sort::Bool(shape)),
            Sort::Int(shape) => Ok(lia::Sort::Int(shape)),
            _ => Err("invalid cast".to_string()),
        }
    }
}

impl TryFrom<Sort> for lra::Sort {
    type Error = String;

    fn try_from(value: Sort) -> Result<Self, Self::Error> {
        match value {
            Sort::Bool(shape) => Ok(lra::Sort::Bool(shape)),
            Sort::Real(shape) => Ok(lra::Sort::Real(shape)),
            _ => Err("invalid cast".to_string()),
        }
    }
}

//============================================================
// The cast lattice
//============================================================
//
// Casts between theories follow the hierarchy
//
// ```text
//     LRA -> Combinatorial, Sequential, Differential
//     LIA -> Combinatorial, Sequential
//     BV  -> Combinatorial, Sequential
//     Combinatorial -> Sequential
//     Sequential    -> Any
//     Differential  -> Any
// ```
//
// with `From` going up. Downward, only the casts from `Any` to the
// catch-alls exist (they serve the Python boundary, which builds typed
// blocks from `Any` terms); casting down to the base theories is unused and
// intentionally not implemented.
//
// `subenum` defines the catch-alls as tagged subsets of `Any` -- generating
// the sub-enums themselves, the casts between each subset and `Any`, and
// (via the propagated derives) the `From` wraps of base-theory ops and the
// `Display` impls.

#[subenum(
    Combinatorial(cfg_attr(feature = "pyo3", pyclass)),
    Differential(cfg_attr(feature = "pyo3", pyclass)),
    Sequential(cfg_attr(feature = "pyo3", pyclass))
)]
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, From, strum::Display)]
pub enum Any {
    #[subenum(Combinatorial, Sequential)]
    #[from(skip)]
    HAVOC(Sort),
    #[subenum(Sequential)]
    #[from(skip)]
    SKIP(Sort),
    #[subenum(Differential)]
    #[from(skip)]
    ZERO(Sort),
    #[subenum(Combinatorial, Differential, Sequential)]
    #[strum(to_string = "{0}")]
    LRA(LRA),
    #[subenum(Combinatorial, Sequential)]
    #[strum(to_string = "{0}")]
    LIA(LIA),
    #[subenum(Combinatorial, Sequential)]
    #[strum(to_string = "{0}")]
    BV(BV),
}

#[cfg(feature = "pyo3")]
impl<'py> FromPyObject<'py> for Any {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(a) = obj.extract::<LRA>() {
            return Ok(Any::LRA(a));
        }
        if let Ok(a) = obj.extract::<LIA>() {
            return Ok(Any::LIA(a));
        }
        if let Ok(a) = obj.extract::<BV>() {
            return Ok(Any::BV(a));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "expected one of LRA, LIA, or BV",
        ))
    }
}

#[cfg(feature = "pyo3")]
impl<'py> IntoPyObject<'py> for Any {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        match self {
            Any::HAVOC(a) => Combinatorial::HAVOC(a)
                .into_pyobject(py)
                .map(Bound::into_any),
            Any::SKIP(a) => Sequential::SKIP(a).into_pyobject(py).map(Bound::into_any),
            Any::ZERO(a) => Differential::ZERO(a).into_pyobject(py).map(Bound::into_any),
            Any::LRA(a) => a.into_pyobject(py).map(Bound::into_any),
            Any::LIA(a) => a.into_pyobject(py).map(Bound::into_any),
            Any::BV(a) => a.into_pyobject(py).map(Bound::into_any),
        }
    }
}

impl crate::Sequential<Sort> for Sequential {
    fn skip(range: &Sort) -> Self {
        Sequential::SKIP(*range)
    }
}

impl crate::Combinatorial<Sort> for Sequential {
    fn havoc(range: &Sort) -> Self {
        Sequential::HAVOC(*range)
    }
}

impl crate::Combinatorial<Sort> for Combinatorial {
    fn havoc(range: &Sort) -> Self {
        Combinatorial::HAVOC(*range)
    }
}

impl crate::Differential<Sort> for Differential {
    fn zero(range: &Sort) -> Self {
        Differential::ZERO(*range)
    }
}

// The one cast `subenum` cannot express: the embedding between two
// sub-enums. Its inverse is intentionally absent -- nothing needs to cast
// down from `Sequential` to `Combinatorial`.
impl From<Combinatorial> for Sequential {
    fn from(op: Combinatorial) -> Self {
        match op {
            Combinatorial::HAVOC(range) => Sequential::HAVOC(range),
            Combinatorial::BV(bv) => Sequential::BV(bv),
            Combinatorial::LRA(lra) => Sequential::LRA(lra),
            Combinatorial::LIA(lia) => Sequential::LIA(lia),
        }
    }
}
