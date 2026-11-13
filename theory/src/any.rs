use crate::bv::BV;
use crate::lia::LIA;
use crate::lra::LRA;
use crate::{bv, lia, lra};
use base::boundary::{Boundary, Mono, Poly};
use base::tangent::Tangent;
use derive_more::From;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use std::fmt;
use subenum::subenum;

// `pyclass` does not see a `cfg_attr` on a variant (inner `cfg_attr`s are not
// evaluated before attribute macros), so the variant attributes that only exist
// with the `pyo3` feature are passed in by the invocations below.
macro_rules! sort {
    ($(#[$real:meta])*) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        #[cfg_attr(feature = "pyo3", pyclass(frozen, eq, str))]
        pub enum Sort {
            Bool([usize; 2]),
            // a real of the given shape and degree: a value at degree 0, its
            // derivative of that order (the tangent of the degree below) at degree >= 1
            $(#[$real])*
            Real([usize; 2], u8),
            Int([usize; 2]),
            BitVec(usize, [usize; 2]),
            // the tangent of `Bool`, `Int`, and `BitVec` (they have only the zero derivative)
            Zero([usize; 2]),
        }
    };
}

// in Python, `Real(shape)` is a value: the degree (`_1`) defaults to 0
#[cfg(feature = "pyo3")]
sort!(#[pyo3(constructor = (_0, _1 = 0))]);
#[cfg(not(feature = "pyo3"))]
sort!();

impl fmt::Display for Sort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sort::Bool(shape) => {
                write!(f, "Bool([{},{}])", shape[0], shape[1])
            }
            Sort::Real(shape, 0) => {
                write!(f, "Real([{},{}])", shape[0], shape[1])
            }
            Sort::Real(shape, degree) => {
                write!(f, "Real([{},{}], {})", shape[0], shape[1], degree)
            }
            Sort::Int(shape) => {
                write!(f, "Int([{},{}])", shape[0], shape[1])
            }
            Sort::BitVec(bw, shape) => {
                write!(f, "Bv{}([{},{}])", bw, shape[0], shape[1])
            }
            Sort::Zero(shape) => {
                write!(f, "Zero([{},{}])", shape[0], shape[1])
            }
        }
    }
}

impl Tangent for Sort {
    fn tangent(&self) -> Self {
        match self {
            Sort::Real(shape, degree) => Sort::Real(*shape, degree + 1),
            Sort::Bool(shape) | Sort::Int(shape) | Sort::BitVec(_, shape) | Sort::Zero(shape) => {
                Sort::Zero(*shape)
            }
        }
    }
}

impl From<bv::Sort> for Sort {
    fn from(value: bv::Sort) -> Self {
        match value {
            bv::Sort::BV(bw, shape) => Sort::BitVec(bw, shape),
            bv::Sort::Zero(shape) => Sort::Zero(shape),
        }
    }
}

impl From<lia::Sort> for Sort {
    fn from(value: lia::Sort) -> Self {
        match value {
            lia::Sort::Int(shape) => Sort::Int(shape),
            lia::Sort::Bool(shape) => Sort::Bool(shape),
            lia::Sort::Zero(shape) => Sort::Zero(shape),
        }
    }
}

impl From<lra::Sort> for Sort {
    fn from(value: lra::Sort) -> Self {
        match value {
            lra::Sort::Bool(shape) => Sort::Bool(shape),
            lra::Sort::Real(shape, degree) => Sort::Real(shape, degree),
            lra::Sort::Zero(shape) => Sort::Zero(shape),
        }
    }
}

impl TryFrom<Sort> for bv::Sort {
    type Error = String;

    fn try_from(value: Sort) -> Result<Self, Self::Error> {
        match value {
            Sort::BitVec(bw, shape) => Ok(bv::Sort::BV(bw, shape)),
            Sort::Zero(shape) => Ok(bv::Sort::Zero(shape)),
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
            Sort::Zero(shape) => Ok(lia::Sort::Zero(shape)),
            _ => Err("invalid cast".to_string()),
        }
    }
}

impl TryFrom<Sort> for lra::Sort {
    type Error = String;

    fn try_from(value: Sort) -> Result<Self, Self::Error> {
        match value {
            Sort::Bool(shape) => Ok(lra::Sort::Bool(shape)),
            Sort::Real(shape, degree) => Ok(lra::Sort::Real(shape, degree)),
            Sort::Zero(shape) => Ok(lra::Sort::Zero(shape)),
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

// The distinguished operations are unary, matching the `Id`, `Any*`/`Havoc`, and
// `*Zerograd` operations of the base theories: `HAVOC` writes any value of its range
// (`I -> range`), `SKIP` copies a value of its range (`range -> range`), and `ZERO` writes
// the zero derivative of its range (`I -> range`). The operations of the base theories
// keep their boundaries, over the sorts of `Any` (see `embed`).

/// The boundary `range` of a single wire
fn wire(range: Sort) -> Poly<Sort> {
    Mono::from_iter([range]).into()
}

/// The type `ty` over the sorts of a base theory as a type over the sorts of `Any`
fn embed<S: Into<Sort>>(ty: Poly<S>) -> Poly<Sort> {
    ty.into_iter()
        .map(|m| m.into_iter().map(Into::into).collect::<Mono<Sort>>())
        .collect()
}

impl Boundary<Sort> for Any {
    fn dom(&self) -> Poly<Sort> {
        match self {
            Any::HAVOC(_) | Any::ZERO(_) => Poly::unit(),
            Any::SKIP(range) => wire(*range),
            Any::LRA(op) => embed(op.dom()),
            Any::LIA(op) => embed(op.dom()),
            Any::BV(op) => embed(op.dom()),
        }
    }

    fn cod(&self) -> Poly<Sort> {
        match self {
            Any::HAVOC(range) | Any::SKIP(range) | Any::ZERO(range) => wire(*range),
            Any::LRA(op) => embed(op.cod()),
            Any::LIA(op) => embed(op.cod()),
            Any::BV(op) => embed(op.cod()),
        }
    }
}

impl Boundary<Sort> for Sequential {
    fn dom(&self) -> Poly<Sort> {
        match self {
            Sequential::HAVOC(_) => Poly::unit(),
            Sequential::SKIP(range) => wire(*range),
            Sequential::LRA(op) => embed(op.dom()),
            Sequential::LIA(op) => embed(op.dom()),
            Sequential::BV(op) => embed(op.dom()),
        }
    }

    fn cod(&self) -> Poly<Sort> {
        match self {
            Sequential::HAVOC(range) | Sequential::SKIP(range) => wire(*range),
            Sequential::LRA(op) => embed(op.cod()),
            Sequential::LIA(op) => embed(op.cod()),
            Sequential::BV(op) => embed(op.cod()),
        }
    }
}

impl Boundary<Sort> for Combinatorial {
    fn dom(&self) -> Poly<Sort> {
        match self {
            Combinatorial::HAVOC(_) => Poly::unit(),
            Combinatorial::LRA(op) => embed(op.dom()),
            Combinatorial::LIA(op) => embed(op.dom()),
            Combinatorial::BV(op) => embed(op.dom()),
        }
    }

    fn cod(&self) -> Poly<Sort> {
        match self {
            Combinatorial::HAVOC(range) => wire(*range),
            Combinatorial::LRA(op) => embed(op.cod()),
            Combinatorial::LIA(op) => embed(op.cod()),
            Combinatorial::BV(op) => embed(op.cod()),
        }
    }
}

impl Boundary<Sort> for Differential {
    fn dom(&self) -> Poly<Sort> {
        match self {
            Differential::ZERO(_) => Poly::unit(),
            Differential::LRA(op) => embed(op.dom()),
        }
    }

    fn cod(&self) -> Poly<Sort> {
        match self {
            Differential::ZERO(range) => wire(*range),
            Differential::LRA(op) => embed(op.cod()),
        }
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
//
#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroU8;

    // Formats the boundary of `op` as `dom -> cod`
    fn show(op: impl Boundary<Sort>) -> String {
        format!("{} -> {}", op.dom(), op.cod())
    }

    // Casts up the hierarchy (Into) wrap the op in the right variant.

    #[test]
    fn base_theories_cast_up() {
        let (lra, lia, bv) = (|| LRA::Add(1, 1), || LIA::Add(1, 1), || BV::And(8, 1, 1));
        assert!(matches!(lra().into(), Combinatorial::LRA(LRA::Add(1, 1))));
        assert!(matches!(lra().into(), Sequential::LRA(LRA::Add(1, 1))));
        assert!(matches!(lra().into(), Differential::LRA(LRA::Add(1, 1))));
        assert!(matches!(lia().into(), Combinatorial::LIA(LIA::Add(1, 1))));
        assert!(matches!(lia().into(), Sequential::LIA(LIA::Add(1, 1))));
        assert!(matches!(bv().into(), Combinatorial::BV(BV::And(8, 1, 1))));
        assert!(matches!(bv().into(), Sequential::BV(BV::And(8, 1, 1))));
    }

    #[test]
    fn catch_alls_cast_up() {
        let comb: Sequential = Combinatorial::LRA(LRA::Add(1, 1)).into();
        assert!(matches!(comb, Sequential::LRA(LRA::Add(1, 1))));
        let t = Sort::Real([1, 1], 0);
        assert!(matches!(
            Combinatorial::HAVOC(t).into(),
            Sequential::HAVOC(_)
        ));
        assert!(matches!(Sequential::SKIP(t).into(), Any::SKIP(_)));
        assert!(matches!(Differential::ZERO(t).into(), Any::ZERO(_)));
    }

    // Casts down the hierarchy (TryInto) recover the op or reject it.

    #[test]
    fn display_is_stable() {
        // Elements print their name, member ops delegate to the base theory.
        let t = Sort::Real([1, 1], 0);
        assert_eq!(Any::HAVOC(t).to_string(), "HAVOC");
        assert_eq!(Sequential::SKIP(t).to_string(), "SKIP");
        assert_eq!(Differential::ZERO(t).to_string(), "ZERO");
        assert_eq!(Any::LRA(LRA::Add(1, 1)).to_string(), "Add");
        assert_eq!(Sequential::LIA(LIA::ReLU(1, 1)).to_string(), "ReLU");
        assert_eq!(
            Combinatorial::BV(BV::MatMul(8, 1, 1, 1)).to_string(),
            "MatMul"
        );
        // Base-theory special cases keep their formats.
        assert_eq!(
            Any::LRA(LRA::UninterpretedReal("f".to_string(), 1, 1)).to_string(),
            "UninterpretedReal(f)"
        );
        let bit_select = BV::BitSelect {
            bits: bv::Bits::new(7, 0, 8).unwrap(),
            rows: 1,
            cols: 1,
        };
        assert_eq!(Any::BV(bit_select).to_string(), "BitSelect[7:0]");
    }

    #[test]
    fn any_casts_down_or_rejects() {
        let t = Sort::Real([1, 1], 0);
        assert!(matches!(Any::SKIP(t).try_into(), Ok(Sequential::SKIP(_))));
        assert!(matches!(
            Any::HAVOC(t).try_into(),
            Ok(Combinatorial::HAVOC(_))
        ));
        assert!(matches!(Any::ZERO(t).try_into(), Ok(Differential::ZERO(_))));
        // ZERO is differential-only; SKIP is not combinatorial or differential.
        assert!(Sequential::try_from(Any::ZERO(t)).is_err());
        assert!(Combinatorial::try_from(Any::SKIP(t)).is_err());
        assert!(Differential::try_from(Any::SKIP(t)).is_err());
    }

    // The distinguished elements are unary at every level of the hierarchy,
    // matching the base theories' `Id`/`Havoc`/`Zero`.

    #[test]
    fn skip_is_unary() {
        // copies a single wire of its range
        let t = Sort::Real([1, 1], 0);
        assert_eq!(show(Any::SKIP(t)), "Real([1,1]) -> Real([1,1])");
        assert_eq!(show(Sequential::SKIP(t)), "Real([1,1]) -> Real([1,1])");
    }

    #[test]
    fn havoc_is_unary() {
        // no reads, exactly one write
        let t = Sort::Real([1, 1], 0);
        assert_eq!(show(Any::HAVOC(t)), "I -> Real([1,1])");
        assert_eq!(show(Sequential::HAVOC(t)), "I -> Real([1,1])");
        assert_eq!(show(Combinatorial::HAVOC(t)), "I -> Real([1,1])");
    }

    #[test]
    fn zero_is_unary() {
        // no reads, exactly one write (of the derivative)
        let t = Sort::Real([1, 1], 0).tangent();
        assert_eq!(show(Any::ZERO(t)), "I -> Real([1,1], 1)");
        assert_eq!(show(Differential::ZERO(t)), "I -> Real([1,1], 1)");
    }

    #[test]
    fn base_theories_keep_their_boundaries() {
        // the operations of the base theories keep their boundaries, over the sorts of `Any`
        assert_eq!(
            show(Any::LRA(LRA::Add(1, 1))),
            "Real([1,1]) ⊗ Real([1,1]) -> Real([1,1])"
        );
        assert_eq!(
            show(Sequential::LIA(LIA::Lt(1, 1))),
            "Int([1,1]) ⊗ Int([1,1]) -> Bool([1,1])"
        );
        assert_eq!(
            show(Combinatorial::BV(BV::Not(8, 1, 1))),
            "Bv8([1,1]) -> Bv8([1,1])"
        );
        assert_eq!(
            show(Differential::LRA(LRA::TAdd(1, 1, NonZeroU8::MIN))),
            "Real([1,1], 1) ⊗ Real([1,1], 1) -> Real([1,1], 1)"
        );
        assert_eq!(show(Any::LRA(LRA::BoolZerograd(1, 1))), "I -> Zero([1,1])");
    }
}
