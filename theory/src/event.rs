/*!
# Events

Defines the sort [`Event`] of events, and one type per operation on it. An event
is encoded as a boolean whose value is whether the event happens in the step
that writes it: set when it happens, reset when it does not.

- [`Set`] — the event happening: writes `1`.
- [`Reset`] — the event not happening: writes `0`.
- [`Test`] — whether the event happens: whether it is set, `= 1`, writing a
  [`Bool`] of the shape `[1, 1]`.
- [`Ite`] — the choice `if c then x else y` between two events, by a
  [`Bool`] `c` of the shape `[1, 1]`: the event set when `c` and `x`, or when
  not `c` and `y`.

The sort has no parameters, so neither do the operations: each is a unit
value, determined by its type alone, and its boundary is fixed. In SMT-LIB2,
an event is a `Bool`, and its operations `true`, `false`, the identity and
`ite`.

## Open issue: `Skip` repeats an event

This encoding is not yet sound under `Skip`. An event that happened in one step
is set, and `Skip`, the identity of every sort, keeps its value, so that the
event happens again in the next step, and in every step after it, until it is
reset. This is the case wherever the next value of an event is its latched
value: returned as is by an update, in a step in which the module that owns it
does not update, and while values are held, as in a jump or a flow. An event
should instead happen only in the step in which it is set, and be reset
wherever it is not written, which `Skip` does not do; this needs to be
addressed, by `Skip` on events, or by the encoding.
*/

pub use crate::Any as Sort;
use crate::bool::Bool;
use crate::zero::Real0;
use base::boundary::{Boundary, Poly};
use base::differentiable::Differentiable;
use base::mono;
use derive_more::From;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use std::fmt;

/// The sort of events: a value that, as a boolean, is whether the event
/// happens in the step that writes it, set when it does and reset when it does
/// not.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash, Default)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.event")
)]
pub struct Event;

/// Python builds it from nothing, and matches it by nothing.
#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Event {
    #[cfg_attr(feature = "pyo3", new)]
    pub fn new() -> Self {
        Event
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__(py: Python<'_>) -> Bound<'_, pyo3::types::PyTuple> {
        pyo3::types::PyTuple::empty(py)
    }

    /// The tangent sort `T(self)`: the sort of the rate of change of its
    /// values.
    #[cfg(feature = "pyo3")]
    #[pyo3(name = "tangent")]
    fn tangent(&self) -> Real0 {
        Differentiable::tangent(self)
    }
}

impl fmt::Display for Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Event()")
    }
}

/// Events are discrete, so their tangent is trivial: the singleton sort
/// [`Real0`].
impl Differentiable for Event {
    type Tangent = Real0;

    fn tangent(self) -> Real0 {
        Real0
    }
}

/// The event happening: set, `1`.
///
/// SMT-LIB2: `true`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.event")
)]
pub struct Set;

impl Boundary<Sort> for Set {
    fn dom(&self) -> Poly<Sort> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Event.into()].into()
    }
}

impl fmt::Display for Set {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Set")
    }
}

/// The event not happening: reset, `0`.
///
/// SMT-LIB2: `false`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.event")
)]
pub struct Reset;

impl Boundary<Sort> for Reset {
    fn dom(&self) -> Poly<Sort> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Event.into()].into()
    }
}

impl fmt::Display for Reset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Reset")
    }
}

/// Whether the event `x` happens: whether it is set, `x = 1`, writing a
/// [`Bool`] of the shape `[1, 1]`.
///
/// SMT-LIB2: `x`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.event")
)]
pub struct Test;

impl Boundary<Sort> for Test {
    fn dom(&self) -> Poly<Sort> {
        mono![Event.into()].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new_unchecked([1, 1]).into()].into()
    }
}

impl fmt::Display for Test {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Test")
    }
}

/// Choice `if c then x else y` between two events `x` and `y`, by the boolean
/// `c`, of the shape `[1, 1]`, read first: a conditional set or reset.
///
/// SMT-LIB2: `(ite c x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.event")
)]
pub struct Ite;

impl Boundary<Sort> for Ite {
    fn dom(&self) -> Poly<Sort> {
        mono![
            Bool::new_unchecked([1, 1]).into(),
            Event.into(),
            Event.into()
        ]
        .into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Event.into()].into()
    }
}

impl fmt::Display for Ite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ite")
    }
}

/// The operations on events, with their boundary in [`Sort`].
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash, From)]
pub enum Op {
    Set(Set),
    Reset(Reset),
    Test(Test),
    Ite(Ite),
}

impl Boundary<Sort> for Op {
    fn dom(&self) -> Poly<Sort> {
        match self {
            Op::Set(op) => op.dom(),
            Op::Reset(op) => op.dom(),
            Op::Test(op) => op.dom(),
            Op::Ite(op) => op.dom(),
        }
    }

    fn cod(&self) -> Poly<Sort> {
        match self {
            Op::Set(op) => op.cod(),
            Op::Reset(op) => op.cod(),
            Op::Test(op) => op.cod(),
            Op::Ite(op) => op.cod(),
        }
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::Set(op) => op.fmt(f),
            Op::Reset(op) => op.fmt(f),
            Op::Test(op) => op.fmt(f),
            Op::Ite(op) => op.fmt(f),
        }
    }
}

// The Python constructors of the operations: each has no parameters and a
// boundary its type fixes, so its domain and codomain are optional, and
// checked when given; each matches by them.

#[cfg(feature = "pyo3")]
#[pymethods]
impl Set {
    /// Python builds it from nothing; its domain `dom` and codomain
    /// `cod`, when given, must be the ones it reads and writes.
    #[new]
    #[pyo3(signature = (dom = None, cod = None))]
    fn py_new(dom: Option<&Bound<'_, PyAny>>, cod: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        crate::check_boundary(Set, dom, cod)
    }

    /// Its domain, then its codomain.
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[getter(dom)]
    fn py_dom<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[getter(cod)]
    fn py_cod<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_sorts(py, Boundary::cod(self))
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl Reset {
    /// Python builds it from nothing; its domain `dom` and codomain
    /// `cod`, when given, must be the ones it reads and writes.
    #[new]
    #[pyo3(signature = (dom = None, cod = None))]
    fn py_new(dom: Option<&Bound<'_, PyAny>>, cod: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        crate::check_boundary(Reset, dom, cod)
    }

    /// Its domain, then its codomain.
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[getter(dom)]
    fn py_dom<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[getter(cod)]
    fn py_cod<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_sorts(py, Boundary::cod(self))
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl Test {
    /// Python builds it from nothing; its domain `dom` and codomain
    /// `cod`, when given, must be the ones it reads and writes.
    #[new]
    #[pyo3(signature = (dom = None, cod = None))]
    fn py_new(dom: Option<&Bound<'_, PyAny>>, cod: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        crate::check_boundary(Test, dom, cod)
    }

    /// Its domain, then its codomain.
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[getter(dom)]
    fn py_dom<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[getter(cod)]
    fn py_cod<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_sorts(py, Boundary::cod(self))
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl Ite {
    /// Python builds it from nothing; its domain `dom` and codomain
    /// `cod`, when given, must be the ones it reads and writes.
    #[new]
    #[pyo3(signature = (dom = None, cod = None))]
    fn py_new(dom: Option<&Bound<'_, PyAny>>, cod: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        crate::check_boundary(Ite, dom, cod)
    }

    /// Its domain, then its codomain.
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[getter(dom)]
    fn py_dom<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[getter(cod)]
    fn py_cod<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_sorts(py, Boundary::cod(self))
    }
}

/// Extracts an operation on events from an instance of its class.
#[cfg(feature = "pyo3")]
impl<'a, 'py> FromPyObject<'a, 'py> for Op {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(op) = ob.extract::<Set>() {
            return Ok(Op::Set(op));
        }
        if let Ok(op) = ob.extract::<Reset>() {
            return Ok(Op::Reset(op));
        }
        if let Ok(op) = ob.extract::<Test>() {
            return Ok(Op::Test(op));
        }
        if let Ok(op) = ob.extract::<Ite>() {
            return Ok(Op::Ite(op));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "expected an operation on events, got {}",
            ob.get_type()
        )))
    }
}

/// Converts an operation on events into an instance of its class.
#[cfg(feature = "pyo3")]
impl<'py> IntoPyObject<'py> for Op {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        match self {
            Op::Set(op) => op.into_pyobject(py).map(Bound::into_any),
            Op::Reset(op) => op.into_pyobject(py).map(Bound::into_any),
            Op::Test(op) => op.into_pyobject(py).map(Bound::into_any),
            Op::Ite(op) => op.into_pyobject(py).map(Bound::into_any),
        }
    }
}
