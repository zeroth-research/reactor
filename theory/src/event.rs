/*!
# Events

Defines the sorts [`Event`] of events, and one type per operation on them. An
event is encoded as a boolean that flips when the event happens: a step
happens the event when it changes its value, and does not when it keeps it.
Its state alone, the parity of the number of times it happened, means nothing.

- [`Toggle`] — the event flipped: an event happens.
- [`Eq`](struct@Eq), [`Ne`] — whether two values of an event are the same or
  differ, writing a [`Bool`] of the shape `[1, 1]`: of its next and its latched
  value, `X(e) != e`, whether it happens.
- [`Ite`] — the choice `if c then x else y` between two values of an event,
  by a [`Bool`] `c` of the shape `[1, 1]`: flipped or kept on a condition.

The event kept, so that none happens, is the operation `Skip` of every sort,
which keeps a value in a step that does not write it.

## One sort per event

Each event is its own sort, with a fresh identity, as a wire is: two events
are never of the same sort. The operations read and write values of a single
event, so that its values meet only those of the same event. This makes an
event memoryless by its typing: no value of an event can be compared with,
chosen against or copied into another one, so that its state, which would
carry the parity of its past, is never observed but through its own change,
`X(e) != e`, and no event is flipped but by the step that owns it.

In SMT-LIB2, an event is a `Bool`, and its operations `not`, `=`,
`distinct` and `ite`.
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
use std::sync::atomic::{AtomicUsize, Ordering};

/// The sort of an event: a value that, as a boolean, flips when the event
/// happens. Each event is its own sort, of a process-unique identity: two
/// events are equal exactly when they are the same event, or its copies.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, hash, str, module = "reactor.event")
)]
pub struct Event {
    id: usize,
}

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

/// Python builds it from nothing, a new event, and matches it by nothing.
#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Event {
    /// A new event, of a fresh identity: a sort no other event has. It has no
    /// `Default`, which would make a fresh sort look like a common one.
    #[allow(clippy::new_without_default)]
    #[cfg_attr(feature = "pyo3", new)]
    pub fn new() -> Self {
        Event {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// Its process-unique identity.
    #[cfg_attr(feature = "pyo3", getter)]
    pub fn id(&self) -> usize {
        self.id
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

/// Writes it by its identity, as wires are: `Event#3`.
impl fmt::Display for Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Event#{}", self.id)
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

/// The event `x` flipped: an event happens.
///
/// SMT-LIB2: `(not x)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.event")
)]
pub struct Toggle {
    sort: Event,
}

/// Whether two values `x` and `y` of an event are the same.
///
/// SMT-LIB2: `(= x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.event")
)]
pub struct Eq {
    sort: Event,
}

/// Whether two values `x` and `y` of an event differ: of its next and its
/// latched value, whether it happens.
///
/// SMT-LIB2: `(distinct x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.event")
)]
pub struct Ne {
    sort: Event,
}

/// Choice `if c then x else y` between two values `x` and `y` of an event, by
/// the boolean `c`, of the shape `[1, 1]`, read first.
///
/// SMT-LIB2: `(ite c x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.event")
)]
pub struct Ite {
    sort: Event,
}

impl Toggle {
    /// The flip of the event of `dom`.
    pub fn with_dom(dom: [Event; 1]) -> Result<Self, crate::Error> {
        let [sort] = dom;
        Ok(Toggle { sort })
    }

    /// The flip writing the event of `cod`.
    pub fn with_cod(cod: [Event; 1]) -> Result<Self, crate::Error> {
        let [sort] = cod;
        Ok(Toggle { sort })
    }
}

impl Eq {
    /// The comparison reading the values of `dom`; fails unless they are of
    /// the same event. Its codomain, a [`Bool`], does not determine it.
    pub fn with_dom(dom: [Event; 2]) -> Result<Self, crate::Error> {
        Ok(Eq {
            sort: crate::same_sort(dom)?,
        })
    }
}

impl Ne {
    /// The comparison reading the values of `dom`; fails unless they are of
    /// the same event. Its codomain, a [`Bool`], does not determine it.
    pub fn with_dom(dom: [Event; 2]) -> Result<Self, crate::Error> {
        Ok(Ne {
            sort: crate::same_sort(dom)?,
        })
    }
}

impl Ite {
    /// The choice reading the boolean condition and the branches of `dom`;
    /// fails unless the condition is a boolean of the shape `[1, 1]` and the
    /// branches of the same event.
    pub fn with_dom(dom: [Sort; 3]) -> Result<Self, crate::Error> {
        match dom {
            [Sort::Bool(c), Sort::Event(x), Sort::Event(y)] if c.shape() == [1, 1] && x == y => {
                Ok(Ite { sort: x })
            }
            _ => Err(crate::unexpected_dom(
                "a condition Bool([1, 1]) and two branches of the same event",
                &dom,
            )),
        }
    }

    /// The choice writing the event of `cod`.
    pub fn with_cod(cod: [Event; 1]) -> Result<Self, crate::Error> {
        let [sort] = cod;
        Ok(Ite { sort })
    }
}

impl Boundary<Sort> for Toggle {
    fn dom(&self) -> Poly<Sort> {
        mono![self.sort.into()].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![self.sort.into()].into()
    }
}

impl Boundary<Sort> for Eq {
    fn dom(&self) -> Poly<Sort> {
        mono![self.sort.into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new_unchecked([1, 1]).into()].into()
    }
}

impl Boundary<Sort> for Ne {
    fn dom(&self) -> Poly<Sort> {
        mono![self.sort.into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new_unchecked([1, 1]).into()].into()
    }
}

impl Boundary<Sort> for Ite {
    fn dom(&self) -> Poly<Sort> {
        mono![
            Bool::new_unchecked([1, 1]).into(),
            self.sort.into(),
            self.sort.into()
        ]
        .into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![self.sort.into()].into()
    }
}

impl fmt::Display for Toggle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Toggle")
    }
}

impl fmt::Display for Eq {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Eq")
    }
}

impl fmt::Display for Ne {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ne")
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
    Toggle(Toggle),
    Eq(Eq),
    Ne(Ne),
    Ite(Ite),
}

impl Boundary<Sort> for Op {
    fn dom(&self) -> Poly<Sort> {
        match self {
            Op::Toggle(op) => op.dom(),
            Op::Eq(op) => op.dom(),
            Op::Ne(op) => op.dom(),
            Op::Ite(op) => op.dom(),
        }
    }

    fn cod(&self) -> Poly<Sort> {
        match self {
            Op::Toggle(op) => op.cod(),
            Op::Eq(op) => op.cod(),
            Op::Ne(op) => op.cod(),
            Op::Ite(op) => op.cod(),
        }
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::Toggle(op) => op.fmt(f),
            Op::Eq(op) => op.fmt(f),
            Op::Ne(op) => op.fmt(f),
            Op::Ite(op) => op.fmt(f),
        }
    }
}

// The Python constructors of the operations: each is built from its domain, the
// event it reads, and its codomain, when given, is checked; each matches by
// them.

#[cfg(feature = "pyo3")]
#[pymethods]
impl Toggle {
    /// Python builds it from its domain `dom`, an event; its codomain `cod`,
    /// when given, must be the one it writes.
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: Event, cod: Option<Event>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom([dom]), cod)
    }

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
impl Eq {
    /// Python builds it from its domain `dom`, two values of the same event;
    /// its codomain `cod`, when given, must be the one it writes.
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Event; 2], cod: Option<Sort>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
    }

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
impl Ne {
    /// Python builds it from its domain `dom`, two values of the same event;
    /// its codomain `cod`, when given, must be the one it writes.
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Event; 2], cod: Option<Sort>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
    }

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
    /// Python builds it from its domain `dom`, the condition, a boolean of
    /// the shape `[1, 1]`, and two values of the same event; its codomain
    /// `cod`, when given, must be the one it writes.
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Sort; 3], cod: Option<Sort>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
    }

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
        if let Ok(op) = ob.extract::<Toggle>() {
            return Ok(Op::Toggle(op));
        }
        if let Ok(op) = ob.extract::<Eq>() {
            return Ok(Op::Eq(op));
        }
        if let Ok(op) = ob.extract::<Ne>() {
            return Ok(Op::Ne(op));
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
            Op::Toggle(op) => op.into_pyobject(py).map(Bound::into_any),
            Op::Eq(op) => op.into_pyobject(py).map(Bound::into_any),
            Op::Ne(op) => op.into_pyobject(py).map(Bound::into_any),
            Op::Ite(op) => op.into_pyobject(py).map(Bound::into_any),
        }
    }
}
