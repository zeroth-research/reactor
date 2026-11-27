/*!
# Events

Defines the sorts [`Event`] of events, and their one operation, [`Xor`]. An
event is encoded as a boolean that flips when the event happens: a step
happens the event when it changes its value, and does not when it keeps it.
Its state alone, the parity of the number of times it happened, means nothing.

The values of an event form a torsor over the booleans: a value is shifted by
a boolean, and two values differ by one, but no boolean is a value. [`Xor`] is
both, chosen by its domain:

- the shift `e ⊕ c`, or `c ⊕ e`, of a value `e` of an event by a [`Bool`] `c`
  of the shape `[1, 1]`, a value of the same event: the event flipped, sent,
  when `c` holds, and kept when it does not;
- the difference `x ⊕ y` of two values of an event, a [`Bool`] of the shape
  `[1, 1]`: of its next and its latched value, `X(e) ⊕ e`, whether it happens.

Every other operation on events is one of these: the flip is `e ⊕ true`, the
comparisons `x ⊕ y` and its negation, and the choice between two values of an
event, `e ⊕ a` and `e ⊕ b`, the shift `e ⊕ ite(c, a, b)`. The event kept, so
that none happens, is the operation `Skip` of every sort, which keeps a value
in a step that does not write it.

## One sort per event

Each event is its own sort, with a fresh identity, as a wire is: two events
are never of the same sort. [`Xor`] reads values of a single event, so that its
values meet only those of the same event. This makes an event memoryless by its
typing: no value of an event is made of booleans alone, nor compared with,
chosen against or copied into another event, so that its state, which would
carry the parity of its past, is never observed but through its own change,
`X(e) ⊕ e`, and no event is flipped but by the step that owns it.

In SMT-LIB2, an event is a `Bool`, and [`Xor`] is `xor`.
*/

pub use crate::Any as Sort;
use crate::Events;
use crate::bool::Bool;
use crate::zero::Real0;
use base::boundary::{Boundary, Poly};
use base::differentiable::Differentiable;
use base::mono;
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

/// The exclusive or `x ⊕ y` of a value of an event with a value of the same
/// event, writing their difference, a [`Bool`] of the shape `[1, 1]`, or with
/// such a boolean, in either order, writing the event shifted by it.
///
/// SMT-LIB2: `(xor x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.event")
)]
pub struct Xor {
    dom: [Events; 2],
    cod: Events,
}

impl Xor {
    /// The exclusive or reading the sorts of `dom`: two values of the same
    /// event, for their difference, or a value of an event and a boolean of the
    /// shape `[1, 1]`, in either order, for its shift. Fails otherwise.
    pub fn with_dom(dom: [Sort; 2]) -> Result<Self, crate::Error> {
        let cod = match &dom {
            // the difference of two values of an event
            [Sort::Event(x), Sort::Event(y)] if x == y => Bool::new_unchecked([1, 1]).into(),
            // the shift of an event by a boolean
            [Sort::Event(e), Sort::Bool(c)] | [Sort::Bool(c), Sort::Event(e)]
                if c.shape() == [1, 1] =>
            {
                (*e).into()
            }
            _ => {
                return Err(crate::unexpected_dom(
                    "two values of the same event, or an event and a Bool([1, 1]) in either order",
                    &dom,
                ));
            }
        };
        let dom = dom.map(|s| Events::try_from(s).expect("a sort of an event or a boolean"));
        Ok(Xor { dom, cod })
    }
}

impl Boundary<Sort> for Xor {
    fn dom(&self) -> Poly<Sort> {
        mono![self.dom[0].into(), self.dom[1].into()].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![self.cod.into()].into()
    }
}

impl fmt::Display for Xor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Xor")
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl Xor {
    /// Python builds it from its domain `dom`, two values of the same event, or
    /// a value of an event and a boolean of the shape `[1, 1]`, in either
    /// order; its codomain `cod`, when given, must be the one it writes.
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Sort; 2], cod: Option<Sort>) -> Result<Self, crate::Error> {
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
