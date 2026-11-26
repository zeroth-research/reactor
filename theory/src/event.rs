/*!
# Events

Defines the sort [`Event`] of events, and one type per operation on it. An event is
encoded as a boolean whose change signals that the event happened, so its
state alone means nothing: it is only compared, flipped or kept.

- [`Eq`](struct@Eq), [`Ne`] — whether two events are in the same state or in
  different ones, writing a [`Bool`] of the shape `[1, 1]`: an event
  happened between two of its values when they differ.
- [`Toggle`] — the event flipped: an event happens.
- [`Hold`] — the event kept: no event happens.
- [`Ite`] — the choice `if c then x else y` between two events, by a
  [`Bool`] `c` of the shape `[1, 1]`.

The sort has no parameters, so neither do the operations: each is a unit
value, determined by its type alone, and its boundary is fixed. In SMT-LIB2,
an event is a `Bool`, and its operations `=`, `distinct`, `not`, the
identity and `ite`.
*/

pub use crate::Any as Sort;
use crate::bool::Bool;
use crate::zero::Real0;
use base::boundary::{Boundary, Poly};
use base::differentiable::Differentiable;
use base::mono;
use derive_more::From;
use std::fmt;

/// The sort of events: a value that, as a boolean, has two states, but is only
/// ever compared, flipped, to signal an event, or kept, to signal none. An
/// event happens when its value changes, so that only the change, not the
/// state, means anything.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash, Default)]
pub struct Event;

impl Event {
    pub fn new() -> Self {
        Event
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

/// Whether two events `x` and `y` are in the same state.
///
/// SMT-LIB2: `(= x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
pub struct Eq;

impl Boundary<Sort> for Eq {
    fn dom(&self) -> Poly<Sort> {
        mono![Event.into(), Event.into()].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new_unchecked([1, 1]).into()].into()
    }
}

impl fmt::Display for Eq {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Eq")
    }
}

/// Whether two events `x` and `y` are in different states: whether an event
/// happened, of `x` and `y` the value of an event before and after.
///
/// SMT-LIB2: `(distinct x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
pub struct Ne;

impl Boundary<Sort> for Ne {
    fn dom(&self) -> Poly<Sort> {
        mono![Event.into(), Event.into()].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new_unchecked([1, 1]).into()].into()
    }
}

impl fmt::Display for Ne {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ne")
    }
}

/// The event `x` flipped: an event happens.
///
/// SMT-LIB2: `(not x)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
pub struct Toggle;

impl Boundary<Sort> for Toggle {
    fn dom(&self) -> Poly<Sort> {
        mono![Event.into()].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Event.into()].into()
    }
}

impl fmt::Display for Toggle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Toggle")
    }
}

/// The event `x` kept: no event happens.
///
/// SMT-LIB2: `x`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
pub struct Hold;

impl Boundary<Sort> for Hold {
    fn dom(&self) -> Poly<Sort> {
        mono![Event.into()].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Event.into()].into()
    }
}

impl fmt::Display for Hold {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hold")
    }
}

/// Choice `if c then x else y` between two events `x` and `y`, by the boolean
/// `c`, of the shape `[1, 1]`, read first.
///
/// SMT-LIB2: `(ite c x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
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
    Eq(Eq),
    Ne(Ne),
    Toggle(Toggle),
    Hold(Hold),
    Ite(Ite),
}

impl Boundary<Sort> for Op {
    fn dom(&self) -> Poly<Sort> {
        match self {
            Op::Eq(op) => op.dom(),
            Op::Ne(op) => op.dom(),
            Op::Toggle(op) => op.dom(),
            Op::Hold(op) => op.dom(),
            Op::Ite(op) => op.dom(),
        }
    }

    fn cod(&self) -> Poly<Sort> {
        match self {
            Op::Eq(op) => op.cod(),
            Op::Ne(op) => op.cod(),
            Op::Toggle(op) => op.cod(),
            Op::Hold(op) => op.cod(),
            Op::Ite(op) => op.cod(),
        }
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::Eq(op) => op.fmt(f),
            Op::Ne(op) => op.fmt(f),
            Op::Toggle(op) => op.fmt(f),
            Op::Hold(op) => op.fmt(f),
            Op::Ite(op) => op.fmt(f),
        }
    }
}
