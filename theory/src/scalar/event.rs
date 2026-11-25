/*!
# Events

Defines one type per operation on the sort [`Event`] of events. An event is
encoded as a boolean whose change signals that the event happened, so its
state alone means nothing: it is only compared, flipped or kept.

- [`Eq`](struct@Eq), [`Ne`] — whether two events are in the same state or in
  different ones, writing a [`Bool`]: an event happened between two of its
  values when they differ.
- [`Toggle`] — the event flipped: an event happens.
- [`Hold`] — the event kept: no event happens.
- [`Ite`] — the choice `if c then x else y` between two events, by a
  boolean `c`.

The sort has no parameters, so neither do the operations: each is a unit
value, determined by its type alone, and its boundary is fixed. In SMT-LIB2,
an event is a `Bool`, and its operations `=`, `distinct`, `not`, the
identity and `ite`.
*/

pub use super::sort::Any as Sort;
use super::sort::{Bool, Event};
use base::boundary::{Boundary, Poly};
use base::mono;
use derive_more::From;
use std::fmt;

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
        mono![Bool.into()].into()
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
        mono![Bool.into()].into()
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
/// `c`, read first.
///
/// SMT-LIB2: `(ite c x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
pub struct Ite;

impl Boundary<Sort> for Ite {
    fn dom(&self) -> Poly<Sort> {
        mono![Bool.into(), Event.into(), Event.into()].into()
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
pub enum Any {
    Eq(Eq),
    Ne(Ne),
    Toggle(Toggle),
    Hold(Hold),
    Ite(Ite),
}

impl Boundary<Sort> for Any {
    fn dom(&self) -> Poly<Sort> {
        match self {
            Any::Eq(op) => op.dom(),
            Any::Ne(op) => op.dom(),
            Any::Toggle(op) => op.dom(),
            Any::Hold(op) => op.dom(),
            Any::Ite(op) => op.dom(),
        }
    }

    fn cod(&self) -> Poly<Sort> {
        match self {
            Any::Eq(op) => op.cod(),
            Any::Ne(op) => op.cod(),
            Any::Toggle(op) => op.cod(),
            Any::Hold(op) => op.cod(),
            Any::Ite(op) => op.cod(),
        }
    }
}

impl fmt::Display for Any {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Any::Eq(op) => op.fmt(f),
            Any::Ne(op) => op.fmt(f),
            Any::Toggle(op) => op.fmt(f),
            Any::Hold(op) => op.fmt(f),
            Any::Ite(op) => op.fmt(f),
        }
    }
}
