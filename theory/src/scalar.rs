/*!
# Scalars

The scalar operations of each theory, one type per operation, in a module per
theory: `enum` for the enumerations of symbolic constants, and [`event`] for
the events, booleans whose change signals them. Each operation reads and
writes the sorts [`Enum`] and [`Event`], which have no shape, and the
booleans of the matrix theories of the shape `[1, 1]`, which their
comparisons write and their choices read, and computes its own boundary.

Unlike the operations of [`crate::matrix`], the scalar operations carry no
matrices, so they are not generic over a [`Matrix`](crate::Matrix) type.

[`Op`] holds the operations of all of them, one variant per theory, with
their boundary in the sorts of all theories, [`crate::Any`], which hold
those booleans. The structural
operations, which belong to no theory, are those of all theories,
[`crate::Havoc`], [`crate::Skip`] and [`crate::Zero`].

## Sorts

The sorts of the scalar theories: [`Enum`], the enumerations of symbolic
constants, and [`Event`], the events. [`Any`] holds them. The booleans that
their comparisons write, and the conditions of their choices read, are those
of the matrix theories, of the shape `[1, 1]`:
[`matrix::bool::Bool`](crate::matrix::bool::Bool).

The sorts are defined here, and the operations on them in the modules of
their theories: `enum` and [`event`].

All of them are discrete, so their tangent is trivial: the singleton sort
[`Real0`], shared with the matrix theories. Each sort gives it, but [`Any`]
neither holds it nor takes tangents: no scalar operation reads or writes it,
and the tangents of all sorts are taken in the sorts of all theories,
[`crate::Any`], which holds it.
*/

pub mod r#enum;
pub mod event;

use crate::zero::Real0;
use base::boundary::{Boundary, Poly};
use base::differentiable::Differentiable;
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

/// An enumeration: the sort whose values are the symbolic constants it names,
/// as nuXmv's symbolic enumeration types, such as `{idle, busy, done}`.
///
/// An enumeration is anonymous: it is identified by its values alone, as a
/// set, so that two enumerations of the same values, in whatever order, are
/// the same sort. It holds them sorted and without repetitions.
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub struct Enum {
    /// The values, strictly increasing: sorted, without repetitions, and never
    /// empty. Membership, inclusion and union rely on it.
    values: Vec<String>,
}

impl Enum {
    /// The enumeration of the symbolic constants `values`, a set: their order
    /// and repetitions do not matter. Fails when there are none, as a sort
    /// with no values cannot be written.
    pub fn new<I>(values: I) -> Result<Self, crate::Error>
    where
        I: IntoIterator,
        I::Item: Into<String>,
    {
        let mut values: Vec<String> = values.into_iter().map(Into::into).collect();
        values.sort();
        values.dedup();
        if values.is_empty() {
            return Err(crate::Error(
                "an enumeration must have at least one value".to_string(),
            ));
        }
        Ok(Enum { values })
    }

    /// The symbolic constants of the enumeration, sorted.
    pub fn values(&self) -> &[String] {
        &self.values
    }

    /// Whether `value` is one of the symbolic constants of the enumeration.
    pub fn contains(&self, value: &str) -> bool {
        self.values
            .binary_search_by(|v| v.as_str().cmp(value))
            .is_ok()
    }

    /// The number of its values.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether `other` holds all the values of `self`: a single walk through
    /// both, sorted.
    pub fn is_subset(&self, other: &Enum) -> bool {
        let mut theirs = other.values.iter();
        self.values
            .iter()
            .all(|v| theirs.by_ref().find(|w| *w >= v) == Some(v))
    }

    /// The enumeration of the values of both `self` and `other`: the merge of
    /// both, sorted, keeping one of the values they share.
    pub fn union(&self, other: &Enum) -> Enum {
        let (mut xs, mut ys) = (
            self.values.iter().peekable(),
            other.values.iter().peekable(),
        );
        let mut values = Vec::with_capacity(self.len() + other.len());
        loop {
            let next = match (xs.peek(), ys.peek()) {
                (Some(x), Some(y)) if x < y => xs.next(),
                (Some(x), Some(y)) if y < x => ys.next(),
                (Some(_), Some(_)) => {
                    ys.next();
                    xs.next()
                }
                (Some(_), None) => xs.next(),
                (None, Some(_)) => ys.next(),
                (None, None) => break,
            };
            values.extend(next.cloned());
        }
        Enum { values }
    }

    /// Never: an enumeration has at least one value.
    pub fn is_empty(&self) -> bool {
        false
    }
}

/// Writes it as it is built, with its values sorted: `Enum(["busy", "idle"])`.
impl fmt::Display for Enum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Enum([")?;
        for (i, v) in self.values.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{v:?}")?;
        }
        write!(f, "])")
    }
}

/// The enumerations are discrete, so their tangent is trivial: the singleton
/// sort [`Real0`].
impl Differentiable for Enum {
    type Tangent = Real0;

    fn tangent(self) -> Real0 {
        Real0
    }
}

/// The tangent of a borrowed enumeration, which is not `Copy`: that of the
/// enumeration.
impl Differentiable for &Enum {
    type Tangent = Real0;

    fn tangent(self) -> Real0 {
        Real0
    }
}

/// The sorts of the scalar theories: a value of `Any` is one of them.
#[derive(Clone, PartialEq, Debug, Eq, Hash, From)]
pub enum Any {
    Enum(Enum),
    Event(Event),
}

impl fmt::Display for Any {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Any::Enum(e) => e.fmt(f),
            Any::Event(e) => e.fmt(f),
        }
    }
}

/// The operations of all the scalar theories: a value of `Op` is an
/// operation of one of them.
#[derive(Clone, PartialEq, Debug, Eq, Hash, From)]
pub enum Op {
    Enum(r#enum::Op),
    Event(event::Op),
}

/// The boundary of each operation, in the sorts of all theories.
impl Boundary<crate::Any> for Op {
    fn dom(&self) -> Poly<crate::Any> {
        match self {
            Op::Enum(op) => op.dom(),
            Op::Event(op) => op.dom(),
        }
    }

    fn cod(&self) -> Poly<crate::Any> {
        match self {
            Op::Enum(op) => op.cod(),
            Op::Event(op) => op.cod(),
        }
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::Enum(op) => op.fmt(f),
            Op::Event(op) => op.fmt(f),
        }
    }
}
