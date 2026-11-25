/*!
# Sorts

The sorts of the scalar theories: [`Enum`], the enumerations of symbolic
constants, [`Bool`], the booleans that their comparisons write, and
[`Event`], the events. [`Any`] holds them.

The sorts are defined here, and the operations on them in the modules of
their theories: `enum`, [`bool`](mod@super::bool) and [`event`](super::event).

All of them are discrete, so their tangent is trivial: the singleton sort
[`Zero`], shared with the matrix theories. Each sort gives it, but [`Any`]
neither holds it nor takes tangents: no scalar operation reads or writes it,
and the tangents of all sorts are taken in the sorts of all theories,
[`crate::sort::Any`], which holds it.
*/

use crate::sort::Zero;
use base::differentiable::Differentiable;
use derive_more::From;
use std::fmt;

/// The sort of scalar booleans: the sort that the comparisons of the scalar
/// theories write, and the conditions of their choices read.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash, Default)]
pub struct Bool;

impl Bool {
    pub fn new() -> Self {
        Bool
    }
}

impl fmt::Display for Bool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bool()")
    }
}

/// The booleans are discrete, so their tangent is trivial: the singleton sort
/// [`Zero`].
impl Differentiable for Bool {
    type Tangent = Zero;

    fn tangent(self) -> Zero {
        Zero
    }
}

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
/// [`Zero`].
impl Differentiable for Event {
    type Tangent = Zero;

    fn tangent(self) -> Zero {
        Zero
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
/// sort [`Zero`].
impl Differentiable for Enum {
    type Tangent = Zero;

    fn tangent(self) -> Zero {
        Zero
    }
}

/// The tangent of a borrowed enumeration, which is not `Copy`: that of the
/// enumeration.
impl Differentiable for &Enum {
    type Tangent = Zero;

    fn tangent(self) -> Zero {
        Zero
    }
}

/// The sorts of the scalar theories: a value of `Any` is one of them.
#[derive(Clone, PartialEq, Debug, Eq, Hash, From)]
pub enum Any {
    Enum(Enum),
    Bool(Bool),
    Event(Event),
}

impl fmt::Display for Any {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Any::Enum(e) => e.fmt(f),
            Any::Bool(b) => b.fmt(f),
            Any::Event(e) => e.fmt(f),
        }
    }
}
