//! Plain-data references: spans into the side arrays of a tape, and the
//! `u32` indices of values, nodes and routes.

use std::fmt;
use std::ops::Range;

/// A contiguous run `start .. start + len` of one of the arrays of a
/// [`Tape`](super::Tape).
///
/// Variable-length lists (a node's arguments, a route's binders, an end node's
/// values) are not stored in their owner. They are concatenated into one
/// array per kind of list, and the owner keeps a span into that array. This
/// keeps every struct of the tape `Copy`, fixed-size and free of heap
/// pointers. (`std::ops::Range<u32>` would do the same job, but it is not
/// `Copy`.)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Span {
    /// The index of the first element.
    pub start: u32,
    /// The number of elements.
    pub len: u32,
}

impl Span {
    /// The empty span.
    pub const EMPTY: Span = Span { start: 0, len: 0 };

    /// The span of `len` elements starting at `start`.
    pub const fn new(start: u32, len: u32) -> Self {
        Self { start, len }
    }

    /// The index one past the last element. It is computed in `usize`, so it
    /// cannot overflow.
    pub fn end(self) -> usize {
        self.start as usize + self.len as usize
    }

    /// The span as a range of indices, for slicing.
    pub fn range(self) -> Range<usize> {
        self.start as usize..self.end()
    }

    /// Whether the span has no elements.
    pub fn is_empty(self) -> bool {
        self.len == 0
    }
}

/// A value in the circuit of one arm: a vertex of the term graph.
///
/// Values are numbered **per arm**, by position. In an arm whose input
/// summand is `Uⱼ`, the values `0 … |Uⱼ| − 1` are the inputs, one per sort of
/// `Uⱼ`, in order. Then come the outputs of the arm's nodes, in node order
/// (a node with `n_outs` outputs defines that many consecutive values). So
/// the same `Value` means different things in different arms.
///
/// Printed as `v3`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Value(pub u32);

/// A node of a tape: an index into [`Tape::nodes`](super::Tape::nodes).
///
/// Node ids are **global**: the nodes of all arms are stored in one
/// array, each arm owning a contiguous span of it.
///
/// Printed as `n3`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub u32);

/// A route of a tape: an index into
/// [`Tape::routes`](super::Tape::routes).
///
/// Route ids are **global**, like [`NodeId`]s. A route may only refer
/// to routes of its own arm.
///
/// Printed as `r3`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RouteId(pub u32);

impl Value {
    /// The index, for indexing a slice.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

impl NodeId {
    /// The index, for indexing a slice.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "n{}", self.0)
    }
}

impl RouteId {
    /// The index, for indexing a slice.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for RouteId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "r{}", self.0)
    }
}
