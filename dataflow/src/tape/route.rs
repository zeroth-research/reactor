//! The routes of an arm: splits on booleans, and end nodes.

use super::{RouteId, Span, Value};

/// A node of an arm's routing DAG.
///
/// A run of arm `j` starts at the arm's root (its last route) and
/// repeats two steps until it reaches an end node:
///
/// 1. it draws the non-deterministic nodes listed in `binds`, in order;
/// 2. at a [`Split`](Route::Split), it moves to `hi` or `lo`. At a
///    [`End`](Route::End), it ends in output summand `output` with the
///    values `outs`.
///
/// Children are always stored before their parents and in the same arm, so
/// the DAG is acyclic and every path ends at an end node. A route may have
/// several parents.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Route {
    /// Continue with `hi` if `cond` is true, with `lo` if it is false: the
    /// tape splits in two, as with the `case` primitive.
    ///
    /// This is a split on a boolean, which may have been drawn at random.
    Split {
        /// The draws made on reaching this route: a span of
        /// [`Tape::bound`](super::Tape::bound).
        binds: Span,
        /// The value tested: any value of the arm of the boolean sort.
        cond: Value,
        /// Where to continue if `cond` is true. It is stored before this
        /// route.
        hi: RouteId,
        /// Where to continue if `cond` is false. It is stored before this
        /// route.
        lo: RouteId,
    },
    /// End the path in output summand `output`.
    End {
        /// The draws made on reaching this route: a span of
        /// [`Tape::bound`](super::Tape::bound).
        binds: Span,
        /// Where the path ends: the index `i` of the output summand `Vᵢ` of
        /// the tape's codomain. (The values carried there are `outs`.)
        output: u32,
        /// The values carried into `Vᵢ`, one per sort of `Vᵢ`, in order: a
        /// span of [`Tape::outs`](super::Tape::outs). A value may appear
        /// several times (a copy). An arm value that no end node, split or node
        /// uses is discarded.
        outs: Span,
    },
}

impl Route {
    /// The draws made on reaching this route: a span of
    /// [`Tape::bound`](super::Tape::bound).
    pub fn binds(&self) -> Span {
        match *self {
            Route::Split { binds, .. } | Route::End { binds, .. } => binds,
        }
    }

    /// The children `(hi, lo)` of a split, or `None` for an end node.
    pub fn children(&self) -> Option<(RouteId, RouteId)> {
        match *self {
            Route::Split { hi, lo, .. } => Some((hi, lo)),
            Route::End { .. } => None,
        }
    }
}
