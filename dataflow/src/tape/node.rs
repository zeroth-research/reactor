//! The nodes of a circuit, and what the tape needs to know about operations.

use super::Span;
use base::boundary::{Boundary, Poly};

/// An opaque operation, as seen by a tape.
///
/// A tape does not interpret operations. It needs exactly two facts about
/// each one:
///
/// - its **boundary** `M → N`, through [`Boundary`]. Both sides must be
///   single products (a `Poly` with exactly one summand), because an
///   operation acts on strings, not on tapes;
/// - whether it is **deterministic**. Deterministic operations can be shared:
///   applying one twice to the same arguments gives the same result, so the
///   two applications are one node. Non-deterministic operations, such as
///   sampling a distribution, cannot be shared: two draws are independent.
///   They are bound at routes instead (see [`NodeKind::NonDet`]).
///
/// All operations are assumed to be **total**: they always produce a result,
/// and sampling always produces a sample. This is what allows the values of
/// every branch to be computed in one shared circuit. (This assumption is to
/// be revisited.)
pub trait Operation<S>: Boundary<S> {
    /// Whether applying the operation twice to the same arguments always
    /// gives the same results.
    fn is_deterministic(&self) -> bool;
}

/// What a [`Node`] computes: an opaque operation, split by determinism,
/// since the two kinds obey different laws.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum NodeKind<O> {
    /// A deterministic operation. In normal form, no two `Det` nodes of a
    /// arm have the same operation and arguments.
    Det(O),
    /// A non-deterministic operation, for instance `sample`.
    ///
    /// Such a node must be *bound* by exactly one route of its arm (see
    /// [`Route::binds`](super::Route::binds)). That is where the draw
    /// happens, and every use of the node must come after it on every path.
    /// `level` is the number of draws made on the path before this one, so
    /// equal draws on different paths at the same depth are the same node.
    NonDet {
        /// The operation.
        op: O,
        /// The number of draws bound before this one on every path that
        /// reaches it (its de Bruijn level).
        level: u32,
    },
}

impl<O> NodeKind<O> {
    /// The operation of the node.
    pub fn op(&self) -> &O {
        match self {
            NodeKind::Det(op) | NodeKind::NonDet { op, .. } => op,
        }
    }
}

/// The boundary of the operation.
impl<S, O: Boundary<S>> Boundary<S> for NodeKind<O> {
    fn dom(&self) -> Poly<S> {
        self.op().dom()
    }

    fn cod(&self) -> Poly<S> {
        self.op().cod()
    }
}

/// A node of a circuit: one application of an operation, a hyperedge of the
/// term graph.
///
/// A node reads its arguments, values defined earlier in its arm, and
/// defines `n_outs` new values. They follow the values defined before it:
/// the first output of the `k`-th node of an arm is the arm's input
/// count plus the outputs of the nodes before it. (This number is not
/// stored, since it can be derived; [`NodeRef`](super::NodeRef) computes it
/// while iterating.)
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Node<O> {
    /// What the node computes.
    pub kind: NodeKind<O>,
    /// The arguments: a span of [`Tape::args`](super::Tape::args). They
    /// match the domain of the operation, sort by sort.
    pub args: Span,
    /// The number of values the node defines: the length of the codomain of
    /// its operation.
    pub n_outs: u32,
}

/// The boundary of the node's operation: a node needs no context for it,
/// unlike arms and routes (see [`ArmRef`](super::ArmRef) and
/// [`RouteRef`](super::RouteRef)).
impl<S, O: Boundary<S>> Boundary<S> for Node<O> {
    fn dom(&self) -> Poly<S> {
        self.kind.dom()
    }

    fn cod(&self) -> Poly<S> {
        self.kind.cod()
    }
}
