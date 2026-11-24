//! The frozen tape and its raw parts.

use super::check;
use super::view::ArmRef;
use super::{Node, NodeId, Operation, Route, Span, Value};
use base::boundary::{Boundary, Mono, Poly};
use std::fmt::Debug;

/// One arm of a tape: the program run on a value of one input summand, like
/// one arm of a `match` on the input.
///
/// Arm `j` reads a value of the `j`-th summand `Uⱼ` of the tape's domain.
/// It owns a contiguous run of nodes, its circuit, and a contiguous run of
/// routes, its routing DAG. Arm `j + 1` starts where arm `j`
/// ends, so the arms partition both arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Arm {
    /// The arm's circuit: a span of [`Tape::nodes`].
    pub nodes: Span,
    /// The arm's routing DAG: a span of [`Tape::routes`]. It is never
    /// empty, and its **last** route is the root, since children come
    /// before their parents.
    pub routes: Span,
}

/// The raw arrays of a [`Tape`], with public fields.
///
/// This is how a tape is put together by hand, and taken apart: fill in a
/// `TapeParts`, then [`Tape::new`] checks that it is well formed. (To put
/// tapes together from combinators, in normal form, use a
/// [`Builder`](super::Builder).) See the
/// [module documentation](super) for the meaning of each array and an
/// example.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TapeParts<S, O> {
    /// The boolean sort: the sort of the conditions of splits. This is the
    /// only thing the tape assumes about booleans: the operations on them
    /// are as opaque as any other.
    pub boolean: S,
    /// The domain `U₁ ⊕ … ⊕ Uₙ`: one arm per summand.
    pub dom: Poly<S>,
    /// The codomain `V₁ ⊕ … ⊕ Vₘ`: every path ends in one of its summands.
    pub cod: Poly<S>,
    /// One arm per summand of `dom`, in order.
    pub arms: Vec<Arm>,
    /// The nodes of all arms, arm after arm, each arm's in
    /// topological order.
    pub nodes: Vec<Node<O>>,
    /// The arguments of all nodes, concatenated.
    pub args: Vec<Value>,
    /// The routes of all arms, arm after arm, each arm's
    /// children before parents.
    pub routes: Vec<Route>,
    /// The binders of all routes, concatenated: the non-deterministic
    /// nodes drawn at each route.
    pub bound: Vec<NodeId>,
    /// The values of all end nodes, concatenated.
    pub outs: Vec<Value>,
}

/// A tape `dom → cod`: a well-formed, immutable, flat representation of a
/// program that branches on data and on chance.
///
/// See the [module documentation](super) for what the parts mean. The fields
/// are private so that every `Tape` is well formed. Read them through the
/// accessors, or iterate over the arms with [`arms`](Self::arms),
/// whose views also give each part its
/// [`Boundary`](base::boundary::Boundary).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tape<S, O> {
    parts: TapeParts<S, O>,
}

impl<S, O> Tape<S, O>
where
    S: Clone + PartialEq + Debug,
    O: Operation<S> + Debug,
{
    /// Checks that `parts` is a well-formed tape, and freezes it.
    ///
    /// The parts are **well formed** when all of the following hold.
    ///
    /// *Arms.* There is one arm per summand of `dom`. The arms'
    /// spans partition `nodes` and `routes` in order, and every arm
    /// has at least one route.
    ///
    /// *Nodes.* Every span is within its array. Every argument is a value
    /// defined before the node (an input of the arm, or an output of an
    /// earlier node of the arm). The argument sorts equal the domain of
    /// the node, and `n_outs` equals the length of its codomain. For opaque
    /// operations, both are taken from [`Boundary`], and both sides must be
    /// single products. `Det` nodes hold deterministic operations and
    /// `NonDet` nodes non-deterministic ones.
    ///
    /// *Routes.* The children of a split are earlier routes of the same
    /// arm. Every route of an arm is reachable from its root. The
    /// condition of a split is a value of the arm of the boolean sort.
    /// An end node's output is a summand of `cod`, and its values match that
    /// summand sort by sort.
    ///
    /// *Binders.* Every non-deterministic node is bound by exactly one
    /// route of its arm, and only non-deterministic nodes are bound.
    /// A route that binds draws, itself or below it, is reached after the
    /// same number `l` of draws on every path from the root. (Routes that
    /// bind nothing, such as a shared end node, may be reached after different
    /// numbers of draws.) The node bound `k`-th at such a route has level
    /// `l + k`. A draw is bound after every draw its
    /// arguments depend on. On every path, a split's condition and an end node's
    /// values only depend on draws already made.
    ///
    /// These conditions do *not* include the normal form (see the module
    /// documentation), which is not checked.
    ///
    /// Fails, with a message naming the offending part, if `parts` is not
    /// well formed.
    pub fn new(parts: TapeParts<S, O>) -> Result<Self, String> {
        check::check(&parts)?;
        Ok(Self { parts })
    }
}

impl<S, O> Tape<S, O> {
    /// Gives the raw arrays back.
    pub fn into_parts(self) -> TapeParts<S, O> {
        self.parts
    }

    /// The raw arrays, read-only.
    pub fn parts(&self) -> &TapeParts<S, O> {
        &self.parts
    }

    /// The boolean sort.
    pub fn boolean(&self) -> &S {
        &self.parts.boolean
    }

    /// The `j`-th summand `Uⱼ` of the domain: the inputs of arm `j`.
    ///
    /// # Panics
    /// If there is no such summand.
    pub fn input(&self, j: usize) -> &Mono<S> {
        self.parts.dom.iter().nth(j).expect("no such input summand")
    }

    /// The `i`-th summand `Vᵢ` of the codomain: where the end nodes with
    /// output `i` end.
    ///
    /// # Panics
    /// If there is no such summand.
    pub fn output(&self, i: usize) -> &Mono<S> {
        self.parts
            .cod
            .iter()
            .nth(i)
            .expect("no such output summand")
    }

    /// The summands `V₁, …, Vₘ` of the codomain, in order: where the end nodes
    /// end.
    pub fn outputs(&self) -> impl ExactSizeIterator<Item = &Mono<S>> {
        self.parts.cod.iter()
    }

    /// The `j`-th arm, as a view.
    ///
    /// # Panics
    /// If `j` is not less than the number of arms, `arms().len()`.
    pub fn arm(&self, j: usize) -> ArmRef<'_, S, O> {
        assert!(j < self.parts.arms.len(), "no such arm");
        ArmRef::new(self, j)
    }

    /// The arms, in order, as views: one per summand of the domain.
    pub fn arms(&self) -> impl ExactSizeIterator<Item = ArmRef<'_, S, O>> {
        (0..self.parts.arms.len()).map(|j| ArmRef::new(self, j))
    }

    /// All the nodes, arm after arm.
    pub fn nodes(&self) -> &[Node<O>] {
        &self.parts.nodes
    }

    /// All the node arguments, concatenated.
    pub fn args(&self) -> &[Value] {
        &self.parts.args
    }

    /// All the routes, arm after arm.
    pub fn routes(&self) -> &[Route] {
        &self.parts.routes
    }

    /// All the binders, concatenated.
    pub fn bound(&self) -> &[NodeId] {
        &self.parts.bound
    }

    /// The values of all end nodes, concatenated.
    pub fn outs(&self) -> &[Value] {
        &self.parts.outs
    }
}

/// The boundary of a tape is `dom → cod`. (To borrow them instead of
/// cloning, use [`Tape::parts`], or [`Tape::input`] and [`Tape::output`] for
/// single summands.)
impl<S: Clone, O> Boundary<S> for Tape<S, O> {
    fn dom(&self) -> Poly<S> {
        self.parts.dom()
    }

    fn cod(&self) -> Poly<S> {
        self.parts.cod()
    }
}

/// The boundary the parts declare, `dom → cod`, whether or not they are well
/// formed.
impl<S: Clone, O> Boundary<S> for TapeParts<S, O> {
    fn dom(&self) -> Poly<S> {
        self.dom.clone()
    }

    fn cod(&self) -> Poly<S> {
        self.cod.clone()
    }
}
