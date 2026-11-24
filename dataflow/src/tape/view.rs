//! Borrowed views of the parts of a tape. Each view knows its context (its
//! tape and arm), so it can give its part a [`Boundary`].

use super::{Arm, Node, NodeId, NodeKind, Route, RouteId, Tape, Value};
use base::boundary::{Boundary, Mono, Poly};

/// An arm of a [`Tape`]: the program run on a value of input summand `Uⱼ`.
///
/// Its boundary is `Uⱼ → cod`: it reads one input summand and may end in any
/// output summand.
pub struct ArmRef<'a, S, O> {
    tape: &'a Tape<S, O>,
    index: usize,
}

impl<'a, S, O> ArmRef<'a, S, O> {
    pub(super) fn new(tape: &'a Tape<S, O>, index: usize) -> Self {
        Self { tape, index }
    }

    /// The arm's index `j`, which is also the index of its input summand.
    pub fn index(&self) -> usize {
        self.index
    }

    /// The arm's spans.
    pub fn arm(&self) -> &'a Arm {
        &self.tape.parts().arms[self.index]
    }

    /// The input summand `Uⱼ`: the sorts of the input values
    /// `v0 … v(|Uⱼ| − 1)`.
    pub fn inputs(&self) -> &'a Mono<S> {
        self.tape.input(self.index)
    }

    /// The circuit's nodes, in order, each with the values it defines.
    pub fn nodes(&self) -> impl Iterator<Item = NodeRef<'a, S, O>> + use<'a, S, O> {
        let tape = self.tape;
        let arm = self.index;
        let mut next = self.inputs().iter().count() as u32;
        self.arm().nodes.range().map(move |n| {
            let first_out = Value(next);
            next += tape.nodes()[n].n_outs;
            NodeRef {
                tape,
                arm,
                id: NodeId(n as u32),
                first_out,
            }
        })
    }

    /// The routes, children before parents. The last one is the root.
    pub fn routes(&self) -> impl Iterator<Item = RouteRef<'a, S, O>> + use<'a, S, O> {
        let tape = self.tape;
        let arm = self.index;
        self.arm().routes.range().map(move |d| RouteRef {
            tape,
            arm,
            id: RouteId(d as u32),
        })
    }

    /// The root of the routing DAG, where every run of the arm starts.
    pub fn root(&self) -> RouteRef<'a, S, O> {
        let routes = self.arm().routes;
        RouteRef {
            tape: self.tape,
            arm: self.index,
            id: RouteId(routes.start + routes.len - 1),
        }
    }
}

impl<S: Clone, O> Boundary<S> for ArmRef<'_, S, O> {
    fn dom(&self) -> Poly<S> {
        Poly::from(self.inputs().clone())
    }

    fn cod(&self) -> Poly<S> {
        self.tape.parts().cod.clone()
    }
}

/// A node of an arm's circuit, with the values it defines.
///
/// Its boundary is that of its [`Node`], that is, of its operation.
pub struct NodeRef<'a, S, O> {
    tape: &'a Tape<S, O>,
    arm: usize,
    id: NodeId,
    first_out: Value,
}

impl<'a, S, O> NodeRef<'a, S, O> {
    /// The node's global id.
    pub fn id(&self) -> NodeId {
        self.id
    }

    /// The index of the node's arm.
    pub fn arm(&self) -> usize {
        self.arm
    }

    /// The node itself.
    pub fn node(&self) -> &'a Node<O> {
        &self.tape.nodes()[self.id.index()]
    }

    /// What the node computes.
    pub fn kind(&self) -> &'a NodeKind<O> {
        &self.node().kind
    }

    /// The node's arguments: values of its arm defined before it.
    pub fn args(&self) -> &'a [Value] {
        &self.tape.args()[self.node().args.range()]
    }

    /// The values the node defines, in order.
    pub fn outputs(&self) -> impl Iterator<Item = Value> + use<S, O> {
        let n = self.node().n_outs;
        (self.first_out.0..self.first_out.0 + n).map(Value)
    }
}

impl<S, O: Boundary<S>> Boundary<S> for NodeRef<'_, S, O> {
    fn dom(&self) -> Poly<S> {
        self.node().dom()
    }

    fn cod(&self) -> Poly<S> {
        self.node().cod()
    }
}

/// A route of an arm's routing DAG.
///
/// Its boundary is that of the part of the arm that starts there. A split
/// is `Uⱼ → cod`: it reads the arm's input and may still end in any output
/// summand. An end node is `Uⱼ → Vᵢ`, for its output `i`.
pub struct RouteRef<'a, S, O> {
    tape: &'a Tape<S, O>,
    arm: usize,
    id: RouteId,
}

impl<'a, S, O> RouteRef<'a, S, O> {
    /// The route's global id.
    pub fn id(&self) -> RouteId {
        self.id
    }

    /// The index of the route's arm.
    pub fn arm(&self) -> usize {
        self.arm
    }

    /// The route itself.
    pub fn route(&self) -> &'a Route {
        &self.tape.routes()[self.id.index()]
    }

    /// The non-deterministic nodes drawn on reaching this route, in order.
    pub fn binds(&self) -> &'a [NodeId] {
        &self.tape.bound()[self.route().binds().range()]
    }

    /// The values carried to the output summand, for an end node, or `None` for a
    /// split.
    pub fn outs(&self) -> Option<&'a [Value]> {
        match *self.route() {
            Route::End { outs, .. } => Some(&self.tape.outs()[outs.range()]),
            Route::Split { .. } => None,
        }
    }

    /// The children `(hi, lo)` of a split, or `None` for an end node.
    pub fn children(&self) -> Option<(Self, Self)> {
        let (hi, lo) = self.route().children()?;
        let at = |id| RouteRef {
            tape: self.tape,
            arm: self.arm,
            id,
        };
        Some((at(hi), at(lo)))
    }
}

impl<S: Clone, O> Boundary<S> for RouteRef<'_, S, O> {
    fn dom(&self) -> Poly<S> {
        Poly::from(self.tape.input(self.arm).clone())
    }

    fn cod(&self) -> Poly<S> {
        match *self.route() {
            Route::End { output, .. } => Poly::from(self.tape.output(output as usize).clone()),
            Route::Split { .. } => self.tape.parts().cod.clone(),
        }
    }
}
