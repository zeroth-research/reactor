//! Canonicalization: from a draft to a [`Tape`] in normal form.
//!
//! Each arm is canonicalized on its own, from a [`LocalArm`]: a self-contained
//! copy of a scratch arm, whose draws have no binders and no levels. One pass
//! of canonicalization:
//!
//! 1. *Analyses* the routing DAG reachable from the root: its dominators,
//!    and, for every draw, the routes that use it, directly or through the
//!    nodes that depend on it. A draw is bound at the lowest route that
//!    dominates all its uses. Draws bound at the same route are ordered by
//!    their first use in the traversal of step 2, which puts a draw after
//!    the draws its arguments depend on.
//! 2. *Emits* the tape from the root, depth first: at a split, its
//!    condition, then `hi`, then `lo`; at an end, its values. A value is
//!    emitted after its arguments, and nodes and routes are numbered in the
//!    order they are emitted. A draw's level is the number of draws bound on
//!    the path before it. Nodes and routes are hash-consed as they are
//!    emitted; a route is emitted once per depth, if it binds draws (itself
//!    or below it), and once per assignment of levels to the draws it uses
//!    but does not bind. So there is no dead code, and a route that binds
//!    draws is never reached after different numbers of draws.
//!
//! Hash-consing can make draws equal that were different in the draft: two
//! draws of the same operation, with the same arguments and level, on two
//! paths. Then a split may get two equal children, and a draw may be bound at
//! two routes. Both are fixed by another pass, on the emitted arm, until
//! nothing changes. Every such pass merges draws or removes a split, so this
//! terminates. This is what makes `case ; (f ⊕ f) ; merge` and `f` equal even
//! when `f` samples.

use super::build::{Arena, SKind, SRoute, push_span};
use super::{Arm, Node, NodeId, NodeKind, Operation, Route, RouteId, Span, Tape, TapeParts, Value};
use base::boundary::Poly;
use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap};
use std::fmt::Debug;
use std::hash::Hash;

/// A node of a [`LocalArm`].
struct LNode<O> {
    draw: bool,
    op: O,
    args: Vec<Value>,
    n_outs: u32,
}

/// A route of a [`LocalArm`]; `hi` and `lo` are earlier routes.
enum LRoute {
    Split { cond: Value, hi: usize, lo: usize },
    End { output: u32, outs: Vec<Value> },
}

/// One arm to canonicalize: its circuit, with draws that are neither bound
/// nor numbered, and its routing DAG, children before parents.
pub(super) struct LocalArm<O> {
    n_inputs: u32,
    nodes: Vec<LNode<O>>,
    routes: Vec<LRoute>,
    root: usize,
}

impl<O: Clone> LocalArm<O> {
    /// The scratch arm `arm` of `arena`, with the outputs of its ends
    /// shifted by `shift`.
    pub(super) fn from_scratch(arena: &Arena<O>, arm: u32, shift: u32) -> Self {
        let arm = arena.arms[arm as usize];
        let nodes = arm
            .nodes
            .range()
            .map(|n| {
                let node = &arena.nodes[n];
                let (draw, op) = match &node.kind {
                    SKind::Det(op) => (false, op.clone()),
                    SKind::Draw(op) => (true, op.clone()),
                };
                LNode {
                    draw,
                    op,
                    args: arena.args[node.args.range()].to_vec(),
                    n_outs: node.n_outs,
                }
            })
            .collect();
        let start = arm.routes.start as usize;
        let routes = arm
            .routes
            .range()
            .map(|r| match arena.routes[r] {
                SRoute::Split { cond, hi, lo } => LRoute::Split {
                    cond,
                    hi: hi as usize - start,
                    lo: lo as usize - start,
                },
                SRoute::End { output, outs } => LRoute::End {
                    output: output + shift,
                    outs: arena.outs[outs.range()].to_vec(),
                },
            })
            .collect();
        Self {
            n_inputs: arm.n_inputs,
            nodes,
            routes,
            root: arm.root as usize - start,
        }
    }

    /// An emitted arm, with root `root`, to canonicalize again. Its
    /// non-deterministic nodes become draws again, and its binders are
    /// dropped.
    fn from_emitted(n_inputs: u32, e: &Emitted<O>, root: RouteId) -> Self {
        let nodes = e
            .nodes
            .iter()
            .map(|node| LNode {
                draw: matches!(node.kind, NodeKind::NonDet { .. }),
                op: node.kind.op().clone(),
                args: e.args[node.args.range()].to_vec(),
                n_outs: node.n_outs,
            })
            .collect();
        let routes = e
            .routes
            .iter()
            .map(|route| match *route {
                Route::Split { cond, hi, lo, .. } => LRoute::Split {
                    cond,
                    hi: hi.index(),
                    lo: lo.index(),
                },
                Route::End { output, outs, .. } => LRoute::End {
                    output,
                    outs: e.outs[outs.range()].to_vec(),
                },
            })
            .collect();
        Self {
            n_inputs,
            nodes,
            routes,
            root: root.index(),
        }
    }
}

/// Builds the tape `dom → cod` with one arm per element of `arms`, each in
/// normal form.
pub(super) fn freeze<S, O>(
    boolean: S,
    dom: Poly<S>,
    cod: Poly<S>,
    arms: Vec<LocalArm<O>>,
) -> Tape<S, O>
where
    S: Clone + PartialEq + Debug,
    O: Operation<S> + Clone + Eq + Hash + Debug,
{
    let mut parts = TapeParts {
        boolean,
        dom,
        cod,
        arms: Vec::new(),
        nodes: Vec::new(),
        args: Vec::new(),
        routes: Vec::new(),
        bound: Vec::new(),
        outs: Vec::new(),
    };
    for arm in arms {
        append(&mut parts, canonicalize(arm));
    }
    Tape::new(parts).unwrap_or_else(|e| panic!("freeze produced an ill-formed tape: {e}"))
}

/// Canonicalizes one arm: emits it until nothing changes.
fn canonicalize<O: Clone + Eq + Hash>(mut arm: LocalArm<O>) -> Emitted<O> {
    loop {
        let mut emitter = Emitter::new(&arm);
        let root = emitter.route(arm.root, 0, &HashMap::new());
        if !emitter.changed {
            return emitter.out;
        }
        arm = LocalArm::from_emitted(arm.n_inputs, &emitter.out, root);
    }
}

/// Appends an emitted arm to `parts`, moving its local ids and spans to
/// their place in the tape's arrays.
fn append<S, O>(parts: &mut TapeParts<S, O>, e: Emitted<O>) {
    let (node_base, route_base) = (parts.nodes.len() as u32, parts.routes.len() as u32);
    let (args_base, bound_base, outs_base) = (
        parts.args.len() as u32,
        parts.bound.len() as u32,
        parts.outs.len() as u32,
    );
    let shift = |span: Span, base: u32| {
        if span.is_empty() {
            Span::EMPTY
        } else {
            Span::new(span.start + base, span.len)
        }
    };
    parts.arms.push(Arm {
        nodes: Span::new(node_base, e.nodes.len() as u32),
        routes: Span::new(route_base, e.routes.len() as u32),
    });
    parts.nodes.extend(e.nodes.into_iter().map(|node| Node {
        args: shift(node.args, args_base),
        ..node
    }));
    parts.args.extend(e.args);
    parts
        .routes
        .extend(e.routes.into_iter().map(|route| match route {
            Route::Split {
                binds,
                cond,
                hi,
                lo,
            } => Route::Split {
                binds: shift(binds, bound_base),
                cond,
                hi: RouteId(hi.0 + route_base),
                lo: RouteId(lo.0 + route_base),
            },
            Route::End {
                binds,
                output,
                outs,
            } => Route::End {
                binds: shift(binds, bound_base),
                output,
                outs: shift(outs, outs_base),
            },
        }));
    parts
        .bound
        .extend(e.bound.into_iter().map(|n| NodeId(n.0 + node_base)));
    parts.outs.extend(e.outs);
}

/// What the emitter needs to know about an arm before emitting it.
struct Analysis {
    /// The first output of each node.
    first_out: Vec<u32>,
    /// The node that defines each value that is not an input, indexed by
    /// value minus the number of inputs.
    producer: Vec<usize>,
    /// The draws each node's outputs depend on, the node itself included if
    /// it is a draw, in increasing order.
    node_deps: Vec<Vec<usize>>,
    /// The draws bound at each route, in order.
    binds: Vec<Vec<usize>>,
    /// The draws used at or below each route but bound above it.
    free: Vec<Vec<usize>>,
    /// Whether each route, or one below it, binds a draw.
    binds_below: Vec<bool>,
}

impl Analysis {
    fn new<O>(arm: &LocalArm<O>) -> Self {
        let n_inputs = arm.n_inputs as usize;

        // Circuit: outputs and dependencies on draws.
        let mut first_out = Vec::with_capacity(arm.nodes.len());
        let mut producer = Vec::new();
        let mut node_deps: Vec<Vec<usize>> = Vec::with_capacity(arm.nodes.len());
        let mut next = arm.n_inputs;
        for (n, node) in arm.nodes.iter().enumerate() {
            first_out.push(next);
            next += node.n_outs;
            producer.extend((0..node.n_outs).map(|_| n));
            let mut deps = BTreeSet::new();
            for v in &node.args {
                if v.index() >= n_inputs {
                    deps.extend(node_deps[producer[v.index() - n_inputs]].iter().copied());
                }
            }
            if node.draw {
                deps.insert(n);
            }
            node_deps.push(deps.into_iter().collect());
        }
        let value_deps = |v: &Value| -> &[usize] {
            if v.index() < n_inputs {
                &[]
            } else {
                &node_deps[producer[v.index() - n_inputs]]
            }
        };
        let used_values = |r: usize| -> Vec<Value> {
            match &arm.routes[r] {
                LRoute::Split { cond, .. } => vec![*cond],
                LRoute::End { outs, .. } => outs.clone(),
            }
        };
        let children = |r: usize| -> Option<(usize, usize)> {
            match arm.routes[r] {
                LRoute::Split { hi, lo, .. } => Some((hi, lo)),
                LRoute::End { .. } => None,
            }
        };

        // Routes reachable from the root, and their parents.
        let n_routes = arm.routes.len();
        let mut reachable = vec![false; n_routes];
        let mut parents: Vec<Vec<usize>> = vec![Vec::new(); n_routes];
        reachable[arm.root] = true;
        for r in (0..=arm.root).rev() {
            if let (true, Some((hi, lo))) = (reachable[r], children(r)) {
                for c in [hi, lo] {
                    reachable[c] = true;
                    parents[c].push(r);
                }
            }
        }

        // Dominators: parents come after their children, so a pass in
        // decreasing order sees all the parents of a route before it.
        let mut idom = vec![usize::MAX; n_routes];
        let mut depth = vec![0usize; n_routes];
        idom[arm.root] = arm.root;
        let lca = |idom: &[usize], depth: &[usize], mut a: usize, mut b: usize| {
            while a != b {
                if depth[a] >= depth[b] {
                    a = idom[a];
                } else {
                    b = idom[b];
                }
            }
            a
        };
        for r in (0..arm.root).rev() {
            if !reachable[r] {
                continue;
            }
            let mut d = parents[r][0];
            for &p in &parents[r][1..] {
                d = lca(&idom, &depth, d, p);
            }
            idom[r] = d;
            depth[r] = depth[d] + 1;
        }

        // Where each draw is bound: the lowest route dominating its uses.
        let mut site: Vec<Option<usize>> = vec![None; arm.nodes.len()];
        for r in (0..n_routes).filter(|&r| reachable[r]) {
            for v in used_values(r) {
                for &d in value_deps(&v) {
                    site[d] = Some(match site[d] {
                        None => r,
                        Some(s) => lca(&idom, &depth, s, r),
                    });
                }
            }
        }

        // The order of first use of the draws, in the emitter's traversal.
        let mut order = vec![usize::MAX; arm.nodes.len()];
        let mut next_order = 0;
        let mut seen_routes = vec![false; n_routes];
        let mut seen_nodes = vec![false; arm.nodes.len()];
        fn visit_value<O>(
            arm: &LocalArm<O>,
            producer: &[usize],
            v: Value,
            seen: &mut [bool],
            order: &mut [usize],
            next: &mut usize,
        ) {
            let n_inputs = arm.n_inputs as usize;
            if v.index() < n_inputs {
                return;
            }
            let n = producer[v.index() - n_inputs];
            if seen[n] {
                return;
            }
            seen[n] = true;
            for &a in &arm.nodes[n].args {
                visit_value(arm, producer, a, seen, order, next);
            }
            if arm.nodes[n].draw {
                order[n] = *next;
                *next += 1;
            }
        }
        let mut stack = vec![arm.root];
        while let Some(r) = stack.pop() {
            if seen_routes[r] {
                continue;
            }
            seen_routes[r] = true;
            for v in used_values(r) {
                visit_value(
                    arm,
                    &producer,
                    v,
                    &mut seen_nodes,
                    &mut order,
                    &mut next_order,
                );
            }
            if let Some((hi, lo)) = children(r) {
                stack.push(lo);
                stack.push(hi);
            }
        }

        let mut binds: Vec<Vec<usize>> = vec![Vec::new(); n_routes];
        for (d, s) in site.iter().enumerate() {
            if let Some(s) = s {
                binds[*s].push(d);
            }
        }
        for b in &mut binds {
            b.sort_by_key(|&d| order[d]);
        }

        // Bottom up: the draws used, and those bound, at or below each route.
        let mut used: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n_routes];
        let mut bound: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n_routes];
        let mut free = vec![Vec::new(); n_routes];
        let mut binds_below = vec![false; n_routes];
        for r in (0..n_routes).filter(|&r| reachable[r]) {
            let mut u: BTreeSet<usize> = used_values(r)
                .iter()
                .flat_map(|v| value_deps(v).iter().copied())
                .collect();
            let mut b: BTreeSet<usize> = binds[r].iter().copied().collect();
            if let Some((hi, lo)) = children(r) {
                for c in [hi, lo] {
                    u.extend(used[c].iter().copied());
                    b.extend(bound[c].iter().copied());
                }
            }
            free[r] = u.difference(&b).copied().collect();
            binds_below[r] = !b.is_empty();
            used[r] = u;
            bound[r] = b;
        }

        Self {
            first_out,
            producer,
            node_deps,
            binds,
            free,
            binds_below,
        }
    }
}

/// An emitted arm: the arrays of a tape for a single arm, with ids and spans
/// local to the arm.
struct Emitted<O> {
    nodes: Vec<Node<O>>,
    args: Vec<Value>,
    routes: Vec<Route>,
    bound: Vec<NodeId>,
    outs: Vec<Value>,
}

/// The key under which an emitted route is hash-consed: its binders, and
/// the rest of the route.
#[derive(Clone, PartialEq, Eq, Hash)]
enum RouteKey {
    Split(Value, RouteId, RouteId),
    End(u32, Vec<Value>),
}

/// Levels of the draws bound so far on the current path.
type Levels = HashMap<usize, u32>;

/// One pass of canonicalization over an arm.
struct Emitter<'a, O> {
    arm: &'a LocalArm<O>,
    a: Analysis,
    out: Emitted<O>,
    next_value: u32,
    /// Emitted nodes, by kind and arguments: the node and its first output.
    nodes: HashMap<(NodeKind<O>, Vec<Value>), (NodeId, Value)>,
    /// Emitted routes, by binders and the rest.
    routes: HashMap<(Vec<NodeId>, RouteKey), RouteId>,
    /// Nodes already emitted for a node of the arm, under the levels of the
    /// draws it depends on.
    node_memo: HashMap<(usize, Vec<u32>), (NodeId, Value)>,
    /// Routes already emitted for a route of the arm, under its depth (if
    /// it binds draws) and the levels of the draws it uses but does not bind.
    route_memo: HashMap<(usize, Option<u32>, Vec<u32>), RouteId>,
    /// The route binding each emitted draw.
    bound_at: HashMap<NodeId, RouteId>,
    /// Whether another pass is needed.
    changed: bool,
}

impl<'a, O: Clone + Eq + Hash> Emitter<'a, O> {
    fn new(arm: &'a LocalArm<O>) -> Self {
        Self {
            arm,
            a: Analysis::new(arm),
            out: Emitted {
                nodes: Vec::new(),
                args: Vec::new(),
                routes: Vec::new(),
                bound: Vec::new(),
                outs: Vec::new(),
            },
            next_value: arm.n_inputs,
            nodes: HashMap::new(),
            routes: HashMap::new(),
            node_memo: HashMap::new(),
            route_memo: HashMap::new(),
            bound_at: HashMap::new(),
            changed: false,
        }
    }

    fn levels(deps: &[usize], env: &Levels) -> Vec<u32> {
        deps.iter()
            .map(|d| *env.get(d).expect("a draw is bound before it is used"))
            .collect()
    }

    /// Emits value `v` of the arm.
    fn value(&mut self, v: Value, env: &Levels) -> Value {
        let n_inputs = self.arm.n_inputs as usize;
        if v.index() < n_inputs {
            return v;
        }
        let n = self.a.producer[v.index() - n_inputs];
        let (_, first) = self.node(n, env);
        Value(first.0 + v.0 - self.a.first_out[n])
    }

    /// Emits node `n` of the arm, and returns it and its first output.
    fn node(&mut self, n: usize, env: &Levels) -> (NodeId, Value) {
        let key = (n, Self::levels(&self.a.node_deps[n], env));
        if let Some(&emitted) = self.node_memo.get(&key) {
            return emitted;
        }
        let arm = self.arm;
        let node = &arm.nodes[n];
        let args: Vec<Value> = node.args.iter().map(|&v| self.value(v, env)).collect();
        let kind = if node.draw {
            NodeKind::NonDet {
                op: node.op.clone(),
                level: env[&n],
            }
        } else {
            NodeKind::Det(node.op.clone())
        };
        let table_key = (kind, args);
        let emitted = match self.nodes.get(&table_key) {
            Some(&emitted) => emitted,
            None => {
                let id = NodeId(self.out.nodes.len() as u32);
                let first = Value(self.next_value);
                self.next_value += node.n_outs;
                let (kind, args) = table_key.clone();
                self.out.nodes.push(Node {
                    kind,
                    args: push_span(&mut self.out.args, args),
                    n_outs: node.n_outs,
                });
                self.nodes.insert(table_key, (id, first));
                (id, first)
            }
        };
        self.node_memo.insert(key, emitted);
        emitted
    }

    /// Emits a route with binders `binds`, unless an equal one exists.
    fn emit(&mut self, binds: Vec<NodeId>, key: RouteKey) -> RouteId {
        let table_key = (binds, key);
        if let Some(&id) = self.routes.get(&table_key) {
            return id;
        }
        let id = RouteId(self.out.routes.len() as u32);
        let (binds, key) = table_key.clone();
        for &n in &binds {
            if let Some(other) = self.bound_at.insert(n, id)
                && other != id
            {
                // the same draw bound on two paths: bind it once, above both
                self.changed = true;
            }
        }
        let binds = push_span(&mut self.out.bound, binds);
        let route = match key {
            RouteKey::Split(cond, hi, lo) => Route::Split {
                binds,
                cond,
                hi,
                lo,
            },
            RouteKey::End(output, outs) => Route::End {
                binds,
                output,
                outs: push_span(&mut self.out.outs, outs),
            },
        };
        self.out.routes.push(route);
        self.routes.insert(table_key, id);
        id
    }

    /// The emitted route `id`, as binders and the rest.
    fn key_of(&self, id: RouteId) -> (Vec<NodeId>, RouteKey) {
        let route = self.out.routes[id.index()];
        let binds = self.out.bound[route.binds().range()].to_vec();
        let key = match route {
            Route::Split { cond, hi, lo, .. } => RouteKey::Split(cond, hi, lo),
            Route::End { output, outs, .. } => {
                RouteKey::End(output, self.out.outs[outs.range()].to_vec())
            }
        };
        (binds, key)
    }

    /// Emits route `r` of the arm, reached after `depth` draws, with the
    /// draws bound so far at levels `env`.
    fn route(&mut self, r: usize, depth: u32, env: &Levels) -> RouteId {
        let free_levels = Self::levels(&self.a.free[r], env);
        let key = (r, self.a.binds_below[r].then_some(depth), free_levels);
        if let Some(&id) = self.route_memo.get(&key) {
            return id;
        }
        let binds = self.a.binds[r].clone();
        let mut env: Cow<'_, Levels> = Cow::Borrowed(env);
        if !binds.is_empty() {
            let env = env.to_mut();
            for (k, &d) in binds.iter().enumerate() {
                env.insert(d, depth + k as u32);
            }
        }
        let depth = depth + binds.len() as u32;
        let arm = self.arm;
        let id = match &arm.routes[r] {
            LRoute::End { output, outs } => {
                let outs = outs.iter().map(|&v| self.value(v, &env)).collect();
                let binds = binds.iter().map(|&d| self.node(d, &env).0).collect();
                self.emit(binds, RouteKey::End(*output, outs))
            }
            &LRoute::Split { cond, hi, lo } => {
                let cond = self.value(cond, &env);
                let hi = self.route(hi, depth, &env);
                let lo = self.route(lo, depth, &env);
                let binds: Vec<NodeId> = binds.iter().map(|&d| self.node(d, &env).0).collect();
                if hi == lo {
                    // a split with equal children: keep the child, with these
                    // binders added in front, and canonicalize again
                    self.changed = true;
                    if binds.is_empty() {
                        hi
                    } else {
                        let (child_binds, child) = self.key_of(hi);
                        self.emit(binds.into_iter().chain(child_binds).collect(), child)
                    }
                } else {
                    self.emit(binds, RouteKey::Split(cond, hi, lo))
                }
            }
        };
        self.route_memo.insert(key, id);
        id
    }
}
