//! The well-formedness check of [`Tape::new`](super::Tape::new).
//!
//! The check is one forward pass over each arm's nodes, which computes
//! the sort of every value and the draws it depends on. It is followed by
//! one pass over each arm's routes from the root down (decreasing
//! index, since children come before parents), which computes how many draws
//! precede each route and which draws are available there on every path.

use super::{Arm, NodeId, NodeKind, Operation, Route, RouteId, Span, TapeParts, Value};
use base::boundary::{Mono, Poly};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;

type Draws = BTreeSet<NodeId>;

pub(super) fn check<S, O>(p: &TapeParts<S, O>) -> Result<(), String>
where
    S: Clone + PartialEq + Debug,
    O: Operation<S> + Debug,
{
    let inputs: Vec<&Mono<S>> = p.dom.iter().collect();
    let outputs: Vec<&Mono<S>> = p.cod.iter().collect();
    if p.arms.len() != inputs.len() {
        return Err(format!(
            "the domain has {} summands, but there are {} arms",
            inputs.len(),
            p.arms.len()
        ));
    }

    let (mut next_node, mut next_route) = (0, 0);
    for (j, (arm, input)) in p.arms.iter().zip(&inputs).enumerate() {
        if arm.nodes.start as usize != next_node {
            return Err(format!("arm {j}: its nodes must start at n{next_node}"));
        }
        if arm.routes.start as usize != next_route {
            return Err(format!("arm {j}: its routes must start at r{next_route}"));
        }
        if arm.routes.is_empty() {
            return Err(format!("arm {j} has no routes"));
        }
        next_node = arm.nodes.end();
        next_route = arm.routes.end();
        if next_node > p.nodes.len() || next_route > p.routes.len() {
            return Err(format!("arm {j}: its spans go past the end of the tape"));
        }
        ArmCheck::new(p, j, arm, input, &outputs)?.routes()?;
    }
    if next_node != p.nodes.len() {
        return Err(format!("n{next_node} and later belong to no arm"));
    }
    if next_route != p.routes.len() {
        return Err(format!("r{next_route} and later belong to no arm"));
    }
    Ok(())
}

/// The slice of `array` under `span`, or an error if it goes past the end.
fn slice<'a, T>(array: &'a [T], span: Span, what: &str) -> Result<&'a [T], String> {
    array
        .get(span.range())
        .ok_or_else(|| format!("{what}: span {:?} goes past the end", span.range()))
}

/// The base types of a single product, or an error if `poly` is not one.
fn single<S: Clone + Debug>(poly: Poly<S>, what: &str) -> Result<Vec<S>, String> {
    let mut summands = poly.into_iter();
    match (summands.next(), summands.next()) {
        (Some(m), None) => Ok(m.into_iter().collect()),
        _ => Err(format!("{what} is not a single product")),
    }
}

/// What the node pass learns about one arm.
struct ArmCheck<'a, S, O> {
    parts: &'a TapeParts<S, O>,
    index: usize,
    arm: &'a Arm,
    outputs: &'a [&'a Mono<S>],
    /// The sort of each value of the arm.
    sorts: Vec<S>,
    /// The draws each value depends on, directly or through other nodes.
    deps: Vec<Draws>,
    /// For each draw of the arm: its level, and the draws its arguments
    /// depend on.
    draws: BTreeMap<NodeId, (u32, Draws)>,
}

impl<'a, S, O> ArmCheck<'a, S, O>
where
    S: Clone + PartialEq + Debug,
    O: Operation<S> + Debug,
{
    /// Checks the nodes of the arm.
    fn new(
        parts: &'a TapeParts<S, O>,
        index: usize,
        arm: &'a Arm,
        input: &Mono<S>,
        outputs: &'a [&'a Mono<S>],
    ) -> Result<Self, String> {
        let sorts: Vec<S> = input.iter().cloned().collect();
        let mut this = Self {
            parts,
            index,
            arm,
            outputs,
            deps: vec![Draws::new(); sorts.len()],
            sorts,
            draws: BTreeMap::new(),
        };
        for n in arm.nodes.range() {
            this.node(NodeId(n as u32))?;
        }
        Ok(this)
    }

    /// Checks one node, then records the values it defines.
    fn node(&mut self, id: NodeId) -> Result<(), String> {
        let node = &self.parts.nodes[id.index()];
        let args = slice(&self.parts.args, node.args, &format!("{id}: arguments"))?;
        let mut deps = Draws::new();
        let mut arg_sorts = Vec::with_capacity(args.len());
        for &a in args {
            let sort = self
                .sorts
                .get(a.index())
                .ok_or_else(|| format!("{id}: argument {a} is not defined before the node"))?;
            arg_sorts.push(sort.clone());
            deps.extend(self.deps[a.index()].iter().copied());
        }

        let op = node.kind.op();
        let det = matches!(node.kind, NodeKind::Det(_));
        if op.is_deterministic() != det {
            let which = if det {
                "non-deterministic"
            } else {
                "deterministic"
            };
            return Err(format!("{id}: {op:?} is {which}"));
        }
        let dom = single(op.dom(), &format!("{id}: the domain of {op:?}"))?;
        let cod = single(op.cod(), &format!("{id}: the codomain of {op:?}"))?;
        if arg_sorts != dom {
            return Err(format!(
                "{id}: the arguments have sorts {arg_sorts:?}, but the node reads {dom:?}"
            ));
        }
        if node.n_outs as usize != cod.len() {
            return Err(format!(
                "{id}: n_outs is {}, but the node writes {cod:?}",
                node.n_outs
            ));
        }

        if let NodeKind::NonDet { level, .. } = node.kind {
            self.draws.insert(id, (level, deps.clone()));
            deps.insert(id);
        }
        for sort in cod {
            self.sorts.push(sort);
            self.deps.push(deps.clone());
        }
        Ok(())
    }

    /// The sort of `v`, or an error if `v` is not a value of the arm.
    fn sort(&self, v: Value, what: &str) -> Result<&S, String> {
        self.sorts
            .get(v.index())
            .ok_or_else(|| format!("{what}: {v} is not a value of arm {}", self.index))
    }

    /// Checks the routes of the arm.
    fn routes(&self) -> Result<(), String> {
        let span = self.arm.routes;
        let first = span.start as usize;
        let end = span.end();

        // Local conditions, and where each draw is bound.
        let mut bound_at: BTreeMap<NodeId, RouteId> = BTreeMap::new();
        for d in first..end {
            let id = RouteId(d as u32);
            let route = &self.parts.routes[d];
            for &n in slice(&self.parts.bound, route.binds(), &format!("{id}: binders"))? {
                if !self.draws.contains_key(&n) {
                    return Err(format!(
                        "{id}: binds {n}, which is not a non-deterministic node of arm {}",
                        self.index
                    ));
                }
                if let Some(other) = bound_at.insert(n, id) {
                    return Err(format!("{n} is bound twice, at {other} and at {id}"));
                }
            }
            match *route {
                Route::Split { cond, hi, lo, .. } => {
                    let sort = self.sort(cond, &format!("{id}: condition"))?;
                    if *sort != self.parts.boolean {
                        return Err(format!(
                            "{id}: condition {cond} has sort {sort:?}, not the boolean sort"
                        ));
                    }
                    for child in [hi, lo] {
                        if child.index() < first || child.index() >= d {
                            return Err(format!(
                                "{id}: child {child} is not an earlier route of arm {}",
                                self.index
                            ));
                        }
                    }
                }
                Route::End { output, outs, .. } => {
                    let summand = self
                        .outputs
                        .get(output as usize)
                        .ok_or_else(|| format!("{id}: there is no output summand {output}"))?;
                    let outs = slice(&self.parts.outs, outs, &format!("{id}: values"))?;
                    let mut sorts = Vec::with_capacity(outs.len());
                    for &v in outs {
                        sorts.push(self.sort(v, &format!("{id}: value"))?.clone());
                    }
                    let expected: Vec<S> = summand.iter().cloned().collect();
                    if sorts != expected {
                        return Err(format!(
                            "{id}: the values have sorts {sorts:?}, but output {output} is {expected:?}"
                        ));
                    }
                }
            }
        }
        if let Some(n) = self.draws.keys().find(|n| !bound_at.contains_key(n)) {
            return Err(format!("{n} is never bound"));
        }

        // Bottom up: whether each route, or one below it, binds a draw.
        // Only for those does the number of draws before them matter.
        let mut binds_below = vec![false; end - first];
        for d in first..end {
            let route = &self.parts.routes[d];
            binds_below[d - first] = !route.binds().is_empty()
                || route.children().is_some_and(|(hi, lo)| {
                    binds_below[hi.index() - first] || binds_below[lo.index() - first]
                });
        }

        // From the root down: the number of draws before each route (the
        // same on every path, where it matters), and the draws made on every
        // path to it.
        let mut depth: Vec<Option<u32>> = vec![None; end - first];
        let mut available: Vec<Draws> = vec![Draws::new(); end - first];
        depth[end - 1 - first] = Some(0);
        for d in (first..end).rev() {
            let id = RouteId(d as u32);
            let Some(mut level) = depth[d - first] else {
                return Err(format!("{id} is not reachable from the root of its arm"));
            };
            let route = &self.parts.routes[d];
            let mut here = std::mem::take(&mut available[d - first]);
            for &n in &self.parts.bound[route.binds().range()] {
                let (expected, args) = &self.draws[&n];
                if *expected != level {
                    return Err(format!(
                        "{n} has level {expected}, but {id} binds it after {level} draws"
                    ));
                }
                if !args.is_subset(&here) {
                    return Err(format!(
                        "{id}: {n} is drawn before a draw its arguments depend on"
                    ));
                }
                here.insert(n);
                level += 1;
            }
            match *route {
                Route::Split { cond, hi, lo, .. } => {
                    if !self.deps[cond.index()].is_subset(&here) {
                        return Err(format!(
                            "{id}: condition {cond} depends on a draw not made on every path"
                        ));
                    }
                    for child in [hi, lo] {
                        let k = child.index() - first;
                        match depth[k] {
                            None => {
                                depth[k] = Some(level);
                                available[k] = here.clone();
                            }
                            Some(other) if other != level && binds_below[k] => {
                                return Err(format!(
                                    "{child} binds draws (itself or below it), but is reached \
                                     after {other} draws on one path and after {level} on another"
                                ));
                            }
                            Some(_) => {
                                available[k] = available[k].intersection(&here).copied().collect();
                            }
                        }
                    }
                }
                Route::End { outs, .. } => {
                    for &v in &self.parts.outs[outs.range()] {
                        if !self.deps[v.index()].is_subset(&here) {
                            return Err(format!(
                                "{id}: value {v} depends on a draw not made on every path"
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
