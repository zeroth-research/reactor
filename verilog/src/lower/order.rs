//! The order of the atoms of a module: by what each awaits, with a message for a loop.

use super::Result;
use super::builder::lvalue_targets;
use crate::ast::{Error, Expr, Item, Name, Reset, Span, Stmt, StmtKind};
use std::collections::{BTreeMap, BTreeSet};

/// An item before sorting, with the signals it controls and those it
/// awaits.
pub(super) struct Pending<'m> {
    pub(super) item: &'m Item,
    /// The asynchronous reset of an `always_ff` block, by canonical name.
    pub(super) reset: Option<Name>,
    pub(super) ctrl: Vec<Name>,
    pub(super) awaits: BTreeSet<Name>,
    /// For an `always_comb` driving several signals, what each depends on.
    pub(super) deps: Option<BTreeMap<Name, BTreeSet<Name>>>,
    pub(super) span: Span,
}

impl Pending<'_> {
    /// A signal that `self` awaits and `other` controls.
    fn awaits_from<'p>(&self, other: &'p Pending) -> Option<&'p Name> {
        other.ctrl.iter().find(|x| self.awaits.contains(*x))
    }
}

/// Sorts the items so that each comes after the items controlling what it
/// awaits, keeping the source order where possible; a cycle is an error.
pub(super) fn await_order<'m>(mut pending: Vec<Pending<'m>>) -> Result<Vec<&'m Item>> {
    let mut sorted = Vec::new();
    while !pending.is_empty() {
        let ready = |a: &Pending| !pending.iter().any(|b| a.awaits_from(b).is_some());
        match pending.iter().position(ready) {
            Some(i) => sorted.push(pending.remove(i).item),
            None => return Err(combinational_loop(&pending)),
        }
    }
    Ok(sorted)
}

/// An error naming a cycle among items none of which is ready, and a block
/// through which the cycle passes only because the block is one atom.
pub(super) fn combinational_loop(pending: &[Pending]) -> Error {
    // every item awaits some other pending item: walk until one repeats,
    // recording the signal by which each item is entered
    let mut path: Vec<(usize, Option<&Name>)> = vec![(0, None)];
    loop {
        let at = path.last().unwrap().0;
        let (next, x) = pending
            .iter()
            .enumerate()
            .find_map(|(i, b)| pending[at].awaits_from(b).map(|x| (i, x)))
            .expect("not ready");
        if let Some(start) = path.iter().position(|&(i, _)| i == next) {
            // the items of the cycle, each with the signal it controls and
            // the one it awaits from the next item
            let items: Vec<usize> = path[start..].iter().map(|&(i, _)| i).collect();
            let mut signals: Vec<&Name> = vec![x];
            signals.extend(path[start + 1..].iter().map(|(_, x)| x.unwrap()));
            let cycle: Vec<&str> = signals.iter().chain([&x]).map(|x| x.as_str()).collect();
            let mut msg = format!("combinational loop: {}", cycle.join(" -> "));
            // a register atom awaits only its clock and its reset: the
            // loop goes through the edge test or the reset
            if let Some(k) = items
                .iter()
                .position(|&i| matches!(pending[i].item, Item::AlwaysFf { .. }))
            {
                let (i, awaited) = (items[k], signals[(k + 1) % signals.len()]);
                let what = if pending[i].reset.as_ref() == Some(awaited) {
                    "reset"
                } else {
                    "clock"
                };
                msg += &format!(
                    " (through the {what} of the {} at {})",
                    pending[i].item.block(),
                    pending[i].span
                );
            }
            for (k, &i) in items.iter().enumerate() {
                let (ctrl, awaited) = (signals[k], signals[(k + 1) % signals.len()]);
                let Some(deps) = &pending[i].deps else {
                    continue;
                };
                if !deps[ctrl].contains(awaited) {
                    msg += &format!(
                        "; in the {} at {}, `{ctrl}` does not depend on `{awaited}`, \
                         but the block is translated as a whole: split it",
                        pending[i].item.block(),
                        pending[i].span
                    );
                    break;
                }
            }
            return Error::new(pending[next].span, msg);
        }
        path.push((next, Some(x)));
    }
}

/// The asynchronous reset of an `always_ff` block and its reset values, if
/// it resets any register.
pub(super) fn reset_of(item: &Item) -> Option<(&Reset, Vec<(&Name, &Expr)>)> {
    let Item::AlwaysFf {
        reset: Some(r),
        body,
        ..
    } = item
    else {
        return None;
    };
    let values = body.reset_values();
    (!values.is_empty()).then_some((r, values))
}

/// For each signal an `always_comb` block drives, the signals outside the
/// block on which its final value depends, through data or control.
pub(super) fn dependencies(body: &Stmt) -> BTreeMap<Name, BTreeSet<Name>> {
    let mut deps = BTreeMap::new();
    depend(body, &BTreeSet::new(), &mut deps);
    deps
}

/// Updates `deps` (the current dependencies of the signals assigned so
/// far) by `s`, run under conditions depending on `ctrl`.
pub(super) fn depend(s: &Stmt, ctrl: &BTreeSet<Name>, deps: &mut BTreeMap<Name, BTreeSet<Name>>) {
    let resolve = |e: &Expr, deps: &BTreeMap<Name, BTreeSet<Name>>| {
        let mut reads = Vec::new();
        e.reads(&mut reads);
        let mut out = BTreeSet::new();
        for (x, _) in reads {
            match deps.get(x) {
                Some(d) => out.extend(d.iter().cloned()),
                None => {
                    out.insert(x.clone());
                }
            }
        }
        out
    };
    // runs each branch under `ctrl` and joins the results; `fallthrough`
    // adds a path that runs no branch
    let join = |branches: &[&Stmt], ctrl, fallthrough, deps: &mut BTreeMap<_, _>| {
        let mut joined = if fallthrough {
            deps.clone()
        } else {
            BTreeMap::new()
        };
        for b in branches {
            let mut out = deps.clone();
            depend(b, ctrl, &mut out);
            for (y, d) in out {
                joined.entry(y).or_insert_with(BTreeSet::new).extend(d);
            }
        }
        *deps = joined;
    };
    match &s.kind {
        StmtKind::Block(stmts) => stmts.iter().for_each(|s| depend(s, ctrl, deps)),
        StmtKind::If(c, a, b) => {
            let mut inner = ctrl.clone();
            inner.extend(resolve(c, deps));
            join(&[a, b], &inner, false, deps);
        }
        StmtKind::Case {
            sel,
            items,
            default,
            ..
        } => {
            let mut inner = ctrl.clone();
            inner.extend(resolve(sel, deps));
            for l in items.iter().flat_map(|i| &i.labels) {
                inner.extend(resolve(l, deps));
            }
            let branches: Vec<&Stmt> = s.children();
            join(&branches, &inner, default.is_none(), deps);
        }
        StmtKind::NonBlocking { lhs, rhs } | StmtKind::Blocking { lhs, rhs } => {
            let mut d = resolve(rhs, deps);
            d.extend(ctrl.iter().cloned());
            // the bits a variable index writes depend on the index
            let mut index = Vec::new();
            lhs.reads(&mut index);
            for (x, _) in index {
                match deps.get(x) {
                    Some(dx) => d.extend(dx.iter().cloned()),
                    None => {
                        d.insert(x.clone());
                    }
                }
            }
            // a select keeps the dependencies of the other bits
            let mut whole = Vec::new();
            let mut targets = Vec::new();
            lvalue_targets(lhs, &mut whole, &mut targets);
            for y in targets {
                let entry = deps.entry(y.clone()).or_default();
                if !whole.contains(&y) {
                    entry.extend(d.iter().cloned());
                } else {
                    *entry = d.clone();
                }
            }
        }
        // no effect (dead, see `typing::definite`)
        StmtKind::Unknown { .. } => {}
    }
}
