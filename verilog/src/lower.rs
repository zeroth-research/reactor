/*!
# From the AST to a reactive module

Every signal becomes a variable of the width of its type: inputs are
external, outputs are interface variables, internal signals are private.

Every `assign` becomes a combinatorial atom that awaits the next values
`X(·)` of the signals it reads and writes `X(lhs)`. The expression becomes
one term per operator, with the source operators rewritten into [`SvOp`],
which has no `>`, `>=` and no logical operators:

| source | ops |
|---|---|
| `a > b`, `a >= b` | `ULt`/`SLt`, `ULe`/`SLe` with swapped operands |
| `<`, `<=` | `ULt`/`ULe`, or `SLt`/`SLe` for signed operands |
| `!a`, `a && b`, `a \|\| b` | `Not`/`And`/`Or` on `RedOr` of the operands (none for 1 bit) |
| `c ? a : b` | `Ite` on `RedOr c` (none for 1 bit) |
| `~^`, `~&`, `~\|` | `Not` of `Xor`, `RedAnd`, `RedOr` |
| `===`, `!==` | `Eq`, `Ne` |
| unary `+` | nothing |

Every `always_ff` block becomes a sequential atom controlling the
registers it assigns. One round of the module is one clock cycle, and
the clock itself is no variable: its rising edge is the step from one
round to the next. So the update reads the *latched* values `*x` (the
values of the previous cycle, just before the edge) and writes the next
values `X(q)` (the values after the edge). It awaits nothing, so a
register never closes a combinational loop. The statements are executed
symbolically with a map from each register to the wire holding its value
after the edge so far, starting with `*q` (a register keeps its value on
paths that do not assign it). `q <= e` sets the entry of `q` to the wire of
`e`, so the last assignment wins; at `if`, both branches start from the
same map and the entries that differ are merged with `Ite`. The init
action sets every register to an external variable `q@init`, so all
initial states are possible.

Every `always_comb` block becomes a combinatorial atom controlling the
signals it assigns. Like an `assign`, it awaits the next values `X(x)`
of the signals it reads, except its own: the statements are executed
symbolically with a map from each of its signals to the wire holding its
current value (SSA, the intermediate values are printed `y@1`, `y@2`, …),
so a read of its own signal reads that wire. The map starts empty; typing
has made sure that every read and the end of the block find an entry on
every path (no stale reads, no latches). `if` and `case` merge entries as in
`always_ff`; an entry missing in a branch is missing after the merge.

`case (s) l1, l2: a … default: d endcase` becomes a chain of `Ite`s on
`s == l1 | s == l2`, …, so the first matching item wins; without `default`,
no item matching leaves the map as it was.

The atoms are sorted so that each comes after the atoms controlling what it
awaits; a cycle is a combinational loop and is rejected. The check is per
atom, as the await relation of reactive modules is: an atom's update
is one action, so it waits for everything any of its terms reads. An
`always_comb` block driving several signals may therefore close a loop
that its data flow does not have (it drives `x` from `a` and `y` from `z`,
while another block drives `z` from `x`). Such designs are rejected, with
a message saying which block to split. Translating each signal of a block
as its own atom would accept them with the same meaning, at the cost of
duplicating shared subterms and of the one-to-one correspondence between
blocks and atoms.
*/

use crate::ast::{
    self, BinOp, Dir, Error, Expr, ExprKind, Item, Name, Span, Stmt, StmtKind, Ty, UnOp,
};
use crate::typing;
use crate::{Atom, Module, Sort, SvOp, Term, Var, Wire};
use base::var::X;
use std::collections::{BTreeMap, BTreeSet, HashMap};

type Result<T> = std::result::Result<T, Error>;

/// A module with the names of its variables.
pub struct Translation {
    pub name: Name,
    pub module: Module,
    /// Every source signal, and `q@init` for every register `q`, by name.
    pub vars: BTreeMap<Name, Var>,
    /// The clock of the `always_ff` blocks; it is not a variable of the
    /// module, since a round is a whole clock cycle.
    pub clock: Option<Name>,
    /// The signals assigned in `always_ff` blocks, in source order.
    pub registers: Vec<Name>,
    /// Inputs other than the clock that nothing reads; they are not
    /// variables of the module.
    pub unread_inputs: Vec<Name>,
    /// Names of wires local to an atom: the intermediate values `y@1`,
    /// `y@2`, … of the signals of `always_comb` blocks.
    pub wire_names: HashMap<Wire, Name>,
}

/// The name of the external variable holding the initial value of the
/// register `q`. Generated names contain `@`, which SystemVerilog
/// identifiers cannot, so they never collide with source names.
pub fn init_name(q: &str) -> Name {
    format!("{q}@init")
}

impl Translation {
    /// The name of a variable of the module.
    pub fn name(&self, var: &Var) -> &str {
        self.vars
            .iter()
            .find(|(_, v)| *v == var)
            .map_or("?", |(n, _)| n.as_str())
    }

    /// The variable of a signal; panics if there is none.
    pub fn var(&self, name: &str) -> Var {
        self.vars[name]
    }

    /// The module printed with the signal names.
    pub fn print(&self) -> String {
        crate::print::module(
            &self.module,
            &self.name,
            |v| self.name(v).to_string(),
            |w| self.wire_names.get(w).cloned(),
        )
    }
}

/// Checks and translates a module.
pub fn lower(m: &ast::Module) -> Result<Translation> {
    let env = typing::check(m)?;
    let clock = m.clock().cloned();
    let registers: Vec<Name> = m.registers().into_iter().cloned().collect();
    // in declaration order, so that variable ids (and printing) follow it
    let decls = m.ports.iter().map(|p| &p.decl).chain(&m.decls);
    let mut vars: BTreeMap<Name, Var> = decls
        .map(|d| (d.name.clone(), Var::new(Sort(d.ty.width))))
        .collect();
    for q in &registers {
        vars.insert(init_name(q), Var::new(Sort(env[q].ty.width)));
    }

    // the await order is computed on the AST, so that a loop is reported
    // as such before any term is built
    let pending: Vec<Pending> = m
        .items
        .iter()
        .map(|item| {
            let ctrl: Vec<Name> = item.targets().into_iter().cloned().collect();
            let awaits = match item {
                // only the init action awaits, and only external variables
                Item::AlwaysFf { .. } => ctrl.iter().map(|q| init_name(q)).collect(),
                // `assign y = y` awaits itself, a block reads its own
                // signals from its local values
                Item::Assign { .. } => item.reads().into_iter().map(|(x, _)| x.clone()).collect(),
                Item::AlwaysComb { .. } => item
                    .reads()
                    .into_iter()
                    .map(|(x, _)| x)
                    .filter(|x| !ctrl.contains(x))
                    .cloned()
                    .collect(),
            };
            let deps = match item {
                Item::AlwaysComb { body, .. } if ctrl.len() > 1 => Some(dependencies(body)),
                _ => None,
            };
            Pending {
                item,
                ctrl,
                awaits,
                deps,
                span: item.span(),
            }
        })
        .collect();
    let read: BTreeSet<&Name> = m
        .items
        .iter()
        .flat_map(|item| item.reads().into_iter().map(|(x, _)| x))
        .collect();
    let unread_inputs = m
        .ports
        .iter()
        .map(|p| &p.decl.name)
        .filter(|x| env[*x].dir == Some(Dir::Input) && !read.contains(x))
        .filter(|x| Some(*x) != clock.as_ref())
        .cloned()
        .collect();

    let mut atoms = Vec::new();
    let mut wire_names = HashMap::new();
    for item in await_order(pending)? {
        let span = item.span();
        let atom = match item {
            Item::Assign { lhs, rhs, .. } => {
                let mut b = Builder::new(&env, &vars, false);
                b.used.insert(lhs.clone());
                b.expr(rhs, Some(X(vars[lhs])))?;
                let scope: Vec<&Var> = b.used.iter().map(|n| &vars[n]).collect();
                Atom::combinatorial(scope, b.terms).map_err(|e| internal(span, e))?
            }
            Item::AlwaysFf { body, .. } => {
                let regs = item.targets();
                let mut b = Builder::new(&env, &vars, true);
                b.used.extend(regs.iter().map(|q| (*q).clone()));
                b.cur = regs
                    .iter()
                    .map(|q| ((*q).clone(), Some(*vars[*q])))
                    .collect();
                b.stmt(body)?;
                b.finish_all(span)?;
                let mut init = Vec::new();
                for q in &regs {
                    let q0 = vars[&init_name(q)];
                    let term = Term::function(SvOp::Id, [X(vars[*q])], [X(q0)]);
                    init.push(term.map_err(|e| internal(span, e))?);
                    b.used.insert(init_name(q));
                }
                let scope: Vec<&Var> = b.used.iter().map(|n| &vars[n]).collect();
                Atom::sequential(scope, init, b.terms).map_err(|e| internal(span, e))?
            }
            Item::AlwaysComb { body, .. } => {
                let targets = item.targets();
                let mut b = Builder::new(&env, &vars, false);
                b.used.extend(targets.iter().map(|y| (*y).clone()));
                b.cur = targets.iter().map(|y| ((*y).clone(), None)).collect();
                b.stmt(body)?;
                b.finish_all(span)?;
                wire_names.extend(b.names);
                let scope: Vec<&Var> = b.used.iter().map(|n| &vars[n]).collect();
                Atom::combinatorial(scope, b.terms).map_err(|e| internal(span, e))?
            }
        };
        atoms.push(atom);
    }

    let private: BTreeSet<Var> = m.decls.iter().map(|d| vars[&d.name]).collect();
    let module = Module::partially_observable(atoms, |v| private.contains(v))
        .map_err(|e| internal(Span::default(), e))?;
    Ok(Translation {
        name: m.name.clone(),
        module,
        vars,
        clock,
        registers,
        unread_inputs,
        wire_names,
    })
}

fn internal(span: Span, e: String) -> Error {
    Error::new(span, format!("internal error: {e}"))
}

/// An item before sorting, with the signals it controls and those it
/// awaits.
struct Pending<'m> {
    item: &'m Item,
    ctrl: Vec<Name>,
    awaits: BTreeSet<Name>,
    /// For an `always_comb` driving several signals, what each depends on.
    deps: Option<BTreeMap<Name, BTreeSet<Name>>>,
    span: Span,
}

impl Pending<'_> {
    /// A signal that `self` awaits and `other` controls.
    fn awaits_from<'p>(&self, other: &'p Pending) -> Option<&'p Name> {
        other.ctrl.iter().find(|x| self.awaits.contains(*x))
    }
}

/// Sorts the items so that each comes after the items controlling what it
/// awaits, keeping the source order where possible; a cycle is an error.
fn await_order<'m>(mut pending: Vec<Pending<'m>>) -> Result<Vec<&'m Item>> {
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
fn combinational_loop(pending: &[Pending]) -> Error {
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
            for (k, &i) in items.iter().enumerate() {
                let (ctrl, awaited) = (signals[k], signals[(k + 1) % signals.len()]);
                let Some(deps) = &pending[i].deps else {
                    continue;
                };
                if !deps[ctrl].contains(awaited) {
                    msg += &format!(
                        "; in the `always_comb` at {}, `{ctrl}` does not depend on `{awaited}`, \
                         but the block is translated as a whole: split it",
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

/// For each signal an `always_comb` block drives, the signals outside the
/// block on which its final value depends, through data or control.
fn dependencies(body: &Stmt) -> BTreeMap<Name, BTreeSet<Name>> {
    let mut deps = BTreeMap::new();
    depend(body, &BTreeSet::new(), &mut deps);
    deps
}

/// Updates `deps` (the current dependencies of the signals assigned so
/// far) by `s`, run under conditions depending on `ctrl`.
fn depend(s: &Stmt, ctrl: &BTreeSet<Name>, deps: &mut BTreeMap<Name, BTreeSet<Name>>) {
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
            deps.insert(lhs.clone(), d);
        }
    }
}

/// Builds the terms of one atom.
struct Builder<'a> {
    env: &'a typing::Env,
    vars: &'a BTreeMap<Name, Var>,
    terms: Vec<Term>,
    /// The signals the atom reads or writes.
    used: BTreeSet<Name>,
    /// Whether signals are read latched (`*x`, in `always_ff`) or awaited
    /// (`X(x)`).
    latched: bool,
    /// The wire holding the value of each signal the block drives so far:
    /// the value after the edge in `always_ff`; in `always_comb`, the
    /// current value, or `None` if the signal may not be assigned yet.
    cur: Vec<(Name, Option<Wire>)>,
    /// The names of the intermediate values in `always_comb`.
    names: HashMap<Wire, Name>,
    /// The number of intermediate values named so far, per signal.
    versions: BTreeMap<Name, usize>,
    /// The wires that [`finish`](Self::finish) renamed, to the variable
    /// wire they became.
    renamed: HashMap<Wire, Wire>,
}

impl<'a> Builder<'a> {
    fn new(env: &'a typing::Env, vars: &'a BTreeMap<Name, Var>, latched: bool) -> Self {
        Self {
            env,
            vars,
            terms: Vec::new(),
            used: BTreeSet::new(),
            latched,
            cur: Vec::new(),
            names: HashMap::new(),
            versions: BTreeMap::new(),
            renamed: HashMap::new(),
        }
    }

    /// Lowers a statement of a block, updating [`cur`](Self::cur).
    fn stmt(&mut self, s: &Stmt) -> Result<()> {
        match &s.kind {
            StmtKind::Block(stmts) => stmts.iter().try_for_each(|s| self.stmt(s)),
            StmtKind::If(c, a, b) => {
                let (wc, tc) = self.expr(c, None)?;
                let wc = self.truth(wc, tc, c.span)?;
                let before = self.cur.clone();
                self.stmt(a)?;
                let then = std::mem::replace(&mut self.cur, before);
                self.stmt(b)?;
                let els = std::mem::take(&mut self.cur);
                self.cur = self.merge(wc, then, els, s.span)?;
                Ok(())
            }
            StmtKind::Case {
                sel,
                items,
                default,
            } => {
                // whether each item matches, from the values before the case
                let (ws, _) = self.expr(sel, None)?;
                let mut hits = Vec::new();
                for item in items {
                    let mut hit = None;
                    for l in &item.labels {
                        let (wl, _) = self.expr(l, None)?;
                        let eq = self.emit(SvOp::Eq, &[ws, wl], Sort(1), None, l.span)?;
                        hit = Some(match hit {
                            None => eq,
                            Some(h) => self.emit(SvOp::Or, &[h, eq], Sort(1), None, l.span)?,
                        });
                    }
                    hits.push(hit.expect("a label"));
                }
                let before = self.cur.clone();
                let mut outs = Vec::new();
                for item in items {
                    self.cur = before.clone();
                    self.stmt(&item.body)?;
                    outs.push(std::mem::take(&mut self.cur));
                }
                self.cur = before;
                if let Some(d) = default {
                    self.stmt(d)?;
                }
                let mut acc = std::mem::take(&mut self.cur);
                for (hit, out) in hits.into_iter().zip(outs).rev() {
                    acc = self.merge(hit, out, acc, s.span)?;
                }
                self.cur = acc;
                Ok(())
            }
            StmtKind::NonBlocking { lhs, rhs } | StmtKind::Blocking { lhs, rhs } => {
                let (w, _) = self.expr(rhs, None)?;
                self.set(lhs, w);
                Ok(())
            }
        }
    }

    /// Makes `w` the current value of the signal `y`.
    fn set(&mut self, y: &Name, w: Wire) {
        self.name(y, w);
        let entry = self.cur.iter_mut().find(|(x, _)| x == y).expect("a target");
        entry.1 = Some(w);
    }

    /// Names `w` as the next intermediate value of `y` in `always_comb`,
    /// unless it is not local or already has a name.
    fn name(&mut self, y: &Name, w: Wire) {
        if !self.latched
            && !self.names.contains_key(&w)
            && self.terms.iter().any(|t| t.write() == [w])
        {
            let k = self.versions.entry(y.clone()).or_insert(0);
            *k += 1;
            self.names.insert(w, format!("{y}@{k}"));
        }
    }

    /// The map after `if (c)` with the maps `then` and `els` after the two
    /// branches: `Ite` where they differ, `None` where either is `None`.
    fn merge(
        &mut self,
        c: Wire,
        then: Vec<(Name, Option<Wire>)>,
        els: Vec<(Name, Option<Wire>)>,
        span: Span,
    ) -> Result<Vec<(Name, Option<Wire>)>> {
        let mut out = Vec::new();
        for ((y, a), (_, b)) in then.into_iter().zip(els) {
            let w = match (a, b) {
                (Some(a), Some(b)) if a == b => Some(a),
                (Some(a), Some(b)) => {
                    let sort = Sort(self.env[&y].ty.width);
                    let w = self.emit(SvOp::Ite, &[c, a, b], sort, None, span)?;
                    self.name(&y, w);
                    Some(w)
                }
                _ => None,
            };
            out.push((y, w));
        }
        Ok(out)
    }

    /// Makes the next value `X(y)` of every signal of the block hold its
    /// final value.
    fn finish_all(&mut self, span: Span) -> Result<()> {
        for (y, w) in std::mem::take(&mut self.cur) {
            let w = w.ok_or_else(|| internal(span, format!("`{y}` not assigned on every path")))?;
            self.finish(w, X(self.vars[&y]), span)?;
        }
        Ok(())
    }

    /// Makes `dest` hold the value of `w`: renames `w` to `dest` if a term
    /// of this atom writes `w` and none reads it, else adds `dest := w`.
    /// A wire already renamed (two signals with the same final value) is
    /// copied from the variable it became.
    fn finish(&mut self, w: Wire, dest: Wire, span: Span) -> Result<()> {
        if let Some(&r) = self.renamed.get(&w) {
            self.emit(SvOp::Id, &[r], *w.dtype(), Some(dest), span)?;
            return Ok(());
        }
        let writer = self.terms.iter().position(|t| t.write() == [w]);
        let read = self.terms.iter().any(|t| t.read().contains(&w));
        match writer {
            Some(i) if !read => {
                let t = &self.terms[i];
                let op = t.itype().clone();
                let term = if t.read().is_empty() {
                    Term::constant(op, [dest])
                } else {
                    Term::function(op, [dest], t.read().iter().copied())
                };
                self.terms[i] = term.map_err(|e| internal(span, e))?;
                self.renamed.insert(w, dest);
                self.names.remove(&w);
            }
            _ => {
                self.emit(SvOp::Id, &[w], *w.dtype(), Some(dest), span)?;
            }
        }
        Ok(())
    }

    /// Adds the term `dest := op(reads)`, with a fresh wire of `sort` for
    /// `dest` if none is given.
    fn emit(
        &mut self,
        op: SvOp,
        reads: &[Wire],
        sort: Sort,
        dest: Option<Wire>,
        span: Span,
    ) -> Result<Wire> {
        let w = dest.unwrap_or_else(|| Wire::scalar(sort));
        let term = if reads.is_empty() {
            Term::constant(op, [w])
        } else {
            Term::function(op, [w], reads.iter().copied())
        };
        self.terms.push(term.map_err(|e| internal(span, e))?);
        Ok(w)
    }

    /// The wire holding whether `w` (of type `t`) is non-zero.
    fn truth(&mut self, w: Wire, t: Ty, span: Span) -> Result<Wire> {
        if t.width == 1 {
            Ok(w)
        } else {
            self.emit(SvOp::RedOr, &[w], Sort(1), None, span)
        }
    }

    /// Lowers `e`, writing its value to `dest` if given, and returns the
    /// wire holding the value and its type.
    fn expr(&mut self, e: &Expr, dest: Option<Wire>) -> Result<(Wire, Ty)> {
        let span = e.span;
        match &e.kind {
            ExprKind::Lit { ty, value } => {
                let w = self.emit(SvOp::Lit(value.clone()), &[], Sort(ty.width), dest, span)?;
                Ok((w, *ty))
            }
            ExprKind::Var(x) => {
                let t = typing::lookup(self.env, x, span)?.ty;
                self.used.insert(x.clone());
                let var = self.vars[x];
                let own = self.cur.iter().find(|(y, _)| y == x);
                let w = match own {
                    _ if self.latched => *var,
                    Some((_, w)) => w.ok_or_else(|| internal(span, format!("`{x}` unassigned")))?,
                    None => X(var),
                };
                match dest {
                    None => Ok((w, t)),
                    Some(_) => Ok((self.emit(SvOp::Id, &[w], Sort(t.width), dest, span)?, t)),
                }
            }
            ExprKind::Unary(op, a) => {
                let (wa, ta) = self.expr(a, None)?;
                let t = typing::unary(*op, ta);
                let s = Sort(t.width);
                let w = match op {
                    UnOp::Plus => match dest {
                        None => wa,
                        Some(_) => self.emit(SvOp::Id, &[wa], s, dest, span)?,
                    },
                    UnOp::BitNot => self.emit(SvOp::Not, &[wa], s, dest, span)?,
                    UnOp::Neg => self.emit(SvOp::Neg, &[wa], s, dest, span)?,
                    UnOp::LogNot => {
                        let b = self.truth(wa, ta, span)?;
                        self.emit(SvOp::Not, &[b], s, dest, span)?
                    }
                    UnOp::RedAnd => self.emit(SvOp::RedAnd, &[wa], s, dest, span)?,
                    UnOp::RedOr => self.emit(SvOp::RedOr, &[wa], s, dest, span)?,
                    UnOp::RedXor => self.emit(SvOp::RedXor, &[wa], s, dest, span)?,
                    UnOp::RedNand | UnOp::RedNor | UnOp::RedXnor => {
                        let red = match op {
                            UnOp::RedNand => SvOp::RedAnd,
                            UnOp::RedNor => SvOp::RedOr,
                            _ => SvOp::RedXor,
                        };
                        let r = self.emit(red, &[wa], s, None, span)?;
                        self.emit(SvOp::Not, &[r], s, dest, span)?
                    }
                };
                Ok((w, t))
            }
            ExprKind::Binary(op, a, b) => {
                let (wa, ta) = self.expr(a, None)?;
                let (wb, tb) = self.expr(b, None)?;
                let t = typing::binary(*op, ta, tb, span)?;
                let s = Sort(t.width);
                let (lt, le) = if ta.signed {
                    (SvOp::SLt, SvOp::SLe)
                } else {
                    (SvOp::ULt, SvOp::ULe)
                };
                let w = match op {
                    BinOp::BitAnd => self.emit(SvOp::And, &[wa, wb], s, dest, span)?,
                    BinOp::BitOr => self.emit(SvOp::Or, &[wa, wb], s, dest, span)?,
                    BinOp::BitXor => self.emit(SvOp::Xor, &[wa, wb], s, dest, span)?,
                    BinOp::BitXnor => {
                        let x = self.emit(SvOp::Xor, &[wa, wb], s, None, span)?;
                        self.emit(SvOp::Not, &[x], s, dest, span)?
                    }
                    BinOp::Add => self.emit(SvOp::Add, &[wa, wb], s, dest, span)?,
                    BinOp::Sub => self.emit(SvOp::Sub, &[wa, wb], s, dest, span)?,
                    BinOp::Mul => self.emit(SvOp::Mul, &[wa, wb], s, dest, span)?,
                    BinOp::Eq | BinOp::CaseEq => self.emit(SvOp::Eq, &[wa, wb], s, dest, span)?,
                    BinOp::Ne | BinOp::CaseNe => self.emit(SvOp::Ne, &[wa, wb], s, dest, span)?,
                    BinOp::Lt => self.emit(lt, &[wa, wb], s, dest, span)?,
                    BinOp::Le => self.emit(le, &[wa, wb], s, dest, span)?,
                    BinOp::Gt => self.emit(lt, &[wb, wa], s, dest, span)?,
                    BinOp::Ge => self.emit(le, &[wb, wa], s, dest, span)?,
                    BinOp::LogAnd | BinOp::LogOr => {
                        let x = self.truth(wa, ta, span)?;
                        let y = self.truth(wb, tb, span)?;
                        let op = if *op == BinOp::LogAnd {
                            SvOp::And
                        } else {
                            SvOp::Or
                        };
                        self.emit(op, &[x, y], s, dest, span)?
                    }
                    BinOp::Shl | BinOp::AShl => self.emit(SvOp::Shl, &[wa, wb], s, dest, span)?,
                    BinOp::Shr => self.emit(SvOp::LShr, &[wa, wb], s, dest, span)?,
                    BinOp::AShr => {
                        let op = if ta.signed { SvOp::AShr } else { SvOp::LShr };
                        self.emit(op, &[wa, wb], s, dest, span)?
                    }
                };
                Ok((w, t))
            }
            ExprKind::Cond(c, a, b) => {
                let (wc, tc) = self.expr(c, None)?;
                let wc = self.truth(wc, tc, span)?;
                let (wa, ta) = self.expr(a, None)?;
                let (wb, tb) = self.expr(b, None)?;
                let t = typing::cond(ta, tb, span)?;
                let w = self.emit(SvOp::Ite, &[wc, wa, wb], Sort(t.width), dest, span)?;
                Ok((w, t))
            }
        }
    }
}
