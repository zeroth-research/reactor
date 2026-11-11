/*!
# The reference semantics on the AST

[`Interp::step`] runs one round, a time step: given the values of the
inputs in the round, clocks included, it returns the value of every signal
in the round. Every signal is computed on demand from its driver. The
value of a register in a round is decided by its `always_ff` block: the
block runs if the edge of its clock happens in the round, that is, its
clock was 0 in the previous round and is 1 now (`posedge`), or 1 and now
0 (`negedge`). The level of the clock now is computed first, like any
signal, so a derived clock (a register, or computed from one) has its new
value. A block runs on the values of the previous round, the values just
before the edge, and its registers take the values it assigns, the last
assignment to a register winning; a register no executed assignment
targets keeps its value. In the first round no edge happens: the
registers have their initial values. In any round, including the first,
in which the asynchronous reset of a block is active (its level now is 1,
or 0 for `negedge rst_n`), the registers the reset branch assigns have
their reset values instead, whether or not the edge happens; the reset
branch still runs at an edge like the rest of the body, reading the reset
from the previous round, so an edge in the round where the reset is
released keeps the reset values.

A signal driven by an `always_comb` block is computed by running the whole
block when one of its signals is first needed, with blocking assignments:
a read of a signal of the block sees its latest assignment (typing makes
sure there is one). It gives meaning to the source operators directly
on numbers (unsigned or two's complement as the types say), with the
widths and signedness of IEEE 1800 worked out on the source expression in
the standard's two phases: the self-determined type of every expression
bottom up, then the type of the context passed down to the
context-determined operands, where values are extended (§11.6 to §11.8;
an assignment evaluates its right-hand side at the larger of both widths
and truncates). This is independent of [`sizing`](crate::sizing) (which
rewrites the expression by bidirectional rules), of [`SvOp`] and of the
rewrites in lowering, so that tests can compare it with the translated
module.

A hierarchy runs as one design: every signal is known by its name (see
[`Inst::naming`]), so a port connected to a signal is that signal, and an
input connected to another expression is computed from it on demand.
Register initial values are given by path (`u.q`).

[`SvOp`]: crate::SvOp
*/

use crate::ast::{
    BinOp, Cast, Dir, Edge, Error, Expr, ExprKind, Item, Name, Select, Span, Stmt, StmtKind, Ty,
    UnOp,
};
use crate::elab::{self, Inst};
use crate::typing::{self, Env};
use crate::value::BitVec;
use num_bigint::{BigInt, BigUint};
use num_traits::{One, ToPrimitive, Zero};
use std::collections::{BTreeMap, BTreeSet};

/// An `always_ff` block: its instance, the item, the name and edge of its
/// clock, and the name and edge of its asynchronous reset.
type Block<'m> = (usize, &'m Item, Name, Edge, Option<(Name, Edge)>);

/// A value for each of some signals.
pub type Values = BTreeMap<Name, BitVec>;

pub struct Interp<'m> {
    /// One context per instance, the top module's first.
    ctxs: Vec<Ctx>,
    /// The driver of each signal driven by an `assign`, an `always_comb`
    /// or an instance connection, with the instance it is in.
    driver: BTreeMap<Name, (usize, Driver<'m>)>,
    /// The `always_ff` blocks.
    blocks: Vec<Block<'m>>,
    /// Every signal, by name, with its type.
    signals: BTreeMap<Name, Ty>,
    /// The inputs of the top module.
    inputs: Vec<(Name, Ty)>,
    /// The registers: path, name and type.
    registers: Vec<(Name, Name, Ty)>,
    /// The initial value of every register, by name.
    init: Values,
    /// The values of the previous round; none before the first round.
    prev: Option<Values>,
}

/// The signals of an instance.
struct Ctx {
    env: Env,
    /// The name of each signal of the instance (see [`Inst::naming`]).
    names: BTreeMap<Name, Name>,
}

#[derive(Clone, Copy)]
enum Driver<'m> {
    Item(&'m Item),
    /// An input of an instance connected to this expression of the
    /// instantiating module.
    Conn(&'m Expr),
    /// A register, assigned by the `always_ff` block with this index in
    /// [`Interp::blocks`].
    Ff(usize),
}

impl<'m> Interp<'m> {
    /// Starts with `init` giving a value for every register, by path (and
    /// nothing else).
    pub fn new(inst: &'m Inst, init: &Values) -> Result<Self, Error> {
        let naming: BTreeMap<Name, Name> = inst.naming().into_iter().collect();
        let mut it = Interp {
            ctxs: Vec::new(),
            driver: BTreeMap::new(),
            blocks: Vec::new(),
            signals: BTreeMap::new(),
            inputs: Vec::new(),
            registers: Vec::new(),
            init: Values::new(),
            prev: None,
        };
        it.add(inst, "", &naming)?;
        for p in &inst.module.ports {
            if p.dir == Dir::Input {
                it.inputs.push((p.decl.name.clone(), p.decl.ty));
            }
        }
        let err = |msg: String| Error::new(Span::default(), msg);
        for (path, name, ty) in &it.registers {
            let v = init
                .get(path)
                .ok_or_else(|| err(format!("no initial value for register `{path}`")))?;
            if v.width() != ty.width {
                return Err(err(format!("register `{path}` given {} bits", v.width())));
            }
            it.init.insert(name.clone(), v.clone());
        }
        if let Some(x) = init
            .keys()
            .find(|x| !it.registers.iter().any(|r| r.0 == **x))
        {
            return Err(err(format!("`{x}` is not a register")));
        }
        Ok(it)
    }

    /// Adds the instance at path `prefix` and those below it.
    fn add(
        &mut self,
        inst: &'m Inst,
        prefix: &str,
        naming: &BTreeMap<Name, Name>,
    ) -> Result<(), Error> {
        let m = &inst.module;
        let env = typing::check(m).map_err(|e| elab::context(e, prefix))?;
        let c = self.ctxs.len();
        let names: BTreeMap<Name, Name> = env
            .keys()
            .map(|x| (x.clone(), naming[&format!("{prefix}{x}")].clone()))
            .collect();
        for (x, sig) in &env {
            self.signals.insert(names[x].clone(), sig.ty);
        }
        for item in &m.items {
            match item {
                Item::AlwaysFf {
                    clock, edge, reset, ..
                } => {
                    let b = self.blocks.len();
                    let reset = reset.as_ref().map(|r| (names[&r.name].clone(), r.edge));
                    self.blocks
                        .push((c, item, names[clock].clone(), *edge, reset));
                    for q in item.targets() {
                        let path = format!("{prefix}{q}");
                        self.registers.push((path, names[q].clone(), env[q].ty));
                        self.driver.insert(names[q].clone(), (c, Driver::Ff(b)));
                    }
                }
                Item::Assign { .. } | Item::AlwaysComb { .. } => {
                    for y in item.targets() {
                        self.driver
                            .insert(names[y].clone(), (c, Driver::Item(item)));
                    }
                }
                Item::Instance(i) => {
                    for (conn, f) in i.conns(Dir::Input) {
                        if let Some(e) = &conn.actual
                            && conn.shared().is_none()
                        {
                            let name = naming[&format!("{prefix}{}.{}", i.name, conn.port)].clone();
                            self.signals.insert(name.clone(), f.ty);
                            self.driver.insert(name, (c, Driver::Conn(e)));
                        }
                    }
                }
            }
        }
        self.ctxs.push(Ctx { env, names });
        for (u, child) in &inst.children {
            self.add(child, &format!("{prefix}{u}."), naming)?;
        }
        Ok(())
    }

    /// The registers, by path, with their types.
    pub fn registers(&self) -> Vec<(&Name, Ty)> {
        self.registers.iter().map(|(p, _, t)| (p, *t)).collect()
    }

    /// The inputs of the top module, clocks included, with their types.
    pub fn inputs(&self) -> &[(Name, Ty)] {
        &self.inputs
    }

    /// Runs one round, with `inputs` giving a value for every input,
    /// clocks included. Returns the value of every signal in this round,
    /// by name.
    pub fn step(&mut self, inputs: &Values) -> Result<Values, String> {
        let mut values = Values::new();
        for (name, ty) in &self.inputs {
            let v = inputs
                .get(name)
                .ok_or_else(|| format!("no value for input `{name}`"))?;
            if v.width() != ty.width {
                return Err(format!("input `{name}` given {} bits", v.width()));
            }
            values.insert(name.clone(), v.clone());
        }
        let mut active = BTreeSet::new();
        for name in self.signals.keys() {
            self.signal(name, &mut values, &mut active)?;
        }
        self.prev = Some(values.clone());
        Ok(values)
    }

    /// Executes a statement of an `always_ff` block (`comb` false) or an
    /// `always_comb` block (`comb` true), adding its assignments to `out`.
    /// In `always_comb`, reads of the signals in `out` see their value
    /// there.
    fn exec(
        &self,
        c: usize,
        s: &Stmt,
        comb: bool,
        values: &mut Values,
        active: &mut BTreeSet<Name>,
        out: &mut Values,
    ) -> Result<(), String> {
        let none = Values::new();
        let scope = |out: &Values| if comb { out.clone() } else { none.clone() };
        match &s.kind {
            StmtKind::Block(stmts) => stmts
                .iter()
                .try_for_each(|s| self.exec(c, s, comb, values, active, out)),
            StmtKind::If(cond, a, b) => {
                let v = self.own(c, cond, &scope(out), values, active)?;
                let branch = if v.is_zero() { b } else { a };
                self.exec(c, branch, comb, values, active, out)
            }
            StmtKind::Case {
                sel,
                items,
                default,
            } => {
                let scope = scope(out);
                // the selector and the labels are evaluated at one type
                let mut t = self.det(c, sel);
                for l in items.iter().flat_map(|i| &i.labels) {
                    let tl = self.det(c, l);
                    t = Ty {
                        width: t.width.max(tl.width),
                        signed: t.signed && tl.signed,
                    };
                }
                let v = self.value(c, sel, t, &scope, values, active)?;
                for item in items {
                    for l in &item.labels {
                        if self.value(c, l, t, &scope, values, active)? == v {
                            return self.exec(c, &item.body, comb, values, active, out);
                        }
                    }
                }
                match default {
                    Some(d) => self.exec(c, d, comb, values, active, out),
                    None => Ok(()),
                }
            }
            StmtKind::NonBlocking { lhs, rhs } | StmtKind::Blocking { lhs, rhs } => {
                let env = &self.ctxs[c].env;
                let width = typing::pieces(env, lhs).iter().map(|p| p.width()).sum();
                let v = self.assigned(c, rhs, width, &scope(out), values, active)?;
                // the variable indices, evaluated before anything is written
                let pieces = typing::pieces(env, lhs);
                let mut indices = Vec::new();
                for p in &pieces {
                    indices.push(match p.variable {
                        Some(var) => {
                            let i = self.own(c, var.index, &scope(out), values, active)?;
                            Some(BigInt::from(i.unsigned().clone()) - var.below)
                        }
                        None => None,
                    });
                }
                // the pieces take the bits of `v` from the top down
                let mut top = v.width();
                for (p, index) in pieces.iter().zip(indices) {
                    let bits = slice(&v, top - 1, top - p.width());
                    top -= p.width();
                    // the other bits of a register keep their value; those
                    // of an `always_comb` signal are assigned before they
                    // are read (typing), so any value does
                    let old = match out.get(&p.name) {
                        Some(old) => old.clone(),
                        None if comb => BitVec::zero(env[&p.name].ty.width),
                        None => values[&self.ctxs[c].names[&p.name]].clone(),
                    };
                    let new = match index {
                        None => write(&old, p.lo, &bits),
                        // bit by bit from the lowest one at `lo`, skipping
                        // those outside the signal
                        Some(lo) => {
                            let mut new = old;
                            for k in 0..p.width() {
                                let at = &lo + k;
                                if let Some(at) = at.to_usize().filter(|at| *at < new.width()) {
                                    new = write(&new, at, &slice(&bits, k, k));
                                }
                            }
                            new
                        }
                    };
                    out.insert(p.name.clone(), new);
                }
                Ok(())
            }
        }
    }

    /// The value of a signal (by name), computed on demand from its
    /// driver.
    fn signal(
        &self,
        name: &str,
        values: &mut Values,
        active: &mut BTreeSet<Name>,
    ) -> Result<BitVec, String> {
        if let Some(v) = values.get(name) {
            return Ok(v.clone());
        }
        if !active.insert(name.to_string()) {
            return Err(format!("combinational loop through `{name}`"));
        }
        let (c, driver) = self
            .driver
            .get(name)
            .copied()
            .ok_or_else(|| format!("`{name}` has no driver"))?;
        let ctx = &self.ctxs[c];
        match driver {
            Driver::Conn(e) => {
                let width = self.signals[name].width;
                let v = self.assigned(c, e, width, &Values::new(), values, active)?;
                values.insert(name.to_string(), v);
            }
            Driver::Item(Item::Assign { lhs, rhs, .. }) => {
                let width = typing::pieces(&ctx.env, lhs)
                    .iter()
                    .map(|p| p.width())
                    .sum();
                let v = self.assigned(c, rhs, width, &Values::new(), values, active)?;
                // an `assign` drives whole signals (typing)
                let mut top = v.width();
                for p in typing::pieces(&ctx.env, lhs) {
                    let bits = slice(&v, top - 1, top - p.width());
                    top -= p.width();
                    let y = &ctx.names[&p.name];
                    let old = values
                        .get(y)
                        .cloned()
                        .unwrap_or_else(|| BitVec::zero(ctx.env[&p.name].ty.width));
                    values.insert(y.clone(), write(&old, p.lo, &bits));
                }
            }
            Driver::Ff(b) => {
                let (c, item, clock, edge, reset) = &self.blocks[b];
                // an active asynchronous reset sets the registers with a
                // reset value, in any round
                let reset_active = match reset {
                    Some((r, edge)) => {
                        let level = !self.signal(r, values, active)?.is_zero();
                        level == (*edge == Edge::Pos)
                    }
                    None => false,
                };
                let mut out = Values::new();
                if reset_active {
                    for (q, e) in item.body().unwrap().reset_values() {
                        let width = self.ctxs[*c].env[q].ty.width;
                        let v = self.assigned(*c, e, width, &Values::new(), values, active)?;
                        out.insert(q.clone(), v);
                    }
                }
                let Some(prev) = self.prev.as_ref() else {
                    // the first round: no edge
                    for q in item.targets() {
                        let y = &self.ctxs[*c].names[q];
                        let v = out.remove(q).unwrap_or_else(|| self.init[y].clone());
                        values.insert(y.clone(), v);
                    }
                    active.remove(name);
                    return Ok(values[name].clone());
                };
                let before = prev[clock].is_zero();
                let after = self.signal(clock, values, active)?.is_zero();
                let fires = match edge {
                    Edge::Pos => before && !after,
                    Edge::Neg => !before && after,
                };
                if fires {
                    // on the values just before the edge; the reset
                    // values win
                    let mut old = prev.clone();
                    let body = item.body().unwrap();
                    let mut run = Values::new();
                    self.exec(*c, body, false, &mut old, &mut BTreeSet::new(), &mut run)?;
                    for (q, v) in run {
                        out.entry(q).or_insert(v);
                    }
                }
                for q in item.targets() {
                    let y = &self.ctxs[*c].names[q];
                    let v = out.remove(q).unwrap_or_else(|| prev[y].clone());
                    values.insert(y.clone(), v);
                }
            }
            Driver::Item(item) => {
                let targets = item.targets();
                for y in &targets {
                    active.insert(ctx.names[*y].clone());
                }
                let mut out = Values::new();
                self.exec(c, item.body().unwrap(), true, values, active, &mut out)?;
                for y in targets {
                    let v = out
                        .remove(y)
                        .ok_or_else(|| format!("`{y}` not assigned (a latch)"))?;
                    active.remove(&ctx.names[y]);
                    values.insert(ctx.names[y].clone(), v);
                }
            }
        }
        active.remove(name);
        Ok(values[name].clone())
    }

    /// The self-determined type of `e`: IEEE 1800 Table 11-21 for the
    /// width, §11.8.1 for the signedness.
    fn det(&self, c: usize, e: &Expr) -> Ty {
        let env = &self.ctxs[c].env;
        let both = |a: Ty, b: Ty| Ty {
            width: a.width.max(b.width),
            signed: a.signed && b.signed,
        };
        match &e.kind {
            ExprKind::Lit { ty, .. } | ExprKind::Const { ty, .. } | ExprKind::Resize(ty, _) => *ty,
            ExprKind::Var(x) => env[x].ty,
            ExprKind::Fill(_) => Ty::BIT,
            ExprKind::Unary(UnOp::BitNot | UnOp::Neg | UnOp::Plus, a) => self.det(c, a),
            ExprKind::Unary(..) => Ty::BIT,
            ExprKind::Binary(op, a, b) => match op {
                BinOp::Shl | BinOp::Shr | BinOp::AShl | BinOp::AShr => self.det(c, a),
                op if compare(*op) || matches!(op, BinOp::LogAnd | BinOp::LogOr) => Ty::BIT,
                _ => both(self.det(c, a), self.det(c, b)),
            },
            ExprKind::Cond(_, a, b) => both(self.det(c, a), self.det(c, b)),
            ExprKind::Concat(es) => Ty::unsigned(es.iter().map(|e| self.det(c, e).width).sum()),
            ExprKind::Repl(n, es) => {
                Ty::unsigned(n * es.iter().map(|e| self.det(c, e).width).sum::<usize>())
            }
            ExprKind::Select(_, sel) => Ty::unsigned(sel.width()),
            ExprKind::Cast(Cast::Size(n), a) => Ty {
                width: *n,
                signed: self.det(c, a).signed,
            },
            ExprKind::Cast(Cast::Signed, a) => Ty {
                signed: true,
                ..self.det(c, a)
            },
            ExprKind::Cast(Cast::Unsigned, a) => Ty {
                signed: false,
                ..self.det(c, a)
            },
        }
    }

    /// The value of `e` assigned to a target of `width` bits: `e` is
    /// evaluated at the larger of both widths with its own signedness
    /// (IEEE 1800 §10.7), then truncated.
    fn assigned(
        &self,
        c: usize,
        e: &Expr,
        width: usize,
        scope: &Values,
        values: &mut Values,
        active: &mut BTreeSet<Name>,
    ) -> Result<BitVec, String> {
        let own = self.det(c, e);
        let t = Ty {
            width: own.width.max(width),
            ..own
        };
        let v = self.value(c, e, t, scope, values, active)?;
        Ok(slice(&v, width - 1, 0))
    }

    /// The value of `e` at its own type.
    fn own(
        &self,
        c: usize,
        e: &Expr,
        scope: &Values,
        values: &mut Values,
        active: &mut BTreeSet<Name>,
    ) -> Result<BitVec, String> {
        let t = self.det(c, e);
        self.value(c, e, t, scope, values, active)
    }

    /// The value of `e` in a context of type `t` (at least as wide as `e`),
    /// where the signals in `scope` have their value there and all others
    /// their value in the round. This is the propagation of IEEE 1800
    /// §11.8.2: `t` is passed down to the context-determined operands, and
    /// every other expression is evaluated at its own type and then
    /// extended to `t`, by sign if `t` is signed.
    fn value(
        &self,
        c: usize,
        e: &Expr,
        t: Ty,
        scope: &Values,
        values: &mut Values,
        active: &mut BTreeSet<Name>,
    ) -> Result<BitVec, String> {
        let ctx = &self.ctxs[c];
        let sig = |x: &Name, values: &mut Values, active: &mut BTreeSet<Name>| match scope.get(x) {
            Some(v) => Ok(v.clone()),
            None => self.signal(&ctx.names[x], values, active),
        };
        let v = match &e.kind {
            ExprKind::Lit { value, .. } | ExprKind::Const { value, .. } => value.clone(),
            ExprKind::Var(x) => sig(x, values, active)?,
            ExprKind::Fill(b) => {
                return Ok(if *b {
                    BitVec::wrap(t.width, ones(t.width))
                } else {
                    BitVec::zero(t.width)
                });
            }
            ExprKind::Unary(op @ (UnOp::BitNot | UnOp::Neg | UnOp::Plus), a) => {
                return Ok(unary(*op, &self.value(c, a, t, scope, values, active)?));
            }
            ExprKind::Unary(op, a) => unary(*op, &self.own(c, a, scope, values, active)?),
            ExprKind::Binary(op, a, b) if compare(*op) => {
                let (ta, tb) = (self.det(c, a), self.det(c, b));
                let both = Ty {
                    width: ta.width.max(tb.width),
                    signed: ta.signed && tb.signed,
                };
                let x = self.value(c, a, both, scope, values, active)?;
                let y = self.value(c, b, both, scope, values, active)?;
                binary(*op, &x, &y, both.signed)?
            }
            ExprKind::Binary(op @ (BinOp::LogAnd | BinOp::LogOr), a, b) => {
                let x = self.own(c, a, scope, values, active)?;
                let y = self.own(c, b, scope, values, active)?;
                binary(*op, &x, &y, false)?
            }
            ExprKind::Binary(op, a, b) => {
                let x = self.value(c, a, t, scope, values, active)?;
                let shift = matches!(op, BinOp::Shl | BinOp::Shr | BinOp::AShl | BinOp::AShr);
                let y = if shift {
                    self.own(c, b, scope, values, active)?
                } else {
                    self.value(c, b, t, scope, values, active)?
                };
                return binary(*op, &x, &y, t.signed);
            }
            ExprKind::Cond(cond, a, b) => {
                let v = self.own(c, cond, scope, values, active)?;
                let arm = if v.is_zero() { b } else { a };
                return self.value(c, arm, t, scope, values, active);
            }
            ExprKind::Concat(es) => self.exprs(c, es, scope, values, active)?,
            ExprKind::Repl(n, es) => {
                let v = self.exprs(c, es, scope, values, active)?;
                let mut acc = v.clone();
                for _ in 1..*n {
                    acc = concat(&acc, &v);
                }
                acc
            }
            ExprKind::Select(x, sel) => {
                let v = sig(x, values, active)?;
                let base = match sel.index() {
                    None => sel.constant().flatten().expect("constant").1,
                    // typing guarantees the index is in range
                    Some(i) => {
                        let i = self.own(c, i, scope, values, active)?;
                        let i = i.unsigned().to_usize().expect("an index in range");
                        match sel {
                            Select::Down(_, w) => i + 1 - w,
                            _ => i,
                        }
                    }
                };
                slice(&v, base + sel.width() - 1, base)
            }
            // an assignment to an `n`-bit value
            ExprKind::Cast(Cast::Size(n), a) => self.assigned(c, a, *n, scope, values, active)?,
            ExprKind::Cast(_, a) => self.own(c, a, scope, values, active)?,
            ExprKind::Resize(to, a) => {
                let v = self.own(c, a, scope, values, active)?;
                extend(&slice(&v, to.width.min(v.width()) - 1, 0), *to)
            }
        };
        Ok(extend(&v, t))
    }

    /// The values of `es`, each at its own type, concatenated.
    fn exprs(
        &self,
        c: usize,
        es: &[Expr],
        scope: &Values,
        values: &mut Values,
        active: &mut BTreeSet<Name>,
    ) -> Result<BitVec, String> {
        let mut acc: Option<BitVec> = None;
        for e in es {
            let v = self.own(c, e, scope, values, active)?;
            acc = Some(match acc {
                None => v,
                Some(a) => concat(&a, &v),
            });
        }
        Ok(acc.expect("an item"))
    }
}

/// Whether `op` is a comparison.
fn compare(op: BinOp) -> bool {
    use BinOp::*;
    matches!(op, Eq | Ne | CaseEq | CaseNe | Lt | Le | Gt | Ge)
}

/// `v` extended to the width of `t` (at least its own): with copies of its
/// top bit if `t` is signed, else with zeros.
fn extend(v: &BitVec, t: Ty) -> BitVec {
    if t.signed {
        BitVec::from_signed(t.width, v.signed())
    } else {
        BitVec::wrap(t.width, v.unsigned().clone())
    }
}

/// Bits `hi` down to `lo` of `v`.
fn slice(v: &BitVec, hi: usize, lo: usize) -> BitVec {
    BitVec::wrap(hi + 1 - lo, v.unsigned() >> lo)
}

/// `{a, b}`
fn concat(a: &BitVec, b: &BitVec) -> BitVec {
    BitVec::wrap(
        a.width() + b.width(),
        (a.unsigned() << b.width()) | b.unsigned(),
    )
}

/// `v` with the bits from `lo` upwards replaced by `bits`.
fn write(v: &BitVec, lo: usize, bits: &BitVec) -> BitVec {
    let mask = ones(bits.width()) << lo;
    let kept = v.unsigned() & (ones(v.width()) ^ mask);
    BitVec::wrap(v.width(), kept | (bits.unsigned() << lo))
}

fn bit(b: bool) -> BitVec {
    BitVec::wrap(1, b as u8)
}

fn ones(width: usize) -> BigUint {
    (BigUint::one() << width) - BigUint::one()
}

fn unary(op: UnOp, a: &BitVec) -> BitVec {
    let w = a.width();
    let v = a.unsigned();
    let parity = v.count_ones() % 2 == 1;
    match op {
        UnOp::BitNot => BitVec::wrap(w, ones(w) ^ v),
        UnOp::Neg => BitVec::from_signed(w, -BigInt::from(v.clone())),
        UnOp::Plus => a.clone(),
        UnOp::LogNot => bit(v.is_zero()),
        UnOp::RedAnd => bit(*v == ones(w)),
        UnOp::RedOr => bit(!v.is_zero()),
        UnOp::RedXor => bit(parity),
        UnOp::RedNand => bit(*v != ones(w)),
        UnOp::RedNor => bit(v.is_zero()),
        UnOp::RedXnor => bit(!parity),
    }
}

/// `signed` is the signedness of the (equally typed) operands. Division by
/// zero is an error (typing rules it out).
fn binary(op: BinOp, a: &BitVec, b: &BitVec, signed: bool) -> Result<BitVec, String> {
    let w = a.width();
    let (x, y) = (a.unsigned(), b.unsigned());
    let cmp = if signed {
        a.signed().cmp(&b.signed())
    } else {
        x.cmp(y)
    };
    // shift amounts beyond any width saturate
    let amount = || y.to_usize().unwrap_or(usize::MAX).min(w);
    if matches!(op, BinOp::Div | BinOp::Mod) && y.is_zero() {
        return Err(format!("division by zero in `{op}`"));
    }
    Ok(match op {
        BinOp::BitAnd => BitVec::wrap(w, x & y),
        BinOp::BitOr => BitVec::wrap(w, x | y),
        BinOp::BitXor => BitVec::wrap(w, x ^ y),
        BinOp::BitXnor => BitVec::wrap(w, ones(w) ^ (x ^ y)),
        BinOp::Add => BitVec::wrap(w, x + y),
        BinOp::Sub => BitVec::from_signed(w, BigInt::from(x.clone()) - BigInt::from(y.clone())),
        BinOp::Mul => BitVec::wrap(w, x * y),
        BinOp::Eq | BinOp::CaseEq => bit(x == y),
        BinOp::Ne | BinOp::CaseNe => bit(x != y),
        BinOp::Lt => bit(cmp.is_lt()),
        BinOp::Le => bit(cmp.is_le()),
        BinOp::Gt => bit(cmp.is_gt()),
        BinOp::Ge => bit(cmp.is_ge()),
        BinOp::LogAnd => bit(!x.is_zero() && !y.is_zero()),
        BinOp::LogOr => bit(!x.is_zero() || !y.is_zero()),
        BinOp::Shl | BinOp::AShl => BitVec::wrap(w, x << amount()),
        BinOp::Shr => BitVec::wrap(w, x >> amount()),
        BinOp::AShr if signed => BitVec::from_signed(w, a.signed() >> amount()),
        BinOp::AShr => BitVec::wrap(w, x >> amount()),
        // `BigInt` division rounds toward zero, as SystemVerilog's does
        BinOp::Div if signed => BitVec::from_signed(w, a.signed() / b.signed()),
        BinOp::Div => BitVec::wrap(w, x / y),
        BinOp::Mod if signed => BitVec::from_signed(w, a.signed() % b.signed()),
        BinOp::Mod => BitVec::wrap(w, x % y),
    })
}
