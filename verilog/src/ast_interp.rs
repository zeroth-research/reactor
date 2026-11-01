/*!
# The reference cycle semantics on the AST

[`Interp::step`] runs one clock cycle: given the inputs, it computes the
value of every signal in the cycle, from the inputs and the registers,
and then the register values of the next cycle by executing every
`always_ff` block on the values of this cycle. The assignments `q <= e`
of all blocks are collected and take effect together at the end of the
cycle (the rising clock edge), the last one to a register winning; a
register that no executed assignment targets keeps its value.

A signal driven by an `always_comb` block is computed by running the whole
block when one of its signals is first needed, with blocking assignments:
a read of a signal of the block sees its latest assignment (typing makes
sure there is one). It gives meaning to the source operators directly
on numbers (unsigned or two's complement as the operand types say; a
size cast extends as its operand's type says),
independently of [`SvOp`] and of the rewrites in lowering, so that tests
can compare it with the translated module.

A hierarchy runs as one design: every signal is known by its name (see
[`Inst::naming`]), so a port connected to a signal is that signal, and an
input connected to another expression is computed from it on demand.
Register initial values are given by path (`u.q`).

[`SvOp`]: crate::SvOp
*/

use crate::ast::{
    BinOp, Dir, Error, Expr, ExprKind, Item, Name, Select, Span, Stmt, StmtKind, Ty, UnOp,
};
use crate::elab::{self, Inst};
use crate::typing::{self, Env};
use crate::value::BitVec;
use num_bigint::{BigInt, BigUint};
use num_traits::{One, ToPrimitive, Zero};
use std::collections::{BTreeMap, BTreeSet};

/// A value for each of some signals.
pub type Values = BTreeMap<Name, BitVec>;

pub struct Interp<'m> {
    /// One context per instance, the top module's first.
    ctxs: Vec<Ctx>,
    /// The driver of each signal driven by an `assign`, an `always_comb`
    /// or an instance connection, with the instance it is in.
    driver: BTreeMap<Name, (usize, Driver<'m>)>,
    /// The bodies of the `always_ff` blocks, with their instances.
    blocks: Vec<(usize, &'m Stmt)>,
    /// The clock of the top module.
    clock: Option<Name>,
    /// Every signal but the clock, by name, with its type.
    signals: BTreeMap<Name, Ty>,
    /// The inputs of the top module but the clock.
    inputs: Vec<(Name, Ty)>,
    /// The registers: path, name and type.
    registers: Vec<(Name, Name, Ty)>,
    /// The value of every register in the current cycle, by name.
    state: Values,
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
            clock: inst.module.clock().cloned(),
            signals: BTreeMap::new(),
            inputs: Vec::new(),
            registers: Vec::new(),
            state: Values::new(),
        };
        it.add(inst, "", &naming)?;
        for p in &inst.module.ports {
            if p.dir == Dir::Input && Some(&p.decl.name) != it.clock.as_ref() {
                it.inputs.push((p.decl.name.clone(), p.decl.ty));
            }
        }
        if let Some(clock) = &it.clock {
            it.signals.remove(clock);
        }
        let err = |msg: String| Error::new(Span::default(), msg);
        for (path, name, ty) in &it.registers {
            let v = init
                .get(path)
                .ok_or_else(|| err(format!("no initial value for register `{path}`")))?;
            if v.width() != ty.width {
                return Err(err(format!("register `{path}` given {} bits", v.width())));
            }
            it.state.insert(name.clone(), v.clone());
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
                Item::AlwaysFf { body, .. } => {
                    self.blocks.push((c, body));
                    for q in item.targets() {
                        let path = format!("{prefix}{q}");
                        self.registers.push((path, names[q].clone(), env[q].ty));
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
                            && !matches!(e.kind, ExprKind::Var(_))
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

    /// The inputs of the top module but the clock, with their types.
    pub fn inputs(&self) -> &[(Name, Ty)] {
        &self.inputs
    }

    /// Runs one cycle, with `inputs` giving a value for every input but
    /// the clock. Returns the value of every signal but the clock in this
    /// cycle, by name.
    pub fn step(&mut self, inputs: &Values) -> Result<Values, String> {
        let mut values = self.state.clone();
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
        let mut updates = Values::new();
        for &(c, body) in &self.blocks {
            let mut out = Values::new();
            self.exec(c, body, false, &mut values, &mut active, &mut out)?;
            let names = &self.ctxs[c].names;
            updates.extend(out.into_iter().map(|(q, v)| (names[&q].clone(), v)));
        }
        self.state.extend(updates);
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
                let (v, _) = self.expr(c, cond, &scope(out), values, active)?;
                let branch = if v.is_zero() { b } else { a };
                self.exec(c, branch, comb, values, active, out)
            }
            StmtKind::Case {
                sel,
                items,
                default,
            } => {
                let scope = scope(out);
                let (v, _) = self.expr(c, sel, &scope, values, active)?;
                for item in items {
                    for l in &item.labels {
                        if self.expr(c, l, &scope, values, active)?.0 == v {
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
                let (v, _) = self.expr(c, rhs, &scope(out), values, active)?;
                let env = &self.ctxs[c].env;
                // the pieces take the bits of `v` from the top down
                let mut top = v.width();
                for p in typing::pieces(env, lhs) {
                    let bits = slice(&v, top - 1, top - p.width());
                    top -= p.width();
                    // the other bits of a register keep their value; those
                    // of an `always_comb` signal are assigned before they
                    // are read (typing), so any value does
                    let old = match out.get(&p.name) {
                        Some(old) => old.clone(),
                        None if comb => BitVec::zero(env[&p.name].ty.width),
                        None => self.state[&self.ctxs[c].names[&p.name]].clone(),
                    };
                    out.insert(p.name.clone(), write(&old, p.lo, &bits));
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
                let (v, _) = self.expr(c, e, &Values::new(), values, active)?;
                values.insert(name.to_string(), v);
            }
            Driver::Item(Item::Assign { lhs, rhs, .. }) => {
                let (v, _) = self.expr(c, rhs, &Values::new(), values, active)?;
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

    /// The value of `e`, where the signals in `scope` have their value
    /// there and all others their value in the cycle.
    fn expr(
        &self,
        c: usize,
        e: &Expr,
        scope: &Values,
        values: &mut Values,
        active: &mut BTreeSet<Name>,
    ) -> Result<(BitVec, Ty), String> {
        let ctx = &self.ctxs[c];
        Ok(match &e.kind {
            ExprKind::Lit { ty, value } | ExprKind::Const { ty, value, .. } => (value.clone(), *ty),
            ExprKind::Var(x) => {
                let v = match scope.get(x) {
                    Some(v) => v.clone(),
                    None => self.signal(&ctx.names[x], values, active)?,
                };
                (v, ctx.env[x].ty)
            }
            ExprKind::Unary(op, a) => {
                let (a, ta) = self.expr(c, a, scope, values, active)?;
                (unary(*op, &a), typing::unary(*op, ta))
            }
            ExprKind::Binary(op, a, b) => {
                let (a, ta) = self.expr(c, a, scope, values, active)?;
                let (b, tb) = self.expr(c, b, scope, values, active)?;
                let t = typing::binary(*op, ta, tb, e.span).map_err(|e| e.to_string())?;
                (binary(*op, &a, &b, ta.signed)?, t)
            }
            ExprKind::Cond(cond, a, b) => {
                let (v, _) = self.expr(c, cond, scope, values, active)?;
                let (a, ta) = self.expr(c, a, scope, values, active)?;
                let (b, tb) = self.expr(c, b, scope, values, active)?;
                let t = typing::cond(ta, tb, e.span).map_err(|e| e.to_string())?;
                (if v.is_zero() { b } else { a }, t)
            }
            ExprKind::Concat(es) => {
                let (v, ts) = self.exprs(c, es, scope, values, active)?;
                (v, typing::concat(&ts))
            }
            ExprKind::Repl(n, es) => {
                let (v, ts) = self.exprs(c, es, scope, values, active)?;
                let t = typing::concat(&ts);
                let mut acc = v.clone();
                for _ in 1..*n {
                    acc = concat(&acc, &v);
                }
                (acc, typing::repl(*n, t))
            }
            ExprKind::Select(x, sel) => {
                let v = match scope.get(x) {
                    Some(v) => v.clone(),
                    None => self.signal(&ctx.names[x], values, active)?,
                };
                let tx = ctx.env[x].ty;
                let index = match sel.index() {
                    Some(i) => Some(self.expr(c, i, scope, values, active)?),
                    None => None,
                };
                let t = typing::select(x, tx, sel, index.as_ref().map(|i| i.1), e.span)
                    .map_err(|e| e.to_string())?;
                let base = match &index {
                    None => sel.constant().flatten().expect("constant").1,
                    // typing guarantees the index is in range
                    Some((i, _)) => {
                        let i = i.unsigned().to_usize().expect("an index in range");
                        match sel {
                            Select::Down(_, w) => i + 1 - w,
                            _ => i,
                        }
                    }
                };
                (slice(&v, base + t.width - 1, base), t)
            }
            ExprKind::Cast(cast, a) => {
                let (v, ta) = self.expr(c, a, scope, values, active)?;
                let t = typing::cast(*cast, ta, typing::context_free(a), e.span)
                    .map_err(|e| e.to_string())?;
                let v = if ta.signed {
                    BitVec::from_signed(t.width, v.signed())
                } else {
                    BitVec::wrap(t.width, v.unsigned().clone())
                };
                (v, t)
            }
        })
    }

    /// The values of `es` concatenated, and their types.
    fn exprs(
        &self,
        c: usize,
        es: &[Expr],
        scope: &Values,
        values: &mut Values,
        active: &mut BTreeSet<Name>,
    ) -> Result<(BitVec, Vec<Ty>), String> {
        let mut acc: Option<BitVec> = None;
        let mut ts = Vec::new();
        for e in es {
            let (v, t) = self.expr(c, e, scope, values, active)?;
            acc = Some(match acc {
                None => v,
                Some(a) => concat(&a, &v),
            });
            ts.push(t);
        }
        Ok((acc.expect("an item"), ts))
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
