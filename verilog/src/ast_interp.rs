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
on numbers (unsigned or two's complement as the operand types say),
independently of [`SvOp`] and of the rewrites in lowering, so that tests
can compare it with the translated module.

[`SvOp`]: crate::SvOp
*/

use crate::ast::{
    BinOp, Dir, Error, Expr, ExprKind, Item, Module, Name, Span, Stmt, StmtKind, Ty, UnOp,
};
use crate::typing::{self, Env};
use crate::value::BitVec;
use num_bigint::{BigInt, BigUint};
use num_traits::{One, ToPrimitive, Zero};
use std::collections::{BTreeMap, BTreeSet};

/// A value for each of some signals.
pub type Values = BTreeMap<Name, BitVec>;

pub struct Interp<'m> {
    env: Env,
    clock: Option<&'m Name>,
    /// The driver of each signal driven by an `assign` or `always_comb`.
    driver: BTreeMap<&'m str, &'m Item>,
    /// The bodies of the `always_ff` blocks.
    blocks: Vec<&'m Stmt>,
    /// The value of every register in the current cycle.
    state: Values,
}

impl<'m> Interp<'m> {
    /// Starts with `init` giving a value for every register (and nothing
    /// else).
    pub fn new(m: &'m Module, init: &Values) -> Result<Self, Error> {
        let env = typing::check(m)?;
        let mut driver = BTreeMap::new();
        let mut blocks = Vec::new();
        for item in &m.items {
            match item {
                Item::AlwaysFf { body, .. } => blocks.push(body),
                Item::Assign { .. } | Item::AlwaysComb { .. } => {
                    for y in item.targets() {
                        driver.insert(y.as_str(), item);
                    }
                }
            }
        }
        let registers = m.registers();
        let err = |msg: String| Error::new(Span::default(), msg);
        for q in &registers {
            let v = init
                .get(*q)
                .ok_or_else(|| err(format!("no initial value for register `{q}`")))?;
            if v.width() != env[*q].ty.width {
                return Err(err(format!("register `{q}` given {} bits", v.width())));
            }
        }
        if let Some(x) = init.keys().find(|x| !registers.contains(x)) {
            return Err(err(format!("`{x}` is not a register")));
        }
        Ok(Self {
            env,
            clock: m.clock(),
            driver,
            blocks,
            state: init.clone(),
        })
    }

    /// Runs one cycle, with `inputs` giving a value for every input but
    /// the clock. Returns the value of every signal but the clock in this
    /// cycle.
    pub fn step(&mut self, inputs: &Values) -> Result<Values, String> {
        let mut values = self.state.clone();
        for (name, sig) in &self.env {
            if sig.dir != Some(Dir::Input) || Some(name) == self.clock {
                continue;
            }
            let v = inputs
                .get(name)
                .ok_or_else(|| format!("no value for input `{name}`"))?;
            if v.width() != sig.ty.width {
                return Err(format!("input `{name}` given {} bits", v.width()));
            }
            values.insert(name.clone(), v.clone());
        }
        let mut active = BTreeSet::new();
        for name in self.env.keys() {
            if Some(name) != self.clock {
                self.signal(name, &mut values, &mut active)?;
            }
        }
        let mut updates = Values::new();
        for body in &self.blocks {
            self.exec(body, false, &mut values, &mut active, &mut updates)?;
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
                .try_for_each(|s| self.exec(s, comb, values, active, out)),
            StmtKind::If(c, a, b) => {
                let (c, _) = self.expr(c, &scope(out), values, active)?;
                let branch = if c.is_zero() { b } else { a };
                self.exec(branch, comb, values, active, out)
            }
            StmtKind::Case {
                sel,
                items,
                default,
            } => {
                let scope = scope(out);
                let (v, _) = self.expr(sel, &scope, values, active)?;
                for item in items {
                    for l in &item.labels {
                        if self.expr(l, &scope, values, active)?.0 == v {
                            return self.exec(&item.body, comb, values, active, out);
                        }
                    }
                }
                match default {
                    Some(d) => self.exec(d, comb, values, active, out),
                    None => Ok(()),
                }
            }
            StmtKind::NonBlocking { lhs, rhs } | StmtKind::Blocking { lhs, rhs } => {
                let (v, _) = self.expr(rhs, &scope(out), values, active)?;
                out.insert(lhs.clone(), v);
                Ok(())
            }
        }
    }

    /// The value of a signal, computed on demand from its driver.
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
        match self.driver[name] {
            Item::Assign { rhs, .. } => {
                let (v, _) = self.expr(rhs, &Values::new(), values, active)?;
                values.insert(name.to_string(), v);
            }
            item => {
                let targets = item.targets();
                for y in &targets {
                    active.insert(y.to_string());
                }
                let mut out = Values::new();
                self.exec(item.body().unwrap(), true, values, active, &mut out)?;
                for y in targets {
                    let v = out
                        .remove(y)
                        .ok_or_else(|| format!("`{y}` not assigned (a latch)"))?;
                    active.remove(y);
                    values.insert(y.clone(), v);
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
        e: &Expr,
        scope: &Values,
        values: &mut Values,
        active: &mut BTreeSet<Name>,
    ) -> Result<(BitVec, Ty), String> {
        Ok(match &e.kind {
            ExprKind::Lit { ty, value } => (value.clone(), *ty),
            ExprKind::Var(x) => {
                let v = match scope.get(x) {
                    Some(v) => v.clone(),
                    None => self.signal(x, values, active)?,
                };
                (v, self.env[x].ty)
            }
            ExprKind::Unary(op, a) => {
                let (a, ta) = self.expr(a, scope, values, active)?;
                (unary(*op, &a), typing::unary(*op, ta))
            }
            ExprKind::Binary(op, a, b) => {
                let (a, ta) = self.expr(a, scope, values, active)?;
                let (b, tb) = self.expr(b, scope, values, active)?;
                let t = typing::binary(*op, ta, tb, e.span).map_err(|e| e.to_string())?;
                (binary(*op, &a, &b, ta.signed), t)
            }
            ExprKind::Cond(c, a, b) => {
                let (c, _) = self.expr(c, scope, values, active)?;
                let (a, ta) = self.expr(a, scope, values, active)?;
                let (b, tb) = self.expr(b, scope, values, active)?;
                let t = typing::cond(ta, tb, e.span).map_err(|e| e.to_string())?;
                (if c.is_zero() { b } else { a }, t)
            }
        })
    }
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

/// `signed` is the signedness of the (equally typed) operands.
fn binary(op: BinOp, a: &BitVec, b: &BitVec, signed: bool) -> BitVec {
    let w = a.width();
    let (x, y) = (a.unsigned(), b.unsigned());
    let cmp = if signed {
        a.signed().cmp(&b.signed())
    } else {
        x.cmp(y)
    };
    // shift amounts beyond any width saturate
    let amount = || y.to_usize().unwrap_or(usize::MAX).min(w);
    match op {
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
    }
}
