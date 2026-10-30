/*!
# The reference cycle semantics on the AST

[`Interp::step`] runs one clock cycle: given the inputs, it computes the
value of every signal. It gives meaning to the source operators directly
on numbers (unsigned or two's complement as the operand types say),
independently of [`SvOp`] and of the rewrites in lowering, so that tests
can compare it with the translated module.

[`SvOp`]: crate::SvOp
*/

use crate::ast::{BinOp, Dir, Error, Expr, ExprKind, Item, Module, Name, Ty, UnOp};
use crate::typing::{self, Env};
use crate::value::BitVec;
use num_bigint::{BigInt, BigUint};
use num_traits::{One, ToPrimitive, Zero};
use std::collections::{BTreeMap, BTreeSet};

/// A value for each of some signals.
pub type Values = BTreeMap<Name, BitVec>;

pub struct Interp<'m> {
    env: Env,
    /// The expression driving each driven signal.
    driver: BTreeMap<&'m str, &'m Expr>,
}

impl<'m> Interp<'m> {
    pub fn new(m: &'m Module) -> Result<Self, Error> {
        let env = typing::check(m)?;
        let driver = m
            .items
            .iter()
            .map(|item| match item {
                Item::Assign { lhs, rhs, .. } => (lhs.as_str(), rhs),
            })
            .collect();
        Ok(Self { env, driver })
    }

    /// Runs one cycle, with `inputs` giving a value for every input.
    /// Returns the value of every signal in this cycle.
    pub fn step(&mut self, inputs: &Values) -> Result<Values, String> {
        let mut values = Values::new();
        for (name, sig) in &self.env {
            if sig.dir != Some(Dir::Input) {
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
            self.signal(name, &mut values, &mut active)?;
        }
        Ok(values)
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
        let e = self.driver[name];
        let (v, _) = self.expr(e, values, active)?;
        active.remove(name);
        values.insert(name.to_string(), v.clone());
        Ok(v)
    }

    fn expr(
        &self,
        e: &Expr,
        values: &mut Values,
        active: &mut BTreeSet<Name>,
    ) -> Result<(BitVec, Ty), String> {
        Ok(match &e.kind {
            ExprKind::Lit { ty, value } => (value.clone(), *ty),
            ExprKind::Var(x) => (self.signal(x, values, active)?, self.env[x].ty),
            ExprKind::Unary(op, a) => {
                let (a, ta) = self.expr(a, values, active)?;
                (unary(*op, &a), typing::unary(*op, ta))
            }
            ExprKind::Binary(op, a, b) => {
                let (a, ta) = self.expr(a, values, active)?;
                let (b, tb) = self.expr(b, values, active)?;
                let t = typing::binary(*op, ta, tb, e.span).map_err(|e| e.to_string())?;
                (binary(*op, &a, &b, ta.signed), t)
            }
            ExprKind::Cond(c, a, b) => {
                let (c, _) = self.expr(c, values, active)?;
                let (a, ta) = self.expr(a, values, active)?;
                let (b, tb) = self.expr(b, values, active)?;
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
