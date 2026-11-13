/*!
# The standard width and signedness rules

[`module`] rewrites every expression of a checked module into the
explicit-width fragment that lowering translates (see [`typing::expr`]):
where IEEE 1800 resizes a value implicitly, it inserts a resize node
[`ExprKind::Resize`], and it replaces `'0`/`'1` by literals of their
width. Afterwards the operands of every operator have equal types and every
assignment copies a value of the target's width, so lowering needs no rule
of its own. The rules are those of IEEE 1800-2023, §11.6 to §11.8:

- An expression has a *self-determined* type ([`Sizer::det`]): a signal
  or constant its declared one, an unsized decimal literal is an `int`
  (signed, 32 bits), an unsized based literal (`'hff`) unsigned 32 bits,
  `'0`/`'1` 1 bit. The arithmetic and bitwise operators and `?:` take the
  larger width of their operands (the arms for `?:`) and are signed only
  if both are; `~ - +` and the shifts the type of their (left) operand.
  Comparisons, `! && ||` and the reductions give 1 unsigned bit;
  concatenations, replications and selects are unsigned, of the width they
  pick; a size cast `N'(e)` has `N` bits and the signedness of `e`;
  `$signed`/`$unsigned` (and `signed'`/`unsigned'`) the width of `e`.
- The type of the whole expression is then passed down to its
  *context-determined* operands ([`Sizer::check`]): the operands of the
  arithmetic and bitwise operators, of `~ - +`, the left operand of a
  shift and the arms of `?:`. Where it reaches any other expression (a
  signal, a constant, a comparison, a concatenation, …), that value is
  extended to the type: with copies of its top bit if the type passed
  down is signed, with zeros if not, whatever the signedness of the value
  itself. So in `s + u`, with `s` signed and `u` unsigned, `s` is
  zero-extended. The other operands are *self-determined*: each is sized
  at its own type (the condition of `?:`, the shift amount, the operands
  of `! && ||` and of the reductions, the items of concatenations and
  replications, select indices and element indices of arrays, the
  argument of `$signed`/`$unsigned`). An element of an array has the
  type of the array's elements.
  The operands of a comparison are sized together, at the larger width,
  and are signed only if both are.
- An assignment (`assign`, `=`, `<=`, an input port connection) evaluates
  its right-hand side at the larger of its own width and the target's,
  with the right-hand side's signedness (the target's does not matter),
  and truncates the result to the target. A size cast `N'(e)` is such an
  assignment to an `N`-bit value. The selector and the labels of a `case`
  are sized together like the operands of a comparison.

These rules are the bidirectional reading of Desfrene et al. (CAV 2026,
Fig. 6 and 7), which is proved there to agree with the LRM's two-phase
description for widths, extended with signedness as IEEE 1800 §11.8.1 and
§11.8.2 give it (the paper and its formalization leave signedness out).
Nothing is narrowed: `q == 15` with a 4-bit `q` compares 32 bits.

Constant operator expressions (`N - 1`, `-1`) are folded into one
constant at the type of their context, after sizing, so their value is
the one of IEEE 1800 at that type rather than the exact integer: `u + -1`
with a 40-bit unsigned `u` adds 40 ones.

Sizing also checks what needs the sized types: selects stay inside their
signal for every value of a variable index ([`typing::select`]), and the
divisor of `/` and `%` is a nonzero constant.
*/

use crate::ast::{
    BinOp, Cast, Dir, Elem, Error, Expr, ExprKind, Index, Item, Lvalue, LvalueKind, Module, Name,
    Select, Span, Stmt, StmtKind, Ty, UnOp,
};
use crate::elab::{self, Inst};
use crate::theory::SvOp;
use crate::typing::{self, Env};
use crate::value::{self, BitVec};
use std::cmp::Ordering;
use std::collections::BTreeMap;

type Result<T> = std::result::Result<T, Error>;

/// The module with every expression sized (see the module documentation).
/// Expects `m` to pass [`typing::check`], which gives `env`.
pub fn module(m: &Module, env: &Env) -> Result<Module> {
    let s = Sizer { env };
    let mut out = m.clone();
    for item in &mut out.items {
        match item {
            Item::Assign { lhs, rhs, .. } => {
                let width = typing::pieces(env, lhs).iter().map(|p| p.width()).sum();
                *rhs = s.assign(width, rhs)?;
            }
            Item::AlwaysFf { body, .. } | Item::AlwaysComb { body, .. } => s.stmt(body)?,
            Item::Instance(i) => {
                for c in &mut i.conns {
                    let (Some(f), Some(e)) = (&c.formal, &mut c.actual) else {
                        continue;
                    };
                    if f.dir == Dir::Input {
                        *e = s.assign(f.ty.width, e)?;
                    }
                }
            }
        }
    }
    // an initial value is assigned to its register
    for init in &mut out.inits {
        let width = typing::lookup(env, &init.target, init.span)?.ty.width;
        init.value = s.assign(width, &init.value)?;
    }
    Ok(out)
}

/// The sizing rules, over the signals `env`.
pub struct Sizer<'e> {
    pub env: &'e Env,
}

/// How an operator treats its operands.
enum Kind {
    /// Arithmetic and bitwise: both operands context-determined.
    Arith,
    /// Shifts: the left operand context-determined, the amount
    /// self-determined.
    Shift,
    /// Comparisons: operands sized together, 1-bit result.
    Compare,
    /// `&&`, `||`: self-determined operands, 1-bit result.
    Logic,
}

fn kind(op: BinOp) -> Kind {
    use BinOp::*;
    match op {
        BitAnd | BitOr | BitXor | BitXnor | Add | Sub | Mul | Div | Mod => Kind::Arith,
        Shl | Shr | AShl | AShr => Kind::Shift,
        Eq | Ne | CaseEq | CaseNe | Lt | Le | Gt | Ge => Kind::Compare,
        LogAnd | LogOr => Kind::Logic,
    }
}

/// The type of two operands sized together: the larger width, signed only
/// if both are.
fn join(a: Ty, b: Ty) -> Ty {
    Ty {
        width: a.width.max(b.width),
        signed: a.signed && b.signed,
    }
}

impl Sizer<'_> {
    fn stmt(&self, s: &mut Stmt) -> Result<()> {
        match &mut s.kind {
            StmtKind::Block(stmts) => stmts.iter_mut().try_for_each(|s| self.stmt(s)),
            StmtKind::If(c, a, b) => {
                *c = self.synth(c)?;
                self.stmt(a)?;
                self.stmt(b)
            }
            StmtKind::Case {
                sel,
                items,
                default,
            } => {
                let mut t = self.det(sel)?;
                for l in items.iter().flat_map(|i| &i.labels) {
                    t = join(t, self.det(l)?);
                }
                *sel = self.check(sel, t)?;
                for item in items.iter_mut() {
                    for l in &mut item.labels {
                        *l = self.check(l, t)?;
                    }
                    self.stmt(&mut item.body)?;
                }
                match default {
                    Some(d) => self.stmt(d),
                    None => Ok(()),
                }
            }
            StmtKind::NonBlocking { lhs, rhs } | StmtKind::Blocking { lhs, rhs } => {
                let width = typing::pieces(self.env, lhs)
                    .iter()
                    .map(|p| p.width())
                    .sum();
                *rhs = self.assign(width, rhs)?;
                self.target(lhs)
            }
        }
    }

    /// Sizes the variable indices of a target, which are self-determined.
    fn target(&self, l: &mut Lvalue) -> Result<()> {
        let span = l.span;
        match &mut l.kind {
            LvalueKind::Var(_) => Ok(()),
            LvalueKind::Concat(ls) => ls.iter_mut().try_for_each(|l| self.target(l)),
            LvalueKind::Select(x, sel) => self.target_select(x, sel, span),
            LvalueKind::Elem(e) => {
                if let Some(i) = e.index() {
                    typing::target_element(e, self.det(i)?, span)?;
                }
                let name = e.name();
                if let Index::Expr(i) = &mut e.index {
                    **i = self.synth(i)?;
                }
                match &mut e.sel {
                    Some(sel) => self.target_select(&name, sel, span),
                    None => Ok(()),
                }
            }
        }
    }

    /// Sizes the variable index of the select `x[…]` in a target.
    fn target_select(&self, x: &str, sel: &mut Select, span: Span) -> Result<()> {
        let Some(i) = sel.index() else {
            return Ok(());
        };
        typing::target_index(x, sel, self.det(i)?, span)?;
        let sized = self.synth(i)?;
        if let Select::Bit(Index::Expr(i))
        | Select::Up(Index::Expr(i), _)
        | Select::Down(Index::Expr(i), _) = sel
        {
            **i = sized;
        }
        Ok(())
    }

    /// `rhs` as assigned to a target of `width` bits: evaluated at the
    /// larger of both widths, with its own signedness, then truncated.
    pub fn assign(&self, width: usize, rhs: &Expr) -> Result<Expr> {
        let own = self.det(rhs)?;
        let t = Ty {
            width: width.max(own.width),
            signed: own.signed,
        };
        let e = self.check(rhs, t)?;
        Ok(resize(e, t, Ty { width, ..own }))
    }

    /// The self-determined type of `e`.
    pub fn det(&self, e: &Expr) -> Result<Ty> {
        Ok(match &e.kind {
            ExprKind::Lit { ty, .. } | ExprKind::Const { ty, .. } | ExprKind::Resize(ty, _) => *ty,
            ExprKind::Var(x) => typing::lookup(self.env, x, e.span)?.ty,
            ExprKind::Fill(_) => Ty::BIT,
            ExprKind::Unary(op, a) => match op {
                UnOp::BitNot | UnOp::Neg | UnOp::Plus => self.det(a)?,
                _ => Ty::BIT,
            },
            ExprKind::Binary(op, a, b) => match kind(*op) {
                Kind::Arith => join(self.det(a)?, self.det(b)?),
                Kind::Shift => self.det(a)?,
                Kind::Compare | Kind::Logic => Ty::BIT,
            },
            ExprKind::Cond(_, a, b) => join(self.det(a)?, self.det(b)?),
            ExprKind::Concat(es) => {
                let ts = es.iter().map(|e| self.det(e)).collect::<Result<Vec<_>>>()?;
                typing::concat(&ts)
            }
            ExprKind::Repl(n, es) => {
                let ts = es.iter().map(|e| self.det(e)).collect::<Result<Vec<_>>>()?;
                typing::repl(*n, typing::concat(&ts))
            }
            ExprKind::Select(x, sel) => {
                let t = typing::lookup(self.env, x, e.span)?.ty;
                let index = match sel.index() {
                    Some(i) => Some(self.det(i)?),
                    None => None,
                };
                typing::select(x, t, sel, index, e.span).map_err(|err| match sel.index() {
                    Some(i) if index.is_some_and(|t| t.width >= 32) && has_const(i) => Error::new(
                        err.span,
                        format!(
                            "{}; an unsized literal or an integer parameter makes the \
                                     index 32 bits wide",
                            err.msg
                        ),
                    ),
                    _ => err,
                })?
            }
            ExprKind::Elem(el) => {
                let t = typing::elem_type(self.env, el);
                let index = match el.index() {
                    Some(i) => Some(self.det(i)?),
                    None => None,
                };
                typing::element(el, index, e.span)?;
                match &el.sel {
                    None => t,
                    Some(sel) => {
                        let index = match sel.index() {
                            Some(i) => Some(self.det(i)?),
                            None => None,
                        };
                        typing::select(&el.name(), t, sel, index, e.span)?
                    }
                }
            }
            ExprKind::Cast(Cast::Size(n), a) => Ty {
                width: *n,
                signed: self.det(a)?.signed,
            },
            ExprKind::Cast(Cast::Signed, a) => Ty {
                signed: true,
                ..self.det(a)?
            },
            ExprKind::Cast(Cast::Unsigned, a) => Ty {
                signed: false,
                ..self.det(a)?
            },
        })
    }

    /// `e` sized at its own type.
    pub fn synth(&self, e: &Expr) -> Result<Expr> {
        let t = self.det(e)?;
        self.check(e, t)
    }

    /// `e` sized at the type `t` of its context, which is at least as wide
    /// as its own: the type passed down to its context-determined operands,
    /// and the type the other expressions are extended to.
    pub fn check(&self, e: &Expr, t: Ty) -> Result<Expr> {
        let span = e.span;
        let node = |kind| Expr { kind, span };
        let sized = match &e.kind {
            ExprKind::Unary(op @ (UnOp::BitNot | UnOp::Neg | UnOp::Plus), a) => {
                node(ExprKind::Unary(*op, Box::new(self.check(a, t)?)))
            }
            ExprKind::Binary(op, a, b) if matches!(kind(*op), Kind::Arith | Kind::Shift) => {
                let a = self.check(a, t)?;
                let b = match kind(*op) {
                    Kind::Arith => self.check(b, t)?,
                    _ => self.synth(b)?,
                };
                if matches!(op, BinOp::Div | BinOp::Mod) {
                    divisor(&b)?;
                }
                node(ExprKind::Binary(*op, Box::new(a), Box::new(b)))
            }
            ExprKind::Cond(c, a, b) => node(ExprKind::Cond(
                Box::new(self.synth(c)?),
                Box::new(self.check(a, t)?),
                Box::new(self.check(b, t)?),
            )),
            ExprKind::Fill(bit) => {
                let value = if *bit {
                    BitVec::wrap(t.width, (num_bigint::BigUint::from(1u8) << t.width) - 1u8)
                } else {
                    BitVec::zero(t.width)
                };
                node(ExprKind::Const {
                    text: e.to_string(),
                    ty: t,
                    value,
                })
            }
            _ => {
                let own = self.det(e)?;
                return Ok(resize(self.atomic(e)?, own, t));
            }
        };
        fold(sized, t)
    }

    /// `e`, which is no context-determined operator, sized at its own type.
    fn atomic(&self, e: &Expr) -> Result<Expr> {
        let span = e.span;
        let node = |kind| Expr { kind, span };
        let boxed = |e: Result<Expr>| e.map(Box::new);
        let sized = match &e.kind {
            ExprKind::Lit { .. } | ExprKind::Const { .. } | ExprKind::Var(_) => {
                return Ok(e.clone());
            }
            ExprKind::Unary(op, a) => node(ExprKind::Unary(*op, boxed(self.synth(a))?)),
            ExprKind::Binary(op, a, b) => match kind(*op) {
                Kind::Compare => {
                    let t = join(self.det(a)?, self.det(b)?);
                    node(ExprKind::Binary(
                        *op,
                        boxed(self.check(a, t))?,
                        boxed(self.check(b, t))?,
                    ))
                }
                _ => node(ExprKind::Binary(
                    *op,
                    boxed(self.synth(a))?,
                    boxed(self.synth(b))?,
                )),
            },
            ExprKind::Concat(es) => node(ExprKind::Concat(self.synth_all(es)?)),
            ExprKind::Repl(n, es) => node(ExprKind::Repl(*n, self.synth_all(es)?)),
            ExprKind::Select(x, sel) => node(ExprKind::Select(x.clone(), self.select(sel)?)),
            ExprKind::Elem(el) => {
                let mut el = el.clone();
                el.index = self.index(&el.index)?;
                el.sel = el.sel.as_ref().map(|s| self.select(s)).transpose()?;
                node(ExprKind::Elem(el))
            }
            // assigning `a` to an `n`-bit value
            ExprKind::Cast(c @ Cast::Size(n), a) => {
                let own = self.det(a)?;
                let t = Ty {
                    width: own.width.max(*n),
                    ..own
                };
                node(ExprKind::Cast(*c, boxed(self.check(a, t))?))
            }
            ExprKind::Cast(c, a) => node(ExprKind::Cast(*c, boxed(self.synth(a))?)),
            ExprKind::Cond(..) | ExprKind::Fill(_) | ExprKind::Resize(..) => {
                return Err(Error::new(
                    span,
                    format!("internal error: `{e}` sized as an atomic expression"),
                ));
            }
        };
        let t = self.det(e)?;
        fold(sized, t)
    }

    fn synth_all(&self, es: &[Expr]) -> Result<Vec<Expr>> {
        es.iter().map(|e| self.synth(e)).collect()
    }

    /// An index, sized at its own type.
    fn index(&self, i: &Index) -> Result<Index> {
        Ok(match i {
            Index::Const(k) => Index::Const(*k),
            Index::Expr(e) => Index::Expr(Box::new(self.synth(e)?)),
        })
    }

    /// A select with its index sized at its own type.
    fn select(&self, sel: &Select) -> Result<Select> {
        Ok(match sel {
            Select::Bit(i) => Select::Bit(self.index(i)?),
            Select::Part { hi, lo } => Select::Part { hi: *hi, lo: *lo },
            Select::Up(i, w) => Select::Up(self.index(i)?, *w),
            Select::Down(i, w) => Select::Down(self.index(i)?, *w),
        })
    }
}

/// Whether `e` contains a constant (a parameter or an unsized literal).
fn has_const(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::Const { .. } => true,
        ExprKind::Unary(_, a) | ExprKind::Cast(_, a) | ExprKind::Resize(_, a) => has_const(a),
        ExprKind::Binary(_, a, b) => has_const(a) || has_const(b),
        ExprKind::Cond(c, a, b) => has_const(c) || has_const(a) || has_const(b),
        ExprKind::Concat(es) | ExprKind::Repl(_, es) => es.iter().any(has_const),
        _ => false,
    }
}

/// `e`, sized at type `from`, resized to `to`; a constant is resized
/// right away.
fn resize(e: Expr, from: Ty, to: Ty) -> Expr {
    if from == to {
        return e;
    }
    if let Some(v) = constant(&e) {
        return Expr {
            kind: ExprKind::Const {
                text: e.to_string(),
                ty: to,
                value: convert(v, to),
            },
            span: e.span,
        };
    }
    Expr {
        span: e.span,
        kind: ExprKind::Resize(to, Box::new(e)),
    }
}

/// `v` truncated or extended to the type `to`, extended as `to` says.
pub fn convert(v: &BitVec, to: Ty) -> BitVec {
    if to.signed && to.width > v.width() {
        BitVec::from_signed(to.width, v.signed())
    } else {
        BitVec::wrap(to.width, v.unsigned().clone())
    }
}

/// The value of a constant leaf.
fn constant(e: &Expr) -> Option<&BitVec> {
    match &e.kind {
        ExprKind::Lit { value, .. } | ExprKind::Const { value, .. } => Some(value),
        _ => None,
    }
}

/// The divisor of `/` and `%`, sized: a nonzero constant.
fn divisor(b: &Expr) -> Result<()> {
    match constant(b) {
        Some(v) if !v.is_zero() => Ok(()),
        _ => Err(Error::new(
            b.span,
            "the divisor must be a nonzero constant (2-state values have no result for a \
             division by zero)",
        )),
    }
}

/// `e`, sized at type `t`, as one constant if it is an operator on
/// constants. Concatenations, replications and selects are left alone.
fn fold(e: Expr, t: Ty) -> Result<Expr> {
    let value = match &e.kind {
        ExprKind::Unary(op, a) => match constant(a) {
            Some(a) => unary(*op, a, t)?,
            None => return Ok(e),
        },
        ExprKind::Binary(op, a, b) => match (constant(a), constant(b)) {
            (Some(va), Some(vb)) => binary(*op, va, operand_signed(a), vb, t)?,
            _ => return Ok(e),
        },
        ExprKind::Cond(c, a, b) => match (constant(c), constant(a), constant(b)) {
            (Some(c), Some(a), Some(b)) => {
                if c.is_zero() {
                    b.clone()
                } else {
                    a.clone()
                }
            }
            _ => return Ok(e),
        },
        ExprKind::Cast(_, a) => match constant(a) {
            Some(a) => BitVec::wrap(t.width, a.unsigned().clone()),
            None => return Ok(e),
        },
        _ => return Ok(e),
    };
    Ok(Expr {
        kind: ExprKind::Const {
            text: e.to_string(),
            ty: t,
            value,
        },
        span: e.span,
    })
}

fn operand_signed(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::Lit { ty, .. } | ExprKind::Const { ty, .. } => ty.signed,
        _ => false,
    }
}

fn eval(op: SvOp, args: &[&BitVec]) -> Result<BitVec> {
    let args: Vec<BitVec> = args.iter().map(|v| (*v).clone()).collect();
    value::eval(&op, &args).map_err(|m| Error::new(Span::default(), m))
}

/// A sized unary operator on a constant, the result of type `t`.
fn unary(op: UnOp, a: &BitVec, t: Ty) -> Result<BitVec> {
    let not = |v: BitVec| eval(SvOp::Not, &[&v]);
    match op {
        UnOp::Plus => Ok(a.clone()),
        UnOp::BitNot => eval(SvOp::Not, &[a]),
        UnOp::Neg => eval(SvOp::Neg, &[a]),
        UnOp::LogNot => not(eval(SvOp::RedOr, &[a])?),
        UnOp::RedAnd => eval(SvOp::RedAnd, &[a]),
        UnOp::RedOr => eval(SvOp::RedOr, &[a]),
        UnOp::RedXor => eval(SvOp::RedXor, &[a]),
        UnOp::RedNand => not(eval(SvOp::RedAnd, &[a])?),
        UnOp::RedNor => not(eval(SvOp::RedOr, &[a])?),
        UnOp::RedXnor => not(eval(SvOp::RedXor, &[a])?),
    }
    .inspect(|v| debug_assert_eq!(v.width(), t.width))
}

/// A sized binary operator on constants: `signed` is the signedness of the
/// left operand (both have the same type except for shift amounts).
fn binary(op: BinOp, a: &BitVec, signed: bool, b: &BitVec, t: Ty) -> Result<BitVec> {
    use BinOp::*;
    let not = |v: BitVec| eval(SvOp::Not, &[&v]);
    let truth = |v: &BitVec| eval(SvOp::RedOr, &[v]);
    let pick = |s, u| if signed { s } else { u };
    let v = match op {
        BitAnd => eval(SvOp::And, &[a, b])?,
        BitOr => eval(SvOp::Or, &[a, b])?,
        BitXor => eval(SvOp::Xor, &[a, b])?,
        BitXnor => not(eval(SvOp::Xor, &[a, b])?)?,
        Add => eval(SvOp::Add, &[a, b])?,
        Sub => eval(SvOp::Sub, &[a, b])?,
        Mul => eval(SvOp::Mul, &[a, b])?,
        Div => eval(pick(SvOp::SDiv, SvOp::UDiv), &[a, b])?,
        Mod => eval(pick(SvOp::SRem, SvOp::URem), &[a, b])?,
        Eq | CaseEq => eval(SvOp::Eq, &[a, b])?,
        Ne | CaseNe => eval(SvOp::Ne, &[a, b])?,
        Lt => eval(pick(SvOp::SLt, SvOp::ULt), &[a, b])?,
        Le => eval(pick(SvOp::SLe, SvOp::ULe), &[a, b])?,
        Gt => eval(pick(SvOp::SLt, SvOp::ULt), &[b, a])?,
        Ge => eval(pick(SvOp::SLe, SvOp::ULe), &[b, a])?,
        LogAnd => eval(SvOp::And, &[&truth(a)?, &truth(b)?])?,
        LogOr => eval(SvOp::Or, &[&truth(a)?, &truth(b)?])?,
        Shl | AShl => eval(SvOp::Shl, &[a, b])?,
        Shr => eval(SvOp::LShr, &[a, b])?,
        AShr => eval(pick(SvOp::AShr, SvOp::LShr), &[a, b])?,
    };
    debug_assert_eq!(v.width(), t.width);
    Ok(v)
}

/// The sized hierarchy as SystemVerilog source in the explicit-width
/// fragment, for reading and for checking the sizing on its own: every
/// implicit resize is written out as casts (`16'(unsigned'(s))` extends
/// `s` with zeros), constants as sized literals, and `'0`/`'1` at their
/// width. Parameters are substituted, so a module instantiated with
/// different values is printed once per variant (`child`, `child_2`, …),
/// instances first. Simulators read it the same as the original.
pub fn print(inst: &Inst) -> Result<String> {
    let mut p = Printer {
        modules: Vec::new(),
    };
    p.inst(inst, "")?;
    Ok(p.modules.into_iter().map(|(_, text)| text).collect())
}

struct Printer {
    /// The printed modules, with the name each is printed under.
    modules: Vec<(Name, String)>,
}

impl Printer {
    /// Prints the instance at path `prefix` and those below it, and gives
    /// the name its module is printed under.
    fn inst(&mut self, inst: &Inst, prefix: &str) -> Result<Name> {
        let m = &inst.module;
        let mut names = BTreeMap::new();
        for (u, child) in &inst.children {
            names.insert(u.clone(), self.inst(child, &format!("{prefix}{u}."))?);
        }
        let ctx = |e: Error| elab::context(e, prefix);
        let env = typing::check(m).map_err(ctx)?;
        let sized = module(m, &env).map_err(ctx)?;
        let body = body(&env, &sized, &names);
        // the same module with the same text is printed once
        let base = &m.name;
        for (name, text) in &self.modules {
            if text.replacen(&format!("module {name} "), "", 1)
                == body.replacen("module {} ", "", 1)
            {
                return Ok(name.clone());
            }
        }
        let taken = |n: &str| self.modules.iter().any(|(m, _)| m == n);
        let mut name = base.clone();
        let mut k = 1;
        while taken(&name) {
            k += 1;
            name = format!("{base}_{k}");
        }
        self.modules.push((
            name.clone(),
            body.replacen("module {} ", &format!("module {name} "), 1),
        ));
        Ok(name)
    }
}

/// A module as source, named `{}`, with `names` the printed module name of
/// each instance.
fn body(env: &Env, m: &Module, names: &BTreeMap<Name, Name>) -> String {
    let mut out = String::from("module {} (\n");
    let ports: Vec<String> = m
        .ports
        .iter()
        .map(|p| {
            let dir = match p.dir {
                Dir::Input => "input ",
                Dir::Output => "output",
            };
            format!("  {dir} {} {}", p.decl.ty, p.decl.name)
        })
        .collect();
    out += &ports.join(",\n");
    out += "\n);\n";
    for d in m.decls.iter().filter(|d| !d.name.contains('[')) {
        out += &format!("  {} {};\n", d.ty, d.name);
    }
    for a in &m.arrays {
        out += &format!("  {} {} [0:{}];\n", a.ty, a.name, a.size - 1);
        if let Some(r) = &a.rom {
            let task = if r.binary { "$readmemb" } else { "$readmemh" };
            out += &format!("  initial {task}(\"{}\", {});\n", r.file, a.name);
        }
    }
    for i in &m.inits {
        out += &format!("  initial {} = {};\n", i.target, expr(env, &i.value));
    }
    for item in &m.items {
        out += "  ";
        out += &match item {
            Item::Assign { lhs, rhs, .. } => format!("assign {lhs} = {};", expr(env, rhs)),
            Item::AlwaysFf {
                clock,
                edge,
                reset,
                body,
                ..
            } => {
                let reset = reset
                    .as_ref()
                    .map(|r| format!(" or {} {}", r.edge, r.name))
                    .unwrap_or_default();
                format!("always_ff @({edge} {clock}{reset}) {}", stmt(env, body))
            }
            Item::AlwaysComb { body, .. } => format!("always_comb {}", stmt(env, body)),
            Item::Instance(i) => {
                let conns: Vec<String> = i
                    .conns
                    .iter()
                    .map(|c| match &c.actual {
                        Some(e) => format!(".{}({})", c.port, expr(env, e)),
                        None => format!(".{}()", c.port),
                    })
                    .collect();
                format!("{} {} ({});", names[&i.name], i.name, conns.join(", "))
            }
        };
        out += "\n";
    }
    out += "endmodule\n\n";
    out
}

fn stmt(env: &Env, s: &Stmt) -> String {
    match &s.kind {
        StmtKind::Block(stmts) => {
            let inner: String = stmts.iter().map(|s| format!(" {}", stmt(env, s))).collect();
            format!("begin{inner} end")
        }
        StmtKind::If(c, a, b) => format!(
            "if ({}) {} else {}",
            expr(env, c),
            stmt(env, a),
            stmt(env, b)
        ),
        StmtKind::Case {
            sel,
            items,
            default,
        } => {
            let mut out = format!("case ({})", expr(env, sel));
            for item in items {
                let labels: Vec<String> = item.labels.iter().map(|l| expr(env, l)).collect();
                out += &format!(" {}: {}", labels.join(", "), stmt(env, &item.body));
            }
            if let Some(d) = default {
                out += &format!(" default: {}", stmt(env, d));
            }
            out + " endcase"
        }
        StmtKind::NonBlocking { lhs, rhs } => {
            format!("{} <= {};", target(env, lhs), expr(env, rhs))
        }
        StmtKind::Blocking { lhs, rhs } => format!("{} = {};", target(env, lhs), expr(env, rhs)),
    }
}

/// A sized assignment target as source.
fn target(env: &Env, l: &Lvalue) -> String {
    match &l.kind {
        LvalueKind::Var(y) => y.clone(),
        LvalueKind::Select(y, sel) => select(env, y, sel),
        LvalueKind::Elem(e) => element(env, e),
        LvalueKind::Concat(ls) => {
            let ls: Vec<String> = ls.iter().map(|l| target(env, l)).collect();
            format!("{{{}}}", ls.join(", "))
        }
    }
}

/// A sized element `m[i]` or `m[i][…]` as source.
fn element(env: &Env, e: &Elem) -> String {
    let index = match &e.index {
        Index::Const(k) => k.to_string(),
        Index::Expr(i) => expr(env, i),
    };
    let name = format!("{}[{index}]", e.array);
    match &e.sel {
        None => name,
        Some(sel) => select(env, &name, sel),
    }
}

/// A sized select `x[…]` as source.
fn select(env: &Env, x: &str, sel: &Select) -> String {
    let index = |i: &Index| match i {
        Index::Const(k) => k.to_string(),
        Index::Expr(e) => expr(env, e),
    };
    match sel {
        Select::Bit(i) => format!("{x}[{}]", index(i)),
        Select::Part { hi, lo } => format!("{x}[{hi}:{lo}]"),
        Select::Up(i, w) => format!("{x}[{} +: {w}]", index(i)),
        Select::Down(i, w) => format!("{x}[{} -: {w}]", index(i)),
    }
}

/// A sized expression as source, fully parenthesized.
fn expr(env: &Env, e: &Expr) -> String {
    let list = |es: &[Expr]| {
        es.iter()
            .map(|e| expr(env, e))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let sign = |signed: bool| if signed { "signed" } else { "unsigned" };
    match &e.kind {
        ExprKind::Lit { ty, value } | ExprKind::Const { ty, value, .. } => {
            let s = if ty.signed { "s" } else { "" };
            format!("{}'{s}d{}", ty.width, value.unsigned())
        }
        ExprKind::Var(x) => x.clone(),
        ExprKind::Unary(op, a) => format!("({op}{})", expr(env, a)),
        ExprKind::Binary(op, a, b) => format!("({} {op} {})", expr(env, a), expr(env, b)),
        ExprKind::Cond(c, a, b) => {
            format!("({} ? {} : {})", expr(env, c), expr(env, a), expr(env, b))
        }
        ExprKind::Concat(es) => format!("{{{}}}", list(es)),
        ExprKind::Repl(n, es) => format!("{{{n}{{{}}}}}", list(es)),
        ExprKind::Select(x, sel) => select(env, x, sel),
        ExprKind::Elem(el) => element(env, el),
        ExprKind::Cast(Cast::Size(n), a) => format!("{n}'({})", expr(env, a)),
        ExprKind::Cast(Cast::Signed, a) => format!("signed'({})", expr(env, a)),
        ExprKind::Cast(Cast::Unsigned, a) => format!("unsigned'({})", expr(env, a)),
        ExprKind::Fill(b) => format!("'{}", u8::from(*b)),
        ExprKind::Resize(to, a) => {
            let from = typing::expr(env, a).expect("a sized expression");
            let inner = expr(env, a);
            match to.width.cmp(&from.width) {
                // extended as `to` says: the sign cast picks it
                Ordering::Greater => format!("{}'({}'({inner}))", to.width, sign(to.signed)),
                Ordering::Less if to.signed == from.signed => format!("{}'({inner})", to.width),
                Ordering::Less => format!("{}'({}'({inner}))", sign(to.signed), to.width),
                Ordering::Equal => format!("{}'({inner})", sign(to.signed)),
            }
        }
    }
}
