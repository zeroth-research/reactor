//! The explicit-width fragment: the typing rule of each expression form, selects, the pieces of
//! a target ([`explicit`], [`pieces`]).

use super::{Checked, Env, Result, lookup};
use crate::ast::{
    BinOp, Cast, Dir, Elem, Elems, Error, Expr, ExprKind, Item, Lvalue, LvalueKind, Module, Name,
    Select, Span, Stmt, StmtKind, Ty, UnOp,
};

/// Checks that a module, after [`sizing`](crate::sizing), is in the
/// explicit-width fragment: the operands of every operator, the arms of
/// every `?:`, the selector and labels of every `case` have equal types,
/// every assignment (also to an input port) copies a value of the target's
/// width, and casts only narrow or keep the width of an operator
/// expression ([`expr`]). [`checked`](super::checked) runs it on every
/// sized module.
pub fn explicit(c: &Checked) -> Result<()> {
    fragment(c.env(), c.sized())
}

/// [`explicit`] for the module as written: whether it needs no implicit
/// resize at all (as the sized source printed by
/// [`sizing::print`](crate::sizing::print), read back).
pub fn explicit_source(c: &Checked) -> Result<()> {
    fragment(c.env(), c.source())
}

/// [`explicit`] for the module `m` with the signals `env`.
fn fragment(env: &Env, m: &Module) -> Result<()> {
    fn stmt(env: &Env, s: &Stmt) -> Result<()> {
        match &s.kind {
            StmtKind::Block(stmts) => stmts.iter().try_for_each(|s| stmt(env, s)),
            StmtKind::If(c, a, b) => {
                expr(env, c)?;
                stmt(env, a)?;
                stmt(env, b)
            }
            StmtKind::Case {
                sel,
                items,
                default,
                ..
            } => {
                let t = expr(env, sel)?;
                for item in items {
                    for l in &item.labels {
                        label(t, expr(env, l)?, l.span)?;
                    }
                    stmt(env, &item.body)?;
                }
                default.as_deref().map_or(Ok(()), |d| stmt(env, d))
            }
            StmtKind::NonBlocking { lhs, rhs } | StmtKind::Blocking { lhs, rhs } => {
                for p in pieces_in(env, lhs) {
                    if let Some(v) = p.variable {
                        expr(env, v.index)?;
                    }
                    if let Some(e) = p.elem {
                        expr(env, e.index)?;
                    }
                }
                assign(width(env, lhs), expr(env, rhs)?, s.span)
            }
            StmtKind::Unknown { .. } => Ok(()),
        }
    }
    for item in &m.items {
        match item {
            Item::Assign { lhs, rhs, span } => assign(width(env, lhs), expr(env, rhs)?, *span)?,
            Item::AlwaysFf { body, .. } | Item::AlwaysComb { body, .. } => stmt(env, body)?,
            Item::Instance(i) => {
                for (c, f) in i.conns(Dir::Input) {
                    if let Some(e) = &c.actual {
                        assign(f.ty.width, expr(env, e)?, e.span)?;
                    }
                }
            }
        }
    }
    for i in &m.inits {
        assign(env[&i.target].ty.width, expr(env, &i.value)?, i.span)?;
    }
    Ok(())
}

/// The width of a checked assignment target.
pub(super) fn width(env: &Env, l: &Lvalue) -> usize {
    pieces_in(env, l).iter().map(Piece::width).sum()
}

/// A part of a checked assignment target: the bits `hi` down to `lo` of a
/// signal, or for a select with a variable index, `hi` down to `lo` counted
/// from the bit the index gives ([`Variable`]).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Piece<'l> {
    /// The signal, or the array for an element with a variable index.
    pub name: Name,
    pub hi: usize,
    pub lo: usize,
    pub variable: Option<Variable<'l>>,
    /// For an element of an array with a variable index: the bits are
    /// those of the element the index picks, if it is an address.
    pub elem: Option<ElemIndex<'l>>,
}

/// The variable index of an element of an array in an assignment target,
/// and the element signals by address.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ElemIndex<'l> {
    pub index: &'l Expr,
    pub elems: &'l [Name],
}

/// The variable index `index` of a piece: with `i` its value (unsigned),
/// the piece is the bits `i - below + hi` down to `i - below + lo` (where
/// `lo` is 0), those of them inside the signal.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Variable<'l> {
    pub index: &'l Expr,
    /// `w - 1` for `x[i -: w]`, else 0.
    pub below: usize,
}

impl Piece<'_> {
    pub fn width(&self) -> usize {
        self.hi + 1 - self.lo
    }
}

/// The parts of an assignment target of the checked module `c`, the one
/// receiving the high bits of the value first.
pub fn pieces<'l>(c: &Checked, l: &'l Lvalue) -> Vec<Piece<'l>> {
    pieces_in(c.env(), l)
}

/// [`pieces`] over the signals `env`, for a target whose names and
/// constant selects have been checked.
pub(crate) fn pieces_in<'l>(env: &Env, l: &'l Lvalue) -> Vec<Piece<'l>> {
    // the bits of a select of a signal of width `n`
    fn bits(sel: Option<&Select>, n: usize) -> (usize, usize, Option<Variable<'_>>) {
        let Some(sel) = sel else {
            return (n - 1, 0, None);
        };
        let variable = sel.index().map(|index| Variable {
            index,
            below: match sel {
                Select::Down(_, w) => w - 1,
                _ => 0,
            },
        });
        let (hi, lo) = match variable {
            Some(_) => (sel.width() - 1, 0),
            None => sel.constant().flatten().expect("a checked constant select"),
        };
        (hi, lo, variable)
    }
    match &l.kind {
        LvalueKind::Var(y) => vec![Piece {
            name: y.clone(),
            hi: env[y].ty.width - 1,
            lo: 0,
            variable: None,
            elem: None,
        }],
        LvalueKind::Select(y, sel) => {
            let (hi, lo, variable) = bits(Some(sel), env[y].ty.width);
            vec![Piece {
                name: y.clone(),
                hi,
                lo,
                variable,
                elem: None,
            }]
        }
        LvalueKind::Elem(e) => {
            let Elems::Signals(elems) = &e.elems else {
                unreachable!("a ROM is not written")
            };
            let (hi, lo, variable) = bits(e.sel.as_ref(), env[&elems[0]].ty.width);
            let index = e.index().expect("a variable element index");
            vec![Piece {
                name: e.array.clone(),
                hi,
                lo,
                variable,
                elem: Some(ElemIndex { index, elems }),
            }]
        }
        LvalueKind::Concat(ls) => ls.iter().flat_map(|l| pieces_in(env, l)).collect(),
    }
}

/// The type of an element of the array of `e`.
pub(crate) fn elem_type(env: &Env, e: &Elem) -> Ty {
    match &e.elems {
        Elems::Signals(ys) => env[&ys[0]].ty,
        Elems::Consts(ty, _) => *ty,
    }
}

/// Checks the type `index` of the variable index of the element `e` that
/// is read: unsigned, and every value of it is an address (`m[i]` with a
/// 3-bit `i` needs at least 8 elements). A constant index has been checked
/// by the frontend.
pub fn element(e: &Elem, index: Option<Ty>, span: Span) -> Result<()> {
    let Some(i) = index else {
        return Ok(());
    };
    let name = e.name();
    unsigned_index(&name, i, span)?;
    let fits = u32::try_from(i.width)
        .ok()
        .and_then(|w| 1usize.checked_shl(w))
        .is_some_and(|n| n <= e.size());
    if fits {
        Ok(())
    } else {
        Err(Error::new(
            span,
            format!(
                "`{name}` may address beyond the last element of `{}` ({}): the index has \
                 {} bits, so every one of its values must be an address (narrow the index)",
                e.array,
                e.size() - 1,
                i.width
            ),
        ))
    }
}

/// Checks the type `index` of the variable index of an element `e` in an
/// assignment target: it must be unsigned; any value is fine (an index that
/// is no address writes nothing).
pub(crate) fn target_element(e: &Elem, index: Ty, span: Span) -> Result<()> {
    unsigned_index(&e.name(), index, span)
}

/// Checks the type `index` of the variable index of a select in an
/// assignment target `x[…]`: it must be unsigned (a negative index would
/// write nothing; not supported).
pub(crate) fn target_index(x: &str, sel: &Select, index: Ty, span: Span) -> Result<()> {
    unsigned_index(&format!("{x}{sel}"), index, span)
}

/// Checks that the variable index of `what` (a select or an element), of
/// type `index`, is unsigned.
fn unsigned_index(what: &str, index: Ty, span: Span) -> Result<()> {
    if index.signed {
        return Err(Error::new(
            span,
            format!("the index of `{what}` must be unsigned"),
        ));
    }
    Ok(())
}

/// The type of an expression.
pub fn expr(env: &Env, e: &Expr) -> Result<Ty> {
    match &e.kind {
        ExprKind::Lit { ty, .. } | ExprKind::Const { ty, .. } => Ok(*ty),
        ExprKind::Var(x) => Ok(lookup(env, x, e.span)?.ty),
        ExprKind::Unary(op, a) => Ok(unary(*op, expr(env, a)?)),
        ExprKind::Binary(op, a, b) => {
            let t = binary(*op, expr(env, a)?, expr(env, b)?, e.span)?;
            if matches!(op, BinOp::Div | BinOp::Mod) {
                divisor(b)?;
            }
            Ok(t)
        }
        ExprKind::Cond(c, a, b) => {
            expr(env, c)?;
            cond(expr(env, a)?, expr(env, b)?, e.span)
        }
        ExprKind::Concat(es) => {
            let ts = es
                .iter()
                .map(|e| expr(env, e))
                .collect::<Result<Vec<_>>>()?;
            Ok(concat(&ts))
        }
        ExprKind::Repl(n, es) => {
            let ts = es
                .iter()
                .map(|e| expr(env, e))
                .collect::<Result<Vec<_>>>()?;
            Ok(repl(*n, concat(&ts)))
        }
        ExprKind::Select(x, sel) => {
            let t = lookup(env, x, e.span)?.ty;
            let index = sel.index().map(|i| expr(env, i)).transpose()?;
            select(x, t, sel, index, e.span)
        }
        ExprKind::Elem(el) => {
            let t = elem_type(env, el);
            let index = el.index().map(|i| expr(env, i)).transpose()?;
            element(el, index, e.span)?;
            match &el.sel {
                None => Ok(t),
                Some(sel) => {
                    let index = sel.index().map(|i| expr(env, i)).transpose()?;
                    select(&el.name(), t, sel, index, e.span)
                }
            }
        }
        ExprKind::Cast(c, a) => cast(*c, expr(env, a)?, context_free(a), e.span),
        ExprKind::Resize(t, a) => {
            expr(env, a)?;
            Ok(*t)
        }
        ExprKind::Fill(_) => Err(Error::new(
            e.span,
            format!("`{e}` has no width outside a context"),
        )),
    }
}

/// Checks that the divisor of `/` or `%` is a nonzero constant.
pub(crate) fn divisor(b: &Expr) -> Result<()> {
    match &b.kind {
        ExprKind::Lit { value, .. } | ExprKind::Const { value, .. } if !value.is_zero() => Ok(()),
        _ => Err(Error::new(
            b.span,
            "the divisor must be a nonzero constant (2-state values have no result \
             for a division by zero)",
        )),
    }
}

/// The type of `{a, b, …}` with items of types `ts`: always unsigned.
pub(crate) fn concat(ts: &[Ty]) -> Ty {
    Ty::unsigned(ts.iter().map(|t| t.width).sum())
}

/// The type of `{n{e}}`, with `e` of type `t`: always unsigned.
pub(crate) fn repl(n: usize, t: Ty) -> Ty {
    Ty::unsigned(n * t.width)
}

/// Rejects `x[b +: 0]` and `x[b -: 0]`.
pub(super) fn nonzero_width(x: &str, sel: &Select, span: Span) -> Result<()> {
    if let Select::Up(_, 0) | Select::Down(_, 0) = sel {
        return Err(Error::new(span, format!("select `{x}{sel}` of width 0")));
    }
    Ok(())
}

/// The type of the select `x[…]` of a signal of type `t`, with `index` the
/// type of a variable index: always unsigned. A constant select must be
/// inside `x`; a variable index must be unsigned, and may select bits
/// outside `x`, which read 0 (see [`may_leave`]).
pub fn select(x: &str, t: Ty, sel: &Select, index: Option<Ty>, span: Span) -> Result<Ty> {
    let width = sel.width();
    nonzero_width(x, sel, span)?;
    let Some(i) = index else {
        constant_select(x, t, sel, span)?;
        return Ok(Ty::unsigned(width));
    };
    unsigned_index(&format!("{x}{sel}"), i, span)?;
    if width > t.width {
        return Err(Error::new(
            span,
            format!(
                "`{x}{sel}` selects {width} bits, but `{x}` has only {}",
                t.width
            ),
        ));
    }
    Ok(Ty::unsigned(width))
}

/// Whether the read `x[sel]` of a signal of type `t`, with a variable
/// index of type `i`, selects bits outside `x` for some index value.
///
/// Such bits read 0: the standard's value for a 2-state signal, and what
/// Yosys builds. For a 4-state `logic` or `reg` the standard gives x, which
/// synthesis may resolve to any value; so where these bits are read, the
/// translation is exact only for synthesis tools that choose 0 (the
/// translator warns about each such read).
pub fn may_leave(t: Ty, sel: &Select, i: Ty) -> bool {
    // the largest index value; `None` if it does not fit a `usize`
    let max = u32::try_from(i.width)
        .ok()
        .and_then(|w| 1usize.checked_shl(w))
        .map(|m| m - 1);
    let top = match sel {
        Select::Bit(_) | Select::Down(..) => max,
        Select::Up(_, w) => max.and_then(|m| m.checked_add(w - 1)),
        Select::Part { .. } => return false,
    };
    let below = matches!(sel, Select::Down(_, w) if *w > 1);
    below || top.is_none_or(|top| top >= t.width)
}

/// The warning for a read `x[sel]` that [`may_leave`] the signal.
pub(crate) fn leave_warning(x: &str, t: Ty, sel: &Select, span: Span) -> Error {
    Error::new(
        span,
        format!(
            "`{x}{sel}` may select bits outside `{x}` (bits {}:0), which read 0 here; \
             for a 4-state signal they are x in the standard, which another synthesis \
             tool may resolve differently",
            t.width - 1
        ),
    )
}

/// The bits `(hi, lo)` of a constant select `x[…]` of a signal of type `t`,
/// checked to be inside `x`.
pub(super) fn constant_select(x: &str, t: Ty, sel: &Select, span: Span) -> Result<(usize, usize)> {
    nonzero_width(x, sel, span)?;
    match sel.constant().expect("a constant select") {
        Some((hi, lo)) if lo <= hi && hi < t.width => Ok((hi, lo)),
        Some((hi, lo)) if lo > hi => Err(Error::new(
            span,
            format!("`{x}{sel}`: write the higher bit first, `[{lo}:{hi}]`"),
        )),
        _ => Err(Error::new(
            span,
            format!("`{x}{sel}` out of range: `{x}` has bits {}:0", t.width - 1),
        )),
    }
}

/// The type of a cast of an operand of type `t`. `context_free` says
/// whether the operand's width is independent of its context; a widening
/// cast needs that (see the module documentation). A size cast keeps the
/// signedness of its operand.
pub(crate) fn cast(c: Cast, t: Ty, context_free: bool, span: Span) -> Result<Ty> {
    match c {
        Cast::Signed => Ok(Ty { signed: true, ..t }),
        Cast::Unsigned => Ok(Ty { signed: false, ..t }),
        Cast::Size(n) if n > t.width && !context_free => Err(Error::new(
            span,
            format!(
                "`{n}'(…)` would evaluate this operator expression at {n} bits rather \
                 than {}: cast its operands instead (e.g. `{n}'(a) + {n}'(b)`)",
                t.width
            ),
        )),
        Cast::Size(n) => Ok(Ty { width: n, ..t }),
    }
}

/// Whether the width at which `e` is evaluated does not depend on its
/// context: signals, selects, literals, constants, concatenations and
/// casts. (Sizing folds an operator on constants into one constant at its
/// final type.) An operator expression is evaluated at the width of its
/// context.
pub fn context_free(e: &Expr) -> bool {
    matches!(
        e.kind,
        ExprKind::Lit { .. }
            | ExprKind::Const { .. }
            | ExprKind::Var(_)
            | ExprKind::Select(..)
            | ExprKind::Elem(_)
            | ExprKind::Concat(_)
            | ExprKind::Repl(..)
            | ExprKind::Cast(..)
    )
}

/// The result type of a unary operator; every operand type is fine.
pub fn unary(op: UnOp, t: Ty) -> Ty {
    use UnOp::*;
    match op {
        BitNot | Neg | Plus => t,
        LogNot | RedAnd | RedOr | RedXor | RedNand | RedNor | RedXnor => Ty::BIT,
    }
}

/// The result type of a binary operator.
pub fn binary(op: BinOp, a: Ty, b: Ty, span: Span) -> Result<Ty> {
    use BinOp::*;
    let same = || {
        if a == b {
            Ok(())
        } else {
            Err(Error::new(
                span,
                format!("operands of `{op}` differ: `{a}` and `{b}`"),
            ))
        }
    };
    match op {
        BitAnd | BitOr | BitXor | BitXnor | Add | Sub | Mul | Div | Mod => same().map(|()| a),
        Eq | Ne | CaseEq | CaseNe | Lt | Le | Gt | Ge => same().map(|()| Ty::BIT),
        LogAnd | LogOr => Ok(Ty::BIT),
        // the shift amount is self-determined and unsigned
        Shl | Shr | AShl | AShr => Ok(a),
        Pow => Err(Error::new(
            span,
            "internal error: `**` is folded by sizing, it has no operator",
        )),
    }
}

/// The result type of `c ? a : b`; the condition can have any type.
pub fn cond(a: Ty, b: Ty, span: Span) -> Result<Ty> {
    if a == b {
        Ok(a)
    } else {
        Err(Error::new(
            span,
            format!("arms of `?:` differ: `{a}` and `{b}`"),
        ))
    }
}

/// Checks a label of type `label` of a `case` whose selector has type
/// `sel`: both must have the same type (no implicit resizing), and they are
/// compared with `==`.
pub(crate) fn label(sel: Ty, label: Ty, span: Span) -> Result<()> {
    if sel == label {
        Ok(())
    } else {
        Err(Error::new(
            span,
            format!("`case` label `{label}` differs from selector `{sel}`"),
        ))
    }
}

/// Checks an assignment of a value of type `rhs` to a target of width
/// `lhs`: only the widths must agree, the bits are copied.
pub(crate) fn assign(lhs: usize, rhs: Ty, span: Span) -> Result<()> {
    if lhs == rhs.width {
        Ok(())
    } else {
        Err(Error::new(
            span,
            format!("assigning {} bits to {} bits", rhs.width, lhs),
        ))
    }
}
