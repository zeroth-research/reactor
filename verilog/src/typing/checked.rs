//! A module that passed every check, sized: [`checked`] and [`Checked`].

use super::{Env, Result, check, explicit};
use crate::ast::{
    Cast, Error, Expr, ExprKind, Index, Item, Lvalue, LvalueKind, Module, Select, Stmt, StmtKind,
};
use crate::elab::Inst;
use crate::sizing::{self, Sizer};

/// The module of an instance after every check of this module, with its
/// signals and its sized form. Only [`checked`] makes one, so a function
/// that takes it may rely on what the checks guarantee: every name is
/// declared, every target has one driver, no block has a latch or a stale
/// read, and the sized module is in the explicit-width fragment.
#[derive(Debug)]
pub struct Checked<'m> {
    env: Env,
    source: &'m Module,
    sized: Module,
}

impl<'m> Checked<'m> {
    /// The signals of the module.
    pub fn env(&self) -> &Env {
        &self.env
    }

    /// The module as written (after elaboration).
    pub fn source(&self) -> &'m Module {
        self.source
    }

    /// The module with every implicit resize made explicit (see
    /// [`sizing`](crate::sizing)), in the explicit-width fragment.
    pub fn sized(&self) -> &Module {
        &self.sized
    }
}

/// Checks the module of the elaborated instance `inst` ([`check`], which
/// also checks its blocks for latches, stale reads and live x, each block
/// in source order), sizes it ([`sizing`](crate::sizing)) and checks that
/// the sized module is in the explicit-width fragment ([`explicit`]; a
/// failure there is an internal error). Errors are positions in the module, without the path
/// of the instance.
pub fn checked(inst: &Inst) -> Result<Checked<'_>> {
    let m = &inst.module;
    debug_assert!(
        m.instances()
            .all(|i| i.conns.iter().all(|c| c.formal.is_some())),
        "an instance of `{}` is not elaborated",
        m.name
    );
    let env = check(m)?;
    let sized = sizing::module(m, &env)?;
    let c = Checked {
        env,
        source: m,
        sized,
    };
    explicit(&c)
        .map_err(|e| Error::new(e.span, format!("internal error: sizing left `{}`", e.msg)))?;
    debug_assert!(
        sized_invariants(&c),
        "sizing broke an invariant of `{}`",
        m.name
    );
    Ok(c)
}

/// The invariants of a sized module, beyond [`explicit`]: no `'0`/`'1`
/// left, no cast that widens its operand, and the type of every
/// expression by the fragment's rules ([`expr`](super::expr)) is the
/// self-determined type by the sizing rules ([`Sizer::det`]).
fn sized_invariants(c: &Checked) -> bool {
    let s = Sizer::new(&c.env);
    let mut ok = true;
    each_expr(&c.sized, &mut |e| {
        let fragment = super::expr(&c.env, e).ok();
        let det = s.det(e).ok();
        let widening = match &e.kind {
            ExprKind::Cast(Cast::Size(n), a) => det_width(&s, a) < Some(*n),
            _ => false,
        };
        if matches!(e.kind, ExprKind::Fill(_)) || widening || fragment.is_none() || fragment != det
        {
            eprintln!(
                "sized `{e}` at {}: fragment {fragment:?}, sizing {det:?}",
                e.span
            );
            ok = false;
        }
    });
    ok
}

fn det_width(s: &Sizer, e: &Expr) -> Option<usize> {
    s.det(e).ok().map(|t| t.width)
}

/// Calls `f` on every expression of the module and on each of their
/// subexpressions, select and element indices included.
fn each_expr(m: &Module, f: &mut impl FnMut(&Expr)) {
    fn expr(e: &Expr, f: &mut impl FnMut(&Expr)) {
        f(e);
        match &e.kind {
            ExprKind::Lit { .. }
            | ExprKind::Const { .. }
            | ExprKind::Var(_)
            | ExprKind::Fill(_) => {}
            ExprKind::Unary(_, a) | ExprKind::Cast(_, a) | ExprKind::Resize(_, a) => expr(a, f),
            ExprKind::Binary(_, a, b) => {
                expr(a, f);
                expr(b, f);
            }
            ExprKind::Cond(c, a, b) => {
                expr(c, f);
                expr(a, f);
                expr(b, f);
            }
            ExprKind::Concat(es) | ExprKind::Repl(_, es) => es.iter().for_each(|e| expr(e, f)),
            ExprKind::Select(_, sel) => select(sel, f),
            ExprKind::Elem(el) => {
                index(&el.index, f);
                if let Some(sel) = &el.sel {
                    select(sel, f);
                }
            }
        }
    }
    fn index(i: &Index, f: &mut impl FnMut(&Expr)) {
        if let Index::Expr(e) = i {
            expr(e, f);
        }
    }
    fn select(sel: &Select, f: &mut impl FnMut(&Expr)) {
        if let Some(e) = sel.index() {
            expr(e, f);
        }
    }
    fn lvalue(l: &Lvalue, f: &mut impl FnMut(&Expr)) {
        match &l.kind {
            LvalueKind::Var(_) => {}
            LvalueKind::Select(_, sel) => select(sel, f),
            LvalueKind::Elem(el) => {
                index(&el.index, f);
                if let Some(sel) = &el.sel {
                    select(sel, f);
                }
            }
            LvalueKind::Concat(ls) => ls.iter().for_each(|l| lvalue(l, f)),
        }
    }
    fn stmt(s: &Stmt, f: &mut impl FnMut(&Expr)) {
        match &s.kind {
            StmtKind::Block(ss) => ss.iter().for_each(|s| stmt(s, f)),
            StmtKind::If(c, a, b) => {
                expr(c, f);
                stmt(a, f);
                stmt(b, f);
            }
            StmtKind::Case {
                sel,
                items,
                default,
                ..
            } => {
                expr(sel, f);
                for item in items {
                    item.labels.iter().for_each(|l| expr(l, f));
                    stmt(&item.body, f);
                }
                if let Some(d) = default {
                    stmt(d, f);
                }
            }
            StmtKind::NonBlocking { lhs, rhs } | StmtKind::Blocking { lhs, rhs } => {
                lvalue(lhs, f);
                expr(rhs, f);
            }
            StmtKind::Unknown { lhs, .. } => lvalue(lhs, f),
        }
    }
    for item in &m.items {
        match item {
            Item::Assign { lhs, rhs, .. } => {
                lvalue(lhs, f);
                expr(rhs, f);
            }
            Item::AlwaysFf { body, .. } | Item::AlwaysComb { body, .. } => stmt(body, f),
            Item::Instance(i) => i
                .conns
                .iter()
                .filter_map(|c| c.actual.as_ref())
                .for_each(|e| expr(e, f)),
        }
    }
    for i in &m.inits {
        expr(&i.value, f);
    }
}
