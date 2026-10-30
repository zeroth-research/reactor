/*!
# Well-formedness and the explicit-width typing

[`check`] resolves names, checks that every signal has exactly one driver
and types every expression.

The typing rules are one function per expression form ([`unary`],
[`binary`], [`cond`]), so that lowering can call the same rules while it
builds terms. They accept only expressions in which no implicit resizing
happens:

- operands of the arithmetic, bitwise and comparison operators, and the two
  arms of `?:`, have the same width and the same signedness;
- the right-hand side of an assignment has the width of the left-hand side;
- the condition of `?:` and the operands of `! && ||` and of the
  reductions are self-determined, so any width is fine.

On this fragment the widths of IEEE 1800 (after propagation) equal the
self-determined widths, so the fragment has the standard's semantics.

An `always_comb` block must also be free of latches and of reads of stale
values ([`definite`]): every signal it drives is assigned on every path
through the block, and read only after it has been assigned on every path
to the read.
*/

use crate::ast::{
    BinOp, Dir, Error, Expr, ExprKind, Item, Module, Name, Span, Stmt, StmtKind, Ty, UnOp,
};
use std::collections::{BTreeMap, BTreeSet};

type Result<T> = std::result::Result<T, Error>;

/// What a name refers to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Sig {
    pub ty: Ty,
    /// `None` for an internal signal.
    pub dir: Option<Dir>,
    pub span: Span,
}

pub type Env = BTreeMap<Name, Sig>;

/// The signals of the module, or the first one declared twice.
pub fn env(m: &Module) -> Result<Env> {
    let mut env = Env::new();
    let ports = m.ports.iter().map(|p| (&p.decl, Some(p.dir)));
    let decls = m.decls.iter().map(|d| (d, None));
    for (d, dir) in ports.chain(decls) {
        if let Some(prev) = env.get(&d.name) {
            return Err(Error::new(
                d.span,
                format!("`{}` already declared at {}", d.name, prev.span),
            ));
        }
        let sig = Sig {
            ty: d.ty,
            dir,
            span: d.span,
        };
        env.insert(d.name.clone(), sig);
    }
    Ok(env)
}

/// Checks the module and returns its signals.
///
/// Drivers: every signal other than an input has exactly one driver, an
/// `assign`, an `always_ff` or an `always_comb` block. Within one block a
/// signal can be assigned several times (the last assignment wins). An
/// `always_ff` block has only non-blocking assignments `<=`, an
/// `always_comb` block only blocking ones `=`, and every block drives
/// something. This is the
/// rule of IEEE 1800-2017 for whole variables: a variable assigned in an
/// `always_ff` "shall not be written to by any other process" (§9.2.2.4),
/// and procedural and continuous assignments to the same variable must not
/// be mixed (§6.5). The standard applies both rules per longest static
/// prefix, so different bits of a vector may be driven from different
/// places; we only assign whole signals, where the two rules coincide.
///
/// The clock: all `always_ff` blocks are clocked by the same 1-bit
/// input, and nothing else reads it. A round of the module is a whole
/// clock cycle, in which the clock is 0 and 1, so it has no value there.
pub fn check(m: &Module) -> Result<Env> {
    let env = env(m)?;
    // the position of the first assignment to each signal, and its item
    let mut driver: BTreeMap<Name, (Span, usize)> = BTreeMap::new();
    let mut clock: Option<(&Name, Span)> = None;
    for (i, item) in m.items.iter().enumerate() {
        match item {
            Item::Assign { lhs, rhs, span } => {
                let ty = drive(&env, &mut driver, lhs, *span, i)?;
                let t = expr(&env, rhs)?;
                assign(ty, t, *span)?;
            }
            Item::AlwaysFf {
                clock: c,
                clock_span,
                body,
                ..
            } => {
                let sig = lookup(&env, c, *clock_span)?;
                if sig.dir != Some(Dir::Input) || sig.ty.width != 1 {
                    return Err(Error::new(
                        *clock_span,
                        format!("the clock `{c}` must be a 1-bit input"),
                    ));
                }
                match clock {
                    Some((first, at)) if first != c => {
                        return Err(Error::new(
                            *clock_span,
                            format!(
                                "second clock `{c}`, only one clock (`{first}` at {at}) supported"
                            ),
                        ));
                    }
                    Some(_) => {}
                    None => clock = Some((c, *clock_span)),
                }
                stmt(&env, body, Proc::Ff, &mut |lhs, span| {
                    drive(&env, &mut driver, lhs, span, i)
                })?;
                drives_something(item)?;
            }
            Item::AlwaysComb { body, .. } => {
                stmt(&env, body, Proc::Comb, &mut |lhs, span| {
                    drive(&env, &mut driver, lhs, span, i)
                })?;
                drives_something(item)?;
                definite(item)?;
            }
        }
    }
    if let Some((clock, _)) = clock {
        for item in &m.items {
            if let Some((_, span)) = item.reads().into_iter().find(|(x, _)| *x == clock) {
                return Err(Error::new(
                    span,
                    format!("clock `{clock}` read (a cycle has no single clock value)"),
                ));
            }
        }
    }
    for (name, sig) in &env {
        if sig.dir != Some(Dir::Input) && !driver.contains_key(name) {
            return Err(Error::new(sig.span, format!("`{name}` is never driven")));
        }
    }
    Ok(env)
}

/// Records `item` as the driver of `lhs` and returns the type of `lhs`.
fn drive(
    env: &Env,
    driver: &mut BTreeMap<Name, (Span, usize)>,
    lhs: &Name,
    span: Span,
    item: usize,
) -> Result<Ty> {
    let sig = lookup(env, lhs, span)?;
    if sig.dir == Some(Dir::Input) {
        return Err(Error::new(span, format!("input `{lhs}` is assigned")));
    }
    match driver.get(lhs) {
        Some((prev, i)) if *i != item => Err(Error::new(
            span,
            format!("`{lhs}` already driven at {prev}"),
        )),
        Some(_) => Ok(sig.ty),
        None => {
            driver.insert(lhs.clone(), (span, item));
            Ok(sig.ty)
        }
    }
}

/// The kind of block a statement is in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Proc {
    Ff,
    Comb,
}

fn drives_something(item: &Item) -> Result<()> {
    if item.targets().is_empty() {
        let kw = match item {
            Item::AlwaysComb { .. } => "always_comb",
            _ => "always_ff",
        };
        return Err(Error::new(
            item.span(),
            format!("this `{kw}` drives nothing"),
        ));
    }
    Ok(())
}

/// Checks a statement of a block of kind `proc`; `drive` checks the target
/// of an assignment and returns its type.
fn stmt(
    env: &Env,
    s: &Stmt,
    proc: Proc,
    drive: &mut dyn FnMut(&Name, Span) -> Result<Ty>,
) -> Result<()> {
    match &s.kind {
        StmtKind::Block(stmts) => stmts.iter().try_for_each(|s| stmt(env, s, proc, drive)),
        StmtKind::If(c, a, b) => {
            expr(env, c)?;
            stmt(env, a, proc, drive)?;
            stmt(env, b, proc, drive)
        }
        StmtKind::Case {
            sel,
            items,
            default,
        } => {
            let t = expr(env, sel)?;
            for item in items {
                for l in &item.labels {
                    label(t, expr(env, l)?, l.span)?;
                }
                stmt(env, &item.body, proc, drive)?;
            }
            match default {
                Some(d) => stmt(env, d, proc, drive),
                None => Ok(()),
            }
        }
        StmtKind::NonBlocking { lhs, rhs } | StmtKind::Blocking { lhs, rhs } => {
            let blocking = matches!(s.kind, StmtKind::Blocking { .. });
            match (proc, blocking) {
                (Proc::Ff, true) => {
                    return Err(Error::new(
                        s.span,
                        "blocking assignment not allowed in `always_ff`, use `<=`",
                    ));
                }
                (Proc::Comb, false) => {
                    return Err(Error::new(
                        s.span,
                        "non-blocking assignment not allowed in `always_comb`, use `=`",
                    ));
                }
                _ => {}
            }
            let ty = drive(lhs, s.span)?;
            assign(ty, expr(env, rhs)?, s.span)
        }
    }
}

/// Checks that an `always_comb` block assigns each signal it drives on
/// every path before reading it and by the end of the block (no
/// latch).
///
/// A signal not assigned on some path would keep its value from the
/// previous evaluation of the block: hardware needs a latch for that, a
/// storage element outside the clocked cycle model. A read of a signal of
/// the block before it is assigned would read that same old value. The
/// analysis is syntactic: a `case` without `default` may assign nothing
/// even if its labels cover every value.
pub fn definite(item: &Item) -> Result<()> {
    let targets = item.targets();
    let body = item.body().expect("a block");
    let mut d = Definite {
        targets: &targets,
        partial: BTreeMap::new(),
    };
    let mut done = BTreeSet::new();
    d.stmt(body, &mut done)?;
    match targets.iter().find(|y| !done.contains(**y)) {
        None => Ok(()),
        Some(y) => {
            let (span, why) = &d.partial[*y];
            Err(Error::new(
                *span,
                format!(
                    "`{y}` is assigned on only some paths through this {why}, \
                     so it would keep its old value on the others (a latch); \
                     assign it on every path or before"
                ),
            ))
        }
    }
}

struct Definite<'t> {
    targets: &'t [&'t Name],
    /// For each target, the first `if` or `case` after which it is
    /// assigned on some paths only, with a description.
    partial: BTreeMap<Name, (Span, &'static str)>,
}

impl Definite<'_> {
    /// Checks the reads in `e`, with `done` the targets assigned on every
    /// path so far.
    fn reads(&self, e: &Expr, done: &BTreeSet<Name>) -> Result<()> {
        let mut reads = Vec::new();
        e.reads(&mut reads);
        match reads
            .into_iter()
            .find(|(x, _)| self.targets.contains(x) && !done.contains(*x))
        {
            None => Ok(()),
            Some((x, span)) => Err(Error::new(
                span,
                format!(
                    "`{x}` read before it is assigned on every path in this `always_comb` \
                     (it would read its value from the previous evaluation)"
                ),
            )),
        }
    }

    /// Adds to `done` the targets that `s` assigns on every path.
    fn stmt(&mut self, s: &Stmt, done: &mut BTreeSet<Name>) -> Result<()> {
        match &s.kind {
            StmtKind::Block(stmts) => stmts.iter().try_for_each(|s| self.stmt(s, done)),
            StmtKind::If(c, a, b) => {
                self.reads(c, done)?;
                let branches = [a.as_ref(), b.as_ref()];
                self.merge(&branches, false, done, s.span, "`if`")
            }
            StmtKind::Case {
                sel,
                items,
                default,
            } => {
                self.reads(sel, done)?;
                for l in items.iter().flat_map(|i| &i.labels) {
                    self.reads(l, done)?;
                }
                let mut branches: Vec<&Stmt> = items.iter().map(|i| &i.body).collect();
                let why = match default {
                    Some(d) => {
                        branches.push(d);
                        "`case`"
                    }
                    None => "`case` (without `default`, nothing is assigned when no label matches)",
                };
                self.merge(&branches, default.is_none(), done, s.span, why)
            }
            StmtKind::NonBlocking { lhs, rhs } | StmtKind::Blocking { lhs, rhs } => {
                self.reads(rhs, done)?;
                done.insert(lhs.clone());
                Ok(())
            }
        }
    }

    /// Runs each branch from `done` and keeps the targets assigned in all
    /// of them; `fallthrough` adds a path that runs no branch.
    fn merge(
        &mut self,
        branches: &[&Stmt],
        fallthrough: bool,
        done: &mut BTreeSet<Name>,
        span: Span,
        why: &'static str,
    ) -> Result<()> {
        let mut outs = Vec::new();
        for b in branches {
            let mut out = done.clone();
            self.stmt(b, &mut out)?;
            outs.push(out);
        }
        if fallthrough {
            outs.push(done.clone());
        }
        let all: BTreeSet<Name> = outs.iter().flatten().cloned().collect();
        let every: BTreeSet<Name> = all
            .iter()
            .filter(|x| outs.iter().all(|o| o.contains(*x)))
            .cloned()
            .collect();
        for x in all.difference(&every) {
            self.partial.entry(x.clone()).or_insert((span, why));
        }
        *done = every;
        Ok(())
    }
}

pub fn lookup<'e>(env: &'e Env, name: &str, span: Span) -> Result<&'e Sig> {
    env.get(name)
        .ok_or_else(|| Error::new(span, format!("`{name}` is not declared")))
}

/// The type of an expression.
pub fn expr(env: &Env, e: &Expr) -> Result<Ty> {
    match &e.kind {
        ExprKind::Lit { ty, .. } => Ok(*ty),
        ExprKind::Var(x) => Ok(lookup(env, x, e.span)?.ty),
        ExprKind::Unary(op, a) => Ok(unary(*op, expr(env, a)?)),
        ExprKind::Binary(op, a, b) => binary(*op, expr(env, a)?, expr(env, b)?, e.span),
        ExprKind::Cond(c, a, b) => {
            expr(env, c)?;
            cond(expr(env, a)?, expr(env, b)?, e.span)
        }
    }
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
                format!("operands of `{op}` differ: `{a}` and `{b}` (resize explicitly)"),
            ))
        }
    };
    match op {
        BitAnd | BitOr | BitXor | BitXnor | Add | Sub | Mul => same().map(|()| a),
        Eq | Ne | CaseEq | CaseNe | Lt | Le | Gt | Ge => same().map(|()| Ty::BIT),
        LogAnd | LogOr => Ok(Ty::BIT),
        // the shift amount is self-determined and unsigned
        Shl | Shr | AShl | AShr => Ok(a),
    }
}

/// The result type of `c ? a : b`; the condition can have any type.
pub fn cond(a: Ty, b: Ty, span: Span) -> Result<Ty> {
    if a == b {
        Ok(a)
    } else {
        Err(Error::new(
            span,
            format!("arms of `?:` differ: `{a}` and `{b}` (resize explicitly)"),
        ))
    }
}

/// Checks a label of type `label` of a `case` whose selector has type
/// `sel`: both must have the same type (no implicit resizing), and they are
/// compared with `==`.
pub fn label(sel: Ty, label: Ty, span: Span) -> Result<()> {
    if sel == label {
        Ok(())
    } else {
        Err(Error::new(
            span,
            format!("`case` label `{label}` differs from selector `{sel}` (resize explicitly)"),
        ))
    }
}

/// Checks an assignment of a value of type `rhs` to a signal of type `lhs`:
/// only the widths must agree, the bits are copied.
pub fn assign(lhs: Ty, rhs: Ty, span: Span) -> Result<()> {
    if lhs.width == rhs.width {
        Ok(())
    } else {
        Err(Error::new(
            span,
            format!(
                "assigning {} bits to {} bits (resize explicitly)",
                rhs.width, lhs.width
            ),
        ))
    }
}
