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
*/

use crate::ast::{
    BinOp, Dir, Error, Expr, ExprKind, Item, Module, Name, Span, Stmt, StmtKind, Ty, UnOp,
};
use std::collections::BTreeMap;

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
/// `assign` or an `always_ff` block. Within one `always_ff` block a signal
/// can be assigned several times (the last assignment wins). This is the
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
                stmt(&env, body, &mut |lhs, span| {
                    drive(&env, &mut driver, lhs, span, i)
                })?;
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

/// Checks a statement of an `always_ff` block; `drive` checks the target
/// of an assignment and returns its type.
fn stmt(env: &Env, s: &Stmt, drive: &mut dyn FnMut(&Name, Span) -> Result<Ty>) -> Result<()> {
    match &s.kind {
        StmtKind::Block(stmts) => stmts.iter().try_for_each(|s| stmt(env, s, drive)),
        StmtKind::If(c, a, b) => {
            expr(env, c)?;
            stmt(env, a, drive)?;
            stmt(env, b, drive)
        }
        StmtKind::NonBlocking { lhs, rhs } => {
            let ty = drive(lhs, s.span)?;
            assign(ty, expr(env, rhs)?, s.span)
        }
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
