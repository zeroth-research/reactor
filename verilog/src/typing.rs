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
  reductions are self-determined, so any width is fine, and so are the
  shift amount and the items of `{…}` and `{n{…}}`;
- resizing is explicit only: selects `x[…]` (always unsigned, like
  concatenations), `{…}`, `{n{…}}` and casts `N'(e)`; `$signed`,
  `$unsigned`, `signed'` and `unsigned'` change the signedness only;
- the divisor of `/` and `%` is a nonzero literal, so no division by zero
  happens (2-state values have no result for it).

On this fragment the widths of IEEE 1800 (after propagation) equal the
self-determined widths, so the fragment has the standard's semantics. The
one place where this needs a restriction is a widening cast: `N'(e)` is
defined as assigning `e` to an `N`-bit variable, which evaluates `e` at `N`
bits (`16'(a + b)` keeps the carry of 8-bit `a + b`). So the operand of a
widening cast must have a width independent of its context ([`context_free`]),
and `16'(a) + 16'(b)` is written instead. A narrowing cast evaluates its
operand at the operand's own width and truncates.

Selects never reach outside the signal: constant indices are checked, and
a variable index is accepted only if every value it can take is in range
(`a[i]` with `i` of 3 bits needs `a` of at least 8 bits).

Assignment targets are signals, constant selects of signals, and
concatenations of targets ([`pieces`]). An `assign` drives whole signals.

Parameters have been substituted by the frontend: a parameter is a
constant of its type, an integer constant (`N - 1`) has type `int`, so
`cnt == N - 1` with an 8-bit `cnt` is rejected like any other width
mismatch (write `8'(N - 1)`, or declare a typed `localparam`).

An instance drives the signals connected to its outputs, which must be
whole signals of the port's width; the expression connected to an input
must have the port's width. A clock port of an instance is connected to
the name of a 1-bit signal, which then counts as a clock of the module.

An `always_comb` block must also be free of latches and of reads of stale
values ([`definite`]): every signal it drives is assigned on every path
through the block, and read only after it has been assigned on every path
to the read.
*/

use crate::ast::{
    BinOp, Cast, Dir, Error, Expr, ExprKind, Index, Item, Lvalue, LvalueKind, Module, Name, Select,
    Span, Stmt, StmtKind, Ty, UnOp,
};
use crate::elab::{self, Inst};
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

/// The signals of the module, or the first name declared twice (signals,
/// parameters and instances share one name space).
pub fn env(m: &Module) -> Result<Env> {
    // all declared names, in source order, with the signal if it is one
    let params = m.params.iter().map(|p| (&p.name, p.span, None));
    let insts = m.instances().map(|i| (&i.name, i.span, None));
    let ports = m.ports.iter().map(|p| (&p.decl, Some(p.dir)));
    let decls = m.decls.iter().map(|d| (d, None));
    let sigs = ports.chain(decls).map(|(d, dir)| {
        let sig = Sig {
            ty: d.ty,
            dir,
            span: d.span,
        };
        (&d.name, d.span, Some(sig))
    });
    let mut names: Vec<_> = params.chain(insts).chain(sigs).collect();
    names.sort_by_key(|(_, span, _)| *span);
    let mut seen: BTreeMap<&Name, Span> = BTreeMap::new();
    let mut env = Env::new();
    for (name, span, sig) in names {
        if let Some(prev) = seen.insert(name, span) {
            return Err(Error::new(
                span,
                format!("`{name}` already declared at {prev}"),
            ));
        }
        if let Some(sig) = sig {
            env.insert(name.clone(), sig);
        }
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
/// places; we apply them per whole signal, which is stricter: assigning
/// `y[3:0]` and `y[7:4]` from two `assign`s is rejected.
///
/// Clocks: an `always_ff` block runs at the rising (`posedge`) or falling
/// (`negedge`) edge of a 1-bit signal, and a clock port of an instance is
/// connected to the name of a 1-bit signal. A module may have several
/// clocks, and clocks may also be read as data. Where a clock comes from
/// (an input, a register, or a copy or bit of one) and which registers may
/// read each other are checked on the whole hierarchy, by [`domains`].
pub fn check(m: &Module) -> Result<Env> {
    let env = env(m)?;
    // the position of the first assignment to each signal, and its item
    let mut driver: BTreeMap<Name, (Span, usize)> = BTreeMap::new();
    // every use of a signal as a clock
    let mut clocks: Vec<(&Name, Span)> = Vec::new();
    for (i, item) in m.items.iter().enumerate() {
        match item {
            Item::Assign { lhs, rhs, span } => {
                let width = lvalue(lhs, &mut |y, span| drive(&env, &mut driver, y, span, i))?;
                let t = expr(&env, rhs)?;
                assign(width, t, *span)?;
                for y in item.targets() {
                    let n = env[y].ty.width;
                    let bits: BTreeSet<usize> = pieces(&env, lhs)
                        .into_iter()
                        .filter(|p| p.name == *y)
                        .flat_map(|p| p.lo..=p.hi)
                        .collect();
                    if bits.len() < n {
                        let missing: Vec<usize> = (0..n).filter(|b| !bits.contains(b)).collect();
                        return Err(Error::new(
                            *span,
                            format!(
                                "this `assign` leaves {} undriven: an `assign` drives whole \
                                 signals (drive `{y}` with one `assign` of a concatenation)",
                                bits_text(y, &missing, n)
                            ),
                        ));
                    }
                }
            }
            Item::AlwaysFf {
                clock: c,
                clock_span,
                body,
                ..
            } => {
                clocks.push((c, *clock_span));
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
                definite(&env, item)?;
            }
            Item::Instance(inst) => {
                for c in &inst.conns {
                    let f = c.formal.ok_or_else(|| {
                        Error::new(c.span, "internal error: instance not elaborated")
                    })?;
                    let port = format!("port `{}` of `{}`", c.port, inst.name);
                    match (&c.actual, f.dir) {
                        (Some(e), _) if f.clock => match &e.kind {
                            ExprKind::Var(x) => clocks.push((x, e.span)),
                            _ => {
                                return Err(Error::new(
                                    e.span,
                                    format!("the clock {port} must be connected to a clock input"),
                                ));
                            }
                        },
                        (Some(e), Dir::Input) => {
                            let t = expr(&env, e)?;
                            if t.width != f.ty.width {
                                return Err(Error::new(
                                    e.span,
                                    format!(
                                        "{port} has {} bits, the signal connected {} \
                                         (resize explicitly){}",
                                        f.ty.width,
                                        t.width,
                                        int_hint(t, f.ty.width)
                                    ),
                                ));
                            }
                        }
                        (Some(e), Dir::Output) => {
                            let ExprKind::Var(y) = &e.kind else {
                                return Err(Error::new(
                                    e.span,
                                    format!(
                                        "output {port} must be connected to a signal name \
                                         (or left unconnected with `.{}()`)",
                                        c.port
                                    ),
                                ));
                            };
                            let t = drive(&env, &mut driver, y, e.span, i)?;
                            if t.width != f.ty.width {
                                return Err(Error::new(
                                    e.span,
                                    format!("{port} has {} bits, `{y}` {}", f.ty.width, t.width),
                                ));
                            }
                        }
                        (None, _) => {}
                    }
                }
            }
        }
    }
    for (name, sig) in &env {
        if sig.dir != Some(Dir::Input) && !driver.contains_key(name) {
            return Err(Error::new(sig.span, format!("`{name}` is never driven")));
        }
    }
    for (c, span) in clocks {
        let sig = lookup(&env, c, span)?;
        if sig.ty.width != 1 {
            let msg = if driver.contains_key(c) {
                format!("the clock `{c}` must have 1 bit")
            } else {
                format!("the clock `{c}` must be a 1-bit input")
            };
            return Err(Error::new(span, msg));
        }
    }
    Ok(env)
}

/// What drives a signal, as [`domains`] sees it: an item of an instance,
/// or the expression connected to an input of an instance (in the
/// instantiating one).
#[derive(Clone, Copy)]
enum Source<'m> {
    Item(&'m Item),
    Conn(&'m Expr, Span),
}

/// Checks where the clocks of an elaborated hierarchy come from and that no
/// register reads a value from another clock domain. Returns the input
/// clocks the design is clocked by, by name. Expects every instance to
/// pass [`check`] and the hierarchy to be free of loops: a register
/// clocked, through the chain below, by itself is a loop of awaits (its
/// edge test awaits its clock), which lowering rejects.
///
/// A clock is either an input of the top module (an *input clock*), or a
/// register (a *derived clock*, e.g. a divider `slow <= !slow`), or a
/// copy or a constant bit select of one of these driven by an `assign`
/// (`assign c = cnt[3]`), also through instance ports. Its *root* is the
/// input clock at the end of this chain. Anything else is rejected: a
/// clock computed by an operator or in an `always_comb` (a gated clock)
/// could glitch within a time step, which rounds cannot express, and an
/// input clock must have 1 bit (a bit of a wider input is no clock).
///
/// The domain of a signal is the set of roots of the clocks of the
/// registers it depends on through combinational logic (`assign`,
/// `always_comb`, instance connections): the root of its own clock for a
/// register, none for an input. A register may read only signals whose
/// domain is empty or its own root: registers on either edge of one input
/// clock and of the clocks derived from it are *related*, and reads among
/// them see the value from before the edge (see the timing assumptions in
/// [`crate::lower`]). Different input clocks are unrelated, and a value
/// sampled from another domain needs a rule for an edge that coincides
/// with a change of the value, which crossings do not have yet. Signals
/// are compared by name (see [`Inst::naming`]), so a port connected to a
/// signal is that signal.
pub fn domains(inst: &Inst) -> Result<BTreeSet<Name>> {
    let naming: BTreeMap<Name, Name> = inst.naming().into_iter().collect();
    // the instances with their paths, each instance before its children
    let mut insts: Vec<(&Inst, String)> = Vec::new();
    let mut walk = vec![(inst, String::new())];
    while let Some((i, prefix)) = walk.pop() {
        for (u, child) in i.children.iter().rev() {
            walk.push((child, format!("{prefix}{u}.")));
        }
        insts.push((i, prefix));
    }
    let canon = |k: usize, x: &str| naming[&format!("{}{x}", insts[k].1)].clone();
    // the driver of every signal, by name, with its instance and its path
    let mut drivers: BTreeMap<Name, (usize, Name, Source)> = BTreeMap::new();
    for (k, (i, prefix)) in insts.iter().enumerate() {
        for item in &i.module.items {
            match item {
                Item::Instance(u) => {
                    for (c, _) in u.conns(Dir::Input) {
                        if let Some(e) = &c.actual
                            && !matches!(e.kind, ExprKind::Var(_))
                        {
                            let path = format!("{prefix}{}.{}", u.name, c.port);
                            let source = Source::Conn(e, c.span);
                            drivers.insert(naming[&path].clone(), (k, path, source));
                        }
                    }
                }
                _ => {
                    for y in item.targets() {
                        let path = format!("{prefix}{y}");
                        drivers.insert(canon(k, y), (k, path, Source::Item(item)));
                    }
                }
            }
        }
    }
    let top: BTreeMap<&Name, usize> = inst
        .module
        .ports
        .iter()
        .filter(|p| p.dir == Dir::Input)
        .map(|p| (&p.decl.name, p.decl.ty.width))
        .collect();

    // the root of the clock `c` of instance `k`, used at `span`
    let root = |k: usize, c: &Name, span: Span| -> Result<Name> {
        let err = |msg: String| elab::context(Error::new(span, msg), &insts[k].1);
        let mut x = canon(k, c);
        // the names on the chain so far
        let mut chain: Vec<Name> = Vec::new();
        loop {
            if chain.contains(&x) {
                return Err(err(format!("internal error: clock loop through `{x}`")));
            }
            let Some((d, path, source)) = drivers.get(&x) else {
                return match top.get(&x) {
                    Some(1) => Ok(x),
                    Some(_) => Err(err(format!(
                        "the clock `{c}` is a bit of the input `{x}`: input clocks must be \
                         1-bit inputs"
                    ))),
                    None => Err(err(format!("internal error: `{x}` has no driver"))),
                };
            };
            // the next signal on the chain, a name in instance `d`
            let next = match source {
                Source::Item(Item::AlwaysFf { clock, .. }) => Some(clock),
                Source::Item(Item::Assign { lhs, rhs, .. })
                    if matches!(lhs.kind, LvalueKind::Var(_)) =>
                {
                    copy(rhs)
                }
                Source::Conn(e, _) => copy(e),
                _ => None,
            };
            let Some(y) = next else {
                let at = match source {
                    Source::Item(item) => item.span(),
                    Source::Conn(_, span) => *span,
                };
                let msg = if chain.is_empty() && *d == k {
                    format!("the clock `{c}` is computed combinationally")
                } else {
                    format!("the clock `{c}` comes from `{path}`, computed combinationally at {at}")
                };
                return Err(err(format!("{msg}: gated clocks are not supported")));
            };
            chain.push(x);
            x = canon(*d, y);
        }
    };

    // the root of every clock, checked at each use (clock ports first,
    // so that a bad clock is reported where it is passed down)
    let mut roots: BTreeMap<Name, Name> = BTreeMap::new();
    for (k, (i, _)) in insts.iter().enumerate() {
        for item in &i.module.items {
            let uses: Vec<(&Name, Span)> = match item {
                Item::AlwaysFf {
                    clock, clock_span, ..
                } => vec![(clock, *clock_span)],
                Item::Instance(u) => u
                    .conns(Dir::Input)
                    .filter(|(_, f)| f.clock)
                    .filter_map(
                        |(c, _)| match c.actual.as_ref().map(|e| (&e.kind, e.span)) {
                            Some((ExprKind::Var(x), span)) => Some((x, span)),
                            _ => None,
                        },
                    )
                    .collect(),
                _ => Vec::new(),
            };
            for (c, span) in uses {
                let r = root(k, c, span)?;
                roots.insert(canon(k, c), r);
            }
        }
    }

    // the domain of every register, and what every combinational signal
    // reads, by name
    let mut dom: BTreeMap<Name, BTreeSet<Name>> = BTreeMap::new();
    let mut comb: Vec<(Name, Name)> = Vec::new();
    for (k, (i, _)) in insts.iter().enumerate() {
        for item in &i.module.items {
            match item {
                Item::AlwaysFf { clock, .. } => {
                    let r = &roots[&canon(k, clock)];
                    for q in item.targets() {
                        dom.insert(canon(k, q), BTreeSet::from([r.clone()]));
                    }
                }
                Item::Assign { .. } | Item::AlwaysComb { .. } => {
                    for y in item.targets() {
                        for (x, _) in item.reads() {
                            comb.push((canon(k, y), canon(k, x)));
                        }
                    }
                }
                Item::Instance(u) => {
                    for (c, _) in u.conns(Dir::Input) {
                        let Some(e) = &c.actual else { continue };
                        let mut reads = Vec::new();
                        e.reads(&mut reads);
                        for (x, _) in reads {
                            comb.push((canon(k, &format!("{}.{}", u.name, c.port)), canon(k, x)));
                        }
                    }
                }
            }
        }
    }
    loop {
        let mut changed = false;
        for (y, x) in &comb {
            let add = dom.get(x).cloned().unwrap_or_default();
            let d = dom.entry(y.clone()).or_default();
            let before = d.len();
            d.extend(add);
            changed |= d.len() != before;
        }
        if !changed {
            break;
        }
    }
    // how a clock is named in messages: with its root if derived
    let clocked = |c: &Name| match &roots[c] {
        r if r == c => format!("`{c}`"),
        r => format!("`{c}`, derived from `{r}`"),
    };
    for (k, (i, prefix)) in insts.iter().enumerate() {
        for item in &i.module.items {
            let Item::AlwaysFf { clock, .. } = item else {
                continue;
            };
            let c = canon(k, clock);
            let r = &roots[&c];
            for (x, span) in item.reads() {
                let Some(other) = dom
                    .get(&canon(k, x))
                    .and_then(|d| d.iter().find(|d| *d != r))
                else {
                    continue;
                };
                let derived = if roots.iter().any(|(c, r)| r == other && c != other) {
                    " or by clocks derived from it"
                } else {
                    ""
                };
                let q = item.targets()[0];
                let msg = format!(
                    "`{q}` (clocked by {}) reads `{x}`, which depends on registers clocked \
                     by `{other}`{derived}: crossings between unrelated clocks are not supported yet",
                    clocked(&c)
                );
                return Err(elab::context(Error::new(span, msg), prefix));
            }
        }
    }
    Ok(roots.into_values().collect())
}

/// The signal `e` copies (all of it or one bit): a name or a constant bit
/// select of a name.
fn copy(e: &Expr) -> Option<&Name> {
    match &e.kind {
        ExprKind::Var(x) | ExprKind::Select(x, Select::Bit(Index::Const(_))) => Some(x),
        _ => None,
    }
}

/// A hint for a width mismatch between a value of type `t` and `width`
/// bits: an integer constant is an `int`, which needs a cast.
fn int_hint(t: Ty, width: usize) -> String {
    if t == Ty::INT && width != 32 {
        format!("; an integer constant is an `int` (32 bits): cast it, e.g. `{width}'(N - 1)`")
    } else {
        String::new()
    }
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
            let width = lvalue(lhs, drive)?;
            assign(width, expr(env, rhs)?, s.span)
        }
    }
}

/// Checks an assignment target, records the driver of each of its signals
/// with `drive`, and returns its width. A bit may be assigned only once in
/// one target, and selects must be constant.
fn lvalue(l: &Lvalue, drive: &mut dyn FnMut(&Name, Span) -> Result<Ty>) -> Result<usize> {
    fn walk(
        l: &Lvalue,
        drive: &mut dyn FnMut(&Name, Span) -> Result<Ty>,
        seen: &mut BTreeSet<(Name, usize)>,
    ) -> Result<usize> {
        let (y, hi, lo) = match &l.kind {
            LvalueKind::Concat(ls) => {
                return ls.iter().map(|l| walk(l, drive, seen)).sum();
            }
            LvalueKind::Var(y) => (y, drive(y, l.span)?.width - 1, 0),
            LvalueKind::Select(y, sel) => {
                let t = drive(y, l.span)?;
                if let Some(i) = sel.index() {
                    return Err(Error::new(
                        i.span,
                        "variable index in an assignment target not supported \
                         (assign under a `case` instead)",
                    ));
                }
                let (hi, lo) = constant_select(y, t, sel, l.span)?;
                (y, hi, lo)
            }
        };
        for b in lo..=hi {
            if !seen.insert((y.clone(), b)) {
                return Err(Error::new(
                    l.span,
                    format!("`{y}[{b}]` assigned twice in one target"),
                ));
            }
        }
        Ok(hi + 1 - lo)
    }
    walk(l, drive, &mut BTreeSet::new())
}

/// A part of a checked assignment target: the bits `hi` down to `lo` of a
/// signal.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Piece {
    pub name: Name,
    pub hi: usize,
    pub lo: usize,
}

impl Piece {
    pub fn width(&self) -> usize {
        self.hi + 1 - self.lo
    }
}

/// The parts of a checked assignment target, the one receiving the high
/// bits of the value first.
pub fn pieces(env: &Env, l: &Lvalue) -> Vec<Piece> {
    match &l.kind {
        LvalueKind::Var(y) => vec![Piece {
            name: y.clone(),
            hi: env[y].ty.width - 1,
            lo: 0,
        }],
        LvalueKind::Select(y, sel) => {
            let (hi, lo) = sel.constant().flatten().expect("a checked constant select");
            vec![Piece {
                name: y.clone(),
                hi,
                lo,
            }]
        }
        LvalueKind::Concat(ls) => ls.iter().flat_map(|l| pieces(env, l)).collect(),
    }
}

/// The bits `bits` (descending or not) of the signal `y` of width `width`,
/// as SystemVerilog: `` `y` `` for all of them, else `` `y[7:4]`, `y[1]` ``.
fn bits_text(y: &str, bits: &[usize], width: usize) -> String {
    if bits.len() == width {
        return format!("`{y}`");
    }
    let mut bits = bits.to_vec();
    bits.sort_unstable_by(|a, b| b.cmp(a));
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    for b in bits {
        match ranges.last_mut() {
            Some((_, lo)) if *lo == b + 1 => *lo = b,
            _ => ranges.push((b, b)),
        }
    }
    let ranges: Vec<String> = ranges
        .into_iter()
        .map(|(hi, lo)| {
            if hi == lo {
                format!("`{y}[{hi}]`")
            } else {
                format!("`{y}[{hi}:{lo}]`")
            }
        })
        .collect();
    ranges.join(", ")
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
///
/// The analysis is per bit: `y[3:0] = a; y[7:4] = b;` assigns all of `y`.
/// A read with a variable index reads every bit of its signal.
pub fn definite(env: &Env, item: &Item) -> Result<()> {
    let targets = item.targets();
    let body = item.body().expect("a block");
    let mut d = Definite {
        env,
        targets: &targets,
        partial: BTreeMap::new(),
    };
    let mut done = Bits::new();
    d.stmt(body, &mut done)?;
    for y in &targets {
        let n = env[*y].ty.width;
        let missing: Vec<usize> = (0..n)
            .filter(|b| !done.contains(&((*y).clone(), *b)))
            .collect();
        if missing.is_empty() {
            continue;
        }
        let bits = bits_text(y, &missing, n);
        return Err(match d.partial.get(*y) {
            Some((span, why)) => Error::new(
                *span,
                format!(
                    "{bits} is assigned on only some paths through this {why}, \
                     so it would keep its old value on the others (a latch); \
                     assign it on every path or before"
                ),
            ),
            None => Error::new(
                item.span(),
                format!(
                    "{bits} is never assigned in this `always_comb`, \
                     so it would keep its old value (a latch)"
                ),
            ),
        });
    }
    Ok(())
}

/// Bits of signals.
type Bits = BTreeSet<(Name, usize)>;

struct Definite<'t> {
    env: &'t Env,
    targets: &'t [&'t Name],
    /// For each target, the first `if` or `case` after which some of its
    /// bits are assigned on some paths only, with a description.
    partial: BTreeMap<Name, (Span, &'static str)>,
}

impl Definite<'_> {
    /// Checks the reads in `e`, with `done` the bits of targets assigned on
    /// every path so far.
    fn reads(&self, e: &Expr, done: &Bits) -> Result<()> {
        let mut reads = Vec::new();
        bit_reads(self.env, e, &mut reads);
        for (x, hi, lo, span) in reads {
            if !self.targets.contains(&x) {
                continue;
            }
            let n = self.env[x].ty.width;
            let missing: Vec<usize> = (lo..=hi)
                .filter(|b| !done.contains(&(x.clone(), *b)))
                .collect();
            if !missing.is_empty() {
                return Err(Error::new(
                    span,
                    format!(
                        "{} read before it is assigned on every path in this `always_comb` \
                         (it would read its value from the previous evaluation)",
                        bits_text(x, &missing, n)
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Adds to `done` the bits of targets that `s` assigns on every path.
    fn stmt(&mut self, s: &Stmt, done: &mut Bits) -> Result<()> {
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
                for p in pieces(self.env, lhs) {
                    done.extend((p.lo..=p.hi).map(|b| (p.name.clone(), b)));
                }
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
        done: &mut Bits,
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
        let all: Bits = outs.iter().flatten().cloned().collect();
        let every: Bits = all
            .iter()
            .filter(|x| outs.iter().all(|o| o.contains(*x)))
            .cloned()
            .collect();
        for (x, _) in all.difference(&every) {
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
        ExprKind::Cast(c, a) => cast(*c, expr(env, a)?, context_free(a), e.span),
    }
}

/// Checks that the divisor of `/` or `%` is a nonzero literal or
/// constant.
fn divisor(b: &Expr) -> Result<()> {
    match &b.kind {
        ExprKind::Lit { value, .. } | ExprKind::Const { value, .. } if !value.is_zero() => Ok(()),
        _ => Err(Error::new(
            b.span,
            "the divisor must be a nonzero literal (2-state values have no result \
             for a division by zero)",
        )),
    }
}

/// The type of `{a, b, …}` with items of types `ts`: always unsigned.
pub fn concat(ts: &[Ty]) -> Ty {
    Ty::unsigned(ts.iter().map(|t| t.width).sum())
}

/// The type of `{n{e}}`, with `e` of type `t`: always unsigned.
pub fn repl(n: usize, t: Ty) -> Ty {
    Ty::unsigned(n * t.width)
}

/// The type of the select `x[…]` of a signal of type `t`, with `index` the
/// type of a variable index: always unsigned. The selected bits must be
/// inside `x` whatever the value of the index is; a variable index must be
/// unsigned.
pub fn select(x: &Name, t: Ty, sel: &Select, index: Option<Ty>, span: Span) -> Result<Ty> {
    let width = sel.width();
    if let Select::Up(_, 0) | Select::Down(_, 0) = sel {
        return Err(Error::new(span, format!("select `{x}{sel}` of width 0")));
    }
    let Some(i) = index else {
        constant_select(x, t, sel, span)?;
        return Ok(Ty::unsigned(width));
    };
    if i.signed {
        return Err(Error::new(
            span,
            format!("the index of `{x}{sel}` must be unsigned"),
        ));
    }
    // the highest bit selected for the largest index, and the lowest for
    // the smallest index (0)
    let max = u32::try_from(i.width)
        .ok()
        .and_then(|w| 1usize.checked_shl(w))
        .map(|m| m - 1);
    let (top, low_ok) = match sel {
        Select::Bit(_) => (max, true),
        Select::Up(_, w) => (max.and_then(|m| m.checked_add(w - 1)), true),
        Select::Down(_, w) => (max, *w == 1),
        Select::Part { .. } => unreachable!("constant"),
    };
    match top {
        Some(top) if top < t.width && low_ok => Ok(Ty::unsigned(width)),
        _ if !low_ok => Err(Error::new(
            span,
            format!("`{x}{sel}` reaches below bit 0 for small indices; use `+:`"),
        )),
        _ => Err(Error::new(
            span,
            format!(
                "`{x}{sel}` may select beyond bit {} of `{x}`: the index has {} bits, \
                 so every one of its values must be in range (narrow the index)",
                t.width - 1,
                i.width
            ),
        )),
    }
}

/// The bits `(hi, lo)` of a constant select `x[…]` of a signal of type `t`,
/// checked to be inside `x`.
fn constant_select(x: &Name, t: Ty, sel: &Select, span: Span) -> Result<(usize, usize)> {
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
pub fn cast(c: Cast, t: Ty, context_free: bool, span: Span) -> Result<Ty> {
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
/// casts. (An integer constant like `N * N` is evaluated exactly, so at any
/// width it fits in.) An
/// operator expression is evaluated at the width of its context.
pub fn context_free(e: &Expr) -> bool {
    matches!(
        e.kind,
        ExprKind::Lit { .. }
            | ExprKind::Const { .. }
            | ExprKind::Var(_)
            | ExprKind::Select(..)
            | ExprKind::Concat(_)
            | ExprKind::Repl(..)
            | ExprKind::Cast(..)
    )
}

/// The reads in `e` with the bits `(hi, lo)` each may read; a variable
/// index may read any bit.
fn bit_reads<'e>(env: &Env, e: &'e Expr, out: &mut Vec<(&'e Name, usize, usize, Span)>) {
    match &e.kind {
        ExprKind::Lit { .. } | ExprKind::Const { .. } => {}
        ExprKind::Var(x) => out.push((x, env[x].ty.width - 1, 0, e.span)),
        ExprKind::Select(x, sel) => match sel.constant() {
            Some(Some((hi, lo))) => out.push((x, hi, lo, e.span)),
            _ => {
                out.push((x, env[x].ty.width - 1, 0, e.span));
                if let Some(i) = sel.index() {
                    bit_reads(env, i, out);
                }
            }
        },
        ExprKind::Unary(_, a) | ExprKind::Cast(_, a) => bit_reads(env, a, out),
        ExprKind::Binary(_, a, b) => {
            bit_reads(env, a, out);
            bit_reads(env, b, out);
        }
        ExprKind::Cond(c, a, b) => {
            bit_reads(env, c, out);
            bit_reads(env, a, out);
            bit_reads(env, b, out);
        }
        ExprKind::Concat(es) | ExprKind::Repl(_, es) => {
            es.iter().for_each(|e| bit_reads(env, e, out));
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
            let hint = if a == Ty::INT {
                int_hint(a, b.width)
            } else {
                int_hint(b, a.width)
            };
            Err(Error::new(
                span,
                format!("operands of `{op}` differ: `{a}` and `{b}` (resize explicitly){hint}"),
            ))
        }
    };
    match op {
        BitAnd | BitOr | BitXor | BitXnor | Add | Sub | Mul | Div | Mod => same().map(|()| a),
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

/// Checks an assignment of a value of type `rhs` to a target of width
/// `lhs`: only the widths must agree, the bits are copied.
pub fn assign(lhs: usize, rhs: Ty, span: Span) -> Result<()> {
    if lhs == rhs.width {
        Ok(())
    } else {
        Err(Error::new(
            span,
            format!(
                "assigning {} bits to {} bits (resize explicitly){}",
                rhs.width,
                lhs,
                int_hint(rhs, lhs)
            ),
        ))
    }
}
