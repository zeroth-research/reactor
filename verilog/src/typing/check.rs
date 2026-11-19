//! Names, drivers, blocks and their event controls: [`check`].

use super::definite::{bits_text, definite, one_kind_of_assignment};
use super::fragment::{constant_select, nonzero_width, pieces};
use super::{Env, Result, env, lookup};
use crate::ast::{
    self, Dir, Error, Expr, ExprKind, Item, Lvalue, LvalueKind, Module, Name, Span, Stmt, StmtKind,
    Ty, Written,
};
use std::collections::{BTreeMap, BTreeSet};

/// Checks the module and returns its signals.
///
/// Drivers: every signal other than an input has exactly one driver, an
/// `assign`, an `always_ff` or an `always_comb` block. Within one block a
/// signal can be assigned several times (the last assignment wins). An
/// `always_comb` block has only blocking assignments `=`; an `always_ff`
/// block has non-blocking ones `<=`, and blocking ones `=` only where they
/// mean the same: each signal is assigned either only with `<=` or only
/// with `=`, and nothing a `=` assigns is read later in the block (checked
/// by [`definite`]; reads before it see the value before the edge, as with
/// `<=`; another block at the same clock edge may not read it either, see
/// [`domains`](super::domains)). Every block drives something. This is the
/// rule of IEEE 1800-2017 for whole variables: a variable assigned in an
/// `always_ff` "shall not be written to by any other process" (§9.2.2.4),
/// and procedural and continuous assignments to the same variable must not
/// be mixed (§6.5). The standard applies both rules per longest static
/// prefix, so different bits of a vector may be driven from different
/// places; we apply them per whole signal, which is stricter: assigning
/// `y[3:0]` and `y[7:4]` from two `assign`s is rejected. Likewise, all
/// elements of an array have one driver.
///
/// Blackboxes and pins: a blackbox (see
/// [`Module::blackbox`](crate::ast::Module::blackbox)) has nothing to check
/// but its ports. An output of a blackbox drives the signal it is
/// connected to like the output of any instance, but that signal is then a
/// free value, like an input, so it may not be an output of the module
/// (an `assign` from an internal signal makes one). A pin (`inout` port)
/// is no signal: any use other than a connection to a pin of a blackbox
/// (checked in [`elab`](crate::elab)) is rejected.
///
/// Clocks: an `always_ff` block runs at the rising (`posedge`) or falling
/// (`negedge`) edge of a 1-bit signal, and a clock port of an instance is
/// connected to the name of a 1-bit signal. A module may have several
/// clocks, and clocks may also be read as data (but not by the blocks they
/// run, see [`domains`](super::domains)). Where a clock comes from
/// (an input, a register, or a copy or bit of one) and which registers may
/// read each other are checked on the whole hierarchy, by [`domains`](super::domains).
///
/// Asynchronous resets: the frontend accepts `@(posedge clk or posedge
/// rst)` with a body that is one `if (rst) … else …` (`negedge rst_n` with
/// `if (!rst_n)`). The reset has 1 bit, and the `then` branch only assigns
/// constants to whole registers, each at most once; registers of the
/// block it does not assign are not reset (they keep their value while the
/// reset is active, as in simulation and synthesis). Where a reset comes
/// from is checked by [`domains`](super::domains).
pub fn check(m: &Module) -> Result<Env> {
    let env = env(m)?;
    if m.blackbox {
        // ports and parameters only (the frontend made sure)
        return Ok(env);
    }
    pin_uses(m)?;
    // the position of the first assignment to each signal, and its item
    let mut driver: BTreeMap<Name, (Span, usize)> = BTreeMap::new();
    // every use of a signal as a clock
    let mut clocks: Vec<(&Name, Span)> = Vec::new();
    for (i, item) in m.items.iter().enumerate() {
        match item {
            Item::Assign { lhs, rhs, span } => {
                lvalue(lhs, &mut |y, span| drive(&env, &mut driver, y, span, i))?;
                target_names(&env, lhs)?;
                names(&env, rhs)?;
                drives_whole(&env, item, lhs, *span)?;
            }
            Item::AlwaysFf {
                clock: c,
                clock_span,
                reset,
                body,
                ..
            } => {
                clocks.push((c, *clock_span));
                stmt(&env, body, Proc::Ff, &mut |lhs, span| {
                    drive(&env, &mut driver, lhs, span, i)
                })?;
                drives_something(item)?;
                one_kind_of_assignment(body, item.block(), &mut BTreeMap::new())?;
                if let Some(r) = reset {
                    if lookup(&env, &r.name, r.span)?.ty.width != 1 {
                        return Err(Error::new(
                            r.span,
                            format!("the reset `{}` must have 1 bit", r.name),
                        ));
                    }
                    if let Some(branch) = body.reset_branch() {
                        reset_branch(branch, &mut BTreeSet::new())?;
                    }
                }
                definite(&env, item)?;
            }
            Item::AlwaysComb { body, written, .. } => {
                stmt(&env, body, Proc::Comb(item.block()), &mut |lhs, span| {
                    drive(&env, &mut driver, lhs, span, i)
                })?;
                drives_something(item)?;
                match written {
                    Written::List(list) => event_list(&env, m, item, list)?,
                    Written::Star => star_reads(item)?,
                    Written::Keyword => {}
                }
                definite(&env, item)?;
            }
            Item::Instance(inst) => connections(&env, m, inst, i, &mut driver, &mut clocks)?,
        }
    }
    for init in &m.inits {
        initial_value(&env, m, &driver, init)?;
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

/// Every signal an `assign` drives is driven whole: the `assign` `item`
/// with the target `lhs` drives every bit of it.
fn drives_whole(env: &Env, item: &Item, lhs: &Lvalue, span: Span) -> Result<()> {
    for y in item.targets() {
        let n = env[y].ty.width;
        let bits: BTreeSet<usize> = pieces(env, lhs)
            .into_iter()
            .filter(|p| p.name == *y)
            .flat_map(|p| p.lo..=p.hi)
            .collect();
        if bits.len() < n {
            let missing: Vec<usize> = (0..n).filter(|b| !bits.contains(b)).collect();
            return Err(Error::new(
                span,
                format!(
                    "this `assign` leaves {} undriven: an `assign` drives whole \
                     signals (drive `{y}` with one `assign` of a concatenation)",
                    bits_text(y, &missing, n)
                ),
            ));
        }
    }
    Ok(())
}

/// Checks the connections of the instance `inst`, item `i` of `m`: records
/// the signals its outputs drive in `driver`, and in `clocks` the signals
/// used as clocks.
fn connections<'m>(
    env: &Env,
    m: &'m Module,
    inst: &'m ast::Instance,
    i: usize,
    driver: &mut BTreeMap<Name, (Span, usize)>,
    clocks: &mut Vec<(&'m Name, Span)>,
) -> Result<()> {
    // the signals the outputs of this instance drive so far
    let mut outputs: BTreeMap<&Name, Span> = BTreeMap::new();
    for c in &inst.conns {
        let f = c
            .formal
            .ok_or_else(|| Error::new(c.span, "internal error: instance not elaborated"))?;
        let port = format!("port `{}` of `{}`", c.port, inst.name);
        match (&c.actual, f.dir) {
            (Some(e), Dir::Input) if f.clock => match &e.kind {
                ExprKind::Var(x) => clocks.push((x, e.span)),
                _ => {
                    return Err(Error::new(
                        e.span,
                        format!("the clock {port} must be connected to a clock input"),
                    ));
                }
            },
            (Some(e), Dir::Input) => names(env, e)?,
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
                // `drive` lets one item assign a signal several
                // times (as a block may); two outputs of one
                // instance are two drivers
                if let Some(prev) = outputs.insert(y, e.span) {
                    return Err(Error::new(
                        e.span,
                        format!("`{y}` already driven at {prev}"),
                    ));
                }
                let t = drive(env, driver, y, e.span, i)?;
                // an output the child also uses as a clock is a
                // clock here too (a derived one)
                if f.clock {
                    clocks.push((y, e.span));
                }
                // a blackbox output acts like an input of the
                // module, so no output of it
                if f.blackbox && env[y].dir == Some(Dir::Output) {
                    return Err(Error::new(
                        e.span,
                        format!(
                            "`{y}` is an output of `{}`, but port `{}` of the \
                             blackbox `{}` gives a free value in every time \
                             step, like an input: connect an internal signal \
                             and `assign` it to `{y}`",
                            m.name, c.port, inst.name
                        ),
                    ));
                }
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
    Ok(())
}

/// Rejects every use of a pin (an `inout` port) other than a connection
/// to a pin of a blackbox (those are no longer among the connections, see
/// [`Instance::pins`](crate::ast::Instance::pins)): a pin is outside the
/// translated design, so nothing in it reads or drives a pin.
fn pin_uses(m: &Module) -> Result<()> {
    if m.pins.is_empty() {
        return Ok(());
    }
    let is_pin = |x: &str| m.pins.iter().any(|p| p.name == x);
    let err = |x: &str, span: Span| {
        Err(Error::new(
            span,
            format!(
                "`{x}` is an `inout` port: it may only be connected to an `inout` port of a \
                 blackbox"
            ),
        ))
    };
    for item in &m.items {
        let mut uses: Vec<(&Name, Span)> = item.reads();
        match item {
            Item::Assign { lhs, .. } => {
                let mut targets = Vec::new();
                lhs.targets(&mut targets);
                uses.extend(targets.into_iter().map(|y| (y, lhs.span)));
            }
            Item::AlwaysFf {
                clock,
                clock_span,
                reset,
                body,
                ..
            } => {
                uses.push((clock, *clock_span));
                uses.extend(reset.iter().map(|r| (&r.name, r.span)));
                let mut lvalues = Vec::new();
                body.lvalues(&mut lvalues);
                for lv in lvalues {
                    let mut targets = Vec::new();
                    lv.targets(&mut targets);
                    uses.extend(targets.into_iter().map(|y| (y, lv.span)));
                }
            }
            Item::AlwaysComb { body, .. } => {
                let mut lvalues = Vec::new();
                body.lvalues(&mut lvalues);
                for lv in lvalues {
                    let mut targets = Vec::new();
                    lv.targets(&mut targets);
                    uses.extend(targets.into_iter().map(|y| (y, lv.span)));
                }
            }
            Item::Instance(i) => {
                for c in &i.conns {
                    if let Some(e) = &c.actual {
                        e.reads(&mut uses);
                    }
                }
            }
        }
        uses.sort_by_key(|(_, span)| *span);
        if let Some((x, span)) = uses.into_iter().find(|(x, _)| is_pin(x)) {
            return err(x, span);
        }
    }
    Ok(())
}

/// Checks the initial value of a signal: a constant (an expression that
/// reads no signal), given to a register. A signal driven by an `assign`,
/// an `always_comb` or an instance is computed in every time step, so an
/// initial value would have no effect (the frontend has made initialized
/// signals that nothing drives constants already, see
/// [`frontend::constants`](crate::frontend::constants)).
fn initial_value(
    env: &Env,
    m: &Module,
    driver: &BTreeMap<Name, (Span, usize)>,
    init: &ast::Init,
) -> Result<()> {
    let y = &init.target;
    let sig = lookup(env, y, init.span)?;
    let mut reads = Vec::new();
    init.value.reads(&mut reads);
    if let Some((x, span)) = reads.first() {
        return Err(Error::new(
            *span,
            format!("the initial value of `{y}` must be a constant, it reads `{x}`"),
        ));
    }
    if sig.dir == Some(Dir::Input) {
        return Err(Error::new(
            init.span,
            format!("`{y}` is an input: only signals of the module have initial values"),
        ));
    }
    let array = ast::array_of(y);
    let first = driver
        .iter()
        .find(|(x, _)| ast::array_of(x) == array)
        .map(|(_, d)| *d);
    let (span, i) = match first {
        Some(d) => d,
        None if array != y => {
            return Err(Error::new(
                init.span,
                format!(
                    "`{y}`: nothing writes the array `{array}`, so its elements are constants: \
                     load it from a file (`$readmemb`, `$readmemh`) instead"
                ),
            ));
        }
        None => return Err(Error::new(sig.span, format!("`{y}` is never driven"))),
    };
    let kind = match &m.items[i] {
        Item::AlwaysFf { .. } => return Ok(()),
        Item::Assign { .. } => "an `assign`".to_string(),
        Item::AlwaysComb { .. } => m.items[i].a_block(),
        Item::Instance(u) => format!("the instance `{}`", u.name),
    };
    Err(Error::new(
        init.span,
        format!(
            "`{y}` is driven by {kind} at {span}, which computes it in every time step: an \
             initial value has no effect on it (only registers have initial values)"
        ),
    ))
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
    // all elements of an array have one driver
    let array = ast::array_of(lhs);
    let key = driver
        .keys()
        .find(|y| ast::array_of(y) == array)
        .cloned()
        .unwrap_or_else(|| lhs.clone());
    match driver.get(&key) {
        Some((prev, i)) if *i != item && array != lhs => Err(Error::new(
            span,
            format!(
                "`{lhs}`: the array `{array}` is already driven at {prev} (an array has one \
                 driver)"
            ),
        )),
        Some((prev, i)) if *i != item => Err(Error::new(
            span,
            format!("`{lhs}` already driven at {prev}"),
        )),
        _ => {
            driver.entry(lhs.clone()).or_insert((span, item));
            Ok(sig.ty)
        }
    }
}

/// The kind of block a statement is in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Proc {
    Ff,
    /// A combinational block, with its kind as written ([`Item::block`]).
    Comb(&'static str),
}

/// Checks the event list `list` of a combinational plain `always` (`@(a or
/// b)`): every name is a signal or an array, and every signal the block
/// reads, other than its own targets, is listed (an array name stands for
/// its elements). Then the block means `always @*`. A simulator runs the
/// block only when a listed signal changes, while synthesis ignores the
/// list, so with a missing signal the two would disagree.
fn event_list(env: &Env, m: &Module, item: &Item, list: &[(Name, Span)]) -> Result<()> {
    for (x, span) in list {
        if !m.arrays.iter().any(|a| a.name == *x) {
            lookup(env, x, *span)?;
        }
    }
    let targets: BTreeSet<&str> = item
        .targets()
        .into_iter()
        .map(|y| ast::array_of(y))
        .collect();
    for (x, span) in item.reads() {
        let listed = |x: &str| list.iter().any(|(l, _)| l == x);
        if targets.contains(ast::array_of(x)) || listed(x) || listed(ast::array_of(x)) {
            continue;
        }
        return Err(Error::new(
            span,
            format!(
                "`{x}` is read here but missing from the event list of the `always` at {}: a \
                 simulator runs the block only when a listed signal changes, synthesis \
                 ignores the list, so the two would disagree; add `{}` to the list or write \
                 `always @*`",
                item.span(),
                ast::array_of(x)
            ),
        ));
    }
    Ok(())
}

/// Checks that an `always @*` block reads a signal other than its own
/// targets. A simulator runs such a block only when a signal it reads
/// changes, and not at time 0, so a block reading none never runs and its
/// targets stay x, while a combinational block computes them in every time
/// step.
fn star_reads(item: &Item) -> Result<()> {
    let targets: BTreeSet<&str> = item
        .targets()
        .into_iter()
        .map(|y| ast::array_of(y))
        .collect();
    if item
        .reads()
        .iter()
        .any(|(x, _)| !targets.contains(ast::array_of(x)))
    {
        return Ok(());
    }
    Err(Error::new(
        item.span(),
        format!(
            "this {} reads no signal: a simulator runs an `always @*` block only when a \
             signal it reads changes, so it never runs this one and its targets stay x; use \
             `assign` or `always_comb`, which also run at time 0",
            item.block()
        ),
    ))
}

/// Checks the reset branch of a block with an asynchronous reset: only
/// assignments `q <= c` of constants `c` (expressions that read no signal)
/// to whole registers, each register at most once. The reset sets these
/// registers to these values whenever it is active, so a value must not
/// depend on the time of the reset.
fn reset_branch<'s>(s: &'s Stmt, seen: &mut BTreeSet<&'s Name>) -> Result<()> {
    match &s.kind {
        StmtKind::Block(stmts) => stmts.iter().try_for_each(|s| reset_branch(s, seen)),
        StmtKind::NonBlocking { lhs, rhs } => {
            let LvalueKind::Var(q) = &lhs.kind else {
                return Err(Error::new(
                    lhs.span,
                    "the reset branch may only assign whole registers (`q <= 8'd0;`)",
                ));
            };
            let mut reads = Vec::new();
            rhs.reads(&mut reads);
            if let Some((x, span)) = reads.first() {
                return Err(Error::new(
                    *span,
                    format!("the reset value of `{q}` must be a constant, it reads `{x}`"),
                ));
            }
            if !seen.insert(q) {
                return Err(Error::new(lhs.span, format!("`{q}` is reset twice")));
            }
            Ok(())
        }
        _ => Err(Error::new(
            s.span,
            "the reset branch may only assign constants to registers (`q <= 8'd0;`)",
        )),
    }
}

fn drives_something(item: &Item) -> Result<()> {
    if item.targets().is_empty() {
        return Err(Error::new(
            item.span(),
            format!("this {} drives nothing", item.block()),
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
            names(env, c)?;
            stmt(env, a, proc, drive)?;
            stmt(env, b, proc, drive)
        }
        StmtKind::Case {
            sel,
            items,
            default,
            ..
        } => {
            names(env, sel)?;
            for item in items {
                for l in &item.labels {
                    names(env, l)?;
                }
                stmt(env, &item.body, proc, drive)?;
            }
            match default {
                Some(d) => stmt(env, d, proc, drive),
                None => Ok(()),
            }
        }
        StmtKind::NonBlocking { .. } | StmtKind::Blocking { .. } | StmtKind::Unknown { .. } => {
            let blocking = matches!(
                s.kind,
                StmtKind::Blocking { .. } | StmtKind::Unknown { blocking: true, .. }
            );
            // an always_ff takes both kinds (see `one_kind_of_assignment`)
            if let (Proc::Comb(kind), false) = (proc, blocking) {
                return Err(Error::new(
                    s.span,
                    format!(
                        "non-blocking assignment not allowed in {}, use `=`",
                        // `always_comb` without an article, as the keyword
                        if kind.starts_with('`') {
                            kind.to_string()
                        } else {
                            ast::with_article(kind)
                        }
                    ),
                ));
            }
            match &s.kind {
                StmtKind::NonBlocking { lhs, rhs } | StmtKind::Blocking { lhs, rhs } => {
                    lvalue(lhs, drive)?;
                    target_names(env, lhs)?;
                    names(env, rhs)
                }
                StmtKind::Unknown { lhs, size, .. } => unknown(env, lhs, *size, drive),
                _ => unreachable!(),
            }
        }
    }
}

/// Checks the target `lhs` of an assignment of an x literal of `size`
/// bits (`None` if unsized). The bits the literal sets must be known, so
/// the target has no variable index; and a sized literal narrower than its
/// target would be extended with zeros, so its x would cover only some
/// bits.
fn unknown(
    env: &Env,
    lhs: &Lvalue,
    size: Option<usize>,
    drive: &mut dyn FnMut(&Name, Span) -> Result<Ty>,
) -> Result<()> {
    let w = lvalue(lhs, drive)?;
    target_names(env, lhs)?;
    if pieces(env, lhs)
        .iter()
        .any(|p| p.variable.is_some() || p.elem.is_some())
    {
        return Err(Error::new(
            lhs.span,
            format!(
                "`{lhs}` has a variable index: x may only be assigned to targets with \
                 constant indices, so that the bits assigned again later are known"
            ),
        ));
    }
    if let Some(n) = size
        && n < w
    {
        return Err(Error::new(
            lhs.span,
            format!(
                "the x literal has {n} bit{}, but `{lhs}` has {w} bits, so the others \
                 would be 0: use `'x` for x in every bit",
                if n == 1 { "" } else { "s" }
            ),
        ));
    }
    Ok(())
}

/// Checks that every signal the indices of a target read is declared.
fn target_names(env: &Env, l: &Lvalue) -> Result<()> {
    let mut reads = Vec::new();
    l.reads(&mut reads);
    reads
        .into_iter()
        .try_for_each(|(x, span)| lookup(env, x, span).map(|_| ()))
}

/// Checks that every signal `e` reads is declared.
pub(super) fn names(env: &Env, e: &Expr) -> Result<()> {
    let mut reads = Vec::new();
    e.reads(&mut reads);
    reads
        .into_iter()
        .try_for_each(|(x, span)| lookup(env, x, span).map(|_| ()))
}

/// Checks an assignment target, records the driver of each of its signals
/// with `drive`, and returns its width. A bit may be assigned only once in
/// one target, so a select with a variable index must be the only piece of
/// its signal (the frontend accepts variable indices in blocks only).
fn lvalue(l: &Lvalue, drive: &mut dyn FnMut(&Name, Span) -> Result<Ty>) -> Result<usize> {
    fn walk(
        l: &Lvalue,
        drive: &mut dyn FnMut(&Name, Span) -> Result<Ty>,
        seen: &mut BTreeSet<(Name, usize)>,
        variable: &mut BTreeSet<Name>,
    ) -> Result<usize> {
        let (y, hi, lo) = match &l.kind {
            LvalueKind::Concat(ls) => {
                return ls.iter().map(|l| walk(l, drive, seen, variable)).sum();
            }
            LvalueKind::Elem(e) => {
                let mut ty = None;
                for y in e.signals() {
                    ty = Some(drive(y, l.span)?);
                }
                let t = ty.expect("an array with elements");
                let name = e.name();
                let w = match &e.sel {
                    None => t.width,
                    Some(sel) if sel.index().is_some() => {
                        nonzero_width(&name, sel, l.span)?;
                        let w = sel.width();
                        if w == 0 || w > t.width {
                            return Err(Error::new(
                                l.span,
                                format!(
                                    "`{l}` selects {w} bits of `{name}`, which has {}",
                                    t.width
                                ),
                            ));
                        }
                        w
                    }
                    Some(sel) => {
                        let (hi, lo) = constant_select(&name, t, sel, l.span)?;
                        hi + 1 - lo
                    }
                };
                let array = &e.array;
                if !variable.insert(array.clone())
                    || seen.iter().any(|(x, _)| ast::array_of(x) == array)
                {
                    return Err(overlap(l, array));
                }
                return Ok(w);
            }
            LvalueKind::Var(y) => (y, drive(y, l.span)?.width - 1, 0),
            LvalueKind::Select(y, sel) if sel.index().is_some() => {
                let t = drive(y, l.span)?;
                nonzero_width(y, sel, l.span)?;
                let w = sel.width();
                if w == 0 || w > t.width {
                    return Err(Error::new(
                        l.span,
                        format!("`{l}` selects {w} bits of `{y}`, which has {}", t.width),
                    ));
                }
                if !variable.insert(y.clone()) || seen.iter().any(|(x, _)| x == y) {
                    return Err(overlap(l, y));
                }
                return Ok(w);
            }
            LvalueKind::Select(y, sel) => {
                let t = drive(y, l.span)?;
                let (hi, lo) = constant_select(y, t, sel, l.span)?;
                (y, hi, lo)
            }
        };
        if variable.contains(ast::array_of(y)) {
            return Err(overlap(l, ast::array_of(y)));
        }
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
    fn overlap(l: &Lvalue, y: &str) -> Error {
        Error::new(
            l.span,
            format!(
                "`{y}` assigned twice in one target: a select or element of `{y}` with a \
                 variable index may overlap its other pieces"
            ),
        )
    }
    walk(l, drive, &mut BTreeSet::new(), &mut BTreeSet::new())
}
