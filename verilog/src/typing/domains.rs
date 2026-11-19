//! Clock domains: the root input clock of every clock, and the reads and writes across them
//! ([`domains`]).

use super::Result;
use super::definite::one_kind_of_assignment;
use crate::ast::{Dir, Edge, Error, Expr, ExprKind, Index, Item, LvalueKind, Name, Select, Span};
use crate::elab::{self, Inst};
use std::collections::{BTreeMap, BTreeSet};

/// What drives a signal, as [`domains`] sees it: an item of an instance,
/// or the expression connected to an input of an instance (in the
/// instantiating one).
#[derive(Clone, Copy)]
pub(super) enum Source<'m> {
    Item(&'m Item),
    Conn(&'m Expr, Span),
}

/// Checks where the clocks of an elaborated hierarchy come from and that no
/// register reads a value from another clock domain. Returns the input
/// clocks the design is clocked by, by name. Expects every instance to
/// pass [`check`](super::check) and the hierarchy to be free of loops: a register
/// clocked, through the chain below, by itself is a loop of awaits (its
/// edge test awaits its clock), which lowering rejects.
///
/// A clock is either an input of the top module or an output of a
/// blackbox (an *input clock*: both are free values, set by the
/// environment in every time step, such as the output of a PLL), or a
/// register (a *derived clock*, e.g. a divider `slow <= !slow`), or a
/// copy or a constant bit select of one of these driven by an `assign`
/// (`assign c = cnt[3]`), also through instance ports. Its *root* is the
/// input clock at the end of this chain. Anything else is rejected: a
/// clock computed by an operator or in an `always_comb` (a gated clock)
/// could glitch within a time step, which rounds cannot express, and an
/// input clock must have 1 bit (a bit of a wider input is no clock).
///
/// An asynchronous reset follows the same chain and must end at an input
/// or a blackbox output (of any width) or at a register: it acts as a level within the time
/// step, and a reset computed by an operator could glitch and reset
/// registers that rounds keep. Releasing the reset is no crossing of its
/// own: the block's `if (rst)` reads the reset at the clock edge like any
/// other value, so the rule below applies to it.
///
/// The domain of a signal is the set of roots of the clocks of the
/// registers it depends on through combinational logic (`assign`,
/// `always_comb`, instance connections): the root of its own clock for a
/// register, none for an input or a blackbox output. A register may read
/// only signals whose
/// domain is empty or its own root: registers on either edge of one input
/// clock and of the clocks derived from it are *related*, and reads among
/// them see the value from before the edge (see the timing assumptions in
/// [`crate::lower`]). Different input clocks are unrelated, and a value
/// sampled from another domain needs a rule for an edge that coincides
/// with a change of the value, which crossings do not have yet. Signals
/// are compared by name (see [`Inst::naming`]), so a port connected to a
/// signal is that signal.
///
/// A register on the chain of a derived clock may have an asynchronous
/// reset from another domain (an input, or a register on another input
/// clock). Asserting it sets the register at a time unrelated to the root,
/// which is an edge of the derived clock: a falling one for reset value 0,
/// a rising one for 1 (either, if the value is not a plain constant). A
/// block that such an edge moves, directly or through further registers on
/// the chain, may then read only signals that depend on no register, or
/// only on registers on the same edge of the same clock: anything else
/// would be sampled at a time unrelated to its changes. Blocks on the
/// other edge never run at a reset, so the rule above suffices for them.
///
/// A blocking `=` in an `always_ff` means the same as `<=` within its own
/// block (see [`check`](super::check)), but another block that runs at the same clock
/// edge and reads the signal, directly or through combinational logic,
/// races with it: IEEE 1800-2017 (§4.7) leaves the order of the two blocks
/// open, so a simulator may give the reader the value before or after the
/// assignment, while synthesis (and the translation) gives the value
/// before. Such a read is rejected.
///
/// An `always_ff` block may not read its own clock, nor anything on the
/// chain its clock comes from (copies, the registers of derived clocks and
/// their clocks, up to the input clock), directly or through
/// combinational logic: such a value changes in the time step of the very
/// edge that runs the block, so the setup and hold times the semantics
/// assumes (see [`crate::lower`]) cannot hold for it. A simulator reads
/// the value after the edge (`if (clk)` at `posedge clk` is true), the
/// translation the value before it (false), and hardware races. Other
/// blocks may read clocks as data.
pub fn domains(inst: &Inst) -> Result<BTreeSet<Name>> {
    let h = Hierarchy::new(inst);
    let roots = h.roots()?;
    let comb = h.comb_reads();
    own_clock_reads(&h, &comb)?;
    // the domain of every register, and of every signal through
    // combinational logic
    let mut dom: BTreeMap<Name, BTreeSet<Name>> = BTreeMap::new();
    for (k, item) in h.blocks() {
        if let Item::AlwaysFf { clock, .. } = item {
            let r = &roots[&h.canon(k, clock)];
            for q in item.targets() {
                dom.insert(h.canon(k, q), BTreeSet::from([r.clone()]));
            }
        }
    }
    propagate(&comb, &mut dom);
    crossings(&h, &roots, &dom)?;
    reset_edge_reads(&h, &roots, &dom, &comb)?;
    blocking_races(&h, &comb)?;
    Ok(roots.into_values().collect())
}

/// A hierarchy as [`domains`] sees it.
pub(super) struct Hierarchy<'i> {
    /// The instances with their paths, each instance before its children.
    pub(super) insts: Vec<(&'i Inst, String)>,
    /// The name of every signal path (see [`Inst::naming`]).
    naming: BTreeMap<Name, Name>,
    /// The driver of every signal, by name, with its instance and its path.
    pub(super) drivers: BTreeMap<Name, (usize, Name, Source<'i>)>,
    /// The inputs of the top module and the outputs of blackboxes: free
    /// values, by name, with their widths.
    free: BTreeMap<Name, usize>,
}

impl<'i> Hierarchy<'i> {
    fn new(inst: &'i Inst) -> Self {
        let naming: BTreeMap<Name, Name> = inst.naming().into_iter().collect();
        let mut insts: Vec<(&Inst, String)> = Vec::new();
        let mut walk = vec![(inst, String::new())];
        while let Some((i, prefix)) = walk.pop() {
            for (u, child) in i.children.iter().rev() {
                walk.push((child, format!("{prefix}{u}.")));
            }
            insts.push((i, prefix));
        }
        let mut h = Hierarchy {
            insts,
            naming,
            drivers: BTreeMap::new(),
            free: BTreeMap::new(),
        };
        for (k, (i, prefix)) in h.insts.iter().enumerate() {
            for item in &i.module.items {
                match item {
                    Item::Instance(u) => {
                        for (c, f) in u.conns(Dir::Input) {
                            if let Some(e) = &c.actual
                                && c.shared().is_none()
                                && !f.blackbox
                            {
                                let path = format!("{prefix}{}.{}", u.name, c.port);
                                let source = Source::Conn(e, c.span);
                                h.drivers.insert(h.naming[&path].clone(), (k, path, source));
                            }
                        }
                    }
                    _ => {
                        for y in item.targets() {
                            let path = format!("{prefix}{y}");
                            h.drivers
                                .insert(h.canon(k, y), (k, path, Source::Item(item)));
                        }
                    }
                }
            }
        }
        h.free = (inst.module.ports.iter())
            .filter(|p| p.dir == Dir::Input)
            .map(|p| (p.decl.name.clone(), p.decl.ty.width))
            .collect();
        for (k, (i, _)) in h.insts.iter().enumerate() {
            for (c, f) in i.module.instances().flat_map(|u| u.conns(Dir::Output)) {
                if let (true, Some(y)) = (f.blackbox, c.shared()) {
                    h.free.insert(h.canon(k, y), f.ty.width);
                }
            }
        }
        h
    }

    /// The name of the signal `x` of instance `k`.
    pub(super) fn canon(&self, k: usize, x: &str) -> Name {
        self.naming[&format!("{}{x}", self.insts[k].1)].clone()
    }

    /// The items of every instance, with the instance's number.
    pub(super) fn blocks(&self) -> impl Iterator<Item = (usize, &'i Item)> + '_ {
        (self.insts.iter().enumerate())
            .flat_map(|(k, (i, _))| i.module.items.iter().map(move |item| (k, item)))
    }

    /// The root of the clock `c` of instance `k`, used at `span`; for a
    /// reset (`reset`), the input or register at the end of the chain.
    pub(super) fn root(&self, k: usize, c: &Name, span: Span, reset: bool) -> Result<Name> {
        let err = |msg: String| elab::context(Error::new(span, msg), &self.insts[k].1);
        let what = if reset { "reset" } else { "clock" };
        let mut x = self.canon(k, c);
        // the names on the chain so far
        let mut chain: Vec<Name> = Vec::new();
        loop {
            if chain.contains(&x) {
                return Err(err(format!("internal error: {what} loop through `{x}`")));
            }
            let Some((d, path, source)) = self.drivers.get(&x) else {
                return match self.free.get(&x) {
                    Some(1) => Ok(x),
                    Some(_) if reset => Ok(x),
                    Some(_) => Err(err(format!(
                        "the clock `{c}` is a bit of `{x}`, which has {} bits: input clocks \
                         must be 1-bit inputs (or blackbox outputs)",
                        self.free[&x]
                    ))),
                    None => Err(err(format!("internal error: `{x}` has no driver"))),
                };
            };
            // the next signal on the chain, a name in instance `d`
            let next = match source {
                Source::Item(Item::AlwaysFf { .. }) if reset => return Ok(x),
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
                    format!("the {what} `{c}` is computed combinationally")
                } else {
                    format!(
                        "the {what} `{c}` comes from `{path}`, computed combinationally at {at}"
                    )
                };
                let why = if reset {
                    "computed asynchronous resets are not supported (they could glitch)"
                } else {
                    "gated clocks are not supported"
                };
                return Err(err(format!("{msg}: {why}")));
            };
            chain.push(x);
            x = self.canon(*d, y);
        }
    }

    /// The root of every clock, by name, checked at each use (clock ports
    /// first, so that a bad clock is reported where it is passed down), and
    /// the chain of every asynchronous reset.
    fn roots(&self) -> Result<BTreeMap<Name, Name>> {
        let mut roots: BTreeMap<Name, Name> = BTreeMap::new();
        for (k, item) in self.blocks() {
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
                let r = self.root(k, c, span, false)?;
                roots.insert(self.canon(k, c), r);
            }
            if let Item::AlwaysFf { reset: Some(r), .. } = item {
                self.root(k, &r.name, r.span, true)?;
            }
        }
        Ok(roots)
    }

    /// What every combinational signal reads, by name: pairs (signal, read).
    pub(super) fn comb_reads(&self) -> Vec<(Name, Name)> {
        let mut comb: Vec<(Name, Name)> = Vec::new();
        for (k, item) in self.blocks() {
            match item {
                Item::AlwaysFf { .. } => {}
                Item::Assign { .. } | Item::AlwaysComb { .. } => {
                    for y in item.targets() {
                        for (x, _) in item.reads() {
                            comb.push((self.canon(k, y), self.canon(k, x)));
                        }
                    }
                }
                Item::Instance(u) => {
                    for (c, _) in u.conns(Dir::Input).filter(|(_, f)| !f.blackbox) {
                        let Some(e) = &c.actual else { continue };
                        let mut reads = Vec::new();
                        e.reads(&mut reads);
                        for (x, _) in reads {
                            let y = self.canon(k, &format!("{}.{}", u.name, c.port));
                            comb.push((y, self.canon(k, x)));
                        }
                    }
                }
            }
        }
        comb
    }
}

/// Extends the sets of `sets` along the combinational reads `comb` (pairs
/// (signal, read)) until nothing changes: afterwards a signal's set holds
/// those of everything it depends on through combinational logic.
pub(super) fn propagate<T: Ord + Clone>(
    comb: &[(Name, Name)],
    sets: &mut BTreeMap<Name, BTreeSet<T>>,
) {
    loop {
        let mut changed = false;
        for (y, x) in comb {
            let add = sets.get(x).cloned().unwrap_or_default();
            let d = sets.entry(y.clone()).or_default();
            let before = d.len();
            d.extend(add);
            changed |= d.len() != before;
        }
        if !changed {
            break;
        }
    }
}

/// How a clock is named in messages: with its root if derived.
fn clocked(roots: &BTreeMap<Name, Name>, c: &Name) -> String {
    match &roots[c] {
        r if r == c => format!("`{c}`"),
        r => format!("`{c}`, derived from `{r}`"),
    }
}

/// Rejects a read, in an `always_ff` block, of a signal from another clock
/// domain (see [`domains`]).
fn crossings(
    h: &Hierarchy,
    roots: &BTreeMap<Name, Name>,
    dom: &BTreeMap<Name, BTreeSet<Name>>,
) -> Result<()> {
    for (k, item) in h.blocks() {
        let Item::AlwaysFf { clock, .. } = item else {
            continue;
        };
        let c = h.canon(k, clock);
        let r = &roots[&c];
        for (x, span) in item.reads() {
            let Some(other) = dom
                .get(&h.canon(k, x))
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
                clocked(roots, &c)
            );
            return Err(elab::context(Error::new(span, msg), &h.insts[k].1));
        }
    }
    Ok(())
}

/// Rejects a read, in an `always_ff` block on an edge that an asynchronous
/// reset from another domain can make, of a signal that depends on
/// registers on other clocks or edges (see [`domains`]). Such a reset sets
/// a register at a time unrelated to its clock, which is an edge of a clock
/// derived from the register; a block on that edge may read only signals
/// that depend on no register, or only on registers on the same edge of the
/// same clock (they change together).
fn reset_edge_reads(
    h: &Hierarchy,
    roots: &BTreeMap<Name, Name>,
    dom: &BTreeMap<Name, BTreeSet<Name>>,
    comb: &[(Name, Name)],
) -> Result<()> {
    // the edges every signal depends on through combinational logic: the
    // clock and edge of each register, by name, like the domains
    let mut edges: BTreeMap<Name, BTreeSet<(Name, bool)>> = BTreeMap::new();
    for (k, item) in h.blocks() {
        if let Item::AlwaysFf { clock, edge, .. } = item {
            for q in item.targets() {
                let e = (h.canon(k, clock), *edge == Edge::Pos);
                edges.insert(h.canon(k, q), BTreeSet::from([e]));
            }
        }
    }
    propagate(comb, &mut edges);
    for (k, item) in h.blocks() {
        let Item::AlwaysFf { clock, edge, .. } = item else {
            continue;
        };
        let c = h.canon(k, clock);
        let r = &roots[&c];
        let Some(made) = reset_edge(h, dom, &c, *edge, r)? else {
            continue;
        };
        let own = BTreeSet::from([(c.clone(), *edge == Edge::Pos)]);
        for (x, span) in item.reads() {
            let Some(e) = edges.get(&h.canon(k, x)) else {
                continue;
            };
            if e.is_empty() || *e == own {
                continue;
            }
            let q = item.targets()[0];
            let msg = format!(
                "`{q}` (clocked by {}) reads `{x}`, which depends on registers on other \
                 clocks or edges, but {made} at a time unrelated to `{r}`: crossings \
                 between unrelated clocks are not supported yet",
                clocked(roots, &c)
            );
            return Err(elab::context(Error::new(span, msg), &h.insts[k].1));
        }
    }
    Ok(())
}

/// An event that runs `always_ff` blocks, as a simulator sees it: a rising
/// (`true`) or falling edge of a bit of a signal, by name, after following
/// copies to the signal that makes the edge.
pub(super) type Event = (Name, usize, bool);

/// The event `edge` of the clock `x` of instance `k`; adds to `with` the
/// event and those it can happen together with in one step of a
/// simulator: a clock assigned with a blocking `=` changes while the blocks
/// at its own clock edge run (not after them, as with `<=`), so its edges
/// belong to that event too. (A register with an asynchronous reset is
/// assigned with `<=`, as its reset branch is.)
pub(super) fn event(
    h: &Hierarchy,
    blocking: &BTreeMap<Name, ((usize, usize), Span)>,
    k: usize,
    x: &Name,
    edge: Edge,
    with: &mut BTreeSet<Event>,
) -> Event {
    let (mut x, mut bit) = (h.canon(k, x), None);
    while let Some((d, _, source)) = h.drivers.get(&x) {
        let next = match source {
            Source::Item(Item::Assign { lhs, rhs, .. })
                if matches!(lhs.kind, LvalueKind::Var(_)) =>
            {
                copy_bit(rhs)
            }
            Source::Conn(e, _) => copy_bit(e),
            _ => None,
        };
        let Some((y, b)) = next else { break };
        bit = b.or(bit);
        x = h.canon(*d, y);
    }
    let e = (x, bit.unwrap_or(0), edge == Edge::Pos);
    if with.insert(e.clone())
        && let Some(((bk, bn), _)) = blocking.get(&e.0)
        && let Item::AlwaysFf { clock, edge, .. } = &h.insts[*bk].0.module.items[*bn]
    {
        event(h, blocking, *bk, clock, *edge, with);
    }
    e
}

/// Rejects a read, in an `always_ff` block, of a signal that another
/// `always_ff` block assigns with a blocking `=` at the same event (see
/// [`domains`]), directly or through combinational logic. Two blocks run
/// together if their clock edges can happen in the same simulator step
/// ([`event`]); the two edges of one bit never do. Reads of `<=`
/// registers are not affected: every block reads them before any of them
/// changes, in simulation as in the translation.
pub(super) fn blocking_races(h: &Hierarchy, comb: &[(Name, Name)]) -> Result<()> {
    // the registers assigned with `=`: their block (instance, item) and the
    // first such assignment
    let mut blocking: BTreeMap<Name, ((usize, usize), Span)> = BTreeMap::new();
    for (k, (i, _)) in h.insts.iter().enumerate() {
        for (n, item) in i.module.items.iter().enumerate() {
            if let Item::AlwaysFf { body, .. } = item {
                let mut first = BTreeMap::new();
                one_kind_of_assignment(body, item.block(), &mut first)?;
                for (q, (span, b)) in first {
                    if b {
                        blocking.insert(h.canon(k, q), ((k, n), span));
                    }
                }
            }
        }
    }
    if blocking.is_empty() {
        return Ok(());
    }
    // the `=` registers every signal depends on through combinational logic
    let mut dep: BTreeMap<Name, BTreeSet<Name>> = blocking
        .keys()
        .map(|q| (q.clone(), BTreeSet::from([q.clone()])))
        .collect();
    propagate(comb, &mut dep);
    // the event of every block, with the events it can coincide with
    let mut events: BTreeMap<(usize, usize), (Event, BTreeSet<Event>)> = BTreeMap::new();
    for (k, (i, _)) in h.insts.iter().enumerate() {
        for (n, item) in i.module.items.iter().enumerate() {
            if let Item::AlwaysFf { clock, edge, .. } = item {
                let mut with = BTreeSet::new();
                let e = event(h, &blocking, k, clock, *edge, &mut with);
                events.insert((k, n), (e, with));
            }
        }
    }
    let coincide = |(e1, w1): &(Event, BTreeSet<Event>), (e2, w2): &(Event, BTreeSet<Event>)| {
        let opposite = e1.0 == e2.0 && e1.1 == e2.1 && e1.2 != e2.2;
        !opposite && !w1.is_disjoint(w2)
    };
    for (k, (i, prefix)) in h.insts.iter().enumerate() {
        for (n, item) in i.module.items.iter().enumerate() {
            let Item::AlwaysFf { clock, .. } = item else {
                continue;
            };
            let reads = &events[&(k, n)];
            for (x, span) in item.reads() {
                let Some(qs) = dep.get(&h.canon(k, x)) else {
                    continue;
                };
                for q in qs {
                    let (writer, at) = blocking[q];
                    if writer == (k, n) || !coincide(&events[&writer], reads) {
                        continue;
                    }
                    let block = h.insts[writer.0].0.module.items[writer.1].block();
                    let what = if h.canon(k, x) == *q {
                        format!("`{x}`,")
                    } else {
                        format!("`{x}`, which depends on `{q}`,")
                    };
                    let y = item.targets()[0];
                    let msg = format!(
                        "`{y}` (clocked by `{clock}`) reads {what} assigned with a blocking `=` \
                         at {at} in another {block} block that runs at the same clock edge: \
                         simulators may run the two blocks in either order, so the read could \
                         see the value from before or after the assignment (a race); assign \
                         `{q}` with `<=`"
                    );
                    return Err(elab::context(Error::new(span, msg), prefix));
                }
            }
        }
    }
    Ok(())
}

/// Rejects a read, in an `always_ff` block, of a signal on the chain of
/// its own clock (see [`domains`]), directly or through combinational
/// logic.
pub(super) fn own_clock_reads(h: &Hierarchy, comb: &[(Name, Name)]) -> Result<()> {
    // the chain of every block's clock, by name, from the clock to its root
    let mut chains: BTreeMap<(usize, usize), Vec<Name>> = BTreeMap::new();
    for (k, (i, _)) in h.insts.iter().enumerate() {
        for (n, item) in i.module.items.iter().enumerate() {
            let Item::AlwaysFf { clock, .. } = item else {
                continue;
            };
            let mut chain = vec![h.canon(k, clock)];
            while let Some((d, _, source)) = h.drivers.get(chain.last().unwrap()) {
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
                // `domains` has rejected clock loops already
                match next.map(|y| h.canon(*d, y)) {
                    Some(y) if !chain.contains(&y) => chain.push(y),
                    _ => break,
                }
            }
            chains.insert((k, n), chain);
        }
    }
    // the signals on some chain that every signal depends on through
    // combinational logic
    let mut dep: BTreeMap<Name, BTreeSet<Name>> = chains
        .values()
        .flatten()
        .map(|c| (c.clone(), BTreeSet::from([c.clone()])))
        .collect();
    propagate(comb, &mut dep);
    for (k, (i, prefix)) in h.insts.iter().enumerate() {
        for (n, item) in i.module.items.iter().enumerate() {
            let Item::AlwaysFf { clock, .. } = item else {
                continue;
            };
            let chain = &chains[&(k, n)];
            for (x, span) in item.reads() {
                let Some(c) = dep
                    .get(&h.canon(k, x))
                    .and_then(|d| chain.iter().find(|c| d.contains(*c)))
                else {
                    continue;
                };
                let what = match (h.canon(k, x) == *c, *c == chain[0]) {
                    (true, true) => format!("`{x}`, its clock"),
                    (true, false) => format!("`{x}`, from which its clock is derived"),
                    (false, true) => format!("`{x}`, which depends on its clock"),
                    (false, false) => {
                        format!("`{x}`, which depends on `{c}`, from which its clock is derived")
                    }
                };
                let y = item.targets()[0];
                let msg = format!(
                    "`{y}` (clocked by `{clock}`) reads {what}: that value changes in the time \
                     step of the edge that runs the block, so the read races with the edge \
                     (a simulator reads the value after the edge, the translation the value \
                     before it); a block may not read the chain of its own clock"
                );
                return Err(elab::context(Error::new(span, msg), prefix));
            }
        }
    }
    Ok(())
}

/// Whether an edge `edge` of the clock `c` (by name) can be made by an
/// asynchronous reset from a domain other than `r`, the root of `c`: a
/// description of that reset if so. Follows the chain of `c` as `root`
/// does, with the edge of the current signal that moves the block and the
/// bit of the next register that the current signal is (a bit select): a
/// reset to 0 makes a falling edge, a reset to 1 a rising one, and a reset
/// value that is not a plain constant is taken to make either. Past a
/// register, any edge of its clock that it runs at moves it.
pub(super) fn reset_edge(
    h: &Hierarchy,
    dom: &BTreeMap<Name, BTreeSet<Name>>,
    c: &Name,
    edge: Edge,
    r: &Name,
) -> Result<Option<String>> {
    let (mut x, mut want, mut bit) = (c.clone(), edge, None);
    while let Some((d, path, source)) = h.drivers.get(&x) {
        let next = match source {
            Source::Item(Item::AlwaysFf {
                clock,
                edge,
                reset,
                body,
                ..
            }) => {
                let local = &path[h.insts[*d].1.len()..];
                let value = body.reset_values().into_iter().find(|(q, _)| *q == local);
                if let (Some(rst), Some((_, v))) = (reset, value) {
                    // the domain of the reset: the root of its register's
                    // clock, or the input itself
                    let end = h.root(*d, &rst.name, rst.span, true)?;
                    let same = match dom.get(&end) {
                        Some(roots) => roots.contains(r),
                        None => end == *r,
                    };
                    let b = reset_bit(v, bit);
                    if !same && b.is_none_or(|b| b == (want == Edge::Pos)) {
                        let of = match bit {
                            Some(b) => format!("bit {b} of it"),
                            None => "it".to_string(),
                        };
                        let what = match (b, want) {
                            (Some(b), Edge::Pos) => {
                                format!("sets {of} to {}, a rising edge", u8::from(b))
                            }
                            (Some(b), Edge::Neg) => {
                                format!("sets {of} to {}, a falling edge", u8::from(b))
                            }
                            (None, _) => format!("sets {of} to `{v}`, which could make an edge"),
                        };
                        return Ok(Some(format!(
                            "the asynchronous reset `{}` of `{path}` {what}",
                            rst.name
                        )));
                    }
                }
                want = *edge;
                Some((clock, None))
            }
            Source::Item(Item::Assign { rhs, .. }) => copy_bit(rhs),
            Source::Conn(e, _) => copy_bit(e),
            _ => None,
        };
        let Some((y, b)) = next else { break };
        bit = match source {
            Source::Item(Item::AlwaysFf { .. }) => None,
            _ => b.or(bit),
        };
        x = h.canon(*d, y);
    }
    Ok(None)
}

/// Bit `bit` (0 if not given) of the reset value `v`, assigned to the
/// register, if `v` is a literal, a parameter or `'0`/`'1`: extended by its
/// own signedness, as an assignment does.
pub(super) fn reset_bit(v: &Expr, bit: Option<usize>) -> Option<bool> {
    let k = bit.unwrap_or(0);
    match &v.kind {
        ExprKind::Lit { ty, value } | ExprKind::Const { ty, value, .. } => Some(if k < ty.width {
            value.unsigned().bit(k as u64)
        } else {
            ty.signed && value.msb()
        }),
        ExprKind::Fill(b) => Some(*b),
        _ => None,
    }
}

/// The signal `e` copies (all of it or one bit): a name or a constant bit
/// select of a name.
pub(super) fn copy(e: &Expr) -> Option<&Name> {
    copy_bit(e).map(|(x, _)| x)
}

/// [`copy`], with the bit for a bit select.
pub(super) fn copy_bit(e: &Expr) -> Option<(&Name, Option<usize>)> {
    match &e.kind {
        ExprKind::Var(x) => Some((x, None)),
        ExprKind::Select(x, Select::Bit(Index::Const(b))) => Some((x, Some(*b))),
        _ => None,
    }
}
