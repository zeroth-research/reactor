//! Definite assignment: no read of a bit before it is written in a block, no latch, no live x
//! (`definite`, run by [`check`](super::check)); and the bits of registers that no assignment
//! writes ([`unwritten`], run by [`checked`](super::checked)).

use super::fragment::{Piece, pieces_in};
use super::{Env, Result};
use crate::ast::{
    BlockKind, Error, Expr, ExprKind, Item, Lvalue, Name, Select, Span, Stmt, StmtKind, Warning,
};
use std::collections::BTreeMap;

/// The bits `bits` (descending or not) of the signal `y` of width `width`,
/// as SystemVerilog: `` `y` `` for all of them, else `` `y[7:4]`, `y[1]` ``.
pub(super) fn bits_text(y: &str, bits: &[usize], width: usize) -> String {
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

/// Checks the block `item` of a module whose names and drivers have been
/// checked (as written, not sized): [`check`](super::check) runs it on
/// each block in source order, so that a design with several errors is
/// reported by its first one. It checks that an `always_comb` block
/// assigns each signal it drives on
/// every path before reading it and by the end of the block (no
/// latch), and that the x a block of either kind assigns is dead.
///
/// A signal not assigned on some path would keep its value from the
/// previous evaluation of the block: hardware needs a latch for that, a
/// storage element outside the clocked cycle model. A read of a signal of
/// the block before it is assigned would read that same old value. The
/// analysis is syntactic: a `case` without `default` may assign nothing
/// even if its labels cover every value.
///
/// The analysis is per bit: `y[3:0] = a; y[7:4] = b;` assigns all of `y`.
/// A read with a variable index reads every bit of its signal; an
/// assignment with a variable index assigns no bit for sure and keeps the
/// others, so the whole signal must be assigned before it.
///
/// Values are 2-state, so an assignment of x (`y = 'x;`, `q <= 'x;`)
/// leaves its bits unassigned, and every path must assign them again: in
/// an `always_comb` before they are read, in both kinds of blocks before
/// the end. (Synthesis reads a live x as a don't-care, free to pick any
/// value; that is not modelled.) In an `always_ff` a register not assigned
/// keeps its value, and reads see the values from before the clock edge;
/// besides x, only blocking assignments are checked there: no read may
/// follow a `=` that may have assigned the bits it reads (on any path), so
/// that every `=` means the same as `<=`.
pub(super) fn definite(env: &Env, item: &Item) -> Result<()> {
    let targets = item.targets();
    let body = item.body().expect("a block");
    let comb = matches!(item, Item::AlwaysComb { .. });
    let mut d = Definite {
        env,
        targets: &targets,
        comb,
        block: item.kind().expect("a block"),
        partial: BTreeMap::new(),
    };
    let mut st = State::default();
    d.stmt(body, &mut st)?;
    for y in &targets {
        let n = env[*y].ty.width;
        let missing: Vec<usize> = (0..n).filter(|b| !st.done.contains(y, *b)).collect();
        if let Some(e) = st.live_x(y, &missing, n, Until::End) {
            return Err(e);
        }
        if missing.is_empty() || !comb {
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
                    "{bits} is never assigned in this {}, \
                     so it would keep its old value (a latch)",
                    item.block()
                ),
            ),
        });
    }
    Ok(())
}

/// A set of bits of signals: for each signal, a bit set (bit `b` is bit
/// `b % 64` of word `b / 64`).
#[derive(Clone, Default)]
struct Bits(BTreeMap<Name, Vec<u64>>);

impl Bits {
    fn contains(&self, y: &str, b: usize) -> bool {
        let word = self.0.get(y).and_then(|ws| ws.get(b / 64));
        word.is_some_and(|w| w >> (b % 64) & 1 == 1)
    }

    fn insert(&mut self, y: &Name, b: usize) {
        if !self.0.contains_key(y) {
            self.0.insert(y.clone(), Vec::new());
        }
        let ws = self.0.get_mut(y).expect("just added");
        if ws.len() <= b / 64 {
            ws.resize(b / 64 + 1, 0);
        }
        ws[b / 64] |= 1 << (b % 64);
    }

    fn remove(&mut self, y: &str, b: usize) {
        if let Some(w) = self.0.get_mut(y).and_then(|ws| ws.get_mut(b / 64)) {
            *w &= !(1 << (b % 64));
        }
    }

    /// The bits in both sets.
    fn intersection(&self, other: &Bits) -> Bits {
        let mut out = BTreeMap::new();
        for (y, a) in &self.0 {
            if let Some(b) = other.0.get(y) {
                out.insert(y.clone(), a.iter().zip(b).map(|(a, b)| a & b).collect());
            }
        }
        Bits(out)
    }

    /// The bits in either set.
    fn union(&self, other: &Bits) -> Bits {
        let mut out = self.0.clone();
        for (y, b) in &other.0 {
            let a = out.entry(y.clone()).or_default();
            if a.len() < b.len() {
                a.resize(b.len(), 0);
            }
            for (a, b) in a.iter_mut().zip(b) {
                *a |= b;
            }
        }
        Bits(out)
    }

    /// The signals with a bit in `self` that is not in `other`, in order.
    fn not_within<'b>(&'b self, other: &'b Bits) -> impl Iterator<Item = &'b Name> {
        self.0.iter().filter_map(move |(y, a)| {
            let b = other.0.get(y).map_or(&[][..], |b| b.as_slice());
            let word = |k: usize| b.get(k).copied().unwrap_or(0);
            a.iter()
                .enumerate()
                .any(|(k, a)| a & !word(k) != 0)
                .then_some(y)
        })
    }
}

/// What the paths to a point of a block have assigned.
#[derive(Clone, Default)]
struct State {
    /// The bits assigned on every path.
    done: Bits,
    /// The bits assigned x and not assigned again on some path, with the
    /// first such assignment of x.
    x: BTreeMap<(Name, usize), Span>,
    /// In an `always_ff`: the bits a blocking `=` may have assigned on some
    /// path, with the first such assignment.
    blocked: BTreeMap<(Name, usize), Span>,
}

impl State {
    /// The error for the bits of `y` among `missing` (of `n`) that hold an
    /// x, if any; `when` says what comes before they are assigned again.
    fn live_x(&self, y: &Name, missing: &[usize], n: usize, when: Until) -> Option<Error> {
        let bits: Vec<usize> = missing
            .iter()
            .copied()
            .filter(|b| self.x.contains_key(&(y.clone(), *b)))
            .collect();
        let span = bits.iter().map(|b| self.x[&(y.clone(), *b)]).min()?;
        Some(Error::new(
            span,
            format!(
                "{} is assigned x here and not assigned again on every path {when}: \
                 x is accepted only when it is overwritten (a live x is a don't-care \
                 in synthesis, which is not modelled)",
                bits_text(y, &bits, n)
            ),
        ))
    }
}

/// What an x reaches before it is assigned again, for the message about a
/// live x.
#[derive(Clone, Copy)]
enum Until<'a> {
    /// The end of the block.
    End,
    /// A read, at this position.
    Read(Span),
    /// An assignment to this target with a variable index, which keeps the
    /// bits it does not assign.
    Keeping(&'a Lvalue),
}

impl std::fmt::Display for Until<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Until::End => write!(f, "before the end of the block"),
            Until::Read(span) => write!(f, "before it is read at {span}"),
            Until::Keeping(lhs) => {
                write!(f, "before `{lhs}`, which keeps the bits it does not assign")
            }
        }
    }
}

/// Checks that every signal assigned in the `always_ff` body `s` is
/// assigned either only with `=` or only with `<=`; `first` maps each
/// signal seen so far to its first assignment (span, blocking); `block` is
/// the kind of block as written.
pub(super) fn one_kind_of_assignment<'s>(
    s: &'s Stmt,
    block: BlockKind,
    first: &mut BTreeMap<&'s Name, (Span, bool)>,
) -> Result<()> {
    let (lhs, blocking) = match &s.kind {
        StmtKind::Blocking { lhs, .. } => (lhs, true),
        StmtKind::NonBlocking { lhs, .. } => (lhs, false),
        StmtKind::Unknown { lhs, blocking, .. } => (lhs, *blocking),
        _ => {
            return s
                .children()
                .into_iter()
                .try_for_each(|c| one_kind_of_assignment(c, block, first));
        }
    };
    let mut ys = Vec::new();
    lhs.targets(&mut ys);
    for y in ys {
        let (at, b) = *first.entry(y).or_insert((s.span, blocking));
        if b != blocking {
            let (this, other) = if blocking { ("=", "<=") } else { ("<=", "=") };
            return Err(Error::new(
                s.span,
                format!(
                    "`{y}` is assigned with `{this}` here and with `{other}` at {at}: in {}, \
                     assign each signal either only with `<=` or only with `=`",
                    block.with_article()
                ),
            ));
        }
    }
    Ok(())
}

/// Checks that no read among `reads` (in an `always_ff`) reads a bit that a
/// blocking `=` before it may have assigned: a `=` is accepted there only
/// where it means the same as `<=`, so its new value must never be read in
/// the block.
fn blocked_reads(
    reads: &[(&Name, usize, usize, Span)],
    st: &State,
    block: BlockKind,
) -> Result<()> {
    for (x, hi, lo, span) in reads {
        let Some(at) = (*lo..=*hi)
            .filter_map(|b| st.blocked.get(&((*x).clone(), b)))
            .min()
        else {
            continue;
        };
        return Err(Error::new(
            *span,
            format!(
                "`{x}` is read here after the blocking assignment at {at} may have \
                 assigned it: in {}, `=` is accepted only where it means \
                 the same as `<=` (nothing it assigns is read later in the block); \
                 use `<=`",
                block.with_article()
            ),
        ));
    }
    Ok(())
}

/// A statement with several paths, where a latch can start.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Branching {
    If,
    Case,
    /// A `case` without `default`: it has a path on which no label matches.
    CaseWithoutDefault,
}

/// `` `if` ``, `` `case` ``, …, for the message about a latch.
impl std::fmt::Display for Branching {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Branching::If => "`if`",
            Branching::Case => "`case`",
            Branching::CaseWithoutDefault => {
                "`case` (without `default`, nothing is assigned when no label matches)"
            }
        })
    }
}

struct Definite<'t> {
    env: &'t Env,
    targets: &'t [&'t Name],
    /// An `always_comb` (else an `always_ff`, where only x is checked).
    comb: bool,
    /// The kind of block as written.
    block: BlockKind,
    /// For each target, the first `if` or `case` after which some of its
    /// bits are assigned on some paths only.
    partial: BTreeMap<Name, (Span, Branching)>,
}

impl Definite<'_> {
    /// Checks the reads in `e` in an `always_comb`, with `st` what the
    /// paths so far assigned.
    fn reads(&self, e: &Expr, st: &State) -> Result<()> {
        let mut reads = Vec::new();
        bit_reads(self.env, e, &mut reads);
        if !self.comb {
            return blocked_reads(&reads, st, self.block);
        }
        for (x, hi, lo, span) in reads {
            if !self.targets.contains(&x) {
                continue;
            }
            let n = self.env[x].ty.width;
            let missing: Vec<usize> = (lo..=hi).filter(|b| !st.done.contains(x, *b)).collect();
            if let Some(e) = st.live_x(x, &missing, n, Until::Read(span)) {
                return Err(e);
            }
            if !missing.is_empty() {
                return Err(Error::new(
                    span,
                    format!(
                        "{} read before it is assigned on every path in this {} \
                         (it would read its value from the previous evaluation)",
                        bits_text(x, &missing, n),
                        self.block
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Adds to `st` what `s` assigns.
    fn stmt(&mut self, s: &Stmt, st: &mut State) -> Result<()> {
        match &s.kind {
            StmtKind::Block(stmts) => stmts.iter().try_for_each(|s| self.stmt(s, st)),
            StmtKind::If(c, a, b) => {
                self.reads(c, st)?;
                let branches = [a.as_ref(), b.as_ref()];
                self.merge(&branches, false, st, s.span, Branching::If)
            }
            StmtKind::Case {
                sel,
                items,
                default,
                ..
            } => {
                self.reads(sel, st)?;
                for l in items.iter().flat_map(|i| &i.labels) {
                    self.reads(l, st)?;
                }
                let mut branches: Vec<&Stmt> = items.iter().map(|i| &i.body).collect();
                let why = match default {
                    Some(d) => {
                        branches.push(d);
                        Branching::Case
                    }
                    None => Branching::CaseWithoutDefault,
                };
                self.merge(&branches, default.is_none(), st, s.span, why)
            }
            StmtKind::Unknown { lhs, blocking, .. } => {
                for p in pieces_in(self.env, lhs) {
                    for b in p.lo..=p.hi {
                        let bit = (p.name.clone(), b);
                        st.done.remove(&p.name, b);
                        st.x.insert(bit, s.span);
                    }
                }
                if *blocking && !self.comb {
                    self.block(lhs, s.span, st);
                }
                Ok(())
            }
            StmtKind::NonBlocking { lhs, rhs } | StmtKind::Blocking { lhs, rhs } => {
                let blocking = matches!(s.kind, StmtKind::Blocking { .. });
                self.assign(lhs, rhs, blocking, s.span, st)
            }
        }
    }

    /// An assignment of `rhs` to `lhs` at `span`, with `=` if `blocking`.
    fn assign(
        &mut self,
        lhs: &Lvalue,
        rhs: &Expr,
        blocking: bool,
        span: Span,
        st: &mut State,
    ) -> Result<()> {
        self.reads(rhs, st)?;
        let pieces = pieces_in(self.env, lhs);
        for p in &pieces {
            if let Some(v) = p.variable {
                self.reads(v.index, st)?;
            }
            if let Some(e) = p.elem {
                self.reads(e.index, st)?;
            }
        }
        if blocking && !self.comb {
            self.block(lhs, span, st);
        }
        for p in pieces {
            if p.variable.is_none() && p.elem.is_none() {
                for b in p.lo..=p.hi {
                    let bit = (p.name.clone(), b);
                    st.x.remove(&bit);
                    st.done.insert(&p.name, b);
                }
            } else if self.comb {
                self.kept(lhs, &p, st)?;
            }
            // in an always_ff a variable index keeps the other bits, x
            // included
        }
        Ok(())
    }

    /// In an `always_comb`, the piece `p` of `lhs` with a variable index
    /// assigns no bit (or element) for sure and keeps the others, so all
    /// of them must be assigned before.
    fn kept(&self, lhs: &Lvalue, p: &Piece, st: &State) -> Result<()> {
        if let Some(e) = p.elem {
            // a variable element index assigns no element for sure and
            // keeps the others
            let mut missing = Vec::new();
            for y in e.elems {
                let n = self.env[y].ty.width;
                let bits: Vec<usize> = (0..n).filter(|b| !st.done.contains(y, *b)).collect();
                if let Some(e) = st.live_x(y, &bits, n, Until::Keeping(lhs)) {
                    return Err(e);
                }
                if !bits.is_empty() {
                    missing.push(bits_text(y, &bits, n));
                }
            }
            if !missing.is_empty() {
                let some = if missing.len() > 3 {
                    format!("{}, …", missing[..3].join(", "))
                } else {
                    missing.join(", ")
                };
                return Err(Error::new(
                    lhs.span,
                    format!(
                        "`{lhs}` has a variable index, so it assigns no element for \
                         sure and keeps the others, but {some} may not be assigned \
                         yet (a latch): assign every element of `{}` before it on \
                         every path",
                        p.name
                    ),
                ));
            }
            return Ok(());
        }
        // a variable index assigns no bit for sure and keeps the other bits
        let n = self.env[&p.name].ty.width;
        let missing: Vec<usize> = (0..n).filter(|b| !st.done.contains(&p.name, *b)).collect();
        if let Some(e) = st.live_x(&p.name, &missing, n, Until::Keeping(lhs)) {
            return Err(e);
        }
        if !missing.is_empty() {
            return Err(Error::new(
                lhs.span,
                format!(
                    "`{lhs}` has a variable index, so it assigns no bit for sure \
                     and keeps the others, but {} may not be assigned yet \
                     (a latch): assign all of `{}` before it on every path, \
                     e.g. `{} = '0;`",
                    bits_text(&p.name, &missing, n),
                    p.name,
                    p.name
                ),
            ));
        }
        Ok(())
    }

    /// Adds to `st` the bits that the blocking assignment to `lhs` at
    /// `span` in an `always_ff` may assign: with a variable index, every
    /// bit of the signal (or of every element).
    fn block(&self, lhs: &Lvalue, span: Span, st: &mut State) {
        for p in pieces_in(self.env, lhs) {
            let names: Vec<&Name> = match p.elem {
                Some(e) => e.elems.iter().collect(),
                None => vec![&p.name],
            };
            for y in names {
                let (hi, lo) = if p.variable.is_some() || p.elem.is_some() {
                    (self.env[y].ty.width - 1, 0)
                } else {
                    (p.hi, p.lo)
                };
                for b in lo..=hi {
                    st.blocked.entry((y.clone(), b)).or_insert(span);
                }
            }
        }
    }

    /// Runs each branch from `st` and keeps the bits assigned in all of
    /// them, and the x of any of them; `fallthrough` adds a path that runs
    /// no branch.
    fn merge(
        &mut self,
        branches: &[&Stmt],
        fallthrough: bool,
        st: &mut State,
        span: Span,
        why: Branching,
    ) -> Result<()> {
        let mut outs = Vec::new();
        for b in branches {
            let mut out = st.clone();
            self.stmt(b, &mut out)?;
            outs.push(out);
        }
        if fallthrough {
            outs.push(st.clone());
        }
        let (first, rest) = outs.split_first().expect("a path");
        let all = rest
            .iter()
            .fold(first.done.clone(), |a, o| a.union(&o.done));
        let every = rest
            .iter()
            .fold(first.done.clone(), |a, o| a.intersection(&o.done));
        for x in all.not_within(&every) {
            self.partial.entry(x.clone()).or_insert((span, why));
        }
        let mut x = BTreeMap::new();
        let mut blocked = BTreeMap::new();
        for o in &outs {
            for (bit, at) in &o.x {
                let first = x.entry(bit.clone()).or_insert(*at);
                *first = (*first).min(*at);
            }
            for (bit, at) in &o.blocked {
                let first = blocked.entry(bit.clone()).or_insert(*at);
                *first = (*first).min(*at);
            }
        }
        *st = State {
            done: every,
            x,
            blocked,
        };
        Ok(())
    }
}

/// The reads in `e` with the bits `(hi, lo)` each may read; a variable
/// index may read any bit.
fn bit_reads<'e>(env: &Env, e: &'e Expr, out: &mut Vec<(&'e Name, usize, usize, Span)>) {
    match &e.kind {
        ExprKind::Lit { .. } | ExprKind::Const { .. } | ExprKind::Fill(_) => {}
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
        // any bit of any element
        ExprKind::Elem(el) => {
            for y in el.signals() {
                out.push((y, env[y].ty.width - 1, 0, e.span));
            }
            if let Some(i) = el.index() {
                bit_reads(env, i, out);
            }
            if let Some(i) = el.sel.as_ref().and_then(Select::index) {
                bit_reads(env, i, out);
            }
        }
        ExprKind::Unary(_, a) | ExprKind::Cast(_, a) | ExprKind::Resize(_, a) => {
            bit_reads(env, a, out)
        }
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

/// Bits of a register that no assignment of its `always_ff` block writes,
/// on any path: they keep their power-on value (the initial value, or any
/// value) forever. One per maximal range of such bits
/// ([`Checked::unwritten`](super::Checked::unwritten)).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Unwritten {
    /// The register (a signal or an element of an array).
    pub register: Name,
    pub hi: usize,
    pub lo: usize,
    /// The width of the register.
    pub width: usize,
    /// The block that drives the register.
    pub span: Span,
}

impl Unwritten {
    /// `` bits 3:1 of register `q` are never written ``, `` bit 3 of … is … ``,
    /// or `` register `q` is never written `` if no bit is written.
    pub fn never_written(&self) -> String {
        let q = &self.register;
        if self.hi + 1 - self.lo == self.width {
            format!("register `{q}` is never written")
        } else if self.hi == self.lo {
            format!("bit {} of register `{q}` is never written", self.hi)
        } else {
            format!(
                "bits {}:{} of register `{q}` are never written",
                self.hi, self.lo
            )
        }
    }

    /// The warning at the block: the bits keep their power-on value.
    pub fn warning(&self) -> Warning {
        let keep = if self.hi == self.lo || self.hi + 1 - self.lo == self.width {
            "it keeps its power-on value"
        } else {
            "they keep their power-on value"
        };
        Warning::new(self.span, format!("{}; {keep}", self.never_written()))
    }
}

/// The bits of the registers of the `always_ff` block `item` of a sized
/// module (with the signals `env`) that no assignment writes on any path,
/// by register in the order of [`Item::targets`], each register's ranges
/// from the top bit down. An assignment with a variable index counts as
/// writing every bit of its signal (every element of its array), also
/// those that no value of a narrow index reaches: so the bits found are
/// the ones Yosys gives no flip-flop. An assignment of x writes nothing
/// (it is a dead store, assigned again).
pub(super) fn unwritten(env: &Env, item: &Item) -> Vec<Unwritten> {
    let Item::AlwaysFf { body, .. } = item else {
        return Vec::new();
    };
    let mut written = Bits::default();
    let mut stack = vec![body];
    while let Some(s) = stack.pop() {
        match &s.kind {
            StmtKind::NonBlocking { lhs, .. } | StmtKind::Blocking { lhs, .. } => {
                for p in pieces_in(env, lhs) {
                    reachable(env, &p, &mut written);
                }
            }
            _ => stack.extend(s.children()),
        }
    }
    let mut out = Vec::new();
    for y in item.targets() {
        let width = env[y].ty.width;
        let missing: Vec<usize> = (0..width)
            .rev()
            .filter(|b| !written.contains(y, *b))
            .collect();
        // maximal ranges, from the top
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for b in missing {
            match ranges.last_mut() {
                Some((_, lo)) if *lo == b + 1 => *lo = b,
                _ => ranges.push((b, b)),
            }
        }
        out.extend(ranges.into_iter().map(|(hi, lo)| Unwritten {
            register: y.clone(),
            hi,
            lo,
            width,
            span: item.span(),
        }));
    }
    out
}

/// Adds to `written` the bits the piece `p` may write: with a variable
/// index, every bit of its signal (or of every element of its array).
fn reachable(env: &Env, p: &Piece, written: &mut Bits) {
    let names: Vec<&Name> = match p.elem {
        Some(e) => e.elems.iter().collect(),
        None => vec![&p.name],
    };
    for y in names {
        let (hi, lo) = match p.variable {
            None => (p.hi, p.lo),
            Some(_) => (env[y].ty.width - 1, 0),
        };
        for b in lo..=hi {
            written.insert(y, b);
        }
    }
}
