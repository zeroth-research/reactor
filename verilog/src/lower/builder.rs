//! The symbolic execution of a block into the terms of an atom.

use super::segments::{Seg, Value, as_wire, coalesce, cuts, split, whole};
use super::{Result, internal};
use crate::ast::{
    BinOp, Edge, Elem, Elems, Error, Expr, ExprKind, Index, Lvalue, LvalueKind, Name, Reset,
    Select, Span, Stmt, StmtKind, Ty, UnOp, Warning,
};
use crate::typing;
use crate::value::BitVec;
use crate::{Sort, SvOp, Term, Var, Wire};
use num_bigint::BigUint;
use num_traits::One;
use reactive::var::X;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// Adds the signals of a target to `targets`, and those it assigns as a
/// whole to `whole`.
pub(super) fn lvalue_targets<'l>(
    l: &'l Lvalue,
    whole: &mut Vec<&'l Name>,
    targets: &mut Vec<&'l Name>,
) {
    match &l.kind {
        LvalueKind::Var(y) => {
            whole.push(y);
            targets.push(y);
        }
        LvalueKind::Select(y, _) => targets.push(y),
        LvalueKind::Elem(e) => targets.extend(e.signals()),
        LvalueKind::Concat(ls) => ls.iter().for_each(|l| lvalue_targets(l, whole, targets)),
    }
}

/// Builds the terms of one atom.
pub(super) struct Builder<'a> {
    env: &'a typing::Env,
    vars: &'a BTreeMap<Name, Var>,
    pub(super) terms: Vec<Term>,
    /// For each wire a term writes, the index of that term.
    writer: HashMap<Wire, usize>,
    /// The wires some term reads.
    read: HashSet<Wire>,
    /// The signals the atom reads or writes.
    pub(super) used: BTreeSet<Name>,
    /// Whether signals are read latched (`*x`, in `always_ff`) or awaited
    /// (`X(x)`).
    latched: bool,
    /// The value of each signal the block drives so far: the value after
    /// the edge in `always_ff`; in `always_comb`, the current value, with
    /// unassigned segments for bits that may not be assigned yet.
    cur: Vec<(Name, Value)>,
    /// The index of each signal of the block in [`cur`](Self::cur) (the
    /// same in every map of the block's run).
    index: HashMap<Name, usize>,
    /// The names of the intermediate values in `always_comb`.
    pub(super) names: HashMap<Wire, Name>,
    /// The number of intermediate values named so far, per signal.
    versions: BTreeMap<Name, usize>,
    /// The wires that [`finish`](Self::finish) renamed, to the variable
    /// wire they became.
    renamed: HashMap<Wire, Wire>,
    /// The reads with a variable index that may select bits outside their
    /// signal (those bits read 0), as warnings.
    pub(super) out_of_range: Vec<Warning>,
}

impl<'a> Builder<'a> {
    pub(super) fn new(env: &'a typing::Env, vars: &'a BTreeMap<Name, Var>, latched: bool) -> Self {
        Self {
            env,
            vars,
            terms: Vec::new(),
            writer: HashMap::new(),
            read: HashSet::new(),
            used: BTreeSet::new(),
            latched,
            cur: Vec::new(),
            index: HashMap::new(),
            names: HashMap::new(),
            versions: BTreeMap::new(),
            renamed: HashMap::new(),
            out_of_range: Vec::new(),
        }
    }

    /// Starts the block driving `targets`, each with all bits of
    /// `init(y)`, or all bits unassigned.
    pub(super) fn start(&mut self, targets: &[&Name], init: impl Fn(&Name) -> Option<Wire>) {
        self.used.extend(targets.iter().map(|y| (*y).clone()));
        self.index = (targets.iter().enumerate())
            .map(|(i, y)| ((*y).clone(), i))
            .collect();
        self.cur = targets
            .iter()
            .map(|y| {
                let value = match init(y) {
                    Some(w) => whole(w),
                    None => vec![Seg {
                        width: self.env[*y].ty.width,
                        src: None,
                    }],
                };
                ((*y).clone(), value)
            })
            .collect();
    }

    /// Lowers a statement of a block, updating [`cur`](Self::cur).
    pub(super) fn stmt(&mut self, s: &Stmt) -> Result<()> {
        match &s.kind {
            StmtKind::Block(stmts) => stmts.iter().try_for_each(|s| self.stmt(s)),
            StmtKind::If(c, a, b) => {
                let (wc, tc) = self.expr(c)?;
                let wc = self.truth(wc, tc, c.span)?;
                let before = self.cur.clone();
                self.stmt(a)?;
                let then = std::mem::replace(&mut self.cur, before);
                self.stmt(b)?;
                let els = std::mem::take(&mut self.cur);
                self.cur = self.merge(wc, then, els, s.span)?;
                Ok(())
            }
            StmtKind::Case {
                sel,
                items,
                default,
                ..
            } => {
                // whether each item matches, from the values before the case
                let (ws, _) = self.expr(sel)?;
                let mut hits = Vec::new();
                for item in items {
                    let mut hit = None;
                    for l in &item.labels {
                        let (wl, _) = self.expr(l)?;
                        let eq = self.emit(SvOp::Eq, &[ws, wl], Sort(1), l.span)?;
                        hit = Some(match hit {
                            None => eq,
                            Some(h) => self.emit(SvOp::Or, &[h, eq], Sort(1), l.span)?,
                        });
                    }
                    hits.push(hit.expect("a label"));
                }
                let before = self.cur.clone();
                let mut outs = Vec::new();
                for item in items {
                    self.cur = before.clone();
                    self.stmt(&item.body)?;
                    outs.push(std::mem::take(&mut self.cur));
                }
                self.cur = before;
                if let Some(d) = default {
                    self.stmt(d)?;
                }
                let mut acc = std::mem::take(&mut self.cur);
                for (hit, out) in hits.into_iter().zip(outs).rev() {
                    acc = self.merge(hit, out, acc, s.span)?;
                }
                self.cur = acc;
                Ok(())
            }
            StmtKind::NonBlocking { lhs, rhs } | StmtKind::Blocking { lhs, rhs } => {
                let (w, _) = self.expr(rhs)?;
                self.assign(lhs, w, s.span)
            }
            // dead (typing): every path assigns the bits again
            StmtKind::Unknown { .. } => Ok(()),
        }
    }

    /// Assigns the value of `w` to the target `lhs`, the first piece taking
    /// the high bits. Variable indices are evaluated before any piece is
    /// written.
    pub(super) fn assign(&mut self, lhs: &Lvalue, w: Wire, span: Span) -> Result<()> {
        let pieces = typing::pieces_in(self.env, lhs);
        let mut indices = Vec::new();
        for p in &pieces {
            let inner = match p.variable {
                Some(v) => Some((self.expr(v.index)?.0, v.below)),
                None => None,
            };
            let elem = match p.elem {
                Some(e) => Some((self.expr(e.index)?.0, e.elems)),
                None => None,
            };
            indices.push((inner, elem));
        }
        let mut top = w.dtype().width();
        for (p, (inner, elem)) in pieces.iter().zip(indices) {
            top -= p.width();
            let Some((i, elems)) = elem else {
                self.write_piece(&p.name, p, inner, w, top, span)?;
                continue;
            };
            // every element the index can pick: its value after the
            // assignment if the index is its address, else its old value
            let iw = i.dtype().width();
            for (k, y) in elems.iter().enumerate() {
                if iw < usize::BITS as usize && k >> iw != 0 {
                    break;
                }
                let n = self.env[y].ty.width;
                let old = self.read_own(y, n - 1, 0, span)?;
                self.write_piece(y, p, inner, w, top, span)?;
                let new = self.read_own(y, n - 1, 0, span)?;
                let address = self.emit(SvOp::Lit(BitVec::wrap(iw, k)), &[], Sort(iw), span)?;
                let hit = self.emit(SvOp::Eq, &[i, address], Sort(1), span)?;
                let value = self.emit(SvOp::Ite, &[hit, new, old], Sort(n), span)?;
                self.write(y, n - 1, 0, value, 0);
            }
        }
        Ok(())
    }

    /// Writes the bits of `w` from `wlo` upwards to the bits of the signal
    /// `y` that the piece `p` gives, with the variable index `inner` (and
    /// its offset) of a select if there is one.
    fn write_piece(
        &mut self,
        y: &Name,
        p: &typing::Piece,
        inner: Option<(Wire, usize)>,
        w: Wire,
        wlo: usize,
        span: Span,
    ) -> Result<()> {
        match inner {
            None => {
                self.write(y, p.hi, p.lo, w, wlo);
                Ok(())
            }
            Some((i, below)) => self.write_at(y, i, below, p.width(), w, wlo, span),
        }
    }

    /// Makes the `width` bits of the signal `y` from bit `i - below` upwards
    /// (`i` the value of the wire `i`) hold the bits of `w` from `wlo`
    /// upwards, as far as they are inside `y`: with a mask of `width` ones
    /// shifted to bit `i`, on `y` extended by `below` zero bits at the
    /// bottom (so no shift is negative), `y := (y & ~mask) | (bits << i)`.
    /// A shift by the width or more gives 0, so bits outside `y` are not
    /// written.
    #[allow(clippy::too_many_arguments)]
    fn write_at(
        &mut self,
        y: &Name,
        i: Wire,
        below: usize,
        width: usize,
        w: Wire,
        wlo: usize,
        span: Span,
    ) -> Result<()> {
        let n = self.env[y].ty.width;
        let m = n + below;
        let old = self.read_own(y, n - 1, 0, span)?;
        let bits = self.bits(w, wlo + width - 1, wlo, span)?;
        let ones = BitVec::wrap(m, (BigUint::one() << width) - BigUint::one());
        let ones = self.emit(SvOp::Lit(ones), &[], Sort(m), span)?;
        let mask = self.emit(SvOp::Shl, &[ones, i], Sort(m), span)?;
        let bits = if width < m {
            self.emit(SvOp::ZExt(m), &[bits], Sort(m), span)?
        } else {
            bits
        };
        let bits = self.emit(SvOp::Shl, &[bits, i], Sort(m), span)?;
        let padded = if below > 0 {
            let zero = self.emit(SvOp::Lit(BitVec::zero(below)), &[], Sort(below), span)?;
            self.emit(SvOp::Concat, &[old, zero], Sort(m), span)?
        } else {
            old
        };
        let keep = self.emit(SvOp::Not, &[mask], Sort(m), span)?;
        let kept = self.emit(SvOp::And, &[padded, keep], Sort(m), span)?;
        let mut new = self.emit(SvOp::Or, &[kept, bits], Sort(m), span)?;
        if below > 0 {
            let op = SvOp::Extract {
                hi: m - 1,
                lo: below,
            };
            new = self.emit(op, &[new], Sort(n), span)?;
        }
        self.write(y, n - 1, 0, new, 0);
        Ok(())
    }

    /// Makes the bits `hi` down to `lo` of the signal `y` hold the bits of
    /// `w` from `wlo` upwards.
    fn write(&mut self, y: &Name, hi: usize, lo: usize, w: Wire, wlo: usize) {
        let i = self.index[y];
        let value = &self.cur[i].1;
        let cut = split(value, &BTreeSet::from([lo, hi + 1]));
        let mut out = Vec::new();
        let mut top: usize = cut.iter().map(|s| s.width).sum();
        for seg in cut {
            top -= seg.width;
            if top == lo {
                out.push(Seg {
                    width: hi + 1 - lo,
                    src: Some((w, wlo)),
                });
            }
            if top > hi || top < lo {
                out.push(seg);
            }
        }
        let value = coalesce(out);
        if let Some(w) = as_wire(&value) {
            self.name(y, w);
        }
        self.cur[i].1 = value;
    }

    /// The wire holding the bits `hi` down to `lo` of `value`; all of them
    /// must be assigned.
    fn build(&mut self, value: &Value, hi: usize, lo: usize, span: Span) -> Result<Wire> {
        let cut = split(value, &BTreeSet::from([lo, hi + 1]));
        let mut picked = Vec::new();
        let mut top: usize = cut.iter().map(|s| s.width).sum();
        for seg in cut {
            top -= seg.width;
            if lo <= top && top <= hi {
                picked.push(seg);
            }
        }
        let mut parts = Vec::new();
        for seg in coalesce(picked) {
            let (w, wlo) = seg
                .src
                .ok_or_else(|| internal(Some(span), "read of unassigned bits".into()))?;
            parts.push(self.bits(w, wlo + seg.width - 1, wlo, span)?);
        }
        self.concat_wires(parts, span)
    }

    /// The wire holding the bits `hi` down to `lo` of the signal `y` of the
    /// block. If they are a concatenation of several wires, it becomes
    /// their value, so that it is built only once.
    fn read_own(&mut self, y: &Name, hi: usize, lo: usize, span: Span) -> Result<Wire> {
        let i = self.index[y];
        let value = self.cur[i].1.clone();
        let w = self.build(&value, hi, lo, span)?;
        if self
            .terms
            .last()
            .is_some_and(|t| t.write() == [w] && *t.itype() == SvOp::Concat)
        {
            self.write(y, hi, lo, w, 0);
        }
        Ok(w)
    }

    /// Names `w` as the next intermediate value of `y` in `always_comb`,
    /// unless it is not local or already has a name.
    fn name(&mut self, y: &Name, w: Wire) {
        if !self.latched && !self.names.contains_key(&w) && self.writer.contains_key(&w) {
            let k = self.versions.entry(y.clone()).or_insert(0);
            *k += 1;
            self.names.insert(w, format!("{y}@{k}"));
        }
    }

    /// The map after `if (c)` with the maps `then` and `els` after the two
    /// branches: `Ite` on each run of bits that are assigned in both and
    /// differ, unassigned where either is unassigned.
    fn merge(
        &mut self,
        c: Wire,
        then: Vec<(Name, Value)>,
        els: Vec<(Name, Value)>,
        span: Span,
    ) -> Result<Vec<(Name, Value)>> {
        let mut out = Vec::new();
        for ((y, a), (_, b)) in then.into_iter().zip(els) {
            if a == b {
                out.push((y, a));
                continue;
            }
            let mut at = cuts(&a);
            at.extend(cuts(&b));
            let (a, b) = (split(&a, &at), split(&b, &at));
            // the aligned segments from the top down, with an empty pair at
            // the end that closes the last run of differing bits
            let end = Seg {
                width: 0,
                src: None,
            };
            let mut value = Vec::new();
            let mut run: Vec<(Seg, Seg)> = Vec::new();
            for (sa, sb) in a.into_iter().zip(b).chain([(end, end)]) {
                let differ = sa.src.is_some() && sb.src.is_some() && sa != sb;
                if differ {
                    run.push((sa, sb));
                    continue;
                }
                if !run.is_empty() {
                    let (ra, rb): (Value, Value) = std::mem::take(&mut run).into_iter().unzip();
                    let width: usize = ra.iter().map(|s| s.width).sum();
                    let wa = self.build(&ra, width - 1, 0, span)?;
                    let wb = self.build(&rb, width - 1, 0, span)?;
                    let w = self.emit(SvOp::Ite, &[c, wa, wb], Sort(width), span)?;
                    value.extend(whole(w));
                }
                if sa.width > 0 {
                    let src = if sb.src.is_some() { sa.src } else { None };
                    value.push(Seg {
                        width: sa.width,
                        src,
                    });
                }
            }
            let value = coalesce(value);
            if let Some(w) = as_wire(&value) {
                self.name(&y, w);
            }
            out.push((y, value));
        }
        Ok(out)
    }

    /// Makes the next value `X(y)` of every signal of the block hold its
    /// final value.
    pub(super) fn finish_all(&mut self, span: Span) -> Result<()> {
        // all final values are built before any wire is renamed, since one
        // may be built from bits of another
        let mut finals = Vec::new();
        self.index.clear();
        for (y, value) in std::mem::take(&mut self.cur) {
            let width = self.env[&y].ty.width;
            finals.push((self.build(&value, width - 1, 0, span)?, y));
        }
        for (w, y) in finals {
            self.finish(w, X(self.vars[&y]), span)?;
        }
        Ok(())
    }

    /// The wire holding whether the edge `edge` of `clock` happens in the
    /// round: the latched level and the next one are 0 and 1 (`posedge`),
    /// or 1 and 0 (`negedge`).
    pub(super) fn edge(&mut self, clock: &Name, edge: Edge, span: Span) -> Result<Wire> {
        self.used.insert(clock.clone());
        let var = self.vars[clock];
        let (before, after) = (*var, X(var));
        Ok(match edge {
            Edge::Pos => {
                let low = self.emit(SvOp::Not, &[before], Sort(1), span)?;
                self.emit(SvOp::And, &[low, after], Sort(1), span)?
            }
            Edge::Neg => {
                let low = self.emit(SvOp::Not, &[after], Sort(1), span)?;
                self.emit(SvOp::And, &[before, low], Sort(1), span)?
            }
        })
    }

    /// Makes the next value `X(q)` of every register of the block its
    /// final value if the edge `e` happens, else its latched value `*q`;
    /// with an asynchronous reset (`reset` = the wire saying whether it is
    /// active, and the reset values), a register with a reset value gets
    /// that value instead while the reset is active.
    pub(super) fn finish_guarded(
        &mut self,
        e: Wire,
        reset: Option<(Wire, &[(&Name, &Expr)])>,
        span: Span,
    ) -> Result<()> {
        let mut finals = Vec::new();
        self.index.clear();
        for (q, value) in std::mem::take(&mut self.cur) {
            let width = self.env[&q].ty.width;
            finals.push((self.build(&value, width - 1, 0, span)?, q, width));
        }
        for (w, q, width) in finals {
            let var = self.vars[&q];
            let value = reset.and_then(|(a, values)| {
                values.iter().find(|(r, _)| **r == q).map(|(_, c)| (a, *c))
            });
            match value {
                Some((a, c)) => {
                    let g = self.emit(SvOp::Ite, &[e, w, *var], Sort(width), span)?;
                    let (c, _) = self.expr(c)?;
                    self.emit_to(SvOp::Ite, &[a, c, g], X(var), span)?;
                }
                None => {
                    self.emit_to(SvOp::Ite, &[e, w, *var], X(var), span)?;
                }
            }
        }
        Ok(())
    }

    /// The wire saying whether the asynchronous reset `r` is active in
    /// this round: its next level `X(rst)`, or `NOT X(rst_n)` for an
    /// active-low reset.
    pub(super) fn reset_level(&mut self, r: &Reset) -> Wire {
        self.used.insert(r.name.clone());
        let level = X(self.vars[&r.name]);
        match r.edge {
            Edge::Pos => level,
            Edge::Neg => {
                let low = Wire::new(Sort(1));
                let term = crate::term(SvOp::Not, [low], [level]);
                self.push(term.expect("a 1-bit reset"));
                low
            }
        }
    }

    /// Makes `dest` hold the value of `w`: renames `w` to `dest` if a term
    /// of this atom writes `w` and none reads it, else adds `dest := w`.
    /// A wire already renamed (two signals with the same final value) is
    /// copied from the variable it became.
    fn finish(&mut self, w: Wire, dest: Wire, span: Span) -> Result<()> {
        if let Some(&r) = self.renamed.get(&w) {
            self.emit_to(SvOp::Id, &[r], dest, span)?;
            return Ok(());
        }
        match self.writer.get(&w).copied() {
            Some(i) if !self.read.contains(&w) => {
                let t = &self.terms[i];
                let op = t.itype().clone();
                let term = crate::term(op, [dest], t.read().iter().copied());
                self.terms[i] = term.map_err(|e| internal(Some(span), e))?;
                self.writer.remove(&w);
                self.writer.entry(dest).or_insert(i);
                self.renamed.insert(w, dest);
                self.names.remove(&w);
            }
            _ => {
                self.emit_to(SvOp::Id, &[w], dest, span)?;
            }
        }
        Ok(())
    }

    /// Adds the term `w := op(reads)` with a fresh wire `w` of `sort`.
    fn emit(&mut self, op: SvOp, reads: &[Wire], sort: Sort, span: Span) -> Result<Wire> {
        self.emit_to(op, reads, Wire::new(sort), span)
    }

    /// Adds the term `dest := op(reads)`.
    pub(super) fn emit_to(
        &mut self,
        op: SvOp,
        reads: &[Wire],
        dest: Wire,
        span: Span,
    ) -> Result<Wire> {
        let term = crate::term(op, [dest], reads.iter().copied());
        self.push(term.map_err(|e| internal(Some(span), e))?);
        Ok(dest)
    }

    /// Adds `term`, recording the wires it writes and reads.
    fn push(&mut self, term: Term) {
        for &w in term.write() {
            self.writer.entry(w).or_insert(self.terms.len());
        }
        self.read.extend(term.read().iter().copied());
        self.terms.push(term);
    }

    /// The wire holding whether `w` (of type `t`) is non-zero.
    fn truth(&mut self, w: Wire, t: Ty, span: Span) -> Result<Wire> {
        if t.width == 1 {
            Ok(w)
        } else {
            self.emit(SvOp::RedOr, &[w], Sort(1), span)
        }
    }

    /// The wire holding the value of the signal `x`, or of its bits `hi`
    /// down to `lo` if given (else all of them, `None` in the result).
    fn read(
        &mut self,
        x: &Name,
        bits: Option<(usize, usize)>,
        span: Span,
    ) -> Result<(Wire, Option<(usize, usize)>)> {
        self.used.insert(x.clone());
        let var = self.vars[x];
        let own = self.index.contains_key(x);
        if self.latched {
            return Ok((*var, bits));
        }
        if !own {
            return Ok((X(var), bits));
        }
        let (hi, lo) = bits.unwrap_or((self.env[x].ty.width - 1, 0));
        Ok((self.read_own(x, hi, lo, span)?, None))
    }

    /// Lowers `e` and returns the wire holding its value and its type.
    pub(super) fn expr(&mut self, e: &Expr) -> Result<(Wire, Ty)> {
        let span = e.span;
        match &e.kind {
            ExprKind::Lit { ty, value } | ExprKind::Const { ty, value, .. } => {
                let w = self.emit(SvOp::Lit(value.clone()), &[], Sort(ty.width), span)?;
                Ok((w, *ty))
            }
            ExprKind::Var(x) => {
                let t = typing::lookup(self.env, x, span)?.ty;
                Ok((self.read(x, None, span)?.0, t))
            }
            ExprKind::Unary(op, a) => {
                let a = self.expr(a)?;
                self.unary(*op, a, span)
            }
            ExprKind::Binary(op, a, b) => {
                let a = self.expr(a)?;
                let b = self.expr(b)?;
                self.binary(*op, a, b, span)
            }
            ExprKind::Cond(c, a, b) => {
                let (wc, tc) = self.expr(c)?;
                let wc = self.truth(wc, tc, span)?;
                let (wa, ta) = self.expr(a)?;
                let (wb, tb) = self.expr(b)?;
                let t = typing::cond(ta, tb, span)?;
                let w = self.emit(SvOp::Ite, &[wc, wa, wb], Sort(t.width), span)?;
                Ok((w, t))
            }
            ExprKind::Concat(es) => {
                let (w, ts) = self.concat(es, span)?;
                Ok((w, typing::concat(&ts)))
            }
            ExprKind::Repl(n, es) => {
                let (w, ts) = self.concat(es, span)?;
                let t = typing::repl(*n, typing::concat(&ts));
                Ok((self.concat_wires(vec![w; *n], span)?, t))
            }
            ExprKind::Elem(el) => self.elem(el, span),
            ExprKind::Select(x, sel) => self.select(x, sel, span),
            ExprKind::Resize(t, a) => {
                let (wa, _) = self.expr(a)?;
                // extended as the type it is resized to says
                Ok((self.resize(wa, *t, t.signed, span)?, *t))
            }
            ExprKind::Fill(_) => Err(Error::internal(span, "internal error: `'0`/`'1` not sized")),
            ExprKind::Cast(c, a) => {
                let (wa, ta) = self.expr(a)?;
                let t = typing::cast(*c, ta, typing::context_free(a), span)?;
                // extended as the operand's type says
                Ok((self.resize(wa, t, ta.signed, span)?, t))
            }
        }
    }

    /// The wire holding `op` applied to the operand `(wa, ta)` (wire and
    /// type), and its type.
    fn unary(&mut self, op: UnOp, (wa, ta): (Wire, Ty), span: Span) -> Result<(Wire, Ty)> {
        let t = typing::unary(op, ta);
        let s = Sort(t.width);
        let w = match op {
            UnOp::Plus => wa,
            UnOp::BitNot => self.emit(SvOp::Not, &[wa], s, span)?,
            UnOp::Neg => self.emit(SvOp::Neg, &[wa], s, span)?,
            UnOp::LogNot => {
                let b = self.truth(wa, ta, span)?;
                self.emit(SvOp::Not, &[b], s, span)?
            }
            UnOp::RedAnd => self.emit(SvOp::RedAnd, &[wa], s, span)?,
            UnOp::RedOr => self.emit(SvOp::RedOr, &[wa], s, span)?,
            UnOp::RedXor => self.emit(SvOp::RedXor, &[wa], s, span)?,
            UnOp::RedNand | UnOp::RedNor | UnOp::RedXnor => {
                let red = match op {
                    UnOp::RedNand => SvOp::RedAnd,
                    UnOp::RedNor => SvOp::RedOr,
                    _ => SvOp::RedXor,
                };
                let r = self.emit(red, &[wa], s, span)?;
                self.emit(SvOp::Not, &[r], s, span)?
            }
        };
        Ok((w, t))
    }

    /// The wire holding `op` applied to the operands `(wa, ta)` and
    /// `(wb, tb)` (wires and types), and its type.
    fn binary(
        &mut self,
        op: BinOp,
        (wa, ta): (Wire, Ty),
        (wb, tb): (Wire, Ty),
        span: Span,
    ) -> Result<(Wire, Ty)> {
        let t = typing::binary(op, ta, tb, span)?;
        let s = Sort(t.width);
        let (lt, le, div, rem) = if ta.signed {
            (SvOp::SLt, SvOp::SLe, SvOp::SDiv, SvOp::SRem)
        } else {
            (SvOp::ULt, SvOp::ULe, SvOp::UDiv, SvOp::URem)
        };
        let w = match op {
            BinOp::Pow => {
                return Err(Error::internal(
                    span,
                    "internal error: `**` is folded by sizing, it has no operator",
                ));
            }
            BinOp::BitAnd => self.emit(SvOp::And, &[wa, wb], s, span)?,
            BinOp::BitOr => self.emit(SvOp::Or, &[wa, wb], s, span)?,
            BinOp::BitXor => self.emit(SvOp::Xor, &[wa, wb], s, span)?,
            BinOp::BitXnor => {
                let x = self.emit(SvOp::Xor, &[wa, wb], s, span)?;
                self.emit(SvOp::Not, &[x], s, span)?
            }
            BinOp::Add => self.emit(SvOp::Add, &[wa, wb], s, span)?,
            BinOp::Sub => self.emit(SvOp::Sub, &[wa, wb], s, span)?,
            BinOp::Mul => self.emit(SvOp::Mul, &[wa, wb], s, span)?,
            BinOp::Div => self.emit(div, &[wa, wb], s, span)?,
            BinOp::Mod => self.emit(rem, &[wa, wb], s, span)?,
            BinOp::Eq | BinOp::CaseEq => self.emit(SvOp::Eq, &[wa, wb], s, span)?,
            BinOp::Ne | BinOp::CaseNe => self.emit(SvOp::Ne, &[wa, wb], s, span)?,
            BinOp::Lt => self.emit(lt, &[wa, wb], s, span)?,
            BinOp::Le => self.emit(le, &[wa, wb], s, span)?,
            BinOp::Gt => self.emit(lt, &[wb, wa], s, span)?,
            BinOp::Ge => self.emit(le, &[wb, wa], s, span)?,
            BinOp::LogAnd | BinOp::LogOr => {
                let x = self.truth(wa, ta, span)?;
                let y = self.truth(wb, tb, span)?;
                let op = if op == BinOp::LogAnd {
                    SvOp::And
                } else {
                    SvOp::Or
                };
                self.emit(op, &[x, y], s, span)?
            }
            BinOp::Shl | BinOp::AShl => self.emit(SvOp::Shl, &[wa, wb], s, span)?,
            BinOp::Shr => self.emit(SvOp::LShr, &[wa, wb], s, span)?,
            BinOp::AShr => {
                let op = if ta.signed { SvOp::AShr } else { SvOp::LShr };
                self.emit(op, &[wa, wb], s, span)?
            }
        };
        Ok((w, t))
    }

    /// The wire holding the element read `el` (and the bits it selects of
    /// the element), and its type. With a variable index, a tree of `Ite`s
    /// picks among all elements.
    fn elem(&mut self, el: &Elem, span: Span) -> Result<(Wire, Ty)> {
        let t = typing::elem_type(self.env, el);
        let w = match &el.index {
            Index::Const(k) => self.element(&el.elems, *k, span)?,
            Index::Expr(i) => {
                let (wi, ti) = self.expr(i)?;
                typing::element(el, Some(ti), span)?;
                // pairs of neighbours picked by bit 0, pairs of
                // these by bit 1, …
                let mut level = Vec::new();
                for k in 0..1usize << ti.width {
                    level.push(self.element(&el.elems, k, span)?);
                }
                for b in 0..ti.width {
                    let bit = self.bits(wi, b, b, span)?;
                    let mut next = Vec::new();
                    for pair in level.chunks(2) {
                        let reads = [bit, pair[1], pair[0]];
                        next.push(self.emit(SvOp::Ite, &reads, Sort(t.width), span)?);
                    }
                    level = next;
                }
                level[0]
            }
        };
        match &el.sel {
            None => Ok((w, t)),
            Some(sel) => self.pick(w, t, &el.name(), sel, span),
        }
    }

    /// The wire holding the bits that `sel` selects from the signal `x`,
    /// and their type.
    fn select(&mut self, x: &Name, sel: &Select, span: Span) -> Result<(Wire, Ty)> {
        let tx = typing::lookup(self.env, x, span)?.ty;
        let index = match sel.index() {
            Some(i) => Some(self.expr(i)?),
            None => None,
        };
        let t = typing::select(x, tx, sel, index.map(|i| i.1), span)?;
        let w = match index {
            None => {
                let (hi, lo) = sel.constant().flatten().expect("constant");
                match self.read(x, Some((hi, lo)), span)? {
                    (w, Some((hi, lo))) => self.bits(w, hi, lo, span)?,
                    (w, None) => w,
                }
            }
            Some((wi, ti)) => {
                let (wx, _) = self.read(x, None, span)?;
                self.shifted_select(wx, tx, x, sel, (wi, ti), span)?
            }
        };
        Ok((w, t))
    }

    /// The wire `w` resized to the width of `t`: its low bits, or extended
    /// with copies of its top bit if `signed`, else with zeros.
    fn resize(&mut self, w: Wire, t: Ty, signed: bool, span: Span) -> Result<Wire> {
        let s = Sort(t.width);
        match t.width.cmp(&w.dtype().width()) {
            Ordering::Equal => Ok(w),
            Ordering::Less => self.bits(w, t.width - 1, 0, span),
            Ordering::Greater if signed => self.emit(SvOp::SExt(t.width), &[w], s, span),
            Ordering::Greater => self.emit(SvOp::ZExt(t.width), &[w], s, span),
        }
    }

    /// The wire holding the bits `hi` down to `lo` of `w`: `w` itself if
    /// they are all of it.
    fn bits(&mut self, w: Wire, hi: usize, lo: usize, span: Span) -> Result<Wire> {
        if lo == 0 && hi + 1 == w.dtype().width() {
            return Ok(w);
        }
        self.emit(SvOp::Extract { hi, lo }, &[w], Sort(hi + 1 - lo), span)
    }

    /// The wire holding the concatenation of `parts` (the first the high
    /// bits), as nested binary `Concat`s `{p1, {p2, …}}`.
    fn concat_wires(&mut self, mut parts: Vec<Wire>, span: Span) -> Result<Wire> {
        let mut acc = parts.pop().expect("a part");
        while let Some(part) = parts.pop() {
            let sort = Sort(part.dtype().width() + acc.dtype().width());
            acc = self.emit(SvOp::Concat, &[part, acc], sort, span)?;
        }
        Ok(acc)
    }

    /// The wire holding the element at `address` of an array with the
    /// elements `elems`.
    fn element(&mut self, elems: &Elems, address: usize, span: Span) -> Result<Wire> {
        match elems {
            Elems::Signals(ys) => Ok(self.read(&ys[address], None, span)?.0),
            Elems::Consts(ty, values) => {
                let op = SvOp::Lit(values[address].clone());
                self.emit(op, &[], Sort(ty.width), span)
            }
            Elems::Unresolved => unreachable!("the frontend resolves every element"),
        }
    }

    /// The wire holding the bits that `sel` selects from `wx`, the value
    /// `name` of type `tx`, and their type.
    fn pick(
        &mut self,
        wx: Wire,
        tx: Ty,
        name: &str,
        sel: &Select,
        span: Span,
    ) -> Result<(Wire, Ty)> {
        let index = match sel.index() {
            Some(i) => Some(self.expr(i)?),
            None => None,
        };
        let t = typing::select(name, tx, sel, index.map(|i| i.1), span)?;
        let w = match index {
            None => {
                let (hi, lo) = sel.constant().flatten().expect("constant");
                self.bits(wx, hi, lo, span)?
            }
            Some(index) => self.shifted_select(wx, tx, name, sel, index, span)?,
        };
        Ok((w, t))
    }

    /// The wire holding the bits that `sel`, with the variable index
    /// `(wi, ti)` (wire and type), selects from `wx`, the value `name` of
    /// type `tx`: `Extract{w-1,0}(LShr(wx, i))`, so bits beyond the top
    /// read 0. For `-:` of width `w > 1`, `wx` is first padded below with
    /// `w-1` zero bits, so that bits below bit 0 read 0 too. Records a
    /// warning if some index value selects bits outside `name`.
    fn shifted_select(
        &mut self,
        wx: Wire,
        tx: Ty,
        name: &str,
        sel: &Select,
        (wi, ti): (Wire, Ty),
        span: Span,
    ) -> Result<Wire> {
        if typing::may_leave(tx, sel, ti) {
            self.out_of_range
                .push(typing::leave_warning(name, tx, sel, span));
        }
        let w = sel.width();
        let (wx, width) = match sel {
            Select::Down(_, w) if *w > 1 => {
                let zeros = BitVec::zero(w - 1);
                let z = self.emit(SvOp::Lit(zeros), &[], Sort(w - 1), span)?;
                let width = tx.width + w - 1;
                (self.emit(SvOp::Concat, &[wx, z], Sort(width), span)?, width)
            }
            // the lowest selected bit is the index
            _ => (wx, tx.width),
        };
        let shifted = self.emit(SvOp::LShr, &[wx, wi], Sort(width), span)?;
        if w == width {
            return Ok(shifted);
        }
        let op = SvOp::Extract { hi: w - 1, lo: 0 };
        self.emit(op, &[shifted], Sort(w), span)
    }

    /// The wire holding `{e1, e2, …}`, as nested binary `Concat`s
    /// `{e1, {e2, …}}`, and the types of the items.
    fn concat(&mut self, es: &[Expr], span: Span) -> Result<(Wire, Vec<Ty>)> {
        let mut parts = Vec::new();
        let mut ts = Vec::new();
        for e in es {
            let (w, t) = self.expr(e)?;
            parts.push(w);
            ts.push(t);
        }
        Ok((self.concat_wires(parts, span)?, ts))
    }
}
