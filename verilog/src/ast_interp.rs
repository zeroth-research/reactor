/*!
# The reference semantics on the AST

[`Interp::step`] runs one round, a time step: given the values of the
inputs in the round, clocks included, it returns the value of every signal
in the round. Every signal is computed on demand from its driver. The
value of a register in a round is decided by its `always_ff` block: the
block runs if the edge of its clock happens in the round, that is, its
clock was 0 in the previous round and is 1 now (`posedge`), or 1 and now
0 (`negedge`). The level of the clock now is computed first, like any
signal, so a derived clock (a register, or computed from one) has its new
value. A block runs on the values of the previous round, the values just
before the edge, and its registers take the values it assigns, the last
assignment to a register winning; a register no executed assignment
targets keeps its value. In the first round no edge happens: the
registers have their initial values: the value of their initializer
(`logic q = c;`, `initial q = c;`, assigned to the register), or else the
value given to [`Interp::new`]. In any round, including the first,
in which the asynchronous reset of a block is active (its level now is 1,
or 0 for `negedge rst_n`), the registers the reset branch assigns have
their reset values instead, whether or not the edge happens; the reset
branch still runs at an edge like the rest of the body, reading the reset
from the previous round, so an edge in the round where the reset is
released keeps the reset values.

A signal driven by an `always_comb` block is computed by running the whole
block when one of its signals is first needed, with blocking assignments:
a read of a signal of the block sees its latest assignment (typing makes
sure there is one). It gives meaning to the source operators directly
on numbers (unsigned or two's complement as the types say), with the
widths and signedness of IEEE 1800 worked out on the source expression in
the standard's two phases: the self-determined type of every expression
bottom up, then the type of the context passed down to the
context-determined operands, where values are extended (§11.6 to §11.8;
an assignment evaluates its right-hand side at the larger of both widths
and truncates). This is independent of [`sizing`](crate::sizing) (which
rewrites the expression by bidirectional rules), of [`SvOp`] and of the
rewrites in lowering, so that tests can compare it with the translated
module. Each module is checked with [`typing::checked`] like for the
translation (so the interpreter runs only modules the translation
accepts), but the interpreter runs the module as written
([`Checked::source`]) and never looks at its sized form. It also finds
the bits an assignment target writes, the signals an item drives and the
reset branch of a block by itself, not with the helpers lowering uses
(`typing::pieces`, `Item::targets`, `Stmt::reset_values`), so that a
mistake in one of those makes the two disagree.

The element `m[i]` of an array is the signal `m[k]` with `k` the value of
`i` (typing makes sure it is an address when read); an assignment to an
element at an index that is no address writes nothing.

The outputs of blackboxes (see [`crate::lower`]) are inputs of the design,
given to [`Interp::step`] like the inputs of the top module; nothing reads
the inputs of a blackbox.

A hierarchy runs as one design: every signal is known by its name (see
[`Inst::naming`]), so a port connected to a signal is that signal, and an
input connected to another expression is computed from it on demand.
Register initial values are given by path (`u.q`), for the registers
without an initializer.

Computing a signal on demand recurses into the signals it reads, so the
stack grows with the longest chain of combinational signals (one `assign`
reading the previous one, about 1 KB of stack per signal in a release
build) and with the depth of expressions: a chain of a few thousand
signals needs more than the 2 MB of a spawned thread or the 8 MB of the
main thread; run the interpreter on a thread with a larger stack for
such designs (as `v2rm` does).

## The timing model

The rounds, edges, the values blocks read at an edge and the asynchronous
resets above are those of the translation, written down in [`crate::lower`]
("Time and clocks", "Asynchronous resets") together with their timing
assumptions; this interpreter implements the same model directly, so the
model itself is checked only against iverilog (below).

## Which check covers which pass (the oracle map)

What the interpreter shares with the translation, a comparison of the two
cannot see: the parser and its re-association of operators
([`crate::frontend`]), constant evaluation ([`crate::constant`]),
elaboration and naming ([`crate::elab`]), acceptance by
[`typing::check`] (and [`typing::checked`]), the connection helpers
`Instance::conns`/`Conn::shared`, and the timing model. Independent of each
other are sizing against this interpreter's own two-phase width rules,
[`SvOp`] and [`crate::value`] against its arithmetic, symbolic against
direct execution, and (since it splits them itself) assignment targets,
driven signals and reset branches.

| check (where) | compares | covers |
|---|---|---|
| golden (`tests/examples.rs` `golden`, `sized_source`) | the printed module and the `--sized` source of every example against files | changes of [`crate::lower`], [`crate::print`], [`crate::sizing`]: regressions only, not correctness |
| RM-vs-AST (`rm_matches_ast` in `tests/examples.rs`, `tests/scaling.rs`) | [`crate::rm_sim`] on the translated module against this interpreter on the source, random registers and input traces | [`crate::sizing`], [`crate::lower`], [`crate::value`]/[`SvOp`], [`crate::rm_sim`]; not the shared passes above |
| SMT-vs-RM (`smt_matches_rm` in `tests/examples.rs`) | the rounds as SMT-LIB ([`crate::smt`]) in z3 against [`crate::rm_sim`] | [`crate::smt`] |
| AST-vs-iverilog (`ast_matches_iverilog`, and the sized source simulated like the original in `sized_source`) | this interpreter against iverilog round by round on random traces (x tolerated only where an out-of-range read reaches) | the frontend, elaboration, acceptance as far as accepted designs go, this interpreter and the timing model, on `examples/` only, minus designs where a simulator schedules derived clocks otherwise ([`crate::iverilog::simulator_differs`]); sizing on its own |
| cycles-vs-iverilog (`rounds_match_cycles`) | iverilog pulsing one clock per cycle against rounds with a toggling clock | that the round semantics is the usual cycle model, for designs with one rising-edge clock |
| the miter ([`crate::miter`]; `tests/yosys.rs`, `v2rm --yosys`, the benchmarks and the case study) | our translation against Yosys's in SMT-LIB, for every state and input of the init round and one step | the whole pipeline from the parser to [`crate::smt`], against Yosys's independent frontend, on every design it runs on; a difference is replayed here and in iverilog ([`crate::miter::second_opinion`]) |
| mutants (`tests/planted.rs`, ignored by default) | every mutant of every example against the original, by the miter both ways and by random simulation | the power of the miter and of the replay: that they notice changed designs and blame the right side |
| accepted snippets (`common::accepted` in `tests/frontend.rs`, `hierarchy.rs`, `blackbox.rs`) | every design those tests accept, through RM-vs-AST and the miter | the same passes as those two checks, on the constructs the frontend tests name |
| scaling (`tests/scaling.rs`) | RM-vs-AST on generated large designs, timed per phase | performance of every pass, and RM-vs-AST at sizes the examples do not reach |

So the frontend, elaboration and [`typing::check`] are covered only by
iverilog (on `examples/`, and on each counterexample of `v2rm --yosys`)
and by Yosys (through the miter): RM-vs-AST runs them once for both sides.
Whether a rejection is right (a design the subset should accept) is
covered by no oracle, only by the negative tests of each rule.

[`SvOp`]: crate::SvOp
*/

use crate::ast::{
    BinOp, Cast, Dir, Edge, Elems, Error, ErrorKind, Expr, ExprKind, Index, Item, Lvalue,
    LvalueKind, Name, Select, Stmt, StmtKind, Ty, UnOp,
};
use crate::elab::{self, Inst, Path};
use crate::typing::{self, Checked};
use crate::value::BitVec;
use num_bigint::{BigInt, BigUint};
use num_traits::{One, ToPrimitive, Zero};
use std::collections::{BTreeMap, BTreeSet};

/// An `always_ff` block: its instance, the item, the name and edge of its
/// clock, and the name and edge of its asynchronous reset.
type Block<'m> = (usize, &'m Item, Name, Edge, Option<(Name, Edge)>);

/// A value for each of some signals.
pub type Values = BTreeMap<Name, BitVec>;

/// The values of the signals computed so far (`now`), over a map that is
/// only read (`below`, if any): the body of an `always_ff` block runs on
/// the values of the previous round, which it reads without copying them.
#[derive(Default)]
struct Overlay<'v> {
    below: Option<&'v Values>,
    now: Values,
}

impl Overlay<'_> {
    fn get(&self, x: &str) -> Option<&BitVec> {
        self.now.get(x).or_else(|| self.below?.get(x))
    }

    fn insert(&mut self, x: Name, v: BitVec) {
        self.now.insert(x, v);
    }
}

impl std::ops::Index<&str> for Overlay<'_> {
    type Output = BitVec;

    fn index(&self, x: &str) -> &BitVec {
        self.get(x).unwrap_or_else(|| panic!("no value for `{x}`"))
    }
}

pub struct Interp<'m> {
    /// One context per instance, the top module's first.
    ctxs: Vec<Ctx<'m>>,
    /// The driver of each signal driven by an `assign`, an `always_comb`
    /// or an instance connection, with the instance it is in.
    driver: BTreeMap<Name, (usize, Driver<'m>)>,
    /// The `always_ff` blocks.
    blocks: Vec<Block<'m>>,
    /// Every signal, by name, with its type.
    signals: BTreeMap<Name, Ty>,
    /// The inputs of the design (see [`Interp::inputs`]).
    inputs: Vec<(Name, Ty)>,
    /// The registers: path, name and type.
    registers: Vec<(Name, Name, Ty)>,
    /// The registers with an initializer: name, instance and value.
    initialized: Vec<(Name, usize, &'m Expr)>,
    /// The initial value of every register, by name.
    init: Values,
    /// The values of the previous round; none before the first round.
    prev: Option<Values>,
}

/// The signals of an instance.
struct Ctx<'m> {
    /// The checked module; the interpreter runs the module as written
    /// ([`Checked::source`]), never the sized one.
    checked: Checked<'m>,
    /// The name of each signal of the instance (see [`Inst::naming`]).
    names: BTreeMap<Name, Name>,
}

#[derive(Clone, Copy)]
enum Driver<'m> {
    Item(&'m Item),
    /// An input of an instance connected to this expression of the
    /// instantiating module.
    Conn(&'m Expr),
    /// A register, assigned by the `always_ff` block with this index in
    /// [`Interp::blocks`].
    Ff(usize),
}

impl<'m> Interp<'m> {
    /// Starts with `init` giving a value for every register without an
    /// initializer, by path (and nothing else).
    pub fn new(inst: &'m Inst, init: &Values) -> Result<Self, Error> {
        let naming: BTreeMap<Name, Name> = inst.naming().into_iter().collect();
        let mut it = Interp {
            ctxs: Vec::new(),
            driver: BTreeMap::new(),
            blocks: Vec::new(),
            signals: BTreeMap::new(),
            inputs: Vec::new(),
            registers: Vec::new(),
            initialized: Vec::new(),
            init: Values::new(),
            prev: None,
        };
        for p in &inst.module.ports {
            if p.dir == Dir::Input {
                it.inputs.push((p.decl.name.clone(), p.decl.ty));
            }
        }
        // the ports of a blackbox are no signals, and it has no instances
        for (path, i) in inst.iter().filter(|(_, i)| !i.module.blackbox) {
            it.add(i, &path, &naming)?;
        }
        let err = |msg: String| Error::unlocated(ErrorKind::User, msg);
        for (name, c, e) in &it.initialized {
            let width = it.signals[name].width;
            let v = it
                .assigned(
                    *c,
                    e,
                    width,
                    &Values::new(),
                    &mut Overlay::default(),
                    &mut BTreeSet::new(),
                )
                .map_err(|msg| Error::new(e.span, msg))?;
            it.init.insert(name.clone(), v);
        }
        for (path, name, ty) in &it.registers {
            if it.initialized.iter().any(|(y, ..)| y == name) {
                if init.contains_key(path) {
                    return Err(err(format!("register `{path}` has an initializer")));
                }
                continue;
            }
            let v = init
                .get(path)
                .ok_or_else(|| err(format!("no initial value for register `{path}`")))?;
            if v.width() != ty.width {
                return Err(err(format!("register `{path}` given {} bits", v.width())));
            }
            it.init.insert(name.clone(), v.clone());
        }
        if let Some(x) = init
            .keys()
            .find(|x| !it.registers.iter().any(|r| r.0 == **x))
        {
            return Err(err(format!("`{x}` is not a register")));
        }
        Ok(it)
    }

    /// Adds the instance at path `prefix` (not those below it).
    fn add(
        &mut self,
        inst: &'m Inst,
        prefix: &Path,
        naming: &BTreeMap<Name, Name>,
    ) -> Result<(), Error> {
        let checked = typing::checked(inst).map_err(|e| elab::context(e, prefix))?;
        let (m, env) = (checked.source(), checked.env());
        let c = self.ctxs.len();
        let names: BTreeMap<Name, Name> = env
            .keys()
            .map(|x| (x.clone(), naming[&prefix.signal(x)].clone()))
            .collect();
        for (x, sig) in env {
            self.signals.insert(names[x].clone(), sig.ty);
        }
        for i in &m.inits {
            self.initialized
                .push((names[&i.target].clone(), c, &i.value));
        }
        for item in &m.items {
            match item {
                Item::AlwaysFf {
                    clock, edge, reset, ..
                } => {
                    let b = self.blocks.len();
                    let reset = reset.as_ref().map(|r| (names[&r.name].clone(), r.edge));
                    self.blocks
                        .push((c, item, names[clock].clone(), *edge, reset));
                    for q in targets(item) {
                        let path = prefix.signal(q);
                        self.registers.push((path, names[q].clone(), env[q].ty));
                        self.driver.insert(names[q].clone(), (c, Driver::Ff(b)));
                    }
                }
                Item::Assign { .. } | Item::AlwaysComb { .. } => {
                    for y in targets(item) {
                        self.driver
                            .insert(names[y].clone(), (c, Driver::Item(item)));
                    }
                }
                // the connected outputs of a blackbox are inputs of the
                // design; nothing reads its inputs
                Item::Instance(i)
                    if i.conns.iter().any(|c| c.formal.is_some_and(|f| f.blackbox)) =>
                {
                    for (conn, f) in i.conns(Dir::Output) {
                        if let Some(y) = conn.shared() {
                            self.inputs.push((names[y].clone(), f.ty));
                        }
                    }
                }
                Item::Instance(i) => {
                    for (conn, f) in i.conns(Dir::Input) {
                        if let Some(e) = &conn.actual
                            && conn.shared().is_none()
                        {
                            let name = naming[&prefix.child(&i.name).signal(&conn.port)].clone();
                            self.signals.insert(name.clone(), f.ty);
                            self.driver.insert(name, (c, Driver::Conn(e)));
                        }
                    }
                }
            }
        }
        self.ctxs.push(Ctx { checked, names });
        Ok(())
    }

    /// The registers, by path, with their types.
    pub fn registers(&self) -> Vec<(&Name, Ty)> {
        self.registers.iter().map(|(p, _, t)| (p, *t)).collect()
    }

    /// The inputs of the design, clocks included, with their types: the
    /// inputs of the top module, then the connected outputs of blackboxes
    /// (free in every round, like inputs).
    pub fn inputs(&self) -> &[(Name, Ty)] {
        &self.inputs
    }

    /// Runs one round, with `inputs` giving a value for every input,
    /// clocks included. Returns the value of every signal in this round,
    /// by name.
    pub fn step(&mut self, inputs: &Values) -> Result<Values, String> {
        let mut values = Overlay::default();
        for (name, ty) in &self.inputs {
            let v = inputs
                .get(name)
                .ok_or_else(|| format!("no value for input `{name}`"))?;
            if v.width() != ty.width {
                return Err(format!("input `{name}` given {} bits", v.width()));
            }
            values.insert(name.clone(), v.clone());
        }
        let mut active = BTreeSet::new();
        for name in self.signals.keys() {
            self.signal(name, &mut values, &mut active)?;
        }
        self.prev = Some(values.now.clone());
        Ok(values.now)
    }

    /// Executes a statement of an `always_ff` block (`comb` false) or an
    /// `always_comb` block (`comb` true), adding its assignments to `out`.
    /// In `always_comb`, reads of the signals in `out` see their value
    /// there.
    fn exec(
        &self,
        c: usize,
        s: &Stmt,
        comb: bool,
        values: &mut Overlay,
        active: &mut BTreeSet<Name>,
        out: &mut Values,
    ) -> Result<(), String> {
        // the values reads see before the round's: in `always_comb`, those
        // the block assigned so far
        let none = Values::new();
        fn scope<'a>(comb: bool, out: &'a Values, none: &'a Values) -> &'a Values {
            if comb { out } else { none }
        }
        match &s.kind {
            StmtKind::Block(stmts) => stmts
                .iter()
                .try_for_each(|s| self.exec(c, s, comb, values, active, out)),
            StmtKind::If(cond, a, b) => {
                let v = self.own(c, cond, scope(comb, out, &none), values, active)?;
                let branch = if v.is_zero() { b } else { a };
                self.exec(c, branch, comb, values, active, out)
            }
            StmtKind::Case {
                sel,
                items,
                default,
                ..
            } => {
                let scope = scope(comb, out, &none);
                // the selector and the labels are evaluated at one type
                let mut t = self.det(c, sel);
                for l in items.iter().flat_map(|i| &i.labels) {
                    let tl = self.det(c, l);
                    t = Ty {
                        width: t.width.max(tl.width),
                        signed: t.signed && tl.signed,
                    };
                }
                let v = self.value(c, sel, t, scope, values, active)?;
                for item in items {
                    for l in &item.labels {
                        if self.value(c, l, t, scope, values, active)? == v {
                            return self.exec(c, &item.body, comb, values, active, out);
                        }
                    }
                }
                match default {
                    Some(d) => self.exec(c, d, comb, values, active, out),
                    None => Ok(()),
                }
            }
            // a dead store of x (typing): every path assigns the bits again
            StmtKind::Unknown { .. } => Ok(()),
            StmtKind::NonBlocking { lhs, rhs } | StmtKind::Blocking { lhs, rhs } => {
                let scope = scope(comb, out, &none);
                // the places, with their indices evaluated before anything
                // is written
                let mut places = Vec::new();
                self.places(c, lhs, scope, values, active, &mut places)?;
                let width = places.iter().map(|p| p.width).sum();
                let v = self.assigned(c, rhs, width, scope, values, active)?;
                let env = self.ctxs[c].checked.env();
                // the places take the bits of `v` from the top down
                let mut top = width;
                for p in places {
                    let bits = slice(&v, top - 1, top - p.width);
                    top -= p.width;
                    let Some(name) = &p.target else {
                        continue;
                    };
                    // the other bits of a register keep their value; those
                    // of an `always_comb` signal are assigned before they
                    // are read (typing), so any value does
                    let w = env[name].ty.width;
                    let new = if p.whole(w) {
                        bits
                    } else {
                        let old = match out.get(name) {
                            Some(old) => old.clone(),
                            None if comb => BitVec::zero(w),
                            None => values[&self.ctxs[c].names[name]].clone(),
                        };
                        put(&old, &p.lo, &bits)
                    };
                    out.insert(name.clone(), new);
                }
                Ok(())
            }
        }
    }

    /// Adds the places the assignment target `l` writes to `out`, the one
    /// taking the high bits of the value first, with the indices in it
    /// evaluated (reads see `scope` first, as in [`Self::value`]).
    fn places(
        &self,
        c: usize,
        l: &Lvalue,
        scope: &Values,
        values: &mut Overlay,
        active: &mut BTreeSet<Name>,
        out: &mut Vec<Place>,
    ) -> Result<(), String> {
        let env = self.ctxs[c].checked.env();
        let whole = |y: &Name| (BigInt::zero(), env[y].ty.width);
        let (target, (lo, width)) = match &l.kind {
            LvalueKind::Var(y) => (Some(y.clone()), whole(y)),
            LvalueKind::Select(y, sel) => {
                (Some(y.clone()), self.range(c, sel, scope, values, active)?)
            }
            LvalueKind::Elem(e) => {
                let Elems::Signals(ys) = &e.elems else {
                    unreachable!("a ROM is not written")
                };
                let bits = match &e.sel {
                    Some(sel) => self.range(c, sel, scope, values, active)?,
                    None => whole(&ys[0]),
                };
                // the element the index picks, if it is an address
                let k = self.index(c, &e.index, scope, values, active)?;
                (k.to_usize().and_then(|k| ys.get(k)).cloned(), bits)
            }
            LvalueKind::Concat(ls) => {
                for l in ls {
                    self.places(c, l, scope, values, active, out)?;
                }
                return Ok(());
            }
        };
        out.push(Place { target, lo, width });
        Ok(())
    }

    /// The bits a select picks: its lowest bit (below 0 or above the top
    /// bit of the signal for some values of a variable index) and its
    /// width.
    fn range(
        &self,
        c: usize,
        sel: &Select,
        scope: &Values,
        values: &mut Overlay,
        active: &mut BTreeSet<Name>,
    ) -> Result<(BigInt, usize), String> {
        Ok(match sel {
            Select::Bit(i) => (self.index(c, i, scope, values, active)?, 1),
            Select::Part { hi, lo } => (BigInt::from(*lo), (hi + 1).saturating_sub(*lo)),
            Select::Up(i, w) => (self.index(c, i, scope, values, active)?, *w),
            Select::Down(i, w) => (self.index(c, i, scope, values, active)? - (w - 1), *w),
        })
    }

    /// The value of an index, unsigned.
    fn index(
        &self,
        c: usize,
        i: &Index,
        scope: &Values,
        values: &mut Overlay,
        active: &mut BTreeSet<Name>,
    ) -> Result<BigInt, String> {
        Ok(match i {
            Index::Const(k) => BigInt::from(*k),
            Index::Expr(e) => {
                BigInt::from(self.own(c, e, scope, values, active)?.unsigned().clone())
            }
        })
    }

    /// The value of a signal (by name), computed on demand from its
    /// driver.
    fn signal(
        &self,
        name: &str,
        values: &mut Overlay,
        active: &mut BTreeSet<Name>,
    ) -> Result<BitVec, String> {
        if let Some(v) = values.get(name) {
            return Ok(v.clone());
        }
        if !active.insert(name.to_string()) {
            return Err(format!("combinational loop through `{name}`"));
        }
        let (c, driver) = self
            .driver
            .get(name)
            .copied()
            .ok_or_else(|| format!("`{name}` has no driver"))?;
        let ctx = &self.ctxs[c];
        match driver {
            Driver::Conn(e) => {
                let width = self.signals[name].width;
                let v = self.assigned(c, e, width, &Values::new(), values, active)?;
                values.insert(name.to_string(), v);
            }
            Driver::Item(Item::Assign { lhs, rhs, .. }) => {
                let mut places = Vec::new();
                self.places(c, lhs, &Values::new(), values, active, &mut places)?;
                let width = places.iter().map(|p| p.width).sum();
                let v = self.assigned(c, rhs, width, &Values::new(), values, active)?;
                // an `assign` drives whole signals (typing), by constant
                // selects
                let mut top = width;
                for p in places {
                    let bits = slice(&v, top - 1, top - p.width);
                    top -= p.width;
                    let y = p.target.expect("a signal");
                    let name = &ctx.names[&y];
                    let w = ctx.checked.env()[&y].ty.width;
                    let old = values.get(name).cloned().unwrap_or_else(|| BitVec::zero(w));
                    values.insert(name.clone(), put(&old, &p.lo, &bits));
                }
            }
            Driver::Ff(b) => {
                let (c, item, clock, edge, reset) = &self.blocks[b];
                // an active asynchronous reset sets the registers with a
                // reset value, in any round
                let reset_active = match reset {
                    Some((r, edge)) => {
                        let level = !self.signal(r, values, active)?.is_zero();
                        level == (*edge == Edge::Pos)
                    }
                    None => false,
                };
                let mut out = Values::new();
                if reset_active {
                    // the branch assigns constants to whole registers
                    // (typing), so it reads nothing
                    let branch = reset_branch(item.body().unwrap());
                    self.exec(*c, branch, false, values, active, &mut out)?;
                }
                let Some(prev) = self.prev.as_ref() else {
                    // the first round: no edge
                    for q in targets(item) {
                        let y = &self.ctxs[*c].names[q];
                        let v = out.remove(q).unwrap_or_else(|| self.init[y].clone());
                        values.insert(y.clone(), v);
                    }
                    active.remove(name);
                    return Ok(values[name].clone());
                };
                let before = prev[clock].is_zero();
                let after = self.signal(clock, values, active)?.is_zero();
                let fires = match edge {
                    Edge::Pos => before && !after,
                    Edge::Neg => !before && after,
                };
                if fires {
                    // on the values just before the edge; the reset
                    // values win
                    let mut old = Overlay {
                        below: Some(prev),
                        now: Values::new(),
                    };
                    let body = item.body().unwrap();
                    let mut run = Values::new();
                    self.exec(*c, body, false, &mut old, &mut BTreeSet::new(), &mut run)?;
                    for (q, v) in run {
                        out.entry(q).or_insert(v);
                    }
                }
                for q in targets(item) {
                    let y = &self.ctxs[*c].names[q];
                    let v = out.remove(q).unwrap_or_else(|| prev[y].clone());
                    values.insert(y.clone(), v);
                }
            }
            Driver::Item(item) => {
                let targets = targets(item);
                for y in &targets {
                    active.insert(ctx.names[*y].clone());
                }
                let mut out = Values::new();
                self.exec(c, item.body().unwrap(), true, values, active, &mut out)?;
                for y in targets {
                    let v = out
                        .remove(y)
                        .ok_or_else(|| format!("`{y}` not assigned (a latch)"))?;
                    active.remove(&ctx.names[y]);
                    values.insert(ctx.names[y].clone(), v);
                }
            }
        }
        active.remove(name);
        Ok(values[name].clone())
    }

    /// The self-determined type of `e`: IEEE 1800 Table 11-21 for the
    /// width, §11.8.1 for the signedness.
    fn det(&self, c: usize, e: &Expr) -> Ty {
        let env = self.ctxs[c].checked.env();
        let both = |a: Ty, b: Ty| Ty {
            width: a.width.max(b.width),
            signed: a.signed && b.signed,
        };
        match &e.kind {
            ExprKind::Lit { ty, .. } | ExprKind::Const { ty, .. } | ExprKind::Resize(ty, _) => *ty,
            ExprKind::Var(x) => env[x].ty,
            ExprKind::Fill(_) => Ty::BIT,
            ExprKind::Unary(UnOp::BitNot | UnOp::Neg | UnOp::Plus, a) => self.det(c, a),
            ExprKind::Unary(..) => Ty::BIT,
            ExprKind::Binary(op, a, b) => match op {
                BinOp::Shl | BinOp::Shr | BinOp::AShl | BinOp::AShr | BinOp::Pow => self.det(c, a),
                op if compare(*op) || matches!(op, BinOp::LogAnd | BinOp::LogOr) => Ty::BIT,
                _ => both(self.det(c, a), self.det(c, b)),
            },
            ExprKind::Cond(_, a, b) => both(self.det(c, a), self.det(c, b)),
            ExprKind::Concat(es) => Ty::unsigned(es.iter().map(|e| self.det(c, e).width).sum()),
            ExprKind::Repl(n, es) => {
                Ty::unsigned(n * es.iter().map(|e| self.det(c, e).width).sum::<usize>())
            }
            ExprKind::Select(_, sel) => Ty::unsigned(sel.width()),
            ExprKind::Elem(el) => match (&el.sel, &el.elems) {
                (Some(sel), _) => Ty::unsigned(sel.width()),
                (None, Elems::Signals(ys)) => env[&ys[0]].ty,
                (None, Elems::Consts(ty, _)) => *ty,
                (None, Elems::Unresolved) => unreachable!("the frontend resolves every element"),
            },
            ExprKind::Cast(Cast::Size(n), a) => Ty {
                width: *n,
                signed: self.det(c, a).signed,
            },
            ExprKind::Cast(Cast::Signed, a) => Ty {
                signed: true,
                ..self.det(c, a)
            },
            ExprKind::Cast(Cast::Unsigned, a) => Ty {
                signed: false,
                ..self.det(c, a)
            },
        }
    }

    /// The value of `e` assigned to a target of `width` bits: `e` is
    /// evaluated at the larger of both widths with its own signedness
    /// (IEEE 1800 §10.7), then truncated.
    fn assigned(
        &self,
        c: usize,
        e: &Expr,
        width: usize,
        scope: &Values,
        values: &mut Overlay,
        active: &mut BTreeSet<Name>,
    ) -> Result<BitVec, String> {
        let own = self.det(c, e);
        let t = Ty {
            width: own.width.max(width),
            ..own
        };
        let v = self.value(c, e, t, scope, values, active)?;
        Ok(slice(&v, width - 1, 0))
    }

    /// The value of `e` at its own type.
    fn own(
        &self,
        c: usize,
        e: &Expr,
        scope: &Values,
        values: &mut Overlay,
        active: &mut BTreeSet<Name>,
    ) -> Result<BitVec, String> {
        let t = self.det(c, e);
        self.value(c, e, t, scope, values, active)
    }

    /// The value of `e` in a context of type `t` (at least as wide as `e`),
    /// where the signals in `scope` have their value there and all others
    /// their value in the round. This is the propagation of IEEE 1800
    /// §11.8.2: `t` is passed down to the context-determined operands, and
    /// every other expression is evaluated at its own type and then
    /// extended to `t`, by sign if `t` is signed.
    fn value(
        &self,
        c: usize,
        e: &Expr,
        t: Ty,
        scope: &Values,
        values: &mut Overlay,
        active: &mut BTreeSet<Name>,
    ) -> Result<BitVec, String> {
        let ctx = &self.ctxs[c];
        let sig = |x: &Name, values: &mut Overlay, active: &mut BTreeSet<Name>| match scope.get(x) {
            Some(v) => Ok(v.clone()),
            None => self.signal(&ctx.names[x], values, active),
        };
        let v = match &e.kind {
            ExprKind::Lit { value, .. } | ExprKind::Const { value, .. } => value.clone(),
            ExprKind::Var(x) => sig(x, values, active)?,
            ExprKind::Fill(b) => {
                return Ok(if *b {
                    BitVec::wrap(t.width, ones(t.width))
                } else {
                    BitVec::zero(t.width)
                });
            }
            ExprKind::Unary(op @ (UnOp::BitNot | UnOp::Neg | UnOp::Plus), a) => {
                return Ok(unary(*op, &self.value(c, a, t, scope, values, active)?));
            }
            ExprKind::Unary(op, a) => unary(*op, &self.own(c, a, scope, values, active)?),
            ExprKind::Binary(op, a, b) if compare(*op) => {
                let (ta, tb) = (self.det(c, a), self.det(c, b));
                let both = Ty {
                    width: ta.width.max(tb.width),
                    signed: ta.signed && tb.signed,
                };
                let x = self.value(c, a, both, scope, values, active)?;
                let y = self.value(c, b, both, scope, values, active)?;
                binary(*op, &x, &y, both.signed)?
            }
            ExprKind::Binary(op @ (BinOp::LogAnd | BinOp::LogOr), a, b) => {
                let x = self.own(c, a, scope, values, active)?;
                let y = self.own(c, b, scope, values, active)?;
                binary(*op, &x, &y, false)?
            }
            // the base context-determined, the exponent self-determined
            ExprKind::Binary(BinOp::Pow, a, b) => {
                let x = self.value(c, a, t, scope, values, active)?;
                let y = self.own(c, b, scope, values, active)?;
                return power(&x, t.signed, &y, self.det(c, b).signed);
            }
            ExprKind::Binary(op, a, b) => {
                let x = self.value(c, a, t, scope, values, active)?;
                let shift = matches!(op, BinOp::Shl | BinOp::Shr | BinOp::AShl | BinOp::AShr);
                let y = if shift {
                    self.own(c, b, scope, values, active)?
                } else {
                    self.value(c, b, t, scope, values, active)?
                };
                return binary(*op, &x, &y, t.signed);
            }
            ExprKind::Cond(cond, a, b) => {
                let v = self.own(c, cond, scope, values, active)?;
                let arm = if v.is_zero() { b } else { a };
                return self.value(c, arm, t, scope, values, active);
            }
            ExprKind::Concat(es) => self.exprs(c, es, scope, values, active)?,
            ExprKind::Repl(n, es) => {
                let v = self.exprs(c, es, scope, values, active)?;
                let mut acc = v.clone();
                for _ in 1..*n {
                    acc = concat(&acc, &v);
                }
                acc
            }
            ExprKind::Elem(el) => {
                // typing guarantees the index is an address
                let k = match &el.index {
                    Index::Const(k) => *k,
                    Index::Expr(i) => {
                        let i = self.own(c, i, scope, values, active)?;
                        i.unsigned().to_usize().expect("an address")
                    }
                };
                let v = match &el.elems {
                    Elems::Signals(ys) => sig(&ys[k], values, active)?,
                    Elems::Consts(_, vs) => vs[k].clone(),
                    Elems::Unresolved => unreachable!("the frontend resolves every element"),
                };
                match &el.sel {
                    None => v,
                    Some(sel) => self.bits(c, &v, sel, scope, values, active)?,
                }
            }
            ExprKind::Select(x, sel) => {
                let v = sig(x, values, active)?;
                self.bits(c, &v, sel, scope, values, active)?
            }
            // an assignment to an `n`-bit value
            ExprKind::Cast(Cast::Size(n), a) => self.assigned(c, a, *n, scope, values, active)?,
            ExprKind::Cast(_, a) => self.own(c, a, scope, values, active)?,
            ExprKind::Resize(to, a) => {
                let v = self.own(c, a, scope, values, active)?;
                extend(&slice(&v, to.width.min(v.width()) - 1, 0), *to)
            }
        };
        Ok(extend(&v, t))
    }

    /// The bits of the value `v` that `sel` selects.
    fn bits(
        &self,
        c: usize,
        v: &BitVec,
        sel: &Select,
        scope: &Values,
        values: &mut Overlay,
        active: &mut BTreeSet<Name>,
    ) -> Result<BitVec, String> {
        let (lo, w) = self.range(c, sel, scope, values, active)?;
        if let Some(lo) = lo.to_usize().filter(|lo| lo + w <= v.width()) {
            return Ok(slice(v, lo + w - 1, lo));
        }
        // bit by bit; bits outside `v` read 0
        let mut out = BigUint::from(0u8);
        for k in 0..w {
            let at = &lo + k;
            let bit = usize::try_from(&at)
                .ok()
                .filter(|at| *at < v.width())
                .is_some_and(|at| v.unsigned().bit(at as u64));
            if bit {
                out.set_bit(k as u64, true);
            }
        }
        Ok(BitVec::wrap(w, out))
    }

    /// The values of `es`, each at its own type, concatenated.
    fn exprs(
        &self,
        c: usize,
        es: &[Expr],
        scope: &Values,
        values: &mut Overlay,
        active: &mut BTreeSet<Name>,
    ) -> Result<BitVec, String> {
        let mut acc: Option<BitVec> = None;
        for e in es {
            let v = self.own(c, e, scope, values, active)?;
            acc = Some(match acc {
                None => v,
                Some(a) => concat(&a, &v),
            });
        }
        Ok(acc.expect("an item"))
    }
}

/// Whether `op` is a comparison.
fn compare(op: BinOp) -> bool {
    use BinOp::*;
    matches!(op, Eq | Ne | CaseEq | CaseNe | Lt | Le | Gt | Ge)
}

/// `v` extended to the width of `t` (at least its own): with copies of its
/// top bit if `t` is signed, else with zeros.
fn extend(v: &BitVec, t: Ty) -> BitVec {
    if t.signed {
        BitVec::from_signed(t.width, v.signed())
    } else {
        BitVec::wrap(t.width, v.unsigned().clone())
    }
}

/// Bits `hi` down to `lo` of `v`.
fn slice(v: &BitVec, hi: usize, lo: usize) -> BitVec {
    BitVec::wrap(hi + 1 - lo, v.unsigned() >> lo)
}

/// `{a, b}`
fn concat(a: &BitVec, b: &BitVec) -> BitVec {
    BitVec::wrap(
        a.width() + b.width(),
        (a.unsigned() << b.width()) | b.unsigned(),
    )
}

/// `v` with the bits from `lo` upwards replaced by `bits`.
fn write(v: &BitVec, lo: usize, bits: &BitVec) -> BitVec {
    let mask = ones(bits.width()) << lo;
    let kept = v.unsigned() & (ones(v.width()) ^ mask);
    BitVec::wrap(v.width(), kept | (bits.unsigned() << lo))
}

/// `v` with the bits from `lo` upwards replaced by `bits`, those of them
/// inside `v`.
fn put(v: &BitVec, lo: &BigInt, bits: &BitVec) -> BitVec {
    match lo.to_usize() {
        Some(lo) if lo + bits.width() <= v.width() => write(v, lo, bits),
        // bit by bit from the lowest one at `lo`, skipping those outside
        _ => {
            let mut new = v.clone();
            for k in 0..bits.width() {
                if let Some(at) = (lo + k).to_usize().filter(|at| *at < v.width()) {
                    new = write(&new, at, &slice(bits, k, k));
                }
            }
            new
        }
    }
}

/// Where an assignment writes some of the bits of its value: `width` bits
/// of the signal `target` from bit `lo` upwards, those of them inside the
/// signal; nowhere if `target` is `None` (an element at an index that is
/// no address).
struct Place {
    target: Option<Name>,
    lo: BigInt,
    width: usize,
}

impl Place {
    /// Whether the place is all of a signal of `width` bits.
    fn whole(&self, width: usize) -> bool {
        self.lo.is_zero() && self.width == width
    }
}

/// The signals an `assign`, `always_comb` or `always_ff` item assigns,
/// each once, in the order of their first assignment (every element of
/// an array written at a variable index).
fn targets(item: &Item) -> Vec<&Name> {
    fn lvalue<'a>(l: &'a Lvalue, out: &mut Vec<&'a Name>) {
        let mut add = |y: &'a Name| {
            if !out.contains(&y) {
                out.push(y);
            }
        };
        match &l.kind {
            LvalueKind::Var(y) | LvalueKind::Select(y, _) => add(y),
            LvalueKind::Elem(e) => match &e.elems {
                Elems::Signals(ys) => ys.iter().for_each(add),
                Elems::Consts(..) | Elems::Unresolved => {}
            },
            LvalueKind::Concat(ls) => ls.iter().for_each(|l| lvalue(l, out)),
        }
    }
    fn stmt<'a>(s: &'a Stmt, out: &mut Vec<&'a Name>) {
        match &s.kind {
            StmtKind::NonBlocking { lhs, .. }
            | StmtKind::Blocking { lhs, .. }
            | StmtKind::Unknown { lhs, .. } => lvalue(lhs, out),
            StmtKind::Block(ss) => ss.iter().for_each(|s| stmt(s, out)),
            StmtKind::If(_, a, b) => {
                stmt(a, out);
                stmt(b, out);
            }
            StmtKind::Case { items, default, .. } => {
                items.iter().for_each(|i| stmt(&i.body, out));
                if let Some(d) = default {
                    stmt(d, out);
                }
            }
        }
    }
    let mut out = Vec::new();
    match item {
        Item::Assign { lhs, .. } => lvalue(lhs, &mut out),
        Item::AlwaysFf { body, .. } | Item::AlwaysComb { body, .. } => stmt(body, &mut out),
        Item::Instance(_) => {}
    }
    out
}

/// The reset branch of the body of an `always_ff` block with an
/// asynchronous reset: the `then` branch of the `if` the body consists of
/// (possibly inside `begin … end`; typing makes sure there is one).
fn reset_branch(body: &Stmt) -> &Stmt {
    match &body.kind {
        StmtKind::If(_, then, _) => then,
        StmtKind::Block(ss) if ss.len() == 1 => reset_branch(&ss[0]),
        _ => unreachable!("the body of a block with a reset is an `if`"),
    }
}

fn bit(b: bool) -> BitVec {
    BitVec::wrap(1, b as u8)
}

fn ones(width: usize) -> BigUint {
    (BigUint::one() << width) - BigUint::one()
}

fn unary(op: UnOp, a: &BitVec) -> BitVec {
    let w = a.width();
    let v = a.unsigned();
    let parity = v.count_ones() % 2 == 1;
    match op {
        UnOp::BitNot => BitVec::wrap(w, ones(w) ^ v),
        UnOp::Neg => BitVec::from_signed(w, -BigInt::from(v.clone())),
        UnOp::Plus => a.clone(),
        UnOp::LogNot => bit(v.is_zero()),
        UnOp::RedAnd => bit(*v == ones(w)),
        UnOp::RedOr => bit(!v.is_zero()),
        UnOp::RedXor => bit(parity),
        UnOp::RedNand => bit(*v != ones(w)),
        UnOp::RedNor => bit(v.is_zero()),
        UnOp::RedXnor => bit(!parity),
    }
}

/// `a ** b` at the width of `a`, the operands read as signed numbers where
/// `signed_a` and `signed_b` say (IEEE 1800 Table 11-4 with 2-state
/// values): a negative exponent gives 0 unless the base is 1 or -1; 0 to a
/// negative power is x, an error (sizing rules it out).
fn power(a: &BitVec, signed_a: bool, b: &BitVec, signed_b: bool) -> Result<BitVec, String> {
    let w = a.width();
    let base = if signed_a {
        a.signed()
    } else {
        BigInt::from(a.unsigned().clone())
    };
    let exp = if signed_b {
        b.signed()
    } else {
        BigInt::from(b.unsigned().clone())
    };
    if exp.sign() == num_bigint::Sign::Minus {
        let one = BigInt::from(1);
        return if base.is_zero() {
            Err("0 to a negative power".into())
        } else if base == one {
            Ok(BitVec::from_signed(w, one))
        } else if base == -one.clone() {
            let odd = (&exp % 2u8) != BigInt::from(0);
            Ok(BitVec::from_signed(w, if odd { -one } else { one }))
        } else {
            Ok(BitVec::zero(w))
        };
    }
    // repeated squaring, keeping `w` bits
    let (mut acc, mut sq, mut e) = (
        BigUint::from(1u8),
        a.unsigned().clone(),
        exp.magnitude().clone(),
    );
    let mask = (BigUint::from(1u8) << w) - 1u8;
    while !e.is_zero() {
        if e.bit(0) {
            acc = (&acc * &sq) & &mask;
        }
        sq = (&sq * &sq) & &mask;
        e >>= 1;
    }
    Ok(BitVec::wrap(w, acc))
}

/// `signed` is the signedness of the (equally typed) operands. Division by
/// zero is an error (typing rules it out).
fn binary(op: BinOp, a: &BitVec, b: &BitVec, signed: bool) -> Result<BitVec, String> {
    let w = a.width();
    let (x, y) = (a.unsigned(), b.unsigned());
    let cmp = if signed {
        a.signed().cmp(&b.signed())
    } else {
        x.cmp(y)
    };
    // shift amounts beyond any width saturate
    let amount = || y.to_usize().unwrap_or(usize::MAX).min(w);
    if matches!(op, BinOp::Div | BinOp::Mod) && y.is_zero() {
        return Err(format!("division by zero in `{op}`"));
    }
    Ok(match op {
        BinOp::BitAnd => BitVec::wrap(w, x & y),
        BinOp::BitOr => BitVec::wrap(w, x | y),
        BinOp::BitXor => BitVec::wrap(w, x ^ y),
        BinOp::BitXnor => BitVec::wrap(w, ones(w) ^ (x ^ y)),
        BinOp::Add => BitVec::wrap(w, x + y),
        BinOp::Sub => BitVec::from_signed(w, BigInt::from(x.clone()) - BigInt::from(y.clone())),
        BinOp::Mul => BitVec::wrap(w, x * y),
        BinOp::Eq | BinOp::CaseEq => bit(x == y),
        BinOp::Ne | BinOp::CaseNe => bit(x != y),
        BinOp::Lt => bit(cmp.is_lt()),
        BinOp::Le => bit(cmp.is_le()),
        BinOp::Gt => bit(cmp.is_gt()),
        BinOp::Ge => bit(cmp.is_ge()),
        BinOp::LogAnd => bit(!x.is_zero() && !y.is_zero()),
        BinOp::LogOr => bit(!x.is_zero() || !y.is_zero()),
        BinOp::Shl | BinOp::AShl => BitVec::wrap(w, x << amount()),
        BinOp::Shr => BitVec::wrap(w, x >> amount()),
        BinOp::AShr if signed => BitVec::from_signed(w, a.signed() >> amount()),
        BinOp::AShr => BitVec::wrap(w, x >> amount()),
        // `BigInt` division rounds toward zero, as SystemVerilog's does
        BinOp::Div if signed => BitVec::from_signed(w, a.signed() / b.signed()),
        BinOp::Div => BitVec::wrap(w, x / y),
        BinOp::Mod if signed => BitVec::from_signed(w, a.signed() % b.signed()),
        BinOp::Mod => BitVec::wrap(w, x % y),
        BinOp::Pow => unreachable!("see `power`"),
    })
}
