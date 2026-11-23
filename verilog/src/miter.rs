/*!
# Comparing a translation with Yosys

Checks that our module computes the same values as Yosys's translation of
the same design ([`yosys`], with `clk2fflogic`, so that one
Yosys transition is one time step, as one of our rounds is): our init
round the values of Yosys's initial states, and our update round the next
values of one transition, for every state and every input, clocks
included. A *miter* is a script that is satisfiable exactly when the two
disagree on a variable.

## The script

[`build`] puts together a part both rounds share ([`Miter::script`]):

1. `(set-logic QF_UFBV)`: bit vectors plus uninterpreted sorts and
   functions, since Yosys declares its state sort `|T_s|` with
   `declare-sort` and its accessors as functions over it (`QF_BV` is not
   enough; the arrays of `QF_AUFBV` would only be needed for memories);
2. Yosys's output ([`Smt2::text`]);
3. two states `|s|` and `|s'|` of sort `|T_s|`, with the hierarchy
   predicate asserted for both and the transition `(|T_t| |s| |s'|)`;

and one part per round of ours ([`Miter::init`], [`Miter::update`]), both
defining the constants `|X(v)|`, so only one of them is asserted at a
time:

- our init round ([`smt::round`] with [`Phase::Init`]) and its ties, with
  `|s|` an initial state (`|T_i|`): the initial value `|X(p@init)|` of every
  register `p` without an initial value of its own is its output sample in
  `|s|` ([`registers`]), every input `|X(v)|` is the wire `v` in `|s|`;
- our update round ([`Phase::Update`]) and its ties, matching names: every
  latched value `|*v|` of the round is the wire `v` in `|s|`, and every next
  input `|X(v)|` is the wire `v` in `|s'|`.

Each controlled variable `v` of our module then gives a [`Query`] per
round: can `|X(v)|` differ from the wire `v`, in `|s|` after our init round,
in `|s'|` after our update round? [`check`] asks all of them, in one z3
run per round, each query between `push` and `pop`, after checking that
the round's script without a query is satisfiable.

## What it shows

If every query is `unsat`, the two translations agree on every variable
in every round, from every initial state and for every input sequence,
clocks included: by induction on the rounds, with the queries as the
steps. Round 0: in every initial state of Yosys, our init round, started
from the output samples, computes Yosys's wires; this includes each
register's own wire, so the output sample is the register's value there
(no edge happens in the first step). [`registers`] checks that Yosys has
one flip-flop per register of ours and that `|T_i|` leaves the output
samples of the registers without initial values free, so every initial
value of ours is one of Yosys's too. A register with an initial value
(`logic q = c;`) starts with it in our init round, and its init query
compares that with its wire in every initial state of Yosys: it fails if
`|T_i|` gives the register another value or none.
Round k + 1: if our values in round k are Yosys's wires in a state `|s|`,
the update queries show that our next values are Yosys's wires in any
`|s'|` after `|s|` with the same inputs. They are asked for every state
`|s|`, reachable or not, and our round reads only latched values tied to
wires in `|s|` and next inputs tied to `|s'|`. In both systems a register
changes only in a round where its clock edge happens (latched level and
new level), and reads the values just before it.

This rests on the two sides having the same interface, which [`build`]
checks ([`interfaces`]): the queries compare only our controlled
variables, and the ties make each external variable of ours equal to a
Yosys wire. An output our translation dropped would never be compared, and
a computed signal our translation made external would be tied to Yosys's
computed value, hiding every behavior of ours where it takes another.

It also rests on Yosys's side constraining no input: an `unsat` query
says nothing about the inputs Yosys's predicates exclude, and if they
exclude all (or the ties contradict them), every query is `unsat`. Two
checks guard this. [`yosys::parse`] accepts Yosys's output only if every
input and every fresh value of a blackbox output is built from declared
symbols, each its own and none a register's, which the transition would
constrain (see [`yosys`]), and
[`check`] requires each round's script without a query, Yosys's
predicates plus the ties, to be satisfiable before it reads the queries.

The update round alone is not enough: it compares next values only in
states a transition reaches. A combinational signal that differs only for
register values no transition produces (a register that is always set to
`0`, read by an `assign`) differs in round 0, and a register latching it
carries the difference on, while every update query is `unsat`.

## Counterexamples

For a query that is `sat`, [`counterexample`] runs z3 again on that query
alone and reads a model: the registers (by path) and inputs in `|s|`, the
inputs in `|s'|`, the latched values of our round and both sides' next
values ([`Cex`]). [`replay`] runs that step in [`rm_sim`](crate::rm_sim)
on our module and in the reference interpreter
[`ast_interp`](crate::ast_interp) on the source, and says which side the
reference agrees with ([`Side`]), also telling a wrong SMT text of our
round apart from a wrong module.

The reference interpreter shares the frontend with the translation (the
parser, elaboration, the checks of `typing`), so where the frontend reads
the source wrongly, the replay sides with us although we are wrong.
[`second_opinion`] therefore also runs the step in iverilog, whose reading
of the source is its own, and says which side it agrees with
([`Opinion`]): when Yosys and iverilog agree against us, the source likely
means what they say (or both deviate from the standard alike, as some
known deviations do: the opinion is evidence, not a verdict).

## Among the other checks

The miter is the only check that runs the whole pipeline, from the
parser to the SMT-LIB rounds, against a translation with a frontend of its
own, and on any design (the examples, the accepted snippets of the tests,
the benchmarks, the case study); together with iverilog it is all that
covers the frontend, elaboration and the checks of `typing`, which the
reference interpreter shares with the translation. The planted-bug suite
checks the miter itself. The map of every check and the passes it
covers is in the docs of [`ast_interp`](crate::ast_interp) ("Which check
covers which pass").
*/

use crate::ast::Name;
use crate::ast_interp::{Interp, Values};
use crate::elab::Inst;
use crate::iverilog;
use crate::lower::Translation;
use crate::rm_sim::{Sim, Valuation};
use crate::smt::{self, Phase, Round};
use crate::yosys::{self, Sample, Smt2};
use crate::{BitVec, Var};
use num_bigint::BigUint;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;
use std::time::Duration;

/// The state before the transition.
const S: &str = "|s|";
/// The state after it.
const NEXT: &str = "|s'|";

/// The miter of a translation and Yosys's output, without the queries.
#[derive(Clone, Debug)]
pub struct Miter {
    /// The script up to and including the transition (see the module
    /// docs): what both rounds share.
    pub script: String,
    /// Our init round with its ties, to be asserted after [`script`](Self::script).
    pub init: String,
    /// Our update round with its ties, likewise.
    pub update: String,
    /// One query per controlled variable and round: the init round's
    /// first, each in the order of the round.
    pub queries: Vec<Query>,
    /// The values a [`counterexample`] reads from a model.
    pub probes: Probes,
    /// The time z3 may take for each single query in [`check`] and
    /// [`counterexample`] (and for the check without a query that comes
    /// first in each round); none if `None`, as [`build`] leaves it. A
    /// query that runs out of time is [`Verdict::Unknown`].
    pub timeout: Option<Duration>,
}

/// The values a [`counterexample`] reads from a model: every register in
/// `|s|`, by path, as the output sample our init round starts from
/// (`initial`) and as the wire, the value after the step (`registers`);
/// every input in `|s|` and in `|s'|` (Yosys wires); every latched value
/// `|*v|` of our update round. A counterexample of the init round reads
/// `initial` and `inputs`, one of the update round the others.
#[derive(Clone, Debug)]
pub struct Probes {
    pub initial: Vec<Probe>,
    pub registers: Vec<Probe>,
    pub inputs: Vec<Probe>,
    pub next_inputs: Vec<Probe>,
    pub latched: Vec<Probe>,
}

/// A named value in the script, as a bit-vector term.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Probe {
    pub name: Name,
    pub term: String,
    pub width: usize,
}

/// Can the next value of a variable differ between the two sides?
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Query {
    /// The round of ours that is compared.
    pub phase: Phase,
    /// The variable.
    pub name: Name,
    /// Our next value, `|X(v)|`.
    pub ours: String,
    /// Yosys's: the wire `v` in `|s|` for the init round, in `|s'|` for
    /// the update round.
    pub theirs: String,
}

impl Miter {
    /// Our round of phase `phase` with its ties.
    pub fn round(&self, phase: Phase) -> &str {
        match phase {
            Phase::Init => &self.init,
            Phase::Update => &self.update,
        }
    }
}

impl Query {
    /// The assertion that the two sides differ.
    pub fn differ(&self) -> String {
        format!("(assert (distinct {} {}))", self.ours, self.theirs)
    }
}

/// The answer to a [`Query`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// `unsat`: equal in every state for every input.
    Equal,
    /// `sat`: some state and input make them differ.
    Differs,
    /// The solver gave up, or ran out of the time [`Miter::timeout`].
    Unknown,
}

/// The error for a register of a translation without a width (a
/// translation not made by `lower`, or changed since).
fn no_width(p: &str) -> String {
    format!("register `{p}` has no width in the translation")
}

/// The output sample of every register of `t` in Yosys's translation
/// `ys`, by path, or the differences between the two, one message each.
///
/// After `clk2fflogic`, Yosys has three registers per flip-flop (see
/// [`yosys`]): an edge sample, a data input sample and an
/// output sample, and a reset sample if the flip-flop has an asynchronous
/// reset. So there must be as many edge samples as registers of ours,
/// twice as many data samples, as many reset samples as registers with an
/// asynchronous reset, and no other Yosys registers. The
/// output sample of a register `p` (a path, as Yosys names it, not the
/// name of its variable in `t`, which can be a parent's signal connected
/// to the instance's output) is the data sample named after `p` that
/// `|T_i|` leaves free ([`Smt2::free_initially`]); a data input sample can
/// carry the same name, but is set to 0 initially. Its width must be that
/// of `p`. [`build`] ties the initial value `p@init` to it, and the
/// init query of `p`'s variable then checks that it is the output sample:
/// Yosys's wire `p` equals it in every initial state.
///
/// A register with an initial value has no `p@init`, and is left out: its
/// output sample is set to the initial value by `|T_i|`, which is no
/// different from a data sample set to 0 when the value is 0. Its init
/// query checks the value.
pub fn registers(t: &Translation, ys: &Smt2) -> Result<BTreeMap<Name, String>, Vec<String>> {
    let mut errors = vec![];
    let (mut edges, mut data, mut controls) = (0, 0, 0);
    // the data samples, by the wire they are named after
    let mut named: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for name in ys.registers.keys() {
        match yosys::sample(name) {
            Some(Sample::Edge(_)) => edges += 1,
            Some(Sample::Control) => controls += 1,
            Some(Sample::Data(wire)) => {
                data += 1;
                if let Some(wire) = wire {
                    named.entry(wire).or_default().push(name);
                }
            }
            None => errors.push(format!(
                "Yosys register `{name}` is not a sample of `clk2fflogic`"
            )),
        }
    }
    let n = t.registers.len();
    if edges != n || data != 2 * n {
        errors.push(format!(
            "Yosys has {edges} clock samples and {data} data samples, \
             for {n} registers there should be {n} and {}",
            2 * n
        ));
    }
    let r = t.reset_registers.len();
    if controls != r {
        errors.push(format!(
            "Yosys has {controls} reset samples, for {r} registers with an asynchronous \
             reset there should be {r}"
        ));
    }
    let mut out = BTreeMap::new();
    for p in t.registers.iter().filter(|p| !t.initialized.contains(p)) {
        let Some(&width) = t.register_widths.get(p) else {
            errors.push(no_width(p));
            continue;
        };
        let free: Vec<&str> = named
            .get(p.as_str())
            .into_iter()
            .flatten()
            .copied()
            .filter(|r| ys.free_initially(r))
            .collect();
        match free.as_slice() {
            [] => errors.push(format!(
                "register `{p}` has no output sample in Yosys that its initial states leave \
                 free (an initial value in Yosys only?)"
            )),
            [r] if ys.registers[*r] != width => errors.push(format!(
                "register `{p}` has width {width}, in Yosys {}",
                ys.registers[*r]
            )),
            [r] => {
                out.insert(p.clone(), r.to_string());
            }
            _ => errors.push(format!(
                "register `{p}` has several output samples in Yosys: {}",
                free.join(", ")
            )),
        }
    }
    if errors.is_empty() {
        Ok(out)
    } else {
        Err(errors)
    }
}

/// Whether `t` can be compared with Yosys at all, checked before Yosys
/// runs: not if a register that is no element of an array has bits that no
/// assignment writes ([`Translation::unwritten`]). Yosys gives those bits
/// no flip-flop: it splits the register into flip-flops for the written
/// bits and leaves the others undriven, which [`registers`] cannot pair
/// with our one variable. The elements of an array are not refused: Yosys
/// turns an array into one flip-flop per element of its full width
/// (`memory_map`), and should it split one all the same, [`registers`]
/// fails.
pub fn comparable(t: &Translation) -> Result<(), String> {
    let bits: Vec<String> = (t.unwritten.iter())
        .filter(|u| !u.bits.element)
        .map(|u| u.never_written())
        .collect();
    if bits.is_empty() {
        return Ok(());
    }
    Err(format!(
        "cannot compare with Yosys: {}; Yosys splits such a register into flip-flops \
         for the written bits only and leaves the others undriven",
        bits.join(", ")
    ))
}

/// The miter of the rounds of `t` and the Yosys translation `ys` of the
/// same design. Fails if `t` cannot be compared ([`comparable`]), or,
/// listing them all, if the registers do not match
/// ([`registers`]), if a variable the rounds use is not a Yosys wire or
/// has a different width there, or if the inputs, outputs and signals of
/// the two sides do not match ([`interfaces`]).
pub fn build(t: &Translation, ys: &Smt2) -> Result<Miter, String> {
    comparable(t)?;
    let round = smt::round(t, Phase::Update);
    let first = smt::round(t, Phase::Init);
    let (samples, mut errors) = match registers(t, ys) {
        Ok(samples) => (samples, vec![]),
        Err(errors) => (BTreeMap::new(), errors),
    };
    // (a translation from `lower` has the width of every register)
    errors.extend(
        (t.registers.iter())
            .filter(|p| !t.register_widths.contains_key(*p))
            .map(|p| no_width(p)),
    );
    let mut ties = Ties { t, ys, errors };

    let mut script = String::from("(set-logic QF_UFBV)\n");
    script += &ys.text;
    script += "; the states\n";
    for s in [S, NEXT] {
        script += &format!("(declare-const {s} {})\n", ys.state_sort());
        script += &format!("(assert {})\n", ys.hierarchy(s));
    }
    script += &format!("(assert {})\n", ys.transition(S, NEXT));

    // the init round in an initial state `|s|`: the initial values are
    // the output samples of the registers
    let mut init = format!("(assert {})\n", ys.is_initial(S));
    init += &first.text;
    init += "; the ties\n";
    for name in &first.inputs {
        let theirs = match name.strip_suffix("@init") {
            // a missing register is reported by `registers`
            Some(path) => samples.get(path).and_then(|r| ys.value(r, S)),
            None => ties.value(name, S),
        };
        if let Some(theirs) = theirs {
            init += &format!("(assert (= {} {theirs}))\n", smt::next_symbol(name));
        }
    }
    let mut update = round.text.clone();
    update += "; the ties\n";
    let latched = round.latched.iter().map(|n| (n, smt::latched_symbol(n), S));
    let inputs = round.inputs.iter().map(|n| (n, smt::next_symbol(n), NEXT));
    for (name, ours, state) in latched.chain(inputs) {
        if let Some(theirs) = ties.value(name, state) {
            update += &format!("(assert (= {ours} {theirs}))\n");
        }
    }
    let mut queries = vec![];
    for (phase, names, state) in [
        (Phase::Init, &first.next, S),
        (Phase::Update, &round.next, NEXT),
    ] {
        for name in names {
            if let Some(theirs) = ties.value(name, state) {
                queries.push(Query {
                    phase,
                    name: name.clone(),
                    ours: smt::next_symbol(name),
                    theirs,
                });
            }
        }
    }
    let probes = ties.probes(&samples, &round);
    let mut errors = ties.errors;
    errors.extend(interfaces(t, ys, &first, &round));
    let mut seen = BTreeSet::new();
    errors.retain(|e| seen.insert(e.clone()));
    if !errors.is_empty() {
        return Err(format!(
            "the translations do not match: {}",
            errors.join("; ")
        ));
    }
    Ok(Miter {
        script,
        init,
        update,
        queries,
        probes,
        timeout: None,
    })
}

/// Yosys's terms for the names of a translation, collecting what does not
/// match on the way.
struct Ties<'a> {
    t: &'a Translation,
    ys: &'a Smt2,
    errors: Vec<String>,
}

impl Ties<'_> {
    /// Yosys's term for the value of our variable `name` in `state`, if it
    /// has a wire of that name and width.
    fn value(&mut self, name: &str, state: &str) -> Option<String> {
        let width = self.t.var(name).dtype().width();
        match self.ys.wires.get(name) {
            Some(&w) if w == width => self.ys.value(name, state),
            Some(&w) => {
                self.errors
                    .push(format!("`{name}` has width {width}, in Yosys {w}"));
                None
            }
            None => {
                self.errors.push(format!("`{name}` is not a Yosys wire"));
                None
            }
        }
    }

    /// The probes of `names` (name, Yosys wire, width) in `state`.
    fn read<'n>(
        &mut self,
        names: impl Iterator<Item = (&'n String, &'n str, usize)>,
        state: &str,
    ) -> Vec<Probe> {
        let mut probes = Vec::new();
        for (name, wire, width) in names {
            match self.ys.value(wire, state) {
                Some(term) => probes.push(Probe {
                    name: name.clone(),
                    term,
                    width,
                }),
                None => self.errors.push(format!("`{wire}` is not a Yosys wire")),
            }
        }
        probes
    }

    /// What a counterexample reads: the registers by path (their output
    /// `samples`, or their wires), the inputs, and the latched values of
    /// our update `round`.
    fn probes(&mut self, samples: &BTreeMap<Name, String>, round: &Round) -> Probes {
        let (t, ys) = (self.t, self.ys);
        // a missing width is an error of `build`
        let width = |p: &String| t.register_widths.get(p).copied().unwrap_or_default();
        let initial = self.read(samples.iter().map(|(p, r)| (p, r.as_str(), width(p))), S);
        let registers = self.read(t.registers.iter().map(|p| (p, p.as_str(), width(p))), S);
        // the outputs of blackboxes are inputs of the design, Yosys's wires
        // with a free value in every step (from `cutpoint`)
        let free: Vec<(String, usize)> = (t.blackbox_outputs.iter())
            .filter(|n| !ys.inputs.contains_key(*n))
            .map(|n| (n.clone(), ys.wires.get(n).copied().unwrap_or_default()))
            .collect();
        let all = || (ys.inputs.iter()).chain(free.iter().map(|(n, w)| (n, w)));
        let inputs = self.read(all().map(|(n, w)| (n, n.as_str(), *w)), S);
        let next_inputs = self.read(all().map(|(n, w)| (n, n.as_str(), *w)), NEXT);
        let latched = round
            .latched
            .iter()
            .map(|name| Probe {
                name: name.clone(),
                term: smt::latched_symbol(name),
                width: t.var(name).dtype().width(),
            })
            .collect();
        Probes {
            initial,
            registers,
            inputs,
            next_inputs,
            latched,
        }
    }
}

/// The differences between the interfaces of `t` and of Yosys's
/// translation `ys`, one message each; `first` and `round` are our init and
/// update rounds. The queries compare only our controlled variables, and
/// the ties make each of our external variables equal to a Yosys wire, so
/// "all equal" means something only if:
///
/// - every external variable of ours (except the initial values `q@init`)
///   is free on Yosys's side too: a Yosys input of the same width, or a
///   signal driven by a blackbox, which Yosys gives a fresh value in every
///   step ([`Smt2::free_value`]), a different one for each signal;
///   otherwise a signal Yosys computes would be tied to one ours leaves
///   free, and every behavior of ours outside Yosys's would go unseen;
/// - every Yosys input is an external variable of ours, an input nothing
///   reads, or a pin (an `inout` port, which Yosys lists as an input and an
///   output, and we have no variable for);
/// - every Yosys output except the pins is an interface variable of ours
///   of the same width, queried in both rounds, and every interface
///   variable of ours is a Yosys output: a dropped output is not compared;
/// - every Yosys wire with a source name (no part of its path starts with
///   `$`) is a Yosys input, or its path names ([`Translation::paths`]) a
///   variable of ours of the same width or a signal connected to a
///   blackbox output that nothing reads (which is no variable of ours):
///   a dropped internal signal would not be compared either. The only
///   source names that are no variables of ours are thus the inputs
///   nothing reads, the pins, and those blackbox outputs.
pub fn interfaces(
    t: &Translation,
    ys: &Smt2,
    first: &smt::Round,
    round: &smt::Round,
) -> Vec<String> {
    let mut errors = vec![];
    let named = |set: &crate::Interface| -> BTreeMap<&str, usize> {
        (t.vars.iter())
            .filter(|(_, v)| set.contains(v))
            .map(|(n, v)| (n.as_str(), v.dtype().width()))
            .collect()
    };
    let (extl, intf, ctrl) = (
        named(t.module.extl()),
        named(t.module.intf()),
        named(t.module.ctrl()),
    );
    let pin = |n: &str| ys.inputs.contains_key(n) && ys.outputs.contains_key(n);
    let mut anyseq: BTreeMap<&str, &str> = BTreeMap::new();
    for (&name, &width) in &extl {
        if name.ends_with("@init") {
            continue;
        }
        let theirs = ys.wires.get(name).copied();
        if let Some(w) = theirs.filter(|&w| w != width) {
            errors.push(format!("`{name}` has width {width}, in Yosys {w}"));
            continue;
        }
        if ys.inputs.contains_key(name) && !pin(name) {
            continue;
        }
        let free = ys.free_value(name);
        match free {
            Some(symbol) if t.blackbox_outputs.iter().any(|b| b == name) => {
                if let Some(other) = anyseq.insert(symbol, name) {
                    errors.push(format!(
                        "`{other}` and `{name}` are free in every round, in Yosys one value"
                    ));
                }
            }
            _ => errors.push(format!(
                "`{name}` is free in every round, in Yosys {}",
                match (theirs, free) {
                    (None, _) => "not a wire",
                    _ if pin(name) => "a pin",
                    (_, Some(_)) => "the output of no blackbox of ours",
                    _ => "neither an input nor a blackbox output",
                }
            )),
        }
    }
    for name in ys.inputs.keys() {
        if !extl.contains_key(name.as_str()) && !t.unread_inputs.contains(name) && !pin(name) {
            errors.push(format!("Yosys input `{name}` is not an input of ours"));
        }
    }
    let queried = |r: &smt::Round, n: &str| r.next.iter().any(|m| m == n);
    for (name, &w) in &ys.outputs {
        if pin(name) {
            if extl.contains_key(name.as_str()) || ctrl.contains_key(name.as_str()) {
                errors.push(format!("Yosys pin `{name}` is a variable of ours"));
            }
            continue;
        }
        match intf.get(name.as_str()) {
            None => errors.push(format!("Yosys output `{name}` is not an output of ours")),
            Some(&width) if width != w => {
                errors.push(format!("`{name}` has width {width}, in Yosys {w}"));
            }
            Some(_) if !queried(first, name) || !queried(round, name) => {
                errors.push(format!("output `{name}` is not compared in both rounds"));
            }
            Some(_) => {}
        }
    }
    for name in intf.keys() {
        if !ys.outputs.contains_key(*name) {
            errors.push(format!("output `{name}` is not a Yosys output"));
        }
    }
    let public = |n: &str| !n.split('.').any(|part| part.starts_with('$'));
    for (path, &w) in ys.wires.iter().filter(|(n, _)| public(n)) {
        if ys.inputs.contains_key(path) {
            continue;
        }
        let Some(name) = t.paths.get(path) else {
            errors.push(format!("Yosys wire `{path}` is no signal of ours"));
            continue;
        };
        match extl.get(name.as_str()).or_else(|| ctrl.get(name.as_str())) {
            Some(&width) if width != w => {
                errors.push(format!("`{path}` has width {width}, in Yosys {w}"))
            }
            Some(_) => {}
            None if t.blackbox_outputs.contains(name) => {}
            None => errors.push(format!("Yosys wire `{path}` is not a variable of ours")),
        }
    }
    errors
}

/// Asks every query of `m`, in one run of the z3 binary `z3` per round,
/// each query between `push` and `pop`. Fails if z3 cannot be started or
/// answers anything but one `sat`, `unsat` or `unknown` per query (z3
/// reports errors in its output and goes on).
///
/// Before the queries, each run asks whether the round's script without a
/// query is satisfiable: Yosys's predicates for the two states and the
/// ties of the round. If it is not, every query is `unsat` whatever the
/// two sides compute, so `check` fails unless z3 answers `sat` (also on
/// `unknown`).
///
/// The rounds run separately so that their ties are at the outermost
/// level, where z3 simplifies with them; asserted within a `push`, they
/// can make a query much slower (`mac`: seconds instead of milliseconds).
pub fn check(z3: &Path, m: &Miter) -> Result<Vec<(Phase, Name, Verdict)>, String> {
    let mut verdicts = vec![];
    for phase in [Phase::Init, Phase::Update] {
        let queries: Vec<&Query> = m.queries.iter().filter(|q| q.phase == phase).collect();
        let mut script = format!("{}{}(check-sat)\n", m.script, m.round(phase));
        for q in &queries {
            script += &format!("(push 1)\n{}\n(check-sat)\n(pop 1)\n", q.differ());
        }
        let out = crate::z3::run_with(z3, &script, m.timeout)?;
        let round = match phase {
            Phase::Init => "init",
            Phase::Update => "update",
        };
        match out.first().map(String::as_str) {
            Some("sat") => {}
            Some(base @ ("unsat" | "unknown")) => {
                let found = match (base, m.timeout) {
                    ("unsat", _) => "have no solution (z3: unsat)".to_string(),
                    (_, None) => "may have no solution (z3: unknown)".to_string(),
                    (_, Some(t)) => format!(
                        "may have no solution (z3: unknown, perhaps out of its time limit \
                         of {} ms per query)",
                        t.as_millis()
                    ),
                };
                return Err(format!(
                    "in the {round} round, Yosys's constraints and the ties {found}, so every \
                     query would be unsat: the comparison is not meaningful"
                ));
            }
            _ => {
                return Err(format!(
                    "unexpected z3 output for the {round} round without a query:\n{}",
                    out.join("\n")
                ));
            }
        }
        let answers: Option<Vec<Verdict>> = out[1..]
            .iter()
            .map(|line| match line.as_str() {
                "unsat" => Some(Verdict::Equal),
                "sat" => Some(Verdict::Differs),
                "unknown" => Some(Verdict::Unknown),
                _ => None,
            })
            .collect();
        match answers {
            Some(v) if v.len() == queries.len() => {
                let names = queries.iter().map(|q| (phase, q.name.clone()));
                verdicts.extend(names.zip(v).map(|((p, n), v)| (p, n, v)));
            }
            _ => {
                return Err(format!(
                    "unexpected z3 output for {} queries:\n{}",
                    queries.len(),
                    out[1..].join("\n")
                ));
            }
        }
    }
    Ok(verdicts)
}

/// A state and inputs on which the two sides compute different next
/// values for a variable: a model of a [`Query`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cex {
    /// The round of the query.
    pub phase: Phase,
    /// The variable.
    pub name: Name,
    /// The registers in `|s|`, by path: their output samples for the init
    /// round, their wires for the update round.
    pub registers: Values,
    /// The inputs in `|s|`, and in `|s'|` (empty for the init round).
    pub inputs: Values,
    pub next_inputs: Values,
    /// The latched values our update round reads, by variable: by the
    /// ties, the values of these Yosys wires in `|s|` (empty for the init
    /// round).
    pub latched: Values,
    /// The value of the variable after the round: ours, `|X(v)|`, and
    /// Yosys's.
    pub ours: BitVec,
    pub theirs: BitVec,
}

/// A model of query `q` of `m`, from a second run of the z3 binary `z3`
/// with only this query and a `get-value` of the probes of `m` and both
/// sides of `q`. Fails if the query is `unsat` (or `unknown`), or if the
/// output cannot be read.
pub fn counterexample(z3: &Path, m: &Miter, q: &Query) -> Result<Cex, String> {
    let none = vec![];
    let (registers, next_inputs, latched) = match q.phase {
        Phase::Init => (&m.probes.initial, &none, &none),
        Phase::Update => (
            &m.probes.registers,
            &m.probes.next_inputs,
            &m.probes.latched,
        ),
    };
    let groups = [registers, &m.probes.inputs, next_inputs, latched];
    let probes = groups.iter().flat_map(|g| g.iter());
    let terms: Vec<&str> = probes
        .clone()
        .map(|p| p.term.as_str())
        .chain([q.ours.as_str(), q.theirs.as_str()])
        .collect();
    let script = format!(
        "(set-option :produce-models true)\n{}{}{}\n(check-sat)\n(get-value ({}))\n",
        m.script,
        m.round(q.phase),
        q.differ(),
        terms.join(" ")
    );
    let out = crate::z3::run_with(z3, &script, m.timeout)?;
    let fail = |why: &str| {
        format!(
            "no counterexample for `{}`: {why}:\n{}",
            q.name,
            out.join("\n")
        )
    };
    match out.first().map(String::as_str) {
        Some("sat") => {}
        Some("unsat") => return Err(fail("the query is unsat")),
        _ => return Err(fail("z3 did not answer `sat`")),
    }
    let mut values = model(&out[1..].join("\n"))
        .filter(|v| v.len() == terms.len())
        .ok_or_else(|| fail("unexpected z3 output"))?
        .into_iter();
    let mut take = |g: &Vec<Probe>| -> Result<Values, String> {
        g.iter()
            .map(|p| {
                let v = values.next().expect("counted");
                if v.width() == p.width {
                    Ok((p.name.clone(), v))
                } else {
                    Err(fail(&format!("`{}` given {} bits", p.name, v.width())))
                }
            })
            .collect()
    };
    let registers = take(registers)?;
    let inputs = take(&m.probes.inputs)?;
    let next_inputs = take(next_inputs)?;
    let latched = take(latched)?;
    let ours = values.next().expect("counted");
    let theirs = values.next().expect("counted");
    Ok(Cex {
        phase: q.phase,
        name: q.name.clone(),
        registers,
        inputs,
        next_inputs,
        latched,
        ours,
        theirs,
    })
}

/// The values of a `get-value` answer `((term value) …)`, in order, or
/// `None` if it is not one or a value is not a bit-vector literal.
fn model(text: &str) -> Option<Vec<BitVec>> {
    let tokens = tokens(text);
    let mut pos = 0;
    let pairs = match sexp(&tokens, &mut pos)? {
        Sexp::List(pairs) if pos == tokens.len() => pairs,
        _ => return None,
    };
    pairs
        .iter()
        .map(|pair| match pair {
            Sexp::List(kv) => match kv.as_slice() {
                [_, Sexp::Atom(v)] => literal(v),
                _ => None,
            },
            Sexp::Atom(_) => None,
        })
        .collect()
}

pub(crate) enum Sexp {
    Atom(String),
    List(Vec<Sexp>),
}

/// Parentheses, quoted symbols `|…|` and other atoms.
pub(crate) fn tokens(text: &str) -> Vec<String> {
    let mut tokens = vec![];
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '(' | ')' => tokens.push(c.to_string()),
            '|' => {
                let mut t = String::from('|');
                for c in chars.by_ref() {
                    t.push(c);
                    if c == '|' {
                        break;
                    }
                }
                tokens.push(t);
            }
            c if c.is_whitespace() => {}
            c => {
                let mut t = String::from(c);
                while let Some(&c) = chars.peek() {
                    if c.is_whitespace() || c == '(' || c == ')' || c == '|' {
                        break;
                    }
                    t.push(c);
                    chars.next();
                }
                tokens.push(t);
            }
        }
    }
    tokens
}

/// The expression starting at token `pos`, moving `pos` past it.
pub(crate) fn sexp(tokens: &[String], pos: &mut usize) -> Option<Sexp> {
    let t = tokens.get(*pos)?;
    *pos += 1;
    match t.as_str() {
        "(" => {
            let mut items = vec![];
            while tokens.get(*pos)? != ")" {
                items.push(sexp(tokens, pos)?);
            }
            *pos += 1;
            Some(Sexp::List(items))
        }
        ")" => None,
        _ => Some(Sexp::Atom(t.clone())),
    }
}

/// A bit-vector literal as z3 prints it: `#x…` (one digit per four bits)
/// or `#b…`.
fn literal(text: &str) -> Option<BitVec> {
    let (digits, radix, bits) = if let Some(d) = text.strip_prefix("#x") {
        (d, 16, 4)
    } else {
        (text.strip_prefix("#b")?, 2, 1)
    };
    let value = BigUint::parse_bytes(digits.as_bytes(), radix)?;
    BitVec::new(digits.len() * bits, value)
}

/// Which side of a [`Cex`] the reference interpreter agrees with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// With ours: Yosys's translation is wrong.
    Ours,
    /// With Yosys's: our module is wrong.
    Theirs,
    /// With neither.
    Neither,
    /// Both modules compute the reference's next value, and our round reads
    /// the values our module has in `|s|`, but our round in SMT-LIB gives
    /// another value: its text does not match our module.
    OurSmt,
}

/// The cycle of a [`Cex`] replayed in both interpreters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Replay {
    /// The next value of the variable in `rm_sim`, running our module.
    pub rm_sim: BitVec,
    /// Its value in [`ast_interp`](crate::ast_interp), the reference.
    pub reference: BitVec,
    pub side: Side,
    /// Set if both modules compute the reference's next value, but a
    /// latched value our round reads, the Yosys wire in `|s|`, is not the
    /// value our module has there: then [`side`](Self::side) is about this
    /// value in `|s|`.
    pub via: Option<Via>,
}

/// A latched value of our round that differs from our module's in `|s|`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Via {
    pub name: Name,
    /// Its value in Yosys's `|s|`.
    pub theirs: BitVec,
    /// Its value in `rm_sim` and in the reference in that cycle.
    pub rm_sim: BitVec,
    pub reference: BitVec,
}

/// Replays the step of `cex` in `rm_sim` on the module of `t` and in the
/// reference interpreter on `inst`, the hierarchy `t` was lowered from.
///
/// `rm_sim` starts in the init round, so the state `|s|` is rebuilt there:
/// round 0 gets the registers of `|s|` (their wires, the values in `|s|`)
/// as the initial values `p@init` and the inputs of `|s|`, which gives the
/// values of every signal in `|s|` (combinational atoms have the same init
/// and update blocks, and no edge happens in round 0); round 1 with the
/// inputs of `|s'|` then computes the next values. The reference
/// interpreter starts from the same registers and runs the same two
/// rounds; the second gives the values in `|s'|`. A counterexample of the
/// init round is replayed in round 0 only, from the output samples. Since
/// `|s|` need not be an initial state, round 0 of an update counterexample
/// sets the registers with initial values to their values in `|s|` too
/// ([`Sim::step_forced`]), and the reference interpreter runs on `inst`
/// with the initial values dropped ([`Inst::ignore_initializers`]): the
/// update rounds are the same. (`inst` has the signals of `t`.)
///
/// The side is decided on values that each come from one side only: the
/// next value in `rm_sim` (ours), in the reference, and in Yosys's `|s'|`,
/// which is a function of Yosys's registers and inputs. Our value in the
/// model, `|X(v)|`, is not one of them: our round computes it from latched
/// values tied to Yosys's wires in `|s|`. If all three agree, the
/// difference is in our round: a latched value differs from our module's
/// in `|s|` (see [`Via`]; the side is then decided on it), or else the
/// SMT text of the round is wrong ([`Side::OurSmt`]).
pub fn replay(inst: &Inst, t: &Translation, cex: &Cex) -> Result<Replay, String> {
    let missing = |name: &str| format!("the counterexample has no value for `{name}`");
    let extl = |inputs: &Values| -> Result<Valuation, String> {
        let value = |var: &Var| {
            let name = t.name(var);
            let v = match name.strip_suffix("@init") {
                Some(path) => cex.registers.get(path),
                None => inputs.get(name),
            };
            v.map(|v| (*var, v.clone())).ok_or_else(|| missing(name))
        };
        t.module.extl().iter().map(value).collect()
    };
    let var = |name: &str| {
        t.vars
            .get(name)
            .ok_or_else(|| format!("`{name}` is not a variable of the module"))
    };
    let get_register = |p: &str| cex.registers.get(p).cloned().ok_or_else(|| missing(p));
    let mut sim = Sim::new(&t.module);
    let naming: BTreeMap<Name, Name> = inst.naming().into_iter().collect();
    let forced: Valuation = match cex.phase {
        Phase::Init => Valuation::new(),
        Phase::Update => (t.initialized.iter())
            .map(|p| {
                let name = naming
                    .get(p)
                    .ok_or_else(|| format!("register `{p}` is not a signal of the instance"))?;
                Ok((*var(name)?, get_register(p)?))
            })
            .collect::<Result<_, String>>()?,
    };
    let now = sim.step_forced(&extl(&cex.inputs)?, &forced)?;
    let mut free;
    let reference_inst = match cex.phase {
        Phase::Init => inst,
        Phase::Update => {
            free = inst.clone();
            free.ignore_initializers();
            &free
        }
    };
    let mut interp = Interp::new(reference_inst, &cex.registers).map_err(|e| e.to_string())?;
    let reference_now = interp.step(&cex.inputs)?;
    let (rm_sim, reference) = match cex.phase {
        Phase::Init => (now[var(&cex.name)?].clone(), reference_now.clone()),
        Phase::Update => {
            let next = sim.step(&extl(&cex.next_inputs)?)?;
            (
                next[var(&cex.name)?].clone(),
                interp.step(&cex.next_inputs)?,
            )
        }
    };
    let get = |values: &Values, name: &str| {
        values
            .get(name)
            .cloned()
            .ok_or_else(|| format!("the reference interpreter has no `{name}`"))
    };
    let reference = get(&reference, &cex.name)?;

    let side = |ours: &BitVec, theirs: &BitVec, reference: &BitVec| {
        if reference == ours {
            Side::Ours
        } else if reference == theirs {
            Side::Theirs
        } else {
            Side::Neither
        }
    };
    if rm_sim != reference || reference != cex.theirs {
        return Ok(Replay {
            side: side(&rm_sim, &cex.theirs, &reference),
            rm_sim,
            reference,
            via: None,
        });
    }
    for (name, theirs) in &cex.latched {
        let ours = &now[var(name)?];
        if ours != theirs {
            let via = Via {
                name: name.clone(),
                theirs: theirs.clone(),
                rm_sim: ours.clone(),
                reference: get(&reference_now, name)?,
            };
            return Ok(Replay {
                side: side(&via.rm_sim, &via.theirs, &via.reference),
                rm_sim,
                reference,
                via: Some(via),
            });
        }
    }
    Ok(Replay {
        rm_sim,
        reference,
        side: Side::OurSmt,
        via: None,
    })
}

/// What iverilog computes for the step of a [`Cex`]: a second reference
/// next to [`ast_interp`](crate::ast_interp), independent of everything in
/// this crate, the frontend included (see [`second_opinion`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Opinion {
    /// The variable compared: the queried one, or the latched one of
    /// [`Replay::via`] (then in `|s|`).
    pub name: Name,
    /// Its value in iverilog; `None` if iverilog gives it x or z bits.
    pub iverilog: Option<BitVec>,
    /// The side iverilog agrees with: [`Side::Ours`] (our module's value),
    /// [`Side::Theirs`] (Yosys's) or [`Side::Neither`]; `None` without a
    /// value.
    pub side: Option<Side>,
}

/// Runs the step of `cex` in iverilog (on the design `files`, whose
/// hierarchy is `inst`, lowered to `t`; `bin` is the directory of
/// iverilog, see [`iverilog::find`]), as [`replay`] runs it in the
/// interpreters: from the registers of `cex` (by hierarchical assignment,
/// see [`iverilog::testbench`]), the inputs of `|s|` in the first round
/// and those of `|s'|` in the second (none for the init round). Compares
/// the value the replay decided on (the queried variable, or the latched
/// one of [`Replay::via`]) with our module's and Yosys's.
///
/// `Ok(None)` if there is nothing to decide: our module and Yosys give the
/// same value ([`Side::OurSmt`]). An error if iverilog fails, or if its
/// scheduling may differ from the round semantics on this design
/// ([`iverilog::simulator_differs`]).
pub fn second_opinion(
    bin: &Path,
    files: &[&Path],
    inst: &Inst,
    t: &Translation,
    cex: &Cex,
    replay: &Replay,
) -> Result<Option<Opinion>, String> {
    let (name, round, ours, theirs) = match &replay.via {
        Some(v) => (&v.name, 0, &v.rm_sim, &v.theirs),
        None => {
            let last = match cex.phase {
                Phase::Init => 0,
                Phase::Update => 1,
            };
            (&cex.name, last, &replay.rm_sim, &cex.theirs)
        }
    };
    if replay.side == Side::OurSmt || ours == theirs {
        return Ok(None);
    }
    if let Some(why) = iverilog::simulator_differs(inst) {
        return Err(format!(
            "iverilog may schedule this design otherwise than the round semantics: {why}"
        ));
    }
    let rounds = match cex.phase {
        Phase::Init => vec![cex.inputs.clone()],
        Phase::Update => vec![cex.inputs.clone(), cex.next_inputs.clone()],
    };
    let values = iverilog::run(bin, files, inst, t, &cex.registers, &rounds)?;
    let value = values[round]
        .get(name)
        .ok_or_else(|| format!("iverilog printed no value of `{name}`"))?
        .as_ref()
        .map(|n| BitVec::wrap(ours.width(), n.clone()));
    let side = value.as_ref().map(|v| {
        if v == ours {
            Side::Ours
        } else if v == theirs {
            Side::Theirs
        } else {
            Side::Neither
        }
    });
    Ok(Some(Opinion {
        name: name.clone(),
        iverilog: value,
        side,
    }))
}

impl fmt::Display for Opinion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Some(v) = &self.iverilog else {
            return write!(
                f,
                "iverilog gives `{}` x bits: no second opinion",
                self.name
            );
        };
        write!(f, "iverilog gives `{}` {v}: ", self.name)?;
        match self.side {
            Some(Side::Theirs) => write!(f, "Yosys and iverilog agree against us"),
            Some(Side::Ours) => write!(f, "iverilog agrees with ours"),
            _ => write!(f, "iverilog agrees with neither"),
        }
    }
}

/// `name = value, …`
fn values(vs: &Values) -> String {
    let vs: Vec<String> = vs.iter().map(|(n, v)| format!("{n} = {v}")).collect();
    vs.join(", ")
}

/// One line: the variable, both values and where they come from.
impl fmt::Display for Cex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "`{}` is {} in ours, {} in Yosys's, ",
            self.name, self.ours, self.theirs
        )?;
        match self.phase {
            Phase::Init => write!(
                f,
                "in the init round, from initial registers [{}], inputs [{}]",
                values(&self.registers),
                values(&self.inputs)
            ),
            Phase::Update => write!(
                f,
                "from registers [{}], inputs [{}], next inputs [{}], latched [{}]",
                values(&self.registers),
                values(&self.inputs),
                values(&self.next_inputs),
                values(&self.latched)
            ),
        }
    }
}

impl fmt::Display for Replay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let verdict = match self.side {
            Side::Ours => "the reference agrees with ours",
            Side::Theirs => "the reference agrees with Yosys's",
            Side::Neither => "the reference agrees with neither",
            Side::OurSmt => "our SMT round does not match our module",
        };
        write!(
            f,
            "rm_sim gives {}, the reference {}",
            self.rm_sim, self.reference
        )?;
        if let Some(v) = &self.via {
            write!(
                f,
                "; but our round reads `{}` as {} from Yosys's state, where rm_sim has {}, \
                 the reference {}",
                v.name, v.theirs, v.rm_sim, v.reference
            )?;
        }
        write!(f, ": {verdict}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_values() {
        let bv = |w, v: u32| BitVec::new(w, v).unwrap();
        let text =
            "((|*q| #xa5)\n (|X(en)| #b1)\n ((ite (|t_n x| |s'|) #b1 #b0) #b0)\n (|y| #x00f))";
        assert_eq!(
            model(text),
            Some(vec![bv(8, 0xa5), bv(1, 1), bv(1, 0), bv(12, 15)])
        );
        assert_eq!(model("((a #b01) (b true))"), None);
        assert_eq!(model("((a #b01)"), None);
        assert_eq!(model("((a #b01)) ()"), None);
        assert_eq!(model("((a #b012))"), None);
        assert_eq!(model("()"), Some(vec![]));
    }
}
