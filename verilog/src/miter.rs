/*!
# Comparing a translation with Yosys

Checks that one round of our module computes the same next values as one
transition of Yosys's translation of the same design ([`yosys`](crate::yosys)), for every
state and every input: a *miter*, a script that is satisfiable exactly when
the two disagree on a variable.

## The script

[`build`] puts together, in this order:

1. `(set-logic QF_UFBV)`: bit vectors plus uninterpreted sorts and
   functions, since Yosys declares its state sort `|T_s|` with
   `declare-sort` and its accessors as functions over it (`QF_BV` is not
   enough; the arrays of `QF_AUFBV` would only be needed for memories);
2. Yosys's output ([`Smt2::text`]);
3. our update round ([`smt::round`] with [`Phase::Update`]);
4. two states `|s|` and `|s'|` of sort `|T_s|`, with the hierarchy
   predicate asserted for both and the transition `(|T_t| |s| |s'|)`;
5. the ties, matching names: every latched value `|*v|` of the round is
   the wire `v` in `|s|`, and every next input `|X(v)|` is the wire `v` in
   `|s'|`.

Each controlled variable `v` of our module then gives a [`Query`]: can
`|X(v)|` differ from the wire `v` in `|s'|`? [`check`] asks all of them
in one z3 run, each between `push` and `pop`.

## What it shows

If every query is `unsat`, the two translations agree on every variable
in every round after the first, from every state and for every input: the
latched values of the registers are arbitrary in `|s|`, and Yosys's
registers are exactly ours, with the same widths ([`build`] checks this
with [`registers`]), so this covers every reachable state without
induction. Latched
values of combinational signals (read by `always_ff` blocks) are tied to
their wires in `|s|`, where Yosys computes them from the registers and
inputs, as our previous round did. The init round is not compared: without
initializers both sides allow every initial register value, and the
combinational atoms have the same init and update blocks.
*/

use crate::ast::Name;
use crate::lower::{Translation, init_name};
use crate::smt::{self, Phase};
use crate::yosys::Smt2;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// The state before the transition.
const S: &str = "|s|";
/// The state after it.
const NEXT: &str = "|s'|";

/// The miter of a translation and Yosys's output, without the queries.
#[derive(Clone, Debug)]
pub struct Miter {
    /// The script up to and including the ties (see the module docs).
    pub script: String,
    /// One query per controlled variable, in the order of the round.
    pub queries: Vec<Query>,
}

/// Can the next value of a variable differ between the two sides?
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Query {
    /// The variable.
    pub name: Name,
    /// Our next value, `|X(v)|`.
    pub ours: String,
    /// Yosys's: the wire `v` in `|s'|`.
    pub theirs: String,
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
    /// The solver gave up.
    Unknown,
}

/// The differences between the registers of `t` and those of Yosys's
/// translation `ys`, one message each, by path: a register on one side
/// only, or of different widths. Registers are compared by path (`u.v.q`),
/// as Yosys names them, not by the name of their variable in `t`, which
/// can be a parent's signal connected to the instance's output; the width
/// of a register `p` is that of its variable `p@init`.
pub fn registers(t: &Translation, ys: &Smt2) -> Vec<String> {
    let ours: BTreeMap<&str, usize> = t
        .registers
        .iter()
        .map(|p| (p.as_str(), t.var(&init_name(p)).dtype().width()))
        .collect();
    let paths: BTreeSet<&str> = ours
        .keys()
        .copied()
        .chain(ys.registers.keys().map(|p| p.as_str()))
        .collect();
    let mut errors = vec![];
    for p in paths {
        match (ours.get(p), ys.registers.get(p)) {
            (Some(w), Some(y)) if w != y => {
                errors.push(format!("register `{p}` has width {w}, in Yosys {y}"));
            }
            (Some(_), None) => errors.push(format!("register `{p}` is not a Yosys register")),
            (None, Some(_)) => errors.push(format!("Yosys register `{p}` is not one of ours")),
            _ => {}
        }
    }
    errors
}

/// The miter of the update round of `t` and the Yosys translation `ys` of
/// the same design. Fails, listing them all, if the registers differ
/// ([`registers`]), or if a variable the round uses is not a Yosys wire or
/// has a different width there.
pub fn build(t: &Translation, ys: &Smt2) -> Result<Miter, String> {
    let round = smt::round(t, Phase::Update);
    let mut errors = registers(t, ys);
    let mut value = |name: &str, state: &str| {
        let width = t.var(name).dtype().width();
        match ys.wires.get(name) {
            Some(&w) if w == width => ys.value(name, state),
            Some(&w) => {
                errors.push(format!("`{name}` has width {width}, in Yosys {w}"));
                None
            }
            None => {
                errors.push(format!("`{name}` is not a Yosys wire"));
                None
            }
        }
    };

    let mut script = String::from("(set-logic QF_UFBV)\n");
    script += &ys.text;
    script += &round.text;
    script += "; the miter\n";
    for s in [S, NEXT] {
        script += &format!("(declare-const {s} {})\n", ys.state_sort());
        script += &format!("(assert {})\n", ys.hierarchy(s));
    }
    script += &format!("(assert {})\n", ys.transition(S, NEXT));
    let latched = round.latched.iter().map(|n| (n, smt::latched_symbol(n), S));
    let inputs = round.inputs.iter().map(|n| (n, smt::next_symbol(n), NEXT));
    for (name, ours, state) in latched.chain(inputs) {
        if let Some(theirs) = value(name, state) {
            script += &format!("(assert (= {ours} {theirs}))\n");
        }
    }
    let mut queries = vec![];
    for name in &round.next {
        if let Some(theirs) = value(name, NEXT) {
            let ours = smt::next_symbol(name);
            queries.push(Query {
                name: name.clone(),
                ours,
                theirs,
            });
        }
    }
    if !errors.is_empty() {
        return Err(format!(
            "the translations do not match: {}",
            errors.join("; ")
        ));
    }
    Ok(Miter { script, queries })
}

/// Asks every query of `m` in one run of the z3 binary `z3`. Fails if z3
/// cannot be started or answers anything but one `sat`, `unsat` or
/// `unknown` per query (z3 reports errors in its output and goes on).
pub fn check(z3: &Path, m: &Miter) -> Result<Vec<(Name, Verdict)>, String> {
    let mut script = m.script.clone();
    for q in &m.queries {
        script += &format!("(push 1)\n{}\n(check-sat)\n(pop 1)\n", q.differ());
    }
    let out = crate::z3::run(z3, &script)?;
    let verdicts: Option<Vec<Verdict>> = out
        .iter()
        .map(|line| match line.as_str() {
            "unsat" => Some(Verdict::Equal),
            "sat" => Some(Verdict::Differs),
            "unknown" => Some(Verdict::Unknown),
            _ => None,
        })
        .collect();
    match verdicts {
        Some(v) if v.len() == m.queries.len() => {
            let names = m.queries.iter().map(|q| q.name.clone());
            Ok(names.zip(v).collect())
        }
        _ => Err(format!(
            "unexpected z3 output for {} queries:\n{}",
            m.queries.len(),
            out.join("\n")
        )),
    }
}
