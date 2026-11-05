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

## Counterexamples

For a query that is `sat`, [`counterexample`] runs z3 again on that query
alone and reads a model: the registers (by path) and inputs in `|s|`, the
inputs in `|s'|`, the latched values of our round and both sides' next
values ([`Cex`]). [`replay`] runs that cycle in [`rm_sim`](crate::rm_sim)
on our module and in the reference interpreter
[`ast_interp`](crate::ast_interp) on the source, and says which side the
reference agrees with ([`Side`]), also telling a wrong SMT text of our
round apart from a wrong module.
*/

use crate::ast::Name;
use crate::ast_interp::{Interp, Values};
use crate::elab::Inst;
use crate::lower::{Translation, init_name};
use crate::rm_sim::{Sim, Valuation};
use crate::smt::{self, Phase};
use crate::yosys::Smt2;
use crate::{BitVec, Var};
use num_bigint::BigUint;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
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
    /// The values a [`counterexample`] reads from a model: every register
    /// in `|s|`, by path, and every input but the clock in `|s|` and in
    /// `|s'|` (Yosys wires); every latched value `|*v|` of our round.
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
    // what a counterexample reads: registers and inputs but the clock
    let mut probes = |names: &BTreeMap<String, usize>, state: &str| -> Vec<Probe> {
        let names = names.iter().filter(|(n, _)| t.clock.as_ref() != Some(*n));
        let probe = |(name, &width): (&String, &usize)| match ys.value(name, state) {
            Some(term) => Some(Probe {
                name: name.clone(),
                term,
                width,
            }),
            None => {
                errors.push(format!("`{name}` is not a Yosys wire"));
                None
            }
        };
        names.filter_map(probe).collect()
    };
    let registers = probes(&ys.registers, S);
    let inputs = probes(&ys.inputs, S);
    let next_inputs = probes(&ys.inputs, NEXT);
    let latched = round
        .latched
        .iter()
        .map(|name| Probe {
            name: name.clone(),
            term: smt::latched_symbol(name),
            width: t.var(name).dtype().width(),
        })
        .collect();
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
        queries,
        registers,
        inputs,
        next_inputs,
        latched,
    })
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

/// A state and inputs on which the two sides compute different next
/// values for a variable: a model of a [`Query`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cex {
    /// The variable.
    pub name: Name,
    /// The registers in `|s|`, by path.
    pub registers: Values,
    /// The inputs but the clock in `|s|`, and in `|s'|`.
    pub inputs: Values,
    pub next_inputs: Values,
    /// The latched values our round reads, by variable: by the ties, the
    /// values of these Yosys wires in `|s|`.
    pub latched: Values,
    /// The next value of the variable: ours, `|X(v)|`, and Yosys's.
    pub ours: BitVec,
    pub theirs: BitVec,
}

/// A model of query `q` of `m`, from a second run of the z3 binary `z3`
/// with only this query and a `get-value` of the probes of `m` and both
/// sides of `q`. Fails if the query is `unsat` (or `unknown`), or if the
/// output cannot be read.
pub fn counterexample(z3: &Path, m: &Miter, q: &Query) -> Result<Cex, String> {
    let groups = [&m.registers, &m.inputs, &m.next_inputs, &m.latched];
    let probes = groups.iter().flat_map(|g| g.iter());
    let terms: Vec<&str> = probes
        .clone()
        .map(|p| p.term.as_str())
        .chain([q.ours.as_str(), q.theirs.as_str()])
        .collect();
    let script = format!(
        "(set-option :produce-models true)\n{}{}\n(check-sat)\n(get-value ({}))\n",
        m.script,
        q.differ(),
        terms.join(" ")
    );
    let out = crate::z3::run(z3, &script)?;
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
    let registers = take(&m.registers)?;
    let inputs = take(&m.inputs)?;
    let next_inputs = take(&m.next_inputs)?;
    let latched = take(&m.latched)?;
    let ours = values.next().expect("counted");
    let theirs = values.next().expect("counted");
    Ok(Cex {
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

enum Sexp {
    Atom(String),
    List(Vec<Sexp>),
}

/// Parentheses, quoted symbols `|…|` and other atoms.
fn tokens(text: &str) -> Vec<String> {
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
fn sexp(tokens: &[String], pos: &mut usize) -> Option<Sexp> {
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

/// Replays the cycle of `cex` in `rm_sim` on the module of `t` and in the
/// reference interpreter on `inst`, the hierarchy `t` was lowered from.
///
/// `rm_sim` starts in the init round, so the state `|s|` is rebuilt there:
/// round 0 gets the registers of `|s|` as the initial values `p@init` and
/// the inputs of `|s|`, which gives the values of every signal in `|s|`
/// (combinational atoms have the same init and update blocks); round 1
/// with the inputs of `|s'|` then computes the next values. The reference
/// interpreter starts from the registers of `|s|` and runs the same two
/// cycles; the second gives the values in `|s'|`.
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
    let mut sim = Sim::new(&t.module);
    let now = sim.step(&extl(&cex.inputs)?)?;
    let next = sim.step(&extl(&cex.next_inputs)?)?;
    let rm_sim = next[var(&cex.name)?].clone();

    let mut interp = Interp::new(inst, &cex.registers).map_err(|e| e.to_string())?;
    let reference_now = interp.step(&cex.inputs)?;
    let reference = interp.step(&cex.next_inputs)?;
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
            "`{}` is {} in ours, {} in Yosys's, from registers [{}], inputs [{}], \
             next inputs [{}], latched [{}]",
            self.name,
            self.ours,
            self.theirs,
            values(&self.registers),
            values(&self.inputs),
            values(&self.next_inputs),
            values(&self.latched)
        )
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
