/*!
# Modules as SMT-LIB

Prints sorts, values and op applications of [`SvOp`] as SMT-LIB 2.6 text
in the logic `QF_BV` ([`sort`], [`lit`], [`apply`]), and one round of a
translated module as a list of declarations and definitions ([`round`]).
A caller combines them into a script (adding `(set-logic QF_BV)`, asserts
and queries).

## Sorts and values

- The sort of width `N` is `(_ BitVec N)`. There is no `Bool` sort for
  1-bit values: conditions and the results of comparisons and reductions
  are `(_ BitVec 1)`, with `#b1` for true. Where SMT-LIB needs a `Bool`
  (inside `ite`) it is `(= c #b1)`; where an SMT-LIB predicate gives one
  (`=`, `bvult`, …) it is turned back with `(ite p #b1 #b0)`.
- A value of width `N` is a binary literal with exactly `N` digits, most
  significant first: `8'd5` is `#b00000101`.
- A name is a quoted symbol `|name|`. Generated names contain `@`, `%`, `.`
  or `*`, which quoting allows; `|` and `\` cannot occur in a name.

## Ops

Every op application has the sort `(_ BitVec w)` of its [result
sort](SvOp::result_sort) and the value [`value::eval`](crate::value::eval)
gives it. For operands `a`, `b`, `c` of width `w` (and `k` for a shift
amount):

| op | SMT-LIB |
|---|---|
| `Lit(v)` | the literal of `v` |
| `Id` | `a` |
| `Not And Or Xor` | `bvnot bvand bvor bvxor` |
| `Neg Add Sub Mul` | `bvneg bvadd bvsub bvmul` |
| `UDiv URem` | `bvudiv bvurem` |
| `SDiv SRem` | `bvsdiv bvsrem` (`bvsrem` has the sign of the dividend, unlike `bvsmod`) |
| `Eq Ne` | `(ite (= a b) #b1 #b0)`, `(ite (= a b) #b0 #b1)` |
| `ULt ULe SLt SLe` | `(ite (bvult a b) #b1 #b0)`, with `bvule bvslt bvsle` |
| `RedAnd` | `(ite (= a #b1…1) #b1 #b0)` |
| `RedOr` | `(ite (= a #b0…0) #b0 #b1)` |
| `RedXor` | `(bvxor ((_ extract 0 0) a) … ((_ extract w-1 w-1) a))`; `a` if `w = 1` |
| `Ite` | `(ite (= a #b1) b c)` |
| `Shl LShr AShr` | `(bvshl a b')`, with `bvlshr bvashr`, and `b'` below |
| `Extract{hi,lo}` | `((_ extract hi lo) a)` |
| `Concat` | `(concat a b)` (the first operand gives the high bits) |
| `ZExt(to)` `SExt(to)` | `((_ zero_extend k) a)`, `((_ sign_extend k) a)` with `k = to - w`; `a` if `k = 0` |

SMT-LIB shifts need an amount of the value's width `w`, while `SvOp` takes
an amount of any width `k`. The amount `b'` is `b` if `k = w`,
`((_ zero_extend w-k) b)` if `k < w`, and otherwise `b` saturated at `w`:
`(ite (bvult b W) ((_ extract w-1 0) b) Wʹ)`, where `W` and `Wʹ` are the
literals of the number `w` in `k` and `w` bits (`w < 2^w`, so it fits). A
shift by `w` or more gives 0 (`AShr`: copies of the top bit) in both
languages; plain truncation of `b` would not.

Division by zero cannot occur in a translated design (the divisor is a
nonzero constant), and [`value::eval`](crate::value::eval) leaves it
undefined. SMT-LIB defines it: `bvudiv` by 0 gives all ones, `bvurem` the
dividend, `bvsdiv` all ones for a non-negative dividend and 1 otherwise,
`bvsrem` the dividend.

`Havoc` (any value) has no closed expression: a caller declares a fresh
constant for it. `Zero` writes a derivative and only occurs in delay
blocks, which have no counterpart here. [`apply`] panics on both.

Operands are repeated in the output (`RedXor`, saturated shift amounts),
so they should be short: names or literals.

## A round of a module

[`round`] prints one round of a [`Translation`]: the init round (the init
block of every atom) or an update round (the update block of every atom).
Its values are constants named after the variables:

| constant | meaning | |
|---|---|---|
| `\|*v\|` | the latched value of `v` (its value in the previous round) | declared |
| `\|X(v)\|`, `v` external | the next value of `v`, given by the environment | declared |
| `\|X(v)\|`, `v` controlled | the next value of `v`, computed by the round | defined |

A variable name is the name of the signal (`q`, `u.q`, `q@init`); `*`, `(`
and `)` cannot occur in one, so these symbols never collide. Only the
values the round reads are declared: the init round reads no latched
values, and an update round does not read `X(q@init)`.

Then come the atoms, in the module's order (which respects the await
relation), each after a comment `; atom controls v, w`, and the terms of
the block in their order. Every term defines the wire it writes:
`(define-fun |name| () (_ BitVec w) expr)`, with `expr` as [`apply`]
prints it on the symbols of the read wires; the wire is `|X(v)|` if it is
the next value of a controlled variable, else a wire local to its atom
named `|y@1|` (an intermediate value of the signal `y`, as in the module
printer) or `|%k|` (numbered across the whole round, so unlike the module
printer's per-atom numbering). The term `Havoc` declares its wire instead:
`(declare-const |name| (_ BitVec w))`. So every symbol is declared or
defined before it is used, and the definitions of a round are equivalent
to the round of the module: a model of the declared constants determines
all defined ones, as [`rm_sim`](crate::rm_sim) computes them.

For tools that read the text, every variable's constant is preceded by a
comment line, like those of Yosys's `write_smt2`:

```text
; v2rm-latched NAME WIDTH
; v2rm-input NAME WIDTH
; v2rm-next NAME WIDTH interface|private
```
*/

use crate::ast::Name;
use crate::lower::Translation;
use crate::theory::{Sort, SvOp};
use crate::value::BitVec;
use crate::{Atom, Term, Var, Wire};
use base::var::X;
use std::collections::{BTreeSet, HashMap, HashSet};

/// `(_ BitVec N)`.
pub fn sort(s: Sort) -> String {
    format!("(_ BitVec {})", s.width())
}

/// The binary literal of `v`, with one digit per bit.
pub fn lit(v: &BitVec) -> String {
    let digits: String = (0..v.width())
        .rev()
        .map(|i| if v.unsigned().bit(i as u64) { '1' } else { '0' })
        .collect();
    format!("#b{digits}")
}

/// The quoted symbol `|name|`. Panics if the name contains `|` or `\`,
/// which a quoted symbol cannot.
pub fn symbol(name: &str) -> String {
    assert!(
        !name.contains(['|', '\\']),
        "name {name:?} cannot be an SMT-LIB symbol"
    );
    format!("|{name}|")
}

/// The application of `op` to the operands, each given as an expression
/// and its sort. Panics if the op does not apply to the sorts, and on
/// `Havoc` and `Zero`.
pub fn apply(op: &SvOp, args: &[(String, Sort)]) -> String {
    use SvOp::*;
    let sorts: Vec<Sort> = args.iter().map(|(_, s)| *s).collect();
    let w = match op.result_sort(&sorts) {
        Ok(s) => s.width(),
        Err(e) => panic!("{e}"),
    };
    let a = |i: usize| args[i].0.as_str();
    let width = |i: usize| args[i].1.width();
    let call = |f: &str| {
        let operands: Vec<&str> = (0..args.len()).map(a).collect();
        format!("({f} {})", operands.join(" "))
    };
    // a `Bool` as a 1-bit vector
    let bit = |p: String| format!("(ite {p} #b1 #b0)");

    match op {
        Lit(v) => lit(v),
        Havoc(_) | Zero(_) => panic!("{op} has no SMT-LIB expression"),
        Id => a(0).to_string(),

        Not => call("bvnot"),
        And => call("bvand"),
        Or => call("bvor"),
        Xor => call("bvxor"),

        Neg => call("bvneg"),
        Add => call("bvadd"),
        Sub => call("bvsub"),
        Mul => call("bvmul"),
        UDiv => call("bvudiv"),
        URem => call("bvurem"),
        SDiv => call("bvsdiv"),
        SRem => call("bvsrem"),

        Eq => bit(call("=")),
        Ne => format!("(ite {} #b0 #b1)", call("=")),
        ULt => bit(call("bvult")),
        ULe => bit(call("bvule")),
        SLt => bit(call("bvslt")),
        SLe => bit(call("bvsle")),

        RedAnd => bit(format!("(= {} {})", a(0), lit(&ones(width(0))))),
        RedOr => format!(
            "(ite (= {} {}) #b0 #b1)",
            a(0),
            lit(&BitVec::zero(width(0)))
        ),
        RedXor if width(0) == 1 => a(0).to_string(),
        RedXor => {
            let bits: Vec<String> = (0..width(0))
                .map(|i| format!("((_ extract {i} {i}) {})", a(0)))
                .collect();
            format!("(bvxor {})", bits.join(" "))
        }

        Ite => format!("(ite (= {} #b1) {} {})", a(0), a(1), a(2)),

        Shl | LShr | AShr => {
            let f = match op {
                Shl => "bvshl",
                LShr => "bvlshr",
                _ => "bvashr",
            };
            format!("({f} {} {})", a(0), amount(a(1), width(1), w))
        }

        Extract { hi, lo } => format!("((_ extract {hi} {lo}) {})", a(0)),
        Concat => call("concat"),
        ZExt(to) | SExt(to) if *to == width(0) => a(0).to_string(),
        ZExt(to) => format!("((_ zero_extend {}) {})", to - width(0), a(0)),
        SExt(to) => format!("((_ sign_extend {}) {})", to - width(0), a(0)),
    }
}

/// The shift amount `b` of width `k` as an amount of width `w`, saturated
/// at `w` if it is wider.
fn amount(b: &str, k: usize, w: usize) -> String {
    use std::cmp::Ordering::*;
    match k.cmp(&w) {
        Equal => b.to_string(),
        Less => format!("((_ zero_extend {}) {b})", w - k),
        Greater => format!(
            "(ite (bvult {b} {}) ((_ extract {} 0) {b}) {})",
            lit(&BitVec::wrap(k, w)),
            w - 1,
            lit(&BitVec::wrap(w, w)),
        ),
    }
}

/// The value with all `w` bits set.
fn ones(w: usize) -> BitVec {
    BitVec::from_signed(w, -1)
}

/// Which block of each atom a round runs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    /// The first round: the init blocks.
    Init,
    /// Every later round: the update blocks.
    Update,
}

/// One round of a module as SMT-LIB declarations and definitions, with
/// the names of the variables whose constants it declares or defines.
#[derive(Clone, Debug)]
pub struct Round {
    pub text: String,
    /// The variables whose latched value `|*v|` is read and declared.
    pub latched: Vec<Name>,
    /// The external variables whose next value `|X(v)|` is read and
    /// declared.
    pub inputs: Vec<Name>,
    /// The controlled variables, whose next value `|X(v)|` is defined.
    pub next: Vec<Name>,
}

/// `|*v|`: the latched value of the variable `v`.
pub fn latched_symbol(v: &str) -> String {
    symbol(&format!("*{v}"))
}

/// `|X(v)|`: the next value of the variable `v`.
pub fn next_symbol(v: &str) -> String {
    symbol(&format!("X({v})"))
}

/// The round `phase` of the module of `t`. Panics if the module reads a
/// wire before writing it (atoms out of await order), if two local wires
/// have the same name, or if the init round reads a latched value.
pub fn round(t: &Translation, phase: Phase) -> Round {
    let m = &t.module;
    let block = |a| block(a, phase);
    let reads: HashSet<Wire> = m
        .atoms()
        .iter()
        .flat_map(block)
        .flat_map(|term| term.read().iter().copied())
        .collect();
    let vars: Vec<&Var> = m
        .extl()
        .iter()
        .chain(m.intf().iter())
        .chain(m.prvt().iter())
        .collect();
    let width = |v: &Var| v.dtype().width();

    let mut text = String::new();
    // the symbol of every wire declared or defined so far
    let mut symbols: HashMap<Wire, String> = HashMap::new();
    let mut latched = vec![];
    for v in vars.iter().filter(|v| reads.contains(&***v)) {
        let name = t.name(v);
        assert!(phase == Phase::Update, "the init round reads `*{name}`");
        let sym = latched_symbol(name);
        text += &format!("; v2rm-latched {name} {}\n", width(v));
        text += &format!("(declare-const {sym} {})\n", sort(*v.dtype()));
        symbols.insert(***v, sym);
        latched.push(name.to_string());
    }
    let mut inputs = vec![];
    for v in m.extl().iter().filter(|v| reads.contains(&X(*v))) {
        let name = t.name(v);
        let sym = next_symbol(name);
        text += &format!("; v2rm-input {name} {}\n", width(v));
        text += &format!("(declare-const {sym} {})\n", sort(*v.dtype()));
        symbols.insert(X(v), sym);
        inputs.push(name.to_string());
    }

    let controlled: HashMap<Wire, &Var> = m.ctrl().iter().map(|v| (X(v), v)).collect();
    let mut next = vec![];
    let mut local: BTreeSet<Name> = BTreeSet::new();
    let mut unnamed = 0;
    for atom in m.atoms() {
        let ctrl: Vec<&str> = atom.ctrl().iter().map(|v| t.name(v)).collect();
        text += &format!("; atom controls {}\n", ctrl.join(", "));
        for term in block(atom) {
            let w = term.write()[0];
            let sym = match controlled.get(&w) {
                Some(v) => {
                    let name = t.name(v);
                    let class = if m.intf().contains(v) {
                        "interface"
                    } else {
                        "private"
                    };
                    text += &format!("; v2rm-next {name} {} {class}\n", width(v));
                    next.push(name.to_string());
                    next_symbol(name)
                }
                None => {
                    let name = t.wire_names.get(&w).cloned().unwrap_or_else(|| {
                        unnamed += 1;
                        format!("%{unnamed}")
                    });
                    assert!(local.insert(name.clone()), "two wires named `{name}`");
                    symbol(&name)
                }
            };
            let s = sort(*w.dtype());
            if let SvOp::Havoc(_) = term.itype() {
                text += &format!("(declare-const {sym} {s})\n");
            } else {
                let args: Vec<(String, Sort)> = term
                    .read()
                    .iter()
                    .map(|r| match symbols.get(r) {
                        Some(sym) => (sym.clone(), *r.dtype()),
                        None => panic!("{sym} reads a wire before it is written"),
                    })
                    .collect();
                text += &format!("(define-fun {sym} () {s} {})\n", apply(term.itype(), &args));
            }
            symbols.insert(w, sym);
        }
    }
    assert_eq!(
        next.len(),
        m.ctrl().len(),
        "a controlled variable is not defined"
    );
    Round {
        text,
        latched,
        inputs,
        next,
    }
}

/// The terms of the block of `a` that round `phase` runs.
fn block(a: &Atom, phase: Phase) -> impl Iterator<Item = &Term> {
    match phase {
        Phase::Init => a.init().iter(),
        Phase::Update => a.update().iter(),
    }
}
