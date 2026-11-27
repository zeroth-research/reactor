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
use crate::qfbv;
use crate::theory::{Sort, SvOp};
use crate::value::BitVec;
use ::theory::{Combinatorial, Differential, Sequential};
use reactive::var::X;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::hash::Hash;

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
/// which a quoted symbol cannot. Neither can happen for the names this
/// crate passes: ours are made of simple identifiers (the frontend rejects
/// escaped identifiers) and the characters of generated names (`@`, `%`,
/// `.`, `*`, `(`, `)`, `[`, `]`), and [`crate::yosys::parse`] rejects
/// Yosys's names with either character.
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
    /// The declarations of the latched and input values: the start of
    /// `text`.
    pub declarations: String,
    /// The prefix of every defined symbol (see [`round_of`]).
    pub prefix: String,
}

impl Round {
    /// The symbol of the next value of the controlled variable `v`, as
    /// this round defines it.
    pub fn next_sym(&self, v: &str) -> String {
        symbol(&format!("{}X({v})", self.prefix))
    }
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
    round_of(
        &t.module,
        |v| t.name(v).to_string(),
        &t.wire_names,
        phase,
        "",
    )
}

/// An op set whose modules [`round_of`] prints: the SMT-LIB text of its
/// sorts and op applications.
pub trait Ops:
    Combinatorial<Self::Sort> + Sequential<Self::Sort> + Differential<Self::Sort> + Display
{
    type Sort: Copy + Eq + Hash + Ord + Debug;
    /// The SMT-LIB sort.
    fn sort(s: &Self::Sort) -> String;
    /// The width of a word (the sort of a variable).
    fn width(s: &Self::Sort) -> usize;
    /// Whether the op is `Havoc`, which declares its wire.
    fn is_havoc(&self) -> bool;
    /// The application of the op to the operands, each given as an
    /// expression and its sort.
    fn expr(&self, args: &[(String, Self::Sort)]) -> String;
}

impl Ops for SvOp {
    type Sort = Sort;
    fn sort(s: &Sort) -> String {
        sort(*s)
    }
    fn width(s: &Sort) -> usize {
        s.width()
    }
    fn is_havoc(&self) -> bool {
        matches!(self, SvOp::Havoc(_))
    }
    fn expr(&self, args: &[(String, Sort)]) -> String {
        apply(self, args)
    }
}

/// The round `phase` of the module `m` over the op set `O`, as [`round`]
/// prints it, with the names `name` of the variables and `wire_names` of
/// local wires. Every symbol the round defines is prefixed with `prefix`
/// (`|pX(v)|`, `|p%3|`), but not those it declares for the latched and
/// input values (`|*v|`, `|X(v)|`): two rounds of modules with the same
/// variables can be put into one script, the second without its
/// [`Round::declarations`], and then read the same values.
pub fn round_of<O: Ops>(
    m: &reactive::Module<O, O, O, O::Sort>,
    name: impl Fn(&reactive::Var<O::Sort>) -> String,
    wire_names: &HashMap<dataflow::diagram::Wire<O::Sort>, Name>,
    phase: Phase,
    prefix: &str,
) -> Round {
    type W<O> = dataflow::diagram::Wire<<O as Ops>::Sort>;
    type V<O> = reactive::Var<<O as Ops>::Sort>;
    let block = |a| block_of(a, phase);
    let reads: HashSet<W<O>> = m
        .atoms()
        .iter()
        .flat_map(block)
        .flat_map(|term| term.read().iter().copied())
        .collect();
    let vars: Vec<&V<O>> = m
        .extl()
        .iter()
        .chain(m.intf().iter())
        .chain(m.prvt().iter())
        .collect();
    let width = |v: &V<O>| O::width(v.dtype());
    let sym = |s: String| symbol(&format!("{prefix}{s}"));

    let mut text = String::new();
    // the symbol of every wire declared or defined so far
    let mut symbols: HashMap<W<O>, String> = HashMap::new();
    let mut latched = vec![];
    for v in vars.iter().filter(|v| reads.contains(&***v)) {
        let name = name(v);
        assert!(phase == Phase::Update, "the init round reads `*{name}`");
        let s = latched_symbol(&name);
        text += &format!("; v2rm-latched {name} {}\n", width(v));
        text += &format!("(declare-const {s} {})\n", O::sort(v.dtype()));
        symbols.insert(***v, s);
        latched.push(name);
    }
    let mut inputs = vec![];
    for v in m.extl().iter().filter(|v| reads.contains(&X(*v))) {
        let name = name(v);
        let s = next_symbol(&name);
        text += &format!("; v2rm-input {name} {}\n", width(v));
        text += &format!("(declare-const {s} {})\n", O::sort(v.dtype()));
        symbols.insert(X(v), s);
        inputs.push(name);
    }

    let declarations = text.clone();
    let controlled: HashMap<W<O>, &V<O>> = m.ctrl().iter().map(|v| (X(v), v)).collect();
    let mut next = vec![];
    let mut local: BTreeSet<Name> = BTreeSet::new();
    let mut unnamed = 0;
    for atom in m.atoms() {
        let ctrl: Vec<String> = atom.ctrl().iter().map(&name).collect();
        text += &format!("; atom controls {}\n", ctrl.join(", "));
        for term in block(atom) {
            let w = term.write()[0];
            let s = match controlled.get(&w) {
                Some(v) => {
                    let name = name(v);
                    let class = if m.intf().contains(v) {
                        "interface"
                    } else {
                        "private"
                    };
                    text += &format!("; v2rm-next {name} {} {class}\n", width(v));
                    let s = sym(format!("X({name})"));
                    next.push(name);
                    s
                }
                None => {
                    let name = wire_names.get(&w).cloned().unwrap_or_else(|| {
                        unnamed += 1;
                        format!("%{unnamed}")
                    });
                    assert!(local.insert(name.clone()), "two wires named `{name}`");
                    sym(name)
                }
            };
            let sort = O::sort(w.dtype());
            if term.itype().is_havoc() {
                text += &format!("(declare-const {s} {sort})\n");
            } else {
                let args: Vec<(String, O::Sort)> = term
                    .read()
                    .iter()
                    .map(|r| match symbols.get(r) {
                        Some(sym) => (sym.clone(), *r.dtype()),
                        None => panic!("{s} reads a wire before it is written"),
                    })
                    .collect();
                let e = term.itype().expr(&args);
                text += &format!("(define-fun {s} () {sort} {e})\n");
            }
            symbols.insert(w, s);
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
        declarations,
        prefix: prefix.to_string(),
    }
}

/// The SMT-LIB script that checks two rounds of modules with the same
/// variables against each other: both rounds on the same latched and input
/// values (`theirs` without its [`Round::declarations`]; its defined symbols
/// must have another prefix), and for every controlled variable in turn a
/// query whether the two next values can differ, between `(push)` and
/// `(pop)`. Each query answers `unsat` if they cannot. Returns the script
/// and the variables in the order of the queries. Panics if the rounds read
/// or define other variables.
///
/// The rounds share the declared values rather than declaring their own
/// and asserting them equal: then an op both keep (a multiplication, say)
/// is the same term on both sides, which z3 sees at once, while from
/// equated copies it would have to prove two multipliers equal bit by bit.
pub fn equivalence(ours: &Round, theirs: &Round) -> (String, Vec<Name>) {
    assert_eq!(
        ours.latched, theirs.latched,
        "the rounds read other latched values"
    );
    assert_eq!(ours.inputs, theirs.inputs, "the rounds read other inputs");
    assert_eq!(ours.next, theirs.next, "the rounds define other variables");
    assert_eq!(
        ours.declarations, theirs.declarations,
        "the rounds declare other values"
    );
    let mut s = String::from("(set-logic QF_BV)\n");
    s += &ours.text;
    s += &theirs.text[theirs.declarations.len()..];
    for n in &ours.next {
        s += &format!(
            "(push)\n(assert (distinct {} {}))\n(check-sat)\n(pop)\n",
            ours.next_sym(n),
            theirs.next_sym(n)
        );
    }
    (s, ours.next.clone())
}

/// Runs the [`equivalence`] script of the init round and of an update
/// round (`script` makes them) with the z3 binary `z3`; returns the
/// variables (with the round) for which z3 does not answer `unsat`, with
/// its answer.
pub fn differences(
    z3: &std::path::Path,
    script: impl Fn(Phase) -> (String, Vec<Name>),
) -> Result<Vec<(Phase, Name, String)>, String> {
    let mut differ = vec![];
    for phase in [Phase::Init, Phase::Update] {
        let (s, names) = script(phase);
        let answers = crate::z3::run(z3, &s)?;
        if answers.len() != names.len() {
            return Err(format!(
                "z3 answered {answers:?} to {} queries",
                names.len()
            ));
        }
        for (n, a) in names.into_iter().zip(answers) {
            if a != "unsat" {
                differ.push((phase, n, a));
            }
        }
    }
    Ok(differ)
}

/// The terms of the block of `a` that round `phase` runs.
fn block_of<O: Ops>(
    a: &reactive::Atom<O, O, O, O::Sort>,
    phase: Phase,
) -> impl Iterator<Item = &dataflow::diagram::Term<O, O::Sort>> {
    match phase {
        Phase::Init => a.init().iter(),
        Phase::Update => a.update().iter(),
    }
}

/// The ops of [`qfbv`] are SMT-LIB functions: `Bool` is
/// `Bool`, a word `(_ BitVec w)`, a constant a binary literal, and every
/// other op the function of its name (`Skip` the operand itself, `Ne`
/// `distinct`). [`Ops::expr`] panics on `Havoc` and `Zero`.
impl Ops for qfbv::Op {
    type Sort = qfbv::Sort;
    fn sort(s: &qfbv::Sort) -> String {
        match s {
            qfbv::Sort::Word(w) => format!("(_ BitVec {w})"),
            qfbv::Sort::Bool => "Bool".to_string(),
        }
    }
    fn width(s: &qfbv::Sort) -> usize {
        match s {
            qfbv::Sort::Word(w) => *w,
            qfbv::Sort::Bool => 1,
        }
    }
    fn is_havoc(&self) -> bool {
        matches!(self, qfbv::Op::Havoc(_))
    }
    fn expr(&self, args: &[(String, qfbv::Sort)]) -> String {
        use qfbv::Op::*;
        let sorts: Vec<qfbv::Sort> = args.iter().map(|(_, s)| *s).collect();
        if let Err(e) = self.result_sort(&sorts) {
            panic!("{e}");
        }
        let call = |f: &str| {
            let operands: Vec<&str> = args.iter().map(|(a, _)| a.as_str()).collect();
            format!("({f} {})", operands.join(" "))
        };
        match self {
            Num { value, width } => format!("#b{value:0width$b}"),
            Havoc(_) | Zero(_) => panic!("{self} has no SMT-LIB expression"),
            Skip => args[0].0.clone(),
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
            Shl => call("bvshl"),
            LShr => call("bvlshr"),
            AShr => call("bvashr"),
            Extract { hi, lo } => call(&format!("(_ extract {hi} {lo})")),
            Concat => call("concat"),
            ZeroExtend(i) => call(&format!("(_ zero_extend {i})")),
            SignExtend(i) => call(&format!("(_ sign_extend {i})")),
            Eq => call("="),
            Ne => call("distinct"),
            ULt => call("bvult"),
            ULe => call("bvule"),
            SLt => call("bvslt"),
            SLe => call("bvsle"),
            Ite => call("ite"),
        }
    }
}

/// The ops of [`any`](crate::any) are SMT-LIB functions on words: a
/// constant is a binary literal, `Id` the operand itself, `UMod` `bvurem`,
/// `BitSelect` and `Extend` `extract` and `zero_extend`; a comparison is 1
/// if it holds, else 0, and `Ite` takes the second operand if the first is
/// 1. [`Ops::expr`] panics on `HAVOC` and `ZERO`.
impl Ops for crate::any::Op {
    type Sort = Sort;
    fn sort(s: &Sort) -> String {
        sort(*s)
    }
    fn width(s: &Sort) -> usize {
        s.width()
    }
    fn is_havoc(&self) -> bool {
        matches!(self, crate::any::Op::Havoc(_))
    }
    fn expr(&self, args: &[(String, Sort)]) -> String {
        use crate::any::Op::*;
        let sorts: Vec<Sort> = args.iter().map(|(_, s)| *s).collect();
        if let Err(e) = self.result_sort(&sorts) {
            panic!("{e}");
        }
        let call = |f: &str| {
            let operands: Vec<&str> = args.iter().map(|(a, _)| a.as_str()).collect();
            format!("({f} {})", operands.join(" "))
        };
        let bit = |f: &str| format!("(ite {} #b1 #b0)", call(f));
        match self {
            Const { value, width } => format!("#b{value:0width$b}"),
            Havoc(_) | Zero(_) => panic!("{self} has no SMT-LIB expression"),
            Id => args[0].0.clone(),
            Not => call("bvnot"),
            And => call("bvand"),
            Or => call("bvor"),
            Xor => call("bvxor"),
            Neg => call("bvneg"),
            Add => call("bvadd"),
            Sub => call("bvsub"),
            Mul => call("bvmul"),
            UDiv => call("bvudiv"),
            UMod => call("bvurem"),
            Eq => bit("="),
            Ne => bit("distinct"),
            ULt => bit("bvult"),
            ULe => bit("bvule"),
            Ite => format!("(ite (= {} #b1) {} {})", args[0].0, args[1].0, args[2].0),
            BitSelect { high, low } => call(&format!("(_ extract {high} {low})")),
            Extend { extra } => call(&format!("(_ zero_extend {extra})")),
        }
    }
}
