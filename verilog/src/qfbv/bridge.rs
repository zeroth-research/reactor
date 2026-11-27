/*!
# From `SvOp` to `qfbv`

[`bridge`] rewrites the module of a [`Translation`] into a module over
[`Op`] with the same variables (one new variable of the same width for
each), the same atoms in the same order, and the same transition relation.
Each term becomes one or a few terms ("bridges"), all on the wires local to
its atom; the reads and writes of variables stay the same, so each atom
controls, awaits and reads the same variables as before.

| `SvOp` term | `qfbv` terms |
|---|---|
| `Lit(v)` | `Num` |
| `Id` | `Skip` |
| `Not` … `SRem`, `Extract`, `Concat` | the op of the same name |
| `Eq Ne ULt ULe SLt SLe` | the comparison, then `Ite(b, 1, 0)` |
| `RedAnd`, `RedOr` | `Eq(x, 1…1)`, `Ne(x, 0)`, then `Ite(b, 1, 0)` |
| `RedXor` | the bits of `x` (`Extract`) xored one by one from bit 0 up; `Skip` if `x` has one bit |
| `Ite(c, a, b)` | `Ite(Eq(c, 1), a, b)` |
| `ZExt(to)`, `SExt(to)` | `ZeroExtend(to - w)`, `SignExtend(to - w)`; `Skip` if `to = w` |
| `Shl LShr AShr` (`w` bits by `k` bits) | `k = w`: the op; `k < w`: the amount zero-extended to `w` bits; `k > w`: the value extended to `k` bits (`ZeroExtend`, for `AShr` `SignExtend`), shifted, and its low `w` bits `Extract`ed |
| `Havoc` | `Havoc` |

The shifts keep their meaning for amounts of `w` or more (0, or copies of
the sign bit): an amount zero-extended to `w` bits has the same value, and
in `k` bits the extended value has nothing above bit `w-1` but zeros (or
copies of its sign bit). `RedXor` is a chain, not a tree of halves (which has fewer
terms, log w steps): it is the form of the SMT-LIB printer and of Yosys, and a solver
proves a parity equal to another of a different shape only with effort
exponential in the width (64 bits do not finish), so the check of the
bridged module against the original stays easy. The constants `0`, `1` and `1…1` a block needs are
written once per block, where first used.

An atom whose init and update blocks are the same terms (a combinatorial
atom) stays one: both blocks are the same rewritten terms. The delay blocks
are not rewritten: like every discrete atom, the new one gets its `ZERO`
delay from its constructor.

For now a constant is a 64-bit number ([`super::MAX_WIDTH`]), so a module
with a wider wire is refused.
*/

use super::{Atom, MAX_WIDTH, Module, Op, Sort, Var, Wire, term};
use crate::SvOp;
use crate::ast::{Error, ErrorKind, Name};
use crate::lower::Translation;
use crate::smt::{self, Phase};
use num_traits::ToPrimitive;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::Path;

/// The module of a translation over [`Op`], with its variables and names.
pub struct Bridged {
    pub module: Module,
    /// The new variable of each variable of the original module.
    pub vars: HashMap<crate::Var, Var>,
    /// The name of each new variable (the name of the original).
    names: HashMap<Var, Name>,
    /// The names of the local wires that write the value of a named local
    /// wire of the original ([`Translation::wire_names`]).
    pub wire_names: HashMap<Wire, Name>,
}

impl Bridged {
    /// The name of `var`, or `?` if it is not a variable of the module.
    pub fn name(&self, var: &Var) -> &str {
        self.names.get(var).map(String::as_str).unwrap_or("?")
    }

    /// The variable named `name`; panics if there is none.
    pub fn var(&self, t: &Translation, name: &str) -> Var {
        self.vars[&t.var(name)]
    }
}

/// The module of `t` over [`Op`]. Fails, as not supported, if a wire of the
/// module is wider than [`MAX_WIDTH`] bits, naming the variables and local
/// signals that are.
pub fn bridge(t: &Translation) -> Result<Bridged, Error> {
    narrow(t)?;
    let internal = |e: String| Error::unlocated(ErrorKind::Internal, e);
    let m = &t.module;
    let old_vars: Vec<&crate::Var> = m
        .extl()
        .iter()
        .chain(m.intf().iter())
        .chain(m.prvt().iter())
        .collect();
    let mut vars = HashMap::new();
    let mut names = HashMap::new();
    let mut wires: HashMap<crate::Wire, Wire> = HashMap::new();
    for old in &old_vars {
        let new = Var::new(word(old.dtype()));
        vars.insert(**old, new);
        names.insert(new, t.name(old).to_string());
        wires.insert(***old, *new);
        wires.insert(reactive::var::X(*old), reactive::var::X(new));
    }
    let mut wire_names = HashMap::new();
    let mut atoms = vec![];
    for a in m.atoms() {
        let scope: BTreeSet<Var> = (a
            .ctrl()
            .iter()
            .chain(a.wait().iter())
            .chain(a.read().iter()))
        .map(|v| vars[v])
        .collect();
        let mut block = |terms: &mut dyn Iterator<Item = &crate::Term>| {
            let mut b = Block::new(&wires, &t.wire_names);
            for old in terms {
                b.term(old).map_err(internal)?;
            }
            wire_names.extend(b.names);
            Ok::<_, Error>(b.terms)
        };
        let atom = if same(a.init(), a.update()) {
            let assign = block(&mut a.init().iter())?;
            Atom::combinatorial::<Op, _, _>(&scope, assign)
        } else {
            let init = block(&mut a.init().iter())?;
            let update = block(&mut a.update().iter())?;
            Atom::sequential(&scope, init, update)
        };
        atoms.push(atom.map_err(internal)?);
    }
    let hidden: HashSet<Var> = m.prvt().iter().map(|v| vars[v]).collect();
    let module = Module::partially_observable(atoms, |v| hidden.contains(v)).map_err(internal)?;
    Ok(Bridged {
        module,
        vars,
        names,
        wire_names,
    })
}

/// Refuses a module with a wire wider than [`MAX_WIDTH`] bits.
fn narrow(t: &Translation) -> Result<(), Error> {
    let m = &t.module;
    let wide = |w: &crate::Wire| w.dtype().width() > MAX_WIDTH;
    let vars = m
        .extl()
        .iter()
        .chain(m.intf().iter())
        .chain(m.prvt().iter());
    let mut named: BTreeSet<String> = vars
        .filter(|v| wide(v))
        .map(|v| format!("`{}` ({} bits)", t.name(v), v.dtype().width()))
        .collect();
    let mut unnamed = 0;
    let locals = m
        .atoms()
        .iter()
        .flat_map(|a| a.init().iter().chain(a.update().iter()));
    let mut seen = HashSet::new();
    for w in locals.flat_map(|term| term.write().iter()) {
        if !wide(w) || !seen.insert(*w) || m.ctrl().iter().any(|v| reactive::var::X(v) == *w) {
            continue;
        }
        match t.wire_names.get(w) {
            Some(n) => {
                named.insert(format!("`{n}` ({} bits)", w.dtype().width()));
            }
            None => unnamed += 1,
        }
    }
    if named.is_empty() && unnamed == 0 {
        return Ok(());
    }
    let mut what: Vec<String> = named.into_iter().collect();
    if unnamed > 0 {
        what.push(format!("{unnamed} intermediate value(s)"));
    }
    Err(Error::unlocated(
        ErrorKind::Unsupported,
        format!(
            "words of more than {MAX_WIDTH} bits are not supported yet by the SMT-LIB \
             bit-vector op set: {}",
            what.join(", ")
        ),
    ))
}

/// The sort of a word of the width of `s`.
fn word(s: &crate::Sort) -> Sort {
    Sort::Word(s.width())
}

/// Whether two blocks are the same terms (as in a combinatorial atom).
fn same(
    a: &dataflow::diagram::Block<SvOp, crate::Sort>,
    b: &dataflow::diagram::Block<SvOp, crate::Sort>,
) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.itype() == y.itype() && x.write() == y.write() && x.read() == y.read())
}

/// The rewriting of one block.
struct Block<'a> {
    /// The wires of the variables, old to new.
    vars: &'a HashMap<crate::Wire, Wire>,
    old_names: &'a HashMap<crate::Wire, Name>,
    /// The local wires written so far, old to new.
    locals: HashMap<crate::Wire, Wire>,
    /// The constants written so far, by value and width.
    nums: HashMap<(u64, usize), Wire>,
    terms: Vec<super::Term>,
    names: HashMap<Wire, Name>,
}

impl<'a> Block<'a> {
    fn new(
        vars: &'a HashMap<crate::Wire, Wire>,
        old_names: &'a HashMap<crate::Wire, Name>,
    ) -> Self {
        Self {
            vars,
            old_names,
            locals: HashMap::new(),
            nums: HashMap::new(),
            terms: vec![],
            names: HashMap::new(),
        }
    }

    /// The new wire of a wire the block reads.
    fn read(&self, w: &crate::Wire) -> Result<Wire, String> {
        (self.vars.get(w).or_else(|| self.locals.get(w)).copied())
            .ok_or_else(|| "a term reads a wire before it is written".to_string())
    }

    /// The new wire for the write wire of a term.
    fn output(&mut self, w: &crate::Wire) -> Wire {
        if let Some(v) = self.vars.get(w) {
            return *v;
        }
        let new = Wire::new(word(w.dtype()));
        self.locals.insert(*w, new);
        if let Some(n) = self.old_names.get(w) {
            self.names.insert(new, n.clone());
        }
        new
    }

    /// Adds the term `op` writing `out` and reading `read`.
    fn put(&mut self, op: Op, out: Wire, read: &[Wire]) -> Result<(), String> {
        self.terms.push(term(op, [out], read.iter().copied())?);
        Ok(())
    }

    /// Adds the term `op` reading `read`, writing a new wire of `sort`.
    fn fresh(&mut self, op: Op, read: &[Wire], sort: Sort) -> Result<Wire, String> {
        let out = Wire::new(sort);
        self.put(op, out, read)?;
        Ok(out)
    }

    /// The wire of the constant `value` of `width` bits.
    fn num(&mut self, value: u64, width: usize) -> Result<Wire, String> {
        if let Some(w) = self.nums.get(&(value, width)) {
            return Ok(*w);
        }
        let w = self.fresh(Op::Num { value, width }, &[], Sort::Word(width))?;
        self.nums.insert((value, width), w);
        Ok(w)
    }

    /// Writes the truth value `b` to `out` as a 1-bit word.
    fn bit(&mut self, b: Wire, out: Wire) -> Result<(), String> {
        let one = self.num(1, 1)?;
        let zero = self.num(0, 1)?;
        self.put(Op::Ite, out, &[b, one, zero])
    }

    /// Rewrites one term.
    fn term(&mut self, t: &crate::Term) -> Result<(), String> {
        let a: Vec<Wire> = t
            .read()
            .iter()
            .map(|r| self.read(r))
            .collect::<Result<_, _>>()?;
        let [old_out] = t.write() else {
            return Err(format!("{}: writes {} wires", t.itype(), t.write().len()));
        };
        let out = self.output(old_out);
        let width = |w: &Wire| match w.dtype() {
            Sort::Word(n) => *n,
            Sort::Bool => 1,
        };
        let w = old_out.dtype().width();
        use SvOp as S;
        let same = |op: &SvOp| -> Option<Op> {
            Some(match op {
                S::Id => Op::Skip,
                S::Not => Op::Not,
                S::And => Op::And,
                S::Or => Op::Or,
                S::Xor => Op::Xor,
                S::Neg => Op::Neg,
                S::Add => Op::Add,
                S::Sub => Op::Sub,
                S::Mul => Op::Mul,
                S::UDiv => Op::UDiv,
                S::URem => Op::URem,
                S::SDiv => Op::SDiv,
                S::SRem => Op::SRem,
                S::Concat => Op::Concat,
                S::Extract { hi, lo } => Op::Extract { hi: *hi, lo: *lo },
                _ => return None,
            })
        };
        let compare = |op: &SvOp| -> Option<Op> {
            Some(match op {
                S::Eq => Op::Eq,
                S::Ne => Op::Ne,
                S::ULt => Op::ULt,
                S::ULe => Op::ULe,
                S::SLt => Op::SLt,
                S::SLe => Op::SLe,
                _ => return None,
            })
        };
        let op = t.itype();
        if let Some(new) = same(op) {
            return self.put(new, out, &a);
        }
        if let Some(cmp) = compare(op) {
            let b = self.fresh(cmp, &a, Sort::Bool)?;
            return self.bit(b, out);
        }
        match op {
            S::Lit(v) => {
                let value =
                    (v.unsigned().to_u64()).ok_or_else(|| format!("{v}: more than 64 bits"))?;
                self.put(Op::Num { value, width: w }, out, &[])
            }
            S::Havoc(_) => self.put(Op::Havoc(Sort::Word(w)), out, &[]),
            S::Zero(_) => Err("ZERO in an init or update block".to_string()),
            S::RedAnd | S::RedOr => {
                let n = width(&a[0]);
                let (cmp, value) = match op {
                    S::RedAnd => (Op::Eq, u64::MAX >> (64 - n)),
                    _ => (Op::Ne, 0),
                };
                let c = self.num(value, n)?;
                let b = self.fresh(cmp, &[a[0], c], Sort::Bool)?;
                self.bit(b, out)
            }
            S::RedXor => {
                let n = width(&a[0]);
                let bit =
                    |b: &mut Self, i| b.fresh(Op::Extract { hi: i, lo: i }, &[a[0]], Sort::Word(1));
                if n == 1 {
                    return self.put(Op::Skip, out, &[a[0]]);
                }
                let mut x = bit(self, 0)?;
                for i in 1..n {
                    let y = bit(self, i)?;
                    if i == n - 1 {
                        return self.put(Op::Xor, out, &[x, y]);
                    }
                    x = self.fresh(Op::Xor, &[x, y], Sort::Word(1))?;
                }
                unreachable!("the loop ends with its last bit")
            }
            S::Ite => {
                let one = self.num(1, 1)?;
                let c = self.fresh(Op::Eq, &[a[0], one], Sort::Bool)?;
                self.put(Op::Ite, out, &[c, a[1], a[2]])
            }
            S::ZExt(to) | S::SExt(to) => {
                let i = to - width(&a[0]);
                let new = match (op, i) {
                    (_, 0) => Op::Skip,
                    (S::ZExt(_), _) => Op::ZeroExtend(i),
                    _ => Op::SignExtend(i),
                };
                self.put(new, out, &a)
            }
            S::Shl | S::LShr | S::AShr => {
                let new = match op {
                    S::Shl => Op::Shl,
                    S::LShr => Op::LShr,
                    _ => Op::AShr,
                };
                let k = width(&a[1]);
                use std::cmp::Ordering::*;
                match k.cmp(&w) {
                    Equal => self.put(new, out, &a),
                    Less => {
                        let amount = self.fresh(Op::ZeroExtend(w - k), &[a[1]], Sort::Word(w))?;
                        self.put(new, out, &[a[0], amount])
                    }
                    Greater => {
                        let extend = match op {
                            S::AShr => Op::SignExtend(k - w),
                            _ => Op::ZeroExtend(k - w),
                        };
                        let value = self.fresh(extend, &[a[0]], Sort::Word(k))?;
                        let shifted = self.fresh(new, &[value, a[1]], Sort::Word(k))?;
                        self.put(Op::Extract { hi: w - 1, lo: 0 }, out, &[shifted])
                    }
                }
            }
            _ => unreachable!("{op} is handled above"),
        }
    }
}

/// The SMT-LIB script that checks round `phase` of `b` against the same
/// round of `t`: both rounds ([`smt::round`] and [`smt::round_of`], the
/// bridged one with its defined symbols prefixed by `b:`) on the same
/// latched and input values, and for every controlled variable in turn a
/// query whether the two next values can differ, between `(push)` and
/// `(pop)`. Each query answers `unsat` if they cannot. Returns the script
/// and the variables in the order of the queries.
///
/// The rounds share the declared values rather than declaring their own
/// and asserting them equal: then an op the bridge keeps (a multiplication,
/// say) is the same term on both sides, which z3 sees at once, while from
/// equated copies it would have to prove two multipliers equal bit by bit.
pub fn script(t: &Translation, b: &Bridged, phase: Phase) -> (String, Vec<Name>) {
    let ours = smt::round(t, phase);
    let theirs = smt::round_of(
        &b.module,
        |v| b.name(v).to_string(),
        &b.wire_names,
        phase,
        "b:",
    );
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
    (s, ours.next)
}

/// Checks with the z3 binary `z3` that the init round and an update round
/// of `b` compute the same next values as those of `t` from the same
/// latched and input values; returns the variables (with the round) for
/// which z3 does not answer `unsat`, with its answer.
pub fn check(
    z3: &Path,
    t: &Translation,
    b: &Bridged,
) -> Result<Vec<(Phase, Name, String)>, String> {
    let mut differ = vec![];
    for phase in [Phase::Init, Phase::Update] {
        let (s, names) = script(t, b, phase);
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
