/*!
# From `qfbv` to the ops of `theory::bv`

[`desugar`] rewrites a [`qfbv`] module into a module over
[`Op`] with the same variables (one new variable of the same width for
each), the same atoms in the same order and the same transition relation,
like [`qfbv::bridge`](crate::qfbv::bridge()) does. A truth value becomes a
1-bit word (1 is true). Each term becomes one or a few terms:

| `qfbv` term | terms over [`Op`] (`w` = width of the operands) |
|---|---|
| `Num` | `Const`, written once per block, where first read |
| `Skip` | none (its readers read the operand), or `Id` if it writes a variable |
| `Not And Or Xor Neg Add Sub Mul Eq Ne ULt ULe Ite` | the op of the same name |
| `Ite(b, 1, 0)`, `Eq(c, 1)` with 1-bit `b`, `c` | none: `b`, `c` (the bridges between truth values and 1-bit words) |
| `Extract`, `ZeroExtend` | `BitSelect`, `Extend` |
| `Concat(a, b)` | `Or(Mul(Extend a, 2^wb), Extend b)` |
| `SignExtend(x)` | `Sub(Xor(Extend x, m), m)` with `m` = the old sign bit, `2^(w-1)` |
| `SLt`, `SLe` | `ULt`, `ULe` of both operands `Xor`ed with `2^(w-1)` |
| `UDiv`, `URem` by `d` | the op if `d` is a constant other than 0; else `Ite(d = 0, 1…1 or the dividend, op(a, Ite(d = 0, 1, d)))` |
| `SDiv`, `SRem` | on the absolute values (`Ite(sign, Neg x, x)`, folded for a constant), then the sign: of the quotient `Xor` of the signs, of the remainder that of the dividend |
| `Shl`, `LShr`, `AShr` by a constant `k` | `Mul` by `2^k`; `BitSelect` of the bits from `k` up, `Extend`ed (for `AShr` sign-extended as above, `k` at most `w-1`); 0 for `k ≥ w` |
| `Shl`, `LShr` by a variable `k` | for each bit `i` of `k` with `2^i < w` an `Ite` on it choosing the value shifted by `2^i` (as above) or not; then `Ite(w ≤ k, 0, …)` |
| `AShr(x, k)` by a variable `k` | `Ite(sign of x, Not(LShr(Not x, k)), LShr(x, k))` |
| `Havoc` | `HAVOC` |

A result that the last term wrote to a fresh wire is written to the
variable directly when the original term wrote a variable.

The guard of a division makes every divisor nonzero, so the module means
the same to an interpreter whose division by zero fails (or is undefined)
as to SMT-LIB. The signed ops, unlike the unsigned ones, depend on reading
the top bit as a sign; they are spelled out with unsigned ops so that their
meaning does not depend on how an interpreter stores words. Verilog
designs only divide by nonzero constants, so only the first form of
division occurs in translations; the guarded one is tested on modules built
by hand.
*/

use super::{Atom, MAX_WIDTH, Module, Op, Var, Wire, term};
use crate::Sort;
use crate::ast::{Error, ErrorKind, Name};
use crate::qfbv::{self, Bridged};
use crate::smt::{self, Phase};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::Path;

/// The module of a [`qfbv`] module over [`Op`], with its variables and
/// names.
pub struct Desugared {
    pub module: Module,
    /// The new variable of each variable of the `qfbv` module.
    pub vars: HashMap<qfbv::Var, Var>,
    names: HashMap<Var, Name>,
    /// The names of the local wires that write the value of a named local
    /// wire of the `qfbv` module.
    pub wire_names: HashMap<Wire, Name>,
}

impl Desugared {
    /// The name of `var`, or `?` if it is not a variable of the module.
    pub fn name(&self, var: &Var) -> &str {
        self.names.get(var).map(String::as_str).unwrap_or("?")
    }
}

/// The module of `b` over [`Op`].
pub fn desugar(b: &Bridged) -> Result<Desugared, Error> {
    let internal = |e: String| Error::unlocated(ErrorKind::Internal, e);
    let m = &b.module;
    let mut vars = HashMap::new();
    let mut names = HashMap::new();
    let mut wires: HashMap<qfbv::Wire, Wire> = HashMap::new();
    for old in m
        .extl()
        .iter()
        .chain(m.intf().iter())
        .chain(m.prvt().iter())
    {
        let new = Var::new(sort(old.dtype()));
        vars.insert(*old, new);
        names.insert(new, b.name(old).to_string());
        wires.insert(**old, *new);
        wires.insert(reactive::var::X(old), reactive::var::X(new));
    }
    let mut wire_names = HashMap::new();
    let mut atoms = vec![];
    for a in m.atoms() {
        let scope: BTreeSet<Var> = (a.ctrl().iter().chain(a.wait().iter()))
            .chain(a.read().iter())
            .map(|v| vars[v])
            .collect();
        let mut block = |terms: &mut dyn Iterator<Item = &qfbv::Term>| {
            let mut blk = Block::new(&wires, &b.wire_names);
            for old in terms {
                blk.term(old).map_err(internal)?;
            }
            let terms = blk.finish().map_err(internal)?;
            wire_names.extend(blk.names);
            Ok::<_, Error>(terms)
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
    Ok(Desugared {
        module,
        vars,
        names,
        wire_names,
    })
}

/// The sort of a `qfbv` sort: a truth value is a 1-bit word.
fn sort(s: &qfbv::Sort) -> Sort {
    match s {
        qfbv::Sort::Word(w) => Sort(*w),
        qfbv::Sort::Bool => Sort(1),
    }
}

/// Whether two blocks are the same terms (as in a combinatorial atom).
fn same(
    a: &dataflow::diagram::Block<qfbv::Op, qfbv::Sort>,
    b: &dataflow::diagram::Block<qfbv::Op, qfbv::Sort>,
) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.itype() == y.itype() && x.write() == y.write() && x.read() == y.read())
}

/// All ones in `w` bits.
fn ones(w: usize) -> u64 {
    u64::MAX >> (64 - w)
}

/// The rewriting of one block.
struct Block<'a> {
    /// The wires of the variables, old to new.
    vars: &'a HashMap<qfbv::Wire, Wire>,
    old_names: &'a HashMap<qfbv::Wire, Name>,
    /// The local wires written so far, old to new.
    locals: HashMap<qfbv::Wire, Wire>,
    /// The old local wires written by a constant, not yet written here.
    nums: HashMap<qfbv::Wire, (u64, usize)>,
    /// The constants written so far, by value and width, and back.
    consts: HashMap<(u64, usize), Wire>,
    values: HashMap<Wire, u64>,
    /// The terms so far: op, write wire, read wires.
    terms: Vec<(Op, Wire, Vec<Wire>)>,
    names: HashMap<Wire, Name>,
}

impl<'a> Block<'a> {
    fn new(vars: &'a HashMap<qfbv::Wire, Wire>, old_names: &'a HashMap<qfbv::Wire, Name>) -> Self {
        Self {
            vars,
            old_names,
            locals: HashMap::new(),
            nums: HashMap::new(),
            consts: HashMap::new(),
            values: HashMap::new(),
            terms: vec![],
            names: HashMap::new(),
        }
    }

    /// The terms of the block.
    fn finish(&self) -> Result<Vec<super::Term>, String> {
        let terms = self.terms.iter().cloned();
        terms.map(|(op, out, read)| term(op, [out], read)).collect()
    }

    /// The new wire of an old wire the block reads.
    fn read(&mut self, w: &qfbv::Wire) -> Result<Wire, String> {
        if let Some(n) = self.vars.get(w).or_else(|| self.locals.get(w)) {
            return Ok(*n);
        }
        match self.nums.get(w) {
            Some(&(value, width)) => Ok(self.num(value, width)),
            None => Err("a term reads a wire before it is written".to_string()),
        }
    }

    /// The value of an old wire written by a constant.
    fn known_old(&self, w: &qfbv::Wire) -> Option<u64> {
        let new = self.vars.get(w).or_else(|| self.locals.get(w));
        match new {
            Some(n) => self.values.get(n).copied(),
            None => self.nums.get(w).map(|(v, _)| *v),
        }
    }

    /// Adds the term `op` reading `read` and writing a new wire, returned.
    fn op(&mut self, op: Op, read: &[Wire]) -> Result<Wire, String> {
        let sorts: Vec<Sort> = read.iter().map(|w| *w.dtype()).collect();
        let out = Wire::new(op.result_sort(&sorts)?);
        self.terms.push((op, out, read.to_vec()));
        Ok(out)
    }

    /// The wire of the constant `value` of `width` bits.
    fn num(&mut self, value: u64, width: usize) -> Wire {
        if let Some(w) = self.consts.get(&(value, width)) {
            return *w;
        }
        let w = Wire::new(Sort(width));
        self.terms.push((Op::Const { value, width }, w, vec![]));
        self.consts.insert((value, width), w);
        self.values.insert(w, value);
        w
    }

    /// Makes `result` the value of the old wire `out`: a local wire is
    /// replaced by `result` in the terms after; a variable gets the value
    /// written to it, by the last term if that wrote `result` to a fresh
    /// wire, else by `Id`.
    fn bind(&mut self, out: &qfbv::Wire, result: Wire) {
        let Some(&var) = self.vars.get(out) else {
            self.locals.insert(*out, result);
            if let Some(n) = self.old_names.get(out)
                && !self.values.contains_key(&result)
                && !self.vars.values().any(|v| *v == result)
            {
                self.names.entry(result).or_insert_with(|| n.clone());
            }
            return;
        };
        let fresh = !self.values.contains_key(&result) && !self.vars.values().any(|v| *v == result);
        match self.terms.last_mut() {
            Some((_, last, _)) if fresh && *last == result => {
                *last = var;
                for l in self.locals.values_mut() {
                    if *l == result {
                        *l = var;
                    }
                }
                self.names.remove(&result);
            }
            _ => self.terms.push((Op::Id, var, vec![result])),
        }
    }

    /// Rewrites one term.
    fn term(&mut self, t: &qfbv::Term) -> Result<(), String> {
        let [out] = t.write() else {
            return Err(format!("{}: writes {} wires", t.itype(), t.write().len()));
        };
        use qfbv::Op as Q;
        let op = *t.itype();
        let w = sort(out.dtype()).width();
        match op {
            Q::Num { value, width } if !self.vars.contains_key(out) => {
                self.nums.insert(*out, (value, width));
                return Ok(());
            }
            Q::Num { value, width } => {
                let n = self.num(value, width);
                self.bind(out, n);
                return Ok(());
            }
            Q::Havoc(_) => {
                let h = self.op(Op::Havoc(Sort(w)), &[])?;
                self.bind(out, h);
                return Ok(());
            }
            Q::Zero(_) => return Err("ZERO in an init or update block".to_string()),
            // the bridges between truth values and 1-bit words
            Q::Ite
                if self.known_old(&t.read()[1]) == Some(1)
                    && self.known_old(&t.read()[2]) == Some(0)
                    && w == 1 =>
            {
                let b = self.read(&t.read()[0])?;
                self.bind(out, b);
                return Ok(());
            }
            Q::Eq
                if self.known_old(&t.read()[1]) == Some(1)
                    && sort(t.read()[0].dtype()) == Sort(1) =>
            {
                let c = self.read(&t.read()[0])?;
                self.bind(out, c);
                return Ok(());
            }
            _ => {}
        }
        let a: Vec<Wire> = t
            .read()
            .iter()
            .map(|r| self.read(r))
            .collect::<Result<_, _>>()?;
        let result = match op {
            Q::Num { .. } | Q::Havoc(_) | Q::Zero(_) => unreachable!("handled above"),
            Q::Skip => a[0],
            Q::Not => self.op(Op::Not, &a)?,
            Q::And => self.op(Op::And, &a)?,
            Q::Or => self.op(Op::Or, &a)?,
            Q::Xor => self.op(Op::Xor, &a)?,
            Q::Neg => self.op(Op::Neg, &a)?,
            Q::Add => self.op(Op::Add, &a)?,
            Q::Sub => self.op(Op::Sub, &a)?,
            Q::Mul => self.op(Op::Mul, &a)?,
            Q::Eq => self.op(Op::Eq, &a)?,
            Q::Ne => self.op(Op::Ne, &a)?,
            Q::ULt => self.op(Op::ULt, &a)?,
            Q::ULe => self.op(Op::ULe, &a)?,
            Q::Ite => self.op(Op::Ite, &a)?,
            Q::Extract { hi, lo } => self.op(Op::BitSelect { high: hi, low: lo }, &a)?,
            Q::ZeroExtend(i) => self.zext(a[0], i)?,
            Q::SignExtend(i) => self.sext(a[0], i)?,
            Q::Concat => self.concat(a[0], a[1])?,
            Q::SLt | Q::SLe => {
                let m = self.num(1 << (width(a[0]) - 1), width(a[0]));
                let x = self.op(Op::Xor, &[a[0], m])?;
                let y = self.op(Op::Xor, &[a[1], m])?;
                let cmp = if op == Q::SLt { Op::ULt } else { Op::ULe };
                self.op(cmp, &[x, y])?
            }
            Q::UDiv => self.udiv(a[0], a[1], false)?,
            Q::URem => self.udiv(a[0], a[1], true)?,
            Q::SDiv => {
                let (sa, sb) = (self.sign(a[0])?, self.sign(a[1])?);
                let (x, y) = (self.abs(a[0])?, self.abs(a[1])?);
                let q = self.udiv(x, y, false)?;
                let s = self.op(Op::Xor, &[sa, sb])?;
                let n = self.op(Op::Neg, &[q])?;
                self.op(Op::Ite, &[s, n, q])?
            }
            Q::SRem => {
                let sa = self.sign(a[0])?;
                let (x, y) = (self.abs(a[0])?, self.abs(a[1])?);
                let r = self.udiv(x, y, true)?;
                let n = self.op(Op::Neg, &[r])?;
                self.op(Op::Ite, &[sa, n, r])?
            }
            Q::Shl | Q::LShr | Q::AShr => self.shift(op, a[0], a[1])?,
        };
        debug_assert_eq!(width(result), w, "{op}");
        self.bind(out, result);
        Ok(())
    }

    /// `x` zero-extended by `i` bits.
    fn zext(&mut self, x: Wire, i: usize) -> Result<Wire, String> {
        if i == 0 {
            return Ok(x);
        }
        self.op(Op::Extend { extra: i }, &[x])
    }

    /// `x` sign-extended by `i` bits: zero-extended, then the old sign bit
    /// `m` flipped and subtracted, which turns it into the sign bit of the
    /// wider word (`m - m = 0` for a positive `x`, `0 - m` for a negative).
    fn sext(&mut self, x: Wire, i: usize) -> Result<Wire, String> {
        if i == 0 {
            return Ok(x);
        }
        let w = width(x);
        let e = self.op(Op::Extend { extra: i }, &[x])?;
        let m = self.num(1 << (w - 1), w + i);
        let f = self.op(Op::Xor, &[e, m])?;
        self.op(Op::Sub, &[f, m])
    }

    /// `a` (the high bits) followed by `b`.
    fn concat(&mut self, a: Wire, b: Wire) -> Result<Wire, String> {
        let (wa, wb) = (width(a), width(b));
        let ea = self.zext(a, wb)?;
        let scale = self.num(1 << wb, wa + wb);
        let high = self.op(Op::Mul, &[ea, scale])?;
        let low = self.zext(b, wa)?;
        self.op(Op::Or, &[high, low])
    }

    /// The top bit of `x`.
    fn sign(&mut self, x: Wire) -> Result<Wire, String> {
        let w = width(x);
        match self.values.get(&x) {
            Some(v) => Ok(self.num(v >> (w - 1), 1)),
            None => self.op(
                Op::BitSelect {
                    high: w - 1,
                    low: w - 1,
                },
                &[x],
            ),
        }
    }

    /// The absolute value of `x` read as signed (`2^(w-1)` for the most
    /// negative value, as `bvneg` gives).
    fn abs(&mut self, x: Wire) -> Result<Wire, String> {
        let w = width(x);
        if let Some(&v) = self.values.get(&x) {
            let a = if v >> (w - 1) == 1 {
                v.wrapping_neg() & ones(w)
            } else {
                v
            };
            return Ok(self.num(a, w));
        }
        let s = self.sign(x)?;
        let n = self.op(Op::Neg, &[x])?;
        self.op(Op::Ite, &[s, n, x])
    }

    /// The unsigned quotient (or, if `rem`, remainder) of `a` by `d` as
    /// SMT-LIB defines it (by 0: all ones, the dividend), dividing only by a
    /// nonzero divisor.
    fn udiv(&mut self, a: Wire, d: Wire, rem: bool) -> Result<Wire, String> {
        let w = width(a);
        let op = if rem { Op::UMod } else { Op::UDiv };
        match self.values.get(&d) {
            Some(v) if *v != 0 => self.op(op, &[a, d]),
            _ => {
                let zero = self.num(0, w);
                let one = self.num(1, w);
                let z = self.op(Op::Eq, &[d, zero])?;
                let safe = self.op(Op::Ite, &[z, one, d])?;
                let q = self.op(op, &[a, safe])?;
                let by_zero = if rem { a } else { self.num(ones(w), w) };
                self.op(Op::Ite, &[z, by_zero, q])
            }
        }
    }

    /// `x` shifted by `k`, both of `w` bits.
    fn shift(&mut self, op: qfbv::Op, x: Wire, k: Wire) -> Result<Wire, String> {
        let w = width(x);
        if let Some(&k) = self.values.get(&k) {
            let k = usize::try_from(k).unwrap_or(usize::MAX);
            return self.shift_by(op, x, k);
        }
        if op == qfbv::Op::AShr {
            let s = self.sign(x)?;
            let nx = self.op(Op::Not, &[x])?;
            let l = self.shift(qfbv::Op::LShr, nx, k)?;
            let neg = self.op(Op::Not, &[l])?;
            let pos = self.shift(qfbv::Op::LShr, x, k)?;
            return self.op(Op::Ite, &[s, neg, pos]);
        }
        let mut y = x;
        let mut i = 0;
        while 1usize << i < w {
            let bit = self.op(Op::BitSelect { high: i, low: i }, &[k])?;
            let shifted = self.shift_by(op, y, 1 << i)?;
            y = self.op(Op::Ite, &[bit, shifted, y])?;
            i += 1;
        }
        let limit = self.num(w as u64, w);
        let big = self.op(Op::ULe, &[limit, k])?;
        let zero = self.num(0, w);
        self.op(Op::Ite, &[big, zero, y])
    }

    /// `x` shifted by the constant `k`.
    fn shift_by(&mut self, op: qfbv::Op, x: Wire, k: usize) -> Result<Wire, String> {
        let w = width(x);
        let k = if op == qfbv::Op::AShr {
            k.min(w - 1)
        } else {
            k
        };
        if k == 0 {
            return Ok(x);
        }
        if k >= w {
            return Ok(self.num(0, w));
        }
        match op {
            qfbv::Op::Shl => {
                let scale = self.num(1 << k, w);
                self.op(Op::Mul, &[x, scale])
            }
            qfbv::Op::LShr => {
                let high = self.op(
                    Op::BitSelect {
                        high: w - 1,
                        low: k,
                    },
                    &[x],
                )?;
                self.zext(high, k)
            }
            _ => {
                let high = self.op(
                    Op::BitSelect {
                        high: w - 1,
                        low: k,
                    },
                    &[x],
                )?;
                self.sext(high, k)
            }
        }
    }
}

/// The width of a wire.
fn width(w: Wire) -> usize {
    w.dtype().width()
}

const _: () = assert!(MAX_WIDTH <= 64, "constants are u64");

/// The SMT-LIB script that checks round `phase` of `d` against the same
/// round of `b`, as [`qfbv::bridge::script`] does for a bridged module.
pub fn script(b: &Bridged, d: &Desugared, phase: Phase) -> (String, Vec<Name>) {
    let ours = smt::round_of(
        &b.module,
        |v| b.name(v).to_string(),
        &b.wire_names,
        phase,
        "b:",
    );
    let theirs = smt::round_of(
        &d.module,
        |v| d.name(v).to_string(),
        &d.wire_names,
        phase,
        "d:",
    );
    smt::equivalence(&ours, &theirs)
}

/// Checks with the z3 binary `z3` that the init round and an update round
/// of `d` compute the same next values as those of `b` from the same
/// latched and input values; returns the variables (with the round) for
/// which z3 does not answer `unsat`, with its answer.
pub fn check(z3: &Path, b: &Bridged, d: &Desugared) -> Result<Vec<(Phase, Name, String)>, String> {
    smt::differences(z3, |phase| script(b, d, phase))
}
