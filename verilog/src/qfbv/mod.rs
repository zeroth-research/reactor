/*!
# The op set `qfbv`: SMT-LIB bit vectors

A second op set, shaped like the SMT-LIB logic `QF_BV`, into which
[`bridge()`] rewrites a finished translation over [`SvOp`](crate::SvOp).
It is the form other tools take modules in: the theories of reactive
modules follow SMT-LIB, where an op has the sort the standard gives it,
so this op set differs from [`SvOp`](crate::SvOp) in three ways:

- There are two kinds of sorts: words [`Sort::Word`] of a width and the
  truth values [`Sort::Bool`]. A comparison writes a `Bool` (not a 1-bit
  word), and the condition of [`Op::Ite`] is a `Bool`.
- Both operands of a shift have the width of the result.
- There are no reductions (`&x`, `|x`, `^x`).

Every op is the SMT-LIB function of the same name, with the same sort
rules ([`crate::smt`] prints them). Only the
ops [`bridge()`] writes are here, so every one of them is tested.

Every variable of a bridged module is a word, as in the original: `Bool`
only occurs on wires local to an atom. Constants are 64-bit numbers for
now, so the words of a bridged module have at most 64 bits ([`MAX_WIDTH`]).
*/

pub mod bridge;

pub use bridge::{Bridged, bridge};

use ::theory::{Combinatorial, Differential, Sequential};
use std::fmt;

/// The widest word: constants are [`u64`] for now.
pub const MAX_WIDTH: usize = 64;

/// The sort of a wire.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub enum Sort {
    /// A bit vector of the given width (at least 1).
    Word(usize),
    /// A truth value.
    Bool,
}

impl fmt::Display for Sort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sort::Word(w) => write!(f, "bv{w}"),
            Sort::Bool => write!(f, "bool"),
        }
    }
}

/// The ops, named like the SMT-LIB functions they are. `w` is the width of
/// the operands, unless said otherwise.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Op {
    /// `[] → w`: the constant, with `value < 2^w`.
    Num { value: u64, width: usize },
    /// `s → s`: copy (the `SKIP` of [`Sequential`]).
    Skip,
    /// `[] → s`: any value (the `HAVOC` of [`Combinatorial`]).
    Havoc(Sort),
    /// `[] → s`, writing a derivative: zero (the `ZERO` of [`Differential`]).
    Zero(Sort),

    /// `bvnot`.
    Not,
    /// `bvand`.
    And,
    /// `bvor`.
    Or,
    /// `bvxor`.
    Xor,

    /// `bvneg`.
    Neg,
    /// `bvadd`.
    Add,
    /// `bvsub`.
    Sub,
    /// `bvmul`.
    Mul,
    /// `bvudiv`.
    UDiv,
    /// `bvurem`.
    URem,
    /// `bvsdiv`.
    SDiv,
    /// `bvsrem` (the sign of the dividend).
    SRem,

    /// `w, w → w`: `bvshl`.
    Shl,
    /// `w, w → w`: `bvlshr`.
    LShr,
    /// `w, w → w`: `bvashr`.
    AShr,

    /// `w → hi-lo+1`: `(_ extract hi lo)`, with `lo ≤ hi < w`.
    Extract { hi: usize, lo: usize },
    /// `a, b → a+b`: `concat`, the first operand gives the high bits.
    Concat,
    /// `w → w+i`: `(_ zero_extend i)`.
    ZeroExtend(usize),
    /// `w → w+i`: `(_ sign_extend i)`.
    SignExtend(usize),

    /// `w, w → Bool`: `=`.
    Eq,
    /// `w, w → Bool`: `distinct`.
    Ne,
    /// `w, w → Bool`: `bvult`.
    ULt,
    /// `w, w → Bool`: `bvule`.
    ULe,
    /// `w, w → Bool`: `bvslt`.
    SLt,
    /// `w, w → Bool`: `bvsle`.
    SLe,

    /// `Bool, s, s → s`: `ite`.
    Ite,
}

impl Op {
    /// The sort of the write wire, given the sorts of the read wires, or why
    /// the op does not apply to them.
    pub fn result_sort(&self, args: &[Sort]) -> Result<Sort, String> {
        use Op::*;
        let arity = |n: usize| {
            if args.len() == n {
                Ok(())
            } else {
                Err(format!("{self}: expects {n} operands, got {}", args.len()))
            }
        };
        let word = |s: Sort| match s {
            Sort::Word(w) if w > 0 => Ok(w),
            _ => Err(format!("{self}: operand of sort {s}")),
        };
        let same = |a: Sort, b: Sort| {
            if a == b {
                Ok(())
            } else {
                Err(format!("{self}: operands of different sorts {a} and {b}"))
            }
        };
        let sort = match self {
            Num { value, width } => {
                arity(0)?;
                if *width == 0 || *width > MAX_WIDTH || (*width < 64 && value >> width != 0) {
                    return Err(format!("{self}: not a value of {width} bits"));
                }
                Sort::Word(*width)
            }
            Havoc(s) | Zero(s) => {
                arity(0)?;
                *s
            }
            Skip => {
                arity(1)?;
                args[0]
            }
            Not | Neg => {
                arity(1)?;
                word(args[0])?;
                args[0]
            }
            And | Or | Xor | Add | Sub | Mul | UDiv | URem | SDiv | SRem | Shl | LShr | AShr => {
                arity(2)?;
                word(args[0])?;
                same(args[0], args[1])?;
                args[0]
            }
            Eq | Ne | ULt | ULe | SLt | SLe => {
                arity(2)?;
                word(args[0])?;
                same(args[0], args[1])?;
                Sort::Bool
            }
            Ite => {
                arity(3)?;
                if args[0] != Sort::Bool {
                    return Err(format!("{self}: condition of sort {}", args[0]));
                }
                same(args[1], args[2])?;
                args[1]
            }
            Extract { hi, lo } => {
                arity(1)?;
                if lo > hi || *hi >= word(args[0])? {
                    return Err(format!("{self}: out of range for sort {}", args[0]));
                }
                Sort::Word(hi - lo + 1)
            }
            Concat => {
                arity(2)?;
                Sort::Word(word(args[0])? + word(args[1])?)
            }
            ZeroExtend(i) | SignExtend(i) => {
                arity(1)?;
                Sort::Word(word(args[0])? + i)
            }
        };
        match sort {
            Sort::Word(w) if w == 0 || w > MAX_WIDTH => Err(format!(
                "{self}: result of sort {sort}, wider than {MAX_WIDTH} bits"
            )),
            _ => Ok(sort),
        }
    }

    /// Checks that the op applies to the read sorts and writes one wire of
    /// its result sort.
    pub fn check(&self, read: &[Sort], write: &[Sort]) -> Result<(), String> {
        let result = self.result_sort(read)?;
        if write != [result] {
            let write: Vec<String> = write.iter().map(Sort::to_string).collect();
            return Err(format!(
                "{self}: writes [{}], but its result has sort {result}",
                write.join(", ")
            ));
        }
        Ok(())
    }
}

impl Combinatorial<Sort> for Op {
    fn havoc(range: &Sort) -> Self {
        Op::Havoc(*range)
    }
}

impl Sequential<Sort> for Op {
    fn skip(_range: &Sort) -> Self {
        Op::Skip
    }
}

impl Differential<Sort> for Op {
    fn zero(range: &Sort) -> Self {
        Op::Zero(*range)
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Op::*;
        match self {
            Num { value, width } => write!(f, "{width}'d{value}"),
            Skip => write!(f, "SKIP"),
            Havoc(_) => write!(f, "HAVOC"),
            Zero(_) => write!(f, "ZERO"),
            Not => write!(f, "NOT"),
            And => write!(f, "AND"),
            Or => write!(f, "OR"),
            Xor => write!(f, "XOR"),
            Neg => write!(f, "NEG"),
            Add => write!(f, "ADD"),
            Sub => write!(f, "SUB"),
            Mul => write!(f, "MUL"),
            UDiv => write!(f, "UDIV"),
            URem => write!(f, "UREM"),
            SDiv => write!(f, "SDIV"),
            SRem => write!(f, "SREM"),
            Shl => write!(f, "SHL"),
            LShr => write!(f, "LSHR"),
            AShr => write!(f, "ASHR"),
            Extract { hi, lo } => write!(f, "EXTRACT[{hi}:{lo}]"),
            Concat => write!(f, "CONCAT"),
            ZeroExtend(i) => write!(f, "ZERO_EXTEND[{i}]"),
            SignExtend(i) => write!(f, "SIGN_EXTEND[{i}]"),
            Eq => write!(f, "EQ"),
            Ne => write!(f, "NE"),
            ULt => write!(f, "ULT"),
            ULe => write!(f, "ULE"),
            SLt => write!(f, "SLT"),
            SLe => write!(f, "SLE"),
            Ite => write!(f, "ITE"),
        }
    }
}

pub type Module = reactive::Module<Op, Op, Op, Sort>;
pub type Atom = reactive::Atom<Op, Op, Op, Sort>;
pub type Term = dataflow::diagram::Term<Op, Sort>;
pub type Var = reactive::Var<Sort>;
pub type Wire = dataflow::diagram::Wire<Sort>;

/// A term of `op` writing `write` and reading `read`, after checking the
/// wire sorts against the op ([`Op::check`]).
pub fn term(
    op: Op,
    write: impl IntoIterator<Item = Wire>,
    read: impl IntoIterator<Item = Wire>,
) -> Result<Term, String> {
    let write: Vec<Wire> = write.into_iter().collect();
    let read: Vec<Wire> = read.into_iter().collect();
    let sorts = |ws: &[Wire]| ws.iter().map(|w| *w.dtype()).collect::<Vec<Sort>>();
    op.check(&sorts(&read), &sorts(&write))?;
    Term::function(op, write, read)
}

#[cfg(test)]
mod tests {
    use super::*;
    use Sort::*;

    #[test]
    fn sorts() {
        let w8 = Word(8);
        assert_eq!(Op::Add.result_sort(&[w8, w8]), Ok(w8));
        assert_eq!(Op::ULt.result_sort(&[w8, w8]), Ok(Bool));
        assert_eq!(Op::Ite.result_sort(&[Bool, w8, w8]), Ok(w8));
        assert_eq!(Op::Ite.result_sort(&[Bool, Bool, Bool]), Ok(Bool));
        assert!(Op::Ite.result_sort(&[Word(1), w8, w8]).is_err());
        assert!(Op::Shl.result_sort(&[w8, Word(3)]).is_err());
        assert!(Op::Add.result_sort(&[Bool, Bool]).is_err());
        assert!(Op::Eq.result_sort(&[Bool, Bool]).is_err());
        assert_eq!(Op::ZeroExtend(4).result_sort(&[w8]), Ok(Word(12)));
        assert_eq!(Op::Extract { hi: 7, lo: 4 }.result_sort(&[w8]), Ok(Word(4)));
        assert!(Op::Extract { hi: 8, lo: 4 }.result_sort(&[w8]).is_err());
        assert_eq!(Op::Concat.result_sort(&[w8, Word(3)]), Ok(Word(11)));
        assert!(Op::Concat.result_sort(&[Word(40), Word(40)]).is_err());
        let num = |value, width| Op::Num { value, width }.result_sort(&[]);
        assert_eq!(num(255, 8), Ok(w8));
        assert!(num(256, 8).is_err());
        assert_eq!(num(u64::MAX, 64), Ok(Word(64)));
        assert!(num(0, 65).is_err());
        assert!(num(0, 0).is_err());
    }
}
