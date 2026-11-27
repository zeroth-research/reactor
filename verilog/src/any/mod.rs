/*!
# The bit-vector ops of the `theory` crate

The reactive modules of the `theory` crate's union `theory::any` (the
form the Python side simulates) have bit vectors through the theory
`theory::bv`, whose ops follow SMT-LIB but are fewer than those of
[`qfbv`](crate::qfbv): there is no concatenation, no shift and no sign
extension, and a comparison writes a 1-bit word (there is no `Bool`).
This module turns a [`qfbv`](crate::qfbv) module into one over those ops
in two steps:

1. [`desugar()`] rewrites it into a module over [`Op`], a copy of the
   part of `theory::bv` it needs, whose constants are plain numbers. The
   missing ops are spelled out with the others; z3 checks the result
   round by round against the [`qfbv`](crate::qfbv) module
   ([`desugar::check`]).
2. [`export()`] renames that module op by op into a module of
   `theory::any`. Constants there hold tensors, which this crate cannot
   build (the tensor type depends on features of `theory`), so the caller
   passes a function that makes the constant op of a value.

Every op here is meant to be evaluated by a simple interpreter of words
of at most 64 bits, such as one storing a word in a machine integer,
as well as by SMT-LIB: [`desugar()`] only divides by a divisor that is
not zero (SMT-LIB defines division by zero, many interpreters do not),
and avoids the signed ops of `theory::bv` altogether, so their meaning
on such a representation does not matter.
*/

pub mod desugar;
pub mod export;

pub use desugar::{Desugared, desugar};
pub use export::{Exported, export};

use crate::Sort;
use ::theory::{Combinatorial, Differential, Sequential};
use std::fmt;

/// The widest word: constants are [`u64`].
pub const MAX_WIDTH: usize = 64;

/// The ops, each the op of `theory::bv` of the same name (`HAVOC` and
/// `ZERO` those of `theory::any`). All operands of an op have the same
/// width `w` unless said otherwise; a truth value is a 1-bit word.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Op {
    /// `[] → w`: the constant, with `value < 2^w`.
    Const { value: u64, width: usize },
    /// `w → w`: copy.
    Id,
    /// `[] → w`: any value.
    Havoc(Sort),
    /// `[] → w`, writing a derivative: zero.
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
    /// `bvudiv`; the divisor is never zero.
    UDiv,
    /// `bvurem`; the divisor is never zero.
    UMod,
    /// `w, w → 1`: `=`.
    Eq,
    /// `w, w → 1`: `distinct`.
    Ne,
    /// `w, w → 1`: `bvult`.
    ULt,
    /// `w, w → 1`: `bvule`.
    ULe,
    /// `1, w, w → w`: the second operand if the first is 1, else the third.
    Ite,
    /// `w → high-low+1`: `(_ extract high low)`.
    BitSelect { high: usize, low: usize },
    /// `w → w+extra`: `(_ zero_extend extra)`.
    Extend { extra: usize },
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
        let same = |a: Sort, b: Sort| {
            if a == b {
                Ok(())
            } else {
                Err(format!("{self}: operands of different sorts {a} and {b}"))
            }
        };
        let sort = match self {
            Const { value, width } => {
                arity(0)?;
                if *width == 0 || *width > MAX_WIDTH || (*width < 64 && value >> width != 0) {
                    return Err(format!("{self}: not a value of {width} bits"));
                }
                Sort(*width)
            }
            Havoc(s) | Zero(s) => {
                arity(0)?;
                *s
            }
            Id | Not | Neg => {
                arity(1)?;
                args[0]
            }
            And | Or | Xor | Add | Sub | Mul | UDiv | UMod => {
                arity(2)?;
                same(args[0], args[1])?;
                args[0]
            }
            Eq | Ne | ULt | ULe => {
                arity(2)?;
                same(args[0], args[1])?;
                Sort(1)
            }
            Ite => {
                arity(3)?;
                if args[0] != Sort(1) {
                    return Err(format!("{self}: condition of sort {}", args[0]));
                }
                same(args[1], args[2])?;
                args[1]
            }
            BitSelect { high, low } => {
                arity(1)?;
                if low > high || *high >= args[0].width() {
                    return Err(format!("{self}: out of range for sort {}", args[0]));
                }
                Sort(high - low + 1)
            }
            Extend { extra } => {
                arity(1)?;
                Sort(args[0].width() + extra)
            }
        };
        if sort.width() == 0 || sort.width() > MAX_WIDTH || args.iter().any(|a| a.width() == 0) {
            return Err(format!("{self}: a word of 0 or more than {MAX_WIDTH} bits"));
        }
        Ok(sort)
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
        Op::Id
    }
}

impl Differential<Sort> for Op {
    fn zero(range: &Sort) -> Self {
        Op::Zero(*range)
    }
}

/// The names of `theory::bv` (and `theory::any`).
impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Op::*;
        match self {
            Const { value, width } => write!(f, "{width}'d{value}"),
            Id => write!(f, "Id"),
            Havoc(_) => write!(f, "HAVOC"),
            Zero(_) => write!(f, "ZERO"),
            Not => write!(f, "Not"),
            And => write!(f, "And"),
            Or => write!(f, "Or"),
            Xor => write!(f, "Xor"),
            Neg => write!(f, "Neg"),
            Add => write!(f, "Add"),
            Sub => write!(f, "Sub"),
            Mul => write!(f, "Mul"),
            UDiv => write!(f, "UDiv"),
            UMod => write!(f, "UMod"),
            Eq => write!(f, "Eq"),
            Ne => write!(f, "Ne"),
            ULt => write!(f, "ULt"),
            ULe => write!(f, "ULe"),
            Ite => write!(f, "Ite"),
            BitSelect { high, low } => write!(f, "BitSelect[{high}:{low}]"),
            Extend { extra } => write!(f, "Extend(+{extra})"),
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

    #[test]
    fn sorts() {
        let w8 = Sort(8);
        assert_eq!(Op::Add.result_sort(&[w8, w8]), Ok(w8));
        assert_eq!(Op::ULt.result_sort(&[w8, w8]), Ok(Sort(1)));
        assert_eq!(Op::Ite.result_sort(&[Sort(1), w8, w8]), Ok(w8));
        assert!(Op::Ite.result_sort(&[Sort(2), w8, w8]).is_err());
        assert!(Op::Add.result_sort(&[w8, Sort(3)]).is_err());
        assert_eq!(Op::Extend { extra: 4 }.result_sort(&[w8]), Ok(Sort(12)));
        assert!(Op::Extend { extra: 60 }.result_sort(&[w8]).is_err());
        let select = |high, low| Op::BitSelect { high, low }.result_sort(&[w8]);
        assert_eq!(select(7, 4), Ok(Sort(4)));
        assert!(select(8, 4).is_err());
        assert!(select(3, 4).is_err());
        let num = |value, width| Op::Const { value, width }.result_sort(&[]);
        assert_eq!(num(255, 8), Ok(w8));
        assert!(num(256, 8).is_err());
        assert_eq!(num(u64::MAX, 64), Ok(Sort(64)));
        assert!(num(0, 65).is_err());
    }
}
