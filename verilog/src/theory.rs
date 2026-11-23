/*!
# The op set `SvOp`

Ops over 2-state bit vectors, with the meaning of SystemVerilog's
operators on equal-width operands. A [`Sort`] is a bit width only;
signedness lives in the ops (`ULt`/`SLt`, `LShr`/`AShr`, `ZExt`/`SExt`).

Every op has exactly one write wire, and the sort of that wire is a
function of the op and its operand sorts ([`SvOp::result_sort`]). Ops that
change the width carry the parameters they need for that (`Extract`,
`ZExt`, `SExt`, and the sort of `Lit`, `Havoc`, `Zero`).
[`SvOp::check`] computes the result sort and compares it with the write
wire; [`crate::term`] runs it on every term the translation builds.

The meaning of the ops is [`crate::value::eval`].
*/

use crate::value::BitVec;
use ::theory::{Combinatorial, Differential, Sequential};
use std::fmt;

/// The sort of a wire: a bit vector of the given width (at least 1).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct Sort(pub usize);

impl Sort {
    pub fn width(&self) -> usize {
        self.0
    }
}

impl fmt::Display for Sort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bv{}", self.0)
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum SvOp {
    /// `[] → w`: the constant.
    Lit(BitVec),
    /// `w → w`: copy (the `SKIP` of [`Sequential`]).
    Id,
    /// `[] → w`: any value (the `HAVOC` of [`Combinatorial`]).
    Havoc(Sort),
    /// `[] → w`, writing a derivative: zero (the `ZERO` of [`Differential`]).
    Zero(Sort),

    /// `w → w`: bitwise not.
    Not,
    /// `w, w → w`: bitwise and.
    And,
    /// `w, w → w`: bitwise or.
    Or,
    /// `w, w → w`: bitwise xor.
    Xor,

    /// `w → w`: negation modulo `2^w`.
    Neg,
    /// `w, w → w`: addition modulo `2^w`.
    Add,
    /// `w, w → w`: subtraction modulo `2^w`.
    Sub,
    /// `w, w → w`: multiplication modulo `2^w`.
    Mul,
    /// `w, w → w`: unsigned division, rounding down; undefined for a zero
    /// divisor.
    UDiv,
    /// `w, w → w`: unsigned remainder; undefined for a zero divisor.
    URem,
    /// `w, w → w`: signed (two's complement) division, rounding toward
    /// zero, modulo `2^w` (the most negative value divided by -1 is
    /// itself); undefined for a zero divisor.
    SDiv,
    /// `w, w → w`: signed remainder, with the sign of the dividend;
    /// undefined for a zero divisor.
    SRem,

    /// `w, w → 1`: equality.
    Eq,
    /// `w, w → 1`: disequality.
    Ne,
    /// `w, w → 1`: unsigned less-than.
    ULt,
    /// `w, w → 1`: unsigned less-or-equal.
    ULe,
    /// `w, w → 1`: signed (two's complement) less-than.
    SLt,
    /// `w, w → 1`: signed (two's complement) less-or-equal.
    SLe,

    /// `w → 1`: and of all bits.
    RedAnd,
    /// `w → 1`: or of all bits (the operand is non-zero).
    RedOr,
    /// `w → 1`: xor of all bits (parity).
    RedXor,

    /// `1, w, w → w`: the second read if the condition is 1, else the third.
    Ite,

    /// `w, k → w`: shift left by an unsigned amount; `≥ w` gives 0.
    Shl,
    /// `w, k → w`: logical shift right by an unsigned amount; `≥ w` gives 0.
    LShr,
    /// `w, k → w`: arithmetic shift right by an unsigned amount, filling
    /// with the top bit; `≥ w` gives all copies of the top bit.
    AShr,

    /// `w → hi-lo+1`: bits `hi` down to `lo`, with `lo ≤ hi < w`.
    Extract { hi: usize, lo: usize },
    /// `a, b → a+b`: the first read gives the high bits.
    Concat,
    /// `w → to`: zero extension, with `to ≥ w`.
    ZExt(usize),
    /// `w → to`: sign extension, with `to ≥ w`.
    SExt(usize),
}

impl SvOp {
    /// The sort of the write wire, given the sorts of the read wires, or why
    /// the op does not apply to them.
    pub fn result_sort(&self, args: &[Sort]) -> Result<Sort, String> {
        use SvOp::*;
        if let Some(s) = args.iter().find(|s| s.width() == 0) {
            return Err(format!("{self}: operand of sort {s}"));
        }
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
            Lit(v) => {
                arity(0)?;
                Sort(v.width())
            }
            Havoc(s) | Zero(s) => {
                arity(0)?;
                *s
            }
            Id | Not | Neg => {
                arity(1)?;
                args[0]
            }
            And | Or | Xor | Add | Sub | Mul | UDiv | URem | SDiv | SRem => {
                arity(2)?;
                same(args[0], args[1])?;
                args[0]
            }
            Eq | Ne | ULt | ULe | SLt | SLe => {
                arity(2)?;
                same(args[0], args[1])?;
                Sort(1)
            }
            RedAnd | RedOr | RedXor => {
                arity(1)?;
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
            Shl | LShr | AShr => {
                arity(2)?;
                args[0]
            }
            Extract { hi, lo } => {
                arity(1)?;
                if lo > hi || *hi >= args[0].width() {
                    return Err(format!("{self}: out of range for sort {}", args[0]));
                }
                Sort(hi - lo + 1)
            }
            Concat => {
                arity(2)?;
                Sort(args[0].width() + args[1].width())
            }
            ZExt(to) | SExt(to) => {
                arity(1)?;
                if *to < args[0].width() {
                    return Err(format!("{self}: narrows sort {}", args[0]));
                }
                Sort(*to)
            }
        };
        if sort.width() == 0 {
            return Err(format!("{self}: result of sort {sort}"));
        }
        Ok(sort)
    }

    /// Checks a term of this op: the operand sorts must be valid for the op
    /// ([`SvOp::result_sort`]), and there must be exactly one write wire,
    /// of the result sort.
    pub fn check(&self, read: &[Sort], write: &[Sort]) -> Result<(), String> {
        let expected = self.result_sort(read)?;
        match write {
            [w] if *w == expected => Ok(()),
            [w] => Err(format!("{self}: writes sort {w}, expected sort {expected}")),
            _ => Err(format!("{self}: expects 1 write, got {}", write.len())),
        }
    }
}

impl Combinatorial<Sort> for SvOp {
    fn havoc(range: &Sort) -> Self {
        SvOp::Havoc(*range)
    }
}

impl Sequential<Sort> for SvOp {
    fn skip(_range: &Sort) -> Self {
        SvOp::Id
    }
}

impl Differential<Sort> for SvOp {
    fn zero(range: &Sort) -> Self {
        SvOp::Zero(*range)
    }
}

impl fmt::Display for SvOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use SvOp::*;
        match self {
            Lit(v) => write!(f, "{v}"),
            Id => write!(f, "ID"),
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
            Eq => write!(f, "EQ"),
            Ne => write!(f, "NE"),
            ULt => write!(f, "ULT"),
            ULe => write!(f, "ULE"),
            SLt => write!(f, "SLT"),
            SLe => write!(f, "SLE"),
            RedAnd => write!(f, "REDAND"),
            RedOr => write!(f, "REDOR"),
            RedXor => write!(f, "REDXOR"),
            Ite => write!(f, "ITE"),
            Shl => write!(f, "SHL"),
            LShr => write!(f, "LSHR"),
            AShr => write!(f, "ASHR"),
            Extract { hi, lo } => write!(f, "EXTRACT[{hi}:{lo}]"),
            Concat => write!(f, "CONCAT"),
            ZExt(to) => write!(f, "ZEXT[{to}]"),
            SExt(to) => write!(f, "SEXT[{to}]"),
        }
    }
}
