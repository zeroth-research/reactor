/*!
# Bit-vector values and the meaning of `SvOp`

A [`BitVec`] is a width `w ≥ 1` and a natural number below `2^w`, like
Lean's `BitVec w`. Arithmetic is modulo `2^w`; the signed ops read the
value in two's complement.
*/

use crate::theory::{Sort, SvOp};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, ToPrimitive, Zero};
use std::fmt;

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct BitVec {
    width: usize,
    value: BigUint,
}

/// `2^w - 1`: the value with all `w` bits set.
fn ones(width: usize) -> BigUint {
    (BigUint::one() << width) - BigUint::one()
}

impl BitVec {
    /// The value `value` of width `width`, or `None` if the width is 0 or
    /// the value does not fit.
    pub fn new(width: usize, value: impl Into<BigUint>) -> Option<Self> {
        let value = value.into();
        (width > 0 && value.bits() <= width as u64).then_some(Self { width, value })
    }

    /// The value `value mod 2^width`. Panics if the width is 0.
    pub fn wrap(width: usize, value: impl Into<BigUint>) -> Self {
        assert!(width > 0, "bit vector of width 0");
        Self {
            width,
            value: value.into() & ones(width),
        }
    }

    /// The two's complement encoding of `value` in `width` bits, modulo `2^width`.
    /// Panics if the width is 0.
    pub fn from_signed(width: usize, value: impl Into<BigInt>) -> Self {
        let modulus = BigInt::one() << width;
        let value = ((value.into() % &modulus) + &modulus) % &modulus;
        Self::wrap(width, value.to_biguint().expect("non-negative"))
    }

    pub fn zero(width: usize) -> Self {
        Self::wrap(width, BigUint::zero())
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn sort(&self) -> Sort {
        Sort(self.width)
    }

    /// The value as an unsigned number.
    pub fn unsigned(&self) -> &BigUint {
        &self.value
    }

    /// The value read in two's complement.
    pub fn signed(&self) -> BigInt {
        if self.msb() {
            BigInt::from(self.value.clone()) - (BigInt::one() << self.width)
        } else {
            BigInt::from(self.value.clone())
        }
    }

    /// The top bit.
    pub fn msb(&self) -> bool {
        self.value.bit(self.width as u64 - 1)
    }

    pub fn is_zero(&self) -> bool {
        self.value.is_zero()
    }

    fn bool(b: bool) -> Self {
        Self::wrap(1, b as u8)
    }
}

/// Verilog literal syntax: `8'd3`.
impl fmt::Display for BitVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}'d{}", self.width, self.value)
    }
}

/// The value of `op` applied to `args`, or why it is undefined: the
/// operands do not fit the op, the op is `Havoc`, which has no single
/// value, or it divides by zero.
pub fn eval(op: &SvOp, args: &[BitVec]) -> Result<BitVec, String> {
    use SvOp::*;
    let sorts: Vec<Sort> = args.iter().map(BitVec::sort).collect();
    let w = op.result_sort(&sorts)?.width();
    let u = |i: usize| args[i].unsigned();
    // a shift amount at least `w`, or too large for `usize`, shifts every bit out
    let amount = || u(1).to_usize().filter(|k| *k < w);

    let value = match op {
        Lit(v) => return Ok(v.clone()),
        Havoc(_) => return Err(format!("{op} has no single value")),
        Zero(_) => return Ok(BitVec::zero(w)),
        Id => return Ok(args[0].clone()),

        Not => u(0) ^ ones(w),
        And => u(0) & u(1),
        Or => u(0) | u(1),
        Xor => u(0) ^ u(1),

        Neg => (BigUint::one() << w) - u(0),
        Add => u(0) + u(1),
        Sub => (BigUint::one() << w) + u(0) - u(1),
        Mul => u(0) * u(1),
        UDiv | URem | SDiv | SRem if args[1].is_zero() => {
            return Err(format!("{op}: division by zero"));
        }
        UDiv => u(0) / u(1),
        URem => u(0) % u(1),
        // `BigInt` division rounds toward zero, the remainder has the sign
        // of the dividend
        SDiv => return Ok(BitVec::from_signed(w, args[0].signed() / args[1].signed())),
        SRem => return Ok(BitVec::from_signed(w, args[0].signed() % args[1].signed())),

        Eq => return Ok(BitVec::bool(u(0) == u(1))),
        Ne => return Ok(BitVec::bool(u(0) != u(1))),
        ULt => return Ok(BitVec::bool(u(0) < u(1))),
        ULe => return Ok(BitVec::bool(u(0) <= u(1))),
        SLt => return Ok(BitVec::bool(args[0].signed() < args[1].signed())),
        SLe => return Ok(BitVec::bool(args[0].signed() <= args[1].signed())),

        RedAnd => return Ok(BitVec::bool(*u(0) == ones(args[0].width()))),
        RedOr => return Ok(BitVec::bool(!args[0].is_zero())),
        RedXor => return Ok(BitVec::bool(u(0).count_ones() % 2 == 1)),

        Ite => {
            return Ok(if args[0].is_zero() {
                &args[2]
            } else {
                &args[1]
            }
            .clone());
        }

        Shl => amount().map_or(BigUint::zero(), |k| u(0) << k),
        LShr => amount().map_or(BigUint::zero(), |k| u(0) >> k),
        AShr => {
            let fill = if args[0].msb() {
                ones(w)
            } else {
                BigUint::zero()
            };
            match amount() {
                // the shifted-in top `k` bits come from `fill`
                Some(k) => (u(0) >> k) | (fill & (ones(w) ^ (ones(w) >> k))),
                None => fill,
            }
        }

        Extract { lo, .. } => u(0) >> *lo,
        Concat => (u(0) << args[1].width()) | u(1),
        ZExt(_) => u(0).clone(),
        SExt(_) => {
            if args[0].msb() {
                u(0) | (ones(w) ^ ones(args[0].width()))
            } else {
                u(0).clone()
            }
        }
    };
    Ok(BitVec::wrap(w, value))
}
