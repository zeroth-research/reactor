/*!
# Bit-vectors of fixed length

Defines the theory [`BV<N>`] of matrices of `N`-bit bit-vectors, where
the width `N` is fixed at compile time via a const generic.

A [`BVType<N>`] value describes the *type* of a term as a matrix shape
`BV(rows, cols)` whose elements are `N`-bit bit-vectors. The operations
in [`BV`] are:

- [`BV::Const`] — an inline bit-vector matrix literal whose shape must
  match the declared write type.
- [`BV::Not`] — unary, shape-preserving.
- [`BV::And`], [`BV::Or`], [`BV::Xor`], [`BV::Add`], [`BV::Mul`] —
  elementwise binary; both inputs and the output share the same shape.
- [`BV::MatMul`] — matrix multiplication: `(m,k) × (k,n) → (m,n)`, with
  the inner dimensions required to agree.
*/

use crate::*;
#[cfg(feature = "pyo3")]
use pyo3::pyclass;

use crate::bv::BV::{Havoc, Id};
use std::fmt;
use std::fmt::Debug;

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub enum Sort {
    // matrix of bitvectors determined by (bitvector-length, [# rows, # cols])
    BV(usize, [usize; 2]),
}

impl Sort {
    pub fn shape(&self) -> &[usize; 2] {
        match self {
            Sort::BV(_, shape) => shape,
        }
    }

    pub fn bw(&self) -> usize {
        match self {
            Sort::BV(bw, _) => *bw,
        }
    }
}

impl fmt::Display for Sort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sort::BV(bw, [i, j]) => write!(f, "BV<{bw}>({i}, {j})"),
        }
    }
}

/// Theory of bitvector matrices. Operations on bitvectors follow the SMT-LIB2 semantics.
#[derive(Clone, Debug, strum::Display)]
#[cfg_attr(feature = "pyo3", pyclass(frozen))]
pub enum BV {
    /// Create constant BV matrix from a given 2-D tensor
    #[strum(to_string = "{0}")]
    Const(crate::PyTensor),
    /// Element-wise addition of two BV matrices (arithmetic modulo `2^bitwidth`)
    Add(),
    /// Element-wise subtraction of two BV matrices (arithmetic modulo `2^bitwidth`)
    Sub(),
    /// Element-wise multiplication of two BV matrices (arithmetic modulo `2^bitwidth`)
    Mul(),
    /// Element-wise **unsigned** division of two BV matrices (arithmetic modulo `2^bitwidth`)
    UDiv(),
    /// Element-wise **signed** division of two BV matrices . Arithmetic modulo `2^bitwidth`.
    /// Edge cases follow SMT-LIB2 specification (e.g., `-128 / -1 = -128` on 8 bits and
    /// `x / 0 = 2^(bitwidth - 1)`)
    SDiv(),
    /// Element-wise **unsigned** modulo of two BV matrices
    UMod(),
    /// Element-wise **signed** modulo of two BV matrices
    SMod(),
    /// Element-wise negation of two BV matrices (arithmetic modulo `2^bitwidth`).
    /// Edge case follows the SMT-LIB2 specification (e.g., `neg(-128) = -128` on 8 bits)
    Neg(),
    /// XXX: remove from here and move the modelling to Python.
    /// XXX What to do on `Abs(BV<n>_MIN)`? Should the result be `BV<n>_MAX`?
    /// (so that `Abs` always returns positive value?)
    /// Element-wise absolute value. Modeled as `Abs(x) = Ite(x < 0, Neg(x), x)`
    Abs(),
    /// Two bitvector matrices multiplication. Modeled through multiplication and sums
    /// as usual.
    MatMul(),
    /// Element-wise bit-wise "and"
    And(),
    /// Element-wise bit-wise "or"
    Or(),
    /// Element-wise bit-wise "xor"
    Xor(),
    /// Element-wise bit-wise "not" (bit inversion)
    Not(),
    /// Element-wise **unsigned** less-or-equal comparison (result is a matrix of BV<1> values)
    ULe(),
    /// Element-wise **unsigned** less-than comparison (result is a matrix of BV<1> values)
    ULt(),
    /// Element-wise **unsigned** greater-or-equal comparison (result is a matrix of BV<1> values)
    UGe(),
    /// Element-wise **unsigned** greater-than comparison (result is a matrix of BV<1> values)
    UGt(),
    /// Element-wise **signed** less-or-equal comparison (result is a matrix of BV<1> values)
    SLe(),
    /// Element-wise **signed** less-than comparison (result is a matrix of BV<1> values)
    SLt(),
    /// Element-wise **signed** greater-or-equal comparison (result is a matrix of BV<1> values)
    SGe(),
    /// Element-wise **signed** greater-than comparison (result is a matrix of BV<1> values)
    SGt(),
    /// Element-wise equality comparison (result is a matrix of BV<1> values)
    Eq(),
    /// Element-wise non-equality comparison (result is a matrix of BV<1> values)
    Ne(),
    /// If-then-else construct, condition is `BV<1>`
    Ite(),
    /// Identity on bitvector matrices
    Id(),
    /// Element-wise BV<n> to BV<1> via "non-zero comparison" (`x != 0`). Corresponds to SMV ToBool,
    /// FIXME: to be removed and replaced by `Ne x 0`
    BVToBool(),
    /// Element-wise extract bits `[high..=low]` (inclusive)
    /// TODO: rename to `extract`, matching MathSAT
    #[strum(to_string = "BitSelect[{high}:{low}]")]
    BitSelect {
        high: usize,
        low: usize,
    },
    /// Element-wise zero-extend by `extra` bits
    #[strum(to_string = "Extend(+{extra})")]
    Extend {
        extra: usize,
    },
    /// Uninterpreted symbol
    #[strum(to_string = "Uninterpreted({0})")]
    Uninterpreted(String),
    Havoc(usize, [usize; 2]),
}

impl Sequential<Sort> for BV {
    fn skip(_range: &Sort) -> Self {
        Id()
    }
}

impl Combinatorial<Sort> for BV {
    fn havoc(range: &Sort) -> Self {
        Havoc(range.bw(), *range.shape())
    }
}

#[cfg(all(test, feature = "torch"))]
mod tests {
    use super::*;

    fn bv(bw: usize, r: usize, c: usize) -> Sort {
        Sort::BV(bw, [r, c])
    }

    // --- Type helpers ---

    #[test]
    fn type_shape_and_bw() {
        let t = bv(8, 2, 3);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.bw(), 8);

        let t = bv(16, 1, 4);
        assert_eq!(t.shape(), &[1, 4]);
        assert_eq!(t.bw(), 16);
    }
}
