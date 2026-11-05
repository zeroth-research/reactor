/*!
# Bit-vectors

Defines the theory [`BV`] of matrices of bit-vectors.

A [`Sort`] value is either `BV<N>(rows, cols)`, a matrix of `N`-bit bit-vectors,
or `Zero(rows, cols)`, the sort of their (zero) derivatives.

The operations carry their parameters (the width of the bit-vectors and the shapes
of the matrices), so that each operation has a fixed [`Boundary`]: e.g., the boundary
of `Add(8, 2, 3)` is `BV<8>(2, 3) ⊗ BV<8>(2, 3) -> BV<8>(2, 3)`. The operations
in [`BV`] are:

- [`BV::Const`] — a bit-vector matrix literal whose values must fit into its width.
- [`BV::Not`], [`BV::Neg`], [`BV::Abs`] — unary, pointwise.
- [`BV::And`], [`BV::Or`], [`BV::Xor`], [`BV::Add`], [`BV::Sub`], [`BV::Mul`],
  [`BV::UDiv`], [`BV::SDiv`], [`BV::UMod`], [`BV::SMod`] — binary, pointwise;
  both inputs and the output have the same sort.
- [`BV::ULe`], [`BV::ULt`], [`BV::UGe`], [`BV::UGt`], [`BV::SLe`], [`BV::SLt`],
  [`BV::SGe`], [`BV::SGt`], [`BV::Eq`], [`BV::Ne`] — pointwise comparisons
  producing a matrix of bits `BV<1>`.
- [`BV::MatMul`] — matrix multiplication: `(m,k) × (k,n) → (m,n)`.
- [`BV::BitSelect`], [`BV::Extend`] — pointwise extraction of bits and zero-extension.
- [`BV::Ite`] — if-then-else: reads a bit guard and two same-typed branches.

`BV` implements [`Theory`]; [`Theory::check`] checks that the sorts of the
read and written wires are the domain and the codomain of the operation.

## Examples

```
use theory::Theory;
use theory::bv::{BV, Sort};

// 8-bit bit-vectors.
let a = Sort::BV(8, [2, 3]);
let b = Sort::BV(8, [3, 4]);
let c = Sort::BV(8, [2, 4]);

let ok = |s| Ok::<_, String>(s);

// Matrix multiply: (2x3) * (3x4) -> (2x4).
assert!(BV::MatMul(8, 2, 3, 4).check([a, b].map(ok), [c].map(ok)).is_ok());

// Elementwise Add requires matching shapes.
let m = Sort::BV(8, [2, 3]);
assert!(BV::Add(8, 2, 3).check([m, m].map(ok), [m].map(ok)).is_ok());
assert!(BV::Add(8, 2, 3).check([a, b].map(ok), [c].map(ok)).is_err());
```
*/

use crate::*;
use base::boundary::{Boundary, Poly};
use base::tangent::Tangent;
#[cfg(feature = "pyo3")]
use pyo3::pyclass;
use std::fmt;

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub enum Sort {
    // matrix of bitvectors determined by (bitvector-length, [# rows, # cols])
    BV(usize, [usize; 2]),
    // the tangent of `BV` (it has only the zero derivative)
    Zero([usize; 2]),
}

impl Sort {
    pub fn shape(&self) -> &[usize; 2] {
        match self {
            Sort::BV(_, shape) | Sort::Zero(shape) => shape,
        }
    }

    /// The bit-width, `0` for the derivatives (zero-width bit-vectors have the only value 0)
    pub fn bw(&self) -> usize {
        match self {
            Sort::BV(bw, _) => *bw,
            Sort::Zero(_) => 0,
        }
    }

    /// Is this sort a tangent (i.e., a sort of derivatives)?
    pub fn is_tangent(&self) -> bool {
        matches!(self, Sort::Zero(..))
    }
}

impl fmt::Display for Sort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sort::BV(bw, [i, j]) => write!(f, "BV<{bw}>({i}, {j})"),
            Sort::Zero([i, j]) => write!(f, "Zero({i}, {j})"),
        }
    }
}

impl Tangent for Sort {
    fn tangent(&self) -> Self {
        Sort::Zero(*self.shape())
    }
}

/// Theory of bitvector matrices. Operations on bitvectors follow the SMT-LIB2 semantics.
/// The parameters `(width, rows, cols)` are the width of the bit-vectors and the shape of
/// the matrices the operation applies to (e.g., of its operands).
#[derive(Clone, Debug, strum::IntoStaticStr)]
#[cfg_attr(feature = "pyo3", pyclass(frozen))]
pub enum BV {
    /// Create constant BV matrix of the given width from a given 2-D tensor
    Const(crate::PyTensor, usize),
    /// Element-wise addition of two BV matrices (arithmetic modulo `2^bitwidth`)
    Add(usize, usize, usize),
    /// Element-wise subtraction of two BV matrices (arithmetic modulo `2^bitwidth`)
    Sub(usize, usize, usize),
    /// Element-wise multiplication of two BV matrices (arithmetic modulo `2^bitwidth`)
    Mul(usize, usize, usize),
    /// Element-wise **unsigned** division of two BV matrices (arithmetic modulo `2^bitwidth`)
    UDiv(usize, usize, usize),
    /// Element-wise **signed** division of two BV matrices . Arithmetic modulo `2^bitwidth`.
    /// Edge cases follow SMT-LIB2 specification (e.g., `-128 / -1 = -128` on 8 bits and
    /// `x / 0 = 2^(bitwidth - 1)`)
    SDiv(usize, usize, usize),
    /// Element-wise **unsigned** modulo of two BV matrices
    UMod(usize, usize, usize),
    /// Element-wise **signed** modulo of two BV matrices
    SMod(usize, usize, usize),
    /// Element-wise negation of two BV matrices (arithmetic modulo `2^bitwidth`).
    /// Edge case follows the SMT-LIB2 specification (e.g., `neg(-128) = -128` on 8 bits)
    Neg(usize, usize, usize),
    /// XXX: remove from here and move the modelling to Python.
    /// XXX What to do on `Abs(BV<n>_MIN)`? Should the result be `BV<n>_MAX`?
    /// (so that `Abs` always returns positive value?)
    /// Element-wise absolute value. Modeled as `Abs(x) = Ite(x < 0, Neg(x), x)`
    Abs(usize, usize, usize),
    /// Two bitvector matrices multiplication `(m,k) × (k,n) → (m,n)`, with the parameters
    /// `(width, m, k, n)`. Modeled through multiplication and sums as usual.
    MatMul(usize, usize, usize, usize),
    /// Element-wise bit-wise "and"
    And(usize, usize, usize),
    /// Element-wise bit-wise "or"
    Or(usize, usize, usize),
    /// Element-wise bit-wise "xor"
    Xor(usize, usize, usize),
    /// Element-wise bit-wise "not" (bit inversion)
    Not(usize, usize, usize),
    /// Element-wise **unsigned** less-or-equal comparison (result is a matrix of BV<1> values)
    ULe(usize, usize, usize),
    /// Element-wise **unsigned** less-than comparison (result is a matrix of BV<1> values)
    ULt(usize, usize, usize),
    /// Element-wise **unsigned** greater-or-equal comparison (result is a matrix of BV<1> values)
    UGe(usize, usize, usize),
    /// Element-wise **unsigned** greater-than comparison (result is a matrix of BV<1> values)
    UGt(usize, usize, usize),
    /// Element-wise **signed** less-or-equal comparison (result is a matrix of BV<1> values)
    SLe(usize, usize, usize),
    /// Element-wise **signed** less-than comparison (result is a matrix of BV<1> values)
    SLt(usize, usize, usize),
    /// Element-wise **signed** greater-or-equal comparison (result is a matrix of BV<1> values)
    SGe(usize, usize, usize),
    /// Element-wise **signed** greater-than comparison (result is a matrix of BV<1> values)
    SGt(usize, usize, usize),
    /// Element-wise equality comparison (result is a matrix of BV<1> values)
    Eq(usize, usize, usize),
    /// Element-wise non-equality comparison (result is a matrix of BV<1> values)
    Ne(usize, usize, usize),
    /// If-then-else construct, condition is `BV<1>`
    Ite(usize, usize, usize),
    /// Identity on bitvector matrices
    Id(usize, usize, usize),
    /// Element-wise BV<n> to BV<1> via "non-zero comparison" (`x != 0`). Corresponds to SMV ToBool,
    /// FIXME: to be removed and replaced by `Ne x 0`
    BVToBool(usize, usize, usize),
    /// Element-wise extract bits `[high..=low]` (inclusive) of bit-vectors of the width `width`
    /// TODO: rename to `extract`, matching MathSAT
    BitSelect {
        high: usize,
        low: usize,
        width: usize,
        rows: usize,
        cols: usize,
    },
    /// Element-wise zero-extend bit-vectors of the width `width` by `extra` bits
    Extend {
        extra: usize,
        width: usize,
        rows: usize,
        cols: usize,
    },
    /// Uninterpreted value (written, used by the builder for untraced calls)
    Uninterpreted(String, usize, usize, usize),
    Havoc(usize, usize, usize),
}

/// Prints the name of the operation, the tensors of the literals, the bits of
/// [`BV::BitSelect`] and [`BV::Extend`], and the names of the uninterpreted values
/// (the other parameters are given by the wires)
impl fmt::Display for BV {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name: &'static str = self.into();
        match self {
            BV::Const(t, _) => write!(f, "{t}"),
            BV::BitSelect { high, low, .. } => write!(f, "{name}[{high}:{low}]"),
            BV::Extend { extra, .. } => write!(f, "{name}(+{extra})"),
            BV::Uninterpreted(x, ..) => write!(f, "{name}({x})"),
            _ => f.write_str(name),
        }
    }
}

impl Sequential for BV {
    fn skip(range: &Self::Sort) -> Self {
        // a zero derivative cannot be copied: the check of the term with `Zero` fails
        let [r, c] = *range.shape();
        BV::Id(range.bw(), r, c)
    }
}

impl Combinatorial for BV {
    fn havoc(range: &Self::Sort) -> Self {
        // a derivative cannot be havoced: the check of the term with `Zero` fails
        let [r, c] = *range.shape();
        BV::Havoc(range.bw(), r, c)
    }
}

impl Theory for BV {
    type Sort = Sort;
    const NAME: &'static str = "BV";

    fn check<R, W, E: fmt::Display>(&self, read: R, write: W) -> Result<(), String>
    where
        R: IntoIterator<Item = Result<Sort, E>>,
        W: IntoIterator<Item = Result<Sort, E>>,
    {
        try_cod(self)
            .and_then(|cod| check_wires(self.dom(), cod, read, write))
            .map_err(|e| format!("{self:?}: {e}"))
    }
}

impl Boundary<Sort> for BV {
    fn dom(&self) -> Poly<Sort> {
        let bv = |w, r, c| Sort::BV(w, [r, c]);
        match *self {
            BV::Const(..) => Poly::unit(),
            BV::Not(w, r, c) | BV::Neg(w, r, c) | BV::Abs(w, r, c) | BV::Id(w, r, c) => {
                prod([bv(w, r, c)])
            }
            BV::Add(w, r, c)
            | BV::Sub(w, r, c)
            | BV::Mul(w, r, c)
            | BV::UDiv(w, r, c)
            | BV::SDiv(w, r, c)
            | BV::UMod(w, r, c)
            | BV::SMod(w, r, c)
            | BV::And(w, r, c)
            | BV::Or(w, r, c)
            | BV::Xor(w, r, c)
            | BV::ULe(w, r, c)
            | BV::ULt(w, r, c)
            | BV::UGe(w, r, c)
            | BV::UGt(w, r, c)
            | BV::SLe(w, r, c)
            | BV::SLt(w, r, c)
            | BV::SGe(w, r, c)
            | BV::SGt(w, r, c)
            | BV::Eq(w, r, c)
            | BV::Ne(w, r, c) => prod([bv(w, r, c), bv(w, r, c)]),
            BV::MatMul(w, m, k, n) => prod([bv(w, m, k), bv(w, k, n)]),
            BV::BVToBool(w, r, c) => prod([bv(w, r, c)]),
            BV::BitSelect {
                width, rows, cols, ..
            }
            | BV::Extend {
                width, rows, cols, ..
            } => prod([bv(width, rows, cols)]),
            BV::Ite(w, r, c) => prod([bv(1, 1, 1), bv(w, r, c), bv(w, r, c)]),
            BV::Uninterpreted(..) | BV::Havoc(..) => Poly::unit(),
        }
    }

    /// # Panics
    ///
    /// If the operation is ill-formed: the tensor of [`BV::Const`] is not 2D or its values
    /// do not fit into the width, or the bits of [`BV::BitSelect`] are out of range
    /// ([`Theory::check`] reports these as errors).
    fn cod(&self) -> Poly<Sort> {
        try_cod(self).unwrap_or_else(|e| panic!("{self:?}: {e}"))
    }
}

/// The codomain of `op`: the sort of its output, or an error if `op` is ill-formed
fn try_cod(op: &BV) -> Result<Poly<Sort>, String> {
    let bv = |w, r, c| Sort::BV(w, [r, c]);
    let sort = match *op {
        BV::Const(ref cm, w) => {
            let [r, c] = shape_2d(cm)?;
            check_fits(cm, w)?;
            bv(w, r, c)
        }
        // TODO: for `Abs`, what if input is the signed min of BV<N>? It's absolute value
        // does not fit into BV<N>
        BV::Not(w, r, c) | BV::Neg(w, r, c) | BV::Abs(w, r, c) | BV::Id(w, r, c) => bv(w, r, c),
        BV::Add(w, r, c)
        | BV::Sub(w, r, c)
        | BV::Mul(w, r, c)
        | BV::UDiv(w, r, c)
        | BV::SDiv(w, r, c)
        | BV::UMod(w, r, c)
        | BV::SMod(w, r, c)
        | BV::And(w, r, c)
        | BV::Or(w, r, c)
        | BV::Xor(w, r, c) => bv(w, r, c),
        BV::ULe(_, r, c)
        | BV::ULt(_, r, c)
        | BV::UGe(_, r, c)
        | BV::UGt(_, r, c)
        | BV::SLe(_, r, c)
        | BV::SLt(_, r, c)
        | BV::SGe(_, r, c)
        | BV::SGt(_, r, c)
        | BV::Eq(_, r, c)
        | BV::Ne(_, r, c) => bv(1, r, c),
        BV::MatMul(w, m, _, n) => bv(w, m, n),
        BV::BVToBool(_, r, c) => bv(1, r, c),
        BV::BitSelect {
            high,
            low,
            width,
            rows,
            cols,
        } => {
            if high < low {
                return Err(format!("high ({high}) must be >= low ({low})"));
            }
            if high >= width {
                return Err(format!("high ({high}) is out of range for width {width}"));
            }
            bv(high - low + 1, rows, cols)
        }
        BV::Extend {
            extra,
            width,
            rows,
            cols,
        } => bv(width + extra, rows, cols),
        BV::Ite(w, r, c) | BV::Uninterpreted(_, w, r, c) | BV::Havoc(w, r, c) => bv(w, r, c),
    };
    Ok(prod([sort]))
}

/// Check that the values of the tensor `cm` fit into (unsigned) bit-vectors of the width `bw`
fn check_fits(cm: &crate::PyTensor, bw: usize) -> Result<(), String> {
    if bw > 64 {
        return Err("Support at most 64-bit atm.".into());
    }
    if bw < 64 {
        let max = (1i64 << bw) - 1;
        let min_val = cm.min().int64_value(&[]);
        let max_val = cm.max().int64_value(&[]);
        if min_val < 0 || max_val > max {
            return Err(format!("tensor values do not fit in {bw} bits"));
        }
    }
    Ok(())
}

#[cfg(all(test, feature = "torch"))]
mod tests {
    use super::*;

    fn bv(bw: usize, r: usize, c: usize) -> Sort {
        Sort::BV(bw, [r, c])
    }

    pub fn ok<S>(s: S) -> Result<S, String> {
        Ok(s)
    }

    fn int_tensor(rows: i64, cols: i64) -> crate::PyTensor {
        tch::Tensor::zeros([rows, cols], (tch::Kind::Int64, tch::Device::Cpu)).into()
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

    // --- Const ---

    #[test]
    fn const_ok() {
        let t = bv(8, 2, 2);
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[0i64, 1], [2, 3]]).into();
        assert!(BV::Const(cm, 8).check([], [t].map(ok)).is_ok());
    }

    #[test]
    fn const_value_overflow_fails() {
        let t = bv(8, 1, 1);
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[256i64]]).into();
        assert!(BV::Const(cm, 8).check(vec![], [t].map(ok)).is_err());
    }

    #[test]
    fn const_value_max_fits() {
        let t = bv(8, 1, 1);
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[255i64]]).into();
        assert!(BV::Const(cm, 8).check(vec![], [t].map(ok)).is_ok());
    }

    #[test]
    fn const_wrong_width_fails() {
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[1i64]]).into();
        assert!(
            BV::Const(cm, 8)
                .check(vec![], [bv(16, 1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn const_wrong_row_count_fails() {
        let t = bv(8, 2, 2);
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[0i64, 1]]).into();
        assert!(BV::Const(cm, 8).check(vec![], [t].map(ok)).is_err());
    }

    #[test]
    fn const_wrong_col_count_fails() {
        let t = bv(8, 1, 2);
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[0i64]]).into();
        assert!(BV::Const(cm, 8).check(vec![], [t].map(ok)).is_err());
    }

    #[test]
    fn const_with_read_fails() {
        let t = bv(8, 1, 1);
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[0i64]]).into();
        assert!(BV::Const(cm, 8).check([t].map(ok), [t].map(ok)).is_err());
    }

    // --- Not / Id ---

    #[test]
    fn not_ok() {
        let t = bv(8, 2, 3);
        assert!(BV::Not(8, 2, 3).check([t].map(ok), [t].map(ok)).is_ok());
    }

    #[test]
    fn not_type_mismatch_fails() {
        assert!(
            BV::Not(8, 1, 1)
                .check([bv(8, 1, 1)].map(ok), [bv(8, 2, 2)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn id_ok() {
        let t = bv(32, 4, 4);
        assert!(BV::Id(32, 4, 4).check([t].map(ok), [t].map(ok)).is_ok());
        // the width is given by the operation
        assert!(BV::Id(16, 4, 4).check([t].map(ok), [t].map(ok)).is_err());
    }

    // --- Binary elementwise (Add, Mul, And, Or, Xor) ---

    #[test]
    fn add_ok() {
        let t = bv(8, 3, 3);
        assert!(BV::Add(8, 3, 3).check([t, t].map(ok), [t].map(ok)).is_ok());
    }

    #[test]
    fn add_shape_mismatch_fails() {
        assert!(
            BV::Add(8, 1, 2)
                .check([bv(8, 1, 2), bv(8, 2, 1)].map(ok), [bv(8, 1, 2)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn mul_ok() {
        let t = bv(16, 1, 1);
        assert!(BV::Mul(16, 1, 1).check([t, t].map(ok), [t].map(ok)).is_ok());
    }

    #[test]
    fn and_ok() {
        let t = bv(1, 2, 2);
        assert!(BV::And(1, 2, 2).check([t, t].map(ok), [t].map(ok)).is_ok());
    }

    #[test]
    fn or_ok() {
        let t = bv(1, 2, 2);
        assert!(BV::Or(1, 2, 2).check([t, t].map(ok), [t].map(ok)).is_ok());
    }

    #[test]
    fn xor_ok() {
        let t = bv(4, 1, 1);
        assert!(BV::Xor(4, 1, 1).check([t, t].map(ok), [t].map(ok)).is_ok());
    }

    // --- Comparisons ---

    #[test]
    fn lt_ok() {
        let t = bv(8, 2, 3);
        let out = bv(1, 2, 3);
        assert!(
            BV::ULt(8, 2, 3)
                .check([t, t].map(ok), [out].map(ok))
                .is_ok()
        );
        assert!(
            BV::SLt(8, 2, 3)
                .check([t, t].map(ok), [out].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn le_ok() {
        let t = bv(8, 1, 1);
        let out = bv(1, 1, 1);
        assert!(
            BV::ULe(8, 1, 1)
                .check([t, t].map(ok), [out].map(ok))
                .is_ok()
        );
        assert!(
            BV::SLe(8, 1, 1)
                .check([t, t].map(ok), [out].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn eq_ok() {
        let t = bv(32, 1, 1);
        let out = bv(1, 1, 1);
        assert!(
            BV::Eq(32, 1, 1)
                .check([t, t].map(ok), [out].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn ne_ok() {
        let t = bv(8, 1, 1);
        let out = bv(1, 1, 1);
        assert!(BV::Ne(8, 1, 1).check([t, t].map(ok), [out].map(ok)).is_ok());
    }

    #[test]
    fn cmp_wrong_output_type_fails() {
        let t = bv(8, 2, 3);
        // output must be U(1, rows, cols), not U(8, ...)
        assert!(BV::ULt(8, 2, 3).check([t, t].map(ok), [t].map(ok)).is_err());
        assert!(BV::SLt(8, 2, 3).check([t, t].map(ok), [t].map(ok)).is_err());
    }

    #[test]
    fn cmp_input_mismatch_fails() {
        let out = bv(1, 1, 1);
        assert!(
            BV::Eq(8, 1, 1)
                .check([bv(8, 1, 1), bv(16, 1, 1)].map(ok), [out].map(ok))
                .is_err()
        );
    }

    // --- UDiv / SDiv ---

    #[test]
    fn udiv_ok() {
        let t = bv(8, 1, 1);
        assert!(BV::UDiv(8, 1, 1).check([t, t].map(ok), [t].map(ok)).is_ok());
    }

    #[test]
    fn sdiv_ok() {
        let t = bv(8, 1, 1);
        assert!(BV::SDiv(8, 1, 1).check([t, t].map(ok), [t].map(ok)).is_ok());
    }

    // --- MatMul ---

    #[test]
    fn matmul_ok() {
        let a = bv(8, 2, 3);
        let b = bv(8, 3, 4);
        let c = bv(8, 2, 4);
        assert!(
            BV::MatMul(8, 2, 3, 4)
                .check([a, b].map(ok), [c].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn matmul_inner_dim_mismatch_fails() {
        let a = bv(8, 2, 3);
        let b = bv(8, 4, 4);
        let c = bv(8, 2, 4);
        assert!(
            BV::MatMul(8, 2, 3, 4)
                .check([a, b].map(ok), [c].map(ok))
                .is_err()
        );
    }

    #[test]
    fn matmul_bw_mismatch_fails() {
        let a = bv(8, 2, 3);
        let b = bv(16, 3, 4);
        let c = bv(8, 2, 4);
        assert!(
            BV::MatMul(8, 2, 3, 4)
                .check([a, b].map(ok), [c].map(ok))
                .is_err()
        );
    }

    // --- Ite ---

    #[test]
    fn ite_ok() {
        let cond = bv(1, 1, 1);
        let t = bv(8, 1, 1);
        assert!(
            BV::Ite(8, 1, 1)
                .check([cond, t, t].map(ok), [t].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn ite_bad_condition_bw_fails() {
        let cond = bv(8, 1, 1);
        let t = bv(8, 1, 1);
        assert!(
            BV::Ite(8, 1, 1)
                .check([cond, t, t].map(ok), [t].map(ok))
                .is_err()
        );
    }

    // --- Sub / Neg / UMod / SMod ---

    #[test]
    fn sub_ok() {
        let t = bv(8, 2, 3);
        assert!(BV::Sub(8, 2, 3).check([t, t].map(ok), [t].map(ok)).is_ok());
    }

    #[test]
    fn neg_ok() {
        let t = bv(8, 2, 3);
        assert!(BV::Neg(8, 2, 3).check([t].map(ok), [t].map(ok)).is_ok());
    }

    #[test]
    fn neg_type_mismatch_fails() {
        assert!(
            BV::Neg(8, 1, 1)
                .check([bv(8, 1, 1)].map(ok), [bv(8, 2, 2)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn umod_ok() {
        let t = bv(8, 1, 1);
        assert!(BV::UMod(8, 1, 1).check([t, t].map(ok), [t].map(ok)).is_ok());
    }

    #[test]
    fn smod_ok() {
        let t = bv(8, 1, 1);
        assert!(BV::SMod(8, 1, 1).check([t, t].map(ok), [t].map(ok)).is_ok());
    }

    // --- BVToBool / BoolToBV (SMV bool(...) / word1(...)) ---

    #[test]
    fn bv_to_bool_ok() {
        // BV<8>(2,3) -> BV<1>(2,3)
        assert!(
            BV::BVToBool(8, 2, 3)
                .check([bv(8, 2, 3)].map(ok), [bv(1, 2, 3)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn bv_to_bool_identity_width_ok() {
        // BV<1> -> BV<1> is also fine (idempotent on already-bool input).
        assert!(
            BV::BVToBool(1, 1, 1)
                .check([bv(1, 1, 1)].map(ok), [bv(1, 1, 1)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn bv_to_bool_wrong_out_width_fails() {
        assert!(
            BV::BVToBool(8, 1, 1)
                .check([bv(8, 1, 1)].map(ok), [bv(8, 1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn bv_to_bool_shape_mismatch_fails() {
        assert!(
            BV::BVToBool(8, 2, 3)
                .check([bv(8, 2, 3)].map(ok), [bv(1, 1, 1)].map(ok))
                .is_err()
        );
    }

    // --- BitSelect / Extend ---

    fn bit_select(high: usize, low: usize, width: usize) -> BV {
        BV::BitSelect {
            high,
            low,
            width,
            rows: 1,
            cols: 1,
        }
    }

    fn extend(extra: usize, width: usize) -> BV {
        BV::Extend {
            extra,
            width,
            rows: 1,
            cols: 1,
        }
    }

    #[test]
    fn bit_select_ok() {
        // BV<16>[11:4] -> BV<8>
        assert!(
            bit_select(11, 4, 16)
                .check([bv(16, 1, 1)].map(ok), [bv(8, 1, 1)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn bit_select_single_bit_ok() {
        let op = BV::BitSelect {
            high: 3,
            low: 3,
            width: 8,
            rows: 2,
            cols: 2,
        };
        assert!(
            op.check([bv(8, 2, 2)].map(ok), [bv(1, 2, 2)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn bit_select_out_of_range_fails() {
        assert!(
            bit_select(8, 0, 8)
                .check([bv(8, 1, 1)].map(ok), [bv(9, 1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn bit_select_wrong_out_width_fails() {
        assert!(
            bit_select(7, 0, 16)
                .check([bv(16, 1, 1)].map(ok), [bv(7, 1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn extend_ok() {
        assert!(
            extend(8, 8)
                .check([bv(8, 1, 1)].map(ok), [bv(16, 1, 1)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn extend_wrong_out_width_fails() {
        assert!(
            extend(4, 8)
                .check([bv(8, 1, 1)].map(ok), [bv(16, 1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn ite_arm_mismatch_fails() {
        let cond = bv(1, 1, 1);
        assert!(
            BV::Ite(8, 1, 1)
                .check(
                    [cond, bv(8, 1, 1), bv(16, 1, 1)].map(ok),
                    [bv(8, 1, 1)].map(ok)
                )
                .is_err()
        );
    }

    #[test]
    fn ite_matrix_guard_fails() {
        // the guard is a single bit, as in LIA and LRA
        let t = bv(8, 2, 2);
        assert!(
            BV::Ite(8, 2, 2)
                .check([bv(1, 2, 2), t, t].map(ok), [t].map(ok))
                .is_err()
        );
    }

    // --- Boundaries ---

    // Formats the boundary of `op` as `dom -> cod`
    fn show(op: BV) -> String {
        format!("{} -> {}", op.dom(), op.cod())
    }

    #[test]
    fn boundary_ops() {
        let m = "BV<8>(2, 3)";
        for op in [
            BV::Add(8, 2, 3),
            BV::Sub(8, 2, 3),
            BV::Mul(8, 2, 3),
            BV::UDiv(8, 2, 3),
            BV::SDiv(8, 2, 3),
            BV::UMod(8, 2, 3),
            BV::SMod(8, 2, 3),
            BV::And(8, 2, 3),
            BV::Or(8, 2, 3),
            BV::Xor(8, 2, 3),
        ] {
            assert_eq!(show(op), format!("{m} ⊗ {m} -> {m}"));
        }
        for op in [
            BV::Not(8, 2, 3),
            BV::Neg(8, 2, 3),
            BV::Abs(8, 2, 3),
            BV::Id(8, 2, 3),
        ] {
            assert_eq!(show(op), format!("{m} -> {m}"));
        }
        for op in [
            BV::ULe(8, 2, 3),
            BV::ULt(8, 2, 3),
            BV::UGe(8, 2, 3),
            BV::UGt(8, 2, 3),
            BV::SLe(8, 2, 3),
            BV::SLt(8, 2, 3),
            BV::SGe(8, 2, 3),
            BV::SGt(8, 2, 3),
            BV::Eq(8, 2, 3),
            BV::Ne(8, 2, 3),
        ] {
            assert_eq!(show(op), format!("{m} ⊗ {m} -> BV<1>(2, 3)"));
        }
        assert_eq!(
            show(BV::MatMul(8, 2, 3, 4)),
            "BV<8>(2, 3) ⊗ BV<8>(3, 4) -> BV<8>(2, 4)"
        );
        assert_eq!(show(BV::BVToBool(8, 2, 3)), format!("{m} -> BV<1>(2, 3)"));
        assert_eq!(show(bit_select(11, 4, 16)), "BV<16>(1, 1) -> BV<8>(1, 1)");
        assert_eq!(show(extend(8, 8)), "BV<8>(1, 1) -> BV<16>(1, 1)");
        assert_eq!(
            show(BV::Ite(8, 2, 3)),
            format!("BV<1>(1, 1) ⊗ {m} ⊗ {m} -> {m}")
        );
        assert_eq!(
            show(BV::Uninterpreted("f".into(), 8, 2, 3)),
            format!("I -> {m}")
        );
        assert_eq!(show(BV::Havoc(8, 2, 3)), format!("I -> {m}"));
        assert_eq!(show(BV::Const(int_tensor(2, 3), 8)), format!("I -> {m}"));
        assert_ne!(show(BV::Havoc(8, 2, 3)), show(BV::Havoc(16, 2, 3)));
    }

    #[test]
    #[should_panic(expected = "high (3) must be >= low (4)")]
    fn boundary_bit_select_ill_formed() {
        let _ = show(bit_select(3, 4, 8));
    }

    #[test]
    fn display() {
        assert_eq!(BV::Add(8, 2, 3).to_string(), "Add");
        assert_eq!(bit_select(7, 0, 8).to_string(), "BitSelect[7:0]");
        assert_eq!(extend(8, 8).to_string(), "Extend(+8)");
        assert_eq!(
            BV::Uninterpreted("f".into(), 8, 2, 3).to_string(),
            "Uninterpreted(f)"
        );
    }

    #[test]
    fn check_errors() {
        assert_eq!(
            BV::Add(8, 1, 1).check([bv(8, 1, 1), bv(16, 1, 1)].map(ok), [bv(8, 1, 1)].map(ok)),
            Err(
                "Add(8, 1, 1): expected `BV<8>(1, 1) ⊗ BV<8>(1, 1) -> BV<8>(1, 1)`, \
                 got `BV<8>(1, 1) ⊗ BV<16>(1, 1) -> BV<8>(1, 1)`"
                    .to_string()
            )
        );
        let t = bv(8, 1, 1);
        assert_eq!(
            BV::Not(8, 1, 1).check([t, t].map(ok), [t].map(ok)),
            Err("Not(8, 1, 1): expected `BV<8>(1, 1) -> BV<8>(1, 1)`, \
                 got `BV<8>(1, 1) ⊗ BV<8>(1, 1) -> BV<8>(1, 1)`"
                .to_string())
        );
        // errors of wires are propagated
        assert_eq!(
            BV::Not(8, 1, 1).check([Err("bad wire")], [t].map(Ok)),
            Err("Not(8, 1, 1): bad wire".to_string())
        );
    }

    #[test]
    fn check_ill_formed_ops() {
        // ill-formed ops are reported, not panicking
        let t: crate::PyTensor = tch::Tensor::from_slice(&[0i64, 1]).into();
        let res = BV::Const(t, 8).check([].map(ok), [bv(8, 1, 2)].map(ok));
        assert!(res.unwrap_err().ends_with("expected a 2D tensor, got 1D"));
        let res = BV::Const(int_tensor(1, 1), 65).check([].map(ok), [bv(65, 1, 1)].map(ok));
        assert!(res.unwrap_err().ends_with("Support at most 64-bit atm."));
        let res = bit_select(3, 4, 8).check([bv(8, 1, 1)].map(ok), [bv(1, 1, 1)].map(ok));
        assert!(res.unwrap_err().ends_with("high (3) must be >= low (4)"));
        let res = bit_select(8, 0, 8).check([bv(8, 1, 1)].map(ok), [bv(9, 1, 1)].map(ok));
        assert!(
            res.unwrap_err()
                .ends_with("high (8) is out of range for width 8")
        );
    }

    #[test]
    fn check_uninterpreted() {
        let f = || BV::Uninterpreted("f".into(), 8, 2, 3);
        let t = bv(8, 2, 3);
        // a single write of the given sort
        assert!(f().check([].map(ok), [t].map(ok)).is_ok());
        assert!(f().check([].map(ok), [bv(1, 2, 3)].map(ok)).is_err());
        assert!(f().check([t].map(ok), [].map(ok)).is_err());
        assert!(f().check([t].map(ok), [t].map(ok)).is_err());
        assert!(f().check([].map(ok), [t, t].map(ok)).is_err());
        assert!(f().check([].map(ok), [].map(ok)).is_err());
    }

    #[test]
    fn check_derivatives() {
        // the derivatives of bit-vectors are zero
        let z = bv(8, 2, 3).tangent();
        assert_eq!(z, Sort::Zero([2, 3]));
        assert_eq!(z.tangent(), z);
        // operations on values do not apply to derivatives
        let t = bv(8, 2, 3);
        assert!(BV::Add(8, 2, 3).check([z, z].map(ok), [z].map(ok)).is_err());
        assert!(BV::Add(8, 2, 3).check([t, z].map(ok), [t].map(ok)).is_err());
        assert!(BV::Not(8, 2, 3).check([z].map(ok), [t].map(ok)).is_err());
        assert!(BV::Havoc(8, 2, 3).check([].map(ok), [z].map(ok)).is_err());
        assert!(BV::havoc(&z).check([].map(ok), [z].map(ok)).is_err());
        assert!(BV::skip(&z).check([z].map(ok), [z].map(ok)).is_err());
        assert!(
            BV::Const(int_tensor(2, 3), 8)
                .check([].map(ok), [z].map(ok))
                .is_err()
        );
        assert!(BV::Id(8, 2, 3).check([t].map(ok), [z].map(ok)).is_err());
        // ... but values can be copied and havoced
        assert!(BV::skip(&t).check([t].map(ok), [t].map(ok)).is_ok());
        assert!(BV::havoc(&t).check([].map(ok), [t].map(ok)).is_ok());
    }
}
