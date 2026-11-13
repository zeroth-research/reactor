/*!
# Bit-vectors

Defines the theory [`BV`] of matrices of bit-vectors.

A [`Sort`] value is either `BV<N>(rows, cols)`, a matrix of `N`-bit bit-vectors,
or `Zero(rows, cols)`, the sort of their (zero) derivatives.

The operations carry their parameters (the width of the bit-vectors and the shapes
of the matrices), so that each operation has a fixed [`Boundary`]: e.g., the boundary
of `Add(8, 2, 3)` is `BV<8>(2, 3) ⊗ BV<8>(2, 3) -> BV<8>(2, 3)`. The parameters that
can be ill-formed are checked on their creation (the literals are [`BVConst`], and the
selected bits are [`Bits`]), so the boundary of every operation exists. The operations
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

An operation applies to the wires whose sorts are its boundary: the read wires
have the sorts of its domain, and the written wires the sorts of its codomain.

## Examples

```
use base::boundary::Boundary;
use theory::bv::BV;

let boundary = |op: BV| format!("{} -> {}", op.dom(), op.cod());

// Matrix multiply of 8-bit bit-vectors: (2x3) * (3x4) -> (2x4).
assert_eq!(
    boundary(BV::MatMul(8, 2, 3, 4)),
    "BV<8>(2, 3) ⊗ BV<8>(3, 4) -> BV<8>(2, 4)"
);

// Elementwise Add reads and writes matrices of one shape.
assert_eq!(
    boundary(BV::Add(8, 2, 3)),
    "BV<8>(2, 3) ⊗ BV<8>(2, 3) -> BV<8>(2, 3)"
);
```
*/

use crate::*;
use base::boundary::{Boundary, Mono, Poly};
use base::tangent::Tangent;
#[cfg(feature = "pyo3")]
use pyo3::{exceptions::PyValueError, prelude::*};
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

    /// The bit-width, `0` for the derivatives
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
    Const(BVConst),
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
    /// Element-wise extract the bits `bits` of bit-vectors
    /// TODO: rename to `extract`, matching MathSAT
    BitSelect {
        bits: Bits,
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
            BV::Const(k) => write!(f, "{}", k.tensor()),
            BV::BitSelect { bits, .. } => write!(f, "{name}[{}:{}]", bits.high(), bits.low()),
            BV::Extend { extra, .. } => write!(f, "{name}(+{extra})"),
            BV::Uninterpreted(x, ..) => write!(f, "{name}({x})"),
            _ => f.write_str(name),
        }
    }
}

impl Sequential<Sort> for BV {
    fn skip(range: &Sort) -> Self {
        // a zero derivative cannot be copied: the check of the term with `Zero` fails
        let [r, c] = *range.shape();
        BV::Id(range.bw(), r, c)
    }
}

impl Combinatorial<Sort> for BV {
    fn havoc(range: &Sort) -> Self {
        // a derivative cannot be havoced: the check of the term with `Zero` fails
        let [r, c] = *range.shape();
        BV::Havoc(range.bw(), r, c)
    }
}

impl Boundary<Sort> for BV {
    fn dom(&self) -> Poly<Sort> {
        let bv = |w, r, c| Sort::BV(w, [r, c]);
        let sorts = match *self {
            BV::Const(..) => vec![],
            BV::Not(w, r, c) | BV::Neg(w, r, c) | BV::Abs(w, r, c) | BV::Id(w, r, c) => {
                vec![bv(w, r, c)]
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
            | BV::Ne(w, r, c) => vec![bv(w, r, c), bv(w, r, c)],
            BV::MatMul(w, m, k, n) => vec![bv(w, m, k), bv(w, k, n)],
            BV::BVToBool(w, r, c) => vec![bv(w, r, c)],
            BV::BitSelect { bits, rows, cols } => vec![bv(bits.width(), rows, cols)],
            BV::Extend {
                width, rows, cols, ..
            } => vec![bv(width, rows, cols)],
            BV::Ite(w, r, c) => vec![bv(1, 1, 1), bv(w, r, c), bv(w, r, c)],
            BV::Uninterpreted(..) | BV::Havoc(..) => vec![],
        };
        sorts.into_iter().product::<Mono<Sort>>().into()
    }

    fn cod(&self) -> Poly<Sort> {
        let bv = |w, r, c| Sort::BV(w, [r, c]);
        let sort = match *self {
            BV::Const(ref k) => {
                let [r, c] = k.shape();
                bv(k.width(), r, c)
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
            BV::BitSelect { bits, rows, cols } => bv(bits.count(), rows, cols),
            BV::Extend {
                extra,
                width,
                rows,
                cols,
            } => bv(width + extra, rows, cols),
            BV::Ite(w, r, c) | BV::Uninterpreted(_, w, r, c) | BV::Havoc(w, r, c) => bv(w, r, c),
        };
        [sort].into_iter().product::<Mono<Sort>>().into()
    }
}

/// A constant matrix of bit-vectors of the width `width` (at most 64): a 2D tensor of the
/// values that fit into (unsigned) bit-vectors of the width
#[derive(Clone, Debug)]
#[cfg_attr(feature = "pyo3", pyclass(frozen))]
pub struct BVConst {
    matrix: Matrix,
    width: usize,
}

impl BVConst {
    /// The constant `tensor` of bit-vectors of the width `width`, or an error if `tensor`
    /// is not 2D, its values do not fit into the width, or the width is over 64
    pub fn new(tensor: crate::PyTensor, width: usize) -> Result<Self, String> {
        let matrix = Matrix::new(tensor)?;
        check_fits(matrix.tensor(), width)?;
        Ok(Self { matrix, width })
    }

    pub fn tensor(&self) -> &crate::PyTensor {
        self.matrix.tensor()
    }

    /// The `[rows, cols]` shape of the matrix
    pub fn shape(&self) -> [usize; 2] {
        self.matrix.shape()
    }

    /// The width of the bit-vectors
    pub fn width(&self) -> usize {
        self.width
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl BVConst {
    #[new]
    fn py_new(tensor: crate::PyTensor, width: usize) -> PyResult<Self> {
        Self::new(tensor, width).map_err(PyValueError::new_err)
    }

    #[getter(tensor)]
    fn py_tensor(&self) -> crate::PyTensor {
        self.tensor().clone()
    }

    #[getter(width)]
    fn py_width(&self) -> usize {
        self.width()
    }

    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("tensor", "width")
    }

    fn __repr__(&self) -> String {
        format!("BVConst({}, {})", self.tensor(), self.width())
    }
}

/// Check that the values of the tensor `cm` fit into (unsigned) bit-vectors of the width `bw`
/// (at most 64)
fn check_fits(cm: &crate::PyTensor, bw: usize) -> Result<(), String> {
    if bw > 64 {
        return Err("Support at most 64-bit atm.".into());
    }
    // the minimum and the maximum of an empty tensor do not exist
    if bw < 64 && cm.numel() > 0 {
        let max = (1i64 << bw) - 1;
        let min_val = cm.min().int64_value(&[]);
        let max_val = cm.max().int64_value(&[]);
        if min_val < 0 || max_val > max {
            return Err(format!("tensor values do not fit in {bw} bits"));
        }
    }
    Ok(())
}

/// The bits `[high..=low]` (inclusive) of bit-vectors of the width `width`
/// (`low <= high < width`)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "pyo3", pyclass(frozen, eq))]
pub struct Bits {
    high: usize,
    low: usize,
    width: usize,
}

impl Bits {
    /// The bits `[high..=low]` of bit-vectors of the width `width`, or an error if
    /// `high < low` or `high >= width`
    pub fn new(high: usize, low: usize, width: usize) -> Result<Self, String> {
        if high < low {
            return Err(format!("high ({high}) must be >= low ({low})"));
        }
        if high >= width {
            return Err(format!("high ({high}) is out of range for width {width}"));
        }
        Ok(Self { high, low, width })
    }

    pub fn high(&self) -> usize {
        self.high
    }

    pub fn low(&self) -> usize {
        self.low
    }

    /// The width of the bit-vectors the bits are selected from
    pub fn width(&self) -> usize {
        self.width
    }

    /// The number of the bits (the width of the selection)
    pub fn count(&self) -> usize {
        self.high - self.low + 1
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl Bits {
    #[new]
    fn py_new(high: usize, low: usize, width: usize) -> PyResult<Self> {
        Self::new(high, low, width).map_err(PyValueError::new_err)
    }

    #[getter(high)]
    fn py_high(&self) -> usize {
        self.high()
    }

    #[getter(low)]
    fn py_low(&self) -> usize {
        self.low()
    }

    #[getter(width)]
    fn py_width(&self) -> usize {
        self.width()
    }

    #[classattr]
    fn __match_args__() -> (&'static str, &'static str, &'static str) {
        ("high", "low", "width")
    }

    fn __repr__(&self) -> String {
        format!(
            "Bits(high={}, low={}, width={})",
            self.high, self.low, self.width
        )
    }
}

#[cfg(all(test, feature = "torch"))]
mod tests {
    use super::*;

    fn bv(bw: usize, r: usize, c: usize) -> Sort {
        Sort::BV(bw, [r, c])
    }

    fn int_tensor(rows: i64, cols: i64) -> crate::PyTensor {
        tch::Tensor::zeros([rows, cols], (tch::Kind::Int64, tch::Device::Cpu)).into()
    }

    fn bv_const(t: crate::PyTensor, width: usize) -> BV {
        BV::Const(BVConst::new(t, width).unwrap())
    }

    fn bit_select(high: usize, low: usize, width: usize) -> BV {
        let bits = Bits::new(high, low, width).unwrap();
        BV::BitSelect {
            bits,
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

    // Formats the boundary of `op` as `dom -> cod`
    fn show(op: BV) -> String {
        format!("{} -> {}", op.dom(), op.cod())
    }

    #[test]
    fn type_shape_and_bw() {
        let t = bv(8, 2, 3);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.bw(), 8);

        let t = bv(16, 1, 4);
        assert_eq!(t.shape(), &[1, 4]);
        assert_eq!(t.bw(), 16);
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
        // the width is given by the operation
        assert_eq!(show(BV::Id(32, 4, 4)), "BV<32>(4, 4) -> BV<32>(4, 4)");
        // the comparisons write bits
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
        // SMV `bool(...)`, also of bits
        assert_eq!(show(BV::BVToBool(8, 2, 3)), format!("{m} -> BV<1>(2, 3)"));
        assert_eq!(show(BV::BVToBool(1, 1, 1)), "BV<1>(1, 1) -> BV<1>(1, 1)");
        // the guard is a single bit, as in LIA and LRA
        assert_eq!(
            show(BV::Ite(8, 2, 3)),
            format!("BV<1>(1, 1) ⊗ {m} ⊗ {m} -> {m}")
        );
        // a single write of the given sort
        assert_eq!(
            show(BV::Uninterpreted("f".into(), 8, 2, 3)),
            format!("I -> {m}")
        );
        assert_eq!(show(BV::Havoc(8, 2, 3)), format!("I -> {m}"));
        assert_eq!(show(BV::Havoc(16, 2, 3)), "I -> BV<16>(2, 3)");
    }

    #[test]
    fn boundary_const() {
        // a constant writes a matrix of the shape of its tensor and of its width
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[0i64, 1], [2, 3]]).into();
        assert_eq!(show(bv_const(cm, 8)), "I -> BV<8>(2, 2)");
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[255i64]]).into();
        assert_eq!(show(bv_const(cm, 8)), "I -> BV<8>(1, 1)");
        assert_eq!(show(bv_const(int_tensor(2, 3), 16)), "I -> BV<16>(2, 3)");
    }

    #[test]
    fn boundary_bits() {
        // BV<16>[11:4] -> BV<8>
        assert_eq!(show(bit_select(11, 4, 16)), "BV<16>(1, 1) -> BV<8>(1, 1)");
        let single = BV::BitSelect {
            bits: Bits::new(3, 3, 8).unwrap(),
            rows: 2,
            cols: 2,
        };
        assert_eq!(show(single), "BV<8>(2, 2) -> BV<1>(2, 2)");
        // zero-extension by `extra` bits
        assert_eq!(show(extend(8, 8)), "BV<8>(1, 1) -> BV<16>(1, 1)");
        assert_eq!(show(extend(4, 8)), "BV<8>(1, 1) -> BV<12>(1, 1)");
    }

    #[test]
    fn skip_and_havoc() {
        let t = bv(8, 2, 3);
        assert_eq!(show(BV::skip(&t)), "BV<8>(2, 3) -> BV<8>(2, 3)");
        assert_eq!(show(BV::havoc(&t)), "I -> BV<8>(2, 3)");
        // derivatives cannot be copied or havoced: `skip` and `havoc` of them are on
        // values (of no bits)
        let z = t.tangent();
        assert_eq!(show(BV::skip(&z)), "BV<0>(2, 3) -> BV<0>(2, 3)");
        assert_eq!(show(BV::havoc(&z)), "I -> BV<0>(2, 3)");
    }

    #[test]
    fn derivatives() {
        // the derivatives of bit-vectors are zero; the operations of BV are on values only
        // (see `boundary_ops`)
        let z = bv(8, 2, 3).tangent();
        assert_eq!(z, Sort::Zero([2, 3]));
        assert_eq!(z.tangent(), z);
    }

    #[test]
    fn ill_formed_parameters() {
        // the parameters are checked on their creation (so the boundaries always exist)
        fn err<T>(r: Result<T, String>) -> String {
            r.err().unwrap()
        }
        assert_eq!(err(Bits::new(3, 4, 8)), "high (3) must be >= low (4)");
        assert_eq!(
            err(Bits::new(8, 0, 8)),
            "high (8) is out of range for width 8"
        );
        let t: crate::PyTensor = tch::Tensor::from_slice(&[0i64, 1]).into();
        assert_eq!(err(BVConst::new(t, 8)), "expected a 2D tensor, got 1D");
        let t: crate::PyTensor = tch::Tensor::from_slice2(&[[256i64]]).into();
        assert_eq!(
            err(BVConst::new(t, 8)),
            "tensor values do not fit in 8 bits"
        );
        let t: crate::PyTensor = tch::Tensor::from_slice2(&[[-1i64]]).into();
        assert_eq!(
            err(BVConst::new(t, 8)),
            "tensor values do not fit in 8 bits"
        );
        assert_eq!(
            err(BVConst::new(int_tensor(1, 1), 65)),
            "Support at most 64-bit atm."
        );
        // an empty constant fits into any width
        assert_eq!(BVConst::new(int_tensor(0, 3), 8).unwrap().shape(), [0, 3]);
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
}
