/*!
# Fixed-size bit-vectors

Defines the sort [`Word`] of matrices of `width`-bit machine words, the local
sort [`Sort`] of the theory, holding words and the booleans that
comparisons write, and one type per operation on them. Each operation is
an operation of the SMT-LIB2 logic QF_BV, lifted pointwise to matrices:
it applies the SMT-LIB2 operation independently to each entry, with the
same semantics, and its documentation gives the SMT-LIB2 term it
computes on one entry.

Every operation carries enough information to compute its [`Boundary`]:
pointwise operations carry the `width` and the shape `[m, n]` of their
arguments, as `Op(width, m, n)`, and indexed operations put their indices
first, as SMT-LIB2 does with `(_ op i)`. The operations whose meaning
depends on signedness are in the submodules [`signed`], reading words in
two's complement, and [`unsigned`], reading them in binary.

| Operation | SMT-LIB2 | | Operation | SMT-LIB2 |
|---|---|---|---|---|
| [`Const`] | `#b…`, `(_ bvN width)` | | [`signed::Div`] | `bvsdiv` |
| [`Concat`] | `concat` | | [`signed::Rem`] | `bvsrem` |
| [`Extract`] | `(_ extract i j)` | | [`signed::Mod`] | `bvsmod` |
| [`Repeat`] | `(_ repeat i)` | | [`signed::Shr`] | `bvashr` |
| [`Not`] | `bvnot` | | [`signed::Extend`] | `(_ sign_extend i)` |
| [`And`] | `bvand` | | [`signed::Lt`] | `bvslt` |
| [`Or`] | `bvor` | | [`signed::Le`] | `bvsle` |
| [`Xor`] | `bvxor` | | [`signed::Gt`] | `bvsgt` |
| [`Nand`] | `bvnand` | | [`signed::Ge`] | `bvsge` |
| [`Nor`] | `bvnor` | | [`unsigned::Div`] | `bvudiv` |
| [`Xnor`] | `bvxnor` | | [`unsigned::Rem`] | `bvurem` |
| [`Neg`] | `bvneg` | | [`unsigned::Shr`] | `bvlshr` |
| [`Add`] | `bvadd` | | [`unsigned::Extend`] | `(_ zero_extend i)` |
| [`Sub`] | `bvsub` | | [`unsigned::Lt`] | `bvult` |
| [`Mul`] | `bvmul` | | [`unsigned::Le`] | `bvule` |
| [`Shl`] | `bvshl` | | [`unsigned::Gt`] | `bvugt` |
| [`RotateLeft`] | `(_ rotate_left i)` | | [`unsigned::Ge`] | `bvuge` |
| [`RotateRight`] | `(_ rotate_right i)` | | [`signed::Abs`] | — |
| [`Eq`](struct@Eq) | `=` | | | |
| [`Ne`] | `distinct` | | | |
| [`Ite`] | `ite` | | | |

`=` and `distinct` belong to the SMT-LIB2 Core theory, which every logic
includes, at the sort `(_ BitVec width)`; all the others belong to the
theory FixedSizeBitVectors or are defined by the logic QF_BV. The
comparisons, `=` and `distinct` write booleans, as in SMT-LIB2.
*/

use super::sort::{Bool, Word, check_width};
use crate::Tensor;
use base::boundary::{Boundary, Poly};
use base::mono;
use derive_more::From;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use std::fmt;

/// The sorts of the theory, as in the signature of SMT-LIB2's
/// FixedSizeBitVectors: words, and the booleans of the Core theory, which
/// comparisons write, together with [`Zero`](crate::tensor::sort::Zero), the tangent of both. It is
/// the subset [`tensor::sort::Words`](crate::tensor::sort::Words) of the sorts of all
/// theories.
pub use super::sort::Words as Sort;

/// A constant matrix literal of `width`-bit words, as `Const(tensor, width)`, held
/// in a tensor of type `T`.
///
/// SMT-LIB2: a literal `#b…` or `(_ bvN width)`.
#[derive(Debug)]
pub struct Const<T> {
    tensor: T,
    width: usize,
}

impl<T: Tensor> Const<T> {
    /// The constant held in `tensor`, writing the words of `cod`; fails
    /// unless `tensor` is of its shape.
    pub fn with_cod(tensor: T, cod: [Word; 1]) -> Result<Self, crate::Error> {
        let [x] = cod;
        if tensor.shape() != x.shape() {
            return Err(crate::tensor::unexpected_cod(
                &format!("1 sort of the shape {:?} of the constant", tensor.shape()),
                &cod,
            ));
        }
        Self::new(tensor, x.width())
    }
}

impl<T> Const<T> {
    /// The constant of `width`-bit words held in `tensor`; fails unless
    /// `width` is positive.
    pub fn new(tensor: T, width: usize) -> Result<Self, crate::Error> {
        check_width(width)?;
        Ok(Const { tensor, width })
    }

    /// The tensor holding the constant.
    pub fn tensor(&self) -> &T {
        &self.tensor
    }

    /// The width of the words.
    pub fn width(&self) -> usize {
        self.width
    }
}

/// Pointwise concatenation `x ++ y` of a `u`-bit word `x`, the high bits
/// of the result, and a `v`-bit word `y`, its low bits, as
/// `Concat([u, v], [m, n])`; the result has `u + v` bits.
///
/// SMT-LIB2: `(concat x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.word")
)]
pub struct Concat {
    /// The widths `[u, v]` of the words read, high bits first.
    ///
    /// Entry `i` belongs to input `i`, as indexed in the domain of the
    /// boundary: an array rather than one field per input, so that the
    /// two match by position.
    widths: [usize; 2],
    shape: [usize; 2],
}

/// Pointwise extraction of the bits `i` down to `j`, both included, from
/// `width`-bit words, as `Extract(i, j, width, [m, n])`; it requires
/// `j ≤ i < width`, and the result has `i - j + 1` bits.
///
/// SMT-LIB2: `((_ extract i j) x)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.word")
)]
pub struct Extract {
    i: usize,
    j: usize,
    width: usize,
    shape: [usize; 2],
}

/// Pointwise concatenation of `i ≥ 1` copies of `width`-bit words, as
/// `Repeat(i, width, [m, n])`; the result has `i·width` bits.
///
/// SMT-LIB2: `((_ repeat i) x)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.word")
)]
pub struct Repeat {
    i: usize,
    width: usize,
    shape: [usize; 2],
}

/// Pointwise bitwise negation `!x`, as `Not(width, [m, n])`.
///
/// SMT-LIB2: `(bvnot x)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.word")
)]
pub struct Not {
    width: usize,
    shape: [usize; 2],
}

/// Pointwise bitwise conjunction `x & y`, as `And(width, [m, n])`.
///
/// SMT-LIB2: `(bvand x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.word")
)]
pub struct And {
    width: usize,
    shape: [usize; 2],
}

/// Pointwise bitwise disjunction `x | y`, as `Or(width, [m, n])`.
///
/// SMT-LIB2: `(bvor x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.word")
)]
pub struct Or {
    width: usize,
    shape: [usize; 2],
}

/// Pointwise bitwise exclusive disjunction `x ^ y`, as `Xor(width, [m, n])`.
///
/// SMT-LIB2: `(bvxor x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.word")
)]
pub struct Xor {
    width: usize,
    shape: [usize; 2],
}

/// Pointwise bitwise negated conjunction `!(x & y)`, as `Nand(width, [m, n])`.
///
/// SMT-LIB2: `(bvnand x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.word")
)]
pub struct Nand {
    width: usize,
    shape: [usize; 2],
}

/// Pointwise bitwise negated disjunction `!(x | y)`, as `Nor(width, [m, n])`.
///
/// SMT-LIB2: `(bvnor x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.word")
)]
pub struct Nor {
    width: usize,
    shape: [usize; 2],
}

/// Pointwise bitwise negated exclusive disjunction `!(x ^ y)`, as
/// `Xnor(width, [m, n])`.
///
/// SMT-LIB2: `(bvxnor x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.word")
)]
pub struct Xnor {
    width: usize,
    shape: [usize; 2],
}

/// Pointwise two's-complement negation `-x`, modulo `2^width`, as
/// `Neg(width, [m, n])`; the minimum signed word is its own negation.
///
/// SMT-LIB2: `(bvneg x)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.word")
)]
pub struct Neg {
    width: usize,
    shape: [usize; 2],
}

/// Pointwise choice `if c then x else y` between `[m, n]` matrices of `width`-bit
/// words `x` and `y`, by the boolean matrix `c`, read first, as `Ite(width, [m, n])`.
///
/// SMT-LIB2: `(ite c x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.word")
)]
pub struct Ite {
    width: usize,
    shape: [usize; 2],
}

/// Pointwise addition `x + y`, modulo `2^width`, as `Add(width, [m, n])`.
///
/// SMT-LIB2: `(bvadd x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.word")
)]
pub struct Add {
    width: usize,
    shape: [usize; 2],
}

/// Pointwise subtraction `x - y`, modulo `2^width`, as `Sub(width, [m, n])`.
///
/// SMT-LIB2: `(bvsub x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.word")
)]
pub struct Sub {
    width: usize,
    shape: [usize; 2],
}

/// Pointwise multiplication `x * y`, modulo `2^width`, as `Mul(width, [m, n])`.
///
/// SMT-LIB2: `(bvmul x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.word")
)]
pub struct Mul {
    width: usize,
    shape: [usize; 2],
}

/// Pointwise left shift `x << y`, filling with zeros, as `Shl(width, [m, n])`.
///
/// SMT-LIB2: `(bvshl x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.word")
)]
pub struct Shl {
    width: usize,
    shape: [usize; 2],
}

/// Pointwise rotation left by a constant `i` bits, as
/// `RotateLeft(i, width, [m, n])`.
///
/// SMT-LIB2: `((_ rotate_left i) x)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.word")
)]
pub struct RotateLeft {
    i: usize,
    width: usize,
    shape: [usize; 2],
}

/// Pointwise rotation right by a constant `i` bits, as
/// `RotateRight(i, width, [m, n])`.
///
/// SMT-LIB2: `((_ rotate_right i) x)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.word")
)]
pub struct RotateRight {
    i: usize,
    width: usize,
    shape: [usize; 2],
}

/// Pointwise equality `x = y` of `width`-bit words, as `Eq(width, [m, n])`.
///
/// SMT-LIB2: `(= x y)`, of the Core theory.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.word")
)]
pub struct Eq {
    width: usize,
    shape: [usize; 2],
}

/// Pointwise disequality `x ≠ y` of `width`-bit words, as `Ne(width, [m, n])`.
///
/// SMT-LIB2: `(distinct x y)`, of the Core theory.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.word")
)]
pub struct Ne {
    width: usize,
    shape: [usize; 2],
}

/// The operations of the theory of fixed-size bit-vectors, SMT-LIB2's
/// `FixedSizeBitVectors`, with their boundary in [`Sort`]. Those of
/// [`signed`] and [`unsigned`] are named after their SMT-LIB2 operation,
/// as `SDiv` for [`signed::Div`] and `UDiv` for [`unsigned::Div`].
#[derive(Debug, From)]
pub enum BV<T> {
    Const(Const<T>),
    Concat(Concat),
    Extract(Extract),
    Repeat(Repeat),
    Not(Not),
    And(And),
    Or(Or),
    Xor(Xor),
    Nand(Nand),
    Nor(Nor),
    Xnor(Xnor),
    Neg(Neg),
    Ite(Ite),
    Add(Add),
    Sub(Sub),
    Mul(Mul),
    Shl(Shl),
    RotateLeft(RotateLeft),
    RotateRight(RotateRight),
    Eq(Eq),
    Ne(Ne),
    SDiv(signed::Div),
    Abs(signed::Abs),
    SRem(signed::Rem),
    SMod(signed::Mod),
    AShr(signed::Shr),
    SignExtend(signed::Extend),
    SLt(signed::Lt),
    SLe(signed::Le),
    SGt(signed::Gt),
    SGe(signed::Ge),
    UDiv(unsigned::Div),
    URem(unsigned::Rem),
    LShr(unsigned::Shr),
    ZeroExtend(unsigned::Extend),
    ULt(unsigned::Lt),
    ULe(unsigned::Le),
    UGt(unsigned::Gt),
    UGe(unsigned::Ge),
}

impl<T> Clone for BV<T>
where
    Const<T>: Clone,
{
    fn clone(&self) -> Self {
        match self {
            BV::Const(op) => BV::Const(op.clone()),
            BV::Concat(op) => BV::Concat(*op),
            BV::Extract(op) => BV::Extract(*op),
            BV::Repeat(op) => BV::Repeat(*op),
            BV::Not(op) => BV::Not(*op),
            BV::And(op) => BV::And(*op),
            BV::Or(op) => BV::Or(*op),
            BV::Xor(op) => BV::Xor(*op),
            BV::Nand(op) => BV::Nand(*op),
            BV::Nor(op) => BV::Nor(*op),
            BV::Xnor(op) => BV::Xnor(*op),
            BV::Neg(op) => BV::Neg(*op),
            BV::Ite(op) => BV::Ite(*op),
            BV::Add(op) => BV::Add(*op),
            BV::Sub(op) => BV::Sub(*op),
            BV::Mul(op) => BV::Mul(*op),
            BV::Shl(op) => BV::Shl(*op),
            BV::RotateLeft(op) => BV::RotateLeft(*op),
            BV::RotateRight(op) => BV::RotateRight(*op),
            BV::Eq(op) => BV::Eq(*op),
            BV::Ne(op) => BV::Ne(*op),
            BV::SDiv(op) => BV::SDiv(*op),
            BV::Abs(op) => BV::Abs(*op),
            BV::SRem(op) => BV::SRem(*op),
            BV::SMod(op) => BV::SMod(*op),
            BV::AShr(op) => BV::AShr(*op),
            BV::SignExtend(op) => BV::SignExtend(*op),
            BV::SLt(op) => BV::SLt(*op),
            BV::SLe(op) => BV::SLe(*op),
            BV::SGt(op) => BV::SGt(*op),
            BV::SGe(op) => BV::SGe(*op),
            BV::UDiv(op) => BV::UDiv(*op),
            BV::URem(op) => BV::URem(*op),
            BV::LShr(op) => BV::LShr(*op),
            BV::ZeroExtend(op) => BV::ZeroExtend(*op),
            BV::ULt(op) => BV::ULt(*op),
            BV::ULe(op) => BV::ULe(*op),
            BV::UGt(op) => BV::UGt(*op),
            BV::UGe(op) => BV::UGe(*op),
        }
    }
}

impl<T: fmt::Display> fmt::Display for BV<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BV::Const(op) => op.fmt(f),
            BV::Concat(op) => op.fmt(f),
            BV::Extract(op) => op.fmt(f),
            BV::Repeat(op) => op.fmt(f),
            BV::Not(op) => op.fmt(f),
            BV::And(op) => op.fmt(f),
            BV::Or(op) => op.fmt(f),
            BV::Xor(op) => op.fmt(f),
            BV::Nand(op) => op.fmt(f),
            BV::Nor(op) => op.fmt(f),
            BV::Xnor(op) => op.fmt(f),
            BV::Neg(op) => op.fmt(f),
            BV::Ite(op) => op.fmt(f),
            BV::Add(op) => op.fmt(f),
            BV::Sub(op) => op.fmt(f),
            BV::Mul(op) => op.fmt(f),
            BV::Shl(op) => op.fmt(f),
            BV::RotateLeft(op) => op.fmt(f),
            BV::RotateRight(op) => op.fmt(f),
            BV::Eq(op) => op.fmt(f),
            BV::Ne(op) => op.fmt(f),
            BV::SDiv(op) => op.fmt(f),
            BV::Abs(op) => op.fmt(f),
            BV::SRem(op) => op.fmt(f),
            BV::SMod(op) => op.fmt(f),
            BV::AShr(op) => op.fmt(f),
            BV::SignExtend(op) => op.fmt(f),
            BV::SLt(op) => op.fmt(f),
            BV::SLe(op) => op.fmt(f),
            BV::SGt(op) => op.fmt(f),
            BV::SGe(op) => op.fmt(f),
            BV::UDiv(op) => op.fmt(f),
            BV::URem(op) => op.fmt(f),
            BV::LShr(op) => op.fmt(f),
            BV::ZeroExtend(op) => op.fmt(f),
            BV::ULt(op) => op.fmt(f),
            BV::ULe(op) => op.fmt(f),
            BV::UGt(op) => op.fmt(f),
            BV::UGe(op) => op.fmt(f),
        }
    }
}

impl<T: Tensor> Boundary<Sort> for BV<T> {
    fn dom(&self) -> Poly<Sort> {
        match self {
            BV::Const(op) => op.dom(),
            BV::Concat(op) => op.dom(),
            BV::Extract(op) => op.dom(),
            BV::Repeat(op) => op.dom(),
            BV::Not(op) => op.dom(),
            BV::And(op) => op.dom(),
            BV::Or(op) => op.dom(),
            BV::Xor(op) => op.dom(),
            BV::Nand(op) => op.dom(),
            BV::Nor(op) => op.dom(),
            BV::Xnor(op) => op.dom(),
            BV::Neg(op) => op.dom(),
            BV::Ite(op) => op.dom(),
            BV::Add(op) => op.dom(),
            BV::Sub(op) => op.dom(),
            BV::Mul(op) => op.dom(),
            BV::Shl(op) => op.dom(),
            BV::RotateLeft(op) => op.dom(),
            BV::RotateRight(op) => op.dom(),
            BV::Eq(op) => op.dom(),
            BV::Ne(op) => op.dom(),
            BV::SDiv(op) => op.dom(),
            BV::Abs(op) => op.dom(),
            BV::SRem(op) => op.dom(),
            BV::SMod(op) => op.dom(),
            BV::AShr(op) => op.dom(),
            BV::SignExtend(op) => op.dom(),
            BV::SLt(op) => op.dom(),
            BV::SLe(op) => op.dom(),
            BV::SGt(op) => op.dom(),
            BV::SGe(op) => op.dom(),
            BV::UDiv(op) => op.dom(),
            BV::URem(op) => op.dom(),
            BV::LShr(op) => op.dom(),
            BV::ZeroExtend(op) => op.dom(),
            BV::ULt(op) => op.dom(),
            BV::ULe(op) => op.dom(),
            BV::UGt(op) => op.dom(),
            BV::UGe(op) => op.dom(),
        }
    }

    fn cod(&self) -> Poly<Sort> {
        match self {
            BV::Const(op) => op.cod(),
            BV::Concat(op) => op.cod(),
            BV::Extract(op) => op.cod(),
            BV::Repeat(op) => op.cod(),
            BV::Not(op) => op.cod(),
            BV::And(op) => op.cod(),
            BV::Or(op) => op.cod(),
            BV::Xor(op) => op.cod(),
            BV::Nand(op) => op.cod(),
            BV::Nor(op) => op.cod(),
            BV::Xnor(op) => op.cod(),
            BV::Neg(op) => op.cod(),
            BV::Ite(op) => op.cod(),
            BV::Add(op) => op.cod(),
            BV::Sub(op) => op.cod(),
            BV::Mul(op) => op.cod(),
            BV::Shl(op) => op.cod(),
            BV::RotateLeft(op) => op.cod(),
            BV::RotateRight(op) => op.cod(),
            BV::Eq(op) => op.cod(),
            BV::Ne(op) => op.cod(),
            BV::SDiv(op) => op.cod(),
            BV::Abs(op) => op.cod(),
            BV::SRem(op) => op.cod(),
            BV::SMod(op) => op.cod(),
            BV::AShr(op) => op.cod(),
            BV::SignExtend(op) => op.cod(),
            BV::SLt(op) => op.cod(),
            BV::SLe(op) => op.cod(),
            BV::SGt(op) => op.cod(),
            BV::SGe(op) => op.cod(),
            BV::UDiv(op) => op.cod(),
            BV::URem(op) => op.cod(),
            BV::LShr(op) => op.cod(),
            BV::ZeroExtend(op) => op.cod(),
            BV::ULt(op) => op.cod(),
            BV::ULe(op) => op.cod(),
            BV::UGt(op) => op.cod(),
            BV::UGe(op) => op.cod(),
        }
    }
}

impl<T: Tensor> Boundary<Sort> for Const<T> {
    fn dom(&self) -> Poly<Sort> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.width, self.tensor.shape()).into()].into()
    }
}

impl Boundary<Sort> for Concat {
    fn dom(&self) -> Poly<Sort> {
        let ([u, v], shape) = (self.widths, self.shape);
        mono![
            Word::new_unchecked(u, shape).into(),
            Word::new_unchecked(v, shape).into()
        ]
        .into()
    }

    fn cod(&self) -> Poly<Sort> {
        let [u, v] = self.widths;
        mono![Word::new_unchecked(u + v, self.shape).into()].into()
    }
}

impl Boundary<Sort> for Extract {
    fn dom(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.width, self.shape).into()].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.i - self.j + 1, self.shape).into()].into()
    }
}

impl Boundary<Sort> for Repeat {
    fn dom(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.width, self.shape).into()].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.i * self.width, self.shape).into()].into()
    }
}

impl Boundary<Sort> for Not {
    fn dom(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.width, self.shape).into()].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.width, self.shape).into()].into()
    }
}

impl Boundary<Sort> for And {
    fn dom(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.width, self.shape).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.width, self.shape).into()].into()
    }
}

impl Boundary<Sort> for Or {
    fn dom(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.width, self.shape).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.width, self.shape).into()].into()
    }
}

impl Boundary<Sort> for Xor {
    fn dom(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.width, self.shape).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.width, self.shape).into()].into()
    }
}

impl Boundary<Sort> for Nand {
    fn dom(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.width, self.shape).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.width, self.shape).into()].into()
    }
}

impl Boundary<Sort> for Nor {
    fn dom(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.width, self.shape).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.width, self.shape).into()].into()
    }
}

impl Boundary<Sort> for Xnor {
    fn dom(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.width, self.shape).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.width, self.shape).into()].into()
    }
}

impl Boundary<Sort> for Neg {
    fn dom(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.width, self.shape).into()].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.width, self.shape).into()].into()
    }
}

impl Boundary<Sort> for Add {
    fn dom(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.width, self.shape).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.width, self.shape).into()].into()
    }
}

impl Boundary<Sort> for Sub {
    fn dom(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.width, self.shape).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.width, self.shape).into()].into()
    }
}

impl Boundary<Sort> for Mul {
    fn dom(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.width, self.shape).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.width, self.shape).into()].into()
    }
}

impl Boundary<Sort> for Shl {
    fn dom(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.width, self.shape).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.width, self.shape).into()].into()
    }
}

impl Boundary<Sort> for RotateLeft {
    fn dom(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.width, self.shape).into()].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.width, self.shape).into()].into()
    }
}

impl Boundary<Sort> for RotateRight {
    fn dom(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.width, self.shape).into()].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.width, self.shape).into()].into()
    }
}

impl Boundary<Sort> for Eq {
    fn dom(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.width, self.shape).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new(self.shape).into()].into()
    }
}

impl Boundary<Sort> for Ne {
    fn dom(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.width, self.shape).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new(self.shape).into()].into()
    }
}

impl<T: fmt::Display> fmt::Display for Const<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.tensor)
    }
}

impl fmt::Display for Concat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Concat")
    }
}

impl fmt::Display for Extract {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Extract[{}:{}]", self.i, self.j)
    }
}

impl fmt::Display for Repeat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Repeat({})", self.i)
    }
}

impl fmt::Display for Not {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Not")
    }
}

impl fmt::Display for And {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "And")
    }
}

impl fmt::Display for Or {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Or")
    }
}

impl fmt::Display for Xor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Xor")
    }
}

impl fmt::Display for Nand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Nand")
    }
}

impl fmt::Display for Nor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Nor")
    }
}

impl fmt::Display for Xnor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Xnor")
    }
}

impl fmt::Display for Neg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Neg")
    }
}

impl Boundary<Sort> for Ite {
    fn dom(&self) -> Poly<Sort> {
        mono![
            Bool::new(self.shape).into(),
            Word::new_unchecked(self.width, self.shape).into(),
            Word::new_unchecked(self.width, self.shape).into()
        ]
        .into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.width, self.shape).into()].into()
    }
}

impl fmt::Display for Ite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ite")
    }
}

impl fmt::Display for Add {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Add")
    }
}

impl fmt::Display for Sub {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sub")
    }
}

impl fmt::Display for Mul {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Mul")
    }
}

impl fmt::Display for Shl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Shl")
    }
}

impl fmt::Display for RotateLeft {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RotateLeft({})", self.i)
    }
}

impl fmt::Display for RotateRight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RotateRight({})", self.i)
    }
}

impl fmt::Display for Eq {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Eq")
    }
}

impl fmt::Display for Ne {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ne")
    }
}

/// Operations on words read in two's complement, whose meaning depends on
/// signedness; the others are in the parent module.
pub mod signed {
    use super::{Sort, Word};
    use crate::tensor::sort::Bool;
    use crate::tensor::sort::check_width;
    use base::boundary::{Boundary, Poly};
    use base::mono;
    #[cfg(feature = "pyo3")]
    use pyo3::prelude::*;
    use std::fmt;

    /// Pointwise division `x / y`, rounding towards zero, as `Div(width, [m, n])`;
    /// the minimum word divided by `-1` is itself, and `x / 0` is `1` if `x`
    /// is negative and `-1` otherwise.
    ///
    /// SMT-LIB2: `(bvsdiv x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.word.signed")
    )]
    pub struct Div {
        width: usize,
        shape: [usize; 2],
    }

    /// Pointwise absolute value of `width`-bit words read in two's complement, as
    /// `Abs(width, [m, n])`: `Ite(x < 0, -x, x)`, so the minimum word is its own
    /// absolute value.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.word.signed")
    )]
    pub struct Abs {
        width: usize,
        shape: [usize; 2],
    }

    /// Pointwise remainder `x % y`, with the sign of `x`, as `Rem(width, [m, n])`;
    /// `x % 0 = x`.
    ///
    /// SMT-LIB2: `(bvsrem x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.word.signed")
    )]
    pub struct Rem {
        width: usize,
        shape: [usize; 2],
    }

    /// Pointwise modulo `x mod y`, with the sign of `y`, as `Mod(width, [m, n])`;
    /// `x mod 0 = x`.
    ///
    /// SMT-LIB2: `(bvsmod x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.word.signed")
    )]
    pub struct Mod {
        width: usize,
        shape: [usize; 2],
    }

    /// Pointwise arithmetic right shift `x >> y`, filling with the sign bit,
    /// as `Shr(width, [m, n])`.
    ///
    /// SMT-LIB2: `(bvashr x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.word.signed")
    )]
    pub struct Shr {
        width: usize,
        shape: [usize; 2],
    }

    /// Pointwise extension of `width`-bit words by `i` copies of the sign bit,
    /// as `Extend(i, width, [m, n])`.
    ///
    /// SMT-LIB2: `((_ sign_extend i) x)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.word.signed")
    )]
    pub struct Extend {
        i: usize,
        width: usize,
        shape: [usize; 2],
    }

    /// Pointwise signed comparison `x < y`, as `Lt(width, [m, n])`.
    ///
    /// SMT-LIB2: `(bvslt x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.word.signed")
    )]
    pub struct Lt {
        width: usize,
        shape: [usize; 2],
    }

    /// Pointwise signed comparison `x ≤ y`, as `Le(width, [m, n])`.
    ///
    /// SMT-LIB2: `(bvsle x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.word.signed")
    )]
    pub struct Le {
        width: usize,
        shape: [usize; 2],
    }

    /// Pointwise signed comparison `x > y`, as `Gt(width, [m, n])`.
    ///
    /// SMT-LIB2: `(bvsgt x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.word.signed")
    )]
    pub struct Gt {
        width: usize,
        shape: [usize; 2],
    }

    /// Pointwise signed comparison `x ≥ y`, as `Ge(width, [m, n])`.
    ///
    /// SMT-LIB2: `(bvsge x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.word.signed")
    )]
    pub struct Ge {
        width: usize,
        shape: [usize; 2],
    }

    impl Boundary<Sort> for Div {
        fn dom(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.width, self.shape).into(); 2].into()
        }

        fn cod(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.width, self.shape).into()].into()
        }
    }

    impl Boundary<Sort> for Rem {
        fn dom(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.width, self.shape).into(); 2].into()
        }

        fn cod(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.width, self.shape).into()].into()
        }
    }

    impl Boundary<Sort> for Mod {
        fn dom(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.width, self.shape).into(); 2].into()
        }

        fn cod(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.width, self.shape).into()].into()
        }
    }

    impl Boundary<Sort> for Shr {
        fn dom(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.width, self.shape).into(); 2].into()
        }

        fn cod(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.width, self.shape).into()].into()
        }
    }

    impl Boundary<Sort> for Extend {
        fn dom(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.width, self.shape).into()].into()
        }

        fn cod(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.width + self.i, self.shape).into()].into()
        }
    }

    impl Boundary<Sort> for Lt {
        fn dom(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.width, self.shape).into(); 2].into()
        }

        fn cod(&self) -> Poly<Sort> {
            mono![Bool::new(self.shape).into()].into()
        }
    }

    impl Boundary<Sort> for Le {
        fn dom(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.width, self.shape).into(); 2].into()
        }

        fn cod(&self) -> Poly<Sort> {
            mono![Bool::new(self.shape).into()].into()
        }
    }

    impl Boundary<Sort> for Gt {
        fn dom(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.width, self.shape).into(); 2].into()
        }

        fn cod(&self) -> Poly<Sort> {
            mono![Bool::new(self.shape).into()].into()
        }
    }

    impl Boundary<Sort> for Ge {
        fn dom(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.width, self.shape).into(); 2].into()
        }

        fn cod(&self) -> Poly<Sort> {
            mono![Bool::new(self.shape).into()].into()
        }
    }

    impl fmt::Display for Div {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "SDiv")
        }
    }

    impl Boundary<Sort> for Abs {
        fn dom(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.width, self.shape).into()].into()
        }

        fn cod(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.width, self.shape).into()].into()
        }
    }

    impl fmt::Display for Abs {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "Abs")
        }
    }

    impl fmt::Display for Rem {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "SRem")
        }
    }

    impl fmt::Display for Mod {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "SMod")
        }
    }

    impl fmt::Display for Shr {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "AShr")
        }
    }

    impl fmt::Display for Extend {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "SignExtend({})", self.i)
        }
    }

    impl fmt::Display for Lt {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "SLt")
        }
    }

    impl fmt::Display for Le {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "SLe")
        }
    }

    impl fmt::Display for Gt {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "SGt")
        }
    }

    impl fmt::Display for Ge {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "SGe")
        }
    }

    impl Div {
        /// The operation reading the words of `dom`; fails unless they are equal.
        pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
            let s = crate::tensor::same_sort(dom)?;
            Self::new(s.width(), s.shape())
        }

        /// The operation writing the words of `cod`.
        pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
            let [s] = cod;
            Self::new(s.width(), s.shape())
        }

        pub fn new(width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
            check_width(width)?;
            Ok(Div { width, shape })
        }
    }

    // The Python constructors and getters of the operations.
    #[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
    impl Div {
        /// Python builds it from the parameters that its boundary does not determine,
        /// and its domain `dom`; its codomain `cod`, when given, must be the one it
        /// writes.
        #[cfg(feature = "pyo3")]
        #[new]
        #[pyo3(signature = (dom, cod = None))]
        fn py_new(dom: [Word; 2], cod: Option<Word>) -> Result<Self, crate::Error> {
            crate::tensor::check_cod(Self::with_dom(dom), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str) {
            ("dom", "cod")
        }

        /// The sorts it reads: the sort alone when it reads one, else their list.
        #[cfg(feature = "pyo3")]
        #[getter(dom)]
        fn py_dom<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::tensor::py_sorts(py, Boundary::dom(self))
        }

        /// The sort it writes.
        #[cfg(feature = "pyo3")]
        #[getter(cod)]
        fn py_cod<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::tensor::py_sorts(py, Boundary::cod(self))
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn width(&self) -> usize {
            self.width
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn shape(&self) -> [usize; 2] {
            self.shape
        }
    }

    impl Abs {
        /// The operation reading the words of `dom`.
        pub fn with_dom(dom: [Word; 1]) -> Result<Self, crate::Error> {
            let s = crate::tensor::same_sort(dom)?;
            Self::new(s.width(), s.shape())
        }

        /// The operation writing the words of `cod`.
        pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
            let [s] = cod;
            Self::new(s.width(), s.shape())
        }

        pub fn new(width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
            check_width(width)?;
            Ok(Abs { width, shape })
        }
    }

    #[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
    impl Abs {
        /// Python builds it from the parameters that its boundary does not determine,
        /// and its domain `dom`; its codomain `cod`, when given, must be the one it
        /// writes.
        #[cfg(feature = "pyo3")]
        #[new]
        #[pyo3(signature = (dom, cod = None))]
        fn py_new(dom: Word, cod: Option<Word>) -> Result<Self, crate::Error> {
            crate::tensor::check_cod(Self::with_dom([dom]), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str) {
            ("dom", "cod")
        }

        /// The sorts it reads: the sort alone when it reads one, else their list.
        #[cfg(feature = "pyo3")]
        #[getter(dom)]
        fn py_dom<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::tensor::py_sorts(py, Boundary::dom(self))
        }

        /// The sort it writes.
        #[cfg(feature = "pyo3")]
        #[getter(cod)]
        fn py_cod<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::tensor::py_sorts(py, Boundary::cod(self))
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn width(&self) -> usize {
            self.width
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn shape(&self) -> [usize; 2] {
            self.shape
        }
    }

    impl Rem {
        /// The operation reading the words of `dom`; fails unless they are equal.
        pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
            let s = crate::tensor::same_sort(dom)?;
            Self::new(s.width(), s.shape())
        }

        /// The operation writing the words of `cod`.
        pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
            let [s] = cod;
            Self::new(s.width(), s.shape())
        }

        pub fn new(width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
            check_width(width)?;
            Ok(Rem { width, shape })
        }
    }

    #[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
    impl Rem {
        /// Python builds it from the parameters that its boundary does not determine,
        /// and its domain `dom`; its codomain `cod`, when given, must be the one it
        /// writes.
        #[cfg(feature = "pyo3")]
        #[new]
        #[pyo3(signature = (dom, cod = None))]
        fn py_new(dom: [Word; 2], cod: Option<Word>) -> Result<Self, crate::Error> {
            crate::tensor::check_cod(Self::with_dom(dom), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str) {
            ("dom", "cod")
        }

        /// The sorts it reads: the sort alone when it reads one, else their list.
        #[cfg(feature = "pyo3")]
        #[getter(dom)]
        fn py_dom<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::tensor::py_sorts(py, Boundary::dom(self))
        }

        /// The sort it writes.
        #[cfg(feature = "pyo3")]
        #[getter(cod)]
        fn py_cod<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::tensor::py_sorts(py, Boundary::cod(self))
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn width(&self) -> usize {
            self.width
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn shape(&self) -> [usize; 2] {
            self.shape
        }
    }

    impl Mod {
        /// The operation reading the words of `dom`; fails unless they are equal.
        pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
            let s = crate::tensor::same_sort(dom)?;
            Self::new(s.width(), s.shape())
        }

        /// The operation writing the words of `cod`.
        pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
            let [s] = cod;
            Self::new(s.width(), s.shape())
        }

        pub fn new(width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
            check_width(width)?;
            Ok(Mod { width, shape })
        }
    }

    #[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
    impl Mod {
        /// Python builds it from the parameters that its boundary does not determine,
        /// and its domain `dom`; its codomain `cod`, when given, must be the one it
        /// writes.
        #[cfg(feature = "pyo3")]
        #[new]
        #[pyo3(signature = (dom, cod = None))]
        fn py_new(dom: [Word; 2], cod: Option<Word>) -> Result<Self, crate::Error> {
            crate::tensor::check_cod(Self::with_dom(dom), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str) {
            ("dom", "cod")
        }

        /// The sorts it reads: the sort alone when it reads one, else their list.
        #[cfg(feature = "pyo3")]
        #[getter(dom)]
        fn py_dom<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::tensor::py_sorts(py, Boundary::dom(self))
        }

        /// The sort it writes.
        #[cfg(feature = "pyo3")]
        #[getter(cod)]
        fn py_cod<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::tensor::py_sorts(py, Boundary::cod(self))
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn width(&self) -> usize {
            self.width
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn shape(&self) -> [usize; 2] {
            self.shape
        }
    }

    impl Shr {
        /// The operation reading the words of `dom`; fails unless they are equal.
        pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
            let s = crate::tensor::same_sort(dom)?;
            Self::new(s.width(), s.shape())
        }

        /// The operation writing the words of `cod`.
        pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
            let [s] = cod;
            Self::new(s.width(), s.shape())
        }

        pub fn new(width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
            check_width(width)?;
            Ok(Shr { width, shape })
        }
    }

    #[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
    impl Shr {
        /// Python builds it from the parameters that its boundary does not determine,
        /// and its domain `dom`; its codomain `cod`, when given, must be the one it
        /// writes.
        #[cfg(feature = "pyo3")]
        #[new]
        #[pyo3(signature = (dom, cod = None))]
        fn py_new(dom: [Word; 2], cod: Option<Word>) -> Result<Self, crate::Error> {
            crate::tensor::check_cod(Self::with_dom(dom), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str) {
            ("dom", "cod")
        }

        /// The sorts it reads: the sort alone when it reads one, else their list.
        #[cfg(feature = "pyo3")]
        #[getter(dom)]
        fn py_dom<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::tensor::py_sorts(py, Boundary::dom(self))
        }

        /// The sort it writes.
        #[cfg(feature = "pyo3")]
        #[getter(cod)]
        fn py_cod<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::tensor::py_sorts(py, Boundary::cod(self))
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn width(&self) -> usize {
            self.width
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn shape(&self) -> [usize; 2] {
            self.shape
        }
    }

    impl Extend {
        /// The operation of indices `i`, reading the words of `dom`.
        pub fn with_dom(i: usize, dom: [Word; 1]) -> Result<Self, crate::Error> {
            let [s] = dom;
            Self::new(i, s.width(), s.shape())
        }

        /// The extension by `i` bits writing the words of `cod`; fails unless
        /// their width exceeds `i`.
        pub fn with_cod(i: usize, cod: [Word; 1]) -> Result<Self, crate::Error> {
            let [s] = cod;
            match s.width().checked_sub(i) {
                Some(width) if width > 0 => Self::new(i, width, s.shape()),
                _ => Err(crate::tensor::unexpected_cod(
                    &format!("1 sort of more than {i} bits"),
                    &cod,
                )),
            }
        }

        /// Fails unless `width` is positive, and `width + i`, the width of
        /// the words written, fits a `usize`.
        pub fn new(i: usize, width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
            check_width(width)?;
            if width.checked_add(i).is_none() {
                return Err(crate::Error(format!(
                    "cannot extend words of {width} bits by {i} bits: the result would have more than {} bits",
                    usize::MAX
                )));
            }
            Ok(Extend { i, width, shape })
        }
    }

    #[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
    impl Extend {
        /// Python builds it from the parameters that its boundary does not determine,
        /// and its domain `dom`; its codomain `cod`, when given, must be the one it
        /// writes.
        #[cfg(feature = "pyo3")]
        #[new]
        #[pyo3(signature = (i, /, dom, cod = None))]
        fn py_new(i: usize, dom: Word, cod: Option<Word>) -> Result<Self, crate::Error> {
            crate::tensor::check_cod(Self::with_dom(i, [dom]), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str, &'static str) {
            ("i", "dom", "cod")
        }

        /// The sorts it reads: the sort alone when it reads one, else their list.
        #[cfg(feature = "pyo3")]
        #[getter(dom)]
        fn py_dom<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::tensor::py_sorts(py, Boundary::dom(self))
        }

        /// The sort it writes.
        #[cfg(feature = "pyo3")]
        #[getter(cod)]
        fn py_cod<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::tensor::py_sorts(py, Boundary::cod(self))
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn i(&self) -> usize {
            self.i
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn width(&self) -> usize {
            self.width
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn shape(&self) -> [usize; 2] {
            self.shape
        }
    }

    impl Lt {
        /// The operation reading the words of `dom`; fails unless they are equal.
        pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
            let s = crate::tensor::same_sort(dom)?;
            Self::new(s.width(), s.shape())
        }

        pub fn new(width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
            check_width(width)?;
            Ok(Lt { width, shape })
        }
    }

    #[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
    impl Lt {
        /// Python builds it from the parameters that its boundary does not determine,
        /// and its domain `dom`; its codomain `cod`, when given, must be the one it
        /// writes.
        #[cfg(feature = "pyo3")]
        #[new]
        #[pyo3(signature = (dom, cod = None))]
        fn py_new(dom: [Word; 2], cod: Option<Bool>) -> Result<Self, crate::Error> {
            crate::tensor::check_cod(Self::with_dom(dom), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str) {
            ("dom", "cod")
        }

        /// The sorts it reads: the sort alone when it reads one, else their list.
        #[cfg(feature = "pyo3")]
        #[getter(dom)]
        fn py_dom<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::tensor::py_sorts(py, Boundary::dom(self))
        }

        /// The sort it writes.
        #[cfg(feature = "pyo3")]
        #[getter(cod)]
        fn py_cod<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::tensor::py_sorts(py, Boundary::cod(self))
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn width(&self) -> usize {
            self.width
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn shape(&self) -> [usize; 2] {
            self.shape
        }
    }

    impl Le {
        /// The operation reading the words of `dom`; fails unless they are equal.
        pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
            let s = crate::tensor::same_sort(dom)?;
            Self::new(s.width(), s.shape())
        }

        pub fn new(width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
            check_width(width)?;
            Ok(Le { width, shape })
        }
    }

    #[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
    impl Le {
        /// Python builds it from the parameters that its boundary does not determine,
        /// and its domain `dom`; its codomain `cod`, when given, must be the one it
        /// writes.
        #[cfg(feature = "pyo3")]
        #[new]
        #[pyo3(signature = (dom, cod = None))]
        fn py_new(dom: [Word; 2], cod: Option<Bool>) -> Result<Self, crate::Error> {
            crate::tensor::check_cod(Self::with_dom(dom), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str) {
            ("dom", "cod")
        }

        /// The sorts it reads: the sort alone when it reads one, else their list.
        #[cfg(feature = "pyo3")]
        #[getter(dom)]
        fn py_dom<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::tensor::py_sorts(py, Boundary::dom(self))
        }

        /// The sort it writes.
        #[cfg(feature = "pyo3")]
        #[getter(cod)]
        fn py_cod<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::tensor::py_sorts(py, Boundary::cod(self))
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn width(&self) -> usize {
            self.width
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn shape(&self) -> [usize; 2] {
            self.shape
        }
    }

    impl Gt {
        /// The operation reading the words of `dom`; fails unless they are equal.
        pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
            let s = crate::tensor::same_sort(dom)?;
            Self::new(s.width(), s.shape())
        }

        pub fn new(width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
            check_width(width)?;
            Ok(Gt { width, shape })
        }
    }

    #[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
    impl Gt {
        /// Python builds it from the parameters that its boundary does not determine,
        /// and its domain `dom`; its codomain `cod`, when given, must be the one it
        /// writes.
        #[cfg(feature = "pyo3")]
        #[new]
        #[pyo3(signature = (dom, cod = None))]
        fn py_new(dom: [Word; 2], cod: Option<Bool>) -> Result<Self, crate::Error> {
            crate::tensor::check_cod(Self::with_dom(dom), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str) {
            ("dom", "cod")
        }

        /// The sorts it reads: the sort alone when it reads one, else their list.
        #[cfg(feature = "pyo3")]
        #[getter(dom)]
        fn py_dom<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::tensor::py_sorts(py, Boundary::dom(self))
        }

        /// The sort it writes.
        #[cfg(feature = "pyo3")]
        #[getter(cod)]
        fn py_cod<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::tensor::py_sorts(py, Boundary::cod(self))
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn width(&self) -> usize {
            self.width
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn shape(&self) -> [usize; 2] {
            self.shape
        }
    }

    impl Ge {
        /// The operation reading the words of `dom`; fails unless they are equal.
        pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
            let s = crate::tensor::same_sort(dom)?;
            Self::new(s.width(), s.shape())
        }

        pub fn new(width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
            check_width(width)?;
            Ok(Ge { width, shape })
        }
    }

    #[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
    impl Ge {
        /// Python builds it from the parameters that its boundary does not determine,
        /// and its domain `dom`; its codomain `cod`, when given, must be the one it
        /// writes.
        #[cfg(feature = "pyo3")]
        #[new]
        #[pyo3(signature = (dom, cod = None))]
        fn py_new(dom: [Word; 2], cod: Option<Bool>) -> Result<Self, crate::Error> {
            crate::tensor::check_cod(Self::with_dom(dom), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str) {
            ("dom", "cod")
        }

        /// The sorts it reads: the sort alone when it reads one, else their list.
        #[cfg(feature = "pyo3")]
        #[getter(dom)]
        fn py_dom<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::tensor::py_sorts(py, Boundary::dom(self))
        }

        /// The sort it writes.
        #[cfg(feature = "pyo3")]
        #[getter(cod)]
        fn py_cod<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::tensor::py_sorts(py, Boundary::cod(self))
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn width(&self) -> usize {
            self.width
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn shape(&self) -> [usize; 2] {
            self.shape
        }
    }
}

/// Operations on words read in binary, whose meaning depends on
/// signedness; the others are in the parent module.
pub mod unsigned {
    use super::{Sort, Word};
    use crate::tensor::sort::Bool;
    use crate::tensor::sort::check_width;
    use base::boundary::{Boundary, Poly};
    use base::mono;
    #[cfg(feature = "pyo3")]
    use pyo3::prelude::*;
    use std::fmt;

    /// Pointwise division `x / y`, rounding down, as `Div(width, [m, n])`; `x / 0`
    /// is the word of all ones.
    ///
    /// SMT-LIB2: `(bvudiv x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(
            frozen,
            from_py_object,
            eq,
            str,
            module = "reactor.tensor.word.unsigned"
        )
    )]
    pub struct Div {
        width: usize,
        shape: [usize; 2],
    }

    /// Pointwise remainder `x % y`, as `Rem(width, [m, n])`; `x % 0 = x`.
    ///
    /// SMT-LIB2: `(bvurem x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(
            frozen,
            from_py_object,
            eq,
            str,
            module = "reactor.tensor.word.unsigned"
        )
    )]
    pub struct Rem {
        width: usize,
        shape: [usize; 2],
    }

    /// Pointwise logical right shift `x >> y`, filling with zeros, as
    /// `Shr(width, [m, n])`.
    ///
    /// SMT-LIB2: `(bvlshr x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(
            frozen,
            from_py_object,
            eq,
            str,
            module = "reactor.tensor.word.unsigned"
        )
    )]
    pub struct Shr {
        width: usize,
        shape: [usize; 2],
    }

    /// Pointwise extension of `width`-bit words by `i` zero bits, as
    /// `Extend(i, width, [m, n])`.
    ///
    /// SMT-LIB2: `((_ zero_extend i) x)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(
            frozen,
            from_py_object,
            eq,
            str,
            module = "reactor.tensor.word.unsigned"
        )
    )]
    pub struct Extend {
        i: usize,
        width: usize,
        shape: [usize; 2],
    }

    /// Pointwise unsigned comparison `x < y`, as `Lt(width, [m, n])`.
    ///
    /// SMT-LIB2: `(bvult x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(
            frozen,
            from_py_object,
            eq,
            str,
            module = "reactor.tensor.word.unsigned"
        )
    )]
    pub struct Lt {
        width: usize,
        shape: [usize; 2],
    }

    /// Pointwise unsigned comparison `x ≤ y`, as `Le(width, [m, n])`.
    ///
    /// SMT-LIB2: `(bvule x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(
            frozen,
            from_py_object,
            eq,
            str,
            module = "reactor.tensor.word.unsigned"
        )
    )]
    pub struct Le {
        width: usize,
        shape: [usize; 2],
    }

    /// Pointwise unsigned comparison `x > y`, as `Gt(width, [m, n])`.
    ///
    /// SMT-LIB2: `(bvugt x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(
            frozen,
            from_py_object,
            eq,
            str,
            module = "reactor.tensor.word.unsigned"
        )
    )]
    pub struct Gt {
        width: usize,
        shape: [usize; 2],
    }

    /// Pointwise unsigned comparison `x ≥ y`, as `Ge(width, [m, n])`.
    ///
    /// SMT-LIB2: `(bvuge x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(
            frozen,
            from_py_object,
            eq,
            str,
            module = "reactor.tensor.word.unsigned"
        )
    )]
    pub struct Ge {
        width: usize,
        shape: [usize; 2],
    }

    impl Boundary<Sort> for Div {
        fn dom(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.width, self.shape).into(); 2].into()
        }

        fn cod(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.width, self.shape).into()].into()
        }
    }

    impl Boundary<Sort> for Rem {
        fn dom(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.width, self.shape).into(); 2].into()
        }

        fn cod(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.width, self.shape).into()].into()
        }
    }

    impl Boundary<Sort> for Shr {
        fn dom(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.width, self.shape).into(); 2].into()
        }

        fn cod(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.width, self.shape).into()].into()
        }
    }

    impl Boundary<Sort> for Extend {
        fn dom(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.width, self.shape).into()].into()
        }

        fn cod(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.width + self.i, self.shape).into()].into()
        }
    }

    impl Boundary<Sort> for Lt {
        fn dom(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.width, self.shape).into(); 2].into()
        }

        fn cod(&self) -> Poly<Sort> {
            mono![Bool::new(self.shape).into()].into()
        }
    }

    impl Boundary<Sort> for Le {
        fn dom(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.width, self.shape).into(); 2].into()
        }

        fn cod(&self) -> Poly<Sort> {
            mono![Bool::new(self.shape).into()].into()
        }
    }

    impl Boundary<Sort> for Gt {
        fn dom(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.width, self.shape).into(); 2].into()
        }

        fn cod(&self) -> Poly<Sort> {
            mono![Bool::new(self.shape).into()].into()
        }
    }

    impl Boundary<Sort> for Ge {
        fn dom(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.width, self.shape).into(); 2].into()
        }

        fn cod(&self) -> Poly<Sort> {
            mono![Bool::new(self.shape).into()].into()
        }
    }

    impl fmt::Display for Div {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "UDiv")
        }
    }

    impl fmt::Display for Rem {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "URem")
        }
    }

    impl fmt::Display for Shr {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "LShr")
        }
    }

    impl fmt::Display for Extend {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "ZeroExtend({})", self.i)
        }
    }

    impl fmt::Display for Lt {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "ULt")
        }
    }

    impl fmt::Display for Le {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "ULe")
        }
    }

    impl fmt::Display for Gt {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "UGt")
        }
    }

    impl fmt::Display for Ge {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "UGe")
        }
    }

    impl Div {
        /// The operation reading the words of `dom`; fails unless they are equal.
        pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
            let s = crate::tensor::same_sort(dom)?;
            Self::new(s.width(), s.shape())
        }

        /// The operation writing the words of `cod`.
        pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
            let [s] = cod;
            Self::new(s.width(), s.shape())
        }

        pub fn new(width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
            check_width(width)?;
            Ok(Div { width, shape })
        }
    }

    // The Python constructors and getters of the operations.
    #[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
    impl Div {
        /// Python builds it from the parameters that its boundary does not determine,
        /// and its domain `dom`; its codomain `cod`, when given, must be the one it
        /// writes.
        #[cfg(feature = "pyo3")]
        #[new]
        #[pyo3(signature = (dom, cod = None))]
        fn py_new(dom: [Word; 2], cod: Option<Word>) -> Result<Self, crate::Error> {
            crate::tensor::check_cod(Self::with_dom(dom), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str) {
            ("dom", "cod")
        }

        /// The sorts it reads: the sort alone when it reads one, else their list.
        #[cfg(feature = "pyo3")]
        #[getter(dom)]
        fn py_dom<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::tensor::py_sorts(py, Boundary::dom(self))
        }

        /// The sort it writes.
        #[cfg(feature = "pyo3")]
        #[getter(cod)]
        fn py_cod<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::tensor::py_sorts(py, Boundary::cod(self))
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn width(&self) -> usize {
            self.width
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn shape(&self) -> [usize; 2] {
            self.shape
        }
    }

    impl Rem {
        /// The operation reading the words of `dom`; fails unless they are equal.
        pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
            let s = crate::tensor::same_sort(dom)?;
            Self::new(s.width(), s.shape())
        }

        /// The operation writing the words of `cod`.
        pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
            let [s] = cod;
            Self::new(s.width(), s.shape())
        }

        pub fn new(width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
            check_width(width)?;
            Ok(Rem { width, shape })
        }
    }

    #[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
    impl Rem {
        /// Python builds it from the parameters that its boundary does not determine,
        /// and its domain `dom`; its codomain `cod`, when given, must be the one it
        /// writes.
        #[cfg(feature = "pyo3")]
        #[new]
        #[pyo3(signature = (dom, cod = None))]
        fn py_new(dom: [Word; 2], cod: Option<Word>) -> Result<Self, crate::Error> {
            crate::tensor::check_cod(Self::with_dom(dom), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str) {
            ("dom", "cod")
        }

        /// The sorts it reads: the sort alone when it reads one, else their list.
        #[cfg(feature = "pyo3")]
        #[getter(dom)]
        fn py_dom<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::tensor::py_sorts(py, Boundary::dom(self))
        }

        /// The sort it writes.
        #[cfg(feature = "pyo3")]
        #[getter(cod)]
        fn py_cod<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::tensor::py_sorts(py, Boundary::cod(self))
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn width(&self) -> usize {
            self.width
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn shape(&self) -> [usize; 2] {
            self.shape
        }
    }

    impl Shr {
        /// The operation reading the words of `dom`; fails unless they are equal.
        pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
            let s = crate::tensor::same_sort(dom)?;
            Self::new(s.width(), s.shape())
        }

        /// The operation writing the words of `cod`.
        pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
            let [s] = cod;
            Self::new(s.width(), s.shape())
        }

        pub fn new(width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
            check_width(width)?;
            Ok(Shr { width, shape })
        }
    }

    #[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
    impl Shr {
        /// Python builds it from the parameters that its boundary does not determine,
        /// and its domain `dom`; its codomain `cod`, when given, must be the one it
        /// writes.
        #[cfg(feature = "pyo3")]
        #[new]
        #[pyo3(signature = (dom, cod = None))]
        fn py_new(dom: [Word; 2], cod: Option<Word>) -> Result<Self, crate::Error> {
            crate::tensor::check_cod(Self::with_dom(dom), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str) {
            ("dom", "cod")
        }

        /// The sorts it reads: the sort alone when it reads one, else their list.
        #[cfg(feature = "pyo3")]
        #[getter(dom)]
        fn py_dom<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::tensor::py_sorts(py, Boundary::dom(self))
        }

        /// The sort it writes.
        #[cfg(feature = "pyo3")]
        #[getter(cod)]
        fn py_cod<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::tensor::py_sorts(py, Boundary::cod(self))
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn width(&self) -> usize {
            self.width
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn shape(&self) -> [usize; 2] {
            self.shape
        }
    }

    impl Extend {
        /// The operation of indices `i`, reading the words of `dom`.
        pub fn with_dom(i: usize, dom: [Word; 1]) -> Result<Self, crate::Error> {
            let [s] = dom;
            Self::new(i, s.width(), s.shape())
        }

        /// The extension by `i` bits writing the words of `cod`; fails unless
        /// their width exceeds `i`.
        pub fn with_cod(i: usize, cod: [Word; 1]) -> Result<Self, crate::Error> {
            let [s] = cod;
            match s.width().checked_sub(i) {
                Some(width) if width > 0 => Self::new(i, width, s.shape()),
                _ => Err(crate::tensor::unexpected_cod(
                    &format!("1 sort of more than {i} bits"),
                    &cod,
                )),
            }
        }

        /// Fails unless `width` is positive, and `width + i`, the width of
        /// the words written, fits a `usize`.
        pub fn new(i: usize, width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
            check_width(width)?;
            if width.checked_add(i).is_none() {
                return Err(crate::Error(format!(
                    "cannot extend words of {width} bits by {i} bits: the result would have more than {} bits",
                    usize::MAX
                )));
            }
            Ok(Extend { i, width, shape })
        }
    }

    #[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
    impl Extend {
        /// Python builds it from the parameters that its boundary does not determine,
        /// and its domain `dom`; its codomain `cod`, when given, must be the one it
        /// writes.
        #[cfg(feature = "pyo3")]
        #[new]
        #[pyo3(signature = (i, /, dom, cod = None))]
        fn py_new(i: usize, dom: Word, cod: Option<Word>) -> Result<Self, crate::Error> {
            crate::tensor::check_cod(Self::with_dom(i, [dom]), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str, &'static str) {
            ("i", "dom", "cod")
        }

        /// The sorts it reads: the sort alone when it reads one, else their list.
        #[cfg(feature = "pyo3")]
        #[getter(dom)]
        fn py_dom<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::tensor::py_sorts(py, Boundary::dom(self))
        }

        /// The sort it writes.
        #[cfg(feature = "pyo3")]
        #[getter(cod)]
        fn py_cod<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::tensor::py_sorts(py, Boundary::cod(self))
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn i(&self) -> usize {
            self.i
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn width(&self) -> usize {
            self.width
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn shape(&self) -> [usize; 2] {
            self.shape
        }
    }

    impl Lt {
        /// The operation reading the words of `dom`; fails unless they are equal.
        pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
            let s = crate::tensor::same_sort(dom)?;
            Self::new(s.width(), s.shape())
        }

        pub fn new(width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
            check_width(width)?;
            Ok(Lt { width, shape })
        }
    }

    #[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
    impl Lt {
        /// Python builds it from the parameters that its boundary does not determine,
        /// and its domain `dom`; its codomain `cod`, when given, must be the one it
        /// writes.
        #[cfg(feature = "pyo3")]
        #[new]
        #[pyo3(signature = (dom, cod = None))]
        fn py_new(dom: [Word; 2], cod: Option<Bool>) -> Result<Self, crate::Error> {
            crate::tensor::check_cod(Self::with_dom(dom), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str) {
            ("dom", "cod")
        }

        /// The sorts it reads: the sort alone when it reads one, else their list.
        #[cfg(feature = "pyo3")]
        #[getter(dom)]
        fn py_dom<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::tensor::py_sorts(py, Boundary::dom(self))
        }

        /// The sort it writes.
        #[cfg(feature = "pyo3")]
        #[getter(cod)]
        fn py_cod<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::tensor::py_sorts(py, Boundary::cod(self))
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn width(&self) -> usize {
            self.width
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn shape(&self) -> [usize; 2] {
            self.shape
        }
    }

    impl Le {
        /// The operation reading the words of `dom`; fails unless they are equal.
        pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
            let s = crate::tensor::same_sort(dom)?;
            Self::new(s.width(), s.shape())
        }

        pub fn new(width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
            check_width(width)?;
            Ok(Le { width, shape })
        }
    }

    #[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
    impl Le {
        /// Python builds it from the parameters that its boundary does not determine,
        /// and its domain `dom`; its codomain `cod`, when given, must be the one it
        /// writes.
        #[cfg(feature = "pyo3")]
        #[new]
        #[pyo3(signature = (dom, cod = None))]
        fn py_new(dom: [Word; 2], cod: Option<Bool>) -> Result<Self, crate::Error> {
            crate::tensor::check_cod(Self::with_dom(dom), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str) {
            ("dom", "cod")
        }

        /// The sorts it reads: the sort alone when it reads one, else their list.
        #[cfg(feature = "pyo3")]
        #[getter(dom)]
        fn py_dom<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::tensor::py_sorts(py, Boundary::dom(self))
        }

        /// The sort it writes.
        #[cfg(feature = "pyo3")]
        #[getter(cod)]
        fn py_cod<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::tensor::py_sorts(py, Boundary::cod(self))
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn width(&self) -> usize {
            self.width
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn shape(&self) -> [usize; 2] {
            self.shape
        }
    }

    impl Gt {
        /// The operation reading the words of `dom`; fails unless they are equal.
        pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
            let s = crate::tensor::same_sort(dom)?;
            Self::new(s.width(), s.shape())
        }

        pub fn new(width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
            check_width(width)?;
            Ok(Gt { width, shape })
        }
    }

    #[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
    impl Gt {
        /// Python builds it from the parameters that its boundary does not determine,
        /// and its domain `dom`; its codomain `cod`, when given, must be the one it
        /// writes.
        #[cfg(feature = "pyo3")]
        #[new]
        #[pyo3(signature = (dom, cod = None))]
        fn py_new(dom: [Word; 2], cod: Option<Bool>) -> Result<Self, crate::Error> {
            crate::tensor::check_cod(Self::with_dom(dom), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str) {
            ("dom", "cod")
        }

        /// The sorts it reads: the sort alone when it reads one, else their list.
        #[cfg(feature = "pyo3")]
        #[getter(dom)]
        fn py_dom<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::tensor::py_sorts(py, Boundary::dom(self))
        }

        /// The sort it writes.
        #[cfg(feature = "pyo3")]
        #[getter(cod)]
        fn py_cod<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::tensor::py_sorts(py, Boundary::cod(self))
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn width(&self) -> usize {
            self.width
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn shape(&self) -> [usize; 2] {
            self.shape
        }
    }

    impl Ge {
        /// The operation reading the words of `dom`; fails unless they are equal.
        pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
            let s = crate::tensor::same_sort(dom)?;
            Self::new(s.width(), s.shape())
        }

        pub fn new(width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
            check_width(width)?;
            Ok(Ge { width, shape })
        }
    }

    #[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
    impl Ge {
        /// Python builds it from the parameters that its boundary does not determine,
        /// and its domain `dom`; its codomain `cod`, when given, must be the one it
        /// writes.
        #[cfg(feature = "pyo3")]
        #[new]
        #[pyo3(signature = (dom, cod = None))]
        fn py_new(dom: [Word; 2], cod: Option<Bool>) -> Result<Self, crate::Error> {
            crate::tensor::check_cod(Self::with_dom(dom), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str) {
            ("dom", "cod")
        }

        /// The sorts it reads: the sort alone when it reads one, else their list.
        #[cfg(feature = "pyo3")]
        #[getter(dom)]
        fn py_dom<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::tensor::py_sorts(py, Boundary::dom(self))
        }

        /// The sort it writes.
        #[cfg(feature = "pyo3")]
        #[getter(cod)]
        fn py_cod<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::tensor::py_sorts(py, Boundary::cod(self))
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn width(&self) -> usize {
            self.width
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn shape(&self) -> [usize; 2] {
            self.shape
        }
    }
}

/// Extracts an operation of fixed-size bit-vectors from a Python object,
/// trying each operation in turn: those carrying no data through their own
/// classes, and [`Const`] through the extraction that the tensor type `T`
/// provides.
#[cfg(feature = "pyo3")]
impl<'a, 'py, T> pyo3::FromPyObject<'a, 'py> for BV<T>
where
    Const<T>: pyo3::FromPyObject<'a, 'py>,
{
    type Error = pyo3::PyErr;

    fn extract(ob: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use pyo3::types::PyAnyMethods;

        if let Ok(op) = ob.extract::<Concat>() {
            return Ok(BV::Concat(op));
        }
        if let Ok(op) = ob.extract::<Extract>() {
            return Ok(BV::Extract(op));
        }
        if let Ok(op) = ob.extract::<Repeat>() {
            return Ok(BV::Repeat(op));
        }
        if let Ok(op) = ob.extract::<Not>() {
            return Ok(BV::Not(op));
        }
        if let Ok(op) = ob.extract::<And>() {
            return Ok(BV::And(op));
        }
        if let Ok(op) = ob.extract::<Or>() {
            return Ok(BV::Or(op));
        }
        if let Ok(op) = ob.extract::<Xor>() {
            return Ok(BV::Xor(op));
        }
        if let Ok(op) = ob.extract::<Nand>() {
            return Ok(BV::Nand(op));
        }
        if let Ok(op) = ob.extract::<Nor>() {
            return Ok(BV::Nor(op));
        }
        if let Ok(op) = ob.extract::<Xnor>() {
            return Ok(BV::Xnor(op));
        }
        if let Ok(op) = ob.extract::<Neg>() {
            return Ok(BV::Neg(op));
        }
        if let Ok(op) = ob.extract::<Ite>() {
            return Ok(BV::Ite(op));
        }
        if let Ok(op) = ob.extract::<Add>() {
            return Ok(BV::Add(op));
        }
        if let Ok(op) = ob.extract::<Sub>() {
            return Ok(BV::Sub(op));
        }
        if let Ok(op) = ob.extract::<Mul>() {
            return Ok(BV::Mul(op));
        }
        if let Ok(op) = ob.extract::<Shl>() {
            return Ok(BV::Shl(op));
        }
        if let Ok(op) = ob.extract::<RotateLeft>() {
            return Ok(BV::RotateLeft(op));
        }
        if let Ok(op) = ob.extract::<RotateRight>() {
            return Ok(BV::RotateRight(op));
        }
        if let Ok(op) = ob.extract::<Eq>() {
            return Ok(BV::Eq(op));
        }
        if let Ok(op) = ob.extract::<Ne>() {
            return Ok(BV::Ne(op));
        }
        if let Ok(op) = ob.extract::<signed::Div>() {
            return Ok(BV::SDiv(op));
        }
        if let Ok(op) = ob.extract::<signed::Abs>() {
            return Ok(BV::Abs(op));
        }
        if let Ok(op) = ob.extract::<signed::Rem>() {
            return Ok(BV::SRem(op));
        }
        if let Ok(op) = ob.extract::<signed::Mod>() {
            return Ok(BV::SMod(op));
        }
        if let Ok(op) = ob.extract::<signed::Shr>() {
            return Ok(BV::AShr(op));
        }
        if let Ok(op) = ob.extract::<signed::Extend>() {
            return Ok(BV::SignExtend(op));
        }
        if let Ok(op) = ob.extract::<signed::Lt>() {
            return Ok(BV::SLt(op));
        }
        if let Ok(op) = ob.extract::<signed::Le>() {
            return Ok(BV::SLe(op));
        }
        if let Ok(op) = ob.extract::<signed::Gt>() {
            return Ok(BV::SGt(op));
        }
        if let Ok(op) = ob.extract::<signed::Ge>() {
            return Ok(BV::SGe(op));
        }
        if let Ok(op) = ob.extract::<unsigned::Div>() {
            return Ok(BV::UDiv(op));
        }
        if let Ok(op) = ob.extract::<unsigned::Rem>() {
            return Ok(BV::URem(op));
        }
        if let Ok(op) = ob.extract::<unsigned::Shr>() {
            return Ok(BV::LShr(op));
        }
        if let Ok(op) = ob.extract::<unsigned::Extend>() {
            return Ok(BV::ZeroExtend(op));
        }
        if let Ok(op) = ob.extract::<unsigned::Lt>() {
            return Ok(BV::ULt(op));
        }
        if let Ok(op) = ob.extract::<unsigned::Le>() {
            return Ok(BV::ULe(op));
        }
        if let Ok(op) = ob.extract::<unsigned::Gt>() {
            return Ok(BV::UGt(op));
        }
        if let Ok(op) = ob.extract::<unsigned::Ge>() {
            return Ok(BV::UGe(op));
        }
        if let Ok(op) = ob.extract::<Const<T>>() {
            return Ok(BV::Const(op));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "expected an operation of fixed-size bit-vectors, got {}",
            ob.get_type()
        )))
    }
}

impl Concat {
    /// The operation reading the words, high bits first, of `dom`; fails unless they are of the same shape.
    pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
        let [x, y] = dom;
        if x.shape() != y.shape() {
            return Err(crate::tensor::unexpected_dom(
                "2 sorts of the same shape",
                &dom,
            ));
        }
        Self::new([x.width(), y.width()], x.shape())
    }

    /// Fails unless both widths are positive, and their sum, the width of
    /// the words written, fits a `usize`.
    pub fn new(widths: [usize; 2], shape: [usize; 2]) -> Result<Self, crate::Error> {
        let [u, v] = widths;
        check_width(u)?;
        check_width(v)?;
        if u.checked_add(v).is_none() {
            return Err(crate::Error(format!(
                "cannot concatenate words of {u} and {v} bits: the result would have more than {} bits",
                usize::MAX
            )));
        }
        Ok(Concat { widths, shape })
    }
}

// The Python constructors of the operations carrying no tensor: each takes
// its sizes positionally, in the order of its fields, the shape as a list.
#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Concat {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Word; 2], cod: Option<Word>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }

    /// The widths `[u, v]` of the words read, high bits first.
    #[cfg_attr(feature = "pyo3", getter)]
    pub fn widths(&self) -> [usize; 2] {
        self.widths
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Extract {
    /// The operation of indices `i`, `j`, reading the words of `dom`.
    pub fn with_dom(i: usize, j: usize, dom: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = dom;
        Self::new(i, j, s.width(), s.shape())
    }

    /// Fails unless `j ≤ i < width`.
    pub fn new(i: usize, j: usize, width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
        if !(j <= i && i < width) {
            return Err(crate::Error(format!(
                "cannot extract the bits {i} down to {j} of {width}-bit words"
            )));
        }
        Ok(Extract { i, j, width, shape })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Extract {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (i, j, /, dom, cod = None))]
    fn py_new(i: usize, j: usize, dom: Word, cod: Option<Word>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom(i, j, [dom]), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str, &'static str, &'static str) {
        ("i", "j", "dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn i(&self) -> usize {
        self.i
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn j(&self) -> usize {
        self.j
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn width(&self) -> usize {
        self.width
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Repeat {
    /// The operation of indices `i`, reading the words of `dom`.
    pub fn with_dom(i: usize, dom: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = dom;
        Self::new(i, s.width(), s.shape())
    }

    /// The repetition `i` times writing the words of `cod`; fails unless `i`
    /// divides their width.
    pub fn with_cod(i: usize, cod: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        if i == 0 || s.width() % i != 0 {
            return Err(crate::tensor::unexpected_cod(
                &format!("1 sort of a width that {i} divides"),
                &cod,
            ));
        }
        Self::new(i, s.width() / i, s.shape())
    }

    /// Fails unless `i` and `width` are positive, and their product, the
    /// width of the words written, fits a `usize`.
    pub fn new(i: usize, width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
        if i == 0 {
            return Err(crate::Error("cannot repeat a word zero times".to_string()));
        }
        check_width(width)?;
        if i.checked_mul(width).is_none() {
            return Err(crate::Error(format!(
                "cannot repeat words of {width} bits {i} times: the result would have more than {} bits",
                usize::MAX
            )));
        }
        Ok(Repeat { i, width, shape })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Repeat {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (i, /, dom, cod = None))]
    fn py_new(i: usize, dom: Word, cod: Option<Word>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom(i, [dom]), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str, &'static str) {
        ("i", "dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn i(&self) -> usize {
        self.i
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn width(&self) -> usize {
        self.width
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Not {
    /// The operation reading the words of `dom`.
    pub fn with_dom(dom: [Word; 1]) -> Result<Self, crate::Error> {
        let s = crate::tensor::same_sort(dom)?;
        Self::new(s.width(), s.shape())
    }

    /// The operation writing the words of `cod`.
    pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Self::new(s.width(), s.shape())
    }

    pub fn new(width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
        check_width(width)?;
        Ok(Not { width, shape })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Not {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: Word, cod: Option<Word>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom([dom]), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn width(&self) -> usize {
        self.width
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl And {
    /// The operation reading the words of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
        let s = crate::tensor::same_sort(dom)?;
        Self::new(s.width(), s.shape())
    }

    /// The operation writing the words of `cod`.
    pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Self::new(s.width(), s.shape())
    }

    pub fn new(width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
        check_width(width)?;
        Ok(And { width, shape })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl And {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Word; 2], cod: Option<Word>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn width(&self) -> usize {
        self.width
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Or {
    /// The operation reading the words of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
        let s = crate::tensor::same_sort(dom)?;
        Self::new(s.width(), s.shape())
    }

    /// The operation writing the words of `cod`.
    pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Self::new(s.width(), s.shape())
    }

    pub fn new(width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
        check_width(width)?;
        Ok(Or { width, shape })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Or {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Word; 2], cod: Option<Word>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn width(&self) -> usize {
        self.width
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Xor {
    /// The operation reading the words of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
        let s = crate::tensor::same_sort(dom)?;
        Self::new(s.width(), s.shape())
    }

    /// The operation writing the words of `cod`.
    pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Self::new(s.width(), s.shape())
    }

    pub fn new(width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
        check_width(width)?;
        Ok(Xor { width, shape })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Xor {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Word; 2], cod: Option<Word>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn width(&self) -> usize {
        self.width
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Nand {
    /// The operation reading the words of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
        let s = crate::tensor::same_sort(dom)?;
        Self::new(s.width(), s.shape())
    }

    /// The operation writing the words of `cod`.
    pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Self::new(s.width(), s.shape())
    }

    pub fn new(width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
        check_width(width)?;
        Ok(Nand { width, shape })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Nand {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Word; 2], cod: Option<Word>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn width(&self) -> usize {
        self.width
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Nor {
    /// The operation reading the words of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
        let s = crate::tensor::same_sort(dom)?;
        Self::new(s.width(), s.shape())
    }

    /// The operation writing the words of `cod`.
    pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Self::new(s.width(), s.shape())
    }

    pub fn new(width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
        check_width(width)?;
        Ok(Nor { width, shape })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Nor {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Word; 2], cod: Option<Word>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn width(&self) -> usize {
        self.width
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Xnor {
    /// The operation reading the words of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
        let s = crate::tensor::same_sort(dom)?;
        Self::new(s.width(), s.shape())
    }

    /// The operation writing the words of `cod`.
    pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Self::new(s.width(), s.shape())
    }

    pub fn new(width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
        check_width(width)?;
        Ok(Xnor { width, shape })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Xnor {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Word; 2], cod: Option<Word>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn width(&self) -> usize {
        self.width
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Neg {
    /// The operation reading the words of `dom`.
    pub fn with_dom(dom: [Word; 1]) -> Result<Self, crate::Error> {
        let s = crate::tensor::same_sort(dom)?;
        Self::new(s.width(), s.shape())
    }

    /// The operation writing the words of `cod`.
    pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Self::new(s.width(), s.shape())
    }

    pub fn new(width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
        check_width(width)?;
        Ok(Neg { width, shape })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Neg {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: Word, cod: Option<Word>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom([dom]), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn width(&self) -> usize {
        self.width
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Ite {
    /// The choice reading the boolean condition and the branches of `dom`;
    /// fails unless the branches are of the same sort, and the condition of
    /// their shape.
    pub fn with_dom(dom: [Sort; 3]) -> Result<Self, crate::Error> {
        match dom {
            [Sort::Bool(c), Sort::Word(x), Sort::Word(y)] if x == y && c.shape() == x.shape() => {
                Self::new(x.width(), x.shape())
            }
            _ => Err(crate::tensor::unexpected_dom(
                "a condition Bool and two branches Word of its shape",
                &dom,
            )),
        }
    }

    /// The operation writing the words of `cod`.
    pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Self::new(s.width(), s.shape())
    }

    pub fn new(width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
        check_width(width)?;
        Ok(Ite { width, shape })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Ite {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Sort; 3], cod: Option<Word>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn width(&self) -> usize {
        self.width
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Add {
    /// The operation reading the words of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
        let s = crate::tensor::same_sort(dom)?;
        Self::new(s.width(), s.shape())
    }

    /// The operation writing the words of `cod`.
    pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Self::new(s.width(), s.shape())
    }

    pub fn new(width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
        check_width(width)?;
        Ok(Add { width, shape })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Add {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Word; 2], cod: Option<Word>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn width(&self) -> usize {
        self.width
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Sub {
    /// The operation reading the words of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
        let s = crate::tensor::same_sort(dom)?;
        Self::new(s.width(), s.shape())
    }

    /// The operation writing the words of `cod`.
    pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Self::new(s.width(), s.shape())
    }

    pub fn new(width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
        check_width(width)?;
        Ok(Sub { width, shape })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Sub {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Word; 2], cod: Option<Word>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn width(&self) -> usize {
        self.width
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Mul {
    /// The operation reading the words of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
        let s = crate::tensor::same_sort(dom)?;
        Self::new(s.width(), s.shape())
    }

    /// The operation writing the words of `cod`.
    pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Self::new(s.width(), s.shape())
    }

    pub fn new(width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
        check_width(width)?;
        Ok(Mul { width, shape })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Mul {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Word; 2], cod: Option<Word>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn width(&self) -> usize {
        self.width
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Shl {
    /// The operation reading the words of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
        let s = crate::tensor::same_sort(dom)?;
        Self::new(s.width(), s.shape())
    }

    /// The operation writing the words of `cod`.
    pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Self::new(s.width(), s.shape())
    }

    pub fn new(width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
        check_width(width)?;
        Ok(Shl { width, shape })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Shl {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Word; 2], cod: Option<Word>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn width(&self) -> usize {
        self.width
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl RotateLeft {
    /// The operation of indices `i`, reading the words of `dom`.
    pub fn with_dom(i: usize, dom: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = dom;
        Self::new(i, s.width(), s.shape())
    }

    /// The operation of index `i`, writing the words of `cod`.
    pub fn with_cod(i: usize, cod: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Self::new(i, s.width(), s.shape())
    }

    pub fn new(i: usize, width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
        check_width(width)?;
        Ok(RotateLeft { i, width, shape })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl RotateLeft {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (i, /, dom, cod = None))]
    fn py_new(i: usize, dom: Word, cod: Option<Word>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom(i, [dom]), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str, &'static str) {
        ("i", "dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn i(&self) -> usize {
        self.i
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn width(&self) -> usize {
        self.width
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl RotateRight {
    /// The operation of indices `i`, reading the words of `dom`.
    pub fn with_dom(i: usize, dom: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = dom;
        Self::new(i, s.width(), s.shape())
    }

    /// The operation of index `i`, writing the words of `cod`.
    pub fn with_cod(i: usize, cod: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Self::new(i, s.width(), s.shape())
    }

    pub fn new(i: usize, width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
        check_width(width)?;
        Ok(RotateRight { i, width, shape })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl RotateRight {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (i, /, dom, cod = None))]
    fn py_new(i: usize, dom: Word, cod: Option<Word>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom(i, [dom]), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str, &'static str) {
        ("i", "dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn i(&self) -> usize {
        self.i
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn width(&self) -> usize {
        self.width
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Eq {
    /// The operation reading the words of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
        let s = crate::tensor::same_sort(dom)?;
        Self::new(s.width(), s.shape())
    }

    pub fn new(width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
        check_width(width)?;
        Ok(Eq { width, shape })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Eq {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Word; 2], cod: Option<Bool>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn width(&self) -> usize {
        self.width
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Ne {
    /// The operation reading the words of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
        let s = crate::tensor::same_sort(dom)?;
        Self::new(s.width(), s.shape())
    }

    pub fn new(width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
        check_width(width)?;
        Ok(Ne { width, shape })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Ne {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Word; 2], cod: Option<Bool>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn width(&self) -> usize {
        self.width
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}
