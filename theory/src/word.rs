/*!
# Fixed-size bit-vectors

Defines the sort [`Word`] of matrices of `width`-bit machine words, the local
sort [`Words`] of the theory, holding words and the booleans that
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
two's complement, and [`unsigned`], reading them in binary; the choice of
whole matrices by a scalar boolean, [`broadcast::Ite`], is in the submodule
[`broadcast`].

| Operation | SMT-LIB2 | | Operation | SMT-LIB2 |
|---|---|---|---|---|
| [`Num`] | `#b…`, `(_ bvN width)` | | [`signed::Div`] | `bvsdiv` |
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
| [`broadcast::Ite`] | `ite` | | | |

`=` and `distinct` belong to the SMT-LIB2 Core theory, which every logic
includes, at the sort `(_ BitVec width)`; all the others belong to the
theory FixedSizeBitVectors or are defined by the logic QF_BV. The
comparisons, `=` and `distinct` write booleans, as in SMT-LIB2.
*/

use super::bool::Bool;
use crate::Matrix;
use crate::zero::Real0;
use base::boundary::Mono;
use base::boundary::{Boundary, Poly};
use base::differentiable::Differentiable;
use base::mono;
use derive_more::{Display, From};
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use std::fmt;

/// The sort of matrices of `width`-bit words with the given `[rows, cols]`
/// shape: entrywise, the SMT-LIB2 sort `(_ BitVec width)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, get_all, module = "reactor.word")
)]
pub struct Word {
    width: usize,
    shape: [usize; 2],
}

/// Fails unless `width` is positive: as in SMT-LIB2, words have at least one
/// bit.
pub(crate) fn check_width(width: usize) -> Result<(), crate::Error> {
    if width == 0 {
        return Err(crate::Error("words have at least one bit".to_string()));
    }
    Ok(())
}

impl Word {
    /// Matrices of `width`-bit words, as [`Word::new`], but panics if
    /// `width` is zero or `shape` empty.
    pub fn new_unchecked(width: usize, shape: [usize; 2]) -> Self {
        Self::new(width, shape).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

/// Python matches a `Word` by `width` and `shape`.
#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Word {
    /// Matrices of `width`-bit words; as in SMT-LIB2, `width` must be
    /// positive, and the shape must have at least one row and one column,
    /// or this returns an error.
    #[cfg_attr(feature = "pyo3", new)]
    pub fn new(width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
        check_width(width)?;
        crate::matrix::check_shape(shape)?;
        Ok(Word { width, shape })
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("width", "shape")
    }

    /// The tangent sort `T(self)`: the sort of the rate of change of its
    /// values.
    #[cfg(feature = "pyo3")]
    #[pyo3(name = "tangent")]
    fn tangent(&self) -> Real0 {
        Differentiable::tangent(self)
    }
}

impl fmt::Display for Word {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [i, j] = self.shape;
        write!(f, "Word{}([{i}, {j}])", self.width)
    }
}

/// Words are discrete, so their tangent is trivial: the singleton sort
/// [`Real0`].
impl Differentiable for Word {
    type Tangent = Real0;

    fn tangent(self) -> Real0 {
        Real0
    }
}

/// The sorts of the theory, as in the signature of SMT-LIB2's
/// FixedSizeBitVectors: words, and the booleans of the Core theory, which
/// comparisons write, together with [`Real0`], the tangent of both. It is
/// the subset [`Words`](crate::Words) of the sorts of all
/// theories.
use crate::Words;

/// A constant matrix literal of `width`-bit words, as `Num(tensor, width)`, held
/// in a tensor of type `T`.
///
/// SMT-LIB2: a literal `#b…` or `(_ bvN width)`.
#[derive(Clone, Copy, Debug, Display)]
#[display("{tensor}")]
pub struct Num<T> {
    tensor: T,
    width: usize,
}

impl<T: Matrix> Num<T> {
    /// The constant held in `tensor`, writing the words of `cod`; fails
    /// unless `tensor` is of its shape.
    pub fn with_cod(tensor: T, cod: [Word; 1]) -> Result<Self, crate::Error> {
        let [x] = cod;
        if tensor.shape() != x.shape() {
            return Err(crate::unexpected_cod(
                &format!("1 sort of the shape {:?} of the constant", tensor.shape()),
                &cod,
            ));
        }
        Ok(Num {
            tensor,
            width: x.width(),
        })
    }

    /// The sorts it writes.
    pub(crate) fn cod(&self) -> Mono<Words> {
        mono![Word::new_unchecked(self.width, self.tensor.shape()).into()]
    }
}

impl<T> Num<T> {
    /// The tensor holding the constant.
    pub fn tensor(&self) -> &T {
        &self.tensor
    }
}

/// Pointwise concatenation `x ++ y` of a `u`-bit word `x`, the high bits
/// of the result, and a `v`-bit word `y`, its low bits, as
/// `Concat([u, v], [m, n])`; the result has `u + v` bits.
///
/// SMT-LIB2: `(concat x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
#[display("Concat")]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word")
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
#[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
#[display("Extract[{i}:{j}]")]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word")
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
#[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
#[display("Repeat({i})")]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word")
)]
pub struct Repeat {
    i: usize,
    width: usize,
    shape: [usize; 2],
}

/// Pointwise bitwise negation `!x`, as `Not(width, [m, n])`.
///
/// SMT-LIB2: `(bvnot x)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
#[display("Not")]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word")
)]
pub struct Not {
    width: usize,
    shape: [usize; 2],
}

/// Pointwise bitwise conjunction `x & y`, as `And(width, [m, n])`.
///
/// SMT-LIB2: `(bvand x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
#[display("And")]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word")
)]
pub struct And {
    width: usize,
    shape: [usize; 2],
}

/// Pointwise bitwise disjunction `x | y`, as `Or(width, [m, n])`.
///
/// SMT-LIB2: `(bvor x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
#[display("Or")]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word")
)]
pub struct Or {
    width: usize,
    shape: [usize; 2],
}

/// Pointwise bitwise exclusive disjunction `x ^ y`, as `Xor(width, [m, n])`.
///
/// SMT-LIB2: `(bvxor x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
#[display("Xor")]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word")
)]
pub struct Xor {
    width: usize,
    shape: [usize; 2],
}

/// Pointwise bitwise negated conjunction `!(x & y)`, as `Nand(width, [m, n])`.
///
/// SMT-LIB2: `(bvnand x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
#[display("Nand")]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word")
)]
pub struct Nand {
    width: usize,
    shape: [usize; 2],
}

/// Pointwise bitwise negated disjunction `!(x | y)`, as `Nor(width, [m, n])`.
///
/// SMT-LIB2: `(bvnor x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
#[display("Nor")]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word")
)]
pub struct Nor {
    width: usize,
    shape: [usize; 2],
}

/// Pointwise bitwise negated exclusive disjunction `!(x ^ y)`, as
/// `Xnor(width, [m, n])`.
///
/// SMT-LIB2: `(bvxnor x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
#[display("Xnor")]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word")
)]
pub struct Xnor {
    width: usize,
    shape: [usize; 2],
}

/// Pointwise two's-complement negation `-x`, modulo `2^width`, as
/// `Neg(width, [m, n])`; the minimum signed word is its own negation.
///
/// SMT-LIB2: `(bvneg x)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
#[display("Neg")]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word")
)]
pub struct Neg {
    width: usize,
    shape: [usize; 2],
}

/// Pointwise choice `if c then x else y` between `[m, n]` matrices of `width`-bit
/// words `x` and `y`, by the boolean matrix `c`, read first, as `Ite(width, [m, n])`.
///
/// SMT-LIB2: `(ite c x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
#[display("Ite")]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word")
)]
pub struct Ite {
    width: usize,
    shape: [usize; 2],
}

/// Pointwise addition `x + y`, modulo `2^width`, as `Add(width, [m, n])`.
///
/// SMT-LIB2: `(bvadd x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
#[display("Add")]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word")
)]
pub struct Add {
    width: usize,
    shape: [usize; 2],
}

/// Pointwise subtraction `x - y`, modulo `2^width`, as `Sub(width, [m, n])`.
///
/// SMT-LIB2: `(bvsub x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
#[display("Sub")]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word")
)]
pub struct Sub {
    width: usize,
    shape: [usize; 2],
}

/// Pointwise multiplication `x * y`, modulo `2^width`, as `Mul(width, [m, n])`.
///
/// SMT-LIB2: `(bvmul x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
#[display("Mul")]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word")
)]
pub struct Mul {
    width: usize,
    shape: [usize; 2],
}

/// Pointwise left shift `x << y`, filling with zeros, as `Shl(width, [m, n])`.
///
/// SMT-LIB2: `(bvshl x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
#[display("Shl")]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word")
)]
pub struct Shl {
    width: usize,
    shape: [usize; 2],
}

/// Pointwise rotation left by a constant `i` bits, as
/// `RotateLeft(i, width, [m, n])`.
///
/// SMT-LIB2: `((_ rotate_left i) x)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
#[display("RotateLeft({i})")]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word")
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
#[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
#[display("RotateRight({i})")]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word")
)]
pub struct RotateRight {
    i: usize,
    width: usize,
    shape: [usize; 2],
}

/// Pointwise equality `x = y` of `width`-bit words, as `Eq(width, [m, n])`.
///
/// SMT-LIB2: `(= x y)`, of the Core theory.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
#[display("Eq")]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word")
)]
pub struct Eq {
    width: usize,
    shape: [usize; 2],
}

/// Pointwise disequality `x ≠ y` of `width`-bit words, as `Ne(width, [m, n])`.
///
/// SMT-LIB2: `(distinct x y)`, of the Core theory.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
#[display("Ne")]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word")
)]
pub struct Ne {
    width: usize,
    shape: [usize; 2],
}

/// The operations of the theory of fixed-size bit-vectors, SMT-LIB2's
/// `FixedSizeBitVectors`, with their boundary in [`Words`]. Those of
/// [`signed`] and [`unsigned`] are named after their SMT-LIB2 operation,
/// as `SDiv` for [`signed::Div`] and `UDiv` for [`unsigned::Div`].
#[derive(Clone, Copy, Debug, From, Display)]
pub enum BV<T> {
    Num(Num<T>),
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
    BroadcastIte(broadcast::Ite),
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

impl<T: Matrix> Boundary<Words> for BV<T> {
    fn dom(&self) -> Poly<Words> {
        match self {
            BV::Num(op) => Boundary::dom(op),
            BV::Concat(op) => Boundary::dom(op),
            BV::Extract(op) => Boundary::dom(op),
            BV::Repeat(op) => Boundary::dom(op),
            BV::Not(op) => Boundary::dom(op),
            BV::And(op) => Boundary::dom(op),
            BV::Or(op) => Boundary::dom(op),
            BV::Xor(op) => Boundary::dom(op),
            BV::Nand(op) => Boundary::dom(op),
            BV::Nor(op) => Boundary::dom(op),
            BV::Xnor(op) => Boundary::dom(op),
            BV::Neg(op) => Boundary::dom(op),
            BV::Ite(op) => Boundary::dom(op),
            BV::BroadcastIte(op) => Boundary::dom(op),
            BV::Add(op) => Boundary::dom(op),
            BV::Sub(op) => Boundary::dom(op),
            BV::Mul(op) => Boundary::dom(op),
            BV::Shl(op) => Boundary::dom(op),
            BV::RotateLeft(op) => Boundary::dom(op),
            BV::RotateRight(op) => Boundary::dom(op),
            BV::Eq(op) => Boundary::dom(op),
            BV::Ne(op) => Boundary::dom(op),
            BV::SDiv(op) => Boundary::dom(op),
            BV::Abs(op) => Boundary::dom(op),
            BV::SRem(op) => Boundary::dom(op),
            BV::SMod(op) => Boundary::dom(op),
            BV::AShr(op) => Boundary::dom(op),
            BV::SignExtend(op) => Boundary::dom(op),
            BV::SLt(op) => Boundary::dom(op),
            BV::SLe(op) => Boundary::dom(op),
            BV::SGt(op) => Boundary::dom(op),
            BV::SGe(op) => Boundary::dom(op),
            BV::UDiv(op) => Boundary::dom(op),
            BV::URem(op) => Boundary::dom(op),
            BV::LShr(op) => Boundary::dom(op),
            BV::ZeroExtend(op) => Boundary::dom(op),
            BV::ULt(op) => Boundary::dom(op),
            BV::ULe(op) => Boundary::dom(op),
            BV::UGt(op) => Boundary::dom(op),
            BV::UGe(op) => Boundary::dom(op),
        }
    }

    fn cod(&self) -> Poly<Words> {
        match self {
            BV::Num(op) => Boundary::cod(op),
            BV::Concat(op) => Boundary::cod(op),
            BV::Extract(op) => Boundary::cod(op),
            BV::Repeat(op) => Boundary::cod(op),
            BV::Not(op) => Boundary::cod(op),
            BV::And(op) => Boundary::cod(op),
            BV::Or(op) => Boundary::cod(op),
            BV::Xor(op) => Boundary::cod(op),
            BV::Nand(op) => Boundary::cod(op),
            BV::Nor(op) => Boundary::cod(op),
            BV::Xnor(op) => Boundary::cod(op),
            BV::Neg(op) => Boundary::cod(op),
            BV::Ite(op) => Boundary::cod(op),
            BV::BroadcastIte(op) => Boundary::cod(op),
            BV::Add(op) => Boundary::cod(op),
            BV::Sub(op) => Boundary::cod(op),
            BV::Mul(op) => Boundary::cod(op),
            BV::Shl(op) => Boundary::cod(op),
            BV::RotateLeft(op) => Boundary::cod(op),
            BV::RotateRight(op) => Boundary::cod(op),
            BV::Eq(op) => Boundary::cod(op),
            BV::Ne(op) => Boundary::cod(op),
            BV::SDiv(op) => Boundary::cod(op),
            BV::Abs(op) => Boundary::cod(op),
            BV::SRem(op) => Boundary::cod(op),
            BV::SMod(op) => Boundary::cod(op),
            BV::AShr(op) => Boundary::cod(op),
            BV::SignExtend(op) => Boundary::cod(op),
            BV::SLt(op) => Boundary::cod(op),
            BV::SLe(op) => Boundary::cod(op),
            BV::SGt(op) => Boundary::cod(op),
            BV::SGe(op) => Boundary::cod(op),
            BV::UDiv(op) => Boundary::cod(op),
            BV::URem(op) => Boundary::cod(op),
            BV::LShr(op) => Boundary::cod(op),
            BV::ZeroExtend(op) => Boundary::cod(op),
            BV::ULt(op) => Boundary::cod(op),
            BV::ULe(op) => Boundary::cod(op),
            BV::UGt(op) => Boundary::cod(op),
            BV::UGe(op) => Boundary::cod(op),
        }
    }
}

impl<T: Matrix> Boundary<Words> for Num<T> {
    fn dom(&self) -> Poly<Words> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Words> {
        Num::cod(self).into()
    }
}

impl Boundary<Words> for Concat {
    fn dom(&self) -> Poly<Words> {
        Concat::dom(self).into()
    }

    fn cod(&self) -> Poly<Words> {
        Concat::cod(self).into()
    }
}

impl Boundary<Words> for Extract {
    fn dom(&self) -> Poly<Words> {
        Extract::dom(self).into()
    }

    fn cod(&self) -> Poly<Words> {
        Extract::cod(self).into()
    }
}

impl Boundary<Words> for Repeat {
    fn dom(&self) -> Poly<Words> {
        Repeat::dom(self).into()
    }

    fn cod(&self) -> Poly<Words> {
        Repeat::cod(self).into()
    }
}

impl Boundary<Words> for Not {
    fn dom(&self) -> Poly<Words> {
        Not::dom(self).into()
    }

    fn cod(&self) -> Poly<Words> {
        Not::cod(self).into()
    }
}

impl Boundary<Words> for And {
    fn dom(&self) -> Poly<Words> {
        And::dom(self).into()
    }

    fn cod(&self) -> Poly<Words> {
        And::cod(self).into()
    }
}

impl Boundary<Words> for Or {
    fn dom(&self) -> Poly<Words> {
        Or::dom(self).into()
    }

    fn cod(&self) -> Poly<Words> {
        Or::cod(self).into()
    }
}

impl Boundary<Words> for Xor {
    fn dom(&self) -> Poly<Words> {
        Xor::dom(self).into()
    }

    fn cod(&self) -> Poly<Words> {
        Xor::cod(self).into()
    }
}

impl Boundary<Words> for Nand {
    fn dom(&self) -> Poly<Words> {
        Nand::dom(self).into()
    }

    fn cod(&self) -> Poly<Words> {
        Nand::cod(self).into()
    }
}

impl Boundary<Words> for Nor {
    fn dom(&self) -> Poly<Words> {
        Nor::dom(self).into()
    }

    fn cod(&self) -> Poly<Words> {
        Nor::cod(self).into()
    }
}

impl Boundary<Words> for Xnor {
    fn dom(&self) -> Poly<Words> {
        Xnor::dom(self).into()
    }

    fn cod(&self) -> Poly<Words> {
        Xnor::cod(self).into()
    }
}

impl Boundary<Words> for Neg {
    fn dom(&self) -> Poly<Words> {
        Neg::dom(self).into()
    }

    fn cod(&self) -> Poly<Words> {
        Neg::cod(self).into()
    }
}

impl Boundary<Words> for Add {
    fn dom(&self) -> Poly<Words> {
        Add::dom(self).into()
    }

    fn cod(&self) -> Poly<Words> {
        Add::cod(self).into()
    }
}

impl Boundary<Words> for Sub {
    fn dom(&self) -> Poly<Words> {
        Sub::dom(self).into()
    }

    fn cod(&self) -> Poly<Words> {
        Sub::cod(self).into()
    }
}

impl Boundary<Words> for Mul {
    fn dom(&self) -> Poly<Words> {
        Mul::dom(self).into()
    }

    fn cod(&self) -> Poly<Words> {
        Mul::cod(self).into()
    }
}

impl Boundary<Words> for Shl {
    fn dom(&self) -> Poly<Words> {
        Shl::dom(self).into()
    }

    fn cod(&self) -> Poly<Words> {
        Shl::cod(self).into()
    }
}

impl Boundary<Words> for RotateLeft {
    fn dom(&self) -> Poly<Words> {
        RotateLeft::dom(self).into()
    }

    fn cod(&self) -> Poly<Words> {
        RotateLeft::cod(self).into()
    }
}

impl Boundary<Words> for RotateRight {
    fn dom(&self) -> Poly<Words> {
        RotateRight::dom(self).into()
    }

    fn cod(&self) -> Poly<Words> {
        RotateRight::cod(self).into()
    }
}

impl Boundary<Words> for Eq {
    fn dom(&self) -> Poly<Words> {
        Eq::dom(self).into()
    }

    fn cod(&self) -> Poly<Words> {
        Eq::cod(self).into()
    }
}

impl Boundary<Words> for Ne {
    fn dom(&self) -> Poly<Words> {
        Ne::dom(self).into()
    }

    fn cod(&self) -> Poly<Words> {
        Ne::cod(self).into()
    }
}

impl Boundary<Words> for Ite {
    fn dom(&self) -> Poly<Words> {
        Ite::dom(self).into()
    }

    fn cod(&self) -> Poly<Words> {
        Ite::cod(self).into()
    }
}

/// Operations on words read in two's complement, whose meaning depends on
/// signedness; the others are in the parent module.
pub mod signed {
    use super::{Word, Words};
    use crate::bool::Bool;
    use base::boundary::Mono;
    use base::boundary::{Boundary, Poly};
    use base::mono;
    use derive_more::Display;
    #[cfg(feature = "pyo3")]
    use pyo3::prelude::*;

    /// Pointwise division `x / y`, rounding towards zero, as `Div(width, [m, n])`;
    /// the minimum word divided by `-1` is itself, and `x / 0` is `1` if `x`
    /// is negative and `-1` otherwise.
    ///
    /// SMT-LIB2: `(bvsdiv x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
    #[display("SDiv")]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word.signed")
    )]
    pub struct Div {
        width: usize,
        shape: [usize; 2],
    }

    /// Pointwise absolute value of `width`-bit words read in two's complement, as
    /// `Abs(width, [m, n])`: `Ite(x < 0, -x, x)`, so the minimum word is its own
    /// absolute value.
    #[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
    #[display("Abs")]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word.signed")
    )]
    pub struct Abs {
        width: usize,
        shape: [usize; 2],
    }

    /// Pointwise remainder `x % y`, with the sign of `x`, as `Rem(width, [m, n])`;
    /// `x % 0 = x`.
    ///
    /// SMT-LIB2: `(bvsrem x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
    #[display("SRem")]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word.signed")
    )]
    pub struct Rem {
        width: usize,
        shape: [usize; 2],
    }

    /// Pointwise modulo `x mod y`, with the sign of `y`, as `Mod(width, [m, n])`;
    /// `x mod 0 = x`.
    ///
    /// SMT-LIB2: `(bvsmod x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
    #[display("SMod")]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word.signed")
    )]
    pub struct Mod {
        width: usize,
        shape: [usize; 2],
    }

    /// Pointwise arithmetic right shift `x >> y`, filling with the sign bit,
    /// as `Shr(width, [m, n])`.
    ///
    /// SMT-LIB2: `(bvashr x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
    #[display("AShr")]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word.signed")
    )]
    pub struct Shr {
        width: usize,
        shape: [usize; 2],
    }

    /// Pointwise extension of `width`-bit words by `i` copies of the sign bit,
    /// as `Extend(i, width, [m, n])`.
    ///
    /// SMT-LIB2: `((_ sign_extend i) x)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
    #[display("SignExtend({i})")]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word.signed")
    )]
    pub struct Extend {
        i: usize,
        width: usize,
        shape: [usize; 2],
    }

    /// Pointwise signed comparison `x < y`, as `Lt(width, [m, n])`.
    ///
    /// SMT-LIB2: `(bvslt x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
    #[display("SLt")]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word.signed")
    )]
    pub struct Lt {
        width: usize,
        shape: [usize; 2],
    }

    /// Pointwise signed comparison `x ≤ y`, as `Le(width, [m, n])`.
    ///
    /// SMT-LIB2: `(bvsle x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
    #[display("SLe")]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word.signed")
    )]
    pub struct Le {
        width: usize,
        shape: [usize; 2],
    }

    /// Pointwise signed comparison `x > y`, as `Gt(width, [m, n])`.
    ///
    /// SMT-LIB2: `(bvsgt x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
    #[display("SGt")]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word.signed")
    )]
    pub struct Gt {
        width: usize,
        shape: [usize; 2],
    }

    /// Pointwise signed comparison `x ≥ y`, as `Ge(width, [m, n])`.
    ///
    /// SMT-LIB2: `(bvsge x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
    #[display("SGe")]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word.signed")
    )]
    pub struct Ge {
        width: usize,
        shape: [usize; 2],
    }

    impl Boundary<Words> for Div {
        fn dom(&self) -> Poly<Words> {
            Div::dom(self).into()
        }

        fn cod(&self) -> Poly<Words> {
            Div::cod(self).into()
        }
    }

    impl Boundary<Words> for Rem {
        fn dom(&self) -> Poly<Words> {
            Rem::dom(self).into()
        }

        fn cod(&self) -> Poly<Words> {
            Rem::cod(self).into()
        }
    }

    impl Boundary<Words> for Mod {
        fn dom(&self) -> Poly<Words> {
            Mod::dom(self).into()
        }

        fn cod(&self) -> Poly<Words> {
            Mod::cod(self).into()
        }
    }

    impl Boundary<Words> for Shr {
        fn dom(&self) -> Poly<Words> {
            Shr::dom(self).into()
        }

        fn cod(&self) -> Poly<Words> {
            Shr::cod(self).into()
        }
    }

    impl Boundary<Words> for Extend {
        fn dom(&self) -> Poly<Words> {
            Extend::dom(self).into()
        }

        fn cod(&self) -> Poly<Words> {
            Extend::cod(self).into()
        }
    }

    impl Boundary<Words> for Lt {
        fn dom(&self) -> Poly<Words> {
            Lt::dom(self).into()
        }

        fn cod(&self) -> Poly<Words> {
            Lt::cod(self).into()
        }
    }

    impl Boundary<Words> for Le {
        fn dom(&self) -> Poly<Words> {
            Le::dom(self).into()
        }

        fn cod(&self) -> Poly<Words> {
            Le::cod(self).into()
        }
    }

    impl Boundary<Words> for Gt {
        fn dom(&self) -> Poly<Words> {
            Gt::dom(self).into()
        }

        fn cod(&self) -> Poly<Words> {
            Gt::cod(self).into()
        }
    }

    impl Boundary<Words> for Ge {
        fn dom(&self) -> Poly<Words> {
            Ge::dom(self).into()
        }

        fn cod(&self) -> Poly<Words> {
            Ge::cod(self).into()
        }
    }

    impl Boundary<Words> for Abs {
        fn dom(&self) -> Poly<Words> {
            Abs::dom(self).into()
        }

        fn cod(&self) -> Poly<Words> {
            Abs::cod(self).into()
        }
    }

    impl Div {
        /// The operation reading the words of `dom`; fails unless they are equal.
        pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
            let s = crate::same_sort(dom)?;
            Ok(Div {
                width: s.width(),
                shape: s.shape(),
            })
        }

        /// The operation writing the words of `cod`.
        pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
            let [s] = cod;
            Ok(Div {
                width: s.width(),
                shape: s.shape(),
            })
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
            crate::check_cod(Self::with_dom(dom), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str) {
            ("dom", "cod")
        }

        /// The sorts it reads, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn dom(&self) -> Mono<Words> {
            mono![Word::new_unchecked(self.width, self.shape).into(); 2]
        }

        /// The sorts it writes, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn cod(&self) -> Mono<Words> {
            mono![Word::new_unchecked(self.width, self.shape).into()]
        }
    }

    impl Abs {
        /// The operation reading the words of `dom`.
        pub fn with_dom(dom: [Word; 1]) -> Result<Self, crate::Error> {
            let s = crate::same_sort(dom)?;
            Ok(Abs {
                width: s.width(),
                shape: s.shape(),
            })
        }

        /// The operation writing the words of `cod`.
        pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
            let [s] = cod;
            Ok(Abs {
                width: s.width(),
                shape: s.shape(),
            })
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
            crate::check_cod(Self::with_dom([dom]), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str) {
            ("dom", "cod")
        }

        /// The sorts it reads, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn dom(&self) -> Mono<Words> {
            mono![Word::new_unchecked(self.width, self.shape).into()]
        }

        /// The sorts it writes, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn cod(&self) -> Mono<Words> {
            mono![Word::new_unchecked(self.width, self.shape).into()]
        }
    }

    impl Rem {
        /// The operation reading the words of `dom`; fails unless they are equal.
        pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
            let s = crate::same_sort(dom)?;
            Ok(Rem {
                width: s.width(),
                shape: s.shape(),
            })
        }

        /// The operation writing the words of `cod`.
        pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
            let [s] = cod;
            Ok(Rem {
                width: s.width(),
                shape: s.shape(),
            })
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
            crate::check_cod(Self::with_dom(dom), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str) {
            ("dom", "cod")
        }

        /// The sorts it reads, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn dom(&self) -> Mono<Words> {
            mono![Word::new_unchecked(self.width, self.shape).into(); 2]
        }

        /// The sorts it writes, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn cod(&self) -> Mono<Words> {
            mono![Word::new_unchecked(self.width, self.shape).into()]
        }
    }

    impl Mod {
        /// The operation reading the words of `dom`; fails unless they are equal.
        pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
            let s = crate::same_sort(dom)?;
            Ok(Mod {
                width: s.width(),
                shape: s.shape(),
            })
        }

        /// The operation writing the words of `cod`.
        pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
            let [s] = cod;
            Ok(Mod {
                width: s.width(),
                shape: s.shape(),
            })
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
            crate::check_cod(Self::with_dom(dom), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str) {
            ("dom", "cod")
        }

        /// The sorts it reads, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn dom(&self) -> Mono<Words> {
            mono![Word::new_unchecked(self.width, self.shape).into(); 2]
        }

        /// The sorts it writes, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn cod(&self) -> Mono<Words> {
            mono![Word::new_unchecked(self.width, self.shape).into()]
        }
    }

    impl Shr {
        /// The operation reading the words of `dom`; fails unless they are equal.
        pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
            let s = crate::same_sort(dom)?;
            Ok(Shr {
                width: s.width(),
                shape: s.shape(),
            })
        }

        /// The operation writing the words of `cod`.
        pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
            let [s] = cod;
            Ok(Shr {
                width: s.width(),
                shape: s.shape(),
            })
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
            crate::check_cod(Self::with_dom(dom), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str) {
            ("dom", "cod")
        }

        /// The sorts it reads, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn dom(&self) -> Mono<Words> {
            mono![Word::new_unchecked(self.width, self.shape).into(); 2]
        }

        /// The sorts it writes, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn cod(&self) -> Mono<Words> {
            mono![Word::new_unchecked(self.width, self.shape).into()]
        }
    }

    impl Extend {
        /// The operation of indices `i`, reading the words of `dom`.
        pub fn with_dom(i: usize, dom: [Word; 1]) -> Result<Self, crate::Error> {
            let [s] = dom;
            let width = s.width();
            if width.checked_add(i).is_none() {
                return Err(crate::Error(format!(
                    "cannot extend words of {width} bits by {i} bits: the result would have more than {} bits",
                    usize::MAX
                )));
            }
            Ok(Extend {
                i,
                width,
                shape: s.shape(),
            })
        }

        /// The extension by `i` bits writing the words of `cod`; fails unless
        /// their width exceeds `i`.
        pub fn with_cod(i: usize, cod: [Word; 1]) -> Result<Self, crate::Error> {
            let [s] = cod;
            match s.width().checked_sub(i) {
                Some(width) if width > 0 => Ok(Extend {
                    i,
                    width,
                    shape: s.shape(),
                }),
                _ => Err(crate::unexpected_cod(
                    &format!("1 sort of more than {i} bits"),
                    &cod,
                )),
            }
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
            crate::check_cod(Self::with_dom(i, [dom]), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str, &'static str) {
            ("i", "dom", "cod")
        }

        /// The sorts it reads, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn dom(&self) -> Mono<Words> {
            mono![Word::new_unchecked(self.width, self.shape).into()]
        }

        /// The sorts it writes, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn cod(&self) -> Mono<Words> {
            mono![Word::new_unchecked(self.width + self.i, self.shape).into()]
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn i(&self) -> usize {
            self.i
        }
    }

    impl Lt {
        /// The operation reading the words of `dom`; fails unless they are equal.
        pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
            let s = crate::same_sort(dom)?;
            Ok(Lt {
                width: s.width(),
                shape: s.shape(),
            })
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
            crate::check_cod(Self::with_dom(dom), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str) {
            ("dom", "cod")
        }

        /// The sorts it reads, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn dom(&self) -> Mono<Words> {
            mono![Word::new_unchecked(self.width, self.shape).into(); 2]
        }

        /// The sorts it writes, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn cod(&self) -> Mono<Words> {
            mono![Bool::new_unchecked(self.shape).into()]
        }
    }

    impl Le {
        /// The operation reading the words of `dom`; fails unless they are equal.
        pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
            let s = crate::same_sort(dom)?;
            Ok(Le {
                width: s.width(),
                shape: s.shape(),
            })
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
            crate::check_cod(Self::with_dom(dom), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str) {
            ("dom", "cod")
        }

        /// The sorts it reads, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn dom(&self) -> Mono<Words> {
            mono![Word::new_unchecked(self.width, self.shape).into(); 2]
        }

        /// The sorts it writes, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn cod(&self) -> Mono<Words> {
            mono![Bool::new_unchecked(self.shape).into()]
        }
    }

    impl Gt {
        /// The operation reading the words of `dom`; fails unless they are equal.
        pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
            let s = crate::same_sort(dom)?;
            Ok(Gt {
                width: s.width(),
                shape: s.shape(),
            })
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
            crate::check_cod(Self::with_dom(dom), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str) {
            ("dom", "cod")
        }

        /// The sorts it reads, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn dom(&self) -> Mono<Words> {
            mono![Word::new_unchecked(self.width, self.shape).into(); 2]
        }

        /// The sorts it writes, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn cod(&self) -> Mono<Words> {
            mono![Bool::new_unchecked(self.shape).into()]
        }
    }

    impl Ge {
        /// The operation reading the words of `dom`; fails unless they are equal.
        pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
            let s = crate::same_sort(dom)?;
            Ok(Ge {
                width: s.width(),
                shape: s.shape(),
            })
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
            crate::check_cod(Self::with_dom(dom), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str) {
            ("dom", "cod")
        }

        /// The sorts it reads, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn dom(&self) -> Mono<Words> {
            mono![Word::new_unchecked(self.width, self.shape).into(); 2]
        }

        /// The sorts it writes, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn cod(&self) -> Mono<Words> {
            mono![Bool::new_unchecked(self.shape).into()]
        }
    }
}

/// Operations on words read in binary, whose meaning depends on
/// signedness; the others are in the parent module.
pub mod unsigned {
    use super::{Word, Words};
    use crate::bool::Bool;
    use base::boundary::Mono;
    use base::boundary::{Boundary, Poly};
    use base::mono;
    use derive_more::Display;
    #[cfg(feature = "pyo3")]
    use pyo3::prelude::*;

    /// Pointwise division `x / y`, rounding down, as `Div(width, [m, n])`; `x / 0`
    /// is the word of all ones.
    ///
    /// SMT-LIB2: `(bvudiv x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
    #[display("UDiv")]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word.unsigned")
    )]
    pub struct Div {
        width: usize,
        shape: [usize; 2],
    }

    /// Pointwise remainder `x % y`, as `Rem(width, [m, n])`; `x % 0 = x`.
    ///
    /// SMT-LIB2: `(bvurem x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
    #[display("URem")]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word.unsigned")
    )]
    pub struct Rem {
        width: usize,
        shape: [usize; 2],
    }

    /// Pointwise logical right shift `x >> y`, filling with zeros, as
    /// `Shr(width, [m, n])`.
    ///
    /// SMT-LIB2: `(bvlshr x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
    #[display("LShr")]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word.unsigned")
    )]
    pub struct Shr {
        width: usize,
        shape: [usize; 2],
    }

    /// Pointwise extension of `width`-bit words by `i` zero bits, as
    /// `Extend(i, width, [m, n])`.
    ///
    /// SMT-LIB2: `((_ zero_extend i) x)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
    #[display("ZeroExtend({i})")]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word.unsigned")
    )]
    pub struct Extend {
        i: usize,
        width: usize,
        shape: [usize; 2],
    }

    /// Pointwise unsigned comparison `x < y`, as `Lt(width, [m, n])`.
    ///
    /// SMT-LIB2: `(bvult x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
    #[display("ULt")]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word.unsigned")
    )]
    pub struct Lt {
        width: usize,
        shape: [usize; 2],
    }

    /// Pointwise unsigned comparison `x ≤ y`, as `Le(width, [m, n])`.
    ///
    /// SMT-LIB2: `(bvule x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
    #[display("ULe")]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word.unsigned")
    )]
    pub struct Le {
        width: usize,
        shape: [usize; 2],
    }

    /// Pointwise unsigned comparison `x > y`, as `Gt(width, [m, n])`.
    ///
    /// SMT-LIB2: `(bvugt x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
    #[display("UGt")]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word.unsigned")
    )]
    pub struct Gt {
        width: usize,
        shape: [usize; 2],
    }

    /// Pointwise unsigned comparison `x ≥ y`, as `Ge(width, [m, n])`.
    ///
    /// SMT-LIB2: `(bvuge x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
    #[display("UGe")]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word.unsigned")
    )]
    pub struct Ge {
        width: usize,
        shape: [usize; 2],
    }

    impl Boundary<Words> for Div {
        fn dom(&self) -> Poly<Words> {
            Div::dom(self).into()
        }

        fn cod(&self) -> Poly<Words> {
            Div::cod(self).into()
        }
    }

    impl Boundary<Words> for Rem {
        fn dom(&self) -> Poly<Words> {
            Rem::dom(self).into()
        }

        fn cod(&self) -> Poly<Words> {
            Rem::cod(self).into()
        }
    }

    impl Boundary<Words> for Shr {
        fn dom(&self) -> Poly<Words> {
            Shr::dom(self).into()
        }

        fn cod(&self) -> Poly<Words> {
            Shr::cod(self).into()
        }
    }

    impl Boundary<Words> for Extend {
        fn dom(&self) -> Poly<Words> {
            Extend::dom(self).into()
        }

        fn cod(&self) -> Poly<Words> {
            Extend::cod(self).into()
        }
    }

    impl Boundary<Words> for Lt {
        fn dom(&self) -> Poly<Words> {
            Lt::dom(self).into()
        }

        fn cod(&self) -> Poly<Words> {
            Lt::cod(self).into()
        }
    }

    impl Boundary<Words> for Le {
        fn dom(&self) -> Poly<Words> {
            Le::dom(self).into()
        }

        fn cod(&self) -> Poly<Words> {
            Le::cod(self).into()
        }
    }

    impl Boundary<Words> for Gt {
        fn dom(&self) -> Poly<Words> {
            Gt::dom(self).into()
        }

        fn cod(&self) -> Poly<Words> {
            Gt::cod(self).into()
        }
    }

    impl Boundary<Words> for Ge {
        fn dom(&self) -> Poly<Words> {
            Ge::dom(self).into()
        }

        fn cod(&self) -> Poly<Words> {
            Ge::cod(self).into()
        }
    }

    impl Div {
        /// The operation reading the words of `dom`; fails unless they are equal.
        pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
            let s = crate::same_sort(dom)?;
            Ok(Div {
                width: s.width(),
                shape: s.shape(),
            })
        }

        /// The operation writing the words of `cod`.
        pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
            let [s] = cod;
            Ok(Div {
                width: s.width(),
                shape: s.shape(),
            })
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
            crate::check_cod(Self::with_dom(dom), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str) {
            ("dom", "cod")
        }

        /// The sorts it reads, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn dom(&self) -> Mono<Words> {
            mono![Word::new_unchecked(self.width, self.shape).into(); 2]
        }

        /// The sorts it writes, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn cod(&self) -> Mono<Words> {
            mono![Word::new_unchecked(self.width, self.shape).into()]
        }
    }

    impl Rem {
        /// The operation reading the words of `dom`; fails unless they are equal.
        pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
            let s = crate::same_sort(dom)?;
            Ok(Rem {
                width: s.width(),
                shape: s.shape(),
            })
        }

        /// The operation writing the words of `cod`.
        pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
            let [s] = cod;
            Ok(Rem {
                width: s.width(),
                shape: s.shape(),
            })
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
            crate::check_cod(Self::with_dom(dom), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str) {
            ("dom", "cod")
        }

        /// The sorts it reads, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn dom(&self) -> Mono<Words> {
            mono![Word::new_unchecked(self.width, self.shape).into(); 2]
        }

        /// The sorts it writes, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn cod(&self) -> Mono<Words> {
            mono![Word::new_unchecked(self.width, self.shape).into()]
        }
    }

    impl Shr {
        /// The operation reading the words of `dom`; fails unless they are equal.
        pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
            let s = crate::same_sort(dom)?;
            Ok(Shr {
                width: s.width(),
                shape: s.shape(),
            })
        }

        /// The operation writing the words of `cod`.
        pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
            let [s] = cod;
            Ok(Shr {
                width: s.width(),
                shape: s.shape(),
            })
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
            crate::check_cod(Self::with_dom(dom), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str) {
            ("dom", "cod")
        }

        /// The sorts it reads, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn dom(&self) -> Mono<Words> {
            mono![Word::new_unchecked(self.width, self.shape).into(); 2]
        }

        /// The sorts it writes, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn cod(&self) -> Mono<Words> {
            mono![Word::new_unchecked(self.width, self.shape).into()]
        }
    }

    impl Extend {
        /// The operation of indices `i`, reading the words of `dom`.
        pub fn with_dom(i: usize, dom: [Word; 1]) -> Result<Self, crate::Error> {
            let [s] = dom;
            let width = s.width();
            if width.checked_add(i).is_none() {
                return Err(crate::Error(format!(
                    "cannot extend words of {width} bits by {i} bits: the result would have more than {} bits",
                    usize::MAX
                )));
            }
            Ok(Extend {
                i,
                width,
                shape: s.shape(),
            })
        }

        /// The extension by `i` bits writing the words of `cod`; fails unless
        /// their width exceeds `i`.
        pub fn with_cod(i: usize, cod: [Word; 1]) -> Result<Self, crate::Error> {
            let [s] = cod;
            match s.width().checked_sub(i) {
                Some(width) if width > 0 => Ok(Extend {
                    i,
                    width,
                    shape: s.shape(),
                }),
                _ => Err(crate::unexpected_cod(
                    &format!("1 sort of more than {i} bits"),
                    &cod,
                )),
            }
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
            crate::check_cod(Self::with_dom(i, [dom]), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str, &'static str) {
            ("i", "dom", "cod")
        }

        /// The sorts it reads, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn dom(&self) -> Mono<Words> {
            mono![Word::new_unchecked(self.width, self.shape).into()]
        }

        /// The sorts it writes, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn cod(&self) -> Mono<Words> {
            mono![Word::new_unchecked(self.width + self.i, self.shape).into()]
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn i(&self) -> usize {
            self.i
        }
    }

    impl Lt {
        /// The operation reading the words of `dom`; fails unless they are equal.
        pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
            let s = crate::same_sort(dom)?;
            Ok(Lt {
                width: s.width(),
                shape: s.shape(),
            })
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
            crate::check_cod(Self::with_dom(dom), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str) {
            ("dom", "cod")
        }

        /// The sorts it reads, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn dom(&self) -> Mono<Words> {
            mono![Word::new_unchecked(self.width, self.shape).into(); 2]
        }

        /// The sorts it writes, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn cod(&self) -> Mono<Words> {
            mono![Bool::new_unchecked(self.shape).into()]
        }
    }

    impl Le {
        /// The operation reading the words of `dom`; fails unless they are equal.
        pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
            let s = crate::same_sort(dom)?;
            Ok(Le {
                width: s.width(),
                shape: s.shape(),
            })
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
            crate::check_cod(Self::with_dom(dom), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str) {
            ("dom", "cod")
        }

        /// The sorts it reads, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn dom(&self) -> Mono<Words> {
            mono![Word::new_unchecked(self.width, self.shape).into(); 2]
        }

        /// The sorts it writes, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn cod(&self) -> Mono<Words> {
            mono![Bool::new_unchecked(self.shape).into()]
        }
    }

    impl Gt {
        /// The operation reading the words of `dom`; fails unless they are equal.
        pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
            let s = crate::same_sort(dom)?;
            Ok(Gt {
                width: s.width(),
                shape: s.shape(),
            })
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
            crate::check_cod(Self::with_dom(dom), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str) {
            ("dom", "cod")
        }

        /// The sorts it reads, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn dom(&self) -> Mono<Words> {
            mono![Word::new_unchecked(self.width, self.shape).into(); 2]
        }

        /// The sorts it writes, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn cod(&self) -> Mono<Words> {
            mono![Bool::new_unchecked(self.shape).into()]
        }
    }

    impl Ge {
        /// The operation reading the words of `dom`; fails unless they are equal.
        pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
            let s = crate::same_sort(dom)?;
            Ok(Ge {
                width: s.width(),
                shape: s.shape(),
            })
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
            crate::check_cod(Self::with_dom(dom), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str) {
            ("dom", "cod")
        }

        /// The sorts it reads, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn dom(&self) -> Mono<Words> {
            mono![Word::new_unchecked(self.width, self.shape).into(); 2]
        }

        /// The sorts it writes, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn cod(&self) -> Mono<Words> {
            mono![Bool::new_unchecked(self.shape).into()]
        }
    }
}

/// Extracts an operation of fixed-size bit-vectors from a Python object,
/// trying each operation in turn: those carrying no data through their own
/// classes, and [`Num`] through the extraction that the tensor type `T`
/// provides.
#[cfg(feature = "pyo3")]
impl<'a, 'py, T> pyo3::FromPyObject<'a, 'py> for BV<T>
where
    Num<T>: pyo3::FromPyObject<'a, 'py>,
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
        if let Ok(op) = ob.extract::<broadcast::Ite>() {
            return Ok(BV::BroadcastIte(op));
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
        if let Ok(op) = ob.extract::<Num<T>>() {
            return Ok(BV::Num(op));
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
            return Err(crate::unexpected_dom("2 sorts of the same shape", &dom));
        }
        let [u, v] = [x.width(), y.width()];
        if u.checked_add(v).is_none() {
            return Err(crate::Error(format!(
                "cannot concatenate words of {u} and {v} bits: the result would have more than {} bits",
                usize::MAX
            )));
        }
        Ok(Concat {
            widths: [u, v],
            shape: x.shape(),
        })
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
        crate::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn dom(&self) -> Mono<Words> {
        let ([u, v], shape) = (self.widths, self.shape);
        mono![
            Word::new_unchecked(u, shape).into(),
            Word::new_unchecked(v, shape).into()
        ]
    }

    /// The sorts it writes, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn cod(&self) -> Mono<Words> {
        let [u, v] = self.widths;
        mono![Word::new_unchecked(u + v, self.shape).into()]
    }
}

impl Extract {
    /// The operation of indices `i`, `j`, reading the words of `dom`.
    pub fn with_dom(i: usize, j: usize, dom: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = dom;
        let width = s.width();
        if !(j <= i && i < width) {
            return Err(crate::Error(format!(
                "cannot extract the bits {i} down to {j} of {width}-bit words"
            )));
        }
        Ok(Extract {
            i,
            j,
            width,
            shape: s.shape(),
        })
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
        crate::check_cod(Self::with_dom(i, j, [dom]), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str, &'static str, &'static str) {
        ("i", "j", "dom", "cod")
    }

    /// The sorts it reads, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn dom(&self) -> Mono<Words> {
        mono![Word::new_unchecked(self.width, self.shape).into()]
    }

    /// The sorts it writes, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn cod(&self) -> Mono<Words> {
        mono![Word::new_unchecked(self.i - self.j + 1, self.shape).into()]
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn i(&self) -> usize {
        self.i
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn j(&self) -> usize {
        self.j
    }
}

impl Repeat {
    /// The operation of indices `i`, reading the words of `dom`.
    pub fn with_dom(i: usize, dom: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = dom;
        let width = s.width();
        if i == 0 {
            return Err(crate::Error("cannot repeat a word zero times".to_string()));
        }
        if i.checked_mul(width).is_none() {
            return Err(crate::Error(format!(
                "cannot repeat words of {width} bits {i} times: the result would have more than {} bits",
                usize::MAX
            )));
        }
        Ok(Repeat {
            i,
            width,
            shape: s.shape(),
        })
    }

    /// The repetition `i` times writing the words of `cod`; fails unless `i`
    /// divides their width.
    pub fn with_cod(i: usize, cod: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        if i == 0 || s.width() % i != 0 {
            return Err(crate::unexpected_cod(
                &format!("1 sort of a width that {i} divides"),
                &cod,
            ));
        }
        Ok(Repeat {
            i,
            width: s.width() / i,
            shape: s.shape(),
        })
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
        crate::check_cod(Self::with_dom(i, [dom]), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str, &'static str) {
        ("i", "dom", "cod")
    }

    /// The sorts it reads, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn dom(&self) -> Mono<Words> {
        mono![Word::new_unchecked(self.width, self.shape).into()]
    }

    /// The sorts it writes, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn cod(&self) -> Mono<Words> {
        mono![Word::new_unchecked(self.i * self.width, self.shape).into()]
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn i(&self) -> usize {
        self.i
    }
}

impl Not {
    /// The operation reading the words of `dom`.
    pub fn with_dom(dom: [Word; 1]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Not {
            width: s.width(),
            shape: s.shape(),
        })
    }

    /// The operation writing the words of `cod`.
    pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Not {
            width: s.width(),
            shape: s.shape(),
        })
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
        crate::check_cod(Self::with_dom([dom]), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn dom(&self) -> Mono<Words> {
        mono![Word::new_unchecked(self.width, self.shape).into()]
    }

    /// The sorts it writes, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn cod(&self) -> Mono<Words> {
        mono![Word::new_unchecked(self.width, self.shape).into()]
    }
}

impl And {
    /// The operation reading the words of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(And {
            width: s.width(),
            shape: s.shape(),
        })
    }

    /// The operation writing the words of `cod`.
    pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(And {
            width: s.width(),
            shape: s.shape(),
        })
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
        crate::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn dom(&self) -> Mono<Words> {
        mono![Word::new_unchecked(self.width, self.shape).into(); 2]
    }

    /// The sorts it writes, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn cod(&self) -> Mono<Words> {
        mono![Word::new_unchecked(self.width, self.shape).into()]
    }
}

impl Or {
    /// The operation reading the words of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Or {
            width: s.width(),
            shape: s.shape(),
        })
    }

    /// The operation writing the words of `cod`.
    pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Or {
            width: s.width(),
            shape: s.shape(),
        })
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
        crate::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn dom(&self) -> Mono<Words> {
        mono![Word::new_unchecked(self.width, self.shape).into(); 2]
    }

    /// The sorts it writes, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn cod(&self) -> Mono<Words> {
        mono![Word::new_unchecked(self.width, self.shape).into()]
    }
}

impl Xor {
    /// The operation reading the words of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Xor {
            width: s.width(),
            shape: s.shape(),
        })
    }

    /// The operation writing the words of `cod`.
    pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Xor {
            width: s.width(),
            shape: s.shape(),
        })
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
        crate::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn dom(&self) -> Mono<Words> {
        mono![Word::new_unchecked(self.width, self.shape).into(); 2]
    }

    /// The sorts it writes, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn cod(&self) -> Mono<Words> {
        mono![Word::new_unchecked(self.width, self.shape).into()]
    }
}

impl Nand {
    /// The operation reading the words of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Nand {
            width: s.width(),
            shape: s.shape(),
        })
    }

    /// The operation writing the words of `cod`.
    pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Nand {
            width: s.width(),
            shape: s.shape(),
        })
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
        crate::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn dom(&self) -> Mono<Words> {
        mono![Word::new_unchecked(self.width, self.shape).into(); 2]
    }

    /// The sorts it writes, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn cod(&self) -> Mono<Words> {
        mono![Word::new_unchecked(self.width, self.shape).into()]
    }
}

impl Nor {
    /// The operation reading the words of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Nor {
            width: s.width(),
            shape: s.shape(),
        })
    }

    /// The operation writing the words of `cod`.
    pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Nor {
            width: s.width(),
            shape: s.shape(),
        })
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
        crate::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn dom(&self) -> Mono<Words> {
        mono![Word::new_unchecked(self.width, self.shape).into(); 2]
    }

    /// The sorts it writes, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn cod(&self) -> Mono<Words> {
        mono![Word::new_unchecked(self.width, self.shape).into()]
    }
}

impl Xnor {
    /// The operation reading the words of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Xnor {
            width: s.width(),
            shape: s.shape(),
        })
    }

    /// The operation writing the words of `cod`.
    pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Xnor {
            width: s.width(),
            shape: s.shape(),
        })
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
        crate::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn dom(&self) -> Mono<Words> {
        mono![Word::new_unchecked(self.width, self.shape).into(); 2]
    }

    /// The sorts it writes, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn cod(&self) -> Mono<Words> {
        mono![Word::new_unchecked(self.width, self.shape).into()]
    }
}

impl Neg {
    /// The operation reading the words of `dom`.
    pub fn with_dom(dom: [Word; 1]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Neg {
            width: s.width(),
            shape: s.shape(),
        })
    }

    /// The operation writing the words of `cod`.
    pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Neg {
            width: s.width(),
            shape: s.shape(),
        })
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
        crate::check_cod(Self::with_dom([dom]), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn dom(&self) -> Mono<Words> {
        mono![Word::new_unchecked(self.width, self.shape).into()]
    }

    /// The sorts it writes, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn cod(&self) -> Mono<Words> {
        mono![Word::new_unchecked(self.width, self.shape).into()]
    }
}

impl Ite {
    /// The choice reading the boolean condition and the branches of `dom`;
    /// fails unless the branches are of the same sort, and the condition of
    /// their shape.
    pub fn with_dom(dom: [Words; 3]) -> Result<Self, crate::Error> {
        match dom {
            [Words::Bool(c), Words::Word(x), Words::Word(y)]
                if x == y && c.shape() == x.shape() =>
            {
                Ok(Ite {
                    width: x.width(),
                    shape: x.shape(),
                })
            }
            _ => Err(crate::unexpected_dom(
                "a condition Bool and two branches Word of its shape",
                &dom,
            )),
        }
    }

    /// The operation writing the words of `cod`.
    pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Ite {
            width: s.width(),
            shape: s.shape(),
        })
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
    fn py_new(dom: [Words; 3], cod: Option<Word>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn dom(&self) -> Mono<Words> {
        mono![
            Bool::new_unchecked(self.shape).into(),
            Word::new_unchecked(self.width, self.shape).into(),
            Word::new_unchecked(self.width, self.shape).into()
        ]
    }

    /// The sorts it writes, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn cod(&self) -> Mono<Words> {
        mono![Word::new_unchecked(self.width, self.shape).into()]
    }
}

impl Add {
    /// The operation reading the words of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Add {
            width: s.width(),
            shape: s.shape(),
        })
    }

    /// The operation writing the words of `cod`.
    pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Add {
            width: s.width(),
            shape: s.shape(),
        })
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
        crate::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn dom(&self) -> Mono<Words> {
        mono![Word::new_unchecked(self.width, self.shape).into(); 2]
    }

    /// The sorts it writes, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn cod(&self) -> Mono<Words> {
        mono![Word::new_unchecked(self.width, self.shape).into()]
    }
}

impl Sub {
    /// The operation reading the words of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Sub {
            width: s.width(),
            shape: s.shape(),
        })
    }

    /// The operation writing the words of `cod`.
    pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Sub {
            width: s.width(),
            shape: s.shape(),
        })
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
        crate::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn dom(&self) -> Mono<Words> {
        mono![Word::new_unchecked(self.width, self.shape).into(); 2]
    }

    /// The sorts it writes, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn cod(&self) -> Mono<Words> {
        mono![Word::new_unchecked(self.width, self.shape).into()]
    }
}

impl Mul {
    /// The operation reading the words of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Mul {
            width: s.width(),
            shape: s.shape(),
        })
    }

    /// The operation writing the words of `cod`.
    pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Mul {
            width: s.width(),
            shape: s.shape(),
        })
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
        crate::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn dom(&self) -> Mono<Words> {
        mono![Word::new_unchecked(self.width, self.shape).into(); 2]
    }

    /// The sorts it writes, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn cod(&self) -> Mono<Words> {
        mono![Word::new_unchecked(self.width, self.shape).into()]
    }
}

impl Shl {
    /// The operation reading the words of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Shl {
            width: s.width(),
            shape: s.shape(),
        })
    }

    /// The operation writing the words of `cod`.
    pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Shl {
            width: s.width(),
            shape: s.shape(),
        })
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
        crate::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn dom(&self) -> Mono<Words> {
        mono![Word::new_unchecked(self.width, self.shape).into(); 2]
    }

    /// The sorts it writes, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn cod(&self) -> Mono<Words> {
        mono![Word::new_unchecked(self.width, self.shape).into()]
    }
}

impl RotateLeft {
    /// The operation of indices `i`, reading the words of `dom`.
    pub fn with_dom(i: usize, dom: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = dom;
        Ok(RotateLeft {
            i,
            width: s.width(),
            shape: s.shape(),
        })
    }

    /// The operation of index `i`, writing the words of `cod`.
    pub fn with_cod(i: usize, cod: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(RotateLeft {
            i,
            width: s.width(),
            shape: s.shape(),
        })
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
        crate::check_cod(Self::with_dom(i, [dom]), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str, &'static str) {
        ("i", "dom", "cod")
    }

    /// The sorts it reads, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn dom(&self) -> Mono<Words> {
        mono![Word::new_unchecked(self.width, self.shape).into()]
    }

    /// The sorts it writes, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn cod(&self) -> Mono<Words> {
        mono![Word::new_unchecked(self.width, self.shape).into()]
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn i(&self) -> usize {
        self.i
    }
}

impl RotateRight {
    /// The operation of indices `i`, reading the words of `dom`.
    pub fn with_dom(i: usize, dom: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = dom;
        Ok(RotateRight {
            i,
            width: s.width(),
            shape: s.shape(),
        })
    }

    /// The operation of index `i`, writing the words of `cod`.
    pub fn with_cod(i: usize, cod: [Word; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(RotateRight {
            i,
            width: s.width(),
            shape: s.shape(),
        })
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
        crate::check_cod(Self::with_dom(i, [dom]), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str, &'static str) {
        ("i", "dom", "cod")
    }

    /// The sorts it reads, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn dom(&self) -> Mono<Words> {
        mono![Word::new_unchecked(self.width, self.shape).into()]
    }

    /// The sorts it writes, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn cod(&self) -> Mono<Words> {
        mono![Word::new_unchecked(self.width, self.shape).into()]
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn i(&self) -> usize {
        self.i
    }
}

impl Eq {
    /// The operation reading the words of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Eq {
            width: s.width(),
            shape: s.shape(),
        })
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
        crate::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn dom(&self) -> Mono<Words> {
        mono![Word::new_unchecked(self.width, self.shape).into(); 2]
    }

    /// The sorts it writes, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn cod(&self) -> Mono<Words> {
        mono![Bool::new_unchecked(self.shape).into()]
    }
}

impl Ne {
    /// The operation reading the words of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Word; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Ne {
            width: s.width(),
            shape: s.shape(),
        })
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
        crate::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn dom(&self) -> Mono<Words> {
        mono![Word::new_unchecked(self.width, self.shape).into(); 2]
    }

    /// The sorts it writes, as a tuple.
    #[cfg_attr(feature = "pyo3", getter)]
    fn cod(&self) -> Mono<Words> {
        mono![Bool::new_unchecked(self.shape).into()]
    }
}

/// The operations of the theory that broadcast a scalar operand, of the
/// shape `[1, 1]`, to the shape of the others; the others are in the parent
/// module.
pub mod broadcast {
    use super::*;

    /// Choice `if c then x else y` between whole `[m, n]` matrices of `width`-bit words `x` and
    /// `y`, by the scalar boolean `c`, of the shape `[1, 1]`, read first: the
    /// pointwise [`super::Ite`] by `c` broadcast to `[m, n]`.
    ///
    /// SMT-LIB2: `(ite c x y)`, by the only entry of `c`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
    #[display("Ite")]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.word.broadcast")
    )]
    pub struct Ite {
        width: usize,
        shape: [usize; 2],
    }

    impl Boundary<Words> for Ite {
        fn dom(&self) -> Poly<Words> {
            Ite::dom(self).into()
        }

        fn cod(&self) -> Poly<Words> {
            Ite::cod(self).into()
        }
    }

    impl Ite {
        /// The choice reading the scalar boolean condition and the branches of
        /// `dom`; fails unless the condition is of the shape `[1, 1]` and the
        /// branches of the same sort.
        pub fn with_dom(dom: [Words; 3]) -> Result<Self, crate::Error> {
            match dom {
                [Words::Bool(c), Words::Word(x), Words::Word(y)]
                    if x == y && c.shape() == [1, 1] =>
                {
                    Ok(Ite {
                        width: x.width(),
                        shape: x.shape(),
                    })
                }
                _ => Err(crate::unexpected_dom(
                    "a condition Bool([1, 1]) and two branches Word of the same sort",
                    &dom,
                )),
            }
        }

        /// The choice writing the words of `cod`.
        pub fn with_cod(cod: [Word; 1]) -> Result<Self, crate::Error> {
            let [s] = cod;
            Ok(Ite {
                width: s.width(),
                shape: s.shape(),
            })
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
        fn py_new(dom: [Words; 3], cod: Option<Word>) -> Result<Self, crate::Error> {
            crate::check_cod(Self::with_dom(dom), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str) {
            ("dom", "cod")
        }

        /// The sorts it reads, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn dom(&self) -> Mono<Words> {
            mono![
                Bool::new_unchecked([1, 1]).into(),
                Word::new_unchecked(self.width, self.shape).into(),
                Word::new_unchecked(self.width, self.shape).into()
            ]
        }

        /// The sorts it writes, as a tuple.
        #[cfg_attr(feature = "pyo3", getter)]
        fn cod(&self) -> Mono<Words> {
            mono![Word::new_unchecked(self.width, self.shape).into()]
        }
    }
}
