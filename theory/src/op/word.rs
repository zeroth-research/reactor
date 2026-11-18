/*!
# Fixed-size bit-vectors

Defines the sort [`Word`] of matrices of `w`-bit machine words, the local
sort [`Sort`] of the theory, holding words and the booleans that
comparisons write, and one type per operation on them. Each operation is
an operation of the SMT-LIB2 logic QF_BV, lifted pointwise to matrices:
it applies the SMT-LIB2 operation independently to each entry, with the
same semantics, and its documentation gives the SMT-LIB2 term it
computes on one entry.

Every operation carries enough information to compute its [`Boundary`]:
pointwise operations carry the width `w` and the shape `[m, n]` of their
arguments, as `Op(w, m, n)`, and indexed operations put their indices
first, as SMT-LIB2 does with `(_ op i)`. The operations whose meaning
depends on signedness are in the submodules [`signed`], reading words in
two's complement, and [`unsigned`], reading them in binary.

| Operation | SMT-LIB2 | | Operation | SMT-LIB2 |
|---|---|---|---|---|
| [`Const`] | `#b…`, `(_ bvN w)` | | [`signed::Div`] | `bvsdiv` |
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
| [`RotateRight`] | `(_ rotate_right i)` | | | |
| [`Eq`](struct@Eq) | `=` | | | |
| [`Ne`] | `distinct` | | | |

`=` and `distinct` belong to the SMT-LIB2 Core theory, which every logic
includes, at the sort `(_ BitVec w)`; all the others belong to the
theory FixedSizeBitVectors or are defined by the logic QF_BV. The
comparisons, `=` and `distinct` write booleans, as in SMT-LIB2.
*/

use crate::Tensor;
use crate::sort::{Bool, Word};
use base::boundary::{Boundary, Poly};
use base::mono;
use derive_more::From;
use std::fmt;

/// The sorts of the theory, as in the signature of SMT-LIB2's
/// FixedSizeBitVectors: words, and the booleans of the Core theory, which
/// comparisons write, together with [`Zero`](crate::sort::Zero), the tangent of both. It is
/// the subset [`sort::Words`](crate::sort::Words) of the sorts of all
/// theories.
pub use crate::sort::Words as Sort;

/// A constant matrix literal of `w`-bit words, as `Const(tensor, w)`, held
/// in a tensor of type `T`.
///
/// SMT-LIB2: a literal `#b…` or `(_ bvN w)`.
#[derive(Clone, Debug)]
pub struct Const<T>(pub T, pub usize);

/// Pointwise concatenation `x ++ y` of a `u`-bit word `x`, the high bits
/// of the result, and a `v`-bit word `y`, its low bits, as
/// `Concat(u, v, m, n)`; the result has `u + v` bits.
///
/// SMT-LIB2: `(concat x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct Concat(pub usize, pub usize, pub usize, pub usize);

/// Pointwise extraction of the bits `i` down to `j`, both included, from
/// `w`-bit words, as `Extract(i, j, w, m, n)`; it requires
/// `j ≤ i < w`, and the result has `i - j + 1` bits.
///
/// SMT-LIB2: `((_ extract i j) x)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct Extract(pub usize, pub usize, pub usize, pub usize, pub usize);

/// Pointwise concatenation of `i ≥ 1` copies of `w`-bit words, as
/// `Repeat(i, w, m, n)`; the result has `i·w` bits.
///
/// SMT-LIB2: `((_ repeat i) x)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct Repeat(pub usize, pub usize, pub usize, pub usize);

/// Pointwise bitwise negation `!x`, as `Not(w, m, n)`.
///
/// SMT-LIB2: `(bvnot x)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct Not(pub usize, pub usize, pub usize);

/// Pointwise bitwise conjunction `x & y`, as `And(w, m, n)`.
///
/// SMT-LIB2: `(bvand x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct And(pub usize, pub usize, pub usize);

/// Pointwise bitwise disjunction `x | y`, as `Or(w, m, n)`.
///
/// SMT-LIB2: `(bvor x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct Or(pub usize, pub usize, pub usize);

/// Pointwise bitwise exclusive disjunction `x ^ y`, as `Xor(w, m, n)`.
///
/// SMT-LIB2: `(bvxor x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct Xor(pub usize, pub usize, pub usize);

/// Pointwise bitwise negated conjunction `!(x & y)`, as `Nand(w, m, n)`.
///
/// SMT-LIB2: `(bvnand x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct Nand(pub usize, pub usize, pub usize);

/// Pointwise bitwise negated disjunction `!(x | y)`, as `Nor(w, m, n)`.
///
/// SMT-LIB2: `(bvnor x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct Nor(pub usize, pub usize, pub usize);

/// Pointwise bitwise negated exclusive disjunction `!(x ^ y)`, as
/// `Xnor(w, m, n)`.
///
/// SMT-LIB2: `(bvxnor x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct Xnor(pub usize, pub usize, pub usize);

/// Pointwise two's-complement negation `-x`, modulo `2^w`, as
/// `Neg(w, m, n)`; the minimum signed word is its own negation.
///
/// SMT-LIB2: `(bvneg x)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct Neg(pub usize, pub usize, pub usize);

/// Pointwise addition `x + y`, modulo `2^w`, as `Add(w, m, n)`.
///
/// SMT-LIB2: `(bvadd x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct Add(pub usize, pub usize, pub usize);

/// Pointwise subtraction `x - y`, modulo `2^w`, as `Sub(w, m, n)`.
///
/// SMT-LIB2: `(bvsub x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct Sub(pub usize, pub usize, pub usize);

/// Pointwise multiplication `x * y`, modulo `2^w`, as `Mul(w, m, n)`.
///
/// SMT-LIB2: `(bvmul x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct Mul(pub usize, pub usize, pub usize);

/// Pointwise left shift `x << y`, filling with zeros, as `Shl(w, m, n)`.
///
/// SMT-LIB2: `(bvshl x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct Shl(pub usize, pub usize, pub usize);

/// Pointwise rotation left by a constant `i` bits, as
/// `RotateLeft(i, w, m, n)`.
///
/// SMT-LIB2: `((_ rotate_left i) x)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct RotateLeft(pub usize, pub usize, pub usize, pub usize);

/// Pointwise rotation right by a constant `i` bits, as
/// `RotateRight(i, w, m, n)`.
///
/// SMT-LIB2: `((_ rotate_right i) x)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct RotateRight(pub usize, pub usize, pub usize, pub usize);

/// Pointwise equality `x = y` of `w`-bit words, as `Eq(w, m, n)`.
///
/// SMT-LIB2: `(= x y)`, of the Core theory.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct Eq(pub usize, pub usize, pub usize);

/// Pointwise disequality `x ≠ y` of `w`-bit words, as `Ne(w, m, n)`.
///
/// SMT-LIB2: `(distinct x y)`, of the Core theory.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct Ne(pub usize, pub usize, pub usize);

/// The operations of the theory of fixed-size bit-vectors, SMT-LIB2's
/// `FixedSizeBitVectors`, with their boundary in [`Sort`]. Those of
/// [`signed`] and [`unsigned`] are named after their SMT-LIB2 operation,
/// as `SDiv` for [`signed::Div`] and `UDiv` for [`unsigned::Div`].
#[derive(Clone, Debug, From)]
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
    Add(Add),
    Sub(Sub),
    Mul(Mul),
    Shl(Shl),
    RotateLeft(RotateLeft),
    RotateRight(RotateRight),
    Eq(Eq),
    Ne(Ne),
    SDiv(signed::Div),
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
            BV::Add(op) => op.dom(),
            BV::Sub(op) => op.dom(),
            BV::Mul(op) => op.dom(),
            BV::Shl(op) => op.dom(),
            BV::RotateLeft(op) => op.dom(),
            BV::RotateRight(op) => op.dom(),
            BV::Eq(op) => op.dom(),
            BV::Ne(op) => op.dom(),
            BV::SDiv(op) => op.dom(),
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
            BV::Add(op) => op.cod(),
            BV::Sub(op) => op.cod(),
            BV::Mul(op) => op.cod(),
            BV::Shl(op) => op.cod(),
            BV::RotateLeft(op) => op.cod(),
            BV::RotateRight(op) => op.cod(),
            BV::Eq(op) => op.cod(),
            BV::Ne(op) => op.cod(),
            BV::SDiv(op) => op.cod(),
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
        mono![Word::new_unchecked(self.1, self.0.shape()).into()].into()
    }
}

impl Boundary<Sort> for Concat {
    fn dom(&self) -> Poly<Sort> {
        let shape = [self.2, self.3];
        mono![
            Word::new_unchecked(self.0, shape).into(),
            Word::new_unchecked(self.1, shape).into()
        ]
        .into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.0 + self.1, [self.2, self.3]).into()].into()
    }
}

impl Boundary<Sort> for Extract {
    fn dom(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.2, [self.3, self.4]).into()].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.0 - self.1 + 1, [self.3, self.4]).into()].into()
    }
}

impl Boundary<Sort> for Repeat {
    fn dom(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.1, [self.2, self.3]).into()].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.0 * self.1, [self.2, self.3]).into()].into()
    }
}

impl Boundary<Sort> for Not {
    fn dom(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.0, [self.1, self.2]).into()].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.0, [self.1, self.2]).into()].into()
    }
}

impl Boundary<Sort> for And {
    fn dom(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.0, [self.1, self.2]).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.0, [self.1, self.2]).into()].into()
    }
}

impl Boundary<Sort> for Or {
    fn dom(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.0, [self.1, self.2]).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.0, [self.1, self.2]).into()].into()
    }
}

impl Boundary<Sort> for Xor {
    fn dom(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.0, [self.1, self.2]).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.0, [self.1, self.2]).into()].into()
    }
}

impl Boundary<Sort> for Nand {
    fn dom(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.0, [self.1, self.2]).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.0, [self.1, self.2]).into()].into()
    }
}

impl Boundary<Sort> for Nor {
    fn dom(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.0, [self.1, self.2]).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.0, [self.1, self.2]).into()].into()
    }
}

impl Boundary<Sort> for Xnor {
    fn dom(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.0, [self.1, self.2]).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.0, [self.1, self.2]).into()].into()
    }
}

impl Boundary<Sort> for Neg {
    fn dom(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.0, [self.1, self.2]).into()].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.0, [self.1, self.2]).into()].into()
    }
}

impl Boundary<Sort> for Add {
    fn dom(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.0, [self.1, self.2]).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.0, [self.1, self.2]).into()].into()
    }
}

impl Boundary<Sort> for Sub {
    fn dom(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.0, [self.1, self.2]).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.0, [self.1, self.2]).into()].into()
    }
}

impl Boundary<Sort> for Mul {
    fn dom(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.0, [self.1, self.2]).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.0, [self.1, self.2]).into()].into()
    }
}

impl Boundary<Sort> for Shl {
    fn dom(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.0, [self.1, self.2]).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.0, [self.1, self.2]).into()].into()
    }
}

impl Boundary<Sort> for RotateLeft {
    fn dom(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.1, [self.2, self.3]).into()].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.1, [self.2, self.3]).into()].into()
    }
}

impl Boundary<Sort> for RotateRight {
    fn dom(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.1, [self.2, self.3]).into()].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.1, [self.2, self.3]).into()].into()
    }
}

impl Boundary<Sort> for Eq {
    fn dom(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.0, [self.1, self.2]).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new([self.1, self.2]).into()].into()
    }
}

impl Boundary<Sort> for Ne {
    fn dom(&self) -> Poly<Sort> {
        mono![Word::new_unchecked(self.0, [self.1, self.2]).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new([self.1, self.2]).into()].into()
    }
}

impl<T: fmt::Display> fmt::Display for Const<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Display for Concat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Concat")
    }
}

impl fmt::Display for Extract {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Extract[{}:{}]", self.0, self.1)
    }
}

impl fmt::Display for Repeat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Repeat({})", self.0)
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
        write!(f, "RotateLeft({})", self.0)
    }
}

impl fmt::Display for RotateRight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RotateRight({})", self.0)
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
    use crate::sort::Bool;
    use base::boundary::{Boundary, Poly};
    use base::mono;
    use std::fmt;

    /// Pointwise division `x / y`, rounding towards zero, as `Div(w, m, n)`;
    /// the minimum word divided by `-1` is itself, and `x / 0` is `1` if `x`
    /// is negative and `-1` otherwise.
    ///
    /// SMT-LIB2: `(bvsdiv x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str, name = "SDiv"))]
    pub struct Div(pub usize, pub usize, pub usize);

    /// Pointwise remainder `x % y`, with the sign of `x`, as `Rem(w, m, n)`;
    /// `x % 0 = x`.
    ///
    /// SMT-LIB2: `(bvsrem x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str, name = "SRem"))]
    pub struct Rem(pub usize, pub usize, pub usize);

    /// Pointwise modulo `x mod y`, with the sign of `y`, as `Mod(w, m, n)`;
    /// `x mod 0 = x`.
    ///
    /// SMT-LIB2: `(bvsmod x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str, name = "SMod"))]
    pub struct Mod(pub usize, pub usize, pub usize);

    /// Pointwise arithmetic right shift `x >> y`, filling with the sign bit,
    /// as `Shr(w, m, n)`.
    ///
    /// SMT-LIB2: `(bvashr x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str, name = "AShr"))]
    pub struct Shr(pub usize, pub usize, pub usize);

    /// Pointwise extension of `w`-bit words by `i` copies of the sign bit,
    /// as `Extend(i, w, m, n)`.
    ///
    /// SMT-LIB2: `((_ sign_extend i) x)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str, name = "SignExtend"))]
    pub struct Extend(pub usize, pub usize, pub usize, pub usize);

    /// Pointwise signed comparison `x < y`, as `Lt(w, m, n)`.
    ///
    /// SMT-LIB2: `(bvslt x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str, name = "SLt"))]
    pub struct Lt(pub usize, pub usize, pub usize);

    /// Pointwise signed comparison `x ≤ y`, as `Le(w, m, n)`.
    ///
    /// SMT-LIB2: `(bvsle x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str, name = "SLe"))]
    pub struct Le(pub usize, pub usize, pub usize);

    /// Pointwise signed comparison `x > y`, as `Gt(w, m, n)`.
    ///
    /// SMT-LIB2: `(bvsgt x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str, name = "SGt"))]
    pub struct Gt(pub usize, pub usize, pub usize);

    /// Pointwise signed comparison `x ≥ y`, as `Ge(w, m, n)`.
    ///
    /// SMT-LIB2: `(bvsge x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str, name = "SGe"))]
    pub struct Ge(pub usize, pub usize, pub usize);

    impl Boundary<Sort> for Div {
        fn dom(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.0, [self.1, self.2]).into(); 2].into()
        }

        fn cod(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.0, [self.1, self.2]).into()].into()
        }
    }

    impl Boundary<Sort> for Rem {
        fn dom(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.0, [self.1, self.2]).into(); 2].into()
        }

        fn cod(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.0, [self.1, self.2]).into()].into()
        }
    }

    impl Boundary<Sort> for Mod {
        fn dom(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.0, [self.1, self.2]).into(); 2].into()
        }

        fn cod(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.0, [self.1, self.2]).into()].into()
        }
    }

    impl Boundary<Sort> for Shr {
        fn dom(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.0, [self.1, self.2]).into(); 2].into()
        }

        fn cod(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.0, [self.1, self.2]).into()].into()
        }
    }

    impl Boundary<Sort> for Extend {
        fn dom(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.1, [self.2, self.3]).into()].into()
        }

        fn cod(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.1 + self.0, [self.2, self.3]).into()].into()
        }
    }

    impl Boundary<Sort> for Lt {
        fn dom(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.0, [self.1, self.2]).into(); 2].into()
        }

        fn cod(&self) -> Poly<Sort> {
            mono![Bool::new([self.1, self.2]).into()].into()
        }
    }

    impl Boundary<Sort> for Le {
        fn dom(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.0, [self.1, self.2]).into(); 2].into()
        }

        fn cod(&self) -> Poly<Sort> {
            mono![Bool::new([self.1, self.2]).into()].into()
        }
    }

    impl Boundary<Sort> for Gt {
        fn dom(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.0, [self.1, self.2]).into(); 2].into()
        }

        fn cod(&self) -> Poly<Sort> {
            mono![Bool::new([self.1, self.2]).into()].into()
        }
    }

    impl Boundary<Sort> for Ge {
        fn dom(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.0, [self.1, self.2]).into(); 2].into()
        }

        fn cod(&self) -> Poly<Sort> {
            mono![Bool::new([self.1, self.2]).into()].into()
        }
    }

    impl fmt::Display for Div {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "SDiv")
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
            write!(f, "SignExtend({})", self.0)
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
}

/// Operations on words read in binary, whose meaning depends on
/// signedness; the others are in the parent module.
pub mod unsigned {
    use super::{Sort, Word};
    use crate::sort::Bool;
    use base::boundary::{Boundary, Poly};
    use base::mono;
    use std::fmt;

    /// Pointwise division `x / y`, rounding down, as `Div(w, m, n)`; `x / 0`
    /// is the word of all ones.
    ///
    /// SMT-LIB2: `(bvudiv x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str, name = "UDiv"))]
    pub struct Div(pub usize, pub usize, pub usize);

    /// Pointwise remainder `x % y`, as `Rem(w, m, n)`; `x % 0 = x`.
    ///
    /// SMT-LIB2: `(bvurem x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str, name = "URem"))]
    pub struct Rem(pub usize, pub usize, pub usize);

    /// Pointwise logical right shift `x >> y`, filling with zeros, as
    /// `Shr(w, m, n)`.
    ///
    /// SMT-LIB2: `(bvlshr x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str, name = "LShr"))]
    pub struct Shr(pub usize, pub usize, pub usize);

    /// Pointwise extension of `w`-bit words by `i` zero bits, as
    /// `Extend(i, w, m, n)`.
    ///
    /// SMT-LIB2: `((_ zero_extend i) x)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str, name = "ZeroExtend"))]
    pub struct Extend(pub usize, pub usize, pub usize, pub usize);

    /// Pointwise unsigned comparison `x < y`, as `Lt(w, m, n)`.
    ///
    /// SMT-LIB2: `(bvult x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str, name = "ULt"))]
    pub struct Lt(pub usize, pub usize, pub usize);

    /// Pointwise unsigned comparison `x ≤ y`, as `Le(w, m, n)`.
    ///
    /// SMT-LIB2: `(bvule x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str, name = "ULe"))]
    pub struct Le(pub usize, pub usize, pub usize);

    /// Pointwise unsigned comparison `x > y`, as `Gt(w, m, n)`.
    ///
    /// SMT-LIB2: `(bvugt x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str, name = "UGt"))]
    pub struct Gt(pub usize, pub usize, pub usize);

    /// Pointwise unsigned comparison `x ≥ y`, as `Ge(w, m, n)`.
    ///
    /// SMT-LIB2: `(bvuge x y)`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str, name = "UGe"))]
    pub struct Ge(pub usize, pub usize, pub usize);

    impl Boundary<Sort> for Div {
        fn dom(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.0, [self.1, self.2]).into(); 2].into()
        }

        fn cod(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.0, [self.1, self.2]).into()].into()
        }
    }

    impl Boundary<Sort> for Rem {
        fn dom(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.0, [self.1, self.2]).into(); 2].into()
        }

        fn cod(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.0, [self.1, self.2]).into()].into()
        }
    }

    impl Boundary<Sort> for Shr {
        fn dom(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.0, [self.1, self.2]).into(); 2].into()
        }

        fn cod(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.0, [self.1, self.2]).into()].into()
        }
    }

    impl Boundary<Sort> for Extend {
        fn dom(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.1, [self.2, self.3]).into()].into()
        }

        fn cod(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.1 + self.0, [self.2, self.3]).into()].into()
        }
    }

    impl Boundary<Sort> for Lt {
        fn dom(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.0, [self.1, self.2]).into(); 2].into()
        }

        fn cod(&self) -> Poly<Sort> {
            mono![Bool::new([self.1, self.2]).into()].into()
        }
    }

    impl Boundary<Sort> for Le {
        fn dom(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.0, [self.1, self.2]).into(); 2].into()
        }

        fn cod(&self) -> Poly<Sort> {
            mono![Bool::new([self.1, self.2]).into()].into()
        }
    }

    impl Boundary<Sort> for Gt {
        fn dom(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.0, [self.1, self.2]).into(); 2].into()
        }

        fn cod(&self) -> Poly<Sort> {
            mono![Bool::new([self.1, self.2]).into()].into()
        }
    }

    impl Boundary<Sort> for Ge {
        fn dom(&self) -> Poly<Sort> {
            mono![Word::new_unchecked(self.0, [self.1, self.2]).into(); 2].into()
        }

        fn cod(&self) -> Poly<Sort> {
            mono![Bool::new([self.1, self.2]).into()].into()
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
            write!(f, "ZeroExtend({})", self.0)
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
}

/// Extracts an operation of fixed-size bit-vectors from a Python object,
/// trying each operation in turn: those carrying no data through their own
/// classes, and [`Const`] through the extraction that the tensor type `T`
/// provides.
#[cfg(feature = "pyo3")]
impl<'py, T> pyo3::FromPyObject<'py> for BV<T>
where
    Const<T>: pyo3::FromPyObject<'py>,
{
    fn extract_bound(ob: &pyo3::Bound<'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn w(width: usize, shape: [usize; 2]) -> Sort {
        Word::new(width, shape).unwrap().into()
    }

    fn b(shape: [usize; 2]) -> Sort {
        Bool::new(shape).into()
    }

    fn one(s: Sort) -> Poly<Sort> {
        mono![s].into()
    }

    fn two(s: Sort) -> Poly<Sort> {
        mono![s, s].into()
    }

    #[test]
    fn pointwise_boundaries_preserve_width() {
        let x = w(8, [2, 3]);
        assert_eq!(Add(8, 2, 3).dom(), two(x));
        assert_eq!(Add(8, 2, 3).cod(), one(x));
        assert_eq!(Xnor(8, 2, 3).cod(), one(x));
        assert_eq!(Neg(8, 2, 3).dom(), one(x));
        assert_eq!(Not(8, 2, 3).cod(), one(x));
        assert_eq!(RotateLeft(3, 8, 2, 3).dom(), one(x));
        assert_eq!(RotateRight(3, 8, 2, 3).cod(), one(x));
        assert_eq!(signed::Div(8, 2, 3).dom(), two(x));
        assert_eq!(signed::Mod(8, 2, 3).cod(), one(x));
        assert_eq!(unsigned::Rem(8, 2, 3).cod(), one(x));
        assert_eq!(unsigned::Shr(8, 2, 3).dom(), two(x));
    }

    #[test]
    fn comparisons_write_booleans() {
        let x = w(8, [2, 3]);
        assert_eq!(Eq(8, 2, 3).dom(), two(x));
        assert_eq!(Eq(8, 2, 3).cod(), one(b([2, 3])));
        assert_eq!(Ne(8, 2, 3).cod(), one(b([2, 3])));
        assert_eq!(signed::Lt(8, 2, 3).dom(), two(x));
        assert_eq!(signed::Lt(8, 2, 3).cod(), one(b([2, 3])));
        assert_eq!(unsigned::Ge(8, 2, 3).cod(), one(b([2, 3])));
    }

    #[test]
    fn width_changing_boundaries() {
        assert_eq!(Extract(7, 4, 16, 2, 3).dom(), one(w(16, [2, 3])));
        assert_eq!(Extract(7, 4, 16, 2, 3).cod(), one(w(4, [2, 3])));
        assert_eq!(Extract(0, 0, 8, 1, 1).cod(), one(w(1, [1, 1])));

        let dom: Poly<Sort> = mono![w(8, [2, 3]), w(4, [2, 3])].into();
        assert_eq!(Concat(8, 4, 2, 3).dom(), dom);
        assert_eq!(Concat(8, 4, 2, 3).cod(), one(w(12, [2, 3])));

        assert_eq!(unsigned::Extend(8, 8, 2, 3).cod(), one(w(16, [2, 3])));
        assert_eq!(signed::Extend(24, 8, 2, 3).cod(), one(w(32, [2, 3])));
        assert_eq!(Repeat(3, 4, 2, 3).cod(), one(w(12, [2, 3])));
    }

    #[test]
    fn ops_display_their_name_and_indices() {
        assert_eq!(Add(8, 1, 1).to_string(), "Add");
        assert_eq!(Ne(8, 1, 1).to_string(), "Ne");
        assert_eq!(Extract(7, 4, 8, 1, 1).to_string(), "Extract[7:4]");
        assert_eq!(RotateLeft(3, 8, 1, 1).to_string(), "RotateLeft(3)");
        assert_eq!(Repeat(2, 8, 1, 1).to_string(), "Repeat(2)");
        assert_eq!(signed::Div(8, 1, 1).to_string(), "SDiv");
        assert_eq!(unsigned::Div(8, 1, 1).to_string(), "UDiv");
        assert_eq!(signed::Shr(8, 1, 1).to_string(), "AShr");
        assert_eq!(unsigned::Shr(8, 1, 1).to_string(), "LShr");
        assert_eq!(signed::Ge(8, 1, 1).to_string(), "SGe");
        assert_eq!(signed::Extend(8, 8, 1, 1).to_string(), "SignExtend(8)");
        assert_eq!(unsigned::Extend(8, 8, 1, 1).to_string(), "ZeroExtend(8)");
    }
}
