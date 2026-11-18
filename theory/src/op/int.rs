/*!
# Integer arithmetic

Defines one type per operation on the sort [`Int`] of integer matrices. Every
operation reads and writes integer matrices, and carries enough
information to compute its [`Boundary`]:

- [`Const`] — a constant integer matrix literal.
- [`Add`], [`Sub`] — pointwise addition and subtraction.
- [`Mul`] — pointwise multiplication `x ⊙ y`.
- [`Div`], [`Mod`] — pointwise Euclidean division and remainder, as
  SMT-LIB2's `div` and `mod`.
- [`Linear`]`(A, B)` — the affine map `x ↦ A·x + B`, with `A` and `B`
  constant integer matrices of compatible shapes.
- [`ReLU`] — the shape-preserving rectified-linear map `x ↦ max(x, 0)`.
- [`Min`], [`Max`] — pointwise minimum and maximum.
- [`Transpose`] — matrix transposition.
- [`MatMul`] — the matrix product `x·y`.
- [`Lt`], [`Le`], [`Gt`], [`Ge`], [`Eq`](struct@Eq), [`Ne`] — pointwise
  comparisons, which write boolean matrices of the same shape.
- [`Id`] — the identity.
- [`Ite`] — if-then-else, on a single boolean.
- [`Argmax`] — the index of a largest entry.
- [`Havoc`] — an arbitrary matrix, chosen nondeterministically.
- [`Uninterpreted`] — a named value of unknown meaning.

The comparisons are those of SMT-LIB2's `Ints` theory, `<`, `<=`, `>` and
`>=`, together with `=` and `distinct` of the Core theory, lifted pointwise
to matrices. As they write booleans, their boundary is in [`Sort`], the
sorts of the theory, rather than in [`Int`]. So is the boundary of
[`Ite`], which reads a boolean.
*/

use crate::sort::{Bool, Int};
use base::boundary::{Boundary, Poly};
use base::mono;
use derive_more::{Display, From};
use std::fmt;
use subenum::subenum;

/// The sorts of the theory, as in the signature of SMT-LIB2's `Ints`:
/// int matrices, and the booleans of the Core theory, which comparisons
/// write, together with [`Zero`](crate::sort::Zero). It is the subset
/// [`sort::Ints`](crate::sort::Ints) of the sorts of all theories.
pub use crate::sort::Ints as Sort;

/// A constant integer matrix literal.
#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, str, module = "reactor.op.int")
)]
pub struct Const(pub crate::PyTensor);

/// Pointwise addition `x + y` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, str, module = "reactor.op.int")
)]
pub struct Add(pub usize, pub usize);

/// Pointwise subtraction `x - y` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, str, module = "reactor.op.int")
)]
pub struct Sub(pub usize, pub usize);

/// Pointwise multiplication `x ⊙ y` of `[m, n]` matrices: the Hadamard
/// product, not the matrix product [`MatMul`].
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, str, module = "reactor.op.int")
)]
pub struct Mul(pub usize, pub usize);

/// Pointwise Euclidean division `x div y` of `[m, n]` matrices: the
/// quotient `q` such that `x = q·y + r` with `0 ≤ r < |y|`. It rounds down
/// for positive `y` and up for negative `y`, unlike Rust's `/`, which
/// rounds towards zero. As in SMT-LIB2, `x div 0` is unspecified.
///
/// SMT-LIB2: `(div x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, str, module = "reactor.op.int")
)]
pub struct Div(pub usize, pub usize);

/// Pointwise Euclidean remainder `x mod y` of `[m, n]` matrices: the `r`
/// such that `x = (x div y)·y + r` with `0 ≤ r < |y|`, so never negative,
/// unlike Rust's `%`. As in SMT-LIB2, `x mod 0` is unspecified.
///
/// SMT-LIB2: `(mod x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, str, module = "reactor.op.int")
)]
pub struct Mod(pub usize, pub usize);

/// The affine map `x ↦ A·x + B`, where `A` and `B` are constants.
#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, str, module = "reactor.op.int")
)]
pub struct Linear(pub crate::PyTensor, pub crate::PyTensor);

/// The rectified-linear map `x ↦ max(x, 0)`, applied pointwise to
/// `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, str, module = "reactor.op.int")
)]
pub struct ReLU(pub usize, pub usize);

/// Pointwise minimum `min(x, y)` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, str, module = "reactor.op.int")
)]
pub struct Min(pub usize, pub usize);

/// Pointwise maximum `max(x, y)` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, str, module = "reactor.op.int")
)]
pub struct Max(pub usize, pub usize);

/// Transposition `xᵀ` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, str, module = "reactor.op.int")
)]
pub struct Transpose(pub usize, pub usize);

/// The matrix product `x·y` of an `[m, n]` matrix `x` and an `[n, k]`
/// matrix `y`, with the sizes given as `MatMul(m, n, k)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, str, module = "reactor.op.int")
)]
pub struct MatMul(pub usize, pub usize, pub usize);

/// Pointwise comparison `x < y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(< x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, str, module = "reactor.op.int")
)]
pub struct Lt(pub usize, pub usize);

/// Pointwise comparison `x ≤ y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(<= x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, str, module = "reactor.op.int")
)]
pub struct Le(pub usize, pub usize);

/// Pointwise comparison `x > y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(> x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, str, module = "reactor.op.int")
)]
pub struct Gt(pub usize, pub usize);

/// Pointwise comparison `x ≥ y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(>= x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, str, module = "reactor.op.int")
)]
pub struct Ge(pub usize, pub usize);

/// Pointwise comparison `x = y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(= x y)`, of the Core theory.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, str, module = "reactor.op.int")
)]
pub struct Eq(pub usize, pub usize);

/// Pointwise comparison `x ≠ y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(distinct x y)`, of the Core theory.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, str, module = "reactor.op.int")
)]
pub struct Ne(pub usize, pub usize);

/// The identity `x ↦ x` on `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, str, module = "reactor.op.int")
)]
pub struct Id(pub usize, pub usize);

/// If-then-else `if c then x else y` of `[m, n]` matrices, on a single
/// boolean `c`, which picks either matrix whole.
///
/// SMT-LIB2: `(ite c x y)`, of the Core theory, with the same `c` for every
/// entry.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, str, module = "reactor.op.int")
)]
pub struct Ite(pub usize, pub usize);

/// The index of the first largest entry of `[m, n]` matrices, counted from
/// `0` in row-major order, as a `[1, 1]` integer matrix.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, str, module = "reactor.op.int")
)]
pub struct Argmax(pub usize, pub usize);

/// An arbitrary integer matrix of shape `[m, n]`, chosen nondeterministically.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, str, module = "reactor.op.int")
)]
pub struct Havoc(pub usize, pub usize);

/// An integer matrix of shape `[m, n]` of unknown meaning, named by the given
/// symbol, as `Uninterpreted(symbol, m, n)`: e.g., the result of a call that
/// the frontend cannot translate.
#[derive(Clone, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, str, module = "reactor.op.int")
)]
pub struct Uninterpreted(pub String, pub usize, pub usize);

/// The operations of the theory of integer arithmetic, SMT-LIB2's `Ints`,
/// with their boundary in [`Sort`].
///
/// `subenum` defines [`LIA`], the linear ones: all but `Mul`, `Div`, `Mod`
/// and `MatMul`, which multiply or divide two variables. It also generates
/// the casts between them: `From` up to `NIA`, and `TryFrom` down from it.
#[subenum(LIA)]
#[derive(Clone, Debug, From, Display)]
pub enum NIA {
    #[subenum(LIA)]
    Const(Const),
    #[subenum(LIA)]
    Add(Add),
    #[subenum(LIA)]
    Sub(Sub),
    Mul(Mul),
    Div(Div),
    Mod(Mod),
    #[subenum(LIA)]
    Linear(Linear),
    #[subenum(LIA)]
    ReLU(ReLU),
    #[subenum(LIA)]
    Min(Min),
    #[subenum(LIA)]
    Max(Max),
    #[subenum(LIA)]
    Transpose(Transpose),
    MatMul(MatMul),
    #[subenum(LIA)]
    Lt(Lt),
    #[subenum(LIA)]
    Le(Le),
    #[subenum(LIA)]
    Gt(Gt),
    #[subenum(LIA)]
    Ge(Ge),
    #[subenum(LIA)]
    Eq(Eq),
    #[subenum(LIA)]
    Ne(Ne),
    #[subenum(LIA)]
    Id(Id),
    #[subenum(LIA)]
    Ite(Ite),
    #[subenum(LIA)]
    Argmax(Argmax),
    #[subenum(LIA)]
    Havoc(Havoc),
    #[subenum(LIA)]
    Uninterpreted(Uninterpreted),
}

impl Boundary<Sort> for NIA {
    fn dom(&self) -> Poly<Sort> {
        match self {
            NIA::Const(op) => op.dom().map(Sort::from),
            NIA::Add(op) => op.dom().map(Sort::from),
            NIA::Sub(op) => op.dom().map(Sort::from),
            NIA::Mul(op) => op.dom().map(Sort::from),
            NIA::Div(op) => op.dom().map(Sort::from),
            NIA::Mod(op) => op.dom().map(Sort::from),
            NIA::Linear(op) => op.dom().map(Sort::from),
            NIA::ReLU(op) => op.dom().map(Sort::from),
            NIA::Min(op) => op.dom().map(Sort::from),
            NIA::Max(op) => op.dom().map(Sort::from),
            NIA::Transpose(op) => op.dom().map(Sort::from),
            NIA::MatMul(op) => op.dom().map(Sort::from),
            NIA::Lt(op) => op.dom(),
            NIA::Le(op) => op.dom(),
            NIA::Gt(op) => op.dom(),
            NIA::Ge(op) => op.dom(),
            NIA::Eq(op) => op.dom(),
            NIA::Ne(op) => op.dom(),
            NIA::Id(op) => op.dom().map(Sort::from),
            NIA::Ite(op) => op.dom(),
            NIA::Argmax(op) => op.dom().map(Sort::from),
            NIA::Havoc(op) => op.dom().map(Sort::from),
            NIA::Uninterpreted(op) => op.dom().map(Sort::from),
        }
    }

    fn cod(&self) -> Poly<Sort> {
        match self {
            NIA::Const(op) => op.cod().map(Sort::from),
            NIA::Add(op) => op.cod().map(Sort::from),
            NIA::Sub(op) => op.cod().map(Sort::from),
            NIA::Mul(op) => op.cod().map(Sort::from),
            NIA::Div(op) => op.cod().map(Sort::from),
            NIA::Mod(op) => op.cod().map(Sort::from),
            NIA::Linear(op) => op.cod().map(Sort::from),
            NIA::ReLU(op) => op.cod().map(Sort::from),
            NIA::Min(op) => op.cod().map(Sort::from),
            NIA::Max(op) => op.cod().map(Sort::from),
            NIA::Transpose(op) => op.cod().map(Sort::from),
            NIA::MatMul(op) => op.cod().map(Sort::from),
            NIA::Lt(op) => op.cod(),
            NIA::Le(op) => op.cod(),
            NIA::Gt(op) => op.cod(),
            NIA::Ge(op) => op.cod(),
            NIA::Eq(op) => op.cod(),
            NIA::Ne(op) => op.cod(),
            NIA::Id(op) => op.cod().map(Sort::from),
            NIA::Ite(op) => op.cod(),
            NIA::Argmax(op) => op.cod().map(Sort::from),
            NIA::Havoc(op) => op.cod().map(Sort::from),
            NIA::Uninterpreted(op) => op.cod().map(Sort::from),
        }
    }
}
impl Boundary<Sort> for LIA {
    fn dom(&self) -> Poly<Sort> {
        match self {
            LIA::Const(op) => op.dom().map(Sort::from),
            LIA::Add(op) => op.dom().map(Sort::from),
            LIA::Sub(op) => op.dom().map(Sort::from),
            LIA::Linear(op) => op.dom().map(Sort::from),
            LIA::ReLU(op) => op.dom().map(Sort::from),
            LIA::Min(op) => op.dom().map(Sort::from),
            LIA::Max(op) => op.dom().map(Sort::from),
            LIA::Transpose(op) => op.dom().map(Sort::from),
            LIA::Lt(op) => op.dom(),
            LIA::Le(op) => op.dom(),
            LIA::Gt(op) => op.dom(),
            LIA::Ge(op) => op.dom(),
            LIA::Eq(op) => op.dom(),
            LIA::Ne(op) => op.dom(),
            LIA::Id(op) => op.dom().map(Sort::from),
            LIA::Ite(op) => op.dom(),
            LIA::Argmax(op) => op.dom().map(Sort::from),
            LIA::Havoc(op) => op.dom().map(Sort::from),
            LIA::Uninterpreted(op) => op.dom().map(Sort::from),
        }
    }

    fn cod(&self) -> Poly<Sort> {
        match self {
            LIA::Const(op) => op.cod().map(Sort::from),
            LIA::Add(op) => op.cod().map(Sort::from),
            LIA::Sub(op) => op.cod().map(Sort::from),
            LIA::Linear(op) => op.cod().map(Sort::from),
            LIA::ReLU(op) => op.cod().map(Sort::from),
            LIA::Min(op) => op.cod().map(Sort::from),
            LIA::Max(op) => op.cod().map(Sort::from),
            LIA::Transpose(op) => op.cod().map(Sort::from),
            LIA::Lt(op) => op.cod(),
            LIA::Le(op) => op.cod(),
            LIA::Gt(op) => op.cod(),
            LIA::Ge(op) => op.cod(),
            LIA::Eq(op) => op.cod(),
            LIA::Ne(op) => op.cod(),
            LIA::Id(op) => op.cod().map(Sort::from),
            LIA::Ite(op) => op.cod(),
            LIA::Argmax(op) => op.cod().map(Sort::from),
            LIA::Havoc(op) => op.cod().map(Sort::from),
            LIA::Uninterpreted(op) => op.cod().map(Sort::from),
        }
    }
}

impl Boundary<Int> for Const {
    fn dom(&self) -> Poly<Int> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new(self.0.shape())].into()
    }
}

impl Boundary<Int> for Add {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1]); 2].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1])].into()
    }
}

impl Boundary<Int> for Sub {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1]); 2].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1])].into()
    }
}

impl Boundary<Int> for Mul {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1]); 2].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1])].into()
    }
}

impl Boundary<Int> for Div {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1]); 2].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1])].into()
    }
}

impl Boundary<Int> for Mod {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1]); 2].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1])].into()
    }
}

/// With `A` of shape `[m, n]` and `B` of shape `[m, k]`, the map takes `x`
/// of shape `[n, k]` to a matrix of shape `[m, k]`.
impl Boundary<Int> for Linear {
    fn dom(&self) -> Poly<Int> {
        let [_, n] = self.0.shape();
        let [_, k] = self.1.shape();
        mono![Int::new([n, k])].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new(self.1.shape())].into()
    }
}

impl Boundary<Int> for ReLU {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1])].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1])].into()
    }
}

impl Boundary<Int> for Min {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1]); 2].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1])].into()
    }
}

impl Boundary<Int> for Max {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1]); 2].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1])].into()
    }
}

/// Takes a matrix of shape `[m, n]` to one of shape `[n, m]`.
impl Boundary<Int> for Transpose {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1])].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new([self.1, self.0])].into()
    }
}

/// Takes matrices of shapes `[m, n]` and `[n, k]` to one of shape `[m, k]`.
impl Boundary<Int> for MatMul {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1]), Int::new([self.1, self.2])].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.2])].into()
    }
}

impl Boundary<Sort> for Lt {
    fn dom(&self) -> Poly<Sort> {
        mono![Int::new([self.0, self.1]).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new([self.0, self.1]).into()].into()
    }
}

impl Boundary<Sort> for Le {
    fn dom(&self) -> Poly<Sort> {
        mono![Int::new([self.0, self.1]).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new([self.0, self.1]).into()].into()
    }
}

impl Boundary<Sort> for Gt {
    fn dom(&self) -> Poly<Sort> {
        mono![Int::new([self.0, self.1]).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new([self.0, self.1]).into()].into()
    }
}

impl Boundary<Sort> for Ge {
    fn dom(&self) -> Poly<Sort> {
        mono![Int::new([self.0, self.1]).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new([self.0, self.1]).into()].into()
    }
}

impl Boundary<Sort> for Eq {
    fn dom(&self) -> Poly<Sort> {
        mono![Int::new([self.0, self.1]).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new([self.0, self.1]).into()].into()
    }
}

impl Boundary<Sort> for Ne {
    fn dom(&self) -> Poly<Sort> {
        mono![Int::new([self.0, self.1]).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new([self.0, self.1]).into()].into()
    }
}

impl Boundary<Int> for Id {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1])].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1])].into()
    }
}

/// Takes a boolean of shape `[1, 1]` and two matrices of shape `[m, n]` to
/// a matrix of shape `[m, n]`.
impl Boundary<Sort> for Ite {
    fn dom(&self) -> Poly<Sort> {
        let x: Sort = Int::new([self.0, self.1]).into();
        mono![Bool::new([1, 1]).into(), x, x].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Int::new([self.0, self.1]).into()].into()
    }
}

/// Takes a matrix of shape `[m, n]` to one of shape `[1, 1]`.
impl Boundary<Int> for Argmax {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1])].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new([1, 1])].into()
    }
}

impl Boundary<Int> for Havoc {
    fn dom(&self) -> Poly<Int> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1])].into()
    }
}

impl Boundary<Int> for Uninterpreted {
    fn dom(&self) -> Poly<Int> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new([self.1, self.2])].into()
    }
}

impl fmt::Display for Const {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Display for Linear {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Linear({}, {})", self.0, self.1)
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

impl fmt::Display for Div {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Div")
    }
}

impl fmt::Display for Mod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Mod")
    }
}

impl fmt::Display for ReLU {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ReLU")
    }
}

impl fmt::Display for Min {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Min")
    }
}

impl fmt::Display for Max {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Max")
    }
}

impl fmt::Display for Transpose {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Transpose")
    }
}

impl fmt::Display for MatMul {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MatMul")
    }
}

impl fmt::Display for Lt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Lt")
    }
}

impl fmt::Display for Le {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Le")
    }
}

impl fmt::Display for Gt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Gt")
    }
}

impl fmt::Display for Ge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ge")
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

impl fmt::Display for Id {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Id")
    }
}

impl fmt::Display for Ite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ite")
    }
}

impl fmt::Display for Argmax {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Argmax")
    }
}

impl fmt::Display for Havoc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Havoc")
    }
}

impl fmt::Display for Uninterpreted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Uninterpreted({})", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(s: Int) -> Poly<Int> {
        mono![s].into()
    }

    fn two(s: Int) -> Poly<Int> {
        mono![s, s].into()
    }

    #[test]
    fn pointwise_boundaries() {
        let r = Int::new([2, 3]);
        assert_eq!(Add(2, 3).dom(), two(r));
        assert_eq!(Add(2, 3).cod(), one(r));
        assert_eq!(Sub(2, 3).dom(), two(r));
        assert_eq!(Mul(2, 3).dom(), two(r));
        assert_eq!(Mul(2, 3).cod(), one(r));
        assert_eq!(Div(2, 3).dom(), two(r));
        assert_eq!(Div(2, 3).cod(), one(r));
        assert_eq!(Mod(2, 3).dom(), two(r));
        assert_eq!(Mod(2, 3).cod(), one(r));
        assert_eq!(Min(2, 3).dom(), two(r));
        assert_eq!(Max(2, 3).cod(), one(r));
        assert_eq!(ReLU(2, 3).dom(), one(r));
        assert_eq!(ReLU(2, 3).cod(), one(r));
    }

    #[test]
    fn transpose_swaps_shape() {
        let t = Transpose(2, 3);
        assert_eq!(t.dom(), one(Int::new([2, 3])));
        assert_eq!(t.cod(), one(Int::new([3, 2])));
    }

    #[test]
    fn matmul_contracts_inner_size() {
        let p = MatMul(2, 3, 4);
        let dom: Poly<Int> = mono![Int::new([2, 3]), Int::new([3, 4])].into();
        assert_eq!(p.dom(), dom);
        assert_eq!(p.cod(), one(Int::new([2, 4])));
    }

    #[test]
    fn id_keeps_sort() {
        let x = Int::new([2, 3]);
        assert_eq!(Id(2, 3).dom(), one(x));
        assert_eq!(Id(2, 3).cod(), one(x));
    }

    #[test]
    fn ite_reads_a_single_boolean() {
        let c: Sort = Bool::new([1, 1]).into();
        let x: Sort = Int::new([2, 3]).into();
        let dom: Poly<Sort> = mono![c, x, x].into();
        let cod: Poly<Sort> = mono![x].into();
        assert_eq!(Ite(2, 3).dom(), dom);
        assert_eq!(Ite(2, 3).cod(), cod);
    }

    #[test]
    fn argmax_writes_a_single_entry() {
        assert_eq!(Argmax(2, 3).dom(), one(Int::new([2, 3])));
        assert_eq!(Argmax(2, 3).cod(), one(Int::new([1, 1])));
    }

    #[test]
    fn havoc_reads_nothing() {
        assert_eq!(Havoc(2, 3).dom(), Poly::unit());
        assert_eq!(Havoc(2, 3).cod(), one(Int::new([2, 3])));
    }

    #[test]
    fn uninterpreted_reads_nothing() {
        let u = Uninterpreted("f".to_string(), 2, 3);
        assert_eq!(u.dom(), Poly::unit());
        assert_eq!(u.cod(), one(Int::new([2, 3])));
    }

    #[test]
    fn ops_display_their_name() {
        assert_eq!(Add(1, 1).to_string(), "Add");
        assert_eq!(Mul(1, 1).to_string(), "Mul");
        assert_eq!(Div(1, 1).to_string(), "Div");
        assert_eq!(Mod(1, 1).to_string(), "Mod");
        assert_eq!(ReLU(1, 1).to_string(), "ReLU");
        assert_eq!(Transpose(1, 1).to_string(), "Transpose");
        assert_eq!(Id(1, 1).to_string(), "Id");
        assert_eq!(Ite(1, 1).to_string(), "Ite");
        assert_eq!(Argmax(1, 1).to_string(), "Argmax");
        assert_eq!(Havoc(1, 1).to_string(), "Havoc");
        assert_eq!(
            Uninterpreted("f".to_string(), 1, 1).to_string(),
            "Uninterpreted(f)"
        );
    }
    #[test]
    fn comparisons_write_booleans() {
        let x: Sort = Int::new([2, 3]).into();
        let b: Sort = Bool::new([2, 3]).into();
        let dom: Poly<Sort> = mono![x, x].into();
        let cod: Poly<Sort> = mono![b].into();
        for (d, c) in [
            (Lt(2, 3).dom(), Lt(2, 3).cod()),
            (Le(2, 3).dom(), Le(2, 3).cod()),
            (Gt(2, 3).dom(), Gt(2, 3).cod()),
            (Ge(2, 3).dom(), Ge(2, 3).cod()),
            (Eq(2, 3).dom(), Eq(2, 3).cod()),
            (Ne(2, 3).dom(), Ne(2, 3).cod()),
        ] {
            assert_eq!(d, dom);
            assert_eq!(c, cod);
        }
        assert_eq!(Le(1, 1).to_string(), "Le");
        assert_eq!(Ne(1, 1).to_string(), "Ne");
    }
}
