/*!
# Integer arithmetic

Defines one type per operation on the sort [`Int`] of integer matrices. Every
operation reads and writes integer matrices, and carries enough
information to compute its [`Boundary`]:

- [`Num`] — a constant integer matrix literal.
- [`Scale`] — the pointwise product `C ⊙ x` with a constant integer matrix `C`.
- [`Add`], [`Sub`] — pointwise addition and subtraction.
- [`Mul`] — pointwise multiplication `x ⊙ y`.
- [`Div`], [`Mod`] — pointwise Euclidean division and remainder, as
  SMT-LIB2's `div` and `mod`.
- [`Linear`]`(A)` — the linear map `x ↦ A·x`, with `A` a constant integer
  matrix.
- [`Affine`]`(A, B)` — the affine map `x ↦ A·x + B`, with `A` and `B`
  constant integer matrices of compatible shapes.
- [`Neg`] — the pointwise negation `x ↦ -x`.
- [`Abs`] — the pointwise absolute value `x ↦ |x|`.
- [`ReLU`] — the shape-preserving rectified-linear map `x ↦ max(x, 0)`.
- [`Min`], [`Max`] — pointwise minimum and maximum.
- [`Transpose`] — matrix transposition.
- [`MatMul`] — the matrix product `x·y`.
- [`Argmax`] — the index of a maximal entry, counted row by row, as a
  `[1, 1]` matrix.
- [`Ite`] — the pointwise choice `if c then x else y`, by a boolean matrix `c`.
- [`broadcast::Ite`] — the choice of the whole matrices, by a scalar boolean
  `c` of the shape `[1, 1]`.
- [`Lt`], [`Le`], [`Gt`], [`Ge`], [`Eq`](struct@Eq), [`Ne`] — pointwise
  comparisons, which write boolean matrices of the same shape.

The comparisons are those of SMT-LIB2's `Ints` theory, `<`, `<=`, `>` and
`>=`, together with `=` and `distinct` of the Core theory, lifted pointwise
to matrices. As they write booleans, their boundary is in [`Ints`], the
sorts of the theory, rather than in [`Int`].
*/

use super::bool::Bool;
use crate::Matrix;
use crate::zero::Real0;
use base::boundary::{Boundary, Poly};
use base::differentiable::Differentiable;
use base::mono;
use derive_more::From;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use std::fmt;
use subenum::subenum;

/// The sort of integer matrices with the given `[rows, cols]` shape.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, get_all, module = "reactor.int")
)]
pub struct Int {
    shape: [usize; 2],
}

impl Int {
    /// Integer matrices of the given shape, as [`Int::new`], but panics if
    /// `shape` is empty.
    pub fn new_unchecked(shape: [usize; 2]) -> Self {
        Self::new(shape).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

/// Python matches an `Int` by `shape`.
#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Int {
    /// Integer matrices of the given shape; fails unless the shape has at
    /// least one row and one column.
    #[cfg_attr(feature = "pyo3", new)]
    pub fn new(shape: [usize; 2]) -> Result<Self, crate::Error> {
        crate::matrix::check_shape(shape)?;
        Ok(Int { shape })
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    /// The tangent sort `T(self)`: the sort of the rate of change of its
    /// values.
    #[cfg(feature = "pyo3")]
    #[pyo3(name = "tangent")]
    fn tangent(&self) -> Real0 {
        Differentiable::tangent(self)
    }
}

impl fmt::Display for Int {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [i, j] = self.shape;
        write!(f, "Int([{i}, {j}])")
    }
}

/// Integers are discrete, so their tangent is trivial: the singleton sort
/// [`Real0`].
impl Differentiable for Int {
    type Tangent = Real0;

    fn tangent(self) -> Real0 {
        Real0
    }
}

/// The sorts of the theory, as in the signature of SMT-LIB2's `Ints`:
/// int matrices, and the booleans of the Core theory, which comparisons
/// write, together with [`Real0`]. It is the subset
/// [`Ints`](crate::Ints) of the sorts of all theories.
use crate::Ints;

/// A constant integer matrix literal, held in a tensor of type `T`.
#[derive(Debug)]
pub struct Num<T> {
    tensor: T,
}

impl<T: Matrix> Num<T> {
    /// The constant held in `tensor`, writing the integer matrix of `cod`; fails
    /// unless `tensor` is of its shape.
    pub fn with_cod(tensor: T, cod: [Int; 1]) -> Result<Self, crate::Error> {
        let [x] = cod;
        if tensor.shape() != x.shape() {
            return Err(crate::unexpected_cod(
                &format!("1 sort of the shape {:?} of the constant", tensor.shape()),
                &cod,
            ));
        }
        Ok(Num { tensor })
    }
}

impl<T> Num<T> {
    /// The tensor holding the constant.
    pub fn tensor(&self) -> &T {
        &self.tensor
    }
}

/// The pointwise product `x ↦ C ⊙ x` of `[m, n]` matrices with a constant
/// `C` of shape `[m, n]`, held in a tensor of type `T`: linear in `x`, unlike
/// [`Mul`], which multiplies two variables.
#[derive(Debug)]
pub struct Scale<T> {
    tensor: T,
}

impl<T: Matrix> Scale<T> {
    /// The product with the constant held in `tensor`, reading the integer
    /// matrix of `dom`; fails unless `tensor` is of its shape.
    pub fn with_dom(tensor: T, dom: [Int; 1]) -> Result<Self, crate::Error> {
        let [x] = dom;
        if tensor.shape() != x.shape() {
            return Err(crate::unexpected_dom(
                &format!("1 sort of the shape {:?} of C", tensor.shape()),
                &dom,
            ));
        }
        Ok(Scale { tensor })
    }

    /// The product with the constant held in `tensor`, writing the integer
    /// matrix of `cod`; fails unless `tensor` is of its shape.
    pub fn with_cod(tensor: T, cod: [Int; 1]) -> Result<Self, crate::Error> {
        let [x] = cod;
        if tensor.shape() != x.shape() {
            return Err(crate::unexpected_cod(
                &format!("1 sort of the shape {:?} of C", tensor.shape()),
                &cod,
            ));
        }
        Ok(Scale { tensor })
    }
}

impl<T> Scale<T> {
    /// The tensor holding the constant `C`.
    pub fn tensor(&self) -> &T {
        &self.tensor
    }
}

/// Pointwise addition `x + y` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.int")
)]
pub struct Add {
    shape: [usize; 2],
}

/// Pointwise subtraction `x - y` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.int")
)]
pub struct Sub {
    shape: [usize; 2],
}

/// Pointwise multiplication `x ⊙ y` of `[m, n]` matrices: the Hadamard
/// product, not the matrix product [`MatMul`].
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.int")
)]
pub struct Mul {
    shape: [usize; 2],
}

/// Pointwise Euclidean division `x div y` of `[m, n]` matrices: the
/// quotient `q` such that `x = q·y + r` with `0 ≤ r < |y|`. It rounds down
/// for positive `y` and up for negative `y`, unlike Rust's `/`, which
/// rounds towards zero. As in SMT-LIB2, `x div 0` is unspecified.
///
/// SMT-LIB2: `(div x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.int")
)]
pub struct Div {
    shape: [usize; 2],
}

/// Pointwise Euclidean remainder `x mod y` of `[m, n]` matrices: the `r`
/// such that `x = (x div y)·y + r` with `0 ≤ r < |y|`, so never negative,
/// unlike Rust's `%`. As in SMT-LIB2, `x mod 0` is unspecified.
///
/// SMT-LIB2: `(mod x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.int")
)]
pub struct Mod {
    shape: [usize; 2],
}

/// The linear map `x ↦ A·x`, where `A` is a constant. `A` determines the
/// rows of `x` but not its columns, which the map holds.
#[derive(Debug)]
pub struct Linear<T> {
    a: T,
    cols: usize,
}

impl<T: Matrix> Linear<T> {
    /// The linear map `x ↦ A·x` reading the integer matrix `x` of `dom`;
    /// fails unless `a` is of shape `[m, n]` and `x` of shape `[n, k]`.
    pub fn with_dom(a: T, dom: [Int; 1]) -> Result<Self, crate::Error> {
        let [x] = dom;
        crate::matrix::check_shape(a.shape())?;
        let ([_, n], [rows, k]) = (a.shape(), x.shape());
        if rows != n {
            return Err(crate::unexpected_dom(
                &format!("1 sort of shape [{n}, k]"),
                &dom,
            ));
        }
        Ok(Linear { a, cols: k })
    }

    /// The linear map `x ↦ A·x` writing the integer matrix of `cod`; fails
    /// unless `a` is of shape `[m, n]` and that matrix of shape `[m, k]`.
    pub fn with_cod(a: T, cod: [Int; 1]) -> Result<Self, crate::Error> {
        let [x] = cod;
        crate::matrix::check_shape(a.shape())?;
        let ([m, _], [rows, k]) = (a.shape(), x.shape());
        if rows != m {
            return Err(crate::unexpected_cod(
                &format!("1 sort of shape [{m}, k]"),
                &cod,
            ));
        }
        Ok(Linear { a, cols: k })
    }
}

impl<T> Linear<T> {
    /// The tensor of `A`.
    pub fn a(&self) -> &T {
        &self.a
    }
}

/// The affine map `x ↦ A·x + B`, where `A` and `B` are constants.
#[derive(Debug)]
pub struct Affine<T> {
    a: T,
    b: T,
}

impl<T: Matrix> Affine<T> {
    /// The affine map `x ↦ A·x + B` reading the integer matrix `x` of
    /// `dom`; fails unless `a` and `b` are of shapes `[m, n]` and `[m, k]`,
    /// and `x` of shape `[n, k]`.
    pub fn with_dom(a: T, b: T, dom: [Int; 1]) -> Result<Self, crate::Error> {
        let [x] = dom;
        crate::matrix::check_affine(a.shape(), b.shape())?;
        let op = Affine { a, b };
        let ([_, n], [_, k]) = (op.a.shape(), op.b.shape());
        if x.shape() != [n, k] {
            return Err(crate::unexpected_dom(
                &format!("1 sort of shape {:?}", [n, k]),
                &dom,
            ));
        }
        Ok(op)
    }

    /// The affine map `x ↦ A·x + B` writing the integer matrix of `cod`;
    /// fails unless `a` and `b` are of shapes `[m, n]` and `[m, k]`, and
    /// that matrix of shape `[m, k]`.
    pub fn with_cod(a: T, b: T, cod: [Int; 1]) -> Result<Self, crate::Error> {
        let [x] = cod;
        crate::matrix::check_affine(a.shape(), b.shape())?;
        let op = Affine { a, b };
        if x.shape() != op.b.shape() {
            return Err(crate::unexpected_cod(
                &format!("1 sort of shape {:?}", op.b.shape()),
                &cod,
            ));
        }
        Ok(op)
    }
}

impl<T> Affine<T> {
    /// The tensor of `A`.
    pub fn a(&self) -> &T {
        &self.a
    }

    /// The tensor of `B`.
    pub fn b(&self) -> &T {
        &self.b
    }
}

/// Pointwise negation `x ↦ -x` of `[m, n]` matrices.
///
/// SMT-LIB2: `(- x)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.int")
)]
pub struct Neg {
    shape: [usize; 2],
}

/// Pointwise absolute value `x ↦ |x|` of `[m, n]` matrices.
///
/// SMT-LIB2: `(abs x)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.int")
)]
pub struct Abs {
    shape: [usize; 2],
}

/// The rectified-linear map `x ↦ max(x, 0)`, applied pointwise to
/// `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.int")
)]
pub struct ReLU {
    shape: [usize; 2],
}

/// The index of a maximal entry of an `[m, n]` integer matrix, counting its
/// entries row by row, written as a `[1, 1]` matrix of the same sort.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.int")
)]
pub struct Argmax {
    shape: [usize; 2],
}

/// Pointwise minimum `min(x, y)` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.int")
)]
pub struct Min {
    shape: [usize; 2],
}

/// Pointwise maximum `max(x, y)` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.int")
)]
pub struct Max {
    shape: [usize; 2],
}

/// Transposition `xᵀ` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.int")
)]
pub struct Transpose {
    shape: [usize; 2],
}

/// The matrix product `x·y` of an `[m, n]` matrix `x` and an `[n, k]`
/// matrix `y`, with the shapes given as `MatMul([[m, n], [n, k]])`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.int")
)]
pub struct MatMul {
    /// The shapes `[[m, n], [n, k]]` of the factors.
    ///
    /// Entry `i` belongs to input `i`, as indexed in the domain of the
    /// boundary: an array rather than one field per input, so that the
    /// two match by position.
    shapes: [[usize; 2]; 2],
}

/// Pointwise comparison `x < y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(< x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.int")
)]
pub struct Lt {
    shape: [usize; 2],
}

/// Pointwise choice `if c then x else y` between `[m, n]` integer matrices `x` and `y`,
/// by the boolean matrix `c`, read first.
///
/// SMT-LIB2: `(ite c x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.int")
)]
pub struct Ite {
    shape: [usize; 2],
}

/// Pointwise comparison `x ≤ y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(<= x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.int")
)]
pub struct Le {
    shape: [usize; 2],
}

/// Pointwise comparison `x > y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(> x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.int")
)]
pub struct Gt {
    shape: [usize; 2],
}

/// Pointwise comparison `x ≥ y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(>= x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.int")
)]
pub struct Ge {
    shape: [usize; 2],
}

/// Pointwise comparison `x = y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(= x y)`, of the Core theory.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.int")
)]
pub struct Eq {
    shape: [usize; 2],
}

/// Pointwise comparison `x ≠ y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(distinct x y)`, of the Core theory.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.int")
)]
pub struct Ne {
    shape: [usize; 2],
}

/// The operations of the theory of integer arithmetic, SMT-LIB2's `Ints`,
/// with their boundary in [`Ints`].
///
/// `subenum` defines [`LIA`], the linear ones: all but `Mul`, `Div`, `Mod`
/// and `MatMul`, which multiply or divide two variables. It also generates
/// the casts between them: `From` up to `NIA`, and `TryFrom` down from it.
#[subenum(LIA)]
#[derive(Debug, From)]
pub enum NIA<T> {
    #[subenum(LIA)]
    Num(Num<T>),
    #[subenum(LIA)]
    Scale(Scale<T>),
    #[subenum(LIA)]
    Add(Add),
    #[subenum(LIA)]
    Sub(Sub),
    Mul(Mul),
    Div(Div),
    Mod(Mod),
    #[subenum(LIA)]
    Linear(Linear<T>),
    #[subenum(LIA)]
    Affine(Affine<T>),
    #[subenum(LIA)]
    ReLU(ReLU),
    #[subenum(LIA)]
    Abs(Abs),
    #[subenum(LIA)]
    Neg(Neg),
    #[subenum(LIA)]
    Argmax(Argmax),
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
    Ite(Ite),
    #[subenum(LIA)]
    BroadcastIte(broadcast::Ite),
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
}

impl<T> Clone for NIA<T>
where
    Num<T>: Clone,
    Scale<T>: Clone,
    Linear<T>: Clone,
    Affine<T>: Clone,
{
    fn clone(&self) -> Self {
        match self {
            NIA::Num(op) => NIA::Num(op.clone()),
            NIA::Scale(op) => NIA::Scale(op.clone()),
            NIA::Add(op) => NIA::Add(*op),
            NIA::Sub(op) => NIA::Sub(*op),
            NIA::Mul(op) => NIA::Mul(*op),
            NIA::Div(op) => NIA::Div(*op),
            NIA::Mod(op) => NIA::Mod(*op),
            NIA::Linear(op) => NIA::Linear(op.clone()),
            NIA::Affine(op) => NIA::Affine(op.clone()),
            NIA::ReLU(op) => NIA::ReLU(*op),
            NIA::Abs(op) => NIA::Abs(*op),
            NIA::Neg(op) => NIA::Neg(*op),
            NIA::Argmax(op) => NIA::Argmax(*op),
            NIA::Min(op) => NIA::Min(*op),
            NIA::Max(op) => NIA::Max(*op),
            NIA::Transpose(op) => NIA::Transpose(*op),
            NIA::MatMul(op) => NIA::MatMul(*op),
            NIA::Lt(op) => NIA::Lt(*op),
            NIA::Ite(op) => NIA::Ite(*op),
            NIA::BroadcastIte(op) => NIA::BroadcastIte(*op),
            NIA::Le(op) => NIA::Le(*op),
            NIA::Gt(op) => NIA::Gt(*op),
            NIA::Ge(op) => NIA::Ge(*op),
            NIA::Eq(op) => NIA::Eq(*op),
            NIA::Ne(op) => NIA::Ne(*op),
        }
    }
}

impl<T: fmt::Display> fmt::Display for NIA<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NIA::Num(op) => op.fmt(f),
            NIA::Scale(op) => op.fmt(f),
            NIA::Add(op) => op.fmt(f),
            NIA::Sub(op) => op.fmt(f),
            NIA::Mul(op) => op.fmt(f),
            NIA::Div(op) => op.fmt(f),
            NIA::Mod(op) => op.fmt(f),
            NIA::Linear(op) => op.fmt(f),
            NIA::Affine(op) => op.fmt(f),
            NIA::ReLU(op) => op.fmt(f),
            NIA::Abs(op) => op.fmt(f),
            NIA::Neg(op) => op.fmt(f),
            NIA::Argmax(op) => op.fmt(f),
            NIA::Min(op) => op.fmt(f),
            NIA::Max(op) => op.fmt(f),
            NIA::Transpose(op) => op.fmt(f),
            NIA::MatMul(op) => op.fmt(f),
            NIA::Lt(op) => op.fmt(f),
            NIA::Ite(op) => op.fmt(f),
            NIA::BroadcastIte(op) => op.fmt(f),
            NIA::Le(op) => op.fmt(f),
            NIA::Gt(op) => op.fmt(f),
            NIA::Ge(op) => op.fmt(f),
            NIA::Eq(op) => op.fmt(f),
            NIA::Ne(op) => op.fmt(f),
        }
    }
}

impl<T> Clone for LIA<T>
where
    Num<T>: Clone,
    Scale<T>: Clone,
    Linear<T>: Clone,
    Affine<T>: Clone,
{
    fn clone(&self) -> Self {
        match self {
            LIA::Num(op) => LIA::Num(op.clone()),
            LIA::Scale(op) => LIA::Scale(op.clone()),
            LIA::Add(op) => LIA::Add(*op),
            LIA::Sub(op) => LIA::Sub(*op),
            LIA::Linear(op) => LIA::Linear(op.clone()),
            LIA::Affine(op) => LIA::Affine(op.clone()),
            LIA::ReLU(op) => LIA::ReLU(*op),
            LIA::Abs(op) => LIA::Abs(*op),
            LIA::Neg(op) => LIA::Neg(*op),
            LIA::Argmax(op) => LIA::Argmax(*op),
            LIA::Min(op) => LIA::Min(*op),
            LIA::Max(op) => LIA::Max(*op),
            LIA::Transpose(op) => LIA::Transpose(*op),
            LIA::Lt(op) => LIA::Lt(*op),
            LIA::Ite(op) => LIA::Ite(*op),
            LIA::BroadcastIte(op) => LIA::BroadcastIte(*op),
            LIA::Le(op) => LIA::Le(*op),
            LIA::Gt(op) => LIA::Gt(*op),
            LIA::Ge(op) => LIA::Ge(*op),
            LIA::Eq(op) => LIA::Eq(*op),
            LIA::Ne(op) => LIA::Ne(*op),
        }
    }
}

impl<T: fmt::Display> fmt::Display for LIA<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LIA::Num(op) => op.fmt(f),
            LIA::Scale(op) => op.fmt(f),
            LIA::Add(op) => op.fmt(f),
            LIA::Sub(op) => op.fmt(f),
            LIA::Linear(op) => op.fmt(f),
            LIA::Affine(op) => op.fmt(f),
            LIA::ReLU(op) => op.fmt(f),
            LIA::Abs(op) => op.fmt(f),
            LIA::Neg(op) => op.fmt(f),
            LIA::Argmax(op) => op.fmt(f),
            LIA::Min(op) => op.fmt(f),
            LIA::Max(op) => op.fmt(f),
            LIA::Transpose(op) => op.fmt(f),
            LIA::Lt(op) => op.fmt(f),
            LIA::Ite(op) => op.fmt(f),
            LIA::BroadcastIte(op) => op.fmt(f),
            LIA::Le(op) => op.fmt(f),
            LIA::Gt(op) => op.fmt(f),
            LIA::Ge(op) => op.fmt(f),
            LIA::Eq(op) => op.fmt(f),
            LIA::Ne(op) => op.fmt(f),
        }
    }
}

impl<T: Matrix> Boundary<Ints> for NIA<T> {
    fn dom(&self) -> Poly<Ints> {
        match self {
            NIA::Num(op) => op.dom().map(Ints::from),
            NIA::Scale(op) => op.dom().map(Ints::from),
            NIA::Add(op) => op.dom().map(Ints::from),
            NIA::Sub(op) => op.dom().map(Ints::from),
            NIA::Mul(op) => op.dom().map(Ints::from),
            NIA::Div(op) => op.dom().map(Ints::from),
            NIA::Mod(op) => op.dom().map(Ints::from),
            NIA::Linear(op) => op.dom().map(Ints::from),
            NIA::Affine(op) => op.dom().map(Ints::from),
            NIA::ReLU(op) => op.dom().map(Ints::from),
            NIA::Abs(op) => op.dom().map(Ints::from),
            NIA::Neg(op) => op.dom().map(Ints::from),
            NIA::Argmax(op) => op.dom().map(Ints::from),
            NIA::Min(op) => op.dom().map(Ints::from),
            NIA::Max(op) => op.dom().map(Ints::from),
            NIA::Transpose(op) => op.dom().map(Ints::from),
            NIA::MatMul(op) => op.dom().map(Ints::from),
            NIA::Lt(op) => op.dom(),
            NIA::Ite(op) => op.dom(),
            NIA::BroadcastIte(op) => op.dom(),
            NIA::Le(op) => op.dom(),
            NIA::Gt(op) => op.dom(),
            NIA::Ge(op) => op.dom(),
            NIA::Eq(op) => op.dom(),
            NIA::Ne(op) => op.dom(),
        }
    }

    fn cod(&self) -> Poly<Ints> {
        match self {
            NIA::Num(op) => op.cod().map(Ints::from),
            NIA::Scale(op) => op.cod().map(Ints::from),
            NIA::Add(op) => op.cod().map(Ints::from),
            NIA::Sub(op) => op.cod().map(Ints::from),
            NIA::Mul(op) => op.cod().map(Ints::from),
            NIA::Div(op) => op.cod().map(Ints::from),
            NIA::Mod(op) => op.cod().map(Ints::from),
            NIA::Linear(op) => op.cod().map(Ints::from),
            NIA::Affine(op) => op.cod().map(Ints::from),
            NIA::ReLU(op) => op.cod().map(Ints::from),
            NIA::Abs(op) => op.cod().map(Ints::from),
            NIA::Neg(op) => op.cod().map(Ints::from),
            NIA::Argmax(op) => op.cod().map(Ints::from),
            NIA::Min(op) => op.cod().map(Ints::from),
            NIA::Max(op) => op.cod().map(Ints::from),
            NIA::Transpose(op) => op.cod().map(Ints::from),
            NIA::MatMul(op) => op.cod().map(Ints::from),
            NIA::Lt(op) => op.cod(),
            NIA::Ite(op) => op.cod(),
            NIA::BroadcastIte(op) => op.cod(),
            NIA::Le(op) => op.cod(),
            NIA::Gt(op) => op.cod(),
            NIA::Ge(op) => op.cod(),
            NIA::Eq(op) => op.cod(),
            NIA::Ne(op) => op.cod(),
        }
    }
}
impl<T: Matrix> Boundary<Ints> for LIA<T> {
    fn dom(&self) -> Poly<Ints> {
        match self {
            LIA::Num(op) => op.dom().map(Ints::from),
            LIA::Scale(op) => op.dom().map(Ints::from),
            LIA::Add(op) => op.dom().map(Ints::from),
            LIA::Sub(op) => op.dom().map(Ints::from),
            LIA::Linear(op) => op.dom().map(Ints::from),
            LIA::Affine(op) => op.dom().map(Ints::from),
            LIA::ReLU(op) => op.dom().map(Ints::from),
            LIA::Abs(op) => op.dom().map(Ints::from),
            LIA::Neg(op) => op.dom().map(Ints::from),
            LIA::Argmax(op) => op.dom().map(Ints::from),
            LIA::Min(op) => op.dom().map(Ints::from),
            LIA::Max(op) => op.dom().map(Ints::from),
            LIA::Transpose(op) => op.dom().map(Ints::from),
            LIA::Lt(op) => op.dom(),
            LIA::Ite(op) => op.dom(),
            LIA::BroadcastIte(op) => op.dom(),
            LIA::Le(op) => op.dom(),
            LIA::Gt(op) => op.dom(),
            LIA::Ge(op) => op.dom(),
            LIA::Eq(op) => op.dom(),
            LIA::Ne(op) => op.dom(),
        }
    }

    fn cod(&self) -> Poly<Ints> {
        match self {
            LIA::Num(op) => op.cod().map(Ints::from),
            LIA::Scale(op) => op.cod().map(Ints::from),
            LIA::Add(op) => op.cod().map(Ints::from),
            LIA::Sub(op) => op.cod().map(Ints::from),
            LIA::Linear(op) => op.cod().map(Ints::from),
            LIA::Affine(op) => op.cod().map(Ints::from),
            LIA::ReLU(op) => op.cod().map(Ints::from),
            LIA::Abs(op) => op.cod().map(Ints::from),
            LIA::Neg(op) => op.cod().map(Ints::from),
            LIA::Argmax(op) => op.cod().map(Ints::from),
            LIA::Min(op) => op.cod().map(Ints::from),
            LIA::Max(op) => op.cod().map(Ints::from),
            LIA::Transpose(op) => op.cod().map(Ints::from),
            LIA::Lt(op) => op.cod(),
            LIA::Ite(op) => op.cod(),
            LIA::BroadcastIte(op) => op.cod(),
            LIA::Le(op) => op.cod(),
            LIA::Gt(op) => op.cod(),
            LIA::Ge(op) => op.cod(),
            LIA::Eq(op) => op.cod(),
            LIA::Ne(op) => op.cod(),
        }
    }
}

impl<T: Matrix> Boundary<Int> for Num<T> {
    fn dom(&self) -> Poly<Int> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new_unchecked(self.tensor.shape())].into()
    }
}

/// The product with `C` of shape `[m, n]` maps `[m, n]` matrices to `[m, n]`
/// matrices.
impl<T: Matrix> Boundary<Int> for Scale<T> {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new_unchecked(self.tensor.shape())].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new_unchecked(self.tensor.shape())].into()
    }
}

impl Boundary<Int> for Add {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new_unchecked(self.shape); 2].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new_unchecked(self.shape)].into()
    }
}

impl Boundary<Int> for Sub {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new_unchecked(self.shape); 2].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new_unchecked(self.shape)].into()
    }
}

impl Boundary<Int> for Mul {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new_unchecked(self.shape); 2].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new_unchecked(self.shape)].into()
    }
}

impl Boundary<Int> for Div {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new_unchecked(self.shape); 2].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new_unchecked(self.shape)].into()
    }
}

impl Boundary<Int> for Mod {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new_unchecked(self.shape); 2].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new_unchecked(self.shape)].into()
    }
}

/// With `A` of shape `[m, n]`, the map takes `x` of shape `[n, k]` to a
/// matrix of shape `[m, k]`.
impl<T: Matrix> Boundary<Int> for Linear<T> {
    fn dom(&self) -> Poly<Int> {
        let [_, n] = self.a.shape();
        mono![Int::new_unchecked([n, self.cols])].into()
    }

    fn cod(&self) -> Poly<Int> {
        let [m, _] = self.a.shape();
        mono![Int::new_unchecked([m, self.cols])].into()
    }
}

/// With `A` of shape `[m, n]` and `B` of shape `[m, k]`, the map takes `x`
/// of shape `[n, k]` to a matrix of shape `[m, k]`.
impl<T: Matrix> Boundary<Int> for Affine<T> {
    fn dom(&self) -> Poly<Int> {
        let [_, n] = self.a.shape();
        let [_, k] = self.b.shape();
        mono![Int::new_unchecked([n, k])].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new_unchecked(self.b.shape())].into()
    }
}

impl Boundary<Int> for Neg {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new_unchecked(self.shape)].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new_unchecked(self.shape)].into()
    }
}

impl Boundary<Int> for Abs {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new_unchecked(self.shape)].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new_unchecked(self.shape)].into()
    }
}

impl Boundary<Int> for ReLU {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new_unchecked(self.shape)].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new_unchecked(self.shape)].into()
    }
}

impl Boundary<Int> for Min {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new_unchecked(self.shape); 2].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new_unchecked(self.shape)].into()
    }
}

impl Boundary<Int> for Max {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new_unchecked(self.shape); 2].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new_unchecked(self.shape)].into()
    }
}

/// Takes a matrix of shape `[m, n]` to one of shape `[n, m]`.
impl Boundary<Int> for Transpose {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new_unchecked(self.shape)].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new_unchecked([self.shape[1], self.shape[0]])].into()
    }
}

/// Takes matrices of shapes `[m, n]` and `[n, k]` to one of shape `[m, k]`.
impl Boundary<Int> for MatMul {
    fn dom(&self) -> Poly<Int> {
        let [lhs, rhs] = self.shapes;
        mono![Int::new_unchecked(lhs), Int::new_unchecked(rhs)].into()
    }

    fn cod(&self) -> Poly<Int> {
        let [[m, _], [_, k]] = self.shapes;
        mono![Int::new_unchecked([m, k])].into()
    }
}

impl Boundary<Ints> for Lt {
    fn dom(&self) -> Poly<Ints> {
        mono![Int::new_unchecked(self.shape).into(); 2].into()
    }

    fn cod(&self) -> Poly<Ints> {
        mono![Bool::new_unchecked(self.shape).into()].into()
    }
}

impl Boundary<Ints> for Le {
    fn dom(&self) -> Poly<Ints> {
        mono![Int::new_unchecked(self.shape).into(); 2].into()
    }

    fn cod(&self) -> Poly<Ints> {
        mono![Bool::new_unchecked(self.shape).into()].into()
    }
}

impl Boundary<Ints> for Gt {
    fn dom(&self) -> Poly<Ints> {
        mono![Int::new_unchecked(self.shape).into(); 2].into()
    }

    fn cod(&self) -> Poly<Ints> {
        mono![Bool::new_unchecked(self.shape).into()].into()
    }
}

impl Boundary<Ints> for Ge {
    fn dom(&self) -> Poly<Ints> {
        mono![Int::new_unchecked(self.shape).into(); 2].into()
    }

    fn cod(&self) -> Poly<Ints> {
        mono![Bool::new_unchecked(self.shape).into()].into()
    }
}

impl Boundary<Ints> for Eq {
    fn dom(&self) -> Poly<Ints> {
        mono![Int::new_unchecked(self.shape).into(); 2].into()
    }

    fn cod(&self) -> Poly<Ints> {
        mono![Bool::new_unchecked(self.shape).into()].into()
    }
}

impl Boundary<Ints> for Ne {
    fn dom(&self) -> Poly<Ints> {
        mono![Int::new_unchecked(self.shape).into(); 2].into()
    }

    fn cod(&self) -> Poly<Ints> {
        mono![Bool::new_unchecked(self.shape).into()].into()
    }
}

impl<T: fmt::Display> fmt::Display for Scale<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Scale({})", self.tensor)
    }
}

impl<T: fmt::Display> fmt::Display for Num<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.tensor)
    }
}

impl<T: fmt::Display> fmt::Display for Linear<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Linear({})", self.a)
    }
}

impl<T: fmt::Display> fmt::Display for Affine<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Affine({}, {})", self.a, self.b)
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

impl fmt::Display for Neg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Neg")
    }
}

impl fmt::Display for Abs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Abs")
    }
}

impl fmt::Display for ReLU {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ReLU")
    }
}

impl Boundary<Int> for Argmax {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new_unchecked(self.shape)].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new_unchecked([1, 1])].into()
    }
}

impl fmt::Display for Argmax {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Argmax")
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

impl Boundary<Ints> for Ite {
    fn dom(&self) -> Poly<Ints> {
        mono![
            Bool::new_unchecked(self.shape).into(),
            Int::new_unchecked(self.shape).into(),
            Int::new_unchecked(self.shape).into()
        ]
        .into()
    }

    fn cod(&self) -> Poly<Ints> {
        mono![Int::new_unchecked(self.shape).into()].into()
    }
}

impl fmt::Display for Ite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ite")
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

/// Extracts an operation of integer arithmetic from a Python object, trying
/// each operation in turn: those carrying no data through their own
/// classes, and [`Num`], [`Scale`], [`Linear`] and [`Affine`] through the extraction that the
/// tensor type `T` provides.
#[cfg(feature = "pyo3")]
impl<'a, 'py, T> pyo3::FromPyObject<'a, 'py> for NIA<T>
where
    Num<T>: pyo3::FromPyObject<'a, 'py>,
    Scale<T>: pyo3::FromPyObject<'a, 'py>,
    Linear<T>: pyo3::FromPyObject<'a, 'py>,
    Affine<T>: pyo3::FromPyObject<'a, 'py>,
{
    type Error = pyo3::PyErr;

    fn extract(ob: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use pyo3::types::PyAnyMethods;

        if let Ok(op) = ob.extract::<Add>() {
            return Ok(NIA::Add(op));
        }
        if let Ok(op) = ob.extract::<Sub>() {
            return Ok(NIA::Sub(op));
        }
        if let Ok(op) = ob.extract::<Mul>() {
            return Ok(NIA::Mul(op));
        }
        if let Ok(op) = ob.extract::<Div>() {
            return Ok(NIA::Div(op));
        }
        if let Ok(op) = ob.extract::<Mod>() {
            return Ok(NIA::Mod(op));
        }
        if let Ok(op) = ob.extract::<ReLU>() {
            return Ok(NIA::ReLU(op));
        }
        if let Ok(op) = ob.extract::<Abs>() {
            return Ok(NIA::Abs(op));
        }
        if let Ok(op) = ob.extract::<Neg>() {
            return Ok(NIA::Neg(op));
        }
        if let Ok(op) = ob.extract::<Argmax>() {
            return Ok(NIA::Argmax(op));
        }
        if let Ok(op) = ob.extract::<Min>() {
            return Ok(NIA::Min(op));
        }
        if let Ok(op) = ob.extract::<Max>() {
            return Ok(NIA::Max(op));
        }
        if let Ok(op) = ob.extract::<Transpose>() {
            return Ok(NIA::Transpose(op));
        }
        if let Ok(op) = ob.extract::<MatMul>() {
            return Ok(NIA::MatMul(op));
        }
        if let Ok(op) = ob.extract::<Lt>() {
            return Ok(NIA::Lt(op));
        }
        if let Ok(op) = ob.extract::<Ite>() {
            return Ok(NIA::Ite(op));
        }
        if let Ok(op) = ob.extract::<broadcast::Ite>() {
            return Ok(NIA::BroadcastIte(op));
        }
        if let Ok(op) = ob.extract::<Le>() {
            return Ok(NIA::Le(op));
        }
        if let Ok(op) = ob.extract::<Gt>() {
            return Ok(NIA::Gt(op));
        }
        if let Ok(op) = ob.extract::<Ge>() {
            return Ok(NIA::Ge(op));
        }
        if let Ok(op) = ob.extract::<Eq>() {
            return Ok(NIA::Eq(op));
        }
        if let Ok(op) = ob.extract::<Ne>() {
            return Ok(NIA::Ne(op));
        }
        if let Ok(op) = ob.extract::<Num<T>>() {
            return Ok(NIA::Num(op));
        }
        if let Ok(op) = ob.extract::<Scale<T>>() {
            return Ok(NIA::Scale(op));
        }
        if let Ok(op) = ob.extract::<Linear<T>>() {
            return Ok(NIA::Linear(op));
        }
        if let Ok(op) = ob.extract::<Affine<T>>() {
            return Ok(NIA::Affine(op));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "expected an operation of integer arithmetic, got {}",
            ob.get_type()
        )))
    }
}

impl Add {
    /// The operation reading the integer matrices of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Int; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Add { shape: s.shape() })
    }

    /// The operation writing the integer matrix of `cod`.
    pub fn with_cod(cod: [Int; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Add { shape: s.shape() })
    }
}

// The Python constructors of the operations carrying no tensor: each takes
// its sizes positionally, in the order of its fields, the shape as a list.
#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Add {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Int; 2], cod: Option<Int>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
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
        crate::py_boundary(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_boundary(py, Boundary::cod(self))
    }
}

impl Sub {
    /// The operation reading the integer matrices of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Int; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Sub { shape: s.shape() })
    }

    /// The operation writing the integer matrix of `cod`.
    pub fn with_cod(cod: [Int; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Sub { shape: s.shape() })
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
    fn py_new(dom: [Int; 2], cod: Option<Int>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
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
        crate::py_boundary(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_boundary(py, Boundary::cod(self))
    }
}

impl Mul {
    /// The operation reading the integer matrices of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Int; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Mul { shape: s.shape() })
    }

    /// The operation writing the integer matrix of `cod`.
    pub fn with_cod(cod: [Int; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Mul { shape: s.shape() })
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
    fn py_new(dom: [Int; 2], cod: Option<Int>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
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
        crate::py_boundary(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_boundary(py, Boundary::cod(self))
    }
}

impl Div {
    /// The operation reading the integer matrices of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Int; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Div { shape: s.shape() })
    }

    /// The operation writing the integer matrix of `cod`.
    pub fn with_cod(cod: [Int; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Div { shape: s.shape() })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Div {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Int; 2], cod: Option<Int>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
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
        crate::py_boundary(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_boundary(py, Boundary::cod(self))
    }
}

impl Mod {
    /// The operation reading the integer matrices of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Int; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Mod { shape: s.shape() })
    }

    /// The operation writing the integer matrix of `cod`.
    pub fn with_cod(cod: [Int; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Mod { shape: s.shape() })
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
    fn py_new(dom: [Int; 2], cod: Option<Int>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
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
        crate::py_boundary(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_boundary(py, Boundary::cod(self))
    }
}

impl Neg {
    /// The operation reading the integer matrix of `dom`.
    pub fn with_dom(dom: [Int; 1]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Neg { shape: s.shape() })
    }

    /// The operation writing the integer matrix of `cod`.
    pub fn with_cod(cod: [Int; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Neg { shape: s.shape() })
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
    fn py_new(dom: Int, cod: Option<Int>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom([dom]), cod)
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
        crate::py_boundary(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_boundary(py, Boundary::cod(self))
    }
}

impl Abs {
    /// The operation reading the integer matrix of `dom`.
    pub fn with_dom(dom: [Int; 1]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Abs { shape: s.shape() })
    }

    /// The operation writing the integer matrix of `cod`.
    pub fn with_cod(cod: [Int; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Abs { shape: s.shape() })
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
    fn py_new(dom: Int, cod: Option<Int>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom([dom]), cod)
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
        crate::py_boundary(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_boundary(py, Boundary::cod(self))
    }
}

impl ReLU {
    /// The operation reading the integer matrix of `dom`.
    pub fn with_dom(dom: [Int; 1]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(ReLU { shape: s.shape() })
    }

    /// The operation writing the integer matrix of `cod`.
    pub fn with_cod(cod: [Int; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(ReLU { shape: s.shape() })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl ReLU {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: Int, cod: Option<Int>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom([dom]), cod)
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
        crate::py_boundary(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_boundary(py, Boundary::cod(self))
    }
}

impl Argmax {
    /// The operation reading the integer matrix of `dom`.
    pub fn with_dom(dom: [Int; 1]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Argmax { shape: s.shape() })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Argmax {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: Int, cod: Option<Int>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom([dom]), cod)
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
        crate::py_boundary(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_boundary(py, Boundary::cod(self))
    }
}

impl Min {
    /// The operation reading the integer matrices of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Int; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Min { shape: s.shape() })
    }

    /// The operation writing the integer matrix of `cod`.
    pub fn with_cod(cod: [Int; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Min { shape: s.shape() })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Min {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Int; 2], cod: Option<Int>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
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
        crate::py_boundary(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_boundary(py, Boundary::cod(self))
    }
}

impl Max {
    /// The operation reading the integer matrices of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Int; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Max { shape: s.shape() })
    }

    /// The operation writing the integer matrix of `cod`.
    pub fn with_cod(cod: [Int; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Max { shape: s.shape() })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Max {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Int; 2], cod: Option<Int>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
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
        crate::py_boundary(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_boundary(py, Boundary::cod(self))
    }
}

impl Transpose {
    /// The operation reading the integer matrix of `dom`.
    pub fn with_dom(dom: [Int; 1]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Transpose { shape: s.shape() })
    }

    /// The transposition writing the integer matrix of `cod`, reading its transpose.
    pub fn with_cod(cod: [Int; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        let [m, n] = s.shape();
        Ok(Transpose { shape: [n, m] })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Transpose {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: Int, cod: Option<Int>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom([dom]), cod)
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
        crate::py_boundary(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_boundary(py, Boundary::cod(self))
    }
}

impl MatMul {
    /// The product of the integer matrices of `dom`; fails unless the columns
    /// of the first are the rows of the second.
    pub fn with_dom(dom: [Int; 2]) -> Result<Self, crate::Error> {
        let [x, y] = dom;
        let shapes = [x.shape(), y.shape()];
        let [lhs, rhs] = shapes;
        if lhs[1] != rhs[0] {
            return Err(crate::Error(format!(
                "cannot multiply a {lhs:?} matrix by a {rhs:?} matrix"
            )));
        }
        Ok(MatMul { shapes })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl MatMul {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Int; 2], cod: Option<Int>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
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
        crate::py_boundary(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_boundary(py, Boundary::cod(self))
    }
}

impl Lt {
    /// The operation reading the integer matrices of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Int; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Lt { shape: s.shape() })
    }

    /// The operation writing the boolean matrix of `cod`.
    pub fn with_cod(cod: [Bool; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Lt { shape: s.shape() })
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
    fn py_new(dom: [Int; 2], cod: Option<Bool>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
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
        crate::py_boundary(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_boundary(py, Boundary::cod(self))
    }
}

impl Ite {
    /// The choice reading the boolean condition and the branches of `dom`;
    /// fails unless the branches are of the same sort, and the condition of
    /// their shape.
    pub fn with_dom(dom: [Ints; 3]) -> Result<Self, crate::Error> {
        match dom {
            [Ints::Bool(c), Ints::Int(x), Ints::Int(y)] if x == y && c.shape() == x.shape() => {
                Ok(Ite { shape: x.shape() })
            }
            _ => Err(crate::unexpected_dom(
                "a condition Bool and two branches Int of its shape",
                &dom,
            )),
        }
    }

    /// The operation writing the integer matrix of `cod`.
    pub fn with_cod(cod: [Int; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Ite { shape: s.shape() })
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
    fn py_new(dom: [Ints; 3], cod: Option<Int>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
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
        crate::py_boundary(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_boundary(py, Boundary::cod(self))
    }
}

impl Le {
    /// The operation reading the integer matrices of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Int; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Le { shape: s.shape() })
    }

    /// The operation writing the boolean matrix of `cod`.
    pub fn with_cod(cod: [Bool; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Le { shape: s.shape() })
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
    fn py_new(dom: [Int; 2], cod: Option<Bool>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
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
        crate::py_boundary(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_boundary(py, Boundary::cod(self))
    }
}

impl Gt {
    /// The operation reading the integer matrices of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Int; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Gt { shape: s.shape() })
    }

    /// The operation writing the boolean matrix of `cod`.
    pub fn with_cod(cod: [Bool; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Gt { shape: s.shape() })
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
    fn py_new(dom: [Int; 2], cod: Option<Bool>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
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
        crate::py_boundary(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_boundary(py, Boundary::cod(self))
    }
}

impl Ge {
    /// The operation reading the integer matrices of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Int; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Ge { shape: s.shape() })
    }

    /// The operation writing the boolean matrix of `cod`.
    pub fn with_cod(cod: [Bool; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Ge { shape: s.shape() })
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
    fn py_new(dom: [Int; 2], cod: Option<Bool>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
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
        crate::py_boundary(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_boundary(py, Boundary::cod(self))
    }
}

impl Eq {
    /// The operation reading the integer matrices of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Int; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Eq { shape: s.shape() })
    }

    /// The operation writing the boolean matrix of `cod`.
    pub fn with_cod(cod: [Bool; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Eq { shape: s.shape() })
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
    fn py_new(dom: [Int; 2], cod: Option<Bool>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
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
        crate::py_boundary(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_boundary(py, Boundary::cod(self))
    }
}

impl Ne {
    /// The operation reading the integer matrices of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Int; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Ne { shape: s.shape() })
    }

    /// The operation writing the boolean matrix of `cod`.
    pub fn with_cod(cod: [Bool; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Ne { shape: s.shape() })
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
    fn py_new(dom: [Int; 2], cod: Option<Bool>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
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
        crate::py_boundary(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_boundary(py, Boundary::cod(self))
    }
}

/// The operations of the theory that broadcast a scalar operand, of the
/// shape `[1, 1]`, to the shape of the others; the others are in the parent
/// module.
pub mod broadcast {
    use super::*;

    /// Choice `if c then x else y` between whole `[m, n]` integer matrices `x` and
    /// `y`, by the scalar boolean `c`, of the shape `[1, 1]`, read first: the
    /// pointwise [`super::Ite`] by `c` broadcast to `[m, n]`.
    ///
    /// SMT-LIB2: `(ite c x y)`, by the only entry of `c`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.int.broadcast")
    )]
    pub struct Ite {
        shape: [usize; 2],
    }

    impl Boundary<Ints> for Ite {
        fn dom(&self) -> Poly<Ints> {
            mono![
                Bool::new_unchecked([1, 1]).into(),
                Int::new_unchecked(self.shape).into(),
                Int::new_unchecked(self.shape).into()
            ]
            .into()
        }

        fn cod(&self) -> Poly<Ints> {
            mono![Int::new_unchecked(self.shape).into()].into()
        }
    }

    impl fmt::Display for Ite {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "Ite")
        }
    }

    impl Ite {
        /// The choice reading the scalar boolean condition and the branches of
        /// `dom`; fails unless the condition is of the shape `[1, 1]` and the
        /// branches of the same sort.
        pub fn with_dom(dom: [Ints; 3]) -> Result<Self, crate::Error> {
            match dom {
                [Ints::Bool(c), Ints::Int(x), Ints::Int(y)] if x == y && c.shape() == [1, 1] => {
                    Ok(Ite { shape: x.shape() })
                }
                _ => Err(crate::unexpected_dom(
                    "a condition Bool([1, 1]) and two branches Int of the same sort",
                    &dom,
                )),
            }
        }

        /// The choice writing the integer matrix of `cod`.
        pub fn with_cod(cod: [Int; 1]) -> Result<Self, crate::Error> {
            let [s] = cod;
            Ok(Ite { shape: s.shape() })
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
        fn py_new(dom: [Ints; 3], cod: Option<Int>) -> Result<Self, crate::Error> {
            crate::check_cod(Self::with_dom(dom), cod)
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
            crate::py_boundary(py, Boundary::dom(self))
        }

        /// The sort it writes.
        #[cfg(feature = "pyo3")]
        #[getter(cod)]
        fn py_cod<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::py_boundary(py, Boundary::cod(self))
        }
    }
}
