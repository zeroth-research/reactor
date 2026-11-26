/*!
# Real arithmetic

Defines one type per operation on the sort [`Real`] of real matrices. Every operation reads and writes real matrices, and carries enough
information to compute its [`Boundary`]:

- [`Num`] — a constant real matrix literal.
- [`Scale`] — the pointwise product `C ⊙ x` with a constant real matrix `C`.
- [`Add`], [`Sub`] — pointwise addition and subtraction.
- [`Mul`] — pointwise multiplication `x ⊙ y`.
- [`Div`] — pointwise division `x / y`.
- [`Linear`]`(A)` — the linear map `x ↦ A·x`, with `A` a constant real
  matrix.
- [`Affine`]`(A, B)` — the affine map `x ↦ A·x + B`, with `A` and `B`
  constant real matrices of compatible shapes.
- [`Zero`] — the zero matrix of a degree: the zero rate of change.
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

The comparisons are those of SMT-LIB2's `Reals` theory, `<`, `<=`, `>` and
`>=`, together with `=` and `distinct` of the Core theory, lifted pointwise
to matrices. As they write booleans, their boundary is in [`Sort`], the
sorts of the theory, rather than in [`Real`].
*/

use super::bool::Bool;
use crate::Matrix;
use base::boundary::{Boundary, Poly};
use base::differentiable::Differentiable;
use base::mono;
use derive_more::From;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use std::fmt;
use subenum::subenum;

/// The sort of real matrices with the given `[rows, cols]` shape, graded by
/// their `degree`: values of degree `d` are `d`-th order rates of change,
/// and plain reals have degree 0.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, get_all, module = "reactor.real")
)]
pub struct Real {
    degree: u8,
    shape: [usize; 2],
}

impl Real {
    /// Real matrices of the given shape, of the given `degree`: the
    /// `degree`-th order rates of change of plain ones. Outside this crate,
    /// they are written with the tangent, `T(Real::new_unchecked(shape))`.
    pub(crate) fn with_degree(shape: [usize; 2], degree: u8) -> Self {
        Real { degree, shape }
    }

    /// Plain real matrices of the given shape, as [`Real::new`], but panics
    /// if `shape` is empty.
    pub fn new_unchecked(shape: [usize; 2]) -> Self {
        Self::new(shape).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn degree(&self) -> u8 {
        self.degree
    }

    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

/// Python matches a `Real` by `shape`.
#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Real {
    /// Plain real matrices of the given shape, of degree 0; those of a
    /// higher degree are their tangents, `T(Real(shape))`. Fails unless the
    /// shape has at least one row and one column.
    #[cfg_attr(feature = "pyo3", new)]
    pub fn new(shape: [usize; 2]) -> Result<Self, crate::Error> {
        crate::matrix::check_shape(shape)?;
        Ok(Real { degree: 0, shape })
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
    fn tangent(&self) -> Real {
        Differentiable::tangent(self)
    }
}

impl fmt::Display for Real {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [i, j] = self.shape;
        match self.degree {
            0 => write!(f, "Real([{i}, {j}])"),
            1 => write!(f, "T(Real([{i}, {j}]))"),
            d => write!(f, "T^{d}(Real([{i}, {j}]))"),
        }
    }
}

/// The rate of change of a real matrix is a real matrix of the same shape,
/// one degree higher.
impl Differentiable for Real {
    type Tangent = Real;

    fn tangent(self) -> Real {
        Real {
            degree: self
                .degree
                .checked_add(1)
                .expect("real sort degree out of range"),
            shape: self.shape,
        }
    }
}

/// The sorts of the theory, as in the signature of SMT-LIB2's `Reals`:
/// real matrices, and the booleans of the Core theory, which comparisons
/// write, together with [`Zero`](crate::zero::Real0). It is the subset
/// [`Reals`](crate::Reals) of the sorts of all theories.
pub use crate::Reals as Sort;

/// A constant real matrix literal, held in a tensor of type `T`, of a
/// degree: a constant rate of change, such as the `1` of `ẋ = 1`, is of
/// degree 1.
#[derive(Debug)]
pub struct Num<T> {
    tensor: T,
    degree: u8,
}

impl<T: Matrix> Num<T> {
    /// The constant held in `tensor`, writing the real matrix of `cod`; fails
    /// unless `tensor` is of its shape.
    pub fn with_cod(tensor: T, cod: [Real; 1]) -> Result<Self, crate::Error> {
        let [x] = cod;
        if tensor.shape() != x.shape() {
            return Err(crate::unexpected_cod(
                &format!("1 sort of the shape {:?} of the constant", tensor.shape()),
                &cod,
            ));
        }
        Ok(Num {
            tensor,
            degree: x.degree(),
        })
    }
}

impl<T> Num<T> {
    /// The tensor holding the constant.
    pub fn tensor(&self) -> &T {
        &self.tensor
    }

    /// The degree of the constant.
    pub fn degree(&self) -> u8 {
        self.degree
    }
}

/// The pointwise product `x ↦ C ⊙ x` of `[m, n]` matrices with a constant
/// `C` of shape `[m, n]`, held in a tensor of type `T`: linear in `x`, unlike
/// [`Mul`], which multiplies two variables.
///
/// `C` is a plain real, so the product keeps the degree of `x`.
#[derive(Debug)]
pub struct Scale<T> {
    tensor: T,
    degree: u8,
}

impl<T: Matrix> Scale<T> {
    /// The product with the constant held in `tensor`, reading the real
    /// matrix of `dom`, whose degree it keeps; fails unless `tensor` is of
    /// its shape.
    pub fn with_dom(tensor: T, dom: [Real; 1]) -> Result<Self, crate::Error> {
        let [x] = dom;
        if tensor.shape() != x.shape() {
            return Err(crate::unexpected_dom(
                &format!("1 sort of the shape {:?} of C", tensor.shape()),
                &dom,
            ));
        }
        Ok(Scale {
            tensor,
            degree: x.degree(),
        })
    }

    /// The product with the constant held in `tensor`, writing the real
    /// matrix of `cod`, whose degree it keeps; fails unless `tensor` is of its shape.
    pub fn with_cod(tensor: T, cod: [Real; 1]) -> Result<Self, crate::Error> {
        let [x] = cod;
        if tensor.shape() != x.shape() {
            return Err(crate::unexpected_cod(
                &format!("1 sort of the shape {:?} of C", tensor.shape()),
                &cod,
            ));
        }
        Ok(Scale {
            tensor,
            degree: x.degree(),
        })
    }
}

impl<T> Scale<T> {
    /// The tensor holding the constant `C`.
    pub fn tensor(&self) -> &T {
        &self.tensor
    }

    /// The degree of the matrices it maps.
    pub fn degree(&self) -> u8 {
        self.degree
    }
}

/// Pointwise addition `x + y` of `[m, n]` matrices, of the same degree,
/// which the sum keeps.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.real")
)]
pub struct Add {
    shape: [usize; 2],
    degree: u8,
}

/// Pointwise subtraction `x - y` of `[m, n]` matrices, of the same degree,
/// which the difference keeps.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.real")
)]
pub struct Sub {
    shape: [usize; 2],
    degree: u8,
}

/// Pointwise multiplication `x ⊙ y` of `[m, n]` matrices: the Hadamard
/// product, not the matrix product [`MatMul`]. The degrees of the factors
/// add up: the product of a rate of change by a plain real is a rate of
/// change, and that of two rates of change is of degree 2.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.real")
)]
pub struct Mul {
    shape: [usize; 2],
    /// The degrees `[p, q]` of the factors.
    ///
    /// Entry `i` belongs to input `i`, as indexed in the domain of the
    /// boundary: an array rather than one field per input, so that the
    /// two match by position.
    degrees: [u8; 2],
}

/// Pointwise division `x / y` of `[m, n]` matrices: the Hadamard division
/// `x ⊘ y`. As in SMT-LIB2, `x / 0` is unspecified.
///
/// SMT-LIB2: `(/ x y)`.
///
/// The degree of the divisor is taken off that of the dividend: dividing a
/// rate of change by a plain real is a rate of change, while dividing by a
/// rate of change of a higher degree than the dividend is not defined, as
/// degrees are not negative.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.real")
)]
pub struct Div {
    shape: [usize; 2],
    /// The degrees `[p, q]` of the dividend and the divisor.
    ///
    /// Entry `i` belongs to input `i`, as indexed in the domain of the
    /// boundary: an array rather than one field per input, so that the
    /// two match by position.
    degrees: [u8; 2],
}

/// The linear map `x ↦ A·x`, where `A` is a plain real constant, so the map
/// keeps the degree of `x`. `A` determines the rows of `x` but not its
/// columns, which the map holds.
#[derive(Debug)]
pub struct Linear<T> {
    a: T,
    cols: usize,
    degree: u8,
}

impl<T: Matrix> Linear<T> {
    /// The linear map `x ↦ A·x` reading the real matrix `x` of `dom`, whose
    /// degree it keeps; fails unless `a` is of shape `[m, n]` and `x` of
    /// shape `[n, k]`.
    pub fn with_dom(a: T, dom: [Real; 1]) -> Result<Self, crate::Error> {
        let [x] = dom;
        crate::matrix::check_shape(a.shape())?;
        let ([_, n], [rows, k]) = (a.shape(), x.shape());
        if rows != n {
            return Err(crate::unexpected_dom(
                &format!("1 sort of shape [{n}, k]"),
                &dom,
            ));
        }
        Ok(Linear {
            a,
            cols: k,
            degree: x.degree(),
        })
    }

    /// The linear map `x ↦ A·x` writing the real matrix of `cod`, whose
    /// degree it keeps; fails unless `a` is of shape `[m, n]` and that
    /// matrix of shape `[m, k]`.
    pub fn with_cod(a: T, cod: [Real; 1]) -> Result<Self, crate::Error> {
        let [x] = cod;
        crate::matrix::check_shape(a.shape())?;
        let ([m, _], [rows, k]) = (a.shape(), x.shape());
        if rows != m {
            return Err(crate::unexpected_cod(
                &format!("1 sort of shape [{m}, k]"),
                &cod,
            ));
        }
        Ok(Linear {
            a,
            cols: k,
            degree: x.degree(),
        })
    }
}

impl<T> Linear<T> {
    /// The tensor of `A`.
    pub fn a(&self) -> &T {
        &self.a
    }

    /// The degree of the matrices it maps.
    pub fn degree(&self) -> u8 {
        self.degree
    }
}

/// The affine map `x ↦ A·x + B`, where `A` and `B` are constants: `A` a
/// plain real, and `B` of the degree of `x`, which the map keeps.
#[derive(Debug)]
pub struct Affine<T> {
    a: T,
    b: T,
    degree: u8,
}

impl<T: Matrix> Affine<T> {
    /// The affine map `x ↦ A·x + B` reading the real matrix `x` of `dom`,
    /// whose degree it keeps; fails unless `a` and `b` are of shapes
    /// `[m, n]` and `[m, k]`, and `x` of shape `[n, k]`.
    pub fn with_dom(a: T, b: T, dom: [Real; 1]) -> Result<Self, crate::Error> {
        let [x] = dom;
        crate::matrix::check_affine(a.shape(), b.shape())?;
        let op = Affine {
            a,
            b,
            degree: x.degree(),
        };
        let ([_, n], [_, k]) = (op.a.shape(), op.b.shape());
        if x.shape() != [n, k] {
            return Err(crate::unexpected_dom(
                &format!("1 sort of shape {:?}", [n, k]),
                &dom,
            ));
        }
        Ok(op)
    }

    /// The affine map `x ↦ A·x + B` writing the real matrix of `cod`, whose degree it keeps;
    /// fails unless `a` and `b` are of shapes `[m, n]` and `[m, k]`, and
    /// that matrix of shape `[m, k]`.
    pub fn with_cod(a: T, b: T, cod: [Real; 1]) -> Result<Self, crate::Error> {
        let [x] = cod;
        crate::matrix::check_affine(a.shape(), b.shape())?;
        let op = Affine {
            a,
            b,
            degree: x.degree(),
        };
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

    /// The degree of the matrices it maps.
    pub fn degree(&self) -> u8 {
        self.degree
    }
}

/// The zero `[m, n]` matrix of a degree, reading nothing: of degree 1, the
/// rate of change of a real matrix that does not change.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.real")
)]
pub struct Zero {
    shape: [usize; 2],
    degree: u8,
}

/// Pointwise negation `x ↦ -x` of `[m, n]` matrices, of a degree which
/// the negation keeps.
///
/// SMT-LIB2: `(- x)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.real")
)]
pub struct Neg {
    shape: [usize; 2],
    degree: u8,
}

/// Pointwise absolute value `x ↦ |x|` of `[m, n]` matrices. SMT-LIB2's
/// `Reals` has no `abs`, unlike `Ints`; it is `(ite (>= x 0) x (- x))`.
///
/// Of a degree, which the absolute value keeps.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.real")
)]
pub struct Abs {
    shape: [usize; 2],
    degree: u8,
}

/// The rectified-linear map `x ↦ max(x, 0)`, applied pointwise to
/// `[m, n]` matrices.
///
/// Of a degree, which the map keeps.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.real")
)]
pub struct ReLU {
    shape: [usize; 2],
    degree: u8,
}

/// The index of a maximal entry of an `[m, n]` real matrix, counting its
/// entries row by row, written as a `[1, 1]` matrix of the same sort.
///
/// Of an operand of any degree: the index it writes is of degree 0.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.real")
)]
pub struct Argmax {
    shape: [usize; 2],
    degree: u8,
}

/// Pointwise minimum `min(x, y)` of `[m, n]` matrices.
///
/// Of operands of the same degree, which the minimum keeps.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.real")
)]
pub struct Min {
    shape: [usize; 2],
    degree: u8,
}

/// Pointwise maximum `max(x, y)` of `[m, n]` matrices.
///
/// Of operands of the same degree, which the maximum keeps.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.real")
)]
pub struct Max {
    shape: [usize; 2],
    degree: u8,
}

/// Transposition `xᵀ` of `[m, n]` matrices.
///
/// Of a degree, which the transposition keeps.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.real")
)]
pub struct Transpose {
    shape: [usize; 2],
    degree: u8,
}

/// The matrix product `x·y` of an `[m, n]` matrix `x` and an `[n, k]`
/// matrix `y`, with the shapes given as `MatMul([[m, n], [n, k]])`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.real")
)]
pub struct MatMul {
    /// The shapes `[[m, n], [n, k]]` of the factors.
    ///
    /// Entry `i` belongs to input `i`, as indexed in the domain of the
    /// boundary: an array rather than one field per input, so that the
    /// two match by position.
    shapes: [[usize; 2]; 2],
    /// The degrees `[p, q]` of the factors, which add up as for [`Mul`],
    /// matching the shapes by position.
    degrees: [u8; 2],
}

/// Pointwise comparison `x < y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(< x y)`.
///
/// Of operands of the same degree.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.real")
)]
pub struct Lt {
    shape: [usize; 2],
    degree: u8,
}

/// Pointwise choice `if c then x else y` between `[m, n]` real matrices `x` and `y`,
/// by the boolean matrix `c`, read first.
///
/// SMT-LIB2: `(ite c x y)`.
///
/// Of branches of the same degree, which the choice keeps.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.real")
)]
pub struct Ite {
    shape: [usize; 2],
    degree: u8,
}

/// Pointwise comparison `x ≤ y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(<= x y)`.
///
/// Of operands of the same degree.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.real")
)]
pub struct Le {
    shape: [usize; 2],
    degree: u8,
}

/// Pointwise comparison `x > y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(> x y)`.
///
/// Of operands of the same degree.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.real")
)]
pub struct Gt {
    shape: [usize; 2],
    degree: u8,
}

/// Pointwise comparison `x ≥ y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(>= x y)`.
///
/// Of operands of the same degree.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.real")
)]
pub struct Ge {
    shape: [usize; 2],
    degree: u8,
}

/// Pointwise comparison `x = y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(= x y)`, of the Core theory.
///
/// Of operands of the same degree.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.real")
)]
pub struct Eq {
    shape: [usize; 2],
    degree: u8,
}

/// Pointwise comparison `x ≠ y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(distinct x y)`, of the Core theory.
///
/// Of operands of the same degree.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.real")
)]
pub struct Ne {
    shape: [usize; 2],
    degree: u8,
}

/// The operations of the theory of real arithmetic, SMT-LIB2's `Reals`,
/// with their boundary in [`Sort`].
///
/// `subenum` defines [`LRA`], the linear ones: all but `Mul`, `Div` and
/// `MatMul`, which multiply or divide two variables. It also generates the
/// casts between them: `From` up to `NRA`, and `TryFrom` down from it.
#[subenum(LRA)]
#[derive(Debug, From)]
pub enum NRA<T> {
    #[subenum(LRA)]
    Num(Num<T>),
    #[subenum(LRA)]
    Scale(Scale<T>),
    #[subenum(LRA)]
    Add(Add),
    #[subenum(LRA)]
    Sub(Sub),
    Mul(Mul),
    Div(Div),
    #[subenum(LRA)]
    Linear(Linear<T>),
    #[subenum(LRA)]
    Affine(Affine<T>),
    #[subenum(LRA)]
    ReLU(ReLU),
    #[subenum(LRA)]
    Abs(Abs),
    #[subenum(LRA)]
    Neg(Neg),
    #[subenum(LRA)]
    Zero(Zero),
    #[subenum(LRA)]
    Argmax(Argmax),
    #[subenum(LRA)]
    Min(Min),
    #[subenum(LRA)]
    Max(Max),
    #[subenum(LRA)]
    Transpose(Transpose),
    MatMul(MatMul),
    #[subenum(LRA)]
    Lt(Lt),
    #[subenum(LRA)]
    Ite(Ite),
    #[subenum(LRA)]
    BroadcastIte(broadcast::Ite),
    #[subenum(LRA)]
    Le(Le),
    #[subenum(LRA)]
    Gt(Gt),
    #[subenum(LRA)]
    Ge(Ge),
    #[subenum(LRA)]
    Eq(Eq),
    #[subenum(LRA)]
    Ne(Ne),
}

impl<T> Clone for NRA<T>
where
    Num<T>: Clone,
    Scale<T>: Clone,
    Linear<T>: Clone,
    Affine<T>: Clone,
{
    fn clone(&self) -> Self {
        match self {
            NRA::Num(op) => NRA::Num(op.clone()),
            NRA::Scale(op) => NRA::Scale(op.clone()),
            NRA::Add(op) => NRA::Add(*op),
            NRA::Sub(op) => NRA::Sub(*op),
            NRA::Mul(op) => NRA::Mul(*op),
            NRA::Div(op) => NRA::Div(*op),
            NRA::Linear(op) => NRA::Linear(op.clone()),
            NRA::Affine(op) => NRA::Affine(op.clone()),
            NRA::ReLU(op) => NRA::ReLU(*op),
            NRA::Abs(op) => NRA::Abs(*op),
            NRA::Neg(op) => NRA::Neg(*op),
            NRA::Zero(op) => NRA::Zero(*op),
            NRA::Argmax(op) => NRA::Argmax(*op),
            NRA::Min(op) => NRA::Min(*op),
            NRA::Max(op) => NRA::Max(*op),
            NRA::Transpose(op) => NRA::Transpose(*op),
            NRA::MatMul(op) => NRA::MatMul(*op),
            NRA::Lt(op) => NRA::Lt(*op),
            NRA::Ite(op) => NRA::Ite(*op),
            NRA::BroadcastIte(op) => NRA::BroadcastIte(*op),
            NRA::Le(op) => NRA::Le(*op),
            NRA::Gt(op) => NRA::Gt(*op),
            NRA::Ge(op) => NRA::Ge(*op),
            NRA::Eq(op) => NRA::Eq(*op),
            NRA::Ne(op) => NRA::Ne(*op),
        }
    }
}

impl<T: fmt::Display> fmt::Display for NRA<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NRA::Num(op) => op.fmt(f),
            NRA::Scale(op) => op.fmt(f),
            NRA::Add(op) => op.fmt(f),
            NRA::Sub(op) => op.fmt(f),
            NRA::Mul(op) => op.fmt(f),
            NRA::Div(op) => op.fmt(f),
            NRA::Linear(op) => op.fmt(f),
            NRA::Affine(op) => op.fmt(f),
            NRA::ReLU(op) => op.fmt(f),
            NRA::Abs(op) => op.fmt(f),
            NRA::Neg(op) => op.fmt(f),
            NRA::Zero(op) => op.fmt(f),
            NRA::Argmax(op) => op.fmt(f),
            NRA::Min(op) => op.fmt(f),
            NRA::Max(op) => op.fmt(f),
            NRA::Transpose(op) => op.fmt(f),
            NRA::MatMul(op) => op.fmt(f),
            NRA::Lt(op) => op.fmt(f),
            NRA::Ite(op) => op.fmt(f),
            NRA::BroadcastIte(op) => op.fmt(f),
            NRA::Le(op) => op.fmt(f),
            NRA::Gt(op) => op.fmt(f),
            NRA::Ge(op) => op.fmt(f),
            NRA::Eq(op) => op.fmt(f),
            NRA::Ne(op) => op.fmt(f),
        }
    }
}

impl<T> Clone for LRA<T>
where
    Num<T>: Clone,
    Scale<T>: Clone,
    Linear<T>: Clone,
    Affine<T>: Clone,
{
    fn clone(&self) -> Self {
        match self {
            LRA::Num(op) => LRA::Num(op.clone()),
            LRA::Scale(op) => LRA::Scale(op.clone()),
            LRA::Add(op) => LRA::Add(*op),
            LRA::Sub(op) => LRA::Sub(*op),
            LRA::Linear(op) => LRA::Linear(op.clone()),
            LRA::Affine(op) => LRA::Affine(op.clone()),
            LRA::ReLU(op) => LRA::ReLU(*op),
            LRA::Abs(op) => LRA::Abs(*op),
            LRA::Neg(op) => LRA::Neg(*op),
            LRA::Zero(op) => LRA::Zero(*op),
            LRA::Argmax(op) => LRA::Argmax(*op),
            LRA::Min(op) => LRA::Min(*op),
            LRA::Max(op) => LRA::Max(*op),
            LRA::Transpose(op) => LRA::Transpose(*op),
            LRA::Lt(op) => LRA::Lt(*op),
            LRA::Ite(op) => LRA::Ite(*op),
            LRA::BroadcastIte(op) => LRA::BroadcastIte(*op),
            LRA::Le(op) => LRA::Le(*op),
            LRA::Gt(op) => LRA::Gt(*op),
            LRA::Ge(op) => LRA::Ge(*op),
            LRA::Eq(op) => LRA::Eq(*op),
            LRA::Ne(op) => LRA::Ne(*op),
        }
    }
}

impl<T: fmt::Display> fmt::Display for LRA<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LRA::Num(op) => op.fmt(f),
            LRA::Scale(op) => op.fmt(f),
            LRA::Add(op) => op.fmt(f),
            LRA::Sub(op) => op.fmt(f),
            LRA::Linear(op) => op.fmt(f),
            LRA::Affine(op) => op.fmt(f),
            LRA::ReLU(op) => op.fmt(f),
            LRA::Abs(op) => op.fmt(f),
            LRA::Neg(op) => op.fmt(f),
            LRA::Zero(op) => op.fmt(f),
            LRA::Argmax(op) => op.fmt(f),
            LRA::Min(op) => op.fmt(f),
            LRA::Max(op) => op.fmt(f),
            LRA::Transpose(op) => op.fmt(f),
            LRA::Lt(op) => op.fmt(f),
            LRA::Ite(op) => op.fmt(f),
            LRA::BroadcastIte(op) => op.fmt(f),
            LRA::Le(op) => op.fmt(f),
            LRA::Gt(op) => op.fmt(f),
            LRA::Ge(op) => op.fmt(f),
            LRA::Eq(op) => op.fmt(f),
            LRA::Ne(op) => op.fmt(f),
        }
    }
}

impl<T: Matrix> Boundary<Sort> for NRA<T> {
    fn dom(&self) -> Poly<Sort> {
        match self {
            NRA::Num(op) => op.dom().map(Sort::from),
            NRA::Scale(op) => op.dom().map(Sort::from),
            NRA::Add(op) => op.dom().map(Sort::from),
            NRA::Sub(op) => op.dom().map(Sort::from),
            NRA::Mul(op) => op.dom().map(Sort::from),
            NRA::Div(op) => op.dom().map(Sort::from),
            NRA::Linear(op) => op.dom().map(Sort::from),
            NRA::Affine(op) => op.dom().map(Sort::from),
            NRA::ReLU(op) => op.dom().map(Sort::from),
            NRA::Abs(op) => op.dom().map(Sort::from),
            NRA::Neg(op) => op.dom().map(Sort::from),
            NRA::Zero(op) => op.dom().map(Sort::from),
            NRA::Argmax(op) => op.dom().map(Sort::from),
            NRA::Min(op) => op.dom().map(Sort::from),
            NRA::Max(op) => op.dom().map(Sort::from),
            NRA::Transpose(op) => op.dom().map(Sort::from),
            NRA::MatMul(op) => op.dom().map(Sort::from),
            NRA::Lt(op) => op.dom(),
            NRA::Ite(op) => op.dom(),
            NRA::BroadcastIte(op) => op.dom(),
            NRA::Le(op) => op.dom(),
            NRA::Gt(op) => op.dom(),
            NRA::Ge(op) => op.dom(),
            NRA::Eq(op) => op.dom(),
            NRA::Ne(op) => op.dom(),
        }
    }

    fn cod(&self) -> Poly<Sort> {
        match self {
            NRA::Num(op) => op.cod().map(Sort::from),
            NRA::Scale(op) => op.cod().map(Sort::from),
            NRA::Add(op) => op.cod().map(Sort::from),
            NRA::Sub(op) => op.cod().map(Sort::from),
            NRA::Mul(op) => op.cod().map(Sort::from),
            NRA::Div(op) => op.cod().map(Sort::from),
            NRA::Linear(op) => op.cod().map(Sort::from),
            NRA::Affine(op) => op.cod().map(Sort::from),
            NRA::ReLU(op) => op.cod().map(Sort::from),
            NRA::Abs(op) => op.cod().map(Sort::from),
            NRA::Neg(op) => op.cod().map(Sort::from),
            NRA::Zero(op) => op.cod().map(Sort::from),
            NRA::Argmax(op) => op.cod().map(Sort::from),
            NRA::Min(op) => op.cod().map(Sort::from),
            NRA::Max(op) => op.cod().map(Sort::from),
            NRA::Transpose(op) => op.cod().map(Sort::from),
            NRA::MatMul(op) => op.cod().map(Sort::from),
            NRA::Lt(op) => op.cod(),
            NRA::Ite(op) => op.cod(),
            NRA::BroadcastIte(op) => op.cod(),
            NRA::Le(op) => op.cod(),
            NRA::Gt(op) => op.cod(),
            NRA::Ge(op) => op.cod(),
            NRA::Eq(op) => op.cod(),
            NRA::Ne(op) => op.cod(),
        }
    }
}
impl<T: Matrix> Boundary<Sort> for LRA<T> {
    fn dom(&self) -> Poly<Sort> {
        match self {
            LRA::Num(op) => op.dom().map(Sort::from),
            LRA::Scale(op) => op.dom().map(Sort::from),
            LRA::Add(op) => op.dom().map(Sort::from),
            LRA::Sub(op) => op.dom().map(Sort::from),
            LRA::Linear(op) => op.dom().map(Sort::from),
            LRA::Affine(op) => op.dom().map(Sort::from),
            LRA::ReLU(op) => op.dom().map(Sort::from),
            LRA::Abs(op) => op.dom().map(Sort::from),
            LRA::Neg(op) => op.dom().map(Sort::from),
            LRA::Zero(op) => op.dom().map(Sort::from),
            LRA::Argmax(op) => op.dom().map(Sort::from),
            LRA::Min(op) => op.dom().map(Sort::from),
            LRA::Max(op) => op.dom().map(Sort::from),
            LRA::Transpose(op) => op.dom().map(Sort::from),
            LRA::Lt(op) => op.dom(),
            LRA::Ite(op) => op.dom(),
            LRA::BroadcastIte(op) => op.dom(),
            LRA::Le(op) => op.dom(),
            LRA::Gt(op) => op.dom(),
            LRA::Ge(op) => op.dom(),
            LRA::Eq(op) => op.dom(),
            LRA::Ne(op) => op.dom(),
        }
    }

    fn cod(&self) -> Poly<Sort> {
        match self {
            LRA::Num(op) => op.cod().map(Sort::from),
            LRA::Scale(op) => op.cod().map(Sort::from),
            LRA::Add(op) => op.cod().map(Sort::from),
            LRA::Sub(op) => op.cod().map(Sort::from),
            LRA::Linear(op) => op.cod().map(Sort::from),
            LRA::Affine(op) => op.cod().map(Sort::from),
            LRA::ReLU(op) => op.cod().map(Sort::from),
            LRA::Abs(op) => op.cod().map(Sort::from),
            LRA::Neg(op) => op.cod().map(Sort::from),
            LRA::Zero(op) => op.cod().map(Sort::from),
            LRA::Argmax(op) => op.cod().map(Sort::from),
            LRA::Min(op) => op.cod().map(Sort::from),
            LRA::Max(op) => op.cod().map(Sort::from),
            LRA::Transpose(op) => op.cod().map(Sort::from),
            LRA::Lt(op) => op.cod(),
            LRA::Ite(op) => op.cod(),
            LRA::BroadcastIte(op) => op.cod(),
            LRA::Le(op) => op.cod(),
            LRA::Gt(op) => op.cod(),
            LRA::Ge(op) => op.cod(),
            LRA::Eq(op) => op.cod(),
            LRA::Ne(op) => op.cod(),
        }
    }
}

impl<T: Matrix> Boundary<Real> for Num<T> {
    fn dom(&self) -> Poly<Real> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::with_degree(self.tensor.shape(), self.degree)].into()
    }
}

/// The product with `C` of shape `[m, n]` maps `[m, n]` matrices to `[m, n]`
/// matrices.
impl<T: Matrix> Boundary<Real> for Scale<T> {
    fn dom(&self) -> Poly<Real> {
        mono![Real::with_degree(self.tensor.shape(), self.degree)].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::with_degree(self.tensor.shape(), self.degree)].into()
    }
}

impl Boundary<Real> for Add {
    fn dom(&self) -> Poly<Real> {
        mono![Real::with_degree(self.shape, self.degree); 2].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::with_degree(self.shape, self.degree)].into()
    }
}

impl Boundary<Real> for Sub {
    fn dom(&self) -> Poly<Real> {
        mono![Real::with_degree(self.shape, self.degree); 2].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::with_degree(self.shape, self.degree)].into()
    }
}

/// Takes matrices of degrees `p` and `q` to one of degree `p + q`.
impl Boundary<Real> for Mul {
    fn dom(&self) -> Poly<Real> {
        let [p, q] = self.degrees;
        mono![
            Real::with_degree(self.shape, p),
            Real::with_degree(self.shape, q)
        ]
        .into()
    }

    fn cod(&self) -> Poly<Real> {
        // in range: `with_dom` checks it
        let [p, q] = self.degrees;
        mono![Real::with_degree(self.shape, p + q)].into()
    }
}

/// Takes matrices of degrees `p` and `q` to one of degree `p - q`.
impl Boundary<Real> for Div {
    fn dom(&self) -> Poly<Real> {
        let [p, q] = self.degrees;
        mono![
            Real::with_degree(self.shape, p),
            Real::with_degree(self.shape, q)
        ]
        .into()
    }

    fn cod(&self) -> Poly<Real> {
        // not negative: `with_dom` checks it
        let [p, q] = self.degrees;
        mono![Real::with_degree(self.shape, p - q)].into()
    }
}

/// With `A` of shape `[m, n]`, the map takes `x` of shape `[n, k]` to a
/// matrix of shape `[m, k]`.
impl<T: Matrix> Boundary<Real> for Linear<T> {
    fn dom(&self) -> Poly<Real> {
        let [_, n] = self.a.shape();
        mono![Real::with_degree([n, self.cols], self.degree)].into()
    }

    fn cod(&self) -> Poly<Real> {
        let [m, _] = self.a.shape();
        mono![Real::with_degree([m, self.cols], self.degree)].into()
    }
}

/// With `A` of shape `[m, n]` and `B` of shape `[m, k]`, the map takes `x`
/// of shape `[n, k]` to a matrix of shape `[m, k]`.
impl<T: Matrix> Boundary<Real> for Affine<T> {
    fn dom(&self) -> Poly<Real> {
        let [_, n] = self.a.shape();
        let [_, k] = self.b.shape();
        mono![Real::with_degree([n, k], self.degree)].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::with_degree(self.b.shape(), self.degree)].into()
    }
}

impl Boundary<Real> for Zero {
    fn dom(&self) -> Poly<Real> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::with_degree(self.shape, self.degree)].into()
    }
}

impl Boundary<Real> for Neg {
    fn dom(&self) -> Poly<Real> {
        mono![Real::with_degree(self.shape, self.degree)].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::with_degree(self.shape, self.degree)].into()
    }
}

impl Boundary<Real> for Abs {
    fn dom(&self) -> Poly<Real> {
        mono![Real::with_degree(self.shape, self.degree)].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::with_degree(self.shape, self.degree)].into()
    }
}

impl Boundary<Real> for ReLU {
    fn dom(&self) -> Poly<Real> {
        mono![Real::with_degree(self.shape, self.degree)].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::with_degree(self.shape, self.degree)].into()
    }
}

impl Boundary<Real> for Min {
    fn dom(&self) -> Poly<Real> {
        mono![Real::with_degree(self.shape, self.degree); 2].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::with_degree(self.shape, self.degree)].into()
    }
}

impl Boundary<Real> for Max {
    fn dom(&self) -> Poly<Real> {
        mono![Real::with_degree(self.shape, self.degree); 2].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::with_degree(self.shape, self.degree)].into()
    }
}

/// Takes a matrix of shape `[m, n]` to one of shape `[n, m]`.
impl Boundary<Real> for Transpose {
    fn dom(&self) -> Poly<Real> {
        mono![Real::with_degree(self.shape, self.degree)].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::with_degree(
            [self.shape[1], self.shape[0]],
            self.degree
        )]
        .into()
    }
}

/// Takes matrices of shapes `[m, n]` and `[n, k]`, and degrees `p` and `q`,
/// to one of shape `[m, k]` and degree `p + q`.
impl Boundary<Real> for MatMul {
    fn dom(&self) -> Poly<Real> {
        let [lhs, rhs] = self.shapes;
        let [p, q] = self.degrees;
        mono![Real::with_degree(lhs, p), Real::with_degree(rhs, q)].into()
    }

    fn cod(&self) -> Poly<Real> {
        let [[m, _], [_, k]] = self.shapes;
        // in range: `with_dom` checks it
        let [p, q] = self.degrees;
        mono![Real::with_degree([m, k], p + q)].into()
    }
}

impl Boundary<Sort> for Lt {
    fn dom(&self) -> Poly<Sort> {
        mono![Real::with_degree(self.shape, self.degree).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new_unchecked(self.shape).into()].into()
    }
}

impl Boundary<Sort> for Le {
    fn dom(&self) -> Poly<Sort> {
        mono![Real::with_degree(self.shape, self.degree).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new_unchecked(self.shape).into()].into()
    }
}

impl Boundary<Sort> for Gt {
    fn dom(&self) -> Poly<Sort> {
        mono![Real::with_degree(self.shape, self.degree).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new_unchecked(self.shape).into()].into()
    }
}

impl Boundary<Sort> for Ge {
    fn dom(&self) -> Poly<Sort> {
        mono![Real::with_degree(self.shape, self.degree).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new_unchecked(self.shape).into()].into()
    }
}

impl Boundary<Sort> for Eq {
    fn dom(&self) -> Poly<Sort> {
        mono![Real::with_degree(self.shape, self.degree).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new_unchecked(self.shape).into()].into()
    }
}

impl Boundary<Sort> for Ne {
    fn dom(&self) -> Poly<Sort> {
        mono![Real::with_degree(self.shape, self.degree).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
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

impl fmt::Display for Zero {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Zero")
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

impl Boundary<Real> for Argmax {
    fn dom(&self) -> Poly<Real> {
        mono![Real::with_degree(self.shape, self.degree)].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new_unchecked([1, 1])].into()
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

impl Boundary<Sort> for Ite {
    fn dom(&self) -> Poly<Sort> {
        mono![
            Bool::new_unchecked(self.shape).into(),
            Real::with_degree(self.shape, self.degree).into(),
            Real::with_degree(self.shape, self.degree).into()
        ]
        .into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Real::with_degree(self.shape, self.degree).into()].into()
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

/// Extracts an operation of real arithmetic from a Python object, trying
/// each operation in turn: those carrying no data through their own
/// classes, and [`Num`], [`Scale`], [`Linear`] and [`Affine`] through the extraction that the
/// tensor type `T` provides.
#[cfg(feature = "pyo3")]
impl<'a, 'py, T> pyo3::FromPyObject<'a, 'py> for NRA<T>
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
            return Ok(NRA::Add(op));
        }
        if let Ok(op) = ob.extract::<Sub>() {
            return Ok(NRA::Sub(op));
        }
        if let Ok(op) = ob.extract::<Mul>() {
            return Ok(NRA::Mul(op));
        }
        if let Ok(op) = ob.extract::<Div>() {
            return Ok(NRA::Div(op));
        }
        if let Ok(op) = ob.extract::<ReLU>() {
            return Ok(NRA::ReLU(op));
        }
        if let Ok(op) = ob.extract::<Abs>() {
            return Ok(NRA::Abs(op));
        }
        if let Ok(op) = ob.extract::<Neg>() {
            return Ok(NRA::Neg(op));
        }
        if let Ok(op) = ob.extract::<Zero>() {
            return Ok(NRA::Zero(op));
        }
        if let Ok(op) = ob.extract::<Argmax>() {
            return Ok(NRA::Argmax(op));
        }
        if let Ok(op) = ob.extract::<Min>() {
            return Ok(NRA::Min(op));
        }
        if let Ok(op) = ob.extract::<Max>() {
            return Ok(NRA::Max(op));
        }
        if let Ok(op) = ob.extract::<Transpose>() {
            return Ok(NRA::Transpose(op));
        }
        if let Ok(op) = ob.extract::<MatMul>() {
            return Ok(NRA::MatMul(op));
        }
        if let Ok(op) = ob.extract::<Lt>() {
            return Ok(NRA::Lt(op));
        }
        if let Ok(op) = ob.extract::<Ite>() {
            return Ok(NRA::Ite(op));
        }
        if let Ok(op) = ob.extract::<broadcast::Ite>() {
            return Ok(NRA::BroadcastIte(op));
        }
        if let Ok(op) = ob.extract::<Le>() {
            return Ok(NRA::Le(op));
        }
        if let Ok(op) = ob.extract::<Gt>() {
            return Ok(NRA::Gt(op));
        }
        if let Ok(op) = ob.extract::<Ge>() {
            return Ok(NRA::Ge(op));
        }
        if let Ok(op) = ob.extract::<Eq>() {
            return Ok(NRA::Eq(op));
        }
        if let Ok(op) = ob.extract::<Ne>() {
            return Ok(NRA::Ne(op));
        }
        if let Ok(op) = ob.extract::<Num<T>>() {
            return Ok(NRA::Num(op));
        }
        if let Ok(op) = ob.extract::<Scale<T>>() {
            return Ok(NRA::Scale(op));
        }
        if let Ok(op) = ob.extract::<Linear<T>>() {
            return Ok(NRA::Linear(op));
        }
        if let Ok(op) = ob.extract::<Affine<T>>() {
            return Ok(NRA::Affine(op));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "expected an operation of real arithmetic, got {}",
            ob.get_type()
        )))
    }
}

// The Python constructors of the operations carrying no tensor: each takes
// its sizes positionally, in the order of its fields, the shape as a list.
impl Add {
    /// The operation reading the real matrices of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Real; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Add {
            shape: s.shape(),
            degree: s.degree(),
        })
    }

    /// The operation writing the real matrix of `cod`.
    pub fn with_cod(cod: [Real; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Add {
            shape: s.shape(),
            degree: s.degree(),
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
    fn py_new(dom: [Real; 2], cod: Option<Real>) -> Result<Self, crate::Error> {
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
        crate::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn degree(&self) -> u8 {
        self.degree
    }
}

impl Sub {
    /// The operation reading the real matrices of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Real; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Sub {
            shape: s.shape(),
            degree: s.degree(),
        })
    }

    /// The operation writing the real matrix of `cod`.
    pub fn with_cod(cod: [Real; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Sub {
            shape: s.shape(),
            degree: s.degree(),
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
    fn py_new(dom: [Real; 2], cod: Option<Real>) -> Result<Self, crate::Error> {
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
        crate::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn degree(&self) -> u8 {
        self.degree
    }
}

impl Mul {
    /// The degrees `[p, q]` of the factors.
    pub fn degrees(&self) -> [u8; 2] {
        self.degrees
    }

    /// The operation reading the real matrices of `dom`; fails unless they are of the same shape.
    pub fn with_dom(dom: [Real; 2]) -> Result<Self, crate::Error> {
        let [x, y] = dom;
        if x.shape() != y.shape() {
            return Err(crate::unexpected_dom("2 sorts of the same shape", &dom));
        }
        let [p, q] = [x.degree(), y.degree()];
        if p.checked_add(q).is_none() {
            return Err(crate::Error(format!(
                "the product of reals of degrees {p} and {q} is of a degree out of range"
            )));
        }
        Ok(Mul {
            shape: x.shape(),
            degrees: [p, q],
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
    fn py_new(dom: [Real; 2], cod: Option<Real>) -> Result<Self, crate::Error> {
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
        crate::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    /// The degrees of the factors, as a list: pyo3 would make `[u8; 2]`
    /// into `bytes`.
    #[cfg(feature = "pyo3")]
    #[getter(degrees)]
    fn py_degrees(&self) -> [usize; 2] {
        self.degrees.map(usize::from)
    }
}

impl Div {
    /// The degrees `[p, q]` of the dividend and the divisor.
    pub fn degrees(&self) -> [u8; 2] {
        self.degrees
    }

    /// The operation reading the real matrices of `dom`; fails unless they are of the same shape.
    pub fn with_dom(dom: [Real; 2]) -> Result<Self, crate::Error> {
        let [x, y] = dom;
        if x.shape() != y.shape() {
            return Err(crate::unexpected_dom("2 sorts of the same shape", &dom));
        }
        let [p, q] = [x.degree(), y.degree()];
        if q > p {
            return Err(crate::Error(format!(
                "the quotient of reals of degrees {p} and {q} is of a negative degree"
            )));
        }
        Ok(Div {
            shape: x.shape(),
            degrees: [p, q],
        })
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
    fn py_new(dom: [Real; 2], cod: Option<Real>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
    }

    /// The degrees of the dividend and the divisor, as a list: pyo3 would
    /// make `[u8; 2]` into `bytes`.
    #[cfg(feature = "pyo3")]
    #[getter(degrees)]
    fn py_degrees(&self) -> [usize; 2] {
        self.degrees.map(usize::from)
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
        crate::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Zero {
    /// The operation writing the real matrix of `cod`.
    pub fn with_cod(cod: [Real; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Zero {
            shape: s.shape(),
            degree: s.degree(),
        })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Zero {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its codomain `cod`.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (cod))]
    fn py_new(cod: Real) -> Result<Self, crate::Error> {
        Self::with_cod([cod])
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("cod",)
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn degree(&self) -> u8 {
        self.degree
    }
}

impl Neg {
    /// The operation reading the real matrix of `dom`.
    pub fn with_dom(dom: [Real; 1]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Neg {
            shape: s.shape(),
            degree: s.degree(),
        })
    }

    /// The operation writing the real matrix of `cod`.
    pub fn with_cod(cod: [Real; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Neg {
            shape: s.shape(),
            degree: s.degree(),
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
    fn py_new(dom: Real, cod: Option<Real>) -> Result<Self, crate::Error> {
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
        crate::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn degree(&self) -> u8 {
        self.degree
    }
}

impl Abs {
    /// The operation reading the real matrix of `dom`.
    pub fn with_dom(dom: [Real; 1]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Abs {
            shape: s.shape(),
            degree: s.degree(),
        })
    }

    /// The operation writing the real matrix of `cod`.
    pub fn with_cod(cod: [Real; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Abs {
            shape: s.shape(),
            degree: s.degree(),
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
    fn py_new(dom: Real, cod: Option<Real>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom([dom]), cod)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn degree(&self) -> u8 {
        self.degree
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
        crate::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl ReLU {
    /// The operation reading the real matrix of `dom`.
    pub fn with_dom(dom: [Real; 1]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(ReLU {
            shape: s.shape(),
            degree: s.degree(),
        })
    }

    /// The operation writing the real matrix of `cod`.
    pub fn with_cod(cod: [Real; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(ReLU {
            shape: s.shape(),
            degree: s.degree(),
        })
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
    fn py_new(dom: Real, cod: Option<Real>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom([dom]), cod)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn degree(&self) -> u8 {
        self.degree
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
        crate::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Argmax {
    /// The operation reading the real matrix of `dom`.
    pub fn with_dom(dom: [Real; 1]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Argmax {
            shape: s.shape(),
            degree: s.degree(),
        })
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
    fn py_new(dom: Real, cod: Option<Real>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom([dom]), cod)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn degree(&self) -> u8 {
        self.degree
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
        crate::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Min {
    /// The operation reading the real matrices of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Real; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Min {
            shape: s.shape(),
            degree: s.degree(),
        })
    }

    /// The operation writing the real matrix of `cod`.
    pub fn with_cod(cod: [Real; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Min {
            shape: s.shape(),
            degree: s.degree(),
        })
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
    fn py_new(dom: [Real; 2], cod: Option<Real>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn degree(&self) -> u8 {
        self.degree
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
        crate::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Max {
    /// The operation reading the real matrices of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Real; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Max {
            shape: s.shape(),
            degree: s.degree(),
        })
    }

    /// The operation writing the real matrix of `cod`.
    pub fn with_cod(cod: [Real; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Max {
            shape: s.shape(),
            degree: s.degree(),
        })
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
    fn py_new(dom: [Real; 2], cod: Option<Real>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn degree(&self) -> u8 {
        self.degree
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
        crate::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Transpose {
    /// The operation reading the real matrix of `dom`.
    pub fn with_dom(dom: [Real; 1]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Transpose {
            shape: s.shape(),
            degree: s.degree(),
        })
    }

    /// The transposition writing the real matrix of `cod`, reading its transpose.
    pub fn with_cod(cod: [Real; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        let [m, n] = s.shape();
        Ok(Transpose {
            shape: [n, m],
            degree: s.degree(),
        })
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
    fn py_new(dom: Real, cod: Option<Real>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom([dom]), cod)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn degree(&self) -> u8 {
        self.degree
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
        crate::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl MatMul {
    /// The degrees `[p, q]` of the factors.
    pub fn degrees(&self) -> [u8; 2] {
        self.degrees
    }

    /// The product of the real matrices of `dom`; fails unless the columns of
    /// the first are the rows of the second.
    pub fn with_dom(dom: [Real; 2]) -> Result<Self, crate::Error> {
        let [x, y] = dom;
        let [lhs, rhs] = [x.shape(), y.shape()];
        if lhs[1] != rhs[0] {
            return Err(crate::Error(format!(
                "cannot multiply a {lhs:?} matrix by a {rhs:?} matrix"
            )));
        }
        let [p, q] = [x.degree(), y.degree()];
        if p.checked_add(q).is_none() {
            return Err(crate::Error(format!(
                "the product of reals of degrees {p} and {q} is of a degree out of range"
            )));
        }
        Ok(MatMul {
            shapes: [lhs, rhs],
            degrees: [p, q],
        })
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
    fn py_new(dom: [Real; 2], cod: Option<Real>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
    }

    /// The degrees of the factors, as a list: pyo3 would make `[u8; 2]`
    /// into `bytes`.
    #[cfg(feature = "pyo3")]
    #[getter(degrees)]
    fn py_degrees(&self) -> [usize; 2] {
        self.degrees.map(usize::from)
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
        crate::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_sorts(py, Boundary::cod(self))
    }

    /// The shapes of the factors, `[[m, n], [n, k]]`.
    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shapes(&self) -> [[usize; 2]; 2] {
        self.shapes
    }
}

impl Lt {
    /// The operation reading the real matrices of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Real; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Lt {
            shape: s.shape(),
            degree: s.degree(),
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
    fn py_new(dom: [Real; 2], cod: Option<Bool>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn degree(&self) -> u8 {
        self.degree
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
        crate::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_sorts(py, Boundary::cod(self))
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
            [Sort::Bool(c), Sort::Real(x), Sort::Real(y)] if x == y && c.shape() == x.shape() => {
                Ok(Ite {
                    shape: x.shape(),
                    degree: x.degree(),
                })
            }
            _ => Err(crate::unexpected_dom(
                "a condition Bool and two branches Real of its shape",
                &dom,
            )),
        }
    }

    /// The operation writing the real matrix of `cod`.
    pub fn with_cod(cod: [Real; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Ite {
            shape: s.shape(),
            degree: s.degree(),
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
    fn py_new(dom: [Sort; 3], cod: Option<Real>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn degree(&self) -> u8 {
        self.degree
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
        crate::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Le {
    /// The operation reading the real matrices of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Real; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Le {
            shape: s.shape(),
            degree: s.degree(),
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
    fn py_new(dom: [Real; 2], cod: Option<Bool>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn degree(&self) -> u8 {
        self.degree
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
        crate::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Gt {
    /// The operation reading the real matrices of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Real; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Gt {
            shape: s.shape(),
            degree: s.degree(),
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
    fn py_new(dom: [Real; 2], cod: Option<Bool>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn degree(&self) -> u8 {
        self.degree
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
        crate::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Ge {
    /// The operation reading the real matrices of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Real; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Ge {
            shape: s.shape(),
            degree: s.degree(),
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
    fn py_new(dom: [Real; 2], cod: Option<Bool>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn degree(&self) -> u8 {
        self.degree
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
        crate::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Eq {
    /// The operation reading the real matrices of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Real; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Eq {
            shape: s.shape(),
            degree: s.degree(),
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
    fn py_new(dom: [Real; 2], cod: Option<Bool>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn degree(&self) -> u8 {
        self.degree
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
        crate::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Ne {
    /// The operation reading the real matrices of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Real; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Ne {
            shape: s.shape(),
            degree: s.degree(),
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
    fn py_new(dom: [Real; 2], cod: Option<Bool>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn degree(&self) -> u8 {
        self.degree
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
        crate::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

/// The operations of the theory that broadcast a scalar operand, of the
/// shape `[1, 1]`, to the shape of the others; the others are in the parent
/// module.
pub mod broadcast {
    use super::*;

    /// Choice `if c then x else y` between whole `[m, n]` real matrices `x` and
    /// `y`, by the scalar boolean `c`, of the shape `[1, 1]`, read first: the
    /// pointwise [`super::Ite`] by `c` broadcast to `[m, n]`.
    ///
    /// SMT-LIB2: `(ite c x y)`, by the only entry of `c`.
    ///
    /// Of branches of the same degree, which the choice keeps.
    #[derive(Clone, Copy, PartialEq, Debug, Eq)]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.real.broadcast")
    )]
    pub struct Ite {
        shape: [usize; 2],
        degree: u8,
    }

    impl Boundary<Sort> for Ite {
        fn dom(&self) -> Poly<Sort> {
            mono![
                Bool::new_unchecked([1, 1]).into(),
                Real::with_degree(self.shape, self.degree).into(),
                Real::with_degree(self.shape, self.degree).into()
            ]
            .into()
        }

        fn cod(&self) -> Poly<Sort> {
            mono![Real::with_degree(self.shape, self.degree).into()].into()
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
        pub fn with_dom(dom: [Sort; 3]) -> Result<Self, crate::Error> {
            match dom {
                [Sort::Bool(c), Sort::Real(x), Sort::Real(y)] if x == y && c.shape() == [1, 1] => {
                    Ok(Ite {
                        shape: x.shape(),
                        degree: x.degree(),
                    })
                }
                _ => Err(crate::unexpected_dom(
                    "a condition Bool([1, 1]) and two branches Real of the same sort",
                    &dom,
                )),
            }
        }

        /// The choice writing the real matrix of `cod`.
        pub fn with_cod(cod: [Real; 1]) -> Result<Self, crate::Error> {
            let [s] = cod;
            Ok(Ite {
                shape: s.shape(),
                degree: s.degree(),
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
        fn py_new(dom: [Sort; 3], cod: Option<Real>) -> Result<Self, crate::Error> {
            crate::check_cod(Self::with_dom(dom), cod)
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn degree(&self) -> u8 {
            self.degree
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
            crate::py_sorts(py, Boundary::dom(self))
        }

        /// The sort it writes.
        #[cfg(feature = "pyo3")]
        #[getter(cod)]
        fn py_cod<'py>(
            &self,
            py: pyo3::Python<'py>,
        ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
            crate::py_sorts(py, Boundary::cod(self))
        }

        #[cfg_attr(feature = "pyo3", getter)]
        pub fn shape(&self) -> [usize; 2] {
            self.shape
        }
    }
}
