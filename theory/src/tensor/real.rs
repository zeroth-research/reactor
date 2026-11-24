/*!
# Real arithmetic

Defines one type per operation on the sort [`Real`] of real matrices. Every operation reads and writes real matrices, and carries enough
information to compute its [`Boundary`]:

- [`Const`] — a constant real matrix literal.
- [`Scale`] — the pointwise product `C ⊙ x` with a constant real matrix `C`.
- [`Add`], [`Sub`] — pointwise addition and subtraction.
- [`Mul`] — pointwise multiplication `x ⊙ y`.
- [`Div`] — pointwise division `x / y`.
- [`Linear`]`(A, B)` — the affine map `x ↦ A·x + B`, with `A` and `B`
  constant real matrices of compatible shapes.
- [`Neg`] — the pointwise negation `x ↦ -x`.
- [`Abs`] — the pointwise absolute value `x ↦ |x|`.
- [`ReLU`] — the shape-preserving rectified-linear map `x ↦ max(x, 0)`.
- [`Min`], [`Max`] — the least and the greatest entry, as a `[1, 1]`
  matrix.
- [`Transpose`] — matrix transposition.
- [`Get`] — one entry of a matrix.
- [`MatMul`] — the matrix product `x·y`.
- [`Argmax`] — the index of a maximal entry, counted row by row, as a
  `[1, 1]` matrix.
- [`Ite`] — the choice `if c then x else y` between whole matrices, by a
  `[1, 1]` boolean matrix `c`.
- [`Lt`], [`Le`], [`Gt`], [`Ge`], [`Eq`](struct@Eq), [`Ne`] — pointwise
  comparisons, which write boolean matrices of the same shape.
- [`Sqrt`], [`Exp`], [`Log`], [`Sin`], [`Cos`], [`Tan`], [`Asin`], [`Acos`],
  [`Atan`], [`Sinh`], [`Cosh`], [`Tanh`] — pointwise transcendental functions
  of plain reals, and [`Atan2`], the pointwise angle `atan2(y, x)`.

The operations are gathered in three nested enums, after the SMT-LIB2 logics
they span: [`LRA`], the linear operations; [`NRA`], which adds the products and
quotients of two variables (polynomial and rational arithmetic); and
[`NRAT`], which adds the transcendental functions, as in the logic `QF_NRAT`
of the solvers that support it (cvc5, MathSAT, dReal). Where a function is not
defined (`sqrt` and `log` of a negative, `log 0`, `asin` and `acos` outside
`[-1, 1]`, `tan` at its poles, `atan2(0, 0)`), its value is unspecified, as
that of `x / 0`.

The comparisons are those of SMT-LIB2's `Reals` theory, `<`, `<=`, `>` and
`>=`, together with `=` and `distinct` of the Core theory, lifted pointwise
to matrices. As they write booleans, their boundary is in [`Sort`], the
sorts of the theory, rather than in [`Real`].
*/

use super::sort::{Bool, Real};
use crate::Tensor;
use base::boundary::{Boundary, Poly};
use base::mono;
use derive_more::From;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use std::fmt;
use subenum::subenum;

/// The sorts of the theory, as in the signature of SMT-LIB2's `Reals`:
/// real matrices, and the booleans of the Core theory, which comparisons
/// write, together with [`Zero`](crate::tensor::sort::Zero). It is the subset
/// [`tensor::sort::Reals`](crate::tensor::sort::Reals) of the sorts of all theories.
pub use super::sort::Reals as Sort;

/// A constant real matrix literal, held in a tensor of type `T`, of a
/// degree: a constant rate of change, such as the `1` of `ẋ = 1`, is of
/// degree 1.
#[derive(Debug)]
pub struct Const<T> {
    tensor: T,
    degree: u8,
}

impl<T> Const<T> {
    /// The plain constant held in `tensor`, of degree 0.
    pub fn new(tensor: T) -> Self {
        Self::with_degree(tensor, 0)
    }

    /// The constant held in `tensor`, of the given `degree`.
    pub fn with_degree(tensor: T, degree: u8) -> Self {
        Const { tensor, degree }
    }

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

impl<T> Scale<T> {
    /// The product with the constant held in `tensor`, of plain real
    /// matrices, of degree 0.
    pub fn new(tensor: T) -> Self {
        Self::with_degree(tensor, 0)
    }

    /// The product with the constant held in `tensor`, of real matrices of
    /// the given `degree`.
    pub fn with_degree(tensor: T, degree: u8) -> Self {
        Scale { tensor, degree }
    }

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
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, from_py_object, eq, str))]
pub struct Add {
    shape: [usize; 2],
    degree: u8,
}

/// Pointwise subtraction `x - y` of `[m, n]` matrices, of the same degree,
/// which the difference keeps.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, from_py_object, eq, str))]
pub struct Sub {
    shape: [usize; 2],
    degree: u8,
}

/// Pointwise multiplication `x ⊙ y` of `[m, n]` matrices: the Hadamard
/// product, not the matrix product [`MatMul`]. The degrees of the factors
/// add up: the product of a rate of change by a plain real is a rate of
/// change, and that of two rates of change is of degree 2.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, from_py_object, eq, str))]
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
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, from_py_object, eq, str))]
pub struct Div {
    shape: [usize; 2],
    /// The degrees `[p, q]` of the dividend and the divisor.
    ///
    /// Entry `i` belongs to input `i`, as indexed in the domain of the
    /// boundary: an array rather than one field per input, so that the
    /// two match by position.
    degrees: [u8; 2],
}

/// The affine map `x ↦ A·x + B`, where `A` and `B` are constants: `A` a
/// plain real, and `B` of the degree of `x`, which the map keeps.
#[derive(Debug)]
pub struct Linear<T> {
    a: T,
    b: T,
    degree: u8,
}

impl<T: Tensor> Linear<T> {
    /// The affine map `x ↦ A·x + B` of plain real matrices, of degree 0;
    /// fails unless `a` and `b` are of shapes `[m, n]` and `[m, k]`.
    pub fn new(a: T, b: T) -> Result<Self, crate::Error> {
        Self::with_degree(a, b, 0)
    }

    /// The affine map `x ↦ A·x + B` of real matrices of the given `degree`;
    /// fails unless `a` and `b` are of shapes `[m, n]` and `[m, k]`.
    pub fn with_degree(a: T, b: T, degree: u8) -> Result<Self, crate::Error> {
        super::check_affine(a.shape(), b.shape())?;
        Ok(Linear { a, b, degree })
    }
}

impl<T> Linear<T> {
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

/// Pointwise negation `x ↦ -x` of `[m, n]` matrices, of a degree which
/// the negation keeps.
///
/// SMT-LIB2: `(- x)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, from_py_object, eq, str))]
pub struct Neg {
    shape: [usize; 2],
    degree: u8,
}

/// Pointwise absolute value `x ↦ |x|` of `[m, n]` matrices. SMT-LIB2's
/// `Reals` has no `abs`, unlike `Ints`; it is `(ite (>= x 0) x (- x))`.
///
/// Of a degree, which the absolute value keeps.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, from_py_object, eq, str))]
pub struct Abs {
    shape: [usize; 2],
    degree: u8,
}

/// The rectified-linear map `x ↦ max(x, 0)`, applied pointwise to
/// `[m, n]` matrices.
///
/// Of a degree, which the map keeps.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, from_py_object, eq, str))]
pub struct ReLU {
    shape: [usize; 2],
    degree: u8,
}

/// The index of a maximal entry of an `[m, n]` real matrix, counting its
/// entries row by row, written as a `[1, 1]` matrix of the same sort.
///
/// Of an operand of any degree: the index it writes is of degree 0.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, from_py_object, eq, str))]
pub struct Argmax {
    shape: [usize; 2],
    degree: u8,
}

/// The least entry `minᵢⱼ xᵢⱼ` of an `[m, n]` real matrix, written as a
/// `[1, 1]` matrix of the same sort.
///
/// Of a degree, which the minimum keeps.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, from_py_object, eq, str))]
pub struct Min {
    shape: [usize; 2],
    degree: u8,
}

/// The greatest entry `maxᵢⱼ xᵢⱼ` of an `[m, n]` real matrix, written as a
/// `[1, 1]` matrix of the same sort.
///
/// Of a degree, which the maximum keeps.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, from_py_object, eq, str))]
pub struct Max {
    shape: [usize; 2],
    degree: u8,
}

/// Transposition `xᵀ` of `[m, n]` matrices.
///
/// Of a degree, which the transposition keeps.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, from_py_object, eq, str))]
pub struct Transpose {
    shape: [usize; 2],
    degree: u8,
}

/// The entry in row `row` and column `col` of an `[m, n]` real matrix,
/// written as a `[1, 1]` matrix.
///
/// Of a degree, which the entry keeps.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, from_py_object, eq, str))]
pub struct Get {
    row: usize,
    col: usize,
    shape: [usize; 2],
    degree: u8,
}

/// The matrix product `x·y` of an `[m, n]` matrix `x` and an `[n, k]`
/// matrix `y`, with the shapes given as `MatMul([[m, n], [n, k]])`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, from_py_object, eq, str))]
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
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, from_py_object, eq, str))]
pub struct Lt {
    shape: [usize; 2],
    degree: u8,
}

/// The choice `if c then x else y` between `[m, n]` real matrices `x` and
/// `y`, by the `[1, 1]` boolean matrix `c`, read first: one of the two whole
/// matrices, not entry by entry.
///
/// SMT-LIB2: `(ite c x y)`.
///
/// Of branches of the same degree, which the choice keeps.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, from_py_object, eq, str))]
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
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, from_py_object, eq, str))]
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
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, from_py_object, eq, str))]
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
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, from_py_object, eq, str))]
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
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, from_py_object, eq, str))]
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
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, from_py_object, eq, str))]
pub struct Ne {
    shape: [usize; 2],
    degree: u8,
}

/// Defines a pointwise transcendental function of plain real matrices: an
/// operation on `[m, n]` matrices of degree 0, as the function of a rate of
/// change is not one, with its Python constructor from the shape, its
/// boundary and its name.
macro_rules! transcendental {
    ($(#[$attr:meta])* $name:ident) => {
        $(#[$attr])*
        #[derive(Clone, Copy, PartialEq, Debug, Eq)]
        #[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, from_py_object, eq, str))]
        pub struct $name {
            shape: [usize; 2],
        }

        #[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
        impl $name {
            /// On plain real matrices of the given `shape`.
            #[cfg_attr(feature = "pyo3", new)]
            #[cfg_attr(feature = "pyo3", pyo3(signature = (shape, /)))]
            pub fn new(shape: [usize; 2]) -> Self {
                $name { shape }
            }

            #[cfg(feature = "pyo3")]
            #[classattr]
            fn __match_args__() -> (&'static str,) {
                ("shape",)
            }

            #[cfg_attr(feature = "pyo3", getter)]
            pub fn shape(&self) -> [usize; 2] {
                self.shape
            }
        }

        impl Boundary<Real> for $name {
            fn dom(&self) -> Poly<Real> {
                mono![Real::new(self.shape)].into()
            }

            fn cod(&self) -> Poly<Real> {
                mono![Real::new(self.shape)].into()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, stringify!($name))
            }
        }
    };
}

transcendental! {
    /// The pointwise square root `x ↦ √x`, unspecified for `x < 0`.
    ///
    /// cvc5 and MathSAT: `(sqrt x)`.
    Sqrt
}

transcendental! {
    /// The pointwise exponential `x ↦ eˣ`.
    ///
    /// cvc5 and MathSAT: `(exp x)`.
    Exp
}

transcendental! {
    /// The pointwise natural logarithm `x ↦ ln x`, unspecified for `x ≤ 0`.
    ///
    /// MathSAT: `(log x)`.
    Log
}

transcendental! {
    /// The pointwise sine `x ↦ sin x`, of an angle in radians.
    ///
    /// cvc5 and MathSAT: `(sin x)`.
    Sin
}

transcendental! {
    /// The pointwise cosine `x ↦ cos x`, of an angle in radians.
    ///
    /// cvc5 and MathSAT: `(cos x)`.
    Cos
}

transcendental! {
    /// The pointwise tangent `x ↦ tan x`, unspecified at its poles.
    ///
    /// cvc5 and MathSAT: `(tan x)`.
    Tan
}

transcendental! {
    /// The pointwise arcsine `x ↦ asin x`, in `[-π/2, π/2]`, unspecified
    /// outside `[-1, 1]`.
    ///
    /// cvc5: `(arcsin x)`; MathSAT: `(asin x)`.
    Asin
}

transcendental! {
    /// The pointwise arccosine `x ↦ acos x`, in `[0, π]`, unspecified outside
    /// `[-1, 1]`.
    ///
    /// cvc5: `(arccos x)`; MathSAT: `(acos x)`.
    Acos
}

transcendental! {
    /// The pointwise arctangent `x ↦ atan x`, in `(-π/2, π/2)`.
    ///
    /// cvc5: `(arctan x)`; MathSAT: `(atan x)`.
    Atan
}

transcendental! {
    /// The pointwise hyperbolic sine `x ↦ sinh x`.
    Sinh
}

transcendental! {
    /// The pointwise hyperbolic cosine `x ↦ cosh x`.
    Cosh
}

transcendental! {
    /// The pointwise hyperbolic tangent `x ↦ tanh x`.
    Tanh
}

/// The pointwise angle `(y, x) ↦ atan2(y, x)` of the point `(x, y)`, in
/// `(-π, π]`, of plain real `[m, n]` matrices, read in the order `y`, `x`,
/// as in C; unspecified at `(0, 0)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, from_py_object, eq, str))]
pub struct Atan2 {
    shape: [usize; 2],
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Atan2 {
    /// On plain real matrices of the given `shape`.
    #[cfg_attr(feature = "pyo3", new)]
    #[cfg_attr(feature = "pyo3", pyo3(signature = (shape, /)))]
    pub fn new(shape: [usize; 2]) -> Self {
        Atan2 { shape }
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Boundary<Real> for Atan2 {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new(self.shape); 2].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new(self.shape)].into()
    }
}

impl fmt::Display for Atan2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Atan2")
    }
}

/// The operations of the theory of real arithmetic, SMT-LIB2's `Reals`,
/// extended with transcendental functions, with their boundary in [`Sort`].
///
/// `subenum` defines [`NRA`], the polynomial and rational ones: all but the
/// transcendental functions; and [`LRA`], the linear ones: those of `NRA`
/// but `Mul`, `Div` and `MatMul`, which multiply or divide two variables. It
/// also generates the casts between `NRAT` and each of them: `From` up, and
/// `TryFrom` down. Those between `NRA` and `LRA` are implemented by hand.
#[subenum(NRA, LRA)]
#[derive(Debug, From)]
pub enum NRAT<T> {
    #[subenum(NRA, LRA)]
    Const(Const<T>),
    #[subenum(NRA, LRA)]
    Scale(Scale<T>),
    #[subenum(NRA, LRA)]
    Add(Add),
    #[subenum(NRA, LRA)]
    Sub(Sub),
    #[subenum(NRA)]
    Mul(Mul),
    #[subenum(NRA)]
    Div(Div),
    #[subenum(NRA, LRA)]
    Linear(Linear<T>),
    #[subenum(NRA, LRA)]
    ReLU(ReLU),
    #[subenum(NRA, LRA)]
    Abs(Abs),
    #[subenum(NRA, LRA)]
    Neg(Neg),
    #[subenum(NRA, LRA)]
    Argmax(Argmax),
    #[subenum(NRA, LRA)]
    Min(Min),
    #[subenum(NRA, LRA)]
    Max(Max),
    #[subenum(NRA, LRA)]
    Transpose(Transpose),
    #[subenum(NRA, LRA)]
    Get(Get),
    #[subenum(NRA)]
    MatMul(MatMul),
    #[subenum(NRA, LRA)]
    Lt(Lt),
    #[subenum(NRA, LRA)]
    Ite(Ite),
    #[subenum(NRA, LRA)]
    Le(Le),
    #[subenum(NRA, LRA)]
    Gt(Gt),
    #[subenum(NRA, LRA)]
    Ge(Ge),
    #[subenum(NRA, LRA)]
    Eq(Eq),
    #[subenum(NRA, LRA)]
    Ne(Ne),
    Sqrt(Sqrt),
    Exp(Exp),
    Log(Log),
    Sin(Sin),
    Cos(Cos),
    Tan(Tan),
    Asin(Asin),
    Acos(Acos),
    Atan(Atan),
    Atan2(Atan2),
    Sinh(Sinh),
    Cosh(Cosh),
    Tanh(Tanh),
}

// The casts between the two subenums, which `subenum` does not generate: up
// from `LRA` to `NRA`, and down from `NRA` to `LRA`, both through `NRAT`.
impl<T> From<LRA<T>> for NRA<T> {
    fn from(op: LRA<T>) -> Self {
        NRA::try_from(NRAT::from(op))
            .unwrap_or_else(|_| unreachable!("every linear operation is in NRA"))
    }
}

impl<T> TryFrom<NRA<T>> for LRA<T> {
    type Error = LRAConvertError;

    fn try_from(op: NRA<T>) -> Result<Self, Self::Error> {
        LRA::try_from(NRAT::from(op))
    }
}

impl<T> Clone for NRAT<T>
where
    Const<T>: Clone,
    Scale<T>: Clone,
    Linear<T>: Clone,
{
    fn clone(&self) -> Self {
        match self {
            NRAT::Const(op) => NRAT::Const(op.clone()),
            NRAT::Scale(op) => NRAT::Scale(op.clone()),
            NRAT::Add(op) => NRAT::Add(*op),
            NRAT::Sub(op) => NRAT::Sub(*op),
            NRAT::Mul(op) => NRAT::Mul(*op),
            NRAT::Div(op) => NRAT::Div(*op),
            NRAT::Linear(op) => NRAT::Linear(op.clone()),
            NRAT::ReLU(op) => NRAT::ReLU(*op),
            NRAT::Abs(op) => NRAT::Abs(*op),
            NRAT::Neg(op) => NRAT::Neg(*op),
            NRAT::Argmax(op) => NRAT::Argmax(*op),
            NRAT::Min(op) => NRAT::Min(*op),
            NRAT::Max(op) => NRAT::Max(*op),
            NRAT::Transpose(op) => NRAT::Transpose(*op),
            NRAT::Get(op) => NRAT::Get(*op),
            NRAT::MatMul(op) => NRAT::MatMul(*op),
            NRAT::Lt(op) => NRAT::Lt(*op),
            NRAT::Ite(op) => NRAT::Ite(*op),
            NRAT::Le(op) => NRAT::Le(*op),
            NRAT::Gt(op) => NRAT::Gt(*op),
            NRAT::Ge(op) => NRAT::Ge(*op),
            NRAT::Eq(op) => NRAT::Eq(*op),
            NRAT::Ne(op) => NRAT::Ne(*op),
            NRAT::Sqrt(op) => NRAT::Sqrt(*op),
            NRAT::Exp(op) => NRAT::Exp(*op),
            NRAT::Log(op) => NRAT::Log(*op),
            NRAT::Sin(op) => NRAT::Sin(*op),
            NRAT::Cos(op) => NRAT::Cos(*op),
            NRAT::Tan(op) => NRAT::Tan(*op),
            NRAT::Asin(op) => NRAT::Asin(*op),
            NRAT::Acos(op) => NRAT::Acos(*op),
            NRAT::Atan(op) => NRAT::Atan(*op),
            NRAT::Atan2(op) => NRAT::Atan2(*op),
            NRAT::Sinh(op) => NRAT::Sinh(*op),
            NRAT::Cosh(op) => NRAT::Cosh(*op),
            NRAT::Tanh(op) => NRAT::Tanh(*op),
        }
    }
}

impl<T: fmt::Display> fmt::Display for NRAT<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NRAT::Const(op) => op.fmt(f),
            NRAT::Scale(op) => op.fmt(f),
            NRAT::Add(op) => op.fmt(f),
            NRAT::Sub(op) => op.fmt(f),
            NRAT::Mul(op) => op.fmt(f),
            NRAT::Div(op) => op.fmt(f),
            NRAT::Linear(op) => op.fmt(f),
            NRAT::ReLU(op) => op.fmt(f),
            NRAT::Abs(op) => op.fmt(f),
            NRAT::Neg(op) => op.fmt(f),
            NRAT::Argmax(op) => op.fmt(f),
            NRAT::Min(op) => op.fmt(f),
            NRAT::Max(op) => op.fmt(f),
            NRAT::Transpose(op) => op.fmt(f),
            NRAT::Get(op) => op.fmt(f),
            NRAT::MatMul(op) => op.fmt(f),
            NRAT::Lt(op) => op.fmt(f),
            NRAT::Ite(op) => op.fmt(f),
            NRAT::Le(op) => op.fmt(f),
            NRAT::Gt(op) => op.fmt(f),
            NRAT::Ge(op) => op.fmt(f),
            NRAT::Eq(op) => op.fmt(f),
            NRAT::Ne(op) => op.fmt(f),
            NRAT::Sqrt(op) => op.fmt(f),
            NRAT::Exp(op) => op.fmt(f),
            NRAT::Log(op) => op.fmt(f),
            NRAT::Sin(op) => op.fmt(f),
            NRAT::Cos(op) => op.fmt(f),
            NRAT::Tan(op) => op.fmt(f),
            NRAT::Asin(op) => op.fmt(f),
            NRAT::Acos(op) => op.fmt(f),
            NRAT::Atan(op) => op.fmt(f),
            NRAT::Atan2(op) => op.fmt(f),
            NRAT::Sinh(op) => op.fmt(f),
            NRAT::Cosh(op) => op.fmt(f),
            NRAT::Tanh(op) => op.fmt(f),
        }
    }
}

impl<T: Tensor> Boundary<Sort> for NRAT<T> {
    fn dom(&self) -> Poly<Sort> {
        match self {
            NRAT::Const(op) => op.dom().map(Sort::from),
            NRAT::Scale(op) => op.dom().map(Sort::from),
            NRAT::Add(op) => op.dom().map(Sort::from),
            NRAT::Sub(op) => op.dom().map(Sort::from),
            NRAT::Mul(op) => op.dom().map(Sort::from),
            NRAT::Div(op) => op.dom().map(Sort::from),
            NRAT::Linear(op) => op.dom().map(Sort::from),
            NRAT::ReLU(op) => op.dom().map(Sort::from),
            NRAT::Abs(op) => op.dom().map(Sort::from),
            NRAT::Neg(op) => op.dom().map(Sort::from),
            NRAT::Argmax(op) => op.dom().map(Sort::from),
            NRAT::Min(op) => op.dom().map(Sort::from),
            NRAT::Max(op) => op.dom().map(Sort::from),
            NRAT::Transpose(op) => op.dom().map(Sort::from),
            NRAT::Get(op) => op.dom().map(Sort::from),
            NRAT::MatMul(op) => op.dom().map(Sort::from),
            NRAT::Lt(op) => op.dom(),
            NRAT::Ite(op) => op.dom(),
            NRAT::Le(op) => op.dom(),
            NRAT::Gt(op) => op.dom(),
            NRAT::Ge(op) => op.dom(),
            NRAT::Eq(op) => op.dom(),
            NRAT::Ne(op) => op.dom(),
            NRAT::Sqrt(op) => op.dom().map(Sort::from),
            NRAT::Exp(op) => op.dom().map(Sort::from),
            NRAT::Log(op) => op.dom().map(Sort::from),
            NRAT::Sin(op) => op.dom().map(Sort::from),
            NRAT::Cos(op) => op.dom().map(Sort::from),
            NRAT::Tan(op) => op.dom().map(Sort::from),
            NRAT::Asin(op) => op.dom().map(Sort::from),
            NRAT::Acos(op) => op.dom().map(Sort::from),
            NRAT::Atan(op) => op.dom().map(Sort::from),
            NRAT::Atan2(op) => op.dom().map(Sort::from),
            NRAT::Sinh(op) => op.dom().map(Sort::from),
            NRAT::Cosh(op) => op.dom().map(Sort::from),
            NRAT::Tanh(op) => op.dom().map(Sort::from),
        }
    }

    fn cod(&self) -> Poly<Sort> {
        match self {
            NRAT::Const(op) => op.cod().map(Sort::from),
            NRAT::Scale(op) => op.cod().map(Sort::from),
            NRAT::Add(op) => op.cod().map(Sort::from),
            NRAT::Sub(op) => op.cod().map(Sort::from),
            NRAT::Mul(op) => op.cod().map(Sort::from),
            NRAT::Div(op) => op.cod().map(Sort::from),
            NRAT::Linear(op) => op.cod().map(Sort::from),
            NRAT::ReLU(op) => op.cod().map(Sort::from),
            NRAT::Abs(op) => op.cod().map(Sort::from),
            NRAT::Neg(op) => op.cod().map(Sort::from),
            NRAT::Argmax(op) => op.cod().map(Sort::from),
            NRAT::Min(op) => op.cod().map(Sort::from),
            NRAT::Max(op) => op.cod().map(Sort::from),
            NRAT::Transpose(op) => op.cod().map(Sort::from),
            NRAT::Get(op) => op.cod().map(Sort::from),
            NRAT::MatMul(op) => op.cod().map(Sort::from),
            NRAT::Lt(op) => op.cod(),
            NRAT::Ite(op) => op.cod(),
            NRAT::Le(op) => op.cod(),
            NRAT::Gt(op) => op.cod(),
            NRAT::Ge(op) => op.cod(),
            NRAT::Eq(op) => op.cod(),
            NRAT::Ne(op) => op.cod(),
            NRAT::Sqrt(op) => op.cod().map(Sort::from),
            NRAT::Exp(op) => op.cod().map(Sort::from),
            NRAT::Log(op) => op.cod().map(Sort::from),
            NRAT::Sin(op) => op.cod().map(Sort::from),
            NRAT::Cos(op) => op.cod().map(Sort::from),
            NRAT::Tan(op) => op.cod().map(Sort::from),
            NRAT::Asin(op) => op.cod().map(Sort::from),
            NRAT::Acos(op) => op.cod().map(Sort::from),
            NRAT::Atan(op) => op.cod().map(Sort::from),
            NRAT::Atan2(op) => op.cod().map(Sort::from),
            NRAT::Sinh(op) => op.cod().map(Sort::from),
            NRAT::Cosh(op) => op.cod().map(Sort::from),
            NRAT::Tanh(op) => op.cod().map(Sort::from),
        }
    }
}

impl<T> Clone for NRA<T>
where
    Const<T>: Clone,
    Scale<T>: Clone,
    Linear<T>: Clone,
{
    fn clone(&self) -> Self {
        match self {
            NRA::Const(op) => NRA::Const(op.clone()),
            NRA::Scale(op) => NRA::Scale(op.clone()),
            NRA::Add(op) => NRA::Add(*op),
            NRA::Sub(op) => NRA::Sub(*op),
            NRA::Mul(op) => NRA::Mul(*op),
            NRA::Div(op) => NRA::Div(*op),
            NRA::Linear(op) => NRA::Linear(op.clone()),
            NRA::ReLU(op) => NRA::ReLU(*op),
            NRA::Abs(op) => NRA::Abs(*op),
            NRA::Neg(op) => NRA::Neg(*op),
            NRA::Argmax(op) => NRA::Argmax(*op),
            NRA::Min(op) => NRA::Min(*op),
            NRA::Max(op) => NRA::Max(*op),
            NRA::Transpose(op) => NRA::Transpose(*op),
            NRA::Get(op) => NRA::Get(*op),
            NRA::MatMul(op) => NRA::MatMul(*op),
            NRA::Lt(op) => NRA::Lt(*op),
            NRA::Ite(op) => NRA::Ite(*op),
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
            NRA::Const(op) => op.fmt(f),
            NRA::Scale(op) => op.fmt(f),
            NRA::Add(op) => op.fmt(f),
            NRA::Sub(op) => op.fmt(f),
            NRA::Mul(op) => op.fmt(f),
            NRA::Div(op) => op.fmt(f),
            NRA::Linear(op) => op.fmt(f),
            NRA::ReLU(op) => op.fmt(f),
            NRA::Abs(op) => op.fmt(f),
            NRA::Neg(op) => op.fmt(f),
            NRA::Argmax(op) => op.fmt(f),
            NRA::Min(op) => op.fmt(f),
            NRA::Max(op) => op.fmt(f),
            NRA::Transpose(op) => op.fmt(f),
            NRA::Get(op) => op.fmt(f),
            NRA::MatMul(op) => op.fmt(f),
            NRA::Lt(op) => op.fmt(f),
            NRA::Ite(op) => op.fmt(f),
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
    Const<T>: Clone,
    Scale<T>: Clone,
    Linear<T>: Clone,
{
    fn clone(&self) -> Self {
        match self {
            LRA::Const(op) => LRA::Const(op.clone()),
            LRA::Scale(op) => LRA::Scale(op.clone()),
            LRA::Add(op) => LRA::Add(*op),
            LRA::Sub(op) => LRA::Sub(*op),
            LRA::Linear(op) => LRA::Linear(op.clone()),
            LRA::ReLU(op) => LRA::ReLU(*op),
            LRA::Abs(op) => LRA::Abs(*op),
            LRA::Neg(op) => LRA::Neg(*op),
            LRA::Argmax(op) => LRA::Argmax(*op),
            LRA::Min(op) => LRA::Min(*op),
            LRA::Max(op) => LRA::Max(*op),
            LRA::Transpose(op) => LRA::Transpose(*op),
            LRA::Get(op) => LRA::Get(*op),
            LRA::Lt(op) => LRA::Lt(*op),
            LRA::Ite(op) => LRA::Ite(*op),
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
            LRA::Const(op) => op.fmt(f),
            LRA::Scale(op) => op.fmt(f),
            LRA::Add(op) => op.fmt(f),
            LRA::Sub(op) => op.fmt(f),
            LRA::Linear(op) => op.fmt(f),
            LRA::ReLU(op) => op.fmt(f),
            LRA::Abs(op) => op.fmt(f),
            LRA::Neg(op) => op.fmt(f),
            LRA::Argmax(op) => op.fmt(f),
            LRA::Min(op) => op.fmt(f),
            LRA::Max(op) => op.fmt(f),
            LRA::Transpose(op) => op.fmt(f),
            LRA::Get(op) => op.fmt(f),
            LRA::Lt(op) => op.fmt(f),
            LRA::Ite(op) => op.fmt(f),
            LRA::Le(op) => op.fmt(f),
            LRA::Gt(op) => op.fmt(f),
            LRA::Ge(op) => op.fmt(f),
            LRA::Eq(op) => op.fmt(f),
            LRA::Ne(op) => op.fmt(f),
        }
    }
}

impl<T: Tensor> Boundary<Sort> for NRA<T> {
    fn dom(&self) -> Poly<Sort> {
        match self {
            NRA::Const(op) => op.dom().map(Sort::from),
            NRA::Scale(op) => op.dom().map(Sort::from),
            NRA::Add(op) => op.dom().map(Sort::from),
            NRA::Sub(op) => op.dom().map(Sort::from),
            NRA::Mul(op) => op.dom().map(Sort::from),
            NRA::Div(op) => op.dom().map(Sort::from),
            NRA::Linear(op) => op.dom().map(Sort::from),
            NRA::ReLU(op) => op.dom().map(Sort::from),
            NRA::Abs(op) => op.dom().map(Sort::from),
            NRA::Neg(op) => op.dom().map(Sort::from),
            NRA::Argmax(op) => op.dom().map(Sort::from),
            NRA::Min(op) => op.dom().map(Sort::from),
            NRA::Max(op) => op.dom().map(Sort::from),
            NRA::Transpose(op) => op.dom().map(Sort::from),
            NRA::Get(op) => op.dom().map(Sort::from),
            NRA::MatMul(op) => op.dom().map(Sort::from),
            NRA::Lt(op) => op.dom(),
            NRA::Ite(op) => op.dom(),
            NRA::Le(op) => op.dom(),
            NRA::Gt(op) => op.dom(),
            NRA::Ge(op) => op.dom(),
            NRA::Eq(op) => op.dom(),
            NRA::Ne(op) => op.dom(),
        }
    }

    fn cod(&self) -> Poly<Sort> {
        match self {
            NRA::Const(op) => op.cod().map(Sort::from),
            NRA::Scale(op) => op.cod().map(Sort::from),
            NRA::Add(op) => op.cod().map(Sort::from),
            NRA::Sub(op) => op.cod().map(Sort::from),
            NRA::Mul(op) => op.cod().map(Sort::from),
            NRA::Div(op) => op.cod().map(Sort::from),
            NRA::Linear(op) => op.cod().map(Sort::from),
            NRA::ReLU(op) => op.cod().map(Sort::from),
            NRA::Abs(op) => op.cod().map(Sort::from),
            NRA::Neg(op) => op.cod().map(Sort::from),
            NRA::Argmax(op) => op.cod().map(Sort::from),
            NRA::Min(op) => op.cod().map(Sort::from),
            NRA::Max(op) => op.cod().map(Sort::from),
            NRA::Transpose(op) => op.cod().map(Sort::from),
            NRA::Get(op) => op.cod().map(Sort::from),
            NRA::MatMul(op) => op.cod().map(Sort::from),
            NRA::Lt(op) => op.cod(),
            NRA::Ite(op) => op.cod(),
            NRA::Le(op) => op.cod(),
            NRA::Gt(op) => op.cod(),
            NRA::Ge(op) => op.cod(),
            NRA::Eq(op) => op.cod(),
            NRA::Ne(op) => op.cod(),
        }
    }
}
impl<T: Tensor> Boundary<Sort> for LRA<T> {
    fn dom(&self) -> Poly<Sort> {
        match self {
            LRA::Const(op) => op.dom().map(Sort::from),
            LRA::Scale(op) => op.dom().map(Sort::from),
            LRA::Add(op) => op.dom().map(Sort::from),
            LRA::Sub(op) => op.dom().map(Sort::from),
            LRA::Linear(op) => op.dom().map(Sort::from),
            LRA::ReLU(op) => op.dom().map(Sort::from),
            LRA::Abs(op) => op.dom().map(Sort::from),
            LRA::Neg(op) => op.dom().map(Sort::from),
            LRA::Argmax(op) => op.dom().map(Sort::from),
            LRA::Min(op) => op.dom().map(Sort::from),
            LRA::Max(op) => op.dom().map(Sort::from),
            LRA::Transpose(op) => op.dom().map(Sort::from),
            LRA::Get(op) => op.dom().map(Sort::from),
            LRA::Lt(op) => op.dom(),
            LRA::Ite(op) => op.dom(),
            LRA::Le(op) => op.dom(),
            LRA::Gt(op) => op.dom(),
            LRA::Ge(op) => op.dom(),
            LRA::Eq(op) => op.dom(),
            LRA::Ne(op) => op.dom(),
        }
    }

    fn cod(&self) -> Poly<Sort> {
        match self {
            LRA::Const(op) => op.cod().map(Sort::from),
            LRA::Scale(op) => op.cod().map(Sort::from),
            LRA::Add(op) => op.cod().map(Sort::from),
            LRA::Sub(op) => op.cod().map(Sort::from),
            LRA::Linear(op) => op.cod().map(Sort::from),
            LRA::ReLU(op) => op.cod().map(Sort::from),
            LRA::Abs(op) => op.cod().map(Sort::from),
            LRA::Neg(op) => op.cod().map(Sort::from),
            LRA::Argmax(op) => op.cod().map(Sort::from),
            LRA::Min(op) => op.cod().map(Sort::from),
            LRA::Max(op) => op.cod().map(Sort::from),
            LRA::Transpose(op) => op.cod().map(Sort::from),
            LRA::Get(op) => op.cod().map(Sort::from),
            LRA::Lt(op) => op.cod(),
            LRA::Ite(op) => op.cod(),
            LRA::Le(op) => op.cod(),
            LRA::Gt(op) => op.cod(),
            LRA::Ge(op) => op.cod(),
            LRA::Eq(op) => op.cod(),
            LRA::Ne(op) => op.cod(),
        }
    }
}

impl<T: Tensor> Boundary<Real> for Const<T> {
    fn dom(&self) -> Poly<Real> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::with_degree(self.tensor.shape(), self.degree)].into()
    }
}

/// The product with `C` of shape `[m, n]` maps `[m, n]` matrices to `[m, n]`
/// matrices.
impl<T: Tensor> Boundary<Real> for Scale<T> {
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
        // in range: `with_degrees` checks it
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
        // not negative: `with_degrees` checks it
        let [p, q] = self.degrees;
        mono![Real::with_degree(self.shape, p - q)].into()
    }
}

/// With `A` of shape `[m, n]` and `B` of shape `[m, k]`, the map takes `x`
/// of shape `[n, k]` to a matrix of shape `[m, k]`.
impl<T: Tensor> Boundary<Real> for Linear<T> {
    fn dom(&self) -> Poly<Real> {
        let [_, n] = self.a.shape();
        let [_, k] = self.b.shape();
        mono![Real::with_degree([n, k], self.degree)].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::with_degree(self.b.shape(), self.degree)].into()
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

/// Takes a matrix of shape `[m, n]` to one of shape `[1, 1]`.
impl Boundary<Real> for Min {
    fn dom(&self) -> Poly<Real> {
        mono![Real::with_degree(self.shape, self.degree)].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::with_degree([1, 1], self.degree)].into()
    }
}

/// Takes a matrix of shape `[m, n]` to one of shape `[1, 1]`.
impl Boundary<Real> for Max {
    fn dom(&self) -> Poly<Real> {
        mono![Real::with_degree(self.shape, self.degree)].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::with_degree([1, 1], self.degree)].into()
    }
}

/// Takes a matrix of shape `[m, n]` to one of shape `[n, m]`.
impl Boundary<Real> for Get {
    fn dom(&self) -> Poly<Real> {
        mono![Real::with_degree(self.shape, self.degree)].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::with_degree([1, 1], self.degree)].into()
    }
}

impl fmt::Display for Get {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Get[{}][{}]", self.row, self.col)
    }
}

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
        // in range: `with_degrees` checks it
        let [p, q] = self.degrees;
        mono![Real::with_degree([m, k], p + q)].into()
    }
}

impl Boundary<Sort> for Lt {
    fn dom(&self) -> Poly<Sort> {
        mono![Real::with_degree(self.shape, self.degree).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new(self.shape).into()].into()
    }
}

impl Boundary<Sort> for Le {
    fn dom(&self) -> Poly<Sort> {
        mono![Real::with_degree(self.shape, self.degree).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new(self.shape).into()].into()
    }
}

impl Boundary<Sort> for Gt {
    fn dom(&self) -> Poly<Sort> {
        mono![Real::with_degree(self.shape, self.degree).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new(self.shape).into()].into()
    }
}

impl Boundary<Sort> for Ge {
    fn dom(&self) -> Poly<Sort> {
        mono![Real::with_degree(self.shape, self.degree).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new(self.shape).into()].into()
    }
}

impl Boundary<Sort> for Eq {
    fn dom(&self) -> Poly<Sort> {
        mono![Real::with_degree(self.shape, self.degree).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new(self.shape).into()].into()
    }
}

impl Boundary<Sort> for Ne {
    fn dom(&self) -> Poly<Sort> {
        mono![Real::with_degree(self.shape, self.degree).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new(self.shape).into()].into()
    }
}

impl<T: fmt::Display> fmt::Display for Scale<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Scale({})", self.tensor)
    }
}

impl<T: fmt::Display> fmt::Display for Const<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.tensor)
    }
}

impl<T: fmt::Display> fmt::Display for Linear<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Linear({}, {})", self.a, self.b)
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
        mono![Real::new([1, 1])].into()
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
            Bool::new([1, 1]).into(),
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
/// classes, and [`Const`], [`Scale`] and [`Linear`] through the extraction that the
/// tensor type `T` provides.
#[cfg(feature = "pyo3")]
impl<'a, 'py, T> pyo3::FromPyObject<'a, 'py> for NRAT<T>
where
    Const<T>: pyo3::FromPyObject<'a, 'py>,
    Scale<T>: pyo3::FromPyObject<'a, 'py>,
    Linear<T>: pyo3::FromPyObject<'a, 'py>,
{
    type Error = pyo3::PyErr;

    fn extract(ob: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use pyo3::types::PyAnyMethods;

        if let Ok(op) = ob.extract::<Add>() {
            return Ok(NRAT::Add(op));
        }
        if let Ok(op) = ob.extract::<Sub>() {
            return Ok(NRAT::Sub(op));
        }
        if let Ok(op) = ob.extract::<Mul>() {
            return Ok(NRAT::Mul(op));
        }
        if let Ok(op) = ob.extract::<Div>() {
            return Ok(NRAT::Div(op));
        }
        if let Ok(op) = ob.extract::<ReLU>() {
            return Ok(NRAT::ReLU(op));
        }
        if let Ok(op) = ob.extract::<Abs>() {
            return Ok(NRAT::Abs(op));
        }
        if let Ok(op) = ob.extract::<Neg>() {
            return Ok(NRAT::Neg(op));
        }
        if let Ok(op) = ob.extract::<Argmax>() {
            return Ok(NRAT::Argmax(op));
        }
        if let Ok(op) = ob.extract::<Min>() {
            return Ok(NRAT::Min(op));
        }
        if let Ok(op) = ob.extract::<Max>() {
            return Ok(NRAT::Max(op));
        }
        if let Ok(op) = ob.extract::<Transpose>() {
            return Ok(NRAT::Transpose(op));
        }
        if let Ok(op) = ob.extract::<Get>() {
            return Ok(NRAT::Get(op));
        }
        if let Ok(op) = ob.extract::<MatMul>() {
            return Ok(NRAT::MatMul(op));
        }
        if let Ok(op) = ob.extract::<Lt>() {
            return Ok(NRAT::Lt(op));
        }
        if let Ok(op) = ob.extract::<Ite>() {
            return Ok(NRAT::Ite(op));
        }
        if let Ok(op) = ob.extract::<Le>() {
            return Ok(NRAT::Le(op));
        }
        if let Ok(op) = ob.extract::<Gt>() {
            return Ok(NRAT::Gt(op));
        }
        if let Ok(op) = ob.extract::<Ge>() {
            return Ok(NRAT::Ge(op));
        }
        if let Ok(op) = ob.extract::<Eq>() {
            return Ok(NRAT::Eq(op));
        }
        if let Ok(op) = ob.extract::<Ne>() {
            return Ok(NRAT::Ne(op));
        }
        if let Ok(op) = ob.extract::<Sqrt>() {
            return Ok(NRAT::Sqrt(op));
        }
        if let Ok(op) = ob.extract::<Exp>() {
            return Ok(NRAT::Exp(op));
        }
        if let Ok(op) = ob.extract::<Log>() {
            return Ok(NRAT::Log(op));
        }
        if let Ok(op) = ob.extract::<Sin>() {
            return Ok(NRAT::Sin(op));
        }
        if let Ok(op) = ob.extract::<Cos>() {
            return Ok(NRAT::Cos(op));
        }
        if let Ok(op) = ob.extract::<Tan>() {
            return Ok(NRAT::Tan(op));
        }
        if let Ok(op) = ob.extract::<Asin>() {
            return Ok(NRAT::Asin(op));
        }
        if let Ok(op) = ob.extract::<Acos>() {
            return Ok(NRAT::Acos(op));
        }
        if let Ok(op) = ob.extract::<Atan>() {
            return Ok(NRAT::Atan(op));
        }
        if let Ok(op) = ob.extract::<Atan2>() {
            return Ok(NRAT::Atan2(op));
        }
        if let Ok(op) = ob.extract::<Sinh>() {
            return Ok(NRAT::Sinh(op));
        }
        if let Ok(op) = ob.extract::<Cosh>() {
            return Ok(NRAT::Cosh(op));
        }
        if let Ok(op) = ob.extract::<Tanh>() {
            return Ok(NRAT::Tanh(op));
        }
        if let Ok(op) = ob.extract::<Const<T>>() {
            return Ok(NRAT::Const(op));
        }
        if let Ok(op) = ob.extract::<Scale<T>>() {
            return Ok(NRAT::Scale(op));
        }
        if let Ok(op) = ob.extract::<Linear<T>>() {
            return Ok(NRAT::Linear(op));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "expected an operation of real arithmetic, got {}",
            ob.get_type()
        )))
    }
}

/// Extracts a polynomial or rational operation of real arithmetic from a
/// Python object: an operation of [`NRAT`] that is not transcendental.
#[cfg(feature = "pyo3")]
impl<'a, 'py, T> pyo3::FromPyObject<'a, 'py> for NRA<T>
where
    Const<T>: pyo3::FromPyObject<'a, 'py>,
    Scale<T>: pyo3::FromPyObject<'a, 'py>,
    Linear<T>: pyo3::FromPyObject<'a, 'py>,
{
    type Error = pyo3::PyErr;

    fn extract(ob: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        let op = ob.extract::<NRAT<T>>()?;
        NRA::try_from(op).map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "expected a polynomial or rational operation of real arithmetic, \
                 got a transcendental one",
            )
        })
    }
}

// The Python constructors of the operations carrying no tensor: each takes
// its sizes positionally, in the order of its fields, the shape as a list.
impl Add {
    /// On plain real matrices, of degree 0.
    pub fn new(shape: [usize; 2]) -> Self {
        Add { shape, degree: 0 }
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Add {
    /// On real matrices of the given `degree`.
    #[cfg_attr(feature = "pyo3", new)]
    #[cfg_attr(feature = "pyo3", pyo3(signature = (shape, /, degree = 0)))]
    pub fn with_degree(shape: [usize; 2], degree: u8) -> Self {
        Add { shape, degree }
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
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
    /// On plain real matrices, of degree 0.
    pub fn new(shape: [usize; 2]) -> Self {
        Sub { shape, degree: 0 }
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Sub {
    /// On real matrices of the given `degree`.
    #[cfg_attr(feature = "pyo3", new)]
    #[cfg_attr(feature = "pyo3", pyo3(signature = (shape, /, degree = 0)))]
    pub fn with_degree(shape: [usize; 2], degree: u8) -> Self {
        Sub { shape, degree }
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
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
    /// On plain real matrices, of degree 0.
    pub fn new(shape: [usize; 2]) -> Self {
        Mul {
            shape,
            degrees: [0, 0],
        }
    }

    /// The degrees `[p, q]` of the factors.
    pub fn degrees(&self) -> [u8; 2] {
        self.degrees
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Mul {
    /// On real matrices of the given `degrees`, `[p, q]`, writing ones of
    /// degree `p + q`; fails when that degree is out of range.
    #[cfg_attr(feature = "pyo3", new)]
    #[cfg_attr(feature = "pyo3", pyo3(signature = (shape, /, degrees = [0, 0])))]
    pub fn with_degrees(shape: [usize; 2], degrees: [u8; 2]) -> Result<Self, crate::Error> {
        let [p, q] = degrees;
        if p.checked_add(q).is_none() {
            return Err(crate::Error(format!(
                "the product of reals of degrees {p} and {q} is of a degree out of range"
            )));
        }
        Ok(Mul { shape, degrees })
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
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
    /// On plain real matrices, of degree 0.
    pub fn new(shape: [usize; 2]) -> Self {
        Div {
            shape,
            degrees: [0, 0],
        }
    }

    /// The degrees `[p, q]` of the dividend and the divisor.
    pub fn degrees(&self) -> [u8; 2] {
        self.degrees
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Div {
    /// On real matrices of the given `degrees`, `[p, q]`, writing ones of
    /// degree `p - q`; fails when `q` exceeds `p`, as that degree would be
    /// negative.
    #[cfg_attr(feature = "pyo3", new)]
    #[cfg_attr(feature = "pyo3", pyo3(signature = (shape, /, degrees = [0, 0])))]
    pub fn with_degrees(shape: [usize; 2], degrees: [u8; 2]) -> Result<Self, crate::Error> {
        let [p, q] = degrees;
        if q > p {
            return Err(crate::Error(format!(
                "the quotient of reals of degrees {p} and {q} is of a negative degree"
            )));
        }
        Ok(Div { shape, degrees })
    }

    /// The degrees of the dividend and the divisor, as a list: pyo3 would
    /// make `[u8; 2]` into `bytes`.
    #[cfg(feature = "pyo3")]
    #[getter(degrees)]
    fn py_degrees(&self) -> [usize; 2] {
        self.degrees.map(usize::from)
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Neg {
    /// On plain real matrices, of degree 0.
    pub fn new(shape: [usize; 2]) -> Self {
        Neg { shape, degree: 0 }
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Neg {
    /// On real matrices of the given `degree`.
    #[cfg_attr(feature = "pyo3", new)]
    #[cfg_attr(feature = "pyo3", pyo3(signature = (shape, /, degree = 0)))]
    pub fn with_degree(shape: [usize; 2], degree: u8) -> Self {
        Neg { shape, degree }
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
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
    /// On plain real matrices, of degree 0.
    pub fn new(shape: [usize; 2]) -> Self {
        Abs { shape, degree: 0 }
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Abs {
    /// On real matrices of the given `degree`.
    #[cfg_attr(feature = "pyo3", new)]
    #[cfg_attr(feature = "pyo3", pyo3(signature = (shape, /, degree = 0)))]
    pub fn with_degree(shape: [usize; 2], degree: u8) -> Self {
        Abs { shape, degree }
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn degree(&self) -> u8 {
        self.degree
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl ReLU {
    /// On plain real matrices, of degree 0.
    pub fn new(shape: [usize; 2]) -> Self {
        ReLU { shape, degree: 0 }
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl ReLU {
    /// On real matrices of the given `degree`.
    #[cfg_attr(feature = "pyo3", new)]
    #[cfg_attr(feature = "pyo3", pyo3(signature = (shape, /, degree = 0)))]
    pub fn with_degree(shape: [usize; 2], degree: u8) -> Self {
        ReLU { shape, degree }
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn degree(&self) -> u8 {
        self.degree
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Argmax {
    /// On plain real matrices, of degree 0.
    pub fn new(shape: [usize; 2]) -> Self {
        Argmax { shape, degree: 0 }
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Argmax {
    /// On real matrices of the given `degree`.
    #[cfg_attr(feature = "pyo3", new)]
    #[cfg_attr(feature = "pyo3", pyo3(signature = (shape, /, degree = 0)))]
    pub fn with_degree(shape: [usize; 2], degree: u8) -> Self {
        Argmax { shape, degree }
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn degree(&self) -> u8 {
        self.degree
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Min {
    /// On plain real matrices, of degree 0.
    pub fn new(shape: [usize; 2]) -> Self {
        Min { shape, degree: 0 }
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Min {
    /// On real matrices of the given `degree`.
    #[cfg_attr(feature = "pyo3", new)]
    #[cfg_attr(feature = "pyo3", pyo3(signature = (shape, /, degree = 0)))]
    pub fn with_degree(shape: [usize; 2], degree: u8) -> Self {
        Min { shape, degree }
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn degree(&self) -> u8 {
        self.degree
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Max {
    /// On plain real matrices, of degree 0.
    pub fn new(shape: [usize; 2]) -> Self {
        Max { shape, degree: 0 }
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Max {
    /// On real matrices of the given `degree`.
    #[cfg_attr(feature = "pyo3", new)]
    #[cfg_attr(feature = "pyo3", pyo3(signature = (shape, /, degree = 0)))]
    pub fn with_degree(shape: [usize; 2], degree: u8) -> Self {
        Max { shape, degree }
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn degree(&self) -> u8 {
        self.degree
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Transpose {
    /// On plain real matrices, of degree 0.
    pub fn new(shape: [usize; 2]) -> Self {
        Transpose { shape, degree: 0 }
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Transpose {
    /// On real matrices of the given `degree`.
    #[cfg_attr(feature = "pyo3", new)]
    #[cfg_attr(feature = "pyo3", pyo3(signature = (shape, /, degree = 0)))]
    pub fn with_degree(shape: [usize; 2], degree: u8) -> Self {
        Transpose { shape, degree }
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn degree(&self) -> u8 {
        self.degree
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl MatMul {
    /// On plain real matrices, of degree 0; fails unless the columns of the
    /// first shape are the rows of the second.
    pub fn new(shapes: [[usize; 2]; 2]) -> Result<Self, crate::Error> {
        Self::with_degrees(shapes, [0, 0])
    }

    /// The degrees `[p, q]` of the factors.
    pub fn degrees(&self) -> [u8; 2] {
        self.degrees
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl MatMul {
    /// On real matrices of the given `degrees`, `[p, q]`, writing one of
    /// degree `p + q`; fails unless the columns of the first shape are the
    /// rows of the second, and that degree is in range.
    #[cfg_attr(feature = "pyo3", new)]
    #[cfg_attr(feature = "pyo3", pyo3(signature = (shapes, /, degrees = [0, 0])))]
    pub fn with_degrees(shapes: [[usize; 2]; 2], degrees: [u8; 2]) -> Result<Self, crate::Error> {
        let [lhs, rhs] = shapes;
        if lhs[1] != rhs[0] {
            return Err(crate::Error(format!(
                "cannot multiply a {lhs:?} matrix by a {rhs:?} matrix"
            )));
        }
        let [p, q] = degrees;
        if p.checked_add(q).is_none() {
            return Err(crate::Error(format!(
                "the product of reals of degrees {p} and {q} is of a degree out of range"
            )));
        }
        Ok(MatMul { shapes, degrees })
    }

    /// The degrees of the factors, as a list: pyo3 would make `[u8; 2]`
    /// into `bytes`.
    #[cfg(feature = "pyo3")]
    #[getter(degrees)]
    fn py_degrees(&self) -> [usize; 2] {
        self.degrees.map(usize::from)
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shapes",)
    }

    /// The shapes of the factors, `[[m, n], [n, k]]`.
    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shapes(&self) -> [[usize; 2]; 2] {
        self.shapes
    }
}

impl Lt {
    /// On plain real matrices, of degree 0.
    pub fn new(shape: [usize; 2]) -> Self {
        Lt { shape, degree: 0 }
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Lt {
    /// On real matrices of the given `degree`.
    #[cfg_attr(feature = "pyo3", new)]
    #[cfg_attr(feature = "pyo3", pyo3(signature = (shape, /, degree = 0)))]
    pub fn with_degree(shape: [usize; 2], degree: u8) -> Self {
        Lt { shape, degree }
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn degree(&self) -> u8 {
        self.degree
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Ite {
    /// On plain real matrices, of degree 0.
    pub fn new(shape: [usize; 2]) -> Self {
        Ite { shape, degree: 0 }
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Ite {
    /// On real matrices of the given `degree`.
    #[cfg_attr(feature = "pyo3", new)]
    #[cfg_attr(feature = "pyo3", pyo3(signature = (shape, /, degree = 0)))]
    pub fn with_degree(shape: [usize; 2], degree: u8) -> Self {
        Ite { shape, degree }
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn degree(&self) -> u8 {
        self.degree
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Le {
    /// On plain real matrices, of degree 0.
    pub fn new(shape: [usize; 2]) -> Self {
        Le { shape, degree: 0 }
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Le {
    /// On real matrices of the given `degree`.
    #[cfg_attr(feature = "pyo3", new)]
    #[cfg_attr(feature = "pyo3", pyo3(signature = (shape, /, degree = 0)))]
    pub fn with_degree(shape: [usize; 2], degree: u8) -> Self {
        Le { shape, degree }
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn degree(&self) -> u8 {
        self.degree
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Gt {
    /// On plain real matrices, of degree 0.
    pub fn new(shape: [usize; 2]) -> Self {
        Gt { shape, degree: 0 }
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Gt {
    /// On real matrices of the given `degree`.
    #[cfg_attr(feature = "pyo3", new)]
    #[cfg_attr(feature = "pyo3", pyo3(signature = (shape, /, degree = 0)))]
    pub fn with_degree(shape: [usize; 2], degree: u8) -> Self {
        Gt { shape, degree }
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn degree(&self) -> u8 {
        self.degree
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Ge {
    /// On plain real matrices, of degree 0.
    pub fn new(shape: [usize; 2]) -> Self {
        Ge { shape, degree: 0 }
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Ge {
    /// On real matrices of the given `degree`.
    #[cfg_attr(feature = "pyo3", new)]
    #[cfg_attr(feature = "pyo3", pyo3(signature = (shape, /, degree = 0)))]
    pub fn with_degree(shape: [usize; 2], degree: u8) -> Self {
        Ge { shape, degree }
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn degree(&self) -> u8 {
        self.degree
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Eq {
    /// On plain real matrices, of degree 0.
    pub fn new(shape: [usize; 2]) -> Self {
        Eq { shape, degree: 0 }
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Eq {
    /// On real matrices of the given `degree`.
    #[cfg_attr(feature = "pyo3", new)]
    #[cfg_attr(feature = "pyo3", pyo3(signature = (shape, /, degree = 0)))]
    pub fn with_degree(shape: [usize; 2], degree: u8) -> Self {
        Eq { shape, degree }
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn degree(&self) -> u8 {
        self.degree
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Ne {
    /// On plain real matrices, of degree 0.
    pub fn new(shape: [usize; 2]) -> Self {
        Ne { shape, degree: 0 }
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Ne {
    /// On real matrices of the given `degree`.
    #[cfg_attr(feature = "pyo3", new)]
    #[cfg_attr(feature = "pyo3", pyo3(signature = (shape, /, degree = 0)))]
    pub fn with_degree(shape: [usize; 2], degree: u8) -> Self {
        Ne { shape, degree }
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn degree(&self) -> u8 {
        self.degree
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl Get {
    /// On plain real matrices, of degree 0; fails unless `row` and `col`
    /// index an entry of an `[m, n]` matrix.
    pub fn new(row: usize, col: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
        Self::with_degree(row, col, shape, 0)
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Get {
    /// On real matrices of the given `degree`; fails unless `row` and `col`
    /// index an entry of an `[m, n]` matrix.
    #[cfg_attr(feature = "pyo3", new)]
    #[cfg_attr(feature = "pyo3", pyo3(signature = (row, col, shape, /, degree = 0)))]
    pub fn with_degree(
        row: usize,
        col: usize,
        shape: [usize; 2],
        degree: u8,
    ) -> Result<Self, crate::Error> {
        super::check_entry(row, col, shape)?;
        Ok(Get {
            row,
            col,
            shape,
            degree,
        })
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn degree(&self) -> u8 {
        self.degree
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str, &'static str) {
        ("row", "col", "shape")
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn row(&self) -> usize {
        self.row
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn col(&self) -> usize {
        self.col
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}
