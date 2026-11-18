/*!
# Real arithmetic

Defines one type per operation on the sort [`Real`] of real matrices. Every operation reads and writes real matrices, and carries enough
information to compute its [`Boundary`]:

- [`Const`] — a constant real matrix literal.
- [`Add`], [`Sub`] — pointwise addition and subtraction.
- [`Mul`] — pointwise multiplication `x ⊙ y`.
- [`Div`] — pointwise division `x / y`.
- [`Linear`]`(A, B)` — the affine map `x ↦ A·x + B`, with `A` and `B`
  constant real matrices of compatible shapes.
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

The comparisons are those of SMT-LIB2's `Reals` theory, `<`, `<=`, `>` and
`>=`, together with `=` and `distinct` of the Core theory, lifted pointwise
to matrices. As they write booleans, their boundary is in [`Sort`], the
sorts of the theory, rather than in [`Real`]. So is the boundary of
[`Ite`], which reads a boolean.
*/

use crate::sort::{Bool, Real};
use base::boundary::{Boundary, Poly};
use base::mono;
use derive_more::From;
use std::fmt;
use subenum::subenum;

/// The sorts of the theory, as in the signature of SMT-LIB2's `Reals`:
/// real matrices, and the booleans of the Core theory, which comparisons
/// write, together with [`Zero`](crate::sort::Zero). It is the subset
/// [`sort::Reals`](crate::sort::Reals) of the sorts of all theories.
pub use crate::sort::Reals as Sort;

/// A constant real matrix literal.
#[derive(Clone, Debug)]
pub struct Const(pub crate::PyTensor);

/// Pointwise addition `x + y` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Add(pub usize, pub usize);

/// Pointwise subtraction `x - y` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Sub(pub usize, pub usize);

/// Pointwise multiplication `x ⊙ y` of `[m, n]` matrices: the Hadamard
/// product, not the matrix product [`MatMul`].
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Mul(pub usize, pub usize);

/// Pointwise division `x / y` of `[m, n]` matrices: the Hadamard division
/// `x ⊘ y`. As in SMT-LIB2, `x / 0` is unspecified.
///
/// SMT-LIB2: `(/ x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Div(pub usize, pub usize);

/// The affine map `x ↦ A·x + B`, where `A` and `B` are constants.
#[derive(Clone, Debug)]
pub struct Linear(pub crate::PyTensor, pub crate::PyTensor);

/// The rectified-linear map `x ↦ max(x, 0)`, applied pointwise to
/// `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct ReLU(pub usize, pub usize);

/// Pointwise minimum `min(x, y)` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Min(pub usize, pub usize);

/// Pointwise maximum `max(x, y)` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Max(pub usize, pub usize);

/// Transposition `xᵀ` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Transpose(pub usize, pub usize);

/// The matrix product `x·y` of an `[m, n]` matrix `x` and an `[n, k]`
/// matrix `y`, with the sizes given as `MatMul(m, n, k)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct MatMul(pub usize, pub usize, pub usize);

/// Pointwise comparison `x < y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(< x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Lt(pub usize, pub usize);

/// Pointwise comparison `x ≤ y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(<= x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Le(pub usize, pub usize);

/// Pointwise comparison `x > y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(> x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Gt(pub usize, pub usize);

/// Pointwise comparison `x ≥ y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(>= x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Ge(pub usize, pub usize);

/// Pointwise comparison `x = y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(= x y)`, of the Core theory.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Eq(pub usize, pub usize);

/// Pointwise comparison `x ≠ y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(distinct x y)`, of the Core theory.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Ne(pub usize, pub usize);

/// The identity `x ↦ x` on `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Id(pub usize, pub usize);

/// If-then-else `if c then x else y` of `[m, n]` matrices, on a single
/// boolean `c`, which picks either matrix whole.
///
/// SMT-LIB2: `(ite c x y)`, of the Core theory, with the same `c` for every
/// entry.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Ite(pub usize, pub usize);

/// The index of the first largest entry of `[m, n]` matrices, counted from
/// `0` in row-major order, as a `[1, 1]` real matrix.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Argmax(pub usize, pub usize);

/// An arbitrary real matrix of shape `[m, n]`, chosen nondeterministically.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Havoc(pub usize, pub usize);

/// The operations of the theory of real arithmetic, SMT-LIB2's `Reals`,
/// with their boundary in [`Sort`].
///
/// `subenum` defines [`LRA`], the linear ones: all but `Mul`, `Div` and
/// `MatMul`, which multiply or divide two variables. It also generates the
/// casts between them: `From` up to `NRA`, and `TryFrom` down from it.
#[subenum(LRA)]
#[derive(Clone, Debug, From)]
pub enum NRA {
    #[subenum(LRA)]
    Const(Const),
    #[subenum(LRA)]
    Add(Add),
    #[subenum(LRA)]
    Sub(Sub),
    Mul(Mul),
    Div(Div),
    #[subenum(LRA)]
    Linear(Linear),
    #[subenum(LRA)]
    ReLU(ReLU),
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
    Le(Le),
    #[subenum(LRA)]
    Gt(Gt),
    #[subenum(LRA)]
    Ge(Ge),
    #[subenum(LRA)]
    Eq(Eq),
    #[subenum(LRA)]
    Ne(Ne),
    #[subenum(LRA)]
    Id(Id),
    #[subenum(LRA)]
    Ite(Ite),
    #[subenum(LRA)]
    Argmax(Argmax),
    #[subenum(LRA)]
    Havoc(Havoc),
}

impl Boundary<Sort> for NRA {
    fn dom(&self) -> Poly<Sort> {
        match self {
            NRA::Const(op) => op.dom().map(Sort::from),
            NRA::Add(op) => op.dom().map(Sort::from),
            NRA::Sub(op) => op.dom().map(Sort::from),
            NRA::Mul(op) => op.dom().map(Sort::from),
            NRA::Div(op) => op.dom().map(Sort::from),
            NRA::Linear(op) => op.dom().map(Sort::from),
            NRA::ReLU(op) => op.dom().map(Sort::from),
            NRA::Min(op) => op.dom().map(Sort::from),
            NRA::Max(op) => op.dom().map(Sort::from),
            NRA::Transpose(op) => op.dom().map(Sort::from),
            NRA::MatMul(op) => op.dom().map(Sort::from),
            NRA::Lt(op) => op.dom(),
            NRA::Le(op) => op.dom(),
            NRA::Gt(op) => op.dom(),
            NRA::Ge(op) => op.dom(),
            NRA::Eq(op) => op.dom(),
            NRA::Ne(op) => op.dom(),
            NRA::Id(op) => op.dom().map(Sort::from),
            NRA::Ite(op) => op.dom(),
            NRA::Argmax(op) => op.dom().map(Sort::from),
            NRA::Havoc(op) => op.dom().map(Sort::from),
        }
    }

    fn cod(&self) -> Poly<Sort> {
        match self {
            NRA::Const(op) => op.cod().map(Sort::from),
            NRA::Add(op) => op.cod().map(Sort::from),
            NRA::Sub(op) => op.cod().map(Sort::from),
            NRA::Mul(op) => op.cod().map(Sort::from),
            NRA::Div(op) => op.cod().map(Sort::from),
            NRA::Linear(op) => op.cod().map(Sort::from),
            NRA::ReLU(op) => op.cod().map(Sort::from),
            NRA::Min(op) => op.cod().map(Sort::from),
            NRA::Max(op) => op.cod().map(Sort::from),
            NRA::Transpose(op) => op.cod().map(Sort::from),
            NRA::MatMul(op) => op.cod().map(Sort::from),
            NRA::Lt(op) => op.cod(),
            NRA::Le(op) => op.cod(),
            NRA::Gt(op) => op.cod(),
            NRA::Ge(op) => op.cod(),
            NRA::Eq(op) => op.cod(),
            NRA::Ne(op) => op.cod(),
            NRA::Id(op) => op.cod().map(Sort::from),
            NRA::Ite(op) => op.cod(),
            NRA::Argmax(op) => op.cod().map(Sort::from),
            NRA::Havoc(op) => op.cod().map(Sort::from),
        }
    }
}
impl Boundary<Sort> for LRA {
    fn dom(&self) -> Poly<Sort> {
        match self {
            LRA::Const(op) => op.dom().map(Sort::from),
            LRA::Add(op) => op.dom().map(Sort::from),
            LRA::Sub(op) => op.dom().map(Sort::from),
            LRA::Linear(op) => op.dom().map(Sort::from),
            LRA::ReLU(op) => op.dom().map(Sort::from),
            LRA::Min(op) => op.dom().map(Sort::from),
            LRA::Max(op) => op.dom().map(Sort::from),
            LRA::Transpose(op) => op.dom().map(Sort::from),
            LRA::Lt(op) => op.dom(),
            LRA::Le(op) => op.dom(),
            LRA::Gt(op) => op.dom(),
            LRA::Ge(op) => op.dom(),
            LRA::Eq(op) => op.dom(),
            LRA::Ne(op) => op.dom(),
            LRA::Id(op) => op.dom().map(Sort::from),
            LRA::Ite(op) => op.dom(),
            LRA::Argmax(op) => op.dom().map(Sort::from),
            LRA::Havoc(op) => op.dom().map(Sort::from),
        }
    }

    fn cod(&self) -> Poly<Sort> {
        match self {
            LRA::Const(op) => op.cod().map(Sort::from),
            LRA::Add(op) => op.cod().map(Sort::from),
            LRA::Sub(op) => op.cod().map(Sort::from),
            LRA::Linear(op) => op.cod().map(Sort::from),
            LRA::ReLU(op) => op.cod().map(Sort::from),
            LRA::Min(op) => op.cod().map(Sort::from),
            LRA::Max(op) => op.cod().map(Sort::from),
            LRA::Transpose(op) => op.cod().map(Sort::from),
            LRA::Lt(op) => op.cod(),
            LRA::Le(op) => op.cod(),
            LRA::Gt(op) => op.cod(),
            LRA::Ge(op) => op.cod(),
            LRA::Eq(op) => op.cod(),
            LRA::Ne(op) => op.cod(),
            LRA::Id(op) => op.cod().map(Sort::from),
            LRA::Ite(op) => op.cod(),
            LRA::Argmax(op) => op.cod().map(Sort::from),
            LRA::Havoc(op) => op.cod().map(Sort::from),
        }
    }
}

impl Boundary<Real> for Const {
    fn dom(&self) -> Poly<Real> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new(self.0.shape())].into()
    }
}

impl Boundary<Real> for Add {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1]); 2].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1])].into()
    }
}

impl Boundary<Real> for Sub {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1]); 2].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1])].into()
    }
}

impl Boundary<Real> for Mul {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1]); 2].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1])].into()
    }
}

impl Boundary<Real> for Div {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1]); 2].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1])].into()
    }
}

/// With `A` of shape `[m, n]` and `B` of shape `[m, k]`, the map takes `x`
/// of shape `[n, k]` to a matrix of shape `[m, k]`.
impl Boundary<Real> for Linear {
    fn dom(&self) -> Poly<Real> {
        let [_, n] = self.0.shape();
        let [_, k] = self.1.shape();
        mono![Real::new([n, k])].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new(self.1.shape())].into()
    }
}

impl Boundary<Real> for ReLU {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1])].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1])].into()
    }
}

impl Boundary<Real> for Min {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1]); 2].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1])].into()
    }
}

impl Boundary<Real> for Max {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1]); 2].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1])].into()
    }
}

/// Takes a matrix of shape `[m, n]` to one of shape `[n, m]`.
impl Boundary<Real> for Transpose {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1])].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new([self.1, self.0])].into()
    }
}

/// Takes matrices of shapes `[m, n]` and `[n, k]` to one of shape `[m, k]`.
impl Boundary<Real> for MatMul {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1]), Real::new([self.1, self.2])].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.2])].into()
    }
}

impl Boundary<Sort> for Lt {
    fn dom(&self) -> Poly<Sort> {
        mono![Real::new([self.0, self.1]).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new([self.0, self.1]).into()].into()
    }
}

impl Boundary<Sort> for Le {
    fn dom(&self) -> Poly<Sort> {
        mono![Real::new([self.0, self.1]).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new([self.0, self.1]).into()].into()
    }
}

impl Boundary<Sort> for Gt {
    fn dom(&self) -> Poly<Sort> {
        mono![Real::new([self.0, self.1]).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new([self.0, self.1]).into()].into()
    }
}

impl Boundary<Sort> for Ge {
    fn dom(&self) -> Poly<Sort> {
        mono![Real::new([self.0, self.1]).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new([self.0, self.1]).into()].into()
    }
}

impl Boundary<Sort> for Eq {
    fn dom(&self) -> Poly<Sort> {
        mono![Real::new([self.0, self.1]).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new([self.0, self.1]).into()].into()
    }
}

impl Boundary<Sort> for Ne {
    fn dom(&self) -> Poly<Sort> {
        mono![Real::new([self.0, self.1]).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new([self.0, self.1]).into()].into()
    }
}

impl Boundary<Real> for Id {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1])].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1])].into()
    }
}

/// Takes a boolean of shape `[1, 1]` and two matrices of shape `[m, n]` to
/// a matrix of shape `[m, n]`.
impl Boundary<Sort> for Ite {
    fn dom(&self) -> Poly<Sort> {
        let x: Sort = Real::new([self.0, self.1]).into();
        mono![Bool::new([1, 1]).into(), x, x].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Real::new([self.0, self.1]).into()].into()
    }
}

/// Takes a matrix of shape `[m, n]` to one of shape `[1, 1]`.
impl Boundary<Real> for Argmax {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1])].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new([1, 1])].into()
    }
}

impl Boundary<Real> for Havoc {
    fn dom(&self) -> Poly<Real> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1])].into()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn one(s: Real) -> Poly<Real> {
        mono![s].into()
    }

    fn two(s: Real) -> Poly<Real> {
        mono![s, s].into()
    }

    #[test]
    fn pointwise_boundaries() {
        let r = Real::new([2, 3]);
        assert_eq!(Add(2, 3).dom(), two(r));
        assert_eq!(Add(2, 3).cod(), one(r));
        assert_eq!(Sub(2, 3).dom(), two(r));
        assert_eq!(Mul(2, 3).dom(), two(r));
        assert_eq!(Mul(2, 3).cod(), one(r));
        assert_eq!(Div(2, 3).dom(), two(r));
        assert_eq!(Div(2, 3).cod(), one(r));
        assert_eq!(Min(2, 3).dom(), two(r));
        assert_eq!(Max(2, 3).cod(), one(r));
        assert_eq!(ReLU(2, 3).dom(), one(r));
        assert_eq!(ReLU(2, 3).cod(), one(r));
    }

    #[test]
    fn transpose_swaps_shape() {
        let t = Transpose(2, 3);
        assert_eq!(t.dom(), one(Real::new([2, 3])));
        assert_eq!(t.cod(), one(Real::new([3, 2])));
    }

    #[test]
    fn matmul_contracts_inner_size() {
        let p = MatMul(2, 3, 4);
        let dom: Poly<Real> = mono![Real::new([2, 3]), Real::new([3, 4])].into();
        assert_eq!(p.dom(), dom);
        assert_eq!(p.cod(), one(Real::new([2, 4])));
    }

    #[test]
    fn id_keeps_sort() {
        let x = Real::new([2, 3]);
        assert_eq!(Id(2, 3).dom(), one(x));
        assert_eq!(Id(2, 3).cod(), one(x));
    }

    #[test]
    fn ite_reads_a_single_boolean() {
        let c: Sort = Bool::new([1, 1]).into();
        let x: Sort = Real::new([2, 3]).into();
        let dom: Poly<Sort> = mono![c, x, x].into();
        let cod: Poly<Sort> = mono![x].into();
        assert_eq!(Ite(2, 3).dom(), dom);
        assert_eq!(Ite(2, 3).cod(), cod);
    }

    #[test]
    fn argmax_writes_a_single_entry() {
        assert_eq!(Argmax(2, 3).dom(), one(Real::new([2, 3])));
        assert_eq!(Argmax(2, 3).cod(), one(Real::new([1, 1])));
    }

    #[test]
    fn havoc_reads_nothing() {
        assert_eq!(Havoc(2, 3).dom(), Poly::unit());
        assert_eq!(Havoc(2, 3).cod(), one(Real::new([2, 3])));
    }

    #[test]
    fn ops_display_their_name() {
        assert_eq!(Add(1, 1).to_string(), "Add");
        assert_eq!(Mul(1, 1).to_string(), "Mul");
        assert_eq!(Div(1, 1).to_string(), "Div");
        assert_eq!(ReLU(1, 1).to_string(), "ReLU");
        assert_eq!(Transpose(1, 1).to_string(), "Transpose");
        assert_eq!(Id(1, 1).to_string(), "Id");
        assert_eq!(Ite(1, 1).to_string(), "Ite");
        assert_eq!(Argmax(1, 1).to_string(), "Argmax");
        assert_eq!(Havoc(1, 1).to_string(), "Havoc");
    }
    #[test]
    fn comparisons_write_booleans() {
        let x: Sort = Real::new([2, 3]).into();
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
