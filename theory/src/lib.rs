pub mod any;
pub mod bool;
pub mod bv;
pub mod int;
pub mod lia;
pub mod lra;
pub mod real;
pub mod tensor;
pub mod zero;

pub use tensor::PyTensor;

use base::boundary::{Boundary, Poly};
use base::differentiable::{Differentiable, T};
use derive_more::From;
use subenum::subenum;

/// The sorts of all theories: a value of `Sort` is a sort of one of them.
#[derive(Clone, Copy, PartialEq, Debug, Eq, From)]
pub enum Sort {
    Real(real::Real),
    Int(int::Int),
    Bool(bool::Bool),
    Zero(zero::Zero),
}

/// Each sort's tangent, taken within `Sort`.
impl Differentiable for Sort {
    type Tangent = Sort;

    fn tangent(self) -> Sort {
        match self {
            Sort::Real(r) => T(r).into(),
            Sort::Int(i) => T(i).into(),
            Sort::Bool(b) => T(b).into(),
            Sort::Zero(z) => T(z).into(),
        }
    }
}

/// The theory of all operations: a value of `Any` is one of its
/// operations, on any of the sorts in [`Sort`].
///
/// `subenum` defines the sub-theories as subsets of its operations:
/// [`Prop`], propositional logic, with the boolean operations; and [`LRA`],
/// linear real arithmetic, and [`LIA`], linear integer arithmetic, each with
/// the operations of its sort but `MatMul`, which multiplies two variables
/// and so is not linear, together with those of `Prop`. It also generates
/// the casts between `Any` and each sub-theory: `From` up to `Any`,
/// and `TryFrom` down from it.
#[subenum(LRA, LIA, Prop)]
#[derive(Clone, Debug, From)]
pub enum Any {
    #[subenum(LRA)]
    RealConst(real::Const),
    #[subenum(LRA)]
    RealAdd(real::Add),
    #[subenum(LRA)]
    RealSub(real::Sub),
    #[subenum(LRA)]
    RealLinear(real::Linear),
    #[subenum(LRA)]
    RealReLU(real::ReLU),
    #[subenum(LRA)]
    RealMin(real::Min),
    #[subenum(LRA)]
    RealMax(real::Max),
    #[subenum(LRA)]
    RealTranspose(real::Transpose),
    RealMatMul(real::MatMul),
    #[subenum(LIA)]
    IntConst(int::Const),
    #[subenum(LIA)]
    IntAdd(int::Add),
    #[subenum(LIA)]
    IntSub(int::Sub),
    #[subenum(LIA)]
    IntLinear(int::Linear),
    #[subenum(LIA)]
    IntReLU(int::ReLU),
    #[subenum(LIA)]
    IntMin(int::Min),
    #[subenum(LIA)]
    IntMax(int::Max),
    #[subenum(LIA)]
    IntTranspose(int::Transpose),
    IntMatMul(int::MatMul),
    #[subenum(LRA, LIA, Prop)]
    Truth(bool::Const),
    #[subenum(LRA, LIA, Prop)]
    And(bool::And),
    #[subenum(LRA, LIA, Prop)]
    Or(bool::Or),
    #[subenum(LRA, LIA, Prop)]
    Xor(bool::Xor),
    #[subenum(LRA, LIA, Prop)]
    Not(bool::Not),
}

// The embeddings of `Prop` into the linear theories, which `subenum` cannot
// express: it casts between `Any` and each sub-theory, but not between
// two sub-theories. Their inverses are absent, as nothing casts down from a
// linear theory to `Prop`.
impl From<Prop> for LRA {
    fn from(op: Prop) -> Self {
        match op {
            Prop::Truth(op) => LRA::Truth(op),
            Prop::And(op) => LRA::And(op),
            Prop::Or(op) => LRA::Or(op),
            Prop::Xor(op) => LRA::Xor(op),
            Prop::Not(op) => LRA::Not(op),
        }
    }
}

impl From<Prop> for LIA {
    fn from(op: Prop) -> Self {
        match op {
            Prop::Truth(op) => LIA::Truth(op),
            Prop::And(op) => LIA::And(op),
            Prop::Or(op) => LIA::Or(op),
            Prop::Xor(op) => LIA::Xor(op),
            Prop::Not(op) => LIA::Not(op),
        }
    }
}

/// Each operation's boundary, with its sorts taken within `Sort`.
impl Boundary<Sort> for Any {
    fn dom(&self) -> Poly<Sort> {
        match self {
            Any::RealConst(op) => op.dom().map(Sort::from),
            Any::RealAdd(op) => op.dom().map(Sort::from),
            Any::RealSub(op) => op.dom().map(Sort::from),
            Any::RealLinear(op) => op.dom().map(Sort::from),
            Any::RealReLU(op) => op.dom().map(Sort::from),
            Any::RealMin(op) => op.dom().map(Sort::from),
            Any::RealMax(op) => op.dom().map(Sort::from),
            Any::RealTranspose(op) => op.dom().map(Sort::from),
            Any::RealMatMul(op) => op.dom().map(Sort::from),
            Any::IntConst(op) => op.dom().map(Sort::from),
            Any::IntAdd(op) => op.dom().map(Sort::from),
            Any::IntSub(op) => op.dom().map(Sort::from),
            Any::IntLinear(op) => op.dom().map(Sort::from),
            Any::IntReLU(op) => op.dom().map(Sort::from),
            Any::IntMin(op) => op.dom().map(Sort::from),
            Any::IntMax(op) => op.dom().map(Sort::from),
            Any::IntTranspose(op) => op.dom().map(Sort::from),
            Any::IntMatMul(op) => op.dom().map(Sort::from),
            Any::Truth(op) => op.dom().map(Sort::from),
            Any::And(op) => op.dom().map(Sort::from),
            Any::Or(op) => op.dom().map(Sort::from),
            Any::Xor(op) => op.dom().map(Sort::from),
            Any::Not(op) => op.dom().map(Sort::from),
        }
    }

    fn cod(&self) -> Poly<Sort> {
        match self {
            Any::RealConst(op) => op.cod().map(Sort::from),
            Any::RealAdd(op) => op.cod().map(Sort::from),
            Any::RealSub(op) => op.cod().map(Sort::from),
            Any::RealLinear(op) => op.cod().map(Sort::from),
            Any::RealReLU(op) => op.cod().map(Sort::from),
            Any::RealMin(op) => op.cod().map(Sort::from),
            Any::RealMax(op) => op.cod().map(Sort::from),
            Any::RealTranspose(op) => op.cod().map(Sort::from),
            Any::RealMatMul(op) => op.cod().map(Sort::from),
            Any::IntConst(op) => op.cod().map(Sort::from),
            Any::IntAdd(op) => op.cod().map(Sort::from),
            Any::IntSub(op) => op.cod().map(Sort::from),
            Any::IntLinear(op) => op.cod().map(Sort::from),
            Any::IntReLU(op) => op.cod().map(Sort::from),
            Any::IntMin(op) => op.cod().map(Sort::from),
            Any::IntMax(op) => op.cod().map(Sort::from),
            Any::IntTranspose(op) => op.cod().map(Sort::from),
            Any::IntMatMul(op) => op.cod().map(Sort::from),
            Any::Truth(op) => op.cod().map(Sort::from),
            Any::And(op) => op.cod().map(Sort::from),
            Any::Or(op) => op.cod().map(Sort::from),
            Any::Xor(op) => op.cod().map(Sort::from),
            Any::Not(op) => op.cod().map(Sort::from),
        }
    }
}

pub trait Combinatorial<S> {
    // Returns the unary itype that chooses an element within the range
    fn havoc(range: &S) -> Self;
}

pub trait Sequential<S> {
    // Returns the unary itype that copies a value within the range
    fn skip(range: &S) -> Self;
}

pub trait Differential<S> {
    // Returns the unary itype that indicates zero derivative
    fn zero(range: &S) -> Self;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sort_tangent_is_each_sorts_tangent() {
        let r = real::Real::new([2, 3]);
        assert_eq!(T(Sort::from(r)), Sort::Real(T(r)));
        assert_eq!(T(Sort::from(int::Int::new([2, 3]))), Sort::Zero(zero::Zero));
        assert_eq!(
            T(Sort::from(bool::Bool::new([2, 3]))),
            Sort::Zero(zero::Zero)
        );
        assert_eq!(T(Sort::from(zero::Zero)), Sort::Zero(zero::Zero));
    }

    #[test]
    fn any_boundary_is_each_operations_boundary_in_sort() {
        let op: Any = real::MatMul(2, 3, 4).into();
        let dom: Poly<Sort> = base::mono![
            real::Real::new([2, 3]).into(),
            real::Real::new([3, 4]).into()
        ]
        .into();
        let cod: Poly<Sort> = base::mono![real::Real::new([2, 4]).into()].into();
        assert_eq!(op.dom(), dom);
        assert_eq!(op.cod(), cod);

        let op: Any = int::Transpose(2, 3).into();
        let cod: Poly<Sort> = base::mono![int::Int::new([3, 2]).into()].into();
        assert_eq!(op.cod(), cod);

        let op: Any = bool::Not(2, 3).into();
        let dom: Poly<Sort> = base::mono![bool::Bool::new([2, 3]).into()].into();
        assert_eq!(op.dom(), dom);
    }

    #[test]
    fn subtheories_cast_to_and_from_any() {
        // operations wrap into their sub-theory, and sub-theories into `Any`
        let lra: LRA = real::Add(2, 3).into();
        assert!(matches!(Any::from(lra), Any::RealAdd(real::Add(2, 3))));
        let lia: LIA = int::Sub(2, 3).into();
        assert!(matches!(Any::from(lia), Any::IntSub(int::Sub(2, 3))));

        // `Any` casts down to the sub-theories holding the operation only
        assert!(LRA::try_from(Any::from(real::ReLU(2, 3))).is_ok());
        assert!(LIA::try_from(Any::from(real::ReLU(2, 3))).is_err());
        assert!(LIA::try_from(Any::from(int::ReLU(2, 3))).is_ok());

        // boolean operations are in both linear theories
        assert!(LRA::try_from(Any::from(bool::And(2, 3))).is_ok());
        assert!(LIA::try_from(Any::from(bool::And(2, 3))).is_ok());

        // `Prop` holds the boolean operations only
        assert!(Prop::try_from(Any::from(bool::Or(2, 3))).is_ok());
        assert!(Prop::try_from(Any::from(real::Add(2, 3))).is_err());

        // `MatMul` is in neither linear theory
        assert!(LRA::try_from(Any::from(real::MatMul(2, 3, 4))).is_err());
        assert!(LIA::try_from(Any::from(int::MatMul(2, 3, 4))).is_err());
    }

    #[test]
    fn prop_embeds_into_linear_theories() {
        let prop: Prop = bool::And(2, 3).into();
        assert!(matches!(LRA::from(prop.clone()), LRA::And(bool::And(2, 3))));
        assert!(matches!(LIA::from(prop), LIA::And(bool::And(2, 3))));
    }

    #[test]
    fn ops_convert_into_any() {
        assert!(matches!(
            real::Add(2, 3).into(),
            Any::RealAdd(real::Add(2, 3))
        ));
        assert!(matches!(int::Add(2, 3).into(), Any::IntAdd(int::Add(2, 3))));
        assert!(matches!(
            real::MatMul(2, 3, 4).into(),
            Any::RealMatMul(real::MatMul(2, 3, 4))
        ));
    }
}
