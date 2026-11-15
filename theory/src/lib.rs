pub mod any;
pub mod bv;
pub mod int;
pub mod lia;
pub mod lra;
pub mod real;
pub mod tensor;
pub mod zero;

pub use tensor::PyTensor;

use base::boundary::{Boundary, Poly};
use base::tangent::Tangent;
use derive_more::From;
use real::Real;
use subenum::subenum;
use zero::Zero;

/// The sorts of all theories: a value of `Any` is a sort of one of them.
#[derive(Clone, Copy, PartialEq, Debug, Eq, From)]
pub enum Any {
    Real(real::Real),
    Int(int::Int),
    Zero(zero::Zero),
}

/// Each sort's tangent, taken within `Any`.
impl Tangent for Any {
    fn of(s: Any) -> Any {
        match s {
            Any::Real(r) => Real::of(r).into(),
            Any::Int(i) => Zero::of(i).into(),
            Any::Zero(z) => Zero::of(z).into(),
        }
    }
}

/// The operations of all theories: a value of `Op` is an operation of one
/// of them.
///
/// `subenum` defines the theories as subsets of `Op`: [`LRA`], linear real
/// arithmetic, and [`LIA`], linear integer arithmetic. Each holds the
/// operations of its sort but `MatMul`, which multiplies two variables and
/// so is not linear. It also generates the casts between each theory and
/// `Op`: `From` up to `Op`, and `TryFrom` down from it.
#[subenum(LRA, LIA)]
#[derive(Clone, Debug, From)]
pub enum Op {
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
}

/// Each operation's boundary, with its sorts taken within `Any`.
impl Boundary<Any> for Op {
    fn dom(&self) -> Poly<Any> {
        match self {
            Op::RealConst(op) => op.dom().map(Any::from),
            Op::RealAdd(op) => op.dom().map(Any::from),
            Op::RealSub(op) => op.dom().map(Any::from),
            Op::RealLinear(op) => op.dom().map(Any::from),
            Op::RealReLU(op) => op.dom().map(Any::from),
            Op::RealMin(op) => op.dom().map(Any::from),
            Op::RealMax(op) => op.dom().map(Any::from),
            Op::RealTranspose(op) => op.dom().map(Any::from),
            Op::RealMatMul(op) => op.dom().map(Any::from),
            Op::IntConst(op) => op.dom().map(Any::from),
            Op::IntAdd(op) => op.dom().map(Any::from),
            Op::IntSub(op) => op.dom().map(Any::from),
            Op::IntLinear(op) => op.dom().map(Any::from),
            Op::IntReLU(op) => op.dom().map(Any::from),
            Op::IntMin(op) => op.dom().map(Any::from),
            Op::IntMax(op) => op.dom().map(Any::from),
            Op::IntTranspose(op) => op.dom().map(Any::from),
            Op::IntMatMul(op) => op.dom().map(Any::from),
        }
    }

    fn cod(&self) -> Poly<Any> {
        match self {
            Op::RealConst(op) => op.cod().map(Any::from),
            Op::RealAdd(op) => op.cod().map(Any::from),
            Op::RealSub(op) => op.cod().map(Any::from),
            Op::RealLinear(op) => op.cod().map(Any::from),
            Op::RealReLU(op) => op.cod().map(Any::from),
            Op::RealMin(op) => op.cod().map(Any::from),
            Op::RealMax(op) => op.cod().map(Any::from),
            Op::RealTranspose(op) => op.cod().map(Any::from),
            Op::RealMatMul(op) => op.cod().map(Any::from),
            Op::IntConst(op) => op.cod().map(Any::from),
            Op::IntAdd(op) => op.cod().map(Any::from),
            Op::IntSub(op) => op.cod().map(Any::from),
            Op::IntLinear(op) => op.cod().map(Any::from),
            Op::IntReLU(op) => op.cod().map(Any::from),
            Op::IntMin(op) => op.cod().map(Any::from),
            Op::IntMax(op) => op.cod().map(Any::from),
            Op::IntTranspose(op) => op.cod().map(Any::from),
            Op::IntMatMul(op) => op.cod().map(Any::from),
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
    use base::tangent::T;

    #[test]
    fn any_tangent_is_each_sorts_tangent() {
        let r = real::Real::new([2, 3]);
        let t: Any = T(Any::from(r));
        assert_eq!(t, Any::Real(T(r)));
        let t: Any = T(Any::from(int::Int::new([2, 3])));
        assert_eq!(t, Any::Zero(zero::Zero));
        let t: Any = T(Any::from(zero::Zero));
        assert_eq!(t, Any::Zero(zero::Zero));
    }

    #[test]
    fn op_boundary_is_each_operations_boundary_in_any() {
        let op: Op = real::MatMul(2, 3, 4).into();
        let dom: Poly<Any> = base::mono![
            real::Real::new([2, 3]).into(),
            real::Real::new([3, 4]).into()
        ]
        .into();
        let cod: Poly<Any> = base::mono![real::Real::new([2, 4]).into()].into();
        assert_eq!(op.dom(), dom);
        assert_eq!(op.cod(), cod);

        let op: Op = int::Transpose(2, 3).into();
        let cod: Poly<Any> = base::mono![int::Int::new([3, 2]).into()].into();
        assert_eq!(op.cod(), cod);
    }

    #[test]
    fn theories_cast_to_and_from_op() {
        // operations wrap into their theory, and theories into `Op`
        let lra: LRA = real::Add(2, 3).into();
        assert!(matches!(Op::from(lra), Op::RealAdd(real::Add(2, 3))));
        let lia: LIA = int::Sub(2, 3).into();
        assert!(matches!(Op::from(lia), Op::IntSub(int::Sub(2, 3))));

        // `Op` casts down to the theory holding the operation, and no other
        assert!(LRA::try_from(Op::from(real::ReLU(2, 3))).is_ok());
        assert!(LIA::try_from(Op::from(real::ReLU(2, 3))).is_err());
        assert!(LIA::try_from(Op::from(int::ReLU(2, 3))).is_ok());

        // `MatMul` is in neither linear theory
        assert!(LRA::try_from(Op::from(real::MatMul(2, 3, 4))).is_err());
        assert!(LIA::try_from(Op::from(int::MatMul(2, 3, 4))).is_err());
    }

    #[test]
    fn ops_convert_into_op() {
        assert!(matches!(
            real::Add(2, 3).into(),
            Op::RealAdd(real::Add(2, 3))
        ));
        assert!(matches!(int::Add(2, 3).into(), Op::IntAdd(int::Add(2, 3))));
        assert!(matches!(
            real::MatMul(2, 3, 4).into(),
            Op::RealMatMul(real::MatMul(2, 3, 4))
        ));
    }
}
