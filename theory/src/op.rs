/*!
# Operations

The operations of each theory, one type per operation, in a module per
theory: [`real`] and [`int`] for arithmetic, [`bool`](mod@bool) for
propositional logic, and [`word`] for fixed-size bit-vectors. Each
operation reads and writes the sorts of [`crate::sort`], and computes its
own boundary; the enum [`Any`] holds the operations of all theories, and
its subenums those of each logic.
*/

pub mod bool;
pub mod int;
pub mod real;
pub mod word;

use crate::sort;
use base::boundary::{Boundary, Poly};
use derive_more::From;
use subenum::subenum;

/// The theory of all operations: a value of `Any` is one of its
/// operations, on any of the sorts in [`sort::Any`].
///
/// `subenum` defines the logics as subsets of its operations: [`Prop`],
/// propositional logic, with the boolean operations; [`NRA`], non-linear
/// real arithmetic, and [`NIA`], non-linear integer arithmetic, with all
/// the operations of [`real`] and [`int`] respectively, together with those
/// of `Prop`; [`LRA`], linear real arithmetic, and [`LIA`], linear integer
/// arithmetic, each with the operations of its non-linear logic but `Mul`,
/// `Div`, `Mod` and `MatMul`, which multiply or divide two variables and so
/// are not linear; and
/// [`BV`], the SMT-LIB2 logic QF_BV of fixed-size bit-vectors, with the
/// operations of [`word`] together with those of `Prop`. It also generates
/// the casts between `Any` and each logic: `From` up to `Any`, and
/// `TryFrom` down from it.
#[subenum(NRA, NIA, LRA, LIA, Prop, BV)]
#[derive(Clone, Debug, From)]
pub enum Any {
    #[subenum(NRA, LRA)]
    Real(real::Const),
    #[subenum(NRA, LRA)]
    RealAdd(real::Add),
    #[subenum(NRA, LRA)]
    RealSub(real::Sub),
    #[subenum(NRA)]
    RealMul(real::Mul),
    #[subenum(NRA)]
    RealDiv(real::Div),
    #[subenum(NRA, LRA)]
    RealLinear(real::Linear),
    #[subenum(NRA, LRA)]
    RealReLU(real::ReLU),
    #[subenum(NRA, LRA)]
    RealMin(real::Min),
    #[subenum(NRA, LRA)]
    RealMax(real::Max),
    #[subenum(NRA, LRA)]
    RealTranspose(real::Transpose),
    #[subenum(NRA)]
    RealMatMul(real::MatMul),
    #[subenum(NRA, LRA)]
    RealLt(real::Lt),
    #[subenum(NRA, LRA)]
    RealLe(real::Le),
    #[subenum(NRA, LRA)]
    RealGt(real::Gt),
    #[subenum(NRA, LRA)]
    RealGe(real::Ge),
    #[subenum(NRA, LRA)]
    RealEq(real::Eq),
    #[subenum(NRA, LRA)]
    RealNe(real::Ne),
    #[subenum(NIA, LIA)]
    Int(int::Const),
    #[subenum(NIA, LIA)]
    IntAdd(int::Add),
    #[subenum(NIA, LIA)]
    IntSub(int::Sub),
    #[subenum(NIA)]
    IntMul(int::Mul),
    #[subenum(NIA)]
    IntDiv(int::Div),
    #[subenum(NIA)]
    IntMod(int::Mod),
    #[subenum(NIA, LIA)]
    IntLinear(int::Linear),
    #[subenum(NIA, LIA)]
    IntReLU(int::ReLU),
    #[subenum(NIA, LIA)]
    IntMin(int::Min),
    #[subenum(NIA, LIA)]
    IntMax(int::Max),
    #[subenum(NIA, LIA)]
    IntTranspose(int::Transpose),
    #[subenum(NIA)]
    IntMatMul(int::MatMul),
    #[subenum(NIA, LIA)]
    IntLt(int::Lt),
    #[subenum(NIA, LIA)]
    IntLe(int::Le),
    #[subenum(NIA, LIA)]
    IntGt(int::Gt),
    #[subenum(NIA, LIA)]
    IntGe(int::Ge),
    #[subenum(NIA, LIA)]
    IntEq(int::Eq),
    #[subenum(NIA, LIA)]
    IntNe(int::Ne),
    #[subenum(NRA, NIA, LRA, LIA, Prop, BV)]
    Bool(bool::Const),
    #[subenum(NRA, NIA, LRA, LIA, Prop, BV)]
    And(bool::And),
    #[subenum(NRA, NIA, LRA, LIA, Prop, BV)]
    Or(bool::Or),
    #[subenum(NRA, NIA, LRA, LIA, Prop, BV)]
    Xor(bool::Xor),
    #[subenum(NRA, NIA, LRA, LIA, Prop, BV)]
    Not(bool::Not),
    #[subenum(BV)]
    Word(word::Const),
    #[subenum(BV)]
    Concat(word::Concat),
    #[subenum(BV)]
    Extract(word::Extract),
    #[subenum(BV)]
    Repeat(word::Repeat),
    #[subenum(BV)]
    BvNot(word::Not),
    #[subenum(BV)]
    BvAnd(word::And),
    #[subenum(BV)]
    BvOr(word::Or),
    #[subenum(BV)]
    BvXor(word::Xor),
    #[subenum(BV)]
    BvNand(word::Nand),
    #[subenum(BV)]
    BvNor(word::Nor),
    #[subenum(BV)]
    BvXnor(word::Xnor),
    #[subenum(BV)]
    BvNeg(word::Neg),
    #[subenum(BV)]
    BvAdd(word::Add),
    #[subenum(BV)]
    BvSub(word::Sub),
    #[subenum(BV)]
    BvMul(word::Mul),
    #[subenum(BV)]
    BvShl(word::Shl),
    #[subenum(BV)]
    RotateLeft(word::RotateLeft),
    #[subenum(BV)]
    RotateRight(word::RotateRight),
    #[subenum(BV)]
    BvEq(word::Eq),
    #[subenum(BV)]
    BvDistinct(word::Ne),
    #[subenum(BV)]
    BvSDiv(word::signed::Div),
    #[subenum(BV)]
    BvSRem(word::signed::Rem),
    #[subenum(BV)]
    BvSMod(word::signed::Mod),
    #[subenum(BV)]
    BvAShr(word::signed::Shr),
    #[subenum(BV)]
    SignExtend(word::signed::Extend),
    #[subenum(BV)]
    BvSLt(word::signed::Lt),
    #[subenum(BV)]
    BvSLe(word::signed::Le),
    #[subenum(BV)]
    BvSGt(word::signed::Gt),
    #[subenum(BV)]
    BvSGe(word::signed::Ge),
    #[subenum(BV)]
    BvUDiv(word::unsigned::Div),
    #[subenum(BV)]
    BvURem(word::unsigned::Rem),
    #[subenum(BV)]
    BvLShr(word::unsigned::Shr),
    #[subenum(BV)]
    ZeroExtend(word::unsigned::Extend),
    #[subenum(BV)]
    BvULt(word::unsigned::Lt),
    #[subenum(BV)]
    BvULe(word::unsigned::Le),
    #[subenum(BV)]
    BvUGt(word::unsigned::Gt),
    #[subenum(BV)]
    BvUGe(word::unsigned::Ge),
}

// The embeddings between logics, which `subenum` cannot express: it casts
// between `Any` and each logic, but not between two of them. `Prop` embeds
// into every other logic, and each linear logic into its non-linear one.
// Their inverses are absent, as nothing casts down between logics.
impl From<Prop> for NRA {
    fn from(op: Prop) -> Self {
        match op {
            Prop::Bool(op) => NRA::Bool(op),
            Prop::And(op) => NRA::And(op),
            Prop::Or(op) => NRA::Or(op),
            Prop::Xor(op) => NRA::Xor(op),
            Prop::Not(op) => NRA::Not(op),
        }
    }
}

impl From<Prop> for NIA {
    fn from(op: Prop) -> Self {
        match op {
            Prop::Bool(op) => NIA::Bool(op),
            Prop::And(op) => NIA::And(op),
            Prop::Or(op) => NIA::Or(op),
            Prop::Xor(op) => NIA::Xor(op),
            Prop::Not(op) => NIA::Not(op),
        }
    }
}

impl From<Prop> for LRA {
    fn from(op: Prop) -> Self {
        match op {
            Prop::Bool(op) => LRA::Bool(op),
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
            Prop::Bool(op) => LIA::Bool(op),
            Prop::And(op) => LIA::And(op),
            Prop::Or(op) => LIA::Or(op),
            Prop::Xor(op) => LIA::Xor(op),
            Prop::Not(op) => LIA::Not(op),
        }
    }
}

impl From<Prop> for BV {
    fn from(op: Prop) -> Self {
        match op {
            Prop::Bool(op) => BV::Bool(op),
            Prop::And(op) => BV::And(op),
            Prop::Or(op) => BV::Or(op),
            Prop::Xor(op) => BV::Xor(op),
            Prop::Not(op) => BV::Not(op),
        }
    }
}

impl From<LRA> for NRA {
    fn from(op: LRA) -> Self {
        match op {
            LRA::Real(op) => NRA::Real(op),
            LRA::RealAdd(op) => NRA::RealAdd(op),
            LRA::RealSub(op) => NRA::RealSub(op),
            LRA::RealLinear(op) => NRA::RealLinear(op),
            LRA::RealReLU(op) => NRA::RealReLU(op),
            LRA::RealMin(op) => NRA::RealMin(op),
            LRA::RealMax(op) => NRA::RealMax(op),
            LRA::RealTranspose(op) => NRA::RealTranspose(op),
            LRA::RealLt(op) => NRA::RealLt(op),
            LRA::RealLe(op) => NRA::RealLe(op),
            LRA::RealGt(op) => NRA::RealGt(op),
            LRA::RealGe(op) => NRA::RealGe(op),
            LRA::RealEq(op) => NRA::RealEq(op),
            LRA::RealNe(op) => NRA::RealNe(op),
            LRA::Bool(op) => NRA::Bool(op),
            LRA::And(op) => NRA::And(op),
            LRA::Or(op) => NRA::Or(op),
            LRA::Xor(op) => NRA::Xor(op),
            LRA::Not(op) => NRA::Not(op),
        }
    }
}

impl From<LIA> for NIA {
    fn from(op: LIA) -> Self {
        match op {
            LIA::Int(op) => NIA::Int(op),
            LIA::IntAdd(op) => NIA::IntAdd(op),
            LIA::IntSub(op) => NIA::IntSub(op),
            LIA::IntLinear(op) => NIA::IntLinear(op),
            LIA::IntReLU(op) => NIA::IntReLU(op),
            LIA::IntMin(op) => NIA::IntMin(op),
            LIA::IntMax(op) => NIA::IntMax(op),
            LIA::IntTranspose(op) => NIA::IntTranspose(op),
            LIA::IntLt(op) => NIA::IntLt(op),
            LIA::IntLe(op) => NIA::IntLe(op),
            LIA::IntGt(op) => NIA::IntGt(op),
            LIA::IntGe(op) => NIA::IntGe(op),
            LIA::IntEq(op) => NIA::IntEq(op),
            LIA::IntNe(op) => NIA::IntNe(op),
            LIA::Bool(op) => NIA::Bool(op),
            LIA::And(op) => NIA::And(op),
            LIA::Or(op) => NIA::Or(op),
            LIA::Xor(op) => NIA::Xor(op),
            LIA::Not(op) => NIA::Not(op),
        }
    }
}

/// Each operation's boundary, with its sorts taken within [`sort::Any`].
impl Boundary<sort::Any> for Any {
    fn dom(&self) -> Poly<sort::Any> {
        match self {
            Any::Real(op) => op.dom().map(sort::Any::from),
            Any::RealAdd(op) => op.dom().map(sort::Any::from),
            Any::RealSub(op) => op.dom().map(sort::Any::from),
            Any::RealMul(op) => op.dom().map(sort::Any::from),
            Any::RealDiv(op) => op.dom().map(sort::Any::from),
            Any::RealLinear(op) => op.dom().map(sort::Any::from),
            Any::RealReLU(op) => op.dom().map(sort::Any::from),
            Any::RealMin(op) => op.dom().map(sort::Any::from),
            Any::RealMax(op) => op.dom().map(sort::Any::from),
            Any::RealTranspose(op) => op.dom().map(sort::Any::from),
            Any::RealMatMul(op) => op.dom().map(sort::Any::from),
            Any::RealLt(op) => op.dom().map(sort::Any::from),
            Any::RealLe(op) => op.dom().map(sort::Any::from),
            Any::RealGt(op) => op.dom().map(sort::Any::from),
            Any::RealGe(op) => op.dom().map(sort::Any::from),
            Any::RealEq(op) => op.dom().map(sort::Any::from),
            Any::RealNe(op) => op.dom().map(sort::Any::from),
            Any::Int(op) => op.dom().map(sort::Any::from),
            Any::IntAdd(op) => op.dom().map(sort::Any::from),
            Any::IntSub(op) => op.dom().map(sort::Any::from),
            Any::IntMul(op) => op.dom().map(sort::Any::from),
            Any::IntDiv(op) => op.dom().map(sort::Any::from),
            Any::IntMod(op) => op.dom().map(sort::Any::from),
            Any::IntLinear(op) => op.dom().map(sort::Any::from),
            Any::IntReLU(op) => op.dom().map(sort::Any::from),
            Any::IntMin(op) => op.dom().map(sort::Any::from),
            Any::IntMax(op) => op.dom().map(sort::Any::from),
            Any::IntTranspose(op) => op.dom().map(sort::Any::from),
            Any::IntMatMul(op) => op.dom().map(sort::Any::from),
            Any::IntLt(op) => op.dom().map(sort::Any::from),
            Any::IntLe(op) => op.dom().map(sort::Any::from),
            Any::IntGt(op) => op.dom().map(sort::Any::from),
            Any::IntGe(op) => op.dom().map(sort::Any::from),
            Any::IntEq(op) => op.dom().map(sort::Any::from),
            Any::IntNe(op) => op.dom().map(sort::Any::from),
            Any::Bool(op) => op.dom().map(sort::Any::from),
            Any::And(op) => op.dom().map(sort::Any::from),
            Any::Or(op) => op.dom().map(sort::Any::from),
            Any::Xor(op) => op.dom().map(sort::Any::from),
            Any::Not(op) => op.dom().map(sort::Any::from),
            Any::Word(op) => op.dom().map(sort::Any::from),
            Any::Concat(op) => op.dom().map(sort::Any::from),
            Any::Extract(op) => op.dom().map(sort::Any::from),
            Any::Repeat(op) => op.dom().map(sort::Any::from),
            Any::BvNot(op) => op.dom().map(sort::Any::from),
            Any::BvAnd(op) => op.dom().map(sort::Any::from),
            Any::BvOr(op) => op.dom().map(sort::Any::from),
            Any::BvXor(op) => op.dom().map(sort::Any::from),
            Any::BvNand(op) => op.dom().map(sort::Any::from),
            Any::BvNor(op) => op.dom().map(sort::Any::from),
            Any::BvXnor(op) => op.dom().map(sort::Any::from),
            Any::BvNeg(op) => op.dom().map(sort::Any::from),
            Any::BvAdd(op) => op.dom().map(sort::Any::from),
            Any::BvSub(op) => op.dom().map(sort::Any::from),
            Any::BvMul(op) => op.dom().map(sort::Any::from),
            Any::BvShl(op) => op.dom().map(sort::Any::from),
            Any::RotateLeft(op) => op.dom().map(sort::Any::from),
            Any::RotateRight(op) => op.dom().map(sort::Any::from),
            Any::BvEq(op) => op.dom().map(sort::Any::from),
            Any::BvDistinct(op) => op.dom().map(sort::Any::from),
            Any::BvSDiv(op) => op.dom().map(sort::Any::from),
            Any::BvSRem(op) => op.dom().map(sort::Any::from),
            Any::BvSMod(op) => op.dom().map(sort::Any::from),
            Any::BvAShr(op) => op.dom().map(sort::Any::from),
            Any::SignExtend(op) => op.dom().map(sort::Any::from),
            Any::BvSLt(op) => op.dom().map(sort::Any::from),
            Any::BvSLe(op) => op.dom().map(sort::Any::from),
            Any::BvSGt(op) => op.dom().map(sort::Any::from),
            Any::BvSGe(op) => op.dom().map(sort::Any::from),
            Any::BvUDiv(op) => op.dom().map(sort::Any::from),
            Any::BvURem(op) => op.dom().map(sort::Any::from),
            Any::BvLShr(op) => op.dom().map(sort::Any::from),
            Any::ZeroExtend(op) => op.dom().map(sort::Any::from),
            Any::BvULt(op) => op.dom().map(sort::Any::from),
            Any::BvULe(op) => op.dom().map(sort::Any::from),
            Any::BvUGt(op) => op.dom().map(sort::Any::from),
            Any::BvUGe(op) => op.dom().map(sort::Any::from),
        }
    }

    fn cod(&self) -> Poly<sort::Any> {
        match self {
            Any::Real(op) => op.cod().map(sort::Any::from),
            Any::RealAdd(op) => op.cod().map(sort::Any::from),
            Any::RealSub(op) => op.cod().map(sort::Any::from),
            Any::RealMul(op) => op.cod().map(sort::Any::from),
            Any::RealDiv(op) => op.cod().map(sort::Any::from),
            Any::RealLinear(op) => op.cod().map(sort::Any::from),
            Any::RealReLU(op) => op.cod().map(sort::Any::from),
            Any::RealMin(op) => op.cod().map(sort::Any::from),
            Any::RealMax(op) => op.cod().map(sort::Any::from),
            Any::RealTranspose(op) => op.cod().map(sort::Any::from),
            Any::RealMatMul(op) => op.cod().map(sort::Any::from),
            Any::RealLt(op) => op.cod().map(sort::Any::from),
            Any::RealLe(op) => op.cod().map(sort::Any::from),
            Any::RealGt(op) => op.cod().map(sort::Any::from),
            Any::RealGe(op) => op.cod().map(sort::Any::from),
            Any::RealEq(op) => op.cod().map(sort::Any::from),
            Any::RealNe(op) => op.cod().map(sort::Any::from),
            Any::Int(op) => op.cod().map(sort::Any::from),
            Any::IntAdd(op) => op.cod().map(sort::Any::from),
            Any::IntSub(op) => op.cod().map(sort::Any::from),
            Any::IntMul(op) => op.cod().map(sort::Any::from),
            Any::IntDiv(op) => op.cod().map(sort::Any::from),
            Any::IntMod(op) => op.cod().map(sort::Any::from),
            Any::IntLinear(op) => op.cod().map(sort::Any::from),
            Any::IntReLU(op) => op.cod().map(sort::Any::from),
            Any::IntMin(op) => op.cod().map(sort::Any::from),
            Any::IntMax(op) => op.cod().map(sort::Any::from),
            Any::IntTranspose(op) => op.cod().map(sort::Any::from),
            Any::IntMatMul(op) => op.cod().map(sort::Any::from),
            Any::IntLt(op) => op.cod().map(sort::Any::from),
            Any::IntLe(op) => op.cod().map(sort::Any::from),
            Any::IntGt(op) => op.cod().map(sort::Any::from),
            Any::IntGe(op) => op.cod().map(sort::Any::from),
            Any::IntEq(op) => op.cod().map(sort::Any::from),
            Any::IntNe(op) => op.cod().map(sort::Any::from),
            Any::Bool(op) => op.cod().map(sort::Any::from),
            Any::And(op) => op.cod().map(sort::Any::from),
            Any::Or(op) => op.cod().map(sort::Any::from),
            Any::Xor(op) => op.cod().map(sort::Any::from),
            Any::Not(op) => op.cod().map(sort::Any::from),
            Any::Word(op) => op.cod().map(sort::Any::from),
            Any::Concat(op) => op.cod().map(sort::Any::from),
            Any::Extract(op) => op.cod().map(sort::Any::from),
            Any::Repeat(op) => op.cod().map(sort::Any::from),
            Any::BvNot(op) => op.cod().map(sort::Any::from),
            Any::BvAnd(op) => op.cod().map(sort::Any::from),
            Any::BvOr(op) => op.cod().map(sort::Any::from),
            Any::BvXor(op) => op.cod().map(sort::Any::from),
            Any::BvNand(op) => op.cod().map(sort::Any::from),
            Any::BvNor(op) => op.cod().map(sort::Any::from),
            Any::BvXnor(op) => op.cod().map(sort::Any::from),
            Any::BvNeg(op) => op.cod().map(sort::Any::from),
            Any::BvAdd(op) => op.cod().map(sort::Any::from),
            Any::BvSub(op) => op.cod().map(sort::Any::from),
            Any::BvMul(op) => op.cod().map(sort::Any::from),
            Any::BvShl(op) => op.cod().map(sort::Any::from),
            Any::RotateLeft(op) => op.cod().map(sort::Any::from),
            Any::RotateRight(op) => op.cod().map(sort::Any::from),
            Any::BvEq(op) => op.cod().map(sort::Any::from),
            Any::BvDistinct(op) => op.cod().map(sort::Any::from),
            Any::BvSDiv(op) => op.cod().map(sort::Any::from),
            Any::BvSRem(op) => op.cod().map(sort::Any::from),
            Any::BvSMod(op) => op.cod().map(sort::Any::from),
            Any::BvAShr(op) => op.cod().map(sort::Any::from),
            Any::SignExtend(op) => op.cod().map(sort::Any::from),
            Any::BvSLt(op) => op.cod().map(sort::Any::from),
            Any::BvSLe(op) => op.cod().map(sort::Any::from),
            Any::BvSGt(op) => op.cod().map(sort::Any::from),
            Any::BvSGe(op) => op.cod().map(sort::Any::from),
            Any::BvUDiv(op) => op.cod().map(sort::Any::from),
            Any::BvURem(op) => op.cod().map(sort::Any::from),
            Any::BvLShr(op) => op.cod().map(sort::Any::from),
            Any::ZeroExtend(op) => op.cod().map(sort::Any::from),
            Any::BvULt(op) => op.cod().map(sort::Any::from),
            Any::BvULe(op) => op.cod().map(sort::Any::from),
            Any::BvUGt(op) => op.cod().map(sort::Any::from),
            Any::BvUGe(op) => op.cod().map(sort::Any::from),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn any_boundary_is_each_operations_boundary_in_sort() {
        let op: Any = real::MatMul(2, 3, 4).into();
        let dom: Poly<sort::Any> = base::mono![
            sort::Real::new([2, 3]).into(),
            sort::Real::new([3, 4]).into()
        ]
        .into();
        let cod: Poly<sort::Any> = base::mono![sort::Real::new([2, 4]).into()].into();
        assert_eq!(op.dom(), dom);
        assert_eq!(op.cod(), cod);

        let op: Any = int::Transpose(2, 3).into();
        let cod: Poly<sort::Any> = base::mono![sort::Int::new([3, 2]).into()].into();
        assert_eq!(op.cod(), cod);

        let op: Any = bool::Not(2, 3).into();
        let dom: Poly<sort::Any> = base::mono![sort::Bool::new([2, 3]).into()].into();
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

        // neither is the pointwise `Mul`, which is in the non-linear logics
        assert!(LRA::try_from(Any::from(real::Mul(2, 3))).is_err());
        assert!(LIA::try_from(Any::from(int::Mul(2, 3))).is_err());
        assert!(NRA::try_from(Any::from(real::Mul(2, 3))).is_ok());
        assert!(NIA::try_from(Any::from(int::Mul(2, 3))).is_ok());

        // nor are division and remainder, which are also non-linear
        assert!(LRA::try_from(Any::from(real::Div(2, 3))).is_err());
        assert!(LIA::try_from(Any::from(int::Div(2, 3))).is_err());
        assert!(LIA::try_from(Any::from(int::Mod(2, 3))).is_err());
        assert!(NRA::try_from(Any::from(real::Div(2, 3))).is_ok());
        assert!(NIA::try_from(Any::from(int::Div(2, 3))).is_ok());
        assert!(NIA::try_from(Any::from(int::Mod(2, 3))).is_ok());
    }

    #[test]
    fn prop_embeds_into_linear_theories() {
        let prop: Prop = bool::And(2, 3).into();
        assert!(matches!(LRA::from(prop.clone()), LRA::And(bool::And(2, 3))));
        assert!(matches!(LIA::from(prop), LIA::And(bool::And(2, 3))));
    }

    #[test]
    fn non_linear_logics_hold_all_operations_on_their_sorts() {
        // `MatMul` is in the non-linear logics, though in neither linear one
        assert!(NRA::try_from(Any::from(real::MatMul(2, 3, 4))).is_ok());
        assert!(NIA::try_from(Any::from(int::MatMul(2, 3, 4))).is_ok());
        assert!(NRA::try_from(Any::from(real::Add(2, 3))).is_ok());
        assert!(NIA::try_from(Any::from(int::ReLU(2, 3))).is_ok());

        // each holds the boolean operations, and no other sort's
        assert!(NRA::try_from(Any::from(bool::And(2, 3))).is_ok());
        assert!(NIA::try_from(Any::from(bool::Not(2, 3))).is_ok());
        assert!(NRA::try_from(Any::from(int::Add(2, 3))).is_err());
        assert!(NIA::try_from(Any::from(real::Add(2, 3))).is_err());
        assert!(NRA::try_from(Any::from(word::Add(8, 2, 3))).is_err());

        // the linear logics and `Prop` embed into the non-linear ones
        let lra: LRA = real::Sub(2, 3).into();
        assert!(matches!(NRA::from(lra), NRA::RealSub(real::Sub(2, 3))));
        let lia: LIA = int::Max(2, 3).into();
        assert!(matches!(NIA::from(lia), NIA::IntMax(int::Max(2, 3))));
        let prop: Prop = bool::Or(2, 3).into();
        assert!(matches!(NRA::from(prop.clone()), NRA::Or(bool::Or(2, 3))));
        assert!(matches!(NIA::from(prop), NIA::Or(bool::Or(2, 3))));
    }

    #[test]
    fn bv_holds_word_and_boolean_operations() {
        // word operations cast to and from `BV`
        let bv: BV = word::signed::Div(8, 2, 3).into();
        assert!(matches!(
            Any::from(bv),
            Any::BvSDiv(word::signed::Div(8, 2, 3))
        ));
        assert!(BV::try_from(Any::from(word::unsigned::Lt(8, 2, 3))).is_ok());
        assert!(BV::try_from(Any::from(word::Concat(8, 4, 2, 3))).is_ok());

        // boolean operations are in `BV`, and `Prop` embeds into it
        assert!(BV::try_from(Any::from(bool::Xor(2, 3))).is_ok());
        let prop: Prop = bool::Not(2, 3).into();
        assert!(matches!(BV::from(prop), BV::Not(bool::Not(2, 3))));

        // arithmetic operations of other sorts are not in `BV`
        assert!(BV::try_from(Any::from(int::Add(2, 3))).is_err());
        assert!(BV::try_from(Any::from(real::Add(2, 3))).is_err());

        // word operations are in none of the other sub-theories
        assert!(LIA::try_from(Any::from(word::Add(8, 2, 3))).is_err());
        assert!(Prop::try_from(Any::from(word::Not(8, 2, 3))).is_err());
    }

    #[test]
    fn word_ops_have_their_boundary_in_sort() {
        let op: Any = word::Add(8, 2, 3).into();
        let w: sort::Any = sort::Word::new_unchecked(8, [2, 3]).into();
        let dom: Poly<sort::Any> = base::mono![w, w].into();
        let cod: Poly<sort::Any> = base::mono![w].into();
        assert_eq!(op.dom(), dom);
        assert_eq!(op.cod(), cod);

        // comparisons write the booleans of `bool`
        let op: Any = word::signed::Lt(8, 2, 3).into();
        let cod: Poly<sort::Any> = base::mono![sort::Bool::new([2, 3]).into()].into();
        assert_eq!(op.dom(), dom);
        assert_eq!(op.cod(), cod);

        // signed and unsigned operations are distinct operations of `Any`
        assert!(matches!(
            word::signed::Div(8, 2, 3).into(),
            Any::BvSDiv(word::signed::Div(8, 2, 3))
        ));
        assert!(matches!(
            word::unsigned::Div(8, 2, 3).into(),
            Any::BvUDiv(word::unsigned::Div(8, 2, 3))
        ));
    }

    #[test]
    fn comparisons_are_in_the_arithmetic_logics() {
        // comparisons are linear, so in both the linear and non-linear logics
        assert!(LRA::try_from(Any::from(real::Lt(2, 3))).is_ok());
        assert!(NRA::try_from(Any::from(real::Ne(2, 3))).is_ok());
        assert!(LIA::try_from(Any::from(int::Ge(2, 3))).is_ok());
        assert!(NIA::try_from(Any::from(int::Eq(2, 3))).is_ok());
        assert!(LIA::try_from(Any::from(real::Lt(2, 3))).is_err());
        assert!(LRA::try_from(Any::from(int::Lt(2, 3))).is_err());
        assert!(Prop::try_from(Any::from(real::Le(2, 3))).is_err());

        let lra: LRA = real::Gt(2, 3).into();
        assert!(matches!(NRA::from(lra), NRA::RealGt(real::Gt(2, 3))));
        let lia: LIA = int::Le(2, 3).into();
        assert!(matches!(NIA::from(lia), NIA::IntLe(int::Le(2, 3))));

        // they read numbers and write booleans, in `sort::Any`
        let op: Any = int::Lt(2, 3).into();
        let i: sort::Any = sort::Int::new([2, 3]).into();
        let dom: Poly<sort::Any> = base::mono![i, i].into();
        let cod: Poly<sort::Any> = base::mono![sort::Bool::new([2, 3]).into()].into();
        assert_eq!(op.dom(), dom);
        assert_eq!(op.cod(), cod);
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
