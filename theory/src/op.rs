/*!
# Operations

The operations of each theory, one type per operation, in a module per
theory: [`real`] and [`int`] for arithmetic, [`bool`](mod@bool) for
propositional logic, and [`word`] for fixed-size bit-vectors. Each
operation reads and writes the sorts of [`crate::sort`], and computes its
own boundary.

The operations are organised in two levels. Each theory module gathers its
operations in an enum named after the logic of that theory alone:
[`real::NRA`], [`int::NIA`], [`bool::Prop`] and [`word::BV`]. The
arithmetic ones mark their linear operations with a subenum named after
the linear logic: [`real::LRA`] and [`int::LIA`].
Here, [`Any`] holds the operations of all theories, one variant per
theory, and each logic picks the theories it combines, restricting
arithmetic to its linear operations where the logic is linear:

| Logic | Operations | SMT-LIB2 logic |
|---|---|---|
| [`Prop`] | [`bool::Prop`] | none: the `Core` theory alone |
| [`NRA`] | [`real::NRA`], [`bool::Prop`] | `QF_NRA` |
| [`LRA`] | [`real::LRA`], [`bool::Prop`] | `QF_LRA` |
| [`NIA`] | [`int::NIA`], [`bool::Prop`] | `QF_NIA` |
| [`LIA`] | [`int::LIA`], [`bool::Prop`] | `QF_LIA` |
| [`BV`] | [`word::BV`], [`bool::Prop`] | `QF_BV` |
*/

pub mod bool;
pub mod int;
pub mod real;
pub mod word;

use crate::sort;
use base::boundary::{Boundary, Poly};
use derive_more::From;
use subenum::subenum;

/// The operations of all theories: a value of `Any` is an operation of one
/// of them, on the sorts in [`sort::Any`].
///
/// `subenum` defines the logics that take whole theories: [`Prop`],
/// propositional logic; [`NRA`] and [`NIA`], non-linear real and integer
/// arithmetic; and [`BV`], the logic `QF_BV` of fixed-size bit-vectors. It
/// also generates their casts: `From` up to `Any`, and `TryFrom` down from
/// it. The linear logics [`LRA`] and [`LIA`] restrict a theory to its
/// linear operations, which `subenum` cannot express, so they are defined
/// on their own, with the same casts.
#[subenum(NRA, NIA, Prop, BV)]
#[derive(Clone, Debug, From)]
pub enum Any {
    #[subenum(NRA)]
    Real(real::NRA),
    #[subenum(NIA)]
    Int(int::NIA),
    #[subenum(NRA, NIA, Prop, BV)]
    Bool(bool::Prop),
    #[subenum(BV)]
    Word(word::BV),
}

/// Linear real arithmetic: the linear operations of [`real`], with the
/// boolean operations.
#[derive(Clone, Debug, From)]
pub enum LRA {
    Real(real::LRA),
    Bool(bool::Prop),
}

/// Linear integer arithmetic: the linear operations of [`int`], with the
/// boolean operations.
#[derive(Clone, Debug, From)]
pub enum LIA {
    Int(int::LIA),
    Bool(bool::Prop),
}

impl Boundary<sort::Any> for Any {
    fn dom(&self) -> Poly<sort::Any> {
        match self {
            Any::Real(op) => op.dom().map(sort::Any::from),
            Any::Int(op) => op.dom().map(sort::Any::from),
            Any::Bool(op) => op.dom().map(sort::Any::from),
            Any::Word(op) => op.dom().map(sort::Any::from),
        }
    }

    fn cod(&self) -> Poly<sort::Any> {
        match self {
            Any::Real(op) => op.cod().map(sort::Any::from),
            Any::Int(op) => op.cod().map(sort::Any::from),
            Any::Bool(op) => op.cod().map(sort::Any::from),
            Any::Word(op) => op.cod().map(sort::Any::from),
        }
    }
}

impl From<LRA> for Any {
    fn from(op: LRA) -> Self {
        match op {
            LRA::Real(op) => Any::Real(op.into()),
            LRA::Bool(op) => Any::Bool(op),
        }
    }
}

impl TryFrom<Any> for LRA {
    type Error = String;

    fn try_from(op: Any) -> Result<Self, Self::Error> {
        match op {
            Any::Real(op) => op
                .try_into()
                .map(LRA::Real)
                .map_err(|_| "LRA holds the linear real operations only".to_string()),
            Any::Bool(op) => Ok(LRA::Bool(op)),
            _ => Err("LRA holds real and boolean operations only".to_string()),
        }
    }
}

impl From<LIA> for Any {
    fn from(op: LIA) -> Self {
        match op {
            LIA::Int(op) => Any::Int(op.into()),
            LIA::Bool(op) => Any::Bool(op),
        }
    }
}

impl TryFrom<Any> for LIA {
    type Error = String;

    fn try_from(op: Any) -> Result<Self, Self::Error> {
        match op {
            Any::Int(op) => op
                .try_into()
                .map(LIA::Int)
                .map_err(|_| "LIA holds the linear integer operations only".to_string()),
            Any::Bool(op) => Ok(LIA::Bool(op)),
            _ => Err("LIA holds integer and boolean operations only".to_string()),
        }
    }
}

// The embeddings between logics, which `subenum` cannot express: it casts
// between `Any` and each logic, but not between two of them. `Prop` embeds
// into every other logic, and each linear logic into its non-linear one.
// Their inverses are absent, as nothing casts down between logics.
impl From<Prop> for NRA {
    fn from(Prop::Bool(op): Prop) -> Self {
        NRA::Bool(op)
    }
}

impl From<Prop> for NIA {
    fn from(Prop::Bool(op): Prop) -> Self {
        NIA::Bool(op)
    }
}

impl From<Prop> for LRA {
    fn from(Prop::Bool(op): Prop) -> Self {
        LRA::Bool(op)
    }
}

impl From<Prop> for LIA {
    fn from(Prop::Bool(op): Prop) -> Self {
        LIA::Bool(op)
    }
}

impl From<Prop> for BV {
    fn from(Prop::Bool(op): Prop) -> Self {
        BV::Bool(op)
    }
}

impl From<LRA> for NRA {
    fn from(op: LRA) -> Self {
        match op {
            LRA::Real(op) => NRA::Real(op.into()),
            LRA::Bool(op) => NRA::Bool(op),
        }
    }
}

impl From<LIA> for NIA {
    fn from(op: LIA) -> Self {
        match op {
            LIA::Int(op) => NIA::Int(op.into()),
            LIA::Bool(op) => NIA::Bool(op),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn real(op: impl Into<real::NRA>) -> Any {
        Any::Real(op.into())
    }

    fn int(op: impl Into<int::NIA>) -> Any {
        Any::Int(op.into())
    }

    fn boolean(op: impl Into<bool::Prop>) -> Any {
        Any::Bool(op.into())
    }

    fn word(op: impl Into<word::BV>) -> Any {
        Any::Word(op.into())
    }

    #[test]
    fn any_boundary_is_each_operations_boundary_in_sort() {
        let op = real(real::MatMul(2, 3, 4));
        let dom: Poly<sort::Any> = base::mono![
            sort::Real::new([2, 3]).into(),
            sort::Real::new([3, 4]).into()
        ]
        .into();
        let cod: Poly<sort::Any> = base::mono![sort::Real::new([2, 4]).into()].into();
        assert_eq!(op.dom(), dom);
        assert_eq!(op.cod(), cod);

        let op = int(int::Transpose(2, 3));
        let cod: Poly<sort::Any> = base::mono![sort::Int::new([3, 2]).into()].into();
        assert_eq!(op.cod(), cod);

        let op = boolean(bool::Not(2, 3));
        let dom: Poly<sort::Any> = base::mono![sort::Bool::new([2, 3]).into()].into();
        assert_eq!(op.dom(), dom);
    }

    #[test]
    fn theories_mark_their_linear_operations() {
        assert!(real::LRA::try_from(real::NRA::from(real::Add(2, 3))).is_ok());
        assert!(real::LRA::try_from(real::NRA::from(real::Lt(2, 3))).is_ok());
        assert!(real::LRA::try_from(real::NRA::from(real::Mul(2, 3))).is_err());
        assert!(real::LRA::try_from(real::NRA::from(real::MatMul(2, 3, 4))).is_err());
        assert!(int::LIA::try_from(int::NIA::from(int::Mod(2, 3))).is_err());
        assert!(int::LIA::try_from(int::NIA::from(int::ReLU(2, 3))).is_ok());
    }

    #[test]
    fn logics_cast_to_and_from_any() {
        // operations wrap into their logic, and logics into `Any`
        let lra = LRA::Real(real::Add(2, 3).into());
        assert!(matches!(
            Any::from(lra),
            Any::Real(real::NRA::Add(real::Add(2, 3)))
        ));
        let lia = LIA::Int(int::Sub(2, 3).into());
        assert!(matches!(
            Any::from(lia),
            Any::Int(int::NIA::Sub(int::Sub(2, 3)))
        ));

        // `Any` casts down to the logics holding the operation only
        assert!(LRA::try_from(real(real::ReLU(2, 3))).is_ok());
        assert!(LIA::try_from(real(real::ReLU(2, 3))).is_err());
        assert!(LIA::try_from(int(int::ReLU(2, 3))).is_ok());

        // boolean operations are in every logic
        let op = boolean(bool::And(2, 3));
        assert!(Prop::try_from(op.clone()).is_ok());
        assert!(NRA::try_from(op.clone()).is_ok());
        assert!(NIA::try_from(op.clone()).is_ok());
        assert!(LRA::try_from(op.clone()).is_ok());
        assert!(LIA::try_from(op.clone()).is_ok());
        assert!(BV::try_from(op).is_ok());

        // `Prop` holds the boolean operations only
        assert!(Prop::try_from(real(real::Add(2, 3))).is_err());
    }

    #[test]
    fn linear_logics_hold_the_linear_operations_only() {
        // `Mul`, `Div`, `Mod` and `MatMul` are in neither linear logic
        assert!(LRA::try_from(real(real::MatMul(2, 3, 4))).is_err());
        assert!(LRA::try_from(real(real::Mul(2, 3))).is_err());
        assert!(LRA::try_from(real(real::Div(2, 3))).is_err());
        assert!(LIA::try_from(int(int::MatMul(2, 3, 4))).is_err());
        assert!(LIA::try_from(int(int::Mul(2, 3))).is_err());
        assert!(LIA::try_from(int(int::Div(2, 3))).is_err());
        assert!(LIA::try_from(int(int::Mod(2, 3))).is_err());

        // but they are in the non-linear ones
        assert!(NRA::try_from(real(real::MatMul(2, 3, 4))).is_ok());
        assert!(NRA::try_from(real(real::Div(2, 3))).is_ok());
        assert!(NIA::try_from(int(int::Mod(2, 3))).is_ok());

        // comparisons are linear
        assert!(LRA::try_from(real(real::Lt(2, 3))).is_ok());
        assert!(LIA::try_from(int(int::Ge(2, 3))).is_ok());
    }

    #[test]
    fn non_linear_logics_hold_their_theory_only() {
        assert!(NRA::try_from(int(int::Add(2, 3))).is_err());
        assert!(NIA::try_from(real(real::Add(2, 3))).is_err());
        assert!(NRA::try_from(word(word::Add(8, 2, 3))).is_err());
    }

    #[test]
    fn logics_embed_into_larger_logics() {
        let lra = LRA::Real(real::Sub(2, 3).into());
        assert!(matches!(
            NRA::from(lra),
            NRA::Real(real::NRA::Sub(real::Sub(2, 3)))
        ));
        let lia = LIA::Int(int::Max(2, 3).into());
        assert!(matches!(
            NIA::from(lia),
            NIA::Int(int::NIA::Max(int::Max(2, 3)))
        ));

        let prop = Prop::Bool(bool::Or(2, 3).into());
        assert!(matches!(
            NRA::from(prop.clone()),
            NRA::Bool(bool::Prop::Or(bool::Or(2, 3)))
        ));
        assert!(matches!(NIA::from(prop.clone()), NIA::Bool(_)));
        assert!(matches!(LRA::from(prop.clone()), LRA::Bool(_)));
        assert!(matches!(LIA::from(prop.clone()), LIA::Bool(_)));
        assert!(matches!(BV::from(prop), BV::Bool(_)));
    }

    #[test]
    fn bv_holds_word_and_boolean_operations() {
        assert!(BV::try_from(word(word::signed::Div(8, 2, 3))).is_ok());
        assert!(BV::try_from(word(word::unsigned::Lt(8, 2, 3))).is_ok());
        assert!(BV::try_from(word(word::Concat(8, 4, 2, 3))).is_ok());
        assert!(BV::try_from(int(int::Add(2, 3))).is_err());
        assert!(BV::try_from(real(real::Add(2, 3))).is_err());

        // word operations are in none of the other logics
        assert!(LIA::try_from(word(word::Add(8, 2, 3))).is_err());
        assert!(Prop::try_from(word(word::Not(8, 2, 3))).is_err());
    }

    #[test]
    fn word_ops_have_their_boundary_in_sort() {
        let op = word(word::Add(8, 2, 3));
        let w: sort::Any = sort::Word::new_unchecked(8, [2, 3]).into();
        let dom: Poly<sort::Any> = base::mono![w, w].into();
        let cod: Poly<sort::Any> = base::mono![w].into();
        assert_eq!(op.dom(), dom);
        assert_eq!(op.cod(), cod);

        // comparisons write the booleans of `bool`
        let op = word(word::signed::Lt(8, 2, 3));
        let cod: Poly<sort::Any> = base::mono![sort::Bool::new([2, 3]).into()].into();
        assert_eq!(op.dom(), dom);
        assert_eq!(op.cod(), cod);

        // signed and unsigned operations are distinct operations
        assert!(matches!(
            word::BV::from(word::signed::Div(8, 2, 3)),
            word::BV::SDiv(_)
        ));
        assert!(matches!(
            word::BV::from(word::unsigned::Div(8, 2, 3)),
            word::BV::UDiv(_)
        ));
    }

    #[test]
    fn comparisons_read_numbers_and_write_booleans() {
        let op = int(int::Lt(2, 3));
        let i: sort::Any = sort::Int::new([2, 3]).into();
        let dom: Poly<sort::Any> = base::mono![i, i].into();
        let cod: Poly<sort::Any> = base::mono![sort::Bool::new([2, 3]).into()].into();
        assert_eq!(op.dom(), dom);
        assert_eq!(op.cod(), cod);
    }

    #[test]
    fn id_is_in_the_logics_of_its_theory() {
        assert!(LRA::try_from(real(real::Id(2, 3))).is_ok());
        assert!(LIA::try_from(real(real::Id(2, 3))).is_err());
        assert!(LIA::try_from(int(int::Id(2, 3))).is_ok());
        assert!(BV::try_from(word(word::Id(8, 2, 3))).is_ok());
        assert!(Prop::try_from(boolean(bool::Id(2, 3))).is_ok());
    }
}
