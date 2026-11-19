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

The constants and the affine maps hold their data in
[`Tensor`](crate::Tensor)s, of any backend.

With the `pyo3` feature, each operation is also a Python class, in `py`.
*/

pub mod bool;
pub mod int;
#[cfg(feature = "pyo3")]
pub mod py;
pub mod real;
pub mod word;

use crate::sort;
use crate::{Combinatorial, Differential, Sequential};
use base::boundary::{Boundary, Poly};
use derive_more::{Display, From};
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
#[derive(Clone, Debug, From, Display)]
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
#[derive(Clone, Debug, From, Display)]
pub enum LRA {
    Real(real::LRA),
    Bool(bool::Prop),
}

/// Linear integer arithmetic: the linear operations of [`int`], with the
/// boolean operations.
#[derive(Clone, Debug, From, Display)]
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

// `Any` and its logics choose, copy and zero the values of `sort::Any`. A
// logic gives the operation that `Any` gives, on the sorts of its theories
// only: `NRA` and `LRA` on `sort::Reals`, `NIA` and `LIA` on `sort::Ints`,
// `Prop` on `sort::Prop`, and `BV` on `sort::Words`. It panics on any other
// sort.

/// The operation `op` of `Any`, its `what` on `range`, as one of the logic
/// `L`, named `logic`.
///
/// # Panics
///
/// If `L` does not hold `op`, as `range` is not one of its sorts.
fn restrict<L: TryFrom<Any>>(op: Any, logic: &str, what: &str, range: &sort::Any) -> L {
    L::try_from(op).unwrap_or_else(|_| panic!("{logic} has no {what} on {range}"))
}

/// Each theory's `Havoc`, and `bool::Zero` on `Zero`, whose only value is
/// the arbitrary one.
///
/// # Panics
///
/// On reals of degree above 0, rates of change: no operation chooses one.
impl Combinatorial<sort::Any> for Any {
    fn havoc(range: &sort::Any) -> Self {
        match *range {
            sort::Any::Real(r) if r.degree() == 0 => {
                let [m, n] = r.shape();
                Any::Real(real::Havoc(m, n).into())
            }
            sort::Any::Real(_) => panic!("Any has no havoc on {range}"),
            sort::Any::Int(i) => {
                let [m, n] = i.shape();
                Any::Int(int::Havoc(m, n).into())
            }
            sort::Any::Bool(b) => {
                let [m, n] = b.shape();
                Any::Bool(bool::Havoc(m, n).into())
            }
            sort::Any::Word(w) => {
                let [m, n] = w.shape();
                Any::Word(word::Havoc(w.width(), m, n).into())
            }
            sort::Any::Zero(_) => Any::Bool(bool::Zero.into()),
        }
    }
}

impl Combinatorial<sort::Any> for NRA {
    fn havoc(range: &sort::Any) -> Self {
        restrict(Any::havoc(range), "NRA", "havoc", range)
    }
}

impl Combinatorial<sort::Any> for LRA {
    fn havoc(range: &sort::Any) -> Self {
        restrict(Any::havoc(range), "LRA", "havoc", range)
    }
}

impl Combinatorial<sort::Any> for NIA {
    fn havoc(range: &sort::Any) -> Self {
        restrict(Any::havoc(range), "NIA", "havoc", range)
    }
}

impl Combinatorial<sort::Any> for LIA {
    fn havoc(range: &sort::Any) -> Self {
        restrict(Any::havoc(range), "LIA", "havoc", range)
    }
}

impl Combinatorial<sort::Any> for Prop {
    fn havoc(range: &sort::Any) -> Self {
        restrict(Any::havoc(range), "Prop", "havoc", range)
    }
}

impl Combinatorial<sort::Any> for BV {
    fn havoc(range: &sort::Any) -> Self {
        restrict(Any::havoc(range), "BV", "havoc", range)
    }
}

/// Each theory's `Id`, and `real::TId` on reals of degree above 0, rates of
/// change.
///
/// # Panics
///
/// On `Zero`: no operation copies its only value.
impl Sequential<sort::Any> for Any {
    fn skip(range: &sort::Any) -> Self {
        match *range {
            sort::Any::Real(r) => {
                let [m, n] = r.shape();
                match r.degree() {
                    0 => Any::Real(real::Id(m, n).into()),
                    d => Any::Real(real::TId(m, n, d).into()),
                }
            }
            sort::Any::Int(i) => {
                let [m, n] = i.shape();
                Any::Int(int::Id(m, n).into())
            }
            sort::Any::Bool(b) => {
                let [m, n] = b.shape();
                Any::Bool(bool::Id(m, n).into())
            }
            sort::Any::Word(w) => {
                let [m, n] = w.shape();
                Any::Word(word::Id(w.width(), m, n).into())
            }
            sort::Any::Zero(_) => panic!("Any has no skip on {range}"),
        }
    }
}

impl Sequential<sort::Any> for NRA {
    fn skip(range: &sort::Any) -> Self {
        restrict(Any::skip(range), "NRA", "skip", range)
    }
}

impl Sequential<sort::Any> for LRA {
    fn skip(range: &sort::Any) -> Self {
        restrict(Any::skip(range), "LRA", "skip", range)
    }
}

impl Sequential<sort::Any> for NIA {
    fn skip(range: &sort::Any) -> Self {
        restrict(Any::skip(range), "NIA", "skip", range)
    }
}

impl Sequential<sort::Any> for LIA {
    fn skip(range: &sort::Any) -> Self {
        restrict(Any::skip(range), "LIA", "skip", range)
    }
}

impl Sequential<sort::Any> for Prop {
    fn skip(range: &sort::Any) -> Self {
        restrict(Any::skip(range), "Prop", "skip", range)
    }
}

impl Sequential<sort::Any> for BV {
    fn skip(range: &sort::Any) -> Self {
        restrict(Any::skip(range), "BV", "skip", range)
    }
}

/// `real::Zero` on reals, of the same degree, and `bool::Zero` on `Zero`,
/// the sort of the rates of change of the discrete sorts.
///
/// # Panics
///
/// On `Int`, `Bool` and `Word`: their rates of change are of sort `Zero`, so
/// no derivative wire is of these sorts.
impl Differential<sort::Any> for Any {
    fn zero(range: &sort::Any) -> Self {
        match *range {
            sort::Any::Real(r) => {
                let [m, n] = r.shape();
                Any::Real(real::Zero(m, n, r.degree()).into())
            }
            sort::Any::Zero(_) => Any::Bool(bool::Zero.into()),
            sort::Any::Int(_) | sort::Any::Bool(_) | sort::Any::Word(_) => {
                panic!("Any has no zero on {range}")
            }
        }
    }
}

impl Differential<sort::Any> for NRA {
    fn zero(range: &sort::Any) -> Self {
        restrict(Any::zero(range), "NRA", "zero", range)
    }
}

impl Differential<sort::Any> for LRA {
    fn zero(range: &sort::Any) -> Self {
        restrict(Any::zero(range), "LRA", "zero", range)
    }
}

impl Differential<sort::Any> for NIA {
    fn zero(range: &sort::Any) -> Self {
        restrict(Any::zero(range), "NIA", "zero", range)
    }
}

impl Differential<sort::Any> for LIA {
    fn zero(range: &sort::Any) -> Self {
        restrict(Any::zero(range), "LIA", "zero", range)
    }
}

impl Differential<sort::Any> for Prop {
    fn zero(range: &sort::Any) -> Self {
        restrict(Any::zero(range), "Prop", "zero", range)
    }
}

impl Differential<sort::Any> for BV {
    fn zero(range: &sort::Any) -> Self {
        restrict(Any::zero(range), "BV", "zero", range)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base::differentiable::T;

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
    fn operations_display_as_themselves_in_any_enum() {
        assert_eq!(real(real::Add(2, 3)).to_string(), "Add");
        assert_eq!(
            int(int::Uninterpreted("f".into(), 2, 3)).to_string(),
            "Uninterpreted(f)"
        );
        assert_eq!(boolean(bool::Zero).to_string(), "Zero");
        assert_eq!(word(word::unsigned::Lt(8, 2, 3)).to_string(), "ULt");
        assert_eq!(
            word(word::Extract(7, 4, 8, 2, 3)).to_string(),
            "Extract[7:4]"
        );
        assert_eq!(LRA::Real(real::TAdd(2, 3, 1).into()).to_string(), "TAdd");
        assert_eq!(LIA::Bool(bool::Not(2, 3).into()).to_string(), "Not");
        assert_eq!(
            NRA::try_from(real(real::Mul(2, 3))).unwrap().to_string(),
            "Mul"
        );
    }

    #[test]
    fn data_operations_take_their_shape_from_their_tensors() {
        use crate::tensor::VecTensor;
        let floats = |[m, n]: [usize; 2]| -> crate::Tensor {
            VecTensor::floats([m, n], vec![vec![0.0; n]; m])
                .unwrap()
                .into()
        };
        let ints = |[m, n]: [usize; 2]| -> crate::Tensor {
            VecTensor::ints([m, n], vec![vec![0; n]; m]).unwrap().into()
        };
        let c = real(real::Const(floats([2, 3])));
        assert_eq!(c.dom(), Poly::unit());
        let cod: Poly<sort::Any> = base::mono![sort::Real::new([2, 3]).into()].into();
        assert_eq!(c.cod(), cod);

        // `A` of shape `[2, 3]` and `B` of shape `[2, 4]` take `[3, 4]` to `[2, 4]`
        let l = int(int::Linear(ints([2, 3]), ints([2, 4])));
        let dom: Poly<sort::Any> = base::mono![sort::Int::new([3, 4]).into()].into();
        let cod: Poly<sort::Any> = base::mono![sort::Int::new([2, 4]).into()].into();
        assert_eq!(l.dom(), dom);
        assert_eq!(l.cod(), cod);

        let w = word(word::Const(ints([1, 2]), 8));
        let cod: Poly<sort::Any> = base::mono![sort::Word::new_unchecked(8, [1, 2]).into()].into();
        assert_eq!(w.cod(), cod);

        // constants are linear
        assert!(LRA::try_from(c).is_ok());
        assert!(LIA::try_from(l).is_ok());
    }

    #[test]
    fn data_operations_check_the_kind_and_shape_of_their_tensors() {
        use crate::tensor::{Error, Kind, VecTensor};
        let floats = |[m, n]: [usize; 2]| -> crate::Tensor {
            VecTensor::floats([m, n], vec![vec![0.0; n]; m])
                .unwrap()
                .into()
        };
        let ints = |[m, n]: [usize; 2]| -> crate::Tensor {
            VecTensor::ints([m, n], vec![vec![0; n]; m]).unwrap().into()
        };

        assert!(real::Const::new(floats([2, 3])).is_ok());
        assert_eq!(
            real::Const::new(ints([2, 3])).unwrap_err(),
            Error::Kind {
                expected: Kind::Float,
                got: Kind::Int
            }
        );
        assert!(int::Linear::new(ints([4, 2]), ints([4, 3])).is_ok());
        assert_eq!(
            int::Linear::new(ints([4, 2]), ints([3, 3])).unwrap_err(),
            Error::Affine {
                a: [4, 2],
                b: [3, 3]
            }
        );
        assert!(real::TLinear::new(floats([4, 2]), ints([4, 3]), 1).is_err());
        assert!(word::Const::new(ints([1, 1]), 8).is_ok());
        assert!(bool::Const::new(ints([1, 1])).is_err());
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

    #[test]
    fn ite_is_in_the_logics_of_its_theory() {
        let op = real(real::Ite(2, 3));
        let x: sort::Any = sort::Real::new([2, 3]).into();
        let dom: Poly<sort::Any> = base::mono![sort::Bool::new([1, 1]).into(), x, x].into();
        assert_eq!(op.dom(), dom);

        assert!(LRA::try_from(real(real::Ite(2, 3))).is_ok());
        assert!(LIA::try_from(int(int::Ite(2, 3))).is_ok());
        assert!(BV::try_from(word(word::Ite(8, 2, 3))).is_ok());
        assert!(Prop::try_from(boolean(bool::Ite(2, 3))).is_ok());
    }

    #[test]
    fn argmax_is_in_the_linear_logic_of_its_theory() {
        assert!(LRA::try_from(real(real::Argmax(2, 3))).is_ok());
        assert!(LIA::try_from(int(int::Argmax(2, 3))).is_ok());
        assert!(LIA::try_from(real(real::Argmax(2, 3))).is_err());
    }

    #[test]
    fn havoc_is_in_the_logics_of_its_theory() {
        let op = int(int::Havoc(2, 3));
        assert_eq!(op.dom(), Poly::unit());
        let cod: Poly<sort::Any> = base::mono![sort::Int::new([2, 3]).into()].into();
        assert_eq!(op.cod(), cod);

        assert!(LRA::try_from(real(real::Havoc(2, 3))).is_ok());
        assert!(LIA::try_from(int(int::Havoc(2, 3))).is_ok());
        assert!(BV::try_from(word(word::Havoc(8, 2, 3))).is_ok());
        assert!(Prop::try_from(boolean(bool::Havoc(2, 3))).is_ok());
    }

    #[test]
    fn uninterpreted_is_in_the_logics_of_its_theory() {
        let op = word(word::Uninterpreted("f".to_string(), 8, 2, 3));
        assert_eq!(op.dom(), Poly::unit());
        let cod: Poly<sort::Any> = base::mono![sort::Word::new_unchecked(8, [2, 3]).into()].into();
        assert_eq!(op.cod(), cod);

        let f = || "f".to_string();
        assert!(LRA::try_from(real(real::Uninterpreted(f(), 2, 3))).is_ok());
        assert!(LIA::try_from(int(int::Uninterpreted(f(), 2, 3))).is_ok());
        assert!(BV::try_from(word(word::Uninterpreted(f(), 8, 2, 3))).is_ok());
        assert!(Prop::try_from(boolean(bool::Uninterpreted(f(), 2, 3))).is_ok());
    }

    #[test]
    fn zeros_write_tangents() {
        let r = sort::Real::new([2, 3]);
        let op = real(real::Zero(2, 3, 1));
        assert_eq!(op.cod(), base::mono![sort::Any::from(T(r))].into());
        let op = boolean(bool::Zero);
        let i = sort::Int::new([2, 3]);
        assert_eq!(op.cod(), base::mono![T(sort::Any::from(i))].into());

        assert!(LRA::try_from(real(real::Zero(2, 3, 1))).is_ok());
        assert!(LIA::try_from(real(real::Zero(2, 3, 1))).is_err());
        // the zero of the discrete sorts is in every logic
        let op = boolean(bool::Zero);
        assert!(Prop::try_from(op.clone()).is_ok());
        assert!(NRA::try_from(op.clone()).is_ok());
        assert!(NIA::try_from(op.clone()).is_ok());
        assert!(LRA::try_from(op.clone()).is_ok());
        assert!(LIA::try_from(op.clone()).is_ok());
        assert!(BV::try_from(op).is_ok());
    }

    #[test]
    fn tangent_ops_are_in_lra() {
        let r = sort::Real::new([2, 3]);
        let op = real(real::TAdd(2, 3, 1));
        let dom: Poly<sort::Any> = base::mono![T(r).into(), T(r).into()].into();
        assert_eq!(op.dom(), dom);
        let op = real(real::TIte(2, 3, 1));
        let dom: Poly<sort::Any> =
            base::mono![sort::Bool::new([1, 1]).into(), T(r).into(), T(r).into()].into();
        assert_eq!(op.dom(), dom);

        assert!(LRA::try_from(real(real::TAdd(2, 3, 1))).is_ok());
        assert!(LRA::try_from(real(real::TId(2, 3, 1))).is_ok());
        assert!(LIA::try_from(real(real::TSub(2, 3, 1))).is_err());
    }

    fn sorts() -> (sort::Any, sort::Any, sort::Any, sort::Any, sort::Any) {
        (
            sort::Real::new([2, 3]).into(),
            sort::Int::new([2, 3]).into(),
            sort::Bool::new([2, 3]).into(),
            sort::Word::new_unchecked(8, [2, 3]).into(),
            sort::Zero.into(),
        )
    }

    #[test]
    fn havoc_chooses_a_value_of_its_range() {
        let (r, i, b, v, z) = sorts();
        for range in [r, i, b, v, z] {
            let op = Any::havoc(&range);
            assert_eq!(op.dom(), Poly::unit());
            assert_eq!(op.cod(), base::mono![range].into());
        }
        assert!(matches!(
            Any::havoc(&v),
            Any::Word(word::BV::Havoc(word::Havoc(8, 2, 3)))
        ));
        assert!(matches!(
            Any::havoc(&z),
            Any::Bool(bool::Prop::Zero(bool::Zero))
        ));
    }

    #[test]
    #[should_panic(expected = "Any has no havoc on T^1(Real(2, 3))")]
    fn havoc_panics_on_rates_of_change_of_reals() {
        let _ = Any::havoc(&T(sort::Any::from(sort::Real::new([2, 3]))));
    }

    #[test]
    fn logics_havoc_on_their_sorts() {
        let (r, i, b, v, z) = sorts();
        assert!(matches!(
            LRA::havoc(&r),
            LRA::Real(real::LRA::Havoc(real::Havoc(2, 3)))
        ));
        assert!(matches!(
            NRA::havoc(&r),
            NRA::Real(real::NRA::Havoc(real::Havoc(2, 3)))
        ));
        assert!(matches!(
            LRA::havoc(&b),
            LRA::Bool(bool::Prop::Havoc(bool::Havoc(2, 3)))
        ));
        assert!(matches!(
            LRA::havoc(&z),
            LRA::Bool(bool::Prop::Zero(bool::Zero))
        ));
        assert!(matches!(
            LIA::havoc(&i),
            LIA::Int(int::LIA::Havoc(int::Havoc(2, 3)))
        ));
        assert!(matches!(
            NIA::havoc(&i),
            NIA::Int(int::NIA::Havoc(int::Havoc(2, 3)))
        ));
        assert!(matches!(
            Prop::havoc(&b),
            Prop::Bool(bool::Prop::Havoc(bool::Havoc(2, 3)))
        ));
        assert!(matches!(
            BV::havoc(&v),
            BV::Word(word::BV::Havoc(word::Havoc(8, 2, 3)))
        ));
        assert!(matches!(
            BV::havoc(&b),
            BV::Bool(bool::Prop::Havoc(bool::Havoc(2, 3)))
        ));
    }

    #[test]
    #[should_panic(expected = "LRA has no havoc on Int(2, 3)")]
    fn logics_havoc_panics_on_other_sorts() {
        let _ = LRA::havoc(&sort::Int::new([2, 3]).into());
    }

    #[test]
    fn skip_copies_a_value_of_its_range() {
        let (r, i, b, v, _) = sorts();
        for range in [r, T(T(r)), i, b, v] {
            let op = Any::skip(&range);
            assert_eq!(op.dom(), base::mono![range].into());
            assert_eq!(op.cod(), base::mono![range].into());
        }
        assert!(matches!(
            Any::skip(&r),
            Any::Real(real::NRA::Id(real::Id(2, 3)))
        ));
        assert!(matches!(
            Any::skip(&T(T(r))),
            Any::Real(real::NRA::TId(real::TId(2, 3, 2)))
        ));
    }

    #[test]
    #[should_panic(expected = "Any has no skip on Zero")]
    fn skip_panics_on_zero() {
        let _ = Any::skip(&sort::Zero.into());
    }

    #[test]
    fn logics_skip_on_their_sorts() {
        let (r, i, b, v, _) = sorts();
        assert!(matches!(
            LRA::skip(&r),
            LRA::Real(real::LRA::Id(real::Id(2, 3)))
        ));
        assert!(matches!(
            LRA::skip(&T(r)),
            LRA::Real(real::LRA::TId(real::TId(2, 3, 1)))
        ));
        assert!(matches!(
            LRA::skip(&b),
            LRA::Bool(bool::Prop::Id(bool::Id(2, 3)))
        ));
        assert!(matches!(
            LIA::skip(&i),
            LIA::Int(int::LIA::Id(int::Id(2, 3)))
        ));
        assert!(matches!(
            Prop::skip(&b),
            Prop::Bool(bool::Prop::Id(bool::Id(2, 3)))
        ));
        assert!(matches!(
            BV::skip(&v),
            BV::Word(word::BV::Id(word::Id(8, 2, 3)))
        ));
    }

    #[test]
    #[should_panic(expected = "Prop has no skip on Real(2, 3)")]
    fn logics_skip_panics_on_other_sorts() {
        let _ = Prop::skip(&sort::Real::new([2, 3]).into());
    }

    #[test]
    fn zero_writes_the_rate_of_change_of_each_sort() {
        let (r, i, b, v, z) = sorts();
        for s in [r, i, b, v, z] {
            let op = Any::zero(&T(s));
            assert_eq!(op.dom(), Poly::unit());
            assert_eq!(op.cod(), base::mono![T(s)].into());
        }
        assert!(matches!(
            Any::zero(&T(r)),
            Any::Real(real::NRA::Zero(real::Zero(2, 3, 1)))
        ));
        assert!(matches!(
            Any::zero(&T(i)),
            Any::Bool(bool::Prop::Zero(bool::Zero))
        ));
    }

    #[test]
    fn zero_on_plain_reals_is_the_zero_matrix() {
        let (r, ..) = sorts();
        let op = Any::zero(&r);
        assert!(matches!(
            op,
            Any::Real(real::NRA::Zero(real::Zero(2, 3, 0)))
        ));
        assert_eq!(op.cod(), base::mono![r].into());
    }

    #[test]
    #[should_panic(expected = "Any has no zero on Int(2, 3)")]
    fn zero_panics_on_discrete_sorts() {
        let _ = Any::zero(&sort::Int::new([2, 3]).into());
    }

    #[test]
    fn logics_zero_on_their_sorts() {
        let (r, .., z) = sorts();
        assert!(matches!(
            LRA::zero(&T(r)),
            LRA::Real(real::LRA::Zero(real::Zero(2, 3, 1)))
        ));
        assert!(matches!(LRA::zero(&z), LRA::Bool(bool::Prop::Zero(_))));
        assert!(matches!(NIA::zero(&z), NIA::Bool(bool::Prop::Zero(_))));
        assert!(matches!(LIA::zero(&z), LIA::Bool(bool::Prop::Zero(_))));
        assert!(matches!(Prop::zero(&z), Prop::Bool(bool::Prop::Zero(_))));
        assert!(matches!(BV::zero(&z), BV::Bool(bool::Prop::Zero(_))));
    }

    #[test]
    #[should_panic(expected = "LIA has no zero on T^1(Real(2, 3))")]
    fn logics_zero_panics_on_other_sorts() {
        let _ = LIA::zero(&T(sort::Any::from(sort::Real::new([2, 3]))));
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod py_tests {
    use super::*;
    use pyo3::prelude::*;

    #[test]
    fn any_extracts_the_operations_of_each_theory() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let real_add = Bound::new(py, real::Add(2, 3)).unwrap();
            let op = real_add.extract::<Any>().unwrap();
            assert!(matches!(op, Any::Real(real::NRA::Add(real::Add(2, 3)))));

            let int_add = Bound::new(py, int::Add(2, 3)).unwrap();
            let op = int_add.extract::<Any>().unwrap();
            assert!(matches!(op, Any::Int(int::NIA::Add(int::Add(2, 3)))));

            let and = Bound::new(py, bool::And(2, 3)).unwrap();
            let op = and.extract::<Any>().unwrap();
            assert!(matches!(op, Any::Bool(bool::Prop::And(bool::And(2, 3)))));

            let sdiv = Bound::new(py, word::signed::Div(8, 2, 3)).unwrap();
            let op = sdiv.extract::<Any>().unwrap();
            assert!(matches!(
                op,
                Any::Word(word::BV::SDiv(word::signed::Div(8, 2, 3)))
            ));

            let not_an_operation = 42i64.into_pyobject(py).unwrap();
            assert!(not_an_operation.extract::<Any>().is_err());
        });
    }

    #[test]
    fn each_theory_extracts_its_own_operations_only() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let real_mul = Bound::new(py, real::Mul(2, 3)).unwrap();
            assert!(real_mul.extract::<real::NRA>().is_ok());
            assert!(real_mul.extract::<int::NIA>().is_err());

            let int_mod = Bound::new(py, int::Mod(2, 3)).unwrap();
            assert!(int_mod.extract::<int::NIA>().is_ok());

            // `Not` is a different class in propositional logic and in QF_BV
            let not = Bound::new(py, bool::Not(2, 3)).unwrap();
            assert!(not.extract::<bool::Prop>().is_ok());
            assert!(not.extract::<word::BV>().is_err());
        });
    }
}
