/*!
# Tensors

The tensor operations of each theory, one type per operation, in a module per
theory: [`real`] and [`int`] for arithmetic, [`bool`](mod@bool) for
propositional logic, and [`word`] for fixed-size bit-vectors. Each
operation reads and writes the sorts of [`sort`], and computes its
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

The operations that carry data, the constants and the coefficients of the
affine maps, hold it in matrices of a type `T` implementing [`Tensor`], over
which every operation, theory enum and logic is generic. [`PyTensor`](crate::PyTensor) is
the tensor that converts to and from Python `torch.Tensor` objects with the
`torch` feature, and a placeholder without it.
*/

pub mod bool;
pub mod int;
pub mod real;
pub mod sort;
pub mod word;

use base::boundary::{Boundary, Poly};
use derive_more::From;
use subenum::subenum;

/// A matrix of values, as held by the operations that carry data: the
/// constants, and the coefficients of `Linear`. The operations are generic
/// over it, so that any tensor representation can back them.
pub trait Tensor {
    /// The `[rows, cols]` shape of the matrix.
    fn shape(&self) -> [usize; 2];
}

/// The operations of all theories: a value of `Any` is an operation of one
/// of them, on the sorts in [`sort::Any`], with its data held in tensors of
/// type `T`.
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
pub enum Any<T> {
    #[subenum(NRA)]
    Real(real::NRA<T>),
    #[subenum(NIA)]
    Int(int::NIA<T>),
    #[subenum(NRA, NIA, Prop, BV)]
    Bool(bool::Prop<T>),
    #[subenum(BV)]
    Word(word::BV<T>),
}

/// Linear real arithmetic: the linear operations of [`real`], with the
/// boolean operations.
#[derive(Clone, Debug, From)]
pub enum LRA<T> {
    Real(real::LRA<T>),
    Bool(bool::Prop<T>),
}

/// Linear integer arithmetic: the linear operations of [`int`], with the
/// boolean operations.
#[derive(Clone, Debug, From)]
pub enum LIA<T> {
    Int(int::LIA<T>),
    Bool(bool::Prop<T>),
}

impl<T: Tensor> Boundary<sort::Any> for Any<T> {
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

/// Extracts an operation of any theory from a Python object, trying each
/// theory in turn through its own extraction: real, integer, boolean, and
/// then bit-vector operations. The operations of each theory are distinct
/// Python classes, so at most one theory accepts a given object.
#[cfg(feature = "pyo3")]
impl<'py, T> pyo3::FromPyObject<'py> for Any<T>
where
    real::NRA<T>: pyo3::FromPyObject<'py>,
    int::NIA<T>: pyo3::FromPyObject<'py>,
    bool::Prop<T>: pyo3::FromPyObject<'py>,
    word::BV<T>: pyo3::FromPyObject<'py>,
{
    fn extract_bound(ob: &pyo3::Bound<'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use pyo3::types::PyAnyMethods;

        if let Ok(op) = ob.extract::<real::NRA<T>>() {
            return Ok(Any::Real(op));
        }
        if let Ok(op) = ob.extract::<int::NIA<T>>() {
            return Ok(Any::Int(op));
        }
        if let Ok(op) = ob.extract::<bool::Prop<T>>() {
            return Ok(Any::Bool(op));
        }
        if let Ok(op) = ob.extract::<word::BV<T>>() {
            return Ok(Any::Word(op));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "expected an operation of real, integer, boolean or bit-vector \
             arithmetic, got {}",
            ob.get_type()
        )))
    }
}

impl<T> From<LRA<T>> for Any<T> {
    fn from(op: LRA<T>) -> Self {
        match op {
            LRA::Real(op) => Any::Real(op.into()),
            LRA::Bool(op) => Any::Bool(op),
        }
    }
}

impl<T> TryFrom<Any<T>> for LRA<T> {
    type Error = String;

    fn try_from(op: Any<T>) -> Result<Self, Self::Error> {
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

impl<T> From<LIA<T>> for Any<T> {
    fn from(op: LIA<T>) -> Self {
        match op {
            LIA::Int(op) => Any::Int(op.into()),
            LIA::Bool(op) => Any::Bool(op),
        }
    }
}

impl<T> TryFrom<Any<T>> for LIA<T> {
    type Error = String;

    fn try_from(op: Any<T>) -> Result<Self, Self::Error> {
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
impl<T> From<Prop<T>> for NRA<T> {
    fn from(Prop::Bool(op): Prop<T>) -> Self {
        NRA::Bool(op)
    }
}

impl<T> From<Prop<T>> for NIA<T> {
    fn from(Prop::Bool(op): Prop<T>) -> Self {
        NIA::Bool(op)
    }
}

impl<T> From<Prop<T>> for LRA<T> {
    fn from(Prop::Bool(op): Prop<T>) -> Self {
        LRA::Bool(op)
    }
}

impl<T> From<Prop<T>> for LIA<T> {
    fn from(Prop::Bool(op): Prop<T>) -> Self {
        LIA::Bool(op)
    }
}

impl<T> From<Prop<T>> for BV<T> {
    fn from(Prop::Bool(op): Prop<T>) -> Self {
        BV::Bool(op)
    }
}

impl<T> From<LRA<T>> for NRA<T> {
    fn from(op: LRA<T>) -> Self {
        match op {
            LRA::Real(op) => NRA::Real(op.into()),
            LRA::Bool(op) => NRA::Bool(op),
        }
    }
}

impl<T> From<LIA<T>> for NIA<T> {
    fn from(op: LIA<T>) -> Self {
        match op {
            LIA::Int(op) => NIA::Int(op.into()),
            LIA::Bool(op) => NIA::Bool(op),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tensor carrying its shape only, enough to compute boundaries.
    #[derive(Clone, Debug)]
    struct Mat([usize; 2]);

    impl Tensor for Mat {
        fn shape(&self) -> [usize; 2] {
            self.0
        }
    }

    // The operations and logics, instantiated with `Mat`; these shadow the
    // generic ones of `super`.
    type Any = super::Any<Mat>;
    #[allow(clippy::upper_case_acronyms)]
    type NRA = super::NRA<Mat>;
    #[allow(clippy::upper_case_acronyms)]
    type NIA = super::NIA<Mat>;
    #[allow(clippy::upper_case_acronyms)]
    type LRA = super::LRA<Mat>;
    #[allow(clippy::upper_case_acronyms)]
    type LIA = super::LIA<Mat>;
    type Prop = super::Prop<Mat>;
    type BV = super::BV<Mat>;

    fn real(op: impl Into<real::NRA<Mat>>) -> Any {
        Any::Real(op.into())
    }

    fn int(op: impl Into<int::NIA<Mat>>) -> Any {
        Any::Int(op.into())
    }

    fn boolean(op: impl Into<bool::Prop<Mat>>) -> Any {
        Any::Bool(op.into())
    }

    fn word(op: impl Into<word::BV<Mat>>) -> Any {
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
    fn data_operations_take_their_shape_from_their_tensors() {
        let c = real(real::Const(Mat([2, 3])));
        assert_eq!(c.dom(), Poly::unit());
        let cod: Poly<sort::Any> = base::mono![sort::Real::new([2, 3]).into()].into();
        assert_eq!(c.cod(), cod);

        // `A` of shape `[2, 3]` and `B` of shape `[2, 4]` take `[3, 4]` to `[2, 4]`
        let l = int(int::Linear(Mat([2, 3]), Mat([2, 4])));
        let dom: Poly<sort::Any> = base::mono![sort::Int::new([3, 4]).into()].into();
        let cod: Poly<sort::Any> = base::mono![sort::Int::new([2, 4]).into()].into();
        assert_eq!(l.dom(), dom);
        assert_eq!(l.cod(), cod);

        let w = word(word::Const(Mat([1, 2]), 8));
        let cod: Poly<sort::Any> = base::mono![sort::Word::new_unchecked(8, [1, 2]).into()].into();
        assert_eq!(w.cod(), cod);

        // constants are linear
        assert!(LRA::try_from(c).is_ok());
        assert!(LIA::try_from(l).is_ok());
    }

    #[test]
    fn theories_mark_their_linear_operations() {
        assert!(real::LRA::try_from(real::NRA::<Mat>::from(real::Add(2, 3))).is_ok());
        assert!(real::LRA::try_from(real::NRA::<Mat>::from(real::Lt(2, 3))).is_ok());
        assert!(real::LRA::try_from(real::NRA::<Mat>::from(real::Mul(2, 3))).is_err());
        assert!(real::LRA::try_from(real::NRA::<Mat>::from(real::MatMul(2, 3, 4))).is_err());
        assert!(int::LIA::try_from(int::NIA::<Mat>::from(int::Mod(2, 3))).is_err());
        assert!(int::LIA::try_from(int::NIA::<Mat>::from(int::ReLU(2, 3))).is_ok());
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
            word::BV::<Mat>::from(word::signed::Div(8, 2, 3)),
            word::BV::SDiv(_)
        ));
        assert!(matches!(
            word::BV::<Mat>::from(word::unsigned::Div(8, 2, 3)),
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
}

#[cfg(all(test, feature = "pyo3"))]
mod py_tests {
    use super::*;
    use pyo3::prelude::*;

    /// A tensor type with no Python form: the operations carrying it never
    /// extract, so that only those carrying no data do.
    #[derive(Clone, Debug)]
    struct NoData;

    fn no_data<T>(ob: &Bound<'_, PyAny>) -> PyResult<T> {
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "no tensor in {}",
            ob.get_type()
        )))
    }

    impl<'py> FromPyObject<'py> for real::Const<NoData> {
        fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
            no_data(ob)
        }
    }

    impl<'py> FromPyObject<'py> for real::Linear<NoData> {
        fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
            no_data(ob)
        }
    }

    impl<'py> FromPyObject<'py> for int::Const<NoData> {
        fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
            no_data(ob)
        }
    }

    impl<'py> FromPyObject<'py> for int::Linear<NoData> {
        fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
            no_data(ob)
        }
    }

    impl<'py> FromPyObject<'py> for bool::Const<NoData> {
        fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
            no_data(ob)
        }
    }

    impl<'py> FromPyObject<'py> for word::Const<NoData> {
        fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
            no_data(ob)
        }
    }

    #[test]
    fn any_extracts_the_operations_of_each_theory() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let real_add = Bound::new(py, real::Add(2, 3)).unwrap();
            let op = real_add.extract::<Any<NoData>>().unwrap();
            assert!(matches!(op, Any::Real(real::NRA::Add(real::Add(2, 3)))));

            let int_add = Bound::new(py, int::Add(2, 3)).unwrap();
            let op = int_add.extract::<Any<NoData>>().unwrap();
            assert!(matches!(op, Any::Int(int::NIA::Add(int::Add(2, 3)))));

            let and = Bound::new(py, bool::And(2, 3)).unwrap();
            let op = and.extract::<Any<NoData>>().unwrap();
            assert!(matches!(op, Any::Bool(bool::Prop::And(bool::And(2, 3)))));

            let sdiv = Bound::new(py, word::signed::Div(8, 2, 3)).unwrap();
            let op = sdiv.extract::<Any<NoData>>().unwrap();
            assert!(matches!(
                op,
                Any::Word(word::BV::SDiv(word::signed::Div(8, 2, 3)))
            ));

            let not_an_operation = 42i64.into_pyobject(py).unwrap();
            assert!(not_an_operation.extract::<Any<NoData>>().is_err());
        });
    }

    #[test]
    fn each_theory_extracts_its_own_operations_only() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let real_mul = Bound::new(py, real::Mul(2, 3)).unwrap();
            assert!(real_mul.extract::<real::NRA<NoData>>().is_ok());
            assert!(real_mul.extract::<int::NIA<NoData>>().is_err());

            let int_mod = Bound::new(py, int::Mod(2, 3)).unwrap();
            assert!(int_mod.extract::<int::NIA<NoData>>().is_ok());

            // `Not` is a different class in propositional logic and in QF_BV
            let not = Bound::new(py, bool::Not(2, 3)).unwrap();
            assert!(not.extract::<bool::Prop<NoData>>().is_ok());
            assert!(not.extract::<word::BV<NoData>>().is_err());
        });
    }
}
