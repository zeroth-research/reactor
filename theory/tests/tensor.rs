//! Tests for the operations of `tensor`, one module per module of `tensor`.

/// Tests for `tensor::Any` and the logics.
mod any {
    use base::boundary::{Boundary, Poly};
    use theory::tensor::*;

    /// A tensor carrying its shape only, enough to compute boundaries.
    #[derive(Debug)]
    struct Mat([usize; 2]);

    impl Tensor for Mat {
        fn shape(&self) -> [usize; 2] {
            self.0
        }
    }

    // The operations and logics, instantiated with `Mat`; these shadow the
    // generic ones of `theory::tensor`.
    type Any = theory::tensor::Any<Mat>;
    #[allow(clippy::upper_case_acronyms)]
    type NRAT = theory::tensor::NRAT<Mat>;
    #[allow(clippy::upper_case_acronyms)]
    type NRA = theory::tensor::NRA<Mat>;
    #[allow(clippy::upper_case_acronyms)]
    type NIA = theory::tensor::NIA<Mat>;
    #[allow(clippy::upper_case_acronyms)]
    type LRA = theory::tensor::LRA<Mat>;
    #[allow(clippy::upper_case_acronyms)]
    type LIA = theory::tensor::LIA<Mat>;
    type Prop = theory::tensor::Prop<Mat>;
    type BV = theory::tensor::BV<Mat>;

    fn real(op: impl Into<real::NRAT<Mat>>) -> Any {
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
        let op = real(real::MatMul::new([[2, 3], [3, 4]]).unwrap());
        let dom: Poly<sort::Any> = base::mono![
            sort::Real::new([2, 3]).into(),
            sort::Real::new([3, 4]).into()
        ]
        .into();
        let cod: Poly<sort::Any> = base::mono![sort::Real::new([2, 4]).into()].into();
        assert_eq!(op.dom(), dom);
        assert_eq!(op.cod(), cod);

        let op = int(int::Transpose::new([2, 3]));
        let cod: Poly<sort::Any> = base::mono![sort::Int::new([3, 2]).into()].into();
        assert_eq!(op.cod(), cod);

        let op = boolean(bool::Not::new([2, 3]));
        let dom: Poly<sort::Any> = base::mono![sort::Bool::new([2, 3]).into()].into();
        assert_eq!(op.dom(), dom);
    }

    #[test]
    fn data_operations_take_their_shape_from_their_tensors() {
        let c = real(real::Const::new(Mat([2, 3])));
        assert_eq!(c.dom(), Poly::unit());
        let cod: Poly<sort::Any> = base::mono![sort::Real::new([2, 3]).into()].into();
        assert_eq!(c.cod(), cod);

        // `A` of shape `[2, 3]` and `B` of shape `[2, 4]` take `[3, 4]` to `[2, 4]`
        let l = int(int::Linear::new(Mat([2, 3]), Mat([2, 4])).unwrap());
        let dom: Poly<sort::Any> = base::mono![sort::Int::new([3, 4]).into()].into();
        let cod: Poly<sort::Any> = base::mono![sort::Int::new([2, 4]).into()].into();
        assert_eq!(l.dom(), dom);
        assert_eq!(l.cod(), cod);

        let w = word(word::Const::new(Mat([1, 2]), 8).unwrap());
        let cod: Poly<sort::Any> = base::mono![sort::Word::new_unchecked(8, [1, 2]).into()].into();
        assert_eq!(w.cod(), cod);

        // constants are linear
        assert!(LRA::try_from(c).is_ok());
        assert!(LIA::try_from(l).is_ok());
    }

    #[test]
    fn data_operations_check_their_tensors() {
        // `B` must have the rows of `A`
        assert!(real::Linear::new(Mat([2, 3]), Mat([3, 4])).is_err());
        assert!(int::Linear::new(Mat([2, 3]), Mat([3, 4])).is_err());
        assert!(real::Linear::new(Mat([0, 3]), Mat([0, 4])).is_ok());

        // words have at least one bit
        assert!(word::Const::new(Mat([1, 2]), 0).is_err());
        assert_eq!(word::Const::new(Mat([1, 2]), 8).unwrap().width(), 8);
    }

    #[test]
    fn data_operations_read_their_tensors() {
        let c = real::Const::new(Mat([2, 3]));
        assert_eq!(c.tensor().0, [2, 3]);
        let l = int::Linear::new(Mat([2, 3]), Mat([2, 4])).unwrap();
        assert_eq!(l.a().0, [2, 3]);
        assert_eq!(l.b().0, [2, 4]);
    }

    #[test]
    fn theories_mark_their_linear_operations() {
        assert!(real::LRA::try_from(real::NRA::<Mat>::from(real::Add::new([2, 3]))).is_ok());
        assert!(real::LRA::try_from(real::NRA::<Mat>::from(real::Lt::new([2, 3]))).is_ok());
        assert!(real::LRA::try_from(real::NRA::<Mat>::from(real::Mul::new([2, 3]))).is_err());
        assert!(
            real::LRA::try_from(real::NRA::<Mat>::from(
                real::MatMul::new([[2, 3], [3, 4]]).unwrap()
            ))
            .is_err()
        );
        assert!(int::LIA::try_from(int::NIA::<Mat>::from(int::Mod::new([2, 3]))).is_err());
        assert!(int::LIA::try_from(int::NIA::<Mat>::from(int::ReLU::new([2, 3]))).is_ok());

        // negation and absolute value are linear
        assert!(real::LRA::try_from(real::NRA::<Mat>::from(real::Neg::new([2, 3]))).is_ok());
        assert!(int::LIA::try_from(int::NIA::<Mat>::from(int::Neg::new([2, 3]))).is_ok());
        assert!(int::LIA::try_from(int::NIA::<Mat>::from(int::Abs::new([2, 3]))).is_ok());
        assert!(real::LRA::try_from(real::NRA::<Mat>::from(real::Abs::new([2, 3]))).is_ok());

        // the product with a constant is linear, unlike that of two variables
        assert!(real::LRA::try_from(real::NRA::from(real::Scale::new(Mat([2, 3])))).is_ok());
        assert!(int::LIA::try_from(int::NIA::from(int::Scale::new(Mat([2, 3])))).is_ok());
    }

    #[test]
    fn real_operations_take_the_degree_of_their_operands() {
        let r = |shape, degree| -> sort::Any { sort::Real::with_degree(shape, degree).into() };
        let b = |shape| -> sort::Any { sort::Bool::new(shape).into() };

        // kept by maps, choices and extrema
        let op = real(real::Abs::with_degree([2, 3], 1));
        assert_eq!(op.cod(), base::mono![r([2, 3], 1)].into());
        let op = real(real::Transpose::with_degree([2, 3], 2));
        assert_eq!(op.dom(), base::mono![r([2, 3], 2)].into());
        assert_eq!(op.cod(), base::mono![r([3, 2], 2)].into());
        let op = real(real::Max::with_degree([2, 3], 1));
        assert_eq!(op.dom(), base::mono![r([2, 3], 1)].into());
        assert_eq!(op.cod(), base::mono![r([1, 1], 1)].into());
        let op = real(real::Ite::with_degree([2, 3], 1));
        assert_eq!(
            op.dom(),
            base::mono![b([1, 1]), r([2, 3], 1), r([2, 3], 1)].into()
        );
        assert_eq!(op.cod(), base::mono![r([2, 3], 1)].into());

        // compared, writing booleans; an index is a plain real
        let op = real(real::Lt::with_degree([2, 3], 1));
        assert_eq!(op.dom(), base::mono![r([2, 3], 1); 2].into());
        assert_eq!(op.cod(), base::mono![b([2, 3])].into());
        let op = real(real::Argmax::with_degree([2, 3], 1));
        assert_eq!(op.dom(), base::mono![r([2, 3], 1)].into());
        assert_eq!(op.cod(), base::mono![r([1, 1], 0)].into());

        // the degrees of the factors of a matrix product add up
        let op = real(real::MatMul::with_degrees([[2, 3], [3, 4]], [1, 1]).unwrap());
        assert_eq!(op.dom(), base::mono![r([2, 3], 1), r([3, 4], 1)].into());
        assert_eq!(op.cod(), base::mono![r([2, 4], 2)].into());
        assert!(real::MatMul::with_degrees([[2, 3], [3, 4]], [u8::MAX, 1]).is_err());

        // constants and the linear maps of a constant are of a degree too
        let op = real(real::Const::with_degree(Mat([2, 3]), 1));
        assert_eq!(op.cod(), base::mono![r([2, 3], 1)].into());
        let op = real(real::Scale::with_degree(Mat([2, 3]), 1));
        assert_eq!(op.dom(), base::mono![r([2, 3], 1)].into());
        let op = real(real::Linear::with_degree(Mat([2, 3]), Mat([2, 4]), 1).unwrap());
        assert_eq!(op.dom(), base::mono![r([3, 4], 1)].into());
        assert_eq!(op.cod(), base::mono![r([2, 4], 1)].into());
    }

    #[test]
    fn division_takes_off_the_degree_of_the_divisor_and_is_partial() {
        let r = |degree| -> sort::Any { sort::Real::with_degree([2, 3], degree).into() };
        let op = real(real::Div::with_degrees([2, 3], [2, 1]).unwrap());
        assert_eq!(op.dom(), base::mono![r(2), r(1)].into());
        assert_eq!(op.cod(), base::mono![r(1)].into());
        assert_eq!(real::Div::new([2, 3]).degrees(), [0, 0]);
        // a negative degree is not defined
        assert!(real::Div::with_degrees([2, 3], [0, 1]).is_err());
        assert!(real::Div::with_degrees([2, 3], [1, 1]).is_ok());
    }

    #[test]
    fn negation_keeps_the_degree_of_reals() {
        let op = real(real::Neg::with_degree([2, 3], 1));
        let sort: Poly<sort::Any> = base::mono![sort::Real::with_degree([2, 3], 1).into()].into();
        assert_eq!(op.dom(), sort);
        assert_eq!(op.cod(), sort);
        assert_eq!(real::Neg::new([2, 3]).degree(), 0);
    }

    #[test]
    fn scale_maps_matrices_of_the_shape_of_its_constant() {
        let op = real(real::Scale::new(Mat([2, 3])));
        let shape: Poly<sort::Any> = base::mono![sort::Real::new([2, 3]).into()].into();
        assert_eq!(op.dom(), shape);
        assert_eq!(op.cod(), shape);
        assert_eq!(real::Scale::new(Mat([2, 3])).tensor().0, [2, 3]);
    }

    #[test]
    fn logics_cast_to_and_from_any() {
        // operations wrap into their logic, and logics into `Any`
        let lra = LRA::Real(real::Add::new([2, 3]).into());
        assert!(matches!(
            Any::from(lra),
            Any::Real(real::NRAT::Add(op)) if op == real::Add::new([2, 3])
        ));
        let lia = LIA::Int(int::Sub::new([2, 3]).into());
        assert!(matches!(
            Any::from(lia),
            Any::Int(int::NIA::Sub(op)) if op == int::Sub::new([2, 3])
        ));

        // `Any` casts down to the logics holding the operation only
        assert!(LRA::try_from(real(real::ReLU::new([2, 3]))).is_ok());
        assert!(LIA::try_from(real(real::ReLU::new([2, 3]))).is_err());
        assert!(LIA::try_from(int(int::ReLU::new([2, 3]))).is_ok());

        // boolean operations are in every logic
        let op = || boolean(bool::And::new([2, 3]));
        assert!(Prop::try_from(op()).is_ok());
        assert!(NRA::try_from(op()).is_ok());
        assert!(NIA::try_from(op()).is_ok());
        assert!(LRA::try_from(op()).is_ok());
        assert!(LIA::try_from(op()).is_ok());
        assert!(BV::try_from(op()).is_ok());

        // `Prop` holds the boolean operations only
        assert!(Prop::try_from(real(real::Add::new([2, 3]))).is_err());
    }

    #[test]
    fn linear_logics_hold_the_linear_operations_only() {
        // `Mul`, `Div`, `Mod` and `MatMul` are in neither linear logic
        assert!(LRA::try_from(real(real::MatMul::new([[2, 3], [3, 4]]).unwrap())).is_err());
        assert!(LRA::try_from(real(real::Mul::new([2, 3]))).is_err());
        assert!(LRA::try_from(real(real::Div::new([2, 3]))).is_err());
        assert!(LIA::try_from(int(int::MatMul::new([[2, 3], [3, 4]]).unwrap())).is_err());
        assert!(LIA::try_from(int(int::Mul::new([2, 3]))).is_err());
        assert!(LIA::try_from(int(int::Div::new([2, 3]))).is_err());
        assert!(LIA::try_from(int(int::Mod::new([2, 3]))).is_err());

        // but they are in the non-linear ones
        assert!(NRA::try_from(real(real::MatMul::new([[2, 3], [3, 4]]).unwrap())).is_ok());
        assert!(NRA::try_from(real(real::Div::new([2, 3]))).is_ok());
        assert!(NIA::try_from(int(int::Mod::new([2, 3]))).is_ok());

        // comparisons are linear
        assert!(LRA::try_from(real(real::Lt::new([2, 3]))).is_ok());
        assert!(LIA::try_from(int(int::Ge::new([2, 3]))).is_ok());
    }

    #[test]
    fn non_linear_logics_hold_their_theory_only() {
        assert!(NRA::try_from(int(int::Add::new([2, 3]))).is_err());
        assert!(NIA::try_from(real(real::Add::new([2, 3]))).is_err());
        assert!(NRA::try_from(word(word::Add::new(8, [2, 3]).unwrap())).is_err());
    }

    #[test]
    fn logics_embed_into_larger_logics() {
        let lra = LRA::Real(real::Sub::new([2, 3]).into());
        assert!(matches!(
            NRA::from(lra),
            NRA::Real(real::NRA::Sub(op)) if op == real::Sub::new([2, 3])
        ));
        let lia = LIA::Int(int::Max::new([2, 3]).into());
        assert!(matches!(
            NIA::from(lia),
            NIA::Int(int::NIA::Max(op)) if op == int::Max::new([2, 3])
        ));

        let prop = || Prop::Bool(bool::Or::new([2, 3]).into());
        assert!(matches!(
            NRA::from(prop()),
            NRA::Bool(bool::Prop::Or(op)) if op == bool::Or::new([2, 3])
        ));
        assert!(matches!(NIA::from(prop()), NIA::Bool(_)));
        assert!(matches!(LRA::from(prop()), LRA::Bool(_)));
        assert!(matches!(LIA::from(prop()), LIA::Bool(_)));
        assert!(matches!(BV::from(prop()), BV::Bool(_)));
    }

    #[test]
    fn transcendental_operations_are_in_nrat_only() {
        let unary: [real::NRAT<Mat>; 12] = [
            real::Sqrt::new([2, 3]).into(),
            real::Exp::new([2, 3]).into(),
            real::Log::new([2, 3]).into(),
            real::Sin::new([2, 3]).into(),
            real::Cos::new([2, 3]).into(),
            real::Tan::new([2, 3]).into(),
            real::Asin::new([2, 3]).into(),
            real::Acos::new([2, 3]).into(),
            real::Atan::new([2, 3]).into(),
            real::Sinh::new([2, 3]).into(),
            real::Cosh::new([2, 3]).into(),
            real::Tanh::new([2, 3]).into(),
        ];
        let plain = |shape| -> sort::Any { sort::Real::new(shape).into() };
        for op in unary {
            // pointwise on plain reals
            let any = Any::Real(op);
            assert_eq!(any.dom(), base::mono![plain([2, 3])].into());
            assert_eq!(any.cod(), base::mono![plain([2, 3])].into());
            let Any::Real(op) = any else { unreachable!() };
            assert!(real::NRA::try_from(op).is_err());
        }
        let atan2 = || real(real::Atan2::new([2, 3]));
        assert_eq!(
            atan2().dom(),
            base::mono![plain([2, 3]), plain([2, 3])].into()
        );
        assert_eq!(atan2().cod(), base::mono![plain([2, 3])].into());

        // the logics: NRAT holds them, NRA and LRA do not
        assert!(NRAT::try_from(atan2()).is_ok());
        assert!(NRAT::try_from(real(real::Sin::new([2, 3]))).is_ok());
        assert!(NRA::try_from(real(real::Sin::new([2, 3]))).is_err());
        assert!(LRA::try_from(real(real::Sqrt::new([2, 3]))).is_err());
        assert!(NRAT::try_from(int(int::Add::new([2, 3]))).is_err());

        // NRA and LRA nest inside NRAT, and keep their own operations
        assert!(NRAT::try_from(real(real::Mul::new([2, 3]))).is_ok());
        assert!(NRA::try_from(real(real::Mul::new([2, 3]))).is_ok());
        let nra = NRA::Real(real::Mul::new([2, 3]).into());
        assert!(matches!(
            NRAT::from(nra),
            NRAT::Real(real::NRAT::Mul(op)) if op == real::Mul::new([2, 3])
        ));
        let lra: real::LRA<Mat> = real::Add::new([2, 3]).into();
        assert!(matches!(real::NRA::from(lra), real::NRA::Add(_)));
        assert_eq!(real::Sin::new([2, 3]).to_string(), "Sin");
        assert_eq!(real::Atan2::new([2, 3]).to_string(), "Atan2");
    }

    #[test]
    fn bv_holds_word_and_boolean_operations() {
        assert!(BV::try_from(word(word::signed::Div::new(8, [2, 3]).unwrap())).is_ok());
        assert!(BV::try_from(word(word::unsigned::Lt::new(8, [2, 3]).unwrap())).is_ok());
        assert!(BV::try_from(word(word::Concat::new([8, 4], [2, 3]).unwrap())).is_ok());
        assert!(BV::try_from(int(int::Add::new([2, 3]))).is_err());
        assert!(BV::try_from(real(real::Add::new([2, 3]))).is_err());

        // word operations are in none of the other logics
        assert!(LIA::try_from(word(word::Add::new(8, [2, 3]).unwrap())).is_err());
        assert!(Prop::try_from(word(word::Not::new(8, [2, 3]).unwrap())).is_err());
    }

    #[test]
    fn word_ops_have_their_boundary_in_sort() {
        let op = word(word::Add::new(8, [2, 3]).unwrap());
        let w: sort::Any = sort::Word::new_unchecked(8, [2, 3]).into();
        let dom: Poly<sort::Any> = base::mono![w, w].into();
        let cod: Poly<sort::Any> = base::mono![w].into();
        assert_eq!(op.dom(), dom);
        assert_eq!(op.cod(), cod);

        // comparisons write the booleans of `bool`
        let op = word(word::signed::Lt::new(8, [2, 3]).unwrap());
        let cod: Poly<sort::Any> = base::mono![sort::Bool::new([2, 3]).into()].into();
        assert_eq!(op.dom(), dom);
        assert_eq!(op.cod(), cod);

        // signed and unsigned operations are distinct operations
        assert!(matches!(
            word::BV::<Mat>::from(word::signed::Div::new(8, [2, 3]).unwrap()),
            word::BV::SDiv(_)
        ));
        assert!(matches!(
            word::BV::<Mat>::from(word::unsigned::Div::new(8, [2, 3]).unwrap()),
            word::BV::UDiv(_)
        ));
    }

    #[test]
    fn comparisons_read_numbers_and_write_booleans() {
        let op = int(int::Lt::new([2, 3]));
        let i: sort::Any = sort::Int::new([2, 3]).into();
        let dom: Poly<sort::Any> = base::mono![i, i].into();
        let cod: Poly<sort::Any> = base::mono![sort::Bool::new([2, 3]).into()].into();
        assert_eq!(op.dom(), dom);
        assert_eq!(op.cod(), cod);
    }
}

/// Tests for `tensor::bool`.
mod bool {
    use base::boundary::{Boundary, Poly};
    use base::mono;
    use theory::tensor::bool::*;
    use theory::tensor::sort::Bool;

    fn one(s: Bool) -> Poly<Bool> {
        mono![s].into()
    }

    fn two(s: Bool) -> Poly<Bool> {
        mono![s, s].into()
    }

    #[test]
    fn pointwise_boundaries() {
        let b = Bool::new([2, 3]);
        assert_eq!(And::new([2, 3]).dom(), two(b));
        assert_eq!(And::new([2, 3]).cod(), one(b));
        assert_eq!(Or::new([2, 3]).dom(), two(b));
        assert_eq!(Xor::new([2, 3]).cod(), one(b));
        assert_eq!(Not::new([2, 3]).dom(), one(b));
        assert_eq!(Not::new([2, 3]).cod(), one(b));
        assert_eq!(Eq::new([2, 3]).dom(), two(b));
        assert_eq!(Eq::new([2, 3]).cod(), one(b));
        assert_eq!(Ne::new([2, 3]).dom(), two(b));
        assert_eq!(Ne::new([2, 3]).cod(), one(b));
    }

    #[test]
    fn all_reduces_a_matrix_to_one_boolean() {
        assert_eq!(All::new([2, 3]).dom(), one(Bool::new([2, 3])));
        assert_eq!(All::new([2, 3]).cod(), one(Bool::new([1, 1])));
    }

    /// The condition is one boolean, whatever the shape of the branches.
    #[test]
    fn ite_reads_a_single_condition() {
        let (c, b) = (Bool::new([1, 1]), Bool::new([2, 3]));
        assert_eq!(Ite::new([2, 3]).dom(), mono![c, b, b].into());
        assert_eq!(Ite::new([2, 3]).cod(), one(b));
    }

    #[test]
    fn ops_display_their_name() {
        assert_eq!(And::new([1, 1]).to_string(), "And");
        assert_eq!(Not::new([1, 1]).to_string(), "Not");
        assert_eq!(Eq::new([1, 1]).to_string(), "Eq");
        assert_eq!(Ne::new([1, 1]).to_string(), "Ne");
        assert_eq!(All::new([1, 1]).to_string(), "All");
    }
}

/// Tests for `tensor::int`.
mod int {
    use base::boundary::{Boundary, Poly};
    use base::mono;
    use theory::tensor::int::*;
    use theory::tensor::sort::{Bool, Int};

    fn one(s: Int) -> Poly<Int> {
        mono![s].into()
    }

    fn two(s: Int) -> Poly<Int> {
        mono![s, s].into()
    }

    #[test]
    fn pointwise_boundaries() {
        let r = Int::new([2, 3]);
        assert_eq!(Add::new([2, 3]).dom(), two(r));
        assert_eq!(Add::new([2, 3]).cod(), one(r));
        assert_eq!(Sub::new([2, 3]).dom(), two(r));
        assert_eq!(Mul::new([2, 3]).dom(), two(r));
        assert_eq!(Mul::new([2, 3]).cod(), one(r));
        assert_eq!(Div::new([2, 3]).dom(), two(r));
        assert_eq!(Div::new([2, 3]).cod(), one(r));
        assert_eq!(Mod::new([2, 3]).dom(), two(r));
        assert_eq!(Mod::new([2, 3]).cod(), one(r));
        assert_eq!(ReLU::new([2, 3]).dom(), one(r));
        assert_eq!(ReLU::new([2, 3]).cod(), one(r));
    }

    #[test]
    fn min_and_max_reduce_a_matrix_to_one_entry() {
        assert_eq!(Min::new([2, 3]).dom(), one(Int::new([2, 3])));
        assert_eq!(Min::new([2, 3]).cod(), one(Int::new([1, 1])));
        assert_eq!(Max::new([2, 3]).dom(), one(Int::new([2, 3])));
        assert_eq!(Max::new([2, 3]).cod(), one(Int::new([1, 1])));
    }

    #[test]
    fn transpose_swaps_shape() {
        let t = Transpose::new([2, 3]);
        assert_eq!(t.dom(), one(Int::new([2, 3])));
        assert_eq!(t.cod(), one(Int::new([3, 2])));
    }

    #[test]
    fn matmul_contracts_inner_size() {
        let p = MatMul::new([[2, 3], [3, 4]]).unwrap();
        let dom: Poly<Int> = mono![Int::new([2, 3]), Int::new([3, 4])].into();
        assert_eq!(p.dom(), dom);
        assert_eq!(p.cod(), one(Int::new([2, 4])));
    }

    #[test]
    fn matmul_checks_inner_size() {
        assert!(MatMul::new([[2, 3], [4, 5]]).is_err());
        let p = MatMul::new([[2, 3], [3, 4]]).unwrap();
        assert_eq!(p.shapes(), [[2, 3], [3, 4]]);
    }

    #[test]
    fn shapes_may_be_empty() {
        assert_eq!(Add::new([0, 3]).shape(), [0, 3]);
        assert!(MatMul::new([[2, 0], [0, 4]]).is_ok());
        assert_eq!(Transpose::new([0, 0]).cod(), one(Int::new([0, 0])));
    }

    #[test]
    fn ops_display_their_name() {
        assert_eq!(Add::new([1, 1]).to_string(), "Add");
        assert_eq!(Mul::new([1, 1]).to_string(), "Mul");
        assert_eq!(Div::new([1, 1]).to_string(), "Div");
        assert_eq!(Mod::new([1, 1]).to_string(), "Mod");
        assert_eq!(ReLU::new([1, 1]).to_string(), "ReLU");
        assert_eq!(Transpose::new([1, 1]).to_string(), "Transpose");
    }
    #[test]
    fn comparisons_write_booleans() {
        let x: Sort = Int::new([2, 3]).into();
        let b: Sort = Bool::new([2, 3]).into();
        let dom: Poly<Sort> = mono![x, x].into();
        let cod: Poly<Sort> = mono![b].into();
        for (d, c) in [
            (Lt::new([2, 3]).dom(), Lt::new([2, 3]).cod()),
            (Le::new([2, 3]).dom(), Le::new([2, 3]).cod()),
            (Gt::new([2, 3]).dom(), Gt::new([2, 3]).cod()),
            (Ge::new([2, 3]).dom(), Ge::new([2, 3]).cod()),
            (Eq::new([2, 3]).dom(), Eq::new([2, 3]).cod()),
            (Ne::new([2, 3]).dom(), Ne::new([2, 3]).cod()),
        ] {
            assert_eq!(d, dom);
            assert_eq!(c, cod);
        }
        assert_eq!(Le::new([1, 1]).to_string(), "Le");
        assert_eq!(Ne::new([1, 1]).to_string(), "Ne");
    }
}

/// Tests for `tensor::real`.
mod real {
    use base::boundary::{Boundary, Poly};
    use base::mono;
    use theory::tensor::real::*;
    use theory::tensor::sort::{Bool, Real};

    fn one(s: Real) -> Poly<Real> {
        mono![s].into()
    }

    fn two(s: Real) -> Poly<Real> {
        mono![s, s].into()
    }

    #[test]
    fn pointwise_boundaries() {
        let r = Real::new([2, 3]);
        assert_eq!(Add::new([2, 3]).dom(), two(r));
        assert_eq!(Add::new([2, 3]).cod(), one(r));
        assert_eq!(Sub::new([2, 3]).dom(), two(r));
        assert_eq!(Mul::new([2, 3]).dom(), two(r));
        assert_eq!(Mul::new([2, 3]).cod(), one(r));
        assert_eq!(Div::new([2, 3]).dom(), two(r));
        assert_eq!(Div::new([2, 3]).cod(), one(r));
        assert_eq!(ReLU::new([2, 3]).dom(), one(r));
        assert_eq!(ReLU::new([2, 3]).cod(), one(r));
    }

    #[test]
    fn min_and_max_reduce_a_matrix_to_one_entry() {
        assert_eq!(Min::new([2, 3]).dom(), one(Real::new([2, 3])));
        assert_eq!(Min::new([2, 3]).cod(), one(Real::new([1, 1])));
        assert_eq!(Max::new([2, 3]).dom(), one(Real::new([2, 3])));
        assert_eq!(Max::new([2, 3]).cod(), one(Real::new([1, 1])));
    }

    #[test]
    fn transpose_swaps_shape() {
        let t = Transpose::new([2, 3]);
        assert_eq!(t.dom(), one(Real::new([2, 3])));
        assert_eq!(t.cod(), one(Real::new([3, 2])));
    }

    #[test]
    fn matmul_contracts_inner_size() {
        let p = MatMul::new([[2, 3], [3, 4]]).unwrap();
        let dom: Poly<Real> = mono![Real::new([2, 3]), Real::new([3, 4])].into();
        assert_eq!(p.dom(), dom);
        assert_eq!(p.cod(), one(Real::new([2, 4])));
    }

    #[test]
    fn matmul_checks_inner_size() {
        assert!(MatMul::new([[2, 3], [4, 5]]).is_err());
        let p = MatMul::new([[2, 3], [3, 4]]).unwrap();
        assert_eq!(p.shapes(), [[2, 3], [3, 4]]);
    }

    #[test]
    fn mul_adds_the_degrees_of_its_factors() {
        let r = |d| Real::with_degree([2, 3], d);
        let rates = Mul::with_degrees([2, 3], [1, 1]).unwrap();
        assert_eq!(rates.dom(), two(r(1)));
        assert_eq!(rates.cod(), one(r(2)));
        let scaled = Mul::with_degrees([2, 3], [1, 0]).unwrap();
        let dom: Poly<Real> = mono![r(1), r(0)].into();
        assert_eq!(scaled.dom(), dom);
        assert_eq!(scaled.cod(), one(r(1)));
        assert_eq!(Mul::new([2, 3]), Mul::with_degrees([2, 3], [0, 0]).unwrap());
    }

    #[test]
    fn mul_checks_the_degree_of_its_product() {
        assert!(Mul::with_degrees([2, 3], [u8::MAX, 1]).is_err());
        assert!(Mul::with_degrees([2, 3], [u8::MAX, 0]).is_ok());
        assert_eq!(Mul::with_degrees([2, 3], [1, 0]).unwrap().degrees(), [1, 0]);
    }

    #[test]
    fn sums_keep_the_degree_of_their_terms() {
        let r = Real::with_degree([2, 3], 1);
        assert_eq!(Add::with_degree([2, 3], 1).dom(), two(r));
        assert_eq!(Add::with_degree([2, 3], 1).cod(), one(r));
        assert_eq!(Sub::with_degree([2, 3], 1).dom(), two(r));
        assert_eq!(Sub::with_degree([2, 3], 1).cod(), one(r));
        assert_eq!(Add::new([2, 3]), Add::with_degree([2, 3], 0));
    }

    #[test]
    fn shapes_may_be_empty() {
        assert_eq!(Add::new([0, 3]).shape(), [0, 3]);
        assert!(MatMul::new([[2, 0], [0, 4]]).is_ok());
        assert_eq!(Transpose::new([0, 0]).cod(), one(Real::new([0, 0])));
    }

    #[test]
    fn ops_display_their_name() {
        assert_eq!(Add::new([1, 1]).to_string(), "Add");
        assert_eq!(Mul::new([1, 1]).to_string(), "Mul");
        assert_eq!(Div::new([1, 1]).to_string(), "Div");
        assert_eq!(ReLU::new([1, 1]).to_string(), "ReLU");
        assert_eq!(Transpose::new([1, 1]).to_string(), "Transpose");
    }
    #[test]
    fn comparisons_write_booleans() {
        let x: Sort = Real::new([2, 3]).into();
        let b: Sort = Bool::new([2, 3]).into();
        let dom: Poly<Sort> = mono![x, x].into();
        let cod: Poly<Sort> = mono![b].into();
        for (d, c) in [
            (Lt::new([2, 3]).dom(), Lt::new([2, 3]).cod()),
            (Le::new([2, 3]).dom(), Le::new([2, 3]).cod()),
            (Gt::new([2, 3]).dom(), Gt::new([2, 3]).cod()),
            (Ge::new([2, 3]).dom(), Ge::new([2, 3]).cod()),
            (Eq::new([2, 3]).dom(), Eq::new([2, 3]).cod()),
            (Ne::new([2, 3]).dom(), Ne::new([2, 3]).cod()),
        ] {
            assert_eq!(d, dom);
            assert_eq!(c, cod);
        }
        assert_eq!(Le::new([1, 1]).to_string(), "Le");
        assert_eq!(Ne::new([1, 1]).to_string(), "Ne");
    }
}

/// Tests for `tensor::sort`.
mod sort {
    use base::differentiable::T;
    use theory::tensor::sort::*;

    #[test]
    fn real_shape_and_display() {
        let r = Real::new([2, 3]);
        assert_eq!(r.shape(), [2, 3]);
        assert_eq!(r.to_string(), "Real(2, 3)");
    }

    #[test]
    fn real_tangent_raises_degree() {
        let r = Real::new([2, 3]);
        assert_eq!(T(r).degree(), 1);
        assert_eq!(T(r).shape(), [2, 3]);
        assert_eq!(T(T(r)).degree(), 2);
        assert_eq!(T(r).to_string(), "T^1(Real(2, 3))");
    }

    #[test]
    fn int_shape_and_display() {
        let r = Int::new([2, 3]);
        assert_eq!(r.shape(), [2, 3]);
        assert_eq!(r.to_string(), "Int(2, 3)");
    }

    #[test]
    fn int_tangent_is_trivial() {
        assert_eq!(T(Int::new([2, 3])), Zero);
        assert_eq!(T(T(Int::new([2, 3]))), Zero);
    }

    #[test]
    fn bool_shape_and_display() {
        let b = Bool::new([2, 3]);
        assert_eq!(b.shape(), [2, 3]);
        assert_eq!(b.to_string(), "Bool(2, 3)");
    }

    #[test]
    fn bool_tangent_is_trivial() {
        assert_eq!(T(Bool::new([2, 3])), Zero);
        assert_eq!(T(T(Bool::new([2, 3]))), Zero);
    }

    #[test]
    fn word_width_shape_and_display() {
        let x = Word::new(8, [2, 3]).unwrap();
        assert_eq!(x.width(), 8);
        assert_eq!(x.shape(), [2, 3]);
        assert_eq!(x.to_string(), "Word<8>(2, 3)");
    }

    #[test]
    fn word_has_positive_width() {
        assert_eq!(
            Word::new(0, [2, 3]),
            Err(theory::Error("words have at least one bit".to_string()))
        );
        assert!(Word::new(1, [2, 3]).is_ok());
    }

    #[test]
    fn word_new_unchecked_builds_positive_widths() {
        assert_eq!(
            Word::new_unchecked(8, [2, 3]),
            Word::new(8, [2, 3]).unwrap()
        );
    }

    #[test]
    #[should_panic(expected = "words have at least one bit")]
    fn word_new_unchecked_panics_on_zero_width() {
        Word::new_unchecked(0, [2, 3]);
    }

    #[test]
    fn words_display_their_sort() {
        let x = Word::new_unchecked(8, [2, 3]);
        assert_eq!(Words::from(x), Words::Word(x));
        assert_eq!(Words::from(x).to_string(), "Word<8>(2, 3)");
        assert_eq!(Words::from(Bool::new([2, 3])).to_string(), "Bool(2, 3)");
        assert_eq!(Words::from(Zero).to_string(), "Zero");
    }

    #[test]
    fn word_tangent_is_trivial() {
        assert_eq!(T(Word::new_unchecked(8, [2, 3])), Zero);
        assert_eq!(T(Words::from(Word::new_unchecked(8, [2, 3]))), Zero);
        assert_eq!(T(Words::from(Bool::new([2, 3]))), Zero);
    }

    #[test]
    fn zero_tangent_is_itself() {
        assert_eq!(T(Zero), Zero);
    }

    #[test]
    fn zero_display() {
        assert_eq!(Zero.to_string(), "Zero");
    }

    #[test]
    fn tangent_is_each_sorts_tangent() {
        let r = Real::new([2, 3]);
        assert_eq!(T(Any::from(r)), Any::Real(T(r)));
        assert_eq!(T(Any::from(Int::new([2, 3]))), Any::Zero(Zero));
        assert_eq!(T(Any::from(Bool::new([2, 3]))), Any::Zero(Zero));
        assert_eq!(
            T(Any::from(Word::new_unchecked(8, [2, 3]))),
            Any::Zero(Zero)
        );
        assert_eq!(T(Any::from(Zero)), Any::Zero(Zero));
    }

    #[test]
    fn subsets_hold_the_tangents_of_their_sorts() {
        let sorts = [
            Any::Real(Real::new([2, 3])),
            Any::Int(Int::new([2, 3])),
            Any::Bool(Bool::new([2, 3])),
            Any::Word(Word::new_unchecked(8, [2, 3])),
            Any::Zero(Zero),
        ];
        for s in sorts {
            if Prop::try_from(s).is_ok() {
                assert!(Prop::try_from(T(s)).is_ok());
            }
            if Reals::try_from(s).is_ok() {
                assert!(Reals::try_from(T(s)).is_ok());
            }
            if Ints::try_from(s).is_ok() {
                assert!(Ints::try_from(T(s)).is_ok());
            }
            if Nums::try_from(s).is_ok() {
                assert!(Nums::try_from(T(s)).is_ok());
            }
            if Words::try_from(s).is_ok() {
                assert!(Words::try_from(T(s)).is_ok());
            }
        }
    }

    #[test]
    fn subsets_cast_to_and_from_any() {
        let r = Real::new([2, 3]);
        let i = Int::new([2, 3]);
        let b = Bool::new([2, 3]);
        let w = Word::new_unchecked(8, [2, 3]);

        // each sort casts up from the subsets that hold it
        assert_eq!(Any::from(Reals::from(r)), Any::Real(r));
        assert_eq!(Any::from(Ints::from(i)), Any::Int(i));
        assert_eq!(Any::from(Nums::from(r)), Any::Real(r));
        assert_eq!(Any::from(Nums::from(i)), Any::Int(i));
        assert_eq!(Any::from(Words::from(w)), Any::Word(w));

        // and down to them only
        assert_eq!(Words::try_from(Any::Word(w)).ok(), Some(Words::Word(w)));
        assert!(Words::try_from(Any::Real(r)).is_err());
        assert!(Reals::try_from(Any::Int(i)).is_err());
        assert!(Ints::try_from(Any::Real(r)).is_err());
        assert!(Nums::try_from(Any::Word(w)).is_err());

        // every subset holds the booleans and `Zero`
        for s in [Any::Bool(b), Any::Zero(Zero)] {
            assert!(Prop::try_from(s).is_ok());
            assert!(Reals::try_from(s).is_ok());
            assert!(Ints::try_from(s).is_ok());
            assert!(Nums::try_from(s).is_ok());
            assert!(Words::try_from(s).is_ok());
        }

        // `Prop` holds only those
        assert!(Prop::try_from(Any::Real(r)).is_err());
        assert!(Prop::try_from(Any::Int(i)).is_err());
        assert!(Prop::try_from(Any::Word(w)).is_err());
    }
}

/// Tests for `tensor::word`.
mod word {
    use base::boundary::{Boundary, Poly};
    use base::mono;
    use theory::tensor::sort::{Bool, Word};
    use theory::tensor::word::*;

    fn w(width: usize, shape: [usize; 2]) -> Sort {
        Word::new(width, shape).unwrap().into()
    }

    fn b(shape: [usize; 2]) -> Sort {
        Bool::new(shape).into()
    }

    fn one(s: Sort) -> Poly<Sort> {
        mono![s].into()
    }

    fn two(s: Sort) -> Poly<Sort> {
        mono![s, s].into()
    }

    #[test]
    fn pointwise_boundaries_preserve_width() {
        let x = w(8, [2, 3]);
        assert_eq!(Add::new(8, [2, 3]).unwrap().dom(), two(x));
        assert_eq!(Add::new(8, [2, 3]).unwrap().cod(), one(x));
        assert_eq!(Xnor::new(8, [2, 3]).unwrap().cod(), one(x));
        assert_eq!(Neg::new(8, [2, 3]).unwrap().dom(), one(x));
        assert_eq!(Not::new(8, [2, 3]).unwrap().cod(), one(x));
        assert_eq!(RotateLeft::new(3, 8, [2, 3]).unwrap().dom(), one(x));
        assert_eq!(RotateRight::new(3, 8, [2, 3]).unwrap().cod(), one(x));
        assert_eq!(signed::Div::new(8, [2, 3]).unwrap().dom(), two(x));
        assert_eq!(signed::Mod::new(8, [2, 3]).unwrap().cod(), one(x));
        assert_eq!(unsigned::Rem::new(8, [2, 3]).unwrap().cod(), one(x));
        assert_eq!(unsigned::Shr::new(8, [2, 3]).unwrap().dom(), two(x));
    }

    #[test]
    fn comparisons_write_booleans() {
        let x = w(8, [2, 3]);
        assert_eq!(Eq::new(8, [2, 3]).unwrap().dom(), two(x));
        assert_eq!(Eq::new(8, [2, 3]).unwrap().cod(), one(b([2, 3])));
        assert_eq!(Ne::new(8, [2, 3]).unwrap().cod(), one(b([2, 3])));
        assert_eq!(signed::Lt::new(8, [2, 3]).unwrap().dom(), two(x));
        assert_eq!(signed::Lt::new(8, [2, 3]).unwrap().cod(), one(b([2, 3])));
        assert_eq!(unsigned::Ge::new(8, [2, 3]).unwrap().cod(), one(b([2, 3])));
    }

    #[test]
    fn width_changing_boundaries() {
        assert_eq!(
            Extract::new(7, 4, 16, [2, 3]).unwrap().dom(),
            one(w(16, [2, 3]))
        );
        assert_eq!(
            Extract::new(7, 4, 16, [2, 3]).unwrap().cod(),
            one(w(4, [2, 3]))
        );
        assert_eq!(
            Extract::new(0, 0, 8, [1, 1]).unwrap().cod(),
            one(w(1, [1, 1]))
        );

        let dom: Poly<Sort> = mono![w(8, [2, 3]), w(4, [2, 3])].into();
        assert_eq!(Concat::new([8, 4], [2, 3]).unwrap().dom(), dom);
        assert_eq!(
            Concat::new([8, 4], [2, 3]).unwrap().cod(),
            one(w(12, [2, 3]))
        );

        assert_eq!(
            unsigned::Extend::new(8, 8, [2, 3]).unwrap().cod(),
            one(w(16, [2, 3]))
        );
        assert_eq!(
            signed::Extend::new(24, 8, [2, 3]).unwrap().cod(),
            one(w(32, [2, 3]))
        );
        assert_eq!(Repeat::new(3, 4, [2, 3]).unwrap().cod(), one(w(12, [2, 3])));
    }

    #[test]
    fn ops_display_their_name_and_indices() {
        assert_eq!(Add::new(8, [1, 1]).unwrap().to_string(), "Add");
        assert_eq!(Ne::new(8, [1, 1]).unwrap().to_string(), "Ne");
        assert_eq!(
            Extract::new(7, 4, 8, [1, 1]).unwrap().to_string(),
            "Extract[7:4]"
        );
        assert_eq!(
            RotateLeft::new(3, 8, [1, 1]).unwrap().to_string(),
            "RotateLeft(3)"
        );
        assert_eq!(Repeat::new(2, 8, [1, 1]).unwrap().to_string(), "Repeat(2)");
        assert_eq!(signed::Div::new(8, [1, 1]).unwrap().to_string(), "SDiv");
        assert_eq!(unsigned::Div::new(8, [1, 1]).unwrap().to_string(), "UDiv");
        assert_eq!(signed::Shr::new(8, [1, 1]).unwrap().to_string(), "AShr");
        assert_eq!(unsigned::Shr::new(8, [1, 1]).unwrap().to_string(), "LShr");
        assert_eq!(signed::Ge::new(8, [1, 1]).unwrap().to_string(), "SGe");
        assert_eq!(
            signed::Extend::new(8, 8, [1, 1]).unwrap().to_string(),
            "SignExtend(8)"
        );
        assert_eq!(
            unsigned::Extend::new(8, 8, [1, 1]).unwrap().to_string(),
            "ZeroExtend(8)"
        );
    }

    #[test]
    fn constructors_check_their_sizes() {
        // words have at least one bit
        assert!(Add::new(0, [2, 3]).is_err());
        assert!(signed::Div::new(0, [2, 3]).is_err());
        assert!(unsigned::Lt::new(0, [2, 3]).is_err());
        assert!(signed::Extend::new(4, 0, [2, 3]).is_err());
        assert!(RotateLeft::new(3, 0, [2, 3]).is_err());

        // concatenation needs two words
        assert!(Concat::new([0, 4], [2, 3]).is_err());
        assert!(Concat::new([8, 0], [2, 3]).is_err());

        // extraction needs `j ≤ i < width`
        assert!(Extract::new(7, 0, 8, [2, 3]).is_ok());
        assert!(Extract::new(8, 0, 8, [2, 3]).is_err());
        assert!(Extract::new(2, 3, 8, [2, 3]).is_err());

        // repetition needs at least one copy
        assert!(Repeat::new(0, 8, [2, 3]).is_err());
        assert!(Repeat::new(1, 0, [2, 3]).is_err());

        // but shapes may be empty
        assert!(Add::new(8, [0, 0]).is_ok());
    }

    #[test]
    fn getters_read_the_fields() {
        let e = Extract::new(7, 4, 8, [2, 3]).unwrap();
        assert_eq!((e.i(), e.j(), e.width(), e.shape()), (7, 4, 8, [2, 3]));
        let c = Concat::new([8, 4], [2, 3]).unwrap();
        assert_eq!((c.widths(), c.shape()), ([8, 4], [2, 3]));
        let r = RotateRight::new(3, 8, [2, 3]).unwrap();
        assert_eq!((r.i(), r.width(), r.shape()), (3, 8, [2, 3]));
    }

    /// The width of the words that an operation writes must fit a `usize`,
    /// or its boundary could not be computed: the constructors reject the
    /// sizes whose width overflows, and accept those whose width is
    /// `usize::MAX` exactly.
    #[test]
    fn constructors_reject_widths_that_overflow() {
        let max = usize::MAX;

        // `Concat(u, v)` writes words of `u + v` bits
        assert!(Concat::new([max, 1], [1, 1]).is_err());
        assert_eq!(
            Concat::new([max - 1, 1], [1, 1]).unwrap().cod(),
            one(w(max, [1, 1]))
        );

        // `Repeat(i, width)` writes words of `i · width` bits
        assert!(Repeat::new(max / 2 + 1, 2, [1, 1]).is_err());
        assert_eq!(
            Repeat::new(max / 2, 2, [1, 1]).unwrap().cod(),
            one(w(max - 1, [1, 1]))
        );

        // `Extend(i, width)` writes words of `width + i` bits
        assert!(signed::Extend::new(1, max, [1, 1]).is_err());
        assert!(unsigned::Extend::new(max, 1, [1, 1]).is_err());
        assert_eq!(
            unsigned::Extend::new(max - 1, 1, [1, 1]).unwrap().cod(),
            one(w(max, [1, 1]))
        );
    }
}

/// Tests for `Implies` and the operations `Get` of each theory, which read one
/// entry of a matrix.
mod entries {
    use base::boundary::{Boundary, Poly};
    use base::mono;
    use theory::tensor::sort::{Bool, Int, Real, Word};
    use theory::tensor::{bool, int, real, word};

    #[test]
    fn implies_is_pointwise() {
        let b = Bool::new([2, 3]);
        let op = bool::Implies::new([2, 3]);
        assert_eq!(op.dom(), Poly::from(mono![b, b]));
        assert_eq!(op.cod(), Poly::from(mono![b]));
        assert_eq!(op.to_string(), "Implies");
    }

    #[test]
    fn get_reads_a_matrix_and_writes_one_entry() {
        let op = int::Get::new(1, 2, [2, 3]).unwrap();
        assert_eq!(op.dom(), Poly::from(mono![Int::new([2, 3])]));
        assert_eq!(op.cod(), Poly::from(mono![Int::new([1, 1])]));
        assert_eq!(op.to_string(), "Get[1][2]");

        let op = bool::Get::new(0, 0, [1, 4]).unwrap();
        assert_eq!(op.cod(), Poly::from(mono![Bool::new([1, 1])]));

        let op = word::Get::new(0, 1, 8, [1, 2]).unwrap();
        assert_eq!(
            op.dom(),
            Poly::from(mono![word::Sort::from(Word::new(8, [1, 2]).unwrap())])
        );
        assert_eq!(
            op.cod(),
            Poly::from(mono![word::Sort::from(Word::new(8, [1, 1]).unwrap())])
        );
    }

    #[test]
    fn get_of_reals_keeps_the_degree() {
        let op = real::Get::with_degree(1, 0, [2, 1], 1).unwrap();
        assert_eq!(op.dom(), Poly::from(mono![Real::with_degree([2, 1], 1)]));
        assert_eq!(op.cod(), Poly::from(mono![Real::with_degree([1, 1], 1)]));
    }

    #[test]
    fn get_rejects_an_entry_outside_the_matrix() {
        assert!(int::Get::new(2, 0, [2, 3]).is_err());
        assert!(int::Get::new(0, 3, [2, 3]).is_err());
        assert!(real::Get::new(0, 0, [0, 1]).is_err());
        assert!(bool::Get::new(1, 1, [1, 1]).is_err());
        assert!(word::Get::new(0, 0, 0, [1, 1]).is_err());
    }
}

/// Tests for `tensor::cast`.
mod cast {
    use base::boundary::{Boundary, Poly};
    use base::mono;
    use theory::tensor::cast::*;
    use theory::tensor::sort::{Any, Int, Real, Word};

    fn one(s: impl Into<Any>) -> Poly<Any> {
        mono![s.into()].into()
    }

    #[test]
    fn casts_keep_the_shape_and_change_the_sort() {
        assert_eq!(ToReal::new([2, 1]).dom(), one(Int::new([2, 1])));
        assert_eq!(ToReal::new([2, 1]).cod(), one(Real::new([2, 1])));
        assert_eq!(ToInt::new([1, 3]).dom(), one(Real::new([1, 3])));
        assert_eq!(ToInt::new([1, 3]).cod(), one(Int::new([1, 3])));
        let op = UbvToInt::new(8, [1, 1]).unwrap();
        assert_eq!(op.dom(), one(Word::new(8, [1, 1]).unwrap()));
        assert_eq!(op.cod(), one(Int::new([1, 1])));
    }

    #[test]
    fn ubv_to_int_has_positive_width() {
        assert!(UbvToInt::new(0, [1, 1]).is_err());
    }

    #[test]
    fn casts_are_operations_of_no_theory() {
        type Any = theory::tensor::Any<Mat>;
        #[derive(Debug)]
        struct Mat([usize; 2]);
        impl theory::Tensor for Mat {
            fn shape(&self) -> [usize; 2] {
                self.0
            }
        }
        let op = Any::Cast(Cast::ToReal(ToReal::new([1, 1])));
        assert_eq!(op.dom(), one(Int::new([1, 1])));
        assert_eq!(op.cod(), one(Real::new([1, 1])));
        assert_eq!(Cast::ToReal(ToReal::new([1, 1])).to_string(), "ToReal");
    }
}
