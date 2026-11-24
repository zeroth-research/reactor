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
    type NRA = theory::tensor::NRA<Mat>;
    #[allow(clippy::upper_case_acronyms)]
    type NIA = theory::tensor::NIA<Mat>;
    #[allow(clippy::upper_case_acronyms)]
    type LRA = theory::tensor::LRA<Mat>;
    #[allow(clippy::upper_case_acronyms)]
    type LIA = theory::tensor::LIA<Mat>;
    type Prop = theory::tensor::Prop<Mat>;
    type BV = theory::tensor::BV<Mat>;

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
        let op = real(
            real::MatMul::with_dom([sort::Real::new([2, 3]), sort::Real::new([3, 4])]).unwrap(),
        );
        let dom: Poly<sort::Any> = base::mono![
            sort::Real::new([2, 3]).into(),
            sort::Real::new([3, 4]).into()
        ]
        .into();
        let cod: Poly<sort::Any> = base::mono![sort::Real::new([2, 4]).into()].into();
        assert_eq!(op.dom(), dom);
        assert_eq!(op.cod(), cod);

        let op = int(int::Transpose::with_dom([sort::Int::new([2, 3]); 1]).unwrap());
        let cod: Poly<sort::Any> = base::mono![sort::Int::new([3, 2]).into()].into();
        assert_eq!(op.cod(), cod);

        let op = boolean(bool::Not::with_dom([sort::Bool::new([2, 3]); 1]).unwrap());
        let dom: Poly<sort::Any> = base::mono![sort::Bool::new([2, 3]).into()].into();
        assert_eq!(op.dom(), dom);
    }

    #[test]
    fn data_operations_take_their_shape_from_their_tensors() {
        let c = real(real::Const::with_cod(Mat([2, 3]), [sort::Real::new([2, 3])]).unwrap());
        assert_eq!(c.dom(), Poly::unit());
        let cod: Poly<sort::Any> = base::mono![sort::Real::new([2, 3]).into()].into();
        assert_eq!(c.cod(), cod);

        // `A` of shape `[2, 3]` and `B` of shape `[2, 4]` take `[3, 4]` to `[2, 4]`
        let l =
            int(int::Affine::with_dom(Mat([2, 3]), Mat([2, 4]), [sort::Int::new([3, 4])]).unwrap());
        let dom: Poly<sort::Any> = base::mono![sort::Int::new([3, 4]).into()].into();
        let cod: Poly<sort::Any> = base::mono![sort::Int::new([2, 4]).into()].into();
        assert_eq!(l.dom(), dom);
        assert_eq!(l.cod(), cod);

        let w = word(
            word::Const::with_cod(Mat([1, 2]), [sort::Word::new(8, [1, 2]).unwrap()]).unwrap(),
        );
        let cod: Poly<sort::Any> = base::mono![sort::Word::new_unchecked(8, [1, 2]).into()].into();
        assert_eq!(w.cod(), cod);

        // constants are linear
        assert!(LRA::try_from(c).is_ok());
        assert!(LIA::try_from(l).is_ok());
    }

    #[test]
    fn data_operations_check_their_tensors() {
        // `B` must have the rows of `A`
        assert!(
            real::Affine::with_dom(Mat([2, 3]), Mat([3, 4]), [sort::Real::new([3, 4])]).is_err()
        );
        assert!(int::Affine::with_dom(Mat([2, 3]), Mat([3, 4]), [sort::Int::new([3, 4])]).is_err());
        assert!(
            real::Affine::with_dom(Mat([0, 3]), Mat([0, 4]), [sort::Real::new([3, 4])]).is_ok()
        );

        // a constant word has the shape of its tensor
        assert!(word::Const::with_cod(Mat([1, 2]), [sort::Word::new(8, [2, 1]).unwrap()]).is_err());
        assert_eq!(
            word::Const::with_cod(Mat([1, 2]), [sort::Word::new(8, [1, 2]).unwrap()])
                .unwrap()
                .width(),
            8
        );
    }

    #[test]
    fn data_operations_read_their_tensors() {
        let c = real::Const::with_cod(Mat([2, 3]), [sort::Real::new([2, 3])]).unwrap();
        assert_eq!(c.tensor().0, [2, 3]);
        let l = int::Affine::with_dom(Mat([2, 3]), Mat([2, 4]), [sort::Int::new([3, 4])]).unwrap();
        assert_eq!(l.a().0, [2, 3]);
        assert_eq!(l.b().0, [2, 4]);
    }

    #[test]
    fn theories_mark_their_linear_operations() {
        assert!(
            real::LRA::try_from(real::NRA::<Mat>::from(
                real::Add::with_dom([sort::Real::new([2, 3]); 2]).unwrap()
            ))
            .is_ok()
        );
        assert!(
            real::LRA::try_from(real::NRA::<Mat>::from(
                real::Lt::with_dom([sort::Real::new([2, 3]); 2]).unwrap()
            ))
            .is_ok()
        );
        assert!(
            real::LRA::try_from(real::NRA::<Mat>::from(
                real::Mul::with_dom([sort::Real::new([2, 3]); 2]).unwrap()
            ))
            .is_err()
        );
        assert!(
            real::LRA::try_from(real::NRA::<Mat>::from(
                real::MatMul::with_dom([sort::Real::new([2, 3]), sort::Real::new([3, 4])]).unwrap()
            ))
            .is_err()
        );
        assert!(
            int::LIA::try_from(int::NIA::<Mat>::from(
                int::Mod::with_dom([sort::Int::new([2, 3]); 2]).unwrap()
            ))
            .is_err()
        );
        assert!(
            int::LIA::try_from(int::NIA::<Mat>::from(
                int::ReLU::with_dom([sort::Int::new([2, 3]); 1]).unwrap()
            ))
            .is_ok()
        );

        // negation and absolute value are linear
        assert!(
            real::LRA::try_from(real::NRA::<Mat>::from(
                real::Neg::with_dom([sort::Real::new([2, 3]); 1]).unwrap()
            ))
            .is_ok()
        );
        assert!(
            int::LIA::try_from(int::NIA::<Mat>::from(
                int::Neg::with_dom([sort::Int::new([2, 3]); 1]).unwrap()
            ))
            .is_ok()
        );
        assert!(
            int::LIA::try_from(int::NIA::<Mat>::from(
                int::Abs::with_dom([sort::Int::new([2, 3]); 1]).unwrap()
            ))
            .is_ok()
        );
        assert!(
            real::LRA::try_from(real::NRA::<Mat>::from(
                real::Abs::with_dom([sort::Real::new([2, 3]); 1]).unwrap()
            ))
            .is_ok()
        );

        // the product with a constant is linear, unlike that of two variables
        assert!(
            real::LRA::try_from(real::NRA::from(
                real::Scale::with_dom(Mat([2, 3]), [sort::Real::new([2, 3])]).unwrap()
            ))
            .is_ok()
        );
        assert!(
            int::LIA::try_from(int::NIA::from(
                int::Scale::with_dom(Mat([2, 3]), [sort::Int::new([2, 3])]).unwrap()
            ))
            .is_ok()
        );
    }

    #[test]
    fn zero_writes_the_singleton_sort() {
        let op = Any::Zero(Zero);
        assert_eq!(op.dom(), Poly::unit());
        let cod: Poly<sort::Any> = base::mono![sort::Zero.into()].into();
        assert_eq!(op.cod(), cod);
    }

    #[test]
    fn the_zero_rate_of_change_of_a_real_is_the_zero_matrix_of_its_degree() {
        use theory::Differential;
        // a real rate of change: the zero matrix, of its degree
        let rate: sort::Any = sort::Real::with_degree([2, 3], 1).into();
        let op = Any::zero(&rate);
        assert!(matches!(&op, Any::Real(real::NRA::Zero(z)) if z.degree() == 1));
        assert!(LRA::try_from(Any::zero(&rate)).is_ok());
        assert_eq!(op.dom(), Poly::unit());
        assert_eq!(op.cod(), base::mono![rate].into());
        // the rate of change of a discrete value: the singleton
        assert!(matches!(Any::zero(&sort::Zero.into()), Any::Zero(Zero)));
    }

    #[test]
    fn real_operations_take_the_degree_of_their_operands() {
        let r = |shape, degree| -> sort::Any { sort::Real::with_degree(shape, degree).into() };
        let b = |shape| -> sort::Any { sort::Bool::new(shape).into() };

        // kept by maps, choices and extrema
        let op = real(real::Abs::with_dom([sort::Real::with_degree([2, 3], 1); 1]).unwrap());
        assert_eq!(op.cod(), base::mono![r([2, 3], 1)].into());
        let op = real(real::Transpose::with_dom([sort::Real::with_degree([2, 3], 2); 1]).unwrap());
        assert_eq!(op.dom(), base::mono![r([2, 3], 2)].into());
        assert_eq!(op.cod(), base::mono![r([3, 2], 2)].into());
        let op = real(real::Max::with_dom([sort::Real::with_degree([2, 3], 1); 2]).unwrap());
        assert_eq!(op.dom(), base::mono![r([2, 3], 1); 2].into());
        let op = real(
            real::Ite::with_dom([
                sort::Bool::new([2, 3]).into(),
                sort::Real::with_degree([2, 3], 1).into(),
                sort::Real::with_degree([2, 3], 1).into(),
            ])
            .unwrap(),
        );
        assert_eq!(
            op.dom(),
            base::mono![b([2, 3]), r([2, 3], 1), r([2, 3], 1)].into()
        );
        assert_eq!(op.cod(), base::mono![r([2, 3], 1)].into());

        // compared, writing booleans; an index is a plain real
        let op = real(real::Lt::with_dom([sort::Real::with_degree([2, 3], 1); 2]).unwrap());
        assert_eq!(op.dom(), base::mono![r([2, 3], 1); 2].into());
        assert_eq!(op.cod(), base::mono![b([2, 3])].into());
        let op = real(real::Argmax::with_dom([sort::Real::with_degree([2, 3], 1); 1]).unwrap());
        assert_eq!(op.dom(), base::mono![r([2, 3], 1)].into());
        assert_eq!(op.cod(), base::mono![r([1, 1], 0)].into());

        // the degrees of the factors of a matrix product add up
        let op = real(
            real::MatMul::with_dom([
                sort::Real::with_degree([2, 3], 1),
                sort::Real::with_degree([3, 4], 1),
            ])
            .unwrap(),
        );
        assert_eq!(op.dom(), base::mono![r([2, 3], 1), r([3, 4], 1)].into());
        assert_eq!(op.cod(), base::mono![r([2, 4], 2)].into());
        assert!(
            real::MatMul::with_dom([
                sort::Real::with_degree([2, 3], u8::MAX),
                sort::Real::with_degree([3, 4], 1)
            ])
            .is_err()
        );

        // constants and the linear maps of a constant are of a degree too
        let op =
            real(real::Const::with_cod(Mat([2, 3]), [sort::Real::with_degree([2, 3], 1)]).unwrap());
        assert_eq!(op.cod(), base::mono![r([2, 3], 1)].into());
        let op =
            real(real::Scale::with_dom(Mat([2, 3]), [sort::Real::with_degree([2, 3], 1)]).unwrap());
        assert_eq!(op.dom(), base::mono![r([2, 3], 1)].into());
        let op = real(
            real::Affine::with_dom(
                Mat([2, 3]),
                Mat([2, 4]),
                [sort::Real::with_degree([3, 4], 1)],
            )
            .unwrap(),
        );
        assert_eq!(op.dom(), base::mono![r([3, 4], 1)].into());
        assert_eq!(op.cod(), base::mono![r([2, 4], 1)].into());
    }

    #[test]
    fn division_takes_off_the_degree_of_the_divisor_and_is_partial() {
        let r = |degree| -> sort::Any { sort::Real::with_degree([2, 3], degree).into() };
        let op = real(
            real::Div::with_dom([
                sort::Real::with_degree([2, 3], 2),
                sort::Real::with_degree([2, 3], 1),
            ])
            .unwrap(),
        );
        assert_eq!(op.dom(), base::mono![r(2), r(1)].into());
        assert_eq!(op.cod(), base::mono![r(1)].into());
        assert_eq!(
            real::Div::with_dom([sort::Real::new([2, 3]); 2])
                .unwrap()
                .degrees(),
            [0, 0]
        );
        // a negative degree is not defined
        assert!(
            real::Div::with_dom([
                sort::Real::with_degree([2, 3], 0),
                sort::Real::with_degree([2, 3], 1)
            ])
            .is_err()
        );
        assert!(
            real::Div::with_dom([
                sort::Real::with_degree([2, 3], 1),
                sort::Real::with_degree([2, 3], 1)
            ])
            .is_ok()
        );
    }

    #[test]
    fn negation_keeps_the_degree_of_reals() {
        let op = real(real::Neg::with_dom([sort::Real::with_degree([2, 3], 1); 1]).unwrap());
        let sort: Poly<sort::Any> = base::mono![sort::Real::with_degree([2, 3], 1).into()].into();
        assert_eq!(op.dom(), sort);
        assert_eq!(op.cod(), sort);
        assert_eq!(
            real::Neg::with_dom([sort::Real::new([2, 3]); 1])
                .unwrap()
                .degree(),
            0
        );
    }

    #[test]
    fn scale_maps_matrices_of_the_shape_of_its_constant() {
        let op = real(real::Scale::with_dom(Mat([2, 3]), [sort::Real::new([2, 3])]).unwrap());
        let shape: Poly<sort::Any> = base::mono![sort::Real::new([2, 3]).into()].into();
        assert_eq!(op.dom(), shape);
        assert_eq!(op.cod(), shape);
        assert_eq!(
            real::Scale::with_dom(Mat([2, 3]), [sort::Real::new([2, 3])])
                .unwrap()
                .tensor()
                .0,
            [2, 3]
        );
    }

    #[test]
    fn logics_cast_to_and_from_any() {
        // operations wrap into their logic, and logics into `Any`
        let lra = LRA::Real(
            real::Add::with_dom([sort::Real::new([2, 3]); 2])
                .unwrap()
                .into(),
        );
        assert!(matches!(
            Any::from(lra),
            Any::Real(real::NRA::Add(op)) if op == real::Add::with_dom([sort::Real::new([2, 3]); 2]).unwrap()
        ));
        let lia = LIA::Int(
            int::Sub::with_dom([sort::Int::new([2, 3]); 2])
                .unwrap()
                .into(),
        );
        assert!(matches!(
            Any::from(lia),
            Any::Int(int::NIA::Sub(op)) if op == int::Sub::with_dom([sort::Int::new([2, 3]); 2]).unwrap()
        ));

        // `Any` casts down to the logics holding the operation only
        assert!(
            LRA::try_from(real(
                real::ReLU::with_dom([sort::Real::new([2, 3]); 1]).unwrap()
            ))
            .is_ok()
        );
        assert!(
            LIA::try_from(real(
                real::ReLU::with_dom([sort::Real::new([2, 3]); 1]).unwrap()
            ))
            .is_err()
        );
        assert!(
            LIA::try_from(int(
                int::ReLU::with_dom([sort::Int::new([2, 3]); 1]).unwrap()
            ))
            .is_ok()
        );

        // boolean operations are in every logic
        let op = || boolean(bool::And::with_dom([sort::Bool::new([2, 3]); 2]).unwrap());
        assert!(Prop::try_from(op()).is_ok());
        assert!(NRA::try_from(op()).is_ok());
        assert!(NIA::try_from(op()).is_ok());
        assert!(LRA::try_from(op()).is_ok());
        assert!(LIA::try_from(op()).is_ok());
        assert!(BV::try_from(op()).is_ok());

        // `Prop` holds the boolean operations only
        assert!(
            Prop::try_from(real(
                real::Add::with_dom([sort::Real::new([2, 3]); 2]).unwrap()
            ))
            .is_err()
        );
    }

    #[test]
    fn linear_logics_hold_the_linear_operations_only() {
        // `Mul`, `Div`, `Mod` and `MatMul` are in neither linear logic
        assert!(
            LRA::try_from(real(
                real::MatMul::with_dom([sort::Real::new([2, 3]), sort::Real::new([3, 4])]).unwrap()
            ))
            .is_err()
        );
        assert!(
            LRA::try_from(real(
                real::Mul::with_dom([sort::Real::new([2, 3]); 2]).unwrap()
            ))
            .is_err()
        );
        assert!(
            LRA::try_from(real(
                real::Div::with_dom([sort::Real::new([2, 3]); 2]).unwrap()
            ))
            .is_err()
        );
        assert!(
            LIA::try_from(int(int::MatMul::with_dom([
                sort::Int::new([2, 3]),
                sort::Int::new([3, 4])
            ])
            .unwrap()))
            .is_err()
        );
        assert!(
            LIA::try_from(int(int::Mul::with_dom([sort::Int::new([2, 3]); 2]).unwrap())).is_err()
        );
        assert!(
            LIA::try_from(int(int::Div::with_dom([sort::Int::new([2, 3]); 2]).unwrap())).is_err()
        );
        assert!(
            LIA::try_from(int(int::Mod::with_dom([sort::Int::new([2, 3]); 2]).unwrap())).is_err()
        );

        // but they are in the non-linear ones
        assert!(
            NRA::try_from(real(
                real::MatMul::with_dom([sort::Real::new([2, 3]), sort::Real::new([3, 4])]).unwrap()
            ))
            .is_ok()
        );
        assert!(
            NRA::try_from(real(
                real::Div::with_dom([sort::Real::new([2, 3]); 2]).unwrap()
            ))
            .is_ok()
        );
        assert!(
            NIA::try_from(int(int::Mod::with_dom([sort::Int::new([2, 3]); 2]).unwrap())).is_ok()
        );

        // comparisons are linear
        assert!(
            LRA::try_from(real(
                real::Lt::with_dom([sort::Real::new([2, 3]); 2]).unwrap()
            ))
            .is_ok()
        );
        assert!(
            LIA::try_from(int(int::Ge::with_dom([sort::Int::new([2, 3]); 2]).unwrap())).is_ok()
        );
    }

    #[test]
    fn non_linear_logics_hold_their_theory_only() {
        assert!(
            NRA::try_from(int(int::Add::with_dom([sort::Int::new([2, 3]); 2]).unwrap())).is_err()
        );
        assert!(
            NIA::try_from(real(
                real::Add::with_dom([sort::Real::new([2, 3]); 2]).unwrap()
            ))
            .is_err()
        );
        assert!(
            NRA::try_from(word(
                word::Add::with_dom([sort::Word::new(8, [2, 3]).unwrap(); 2]).unwrap()
            ))
            .is_err()
        );
    }

    #[test]
    fn logics_embed_into_larger_logics() {
        let lra = LRA::Real(
            real::Sub::with_dom([sort::Real::new([2, 3]); 2])
                .unwrap()
                .into(),
        );
        assert!(matches!(
            NRA::from(lra),
            NRA::Real(real::NRA::Sub(op)) if op == real::Sub::with_dom([sort::Real::new([2, 3]); 2]).unwrap()
        ));
        let lia = LIA::Int(
            int::Max::with_dom([sort::Int::new([2, 3]); 2])
                .unwrap()
                .into(),
        );
        assert!(matches!(
            NIA::from(lia),
            NIA::Int(int::NIA::Max(op)) if op == int::Max::with_dom([sort::Int::new([2, 3]); 2]).unwrap()
        ));

        let prop = || {
            Prop::Bool(
                bool::Or::with_dom([sort::Bool::new([2, 3]); 2])
                    .unwrap()
                    .into(),
            )
        };
        assert!(matches!(
            NRA::from(prop()),
            NRA::Bool(bool::Prop::Or(op)) if op == bool::Or::with_dom([sort::Bool::new([2, 3]); 2]).unwrap()
        ));
        assert!(matches!(NIA::from(prop()), NIA::Bool(_)));
        assert!(matches!(LRA::from(prop()), LRA::Bool(_)));
        assert!(matches!(LIA::from(prop()), LIA::Bool(_)));
        assert!(matches!(BV::from(prop()), BV::Bool(_)));
    }

    #[test]
    fn bv_holds_word_and_boolean_operations() {
        assert!(
            BV::try_from(word(
                word::signed::Div::with_dom([sort::Word::new(8, [2, 3]).unwrap(); 2]).unwrap()
            ))
            .is_ok()
        );
        assert!(
            BV::try_from(word(
                word::unsigned::Lt::with_dom([sort::Word::new(8, [2, 3]).unwrap(); 2]).unwrap()
            ))
            .is_ok()
        );
        assert!(
            BV::try_from(word(
                word::Concat::with_dom([
                    sort::Word::new(8, [2, 3]).unwrap(),
                    sort::Word::new(4, [2, 3]).unwrap()
                ])
                .unwrap()
            ))
            .is_ok()
        );
        assert!(
            BV::try_from(int(int::Add::with_dom([sort::Int::new([2, 3]); 2]).unwrap())).is_err()
        );
        assert!(
            BV::try_from(real(
                real::Add::with_dom([sort::Real::new([2, 3]); 2]).unwrap()
            ))
            .is_err()
        );

        // word operations are in none of the other logics
        assert!(
            LIA::try_from(word(
                word::Add::with_dom([sort::Word::new(8, [2, 3]).unwrap(); 2]).unwrap()
            ))
            .is_err()
        );
        assert!(
            Prop::try_from(word(
                word::Not::with_dom([sort::Word::new(8, [2, 3]).unwrap(); 1]).unwrap()
            ))
            .is_err()
        );
    }

    #[test]
    fn word_ops_have_their_boundary_in_sort() {
        let op = word(word::Add::with_dom([sort::Word::new(8, [2, 3]).unwrap(); 2]).unwrap());
        let w: sort::Any = sort::Word::new_unchecked(8, [2, 3]).into();
        let dom: Poly<sort::Any> = base::mono![w, w].into();
        let cod: Poly<sort::Any> = base::mono![w].into();
        assert_eq!(op.dom(), dom);
        assert_eq!(op.cod(), cod);

        // comparisons write the booleans of `bool`
        let op =
            word(word::signed::Lt::with_dom([sort::Word::new(8, [2, 3]).unwrap(); 2]).unwrap());
        let cod: Poly<sort::Any> = base::mono![sort::Bool::new([2, 3]).into()].into();
        assert_eq!(op.dom(), dom);
        assert_eq!(op.cod(), cod);

        // signed and unsigned operations are distinct operations
        assert!(matches!(
            word::BV::<Mat>::from(
                word::signed::Div::with_dom([sort::Word::new(8, [2, 3]).unwrap(); 2]).unwrap()
            ),
            word::BV::SDiv(_)
        ));
        assert!(matches!(
            word::BV::<Mat>::from(
                word::unsigned::Div::with_dom([sort::Word::new(8, [2, 3]).unwrap(); 2]).unwrap()
            ),
            word::BV::UDiv(_)
        ));
    }

    #[test]
    fn comparisons_read_numbers_and_write_booleans() {
        let op = int(int::Lt::with_dom([sort::Int::new([2, 3]); 2]).unwrap());
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
        assert_eq!(And::with_dom([Bool::new([2, 3]); 2]).unwrap().dom(), two(b));
        assert_eq!(And::with_dom([Bool::new([2, 3]); 2]).unwrap().cod(), one(b));
        assert_eq!(Or::with_dom([Bool::new([2, 3]); 2]).unwrap().dom(), two(b));
        assert_eq!(Xor::with_dom([Bool::new([2, 3]); 2]).unwrap().cod(), one(b));
        assert_eq!(Not::with_dom([Bool::new([2, 3]); 1]).unwrap().dom(), one(b));
        assert_eq!(Not::with_dom([Bool::new([2, 3]); 1]).unwrap().cod(), one(b));
        assert_eq!(Eq::with_dom([Bool::new([2, 3]); 2]).unwrap().dom(), two(b));
        assert_eq!(Eq::with_dom([Bool::new([2, 3]); 2]).unwrap().cod(), one(b));
        assert_eq!(Ne::with_dom([Bool::new([2, 3]); 2]).unwrap().dom(), two(b));
        assert_eq!(Ne::with_dom([Bool::new([2, 3]); 2]).unwrap().cod(), one(b));
    }

    #[test]
    fn ops_display_their_name() {
        assert_eq!(
            And::with_dom([Bool::new([1, 1]); 2]).unwrap().to_string(),
            "And"
        );
        assert_eq!(
            Not::with_dom([Bool::new([1, 1]); 1]).unwrap().to_string(),
            "Not"
        );
        assert_eq!(
            Eq::with_dom([Bool::new([1, 1]); 2]).unwrap().to_string(),
            "Eq"
        );
        assert_eq!(
            Ne::with_dom([Bool::new([1, 1]); 2]).unwrap().to_string(),
            "Ne"
        );
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
        assert_eq!(Add::with_dom([Int::new([2, 3]); 2]).unwrap().dom(), two(r));
        assert_eq!(Add::with_dom([Int::new([2, 3]); 2]).unwrap().cod(), one(r));
        assert_eq!(Sub::with_dom([Int::new([2, 3]); 2]).unwrap().dom(), two(r));
        assert_eq!(Mul::with_dom([Int::new([2, 3]); 2]).unwrap().dom(), two(r));
        assert_eq!(Mul::with_dom([Int::new([2, 3]); 2]).unwrap().cod(), one(r));
        assert_eq!(Div::with_dom([Int::new([2, 3]); 2]).unwrap().dom(), two(r));
        assert_eq!(Div::with_dom([Int::new([2, 3]); 2]).unwrap().cod(), one(r));
        assert_eq!(Mod::with_dom([Int::new([2, 3]); 2]).unwrap().dom(), two(r));
        assert_eq!(Mod::with_dom([Int::new([2, 3]); 2]).unwrap().cod(), one(r));
        assert_eq!(Min::with_dom([Int::new([2, 3]); 2]).unwrap().dom(), two(r));
        assert_eq!(Max::with_dom([Int::new([2, 3]); 2]).unwrap().cod(), one(r));
        assert_eq!(ReLU::with_dom([Int::new([2, 3]); 1]).unwrap().dom(), one(r));
        assert_eq!(ReLU::with_dom([Int::new([2, 3]); 1]).unwrap().cod(), one(r));
    }

    #[test]
    fn transpose_swaps_shape() {
        let t = Transpose::with_dom([Int::new([2, 3]); 1]).unwrap();
        assert_eq!(t.dom(), one(Int::new([2, 3])));
        assert_eq!(t.cod(), one(Int::new([3, 2])));
    }

    #[test]
    fn matmul_contracts_inner_size() {
        let p = MatMul::with_dom([Int::new([2, 3]), Int::new([3, 4])]).unwrap();
        let dom: Poly<Int> = mono![Int::new([2, 3]), Int::new([3, 4])].into();
        assert_eq!(p.dom(), dom);
        assert_eq!(p.cod(), one(Int::new([2, 4])));
    }

    #[test]
    fn matmul_checks_inner_size() {
        assert!(MatMul::with_dom([Int::new([2, 3]), Int::new([4, 5])]).is_err());
        let p = MatMul::with_dom([Int::new([2, 3]), Int::new([3, 4])]).unwrap();
        assert_eq!(p.shapes(), [[2, 3], [3, 4]]);
    }

    #[test]
    fn shapes_may_be_empty() {
        assert_eq!(
            Add::with_dom([Int::new([0, 3]); 2]).unwrap().shape(),
            [0, 3]
        );
        assert!(MatMul::with_dom([Int::new([2, 0]), Int::new([0, 4])]).is_ok());
        assert_eq!(
            Transpose::with_dom([Int::new([0, 0]); 1]).unwrap().cod(),
            one(Int::new([0, 0]))
        );
    }

    #[test]
    fn ops_display_their_name() {
        assert_eq!(
            Add::with_dom([Int::new([1, 1]); 2]).unwrap().to_string(),
            "Add"
        );
        assert_eq!(
            Mul::with_dom([Int::new([1, 1]); 2]).unwrap().to_string(),
            "Mul"
        );
        assert_eq!(
            Div::with_dom([Int::new([1, 1]); 2]).unwrap().to_string(),
            "Div"
        );
        assert_eq!(
            Mod::with_dom([Int::new([1, 1]); 2]).unwrap().to_string(),
            "Mod"
        );
        assert_eq!(
            ReLU::with_dom([Int::new([1, 1]); 1]).unwrap().to_string(),
            "ReLU"
        );
        assert_eq!(
            Transpose::with_dom([Int::new([1, 1]); 1])
                .unwrap()
                .to_string(),
            "Transpose"
        );
    }
    #[test]
    fn comparisons_write_booleans() {
        let x: Sort = Int::new([2, 3]).into();
        let b: Sort = Bool::new([2, 3]).into();
        let dom: Poly<Sort> = mono![x, x].into();
        let cod: Poly<Sort> = mono![b].into();
        for (d, c) in [
            (
                Lt::with_dom([Int::new([2, 3]); 2]).unwrap().dom(),
                Lt::with_dom([Int::new([2, 3]); 2]).unwrap().cod(),
            ),
            (
                Le::with_dom([Int::new([2, 3]); 2]).unwrap().dom(),
                Le::with_dom([Int::new([2, 3]); 2]).unwrap().cod(),
            ),
            (
                Gt::with_dom([Int::new([2, 3]); 2]).unwrap().dom(),
                Gt::with_dom([Int::new([2, 3]); 2]).unwrap().cod(),
            ),
            (
                Ge::with_dom([Int::new([2, 3]); 2]).unwrap().dom(),
                Ge::with_dom([Int::new([2, 3]); 2]).unwrap().cod(),
            ),
            (
                Eq::with_dom([Int::new([2, 3]); 2]).unwrap().dom(),
                Eq::with_dom([Int::new([2, 3]); 2]).unwrap().cod(),
            ),
            (
                Ne::with_dom([Int::new([2, 3]); 2]).unwrap().dom(),
                Ne::with_dom([Int::new([2, 3]); 2]).unwrap().cod(),
            ),
        ] {
            assert_eq!(d, dom);
            assert_eq!(c, cod);
        }
        assert_eq!(
            Le::with_dom([Int::new([1, 1]); 2]).unwrap().to_string(),
            "Le"
        );
        assert_eq!(
            Ne::with_dom([Int::new([1, 1]); 2]).unwrap().to_string(),
            "Ne"
        );
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
        assert_eq!(Add::with_dom([Real::new([2, 3]); 2]).unwrap().dom(), two(r));
        assert_eq!(Add::with_dom([Real::new([2, 3]); 2]).unwrap().cod(), one(r));
        assert_eq!(Sub::with_dom([Real::new([2, 3]); 2]).unwrap().dom(), two(r));
        assert_eq!(Mul::with_dom([Real::new([2, 3]); 2]).unwrap().dom(), two(r));
        assert_eq!(Mul::with_dom([Real::new([2, 3]); 2]).unwrap().cod(), one(r));
        assert_eq!(Div::with_dom([Real::new([2, 3]); 2]).unwrap().dom(), two(r));
        assert_eq!(Div::with_dom([Real::new([2, 3]); 2]).unwrap().cod(), one(r));
        assert_eq!(Min::with_dom([Real::new([2, 3]); 2]).unwrap().dom(), two(r));
        assert_eq!(Max::with_dom([Real::new([2, 3]); 2]).unwrap().cod(), one(r));
        assert_eq!(
            ReLU::with_dom([Real::new([2, 3]); 1]).unwrap().dom(),
            one(r)
        );
        assert_eq!(
            ReLU::with_dom([Real::new([2, 3]); 1]).unwrap().cod(),
            one(r)
        );
    }

    #[test]
    fn transpose_swaps_shape() {
        let t = Transpose::with_dom([Real::new([2, 3]); 1]).unwrap();
        assert_eq!(t.dom(), one(Real::new([2, 3])));
        assert_eq!(t.cod(), one(Real::new([3, 2])));
    }

    #[test]
    fn matmul_contracts_inner_size() {
        let p = MatMul::with_dom([Real::new([2, 3]), Real::new([3, 4])]).unwrap();
        let dom: Poly<Real> = mono![Real::new([2, 3]), Real::new([3, 4])].into();
        assert_eq!(p.dom(), dom);
        assert_eq!(p.cod(), one(Real::new([2, 4])));
    }

    #[test]
    fn matmul_checks_inner_size() {
        assert!(MatMul::with_dom([Real::new([2, 3]), Real::new([4, 5])]).is_err());
        let p = MatMul::with_dom([Real::new([2, 3]), Real::new([3, 4])]).unwrap();
        assert_eq!(p.shapes(), [[2, 3], [3, 4]]);
    }

    #[test]
    fn mul_adds_the_degrees_of_its_factors() {
        let r = |d| Real::with_degree([2, 3], d);
        let rates =
            Mul::with_dom([Real::with_degree([2, 3], 1), Real::with_degree([2, 3], 1)]).unwrap();
        assert_eq!(rates.dom(), two(r(1)));
        assert_eq!(rates.cod(), one(r(2)));
        let scaled =
            Mul::with_dom([Real::with_degree([2, 3], 1), Real::with_degree([2, 3], 0)]).unwrap();
        let dom: Poly<Real> = mono![r(1), r(0)].into();
        assert_eq!(scaled.dom(), dom);
        assert_eq!(scaled.cod(), one(r(1)));
        assert_eq!(
            Mul::with_dom([Real::new([2, 3]); 2]).unwrap(),
            Mul::with_dom([Real::with_degree([2, 3], 0), Real::with_degree([2, 3], 0)]).unwrap()
        );
    }

    #[test]
    fn mul_checks_the_degree_of_its_product() {
        assert!(
            Mul::with_dom([
                Real::with_degree([2, 3], u8::MAX),
                Real::with_degree([2, 3], 1)
            ])
            .is_err()
        );
        assert!(
            Mul::with_dom([
                Real::with_degree([2, 3], u8::MAX),
                Real::with_degree([2, 3], 0)
            ])
            .is_ok()
        );
        assert_eq!(
            Mul::with_dom([Real::with_degree([2, 3], 1), Real::with_degree([2, 3], 0)])
                .unwrap()
                .degrees(),
            [1, 0]
        );
    }

    #[test]
    fn sums_keep_the_degree_of_their_terms() {
        let r = Real::with_degree([2, 3], 1);
        assert_eq!(
            Add::with_dom([Real::with_degree([2, 3], 1); 2])
                .unwrap()
                .dom(),
            two(r)
        );
        assert_eq!(
            Add::with_dom([Real::with_degree([2, 3], 1); 2])
                .unwrap()
                .cod(),
            one(r)
        );
        assert_eq!(
            Sub::with_dom([Real::with_degree([2, 3], 1); 2])
                .unwrap()
                .dom(),
            two(r)
        );
        assert_eq!(
            Sub::with_dom([Real::with_degree([2, 3], 1); 2])
                .unwrap()
                .cod(),
            one(r)
        );
        assert_eq!(
            Add::with_dom([Real::new([2, 3]); 2]).unwrap(),
            Add::with_dom([Real::with_degree([2, 3], 0); 2]).unwrap()
        );
    }

    #[test]
    fn shapes_may_be_empty() {
        assert_eq!(
            Add::with_dom([Real::new([0, 3]); 2]).unwrap().shape(),
            [0, 3]
        );
        assert!(MatMul::with_dom([Real::new([2, 0]), Real::new([0, 4])]).is_ok());
        assert_eq!(
            Transpose::with_dom([Real::new([0, 0]); 1]).unwrap().cod(),
            one(Real::new([0, 0]))
        );
    }

    #[test]
    fn ops_display_their_name() {
        assert_eq!(
            Add::with_dom([Real::new([1, 1]); 2]).unwrap().to_string(),
            "Add"
        );
        assert_eq!(
            Mul::with_dom([Real::new([1, 1]); 2]).unwrap().to_string(),
            "Mul"
        );
        assert_eq!(
            Div::with_dom([Real::new([1, 1]); 2]).unwrap().to_string(),
            "Div"
        );
        assert_eq!(
            ReLU::with_dom([Real::new([1, 1]); 1]).unwrap().to_string(),
            "ReLU"
        );
        assert_eq!(
            Transpose::with_dom([Real::new([1, 1]); 1])
                .unwrap()
                .to_string(),
            "Transpose"
        );
    }
    #[test]
    fn comparisons_write_booleans() {
        let x: Sort = Real::new([2, 3]).into();
        let b: Sort = Bool::new([2, 3]).into();
        let dom: Poly<Sort> = mono![x, x].into();
        let cod: Poly<Sort> = mono![b].into();
        for (d, c) in [
            (
                Lt::with_dom([Real::new([2, 3]); 2]).unwrap().dom(),
                Lt::with_dom([Real::new([2, 3]); 2]).unwrap().cod(),
            ),
            (
                Le::with_dom([Real::new([2, 3]); 2]).unwrap().dom(),
                Le::with_dom([Real::new([2, 3]); 2]).unwrap().cod(),
            ),
            (
                Gt::with_dom([Real::new([2, 3]); 2]).unwrap().dom(),
                Gt::with_dom([Real::new([2, 3]); 2]).unwrap().cod(),
            ),
            (
                Ge::with_dom([Real::new([2, 3]); 2]).unwrap().dom(),
                Ge::with_dom([Real::new([2, 3]); 2]).unwrap().cod(),
            ),
            (
                Eq::with_dom([Real::new([2, 3]); 2]).unwrap().dom(),
                Eq::with_dom([Real::new([2, 3]); 2]).unwrap().cod(),
            ),
            (
                Ne::with_dom([Real::new([2, 3]); 2]).unwrap().dom(),
                Ne::with_dom([Real::new([2, 3]); 2]).unwrap().cod(),
            ),
        ] {
            assert_eq!(d, dom);
            assert_eq!(c, cod);
        }
        assert_eq!(
            Le::with_dom([Real::new([1, 1]); 2]).unwrap().to_string(),
            "Le"
        );
        assert_eq!(
            Ne::with_dom([Real::new([1, 1]); 2]).unwrap().to_string(),
            "Ne"
        );
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
        assert_eq!(r.to_string(), "Real([2, 3])");
    }

    #[test]
    fn real_tangent_raises_degree() {
        let r = Real::new([2, 3]);
        assert_eq!(T(r).degree(), 1);
        assert_eq!(T(r).shape(), [2, 3]);
        assert_eq!(T(T(r)).degree(), 2);
        assert_eq!(T(r).to_string(), "T(Real([2, 3]))");
        assert_eq!(T(T(r)).to_string(), "T^2(Real([2, 3]))");
    }

    #[test]
    fn int_shape_and_display() {
        let r = Int::new([2, 3]);
        assert_eq!(r.shape(), [2, 3]);
        assert_eq!(r.to_string(), "Int([2, 3])");
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
        assert_eq!(b.to_string(), "Bool([2, 3])");
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
        assert_eq!(x.to_string(), "Word8([2, 3])");
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
        assert_eq!(Words::from(x).to_string(), "Word8([2, 3])");
        assert_eq!(Words::from(Bool::new([2, 3])).to_string(), "Bool([2, 3])");
        assert_eq!(Words::from(Zero).to_string(), "Zero()");
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
        assert_eq!(Zero.to_string(), "Zero()");
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
        assert_eq!(
            Add::with_dom([Word::new(8, [2, 3]).unwrap(); 2])
                .unwrap()
                .dom(),
            two(x)
        );
        assert_eq!(
            Add::with_dom([Word::new(8, [2, 3]).unwrap(); 2])
                .unwrap()
                .cod(),
            one(x)
        );
        assert_eq!(
            Xnor::with_dom([Word::new(8, [2, 3]).unwrap(); 2])
                .unwrap()
                .cod(),
            one(x)
        );
        assert_eq!(
            Neg::with_dom([Word::new(8, [2, 3]).unwrap(); 1])
                .unwrap()
                .dom(),
            one(x)
        );
        assert_eq!(
            Not::with_dom([Word::new(8, [2, 3]).unwrap(); 1])
                .unwrap()
                .cod(),
            one(x)
        );
        assert_eq!(
            RotateLeft::with_dom(3, [Word::new(8, [2, 3]).unwrap(); 1])
                .unwrap()
                .dom(),
            one(x)
        );
        assert_eq!(
            RotateRight::with_dom(3, [Word::new(8, [2, 3]).unwrap(); 1])
                .unwrap()
                .cod(),
            one(x)
        );
        assert_eq!(
            signed::Div::with_dom([Word::new(8, [2, 3]).unwrap(); 2])
                .unwrap()
                .dom(),
            two(x)
        );
        assert_eq!(
            signed::Mod::with_dom([Word::new(8, [2, 3]).unwrap(); 2])
                .unwrap()
                .cod(),
            one(x)
        );
        assert_eq!(
            unsigned::Rem::with_dom([Word::new(8, [2, 3]).unwrap(); 2])
                .unwrap()
                .cod(),
            one(x)
        );
        assert_eq!(
            unsigned::Shr::with_dom([Word::new(8, [2, 3]).unwrap(); 2])
                .unwrap()
                .dom(),
            two(x)
        );
    }

    #[test]
    fn comparisons_write_booleans() {
        let x = w(8, [2, 3]);
        assert_eq!(
            Eq::with_dom([Word::new(8, [2, 3]).unwrap(); 2])
                .unwrap()
                .dom(),
            two(x)
        );
        assert_eq!(
            Eq::with_dom([Word::new(8, [2, 3]).unwrap(); 2])
                .unwrap()
                .cod(),
            one(b([2, 3]))
        );
        assert_eq!(
            Ne::with_dom([Word::new(8, [2, 3]).unwrap(); 2])
                .unwrap()
                .cod(),
            one(b([2, 3]))
        );
        assert_eq!(
            signed::Lt::with_dom([Word::new(8, [2, 3]).unwrap(); 2])
                .unwrap()
                .dom(),
            two(x)
        );
        assert_eq!(
            signed::Lt::with_dom([Word::new(8, [2, 3]).unwrap(); 2])
                .unwrap()
                .cod(),
            one(b([2, 3]))
        );
        assert_eq!(
            unsigned::Ge::with_dom([Word::new(8, [2, 3]).unwrap(); 2])
                .unwrap()
                .cod(),
            one(b([2, 3]))
        );
    }

    #[test]
    fn width_changing_boundaries() {
        assert_eq!(
            Extract::with_dom(7, 4, [Word::new(16, [2, 3]).unwrap(); 1])
                .unwrap()
                .dom(),
            one(w(16, [2, 3]))
        );
        assert_eq!(
            Extract::with_dom(7, 4, [Word::new(16, [2, 3]).unwrap(); 1])
                .unwrap()
                .cod(),
            one(w(4, [2, 3]))
        );
        assert_eq!(
            Extract::with_dom(0, 0, [Word::new(8, [1, 1]).unwrap(); 1])
                .unwrap()
                .cod(),
            one(w(1, [1, 1]))
        );

        let dom: Poly<Sort> = mono![w(8, [2, 3]), w(4, [2, 3])].into();
        assert_eq!(
            Concat::with_dom([Word::new(8, [2, 3]).unwrap(), Word::new(4, [2, 3]).unwrap()])
                .unwrap()
                .dom(),
            dom
        );
        assert_eq!(
            Concat::with_dom([Word::new(8, [2, 3]).unwrap(), Word::new(4, [2, 3]).unwrap()])
                .unwrap()
                .cod(),
            one(w(12, [2, 3]))
        );

        assert_eq!(
            unsigned::Extend::with_dom(8, [Word::new(8, [2, 3]).unwrap(); 1])
                .unwrap()
                .cod(),
            one(w(16, [2, 3]))
        );
        assert_eq!(
            signed::Extend::with_dom(24, [Word::new(8, [2, 3]).unwrap(); 1])
                .unwrap()
                .cod(),
            one(w(32, [2, 3]))
        );
        assert_eq!(
            Repeat::with_dom(3, [Word::new(4, [2, 3]).unwrap(); 1])
                .unwrap()
                .cod(),
            one(w(12, [2, 3]))
        );
    }

    #[test]
    fn ops_display_their_name_and_indices() {
        assert_eq!(
            Add::with_dom([Word::new(8, [1, 1]).unwrap(); 2])
                .unwrap()
                .to_string(),
            "Add"
        );
        assert_eq!(
            Ne::with_dom([Word::new(8, [1, 1]).unwrap(); 2])
                .unwrap()
                .to_string(),
            "Ne"
        );
        assert_eq!(
            Extract::with_dom(7, 4, [Word::new(8, [1, 1]).unwrap(); 1])
                .unwrap()
                .to_string(),
            "Extract[7:4]"
        );
        assert_eq!(
            RotateLeft::with_dom(3, [Word::new(8, [1, 1]).unwrap(); 1])
                .unwrap()
                .to_string(),
            "RotateLeft(3)"
        );
        assert_eq!(
            Repeat::with_dom(2, [Word::new(8, [1, 1]).unwrap(); 1])
                .unwrap()
                .to_string(),
            "Repeat(2)"
        );
        assert_eq!(
            signed::Div::with_dom([Word::new(8, [1, 1]).unwrap(); 2])
                .unwrap()
                .to_string(),
            "SDiv"
        );
        assert_eq!(
            unsigned::Div::with_dom([Word::new(8, [1, 1]).unwrap(); 2])
                .unwrap()
                .to_string(),
            "UDiv"
        );
        assert_eq!(
            signed::Shr::with_dom([Word::new(8, [1, 1]).unwrap(); 2])
                .unwrap()
                .to_string(),
            "AShr"
        );
        assert_eq!(
            unsigned::Shr::with_dom([Word::new(8, [1, 1]).unwrap(); 2])
                .unwrap()
                .to_string(),
            "LShr"
        );
        assert_eq!(
            signed::Ge::with_dom([Word::new(8, [1, 1]).unwrap(); 2])
                .unwrap()
                .to_string(),
            "SGe"
        );
        assert_eq!(
            signed::Extend::with_dom(8, [Word::new(8, [1, 1]).unwrap(); 1])
                .unwrap()
                .to_string(),
            "SignExtend(8)"
        );
        assert_eq!(
            unsigned::Extend::with_dom(8, [Word::new(8, [1, 1]).unwrap(); 1])
                .unwrap()
                .to_string(),
            "ZeroExtend(8)"
        );
    }

    #[test]
    fn constructors_check_their_sizes() {
        // words have at least one bit: the sort `Word` checks it, so no
        // operation reads or writes words of no bits

        // extraction needs `j ≤ i < width`
        assert!(Extract::with_dom(7, 0, [Word::new(8, [2, 3]).unwrap(); 1]).is_ok());
        assert!(Extract::with_dom(8, 0, [Word::new(8, [2, 3]).unwrap(); 1]).is_err());
        assert!(Extract::with_dom(2, 3, [Word::new(8, [2, 3]).unwrap(); 1]).is_err());

        // repetition needs at least one copy
        assert!(Repeat::with_dom(0, [Word::new(8, [2, 3]).unwrap(); 1]).is_err());

        // but shapes may be empty
        assert!(Add::with_dom([Word::new(8, [0, 0]).unwrap(); 2]).is_ok());
    }

    #[test]
    fn getters_read_the_fields() {
        let e = Extract::with_dom(7, 4, [Word::new(8, [2, 3]).unwrap(); 1]).unwrap();
        assert_eq!((e.i(), e.j(), e.width(), e.shape()), (7, 4, 8, [2, 3]));
        let c = Concat::with_dom([Word::new(8, [2, 3]).unwrap(), Word::new(4, [2, 3]).unwrap()])
            .unwrap();
        assert_eq!((c.widths(), c.shape()), ([8, 4], [2, 3]));
        let r = RotateRight::with_dom(3, [Word::new(8, [2, 3]).unwrap(); 1]).unwrap();
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
        assert!(
            Concat::with_dom([
                Word::new(max, [1, 1]).unwrap(),
                Word::new(1, [1, 1]).unwrap()
            ])
            .is_err()
        );
        assert_eq!(
            Concat::with_dom([
                Word::new(max - 1, [1, 1]).unwrap(),
                Word::new(1, [1, 1]).unwrap()
            ])
            .unwrap()
            .cod(),
            one(w(max, [1, 1]))
        );

        // `Repeat(i, width)` writes words of `i · width` bits
        assert!(Repeat::with_dom(max / 2 + 1, [Word::new(2, [1, 1]).unwrap(); 1]).is_err());
        assert_eq!(
            Repeat::with_dom(max / 2, [Word::new(2, [1, 1]).unwrap(); 1])
                .unwrap()
                .cod(),
            one(w(max - 1, [1, 1]))
        );

        // `Extend(i, width)` writes words of `width + i` bits
        assert!(signed::Extend::with_dom(1, [Word::new(max, [1, 1]).unwrap(); 1]).is_err());
        assert!(unsigned::Extend::with_dom(max, [Word::new(1, [1, 1]).unwrap(); 1]).is_err());
        assert_eq!(
            unsigned::Extend::with_dom(max - 1, [Word::new(1, [1, 1]).unwrap(); 1])
                .unwrap()
                .cod(),
            one(w(max, [1, 1]))
        );
    }
}

/// Tests for the `with_dom` constructors: each builds the operation whose
/// domain is the one given, and fails on a domain of no instance.
mod with_dom {
    use base::boundary::{Boundary, Mono, Poly};
    use theory::tensor::sort::{Bool, Int, Real, Word};
    use theory::tensor::*;

    #[derive(Debug)]
    struct Mat([usize; 2]);

    impl Tensor for Mat {
        fn shape(&self) -> [usize; 2] {
            self.0
        }
    }

    /// Fails unless `op` reads exactly `dom`.
    fn reads<S, B>(op: B, dom: impl IntoIterator<Item = S>)
    where
        S: Clone + PartialEq + std::fmt::Debug,
        B: Boundary<S>,
    {
        let expected: Poly<S> = dom.into_iter().collect::<Mono<S>>().into();
        assert_eq!(op.dom(), expected);
    }

    fn word(width: usize, shape: [usize; 2]) -> Word {
        Word::new_unchecked(width, shape)
    }

    #[test]
    fn real_operations_read_their_domain() {
        let x = Real::with_degree([2, 3], 1);
        let y = Real::with_degree([2, 3], 2);
        reads(real::Add::with_dom([x, x]).unwrap(), [x, x]);
        reads(real::Neg::with_dom([y]).unwrap(), [y]);
        reads(real::Transpose::with_dom([x]).unwrap(), [x]);
        reads(real::Mul::with_dom([x, y]).unwrap(), [x, y]);
        reads(real::Div::with_dom([y, x]).unwrap(), [y, x]);
        let z = Real::new([3, 4]);
        reads(real::MatMul::with_dom([x, z]).unwrap(), [x, z]);
        reads(real::Scale::with_dom(Mat([2, 3]), [y]).unwrap(), [y]);
        reads(
            real::Affine::with_dom(Mat([5, 2]), Mat([5, 3]), [x]).unwrap(),
            [x],
        );
        // `A` of shape `[5, 2]` takes `x` of shape `[2, 3]` to `[5, 3]`
        let l = real::Linear::with_dom(Mat([5, 2]), [x]).unwrap();
        reads(real::Linear::with_dom(Mat([5, 2]), [x]).unwrap(), [x]);
        assert_eq!(l.cod(), base::mono![Real::with_degree([5, 3], 1)].into());
        let c = Bool::new([2, 3]);
        let dom = [c.into(), x.into(), x.into()];
        reads(real::Ite::with_dom(dom).unwrap(), dom);
        let cmp = real::Lt::with_dom([y, y]).unwrap();
        assert_eq!(
            cmp,
            real::Lt::with_dom([Real::with_degree([2, 3], 2); 2]).unwrap()
        );
    }

    #[test]
    fn real_operations_reject_domains_of_no_instance() {
        let x = Real::new([2, 3]);
        let y = Real::with_degree([2, 3], 1);
        let z = Real::new([3, 2]);
        assert!(real::Add::with_dom([x, y]).is_err());
        assert!(real::Eq::with_dom([x, z]).is_err());
        assert!(real::Mul::with_dom([x, z]).is_err());
        // the degree of the divisor exceeds that of the dividend
        assert!(real::Div::with_dom([x, y]).is_err());
        assert!(real::MatMul::with_dom([x, x]).is_err());
        assert!(real::Scale::with_dom(Mat([3, 2]), [x]).is_err());
        assert!(real::Affine::with_dom(Mat([5, 2]), Mat([5, 3]), [z]).is_err());
        assert!(real::Linear::with_dom(Mat([5, 2]), [z]).is_err());
        let c = Bool::new([3, 2]);
        assert!(real::Ite::with_dom([c.into(), x.into(), x.into()]).is_err());
        assert!(real::Ite::with_dom([x.into(), x.into(), x.into()]).is_err());
        let e = real::Add::with_dom([x, y]).unwrap_err();
        assert_eq!(
            e.0,
            "expected a domain of 2 equal sorts, got [Real([2, 3]), T(Real([2, 3]))]"
        );
    }

    #[test]
    fn int_operations_read_their_domain() {
        let x = Int::new([2, 3]);
        let z = Int::new([3, 1]);
        reads(int::Mod::with_dom([x, x]).unwrap(), [x, x]);
        reads(int::Argmax::with_dom([x]).unwrap(), [x]);
        reads(int::MatMul::with_dom([x, z]).unwrap(), [x, z]);
        reads(int::Scale::with_dom(Mat([2, 3]), [x]).unwrap(), [x]);
        reads(
            int::Affine::with_dom(Mat([4, 2]), Mat([4, 3]), [x]).unwrap(),
            [x],
        );
        reads(int::Linear::with_dom(Mat([4, 2]), [x]).unwrap(), [x]);
        assert!(int::Linear::with_dom(Mat([4, 3]), [x]).is_err());
        let dom = [Bool::new([2, 3]).into(), x.into(), x.into()];
        reads(int::Ite::with_dom(dom).unwrap(), dom);
        assert!(int::Sub::with_dom([x, z]).is_err());
        assert!(int::MatMul::with_dom([z, x]).is_err());
        assert!(int::Ite::with_dom([Bool::new([2, 3]).into(), x.into(), z.into()]).is_err());
    }

    #[test]
    fn bool_operations_read_their_domain() {
        let x = Bool::new([1, 4]);
        reads(bool::And::with_dom([x, x]).unwrap(), [x, x]);
        reads(bool::Not::with_dom([x]).unwrap(), [x]);
        reads(bool::Ite::with_dom([x, x, x]).unwrap(), [x, x, x]);
        assert!(bool::Ite::with_dom([x, x, Bool::new([4, 1])]).is_err());
    }

    #[test]
    fn word_operations_read_their_domain() {
        let x = word(8, [2, 2]);
        let y = word(4, [2, 2]);
        let dom = |ws: &[Word]| ws.iter().map(|&w| sort::Words::from(w)).collect::<Vec<_>>();
        reads(word::Add::with_dom([x, x]).unwrap(), dom(&[x, x]));
        reads(word::Not::with_dom([x]).unwrap(), dom(&[x]));
        reads(word::Concat::with_dom([x, y]).unwrap(), dom(&[x, y]));
        reads(word::Extract::with_dom(5, 2, [x]).unwrap(), dom(&[x]));
        reads(word::Repeat::with_dom(3, [y]).unwrap(), dom(&[y]));
        reads(word::RotateLeft::with_dom(1, [x]).unwrap(), dom(&[x]));
        reads(word::signed::Rem::with_dom([y, y]).unwrap(), dom(&[y, y]));
        reads(word::signed::Abs::with_dom([y]).unwrap(), dom(&[y]));
        reads(word::signed::Extend::with_dom(8, [x]).unwrap(), dom(&[x]));
        reads(word::unsigned::Ge::with_dom([x, x]).unwrap(), dom(&[x, x]));
        reads(word::unsigned::Extend::with_dom(8, [y]).unwrap(), dom(&[y]));
        let ite = [Bool::new([2, 2]).into(), x.into(), x.into()];
        reads(word::Ite::with_dom(ite).unwrap(), ite);

        assert!(word::Add::with_dom([x, y]).is_err());
        assert!(word::Concat::with_dom([x, word(4, [2, 1])]).is_err());
        assert!(word::Extract::with_dom(8, 2, [x]).is_err());
        assert!(word::Repeat::with_dom(0, [x]).is_err());
        assert!(word::Ite::with_dom([Bool::new([2, 2]).into(), x.into(), y.into()]).is_err());
    }
}

/// Tests for the `with_cod` constructors: each builds the operation whose
/// codomain is the one given, when the codomain determines it, and fails on
/// a codomain of no instance.
mod with_cod {
    use base::boundary::{Boundary, Mono, Poly};
    use theory::tensor::sort::{Bool, Int, Real, Word};
    use theory::tensor::*;

    #[derive(Debug)]
    struct Mat([usize; 2]);

    impl Tensor for Mat {
        fn shape(&self) -> [usize; 2] {
            self.0
        }
    }

    /// Fails unless `op` writes exactly `cod`.
    fn writes<S, B>(op: B, cod: impl IntoIterator<Item = S>)
    where
        S: Clone + PartialEq + std::fmt::Debug,
        B: Boundary<S>,
    {
        let expected: Poly<S> = cod.into_iter().collect::<Mono<S>>().into();
        assert_eq!(op.cod(), expected);
    }

    fn word(width: usize, shape: [usize; 2]) -> sort::Words {
        Word::new_unchecked(width, shape).into()
    }

    #[test]
    fn real_operations_write_their_codomain() {
        let x = Real::with_degree([2, 3], 1);
        writes(real::Zero::with_cod([x]).unwrap(), [x]);
        writes(real::Add::with_cod([x]).unwrap(), [x]);
        writes(real::ReLU::with_cod([x]).unwrap(), [x]);
        writes(real::Transpose::with_cod([x]).unwrap(), [x]);
        assert_eq!(
            real::Transpose::with_cod([x]).unwrap(),
            real::Transpose::with_dom([Real::with_degree([3, 2], 1); 1]).unwrap()
        );
        writes(real::Ite::with_cod([x]).unwrap(), [x.into()]);
        writes(real::Scale::with_cod(Mat([2, 3]), [x]).unwrap(), [x]);
        writes(
            real::Affine::with_cod(Mat([2, 5]), Mat([2, 3]), [x]).unwrap(),
            [x],
        );
        // `A` of shape `[2, 5]` writes `[2, 3]` from `[5, 3]`, of the same degree
        let l = real::Linear::with_cod(Mat([2, 5]), [x]).unwrap();
        writes(real::Linear::with_cod(Mat([2, 5]), [x]).unwrap(), [x]);
        assert_eq!(l.dom(), base::mono![Real::with_degree([5, 3], 1)].into());
        assert!(real::Linear::with_cod(Mat([3, 5]), [x]).is_err());
        assert!(real::Scale::with_cod(Mat([3, 2]), [x]).is_err());
        let e = real::Affine::with_cod(Mat([3, 5]), Mat([3, 3]), [x]).unwrap_err();
        assert_eq!(
            e.0,
            "expected a codomain of 1 sort of shape [3, 3], got [T(Real([2, 3]))]"
        );
    }

    #[test]
    fn int_operations_write_their_codomain() {
        let x = Int::new([2, 3]);
        let b = Bool::new([2, 3]);
        writes(int::Mod::with_cod([x]).unwrap(), [x]);
        writes(int::Transpose::with_cod([x]).unwrap(), [x]);
        writes(int::Ite::with_cod([x]).unwrap(), [x.into()]);
        writes(int::Le::with_cod([b]).unwrap(), [b.into()]);
        writes(int::Scale::with_cod(Mat([2, 3]), [x]).unwrap(), [x]);
        writes(
            int::Affine::with_cod(Mat([2, 4]), Mat([2, 3]), [x]).unwrap(),
            [x],
        );
        assert!(int::Affine::with_cod(Mat([3, 4]), Mat([3, 3]), [x]).is_err());
        writes(int::Linear::with_cod(Mat([2, 4]), [x]).unwrap(), [x]);
        assert!(int::Linear::with_cod(Mat([3, 4]), [x]).is_err());
    }

    #[test]
    fn bool_operations_write_their_codomain() {
        let x = Bool::new([1, 4]);
        writes(bool::Xor::with_cod([x]).unwrap(), [x]);
        writes(bool::Not::with_cod([x]).unwrap(), [x]);
        writes(bool::Ite::with_cod([x]).unwrap(), [x]);
    }

    #[test]
    fn word_operations_write_their_codomain() {
        let w = |width| Word::new_unchecked(width, [2, 2]);
        writes(word::Nand::with_cod([w(8)]).unwrap(), [word(8, [2, 2])]);
        writes(word::Ite::with_cod([w(8)]).unwrap(), [word(8, [2, 2])]);
        writes(
            word::RotateRight::with_cod(3, [w(8)]).unwrap(),
            [word(8, [2, 2])],
        );
        writes(
            word::signed::Mod::with_cod([w(8)]).unwrap(),
            [word(8, [2, 2])],
        );
        writes(
            word::unsigned::Shr::with_cod([w(8)]).unwrap(),
            [word(8, [2, 2])],
        );

        let repeat = word::Repeat::with_cod(3, [w(12)]).unwrap();
        assert_eq!(
            repeat,
            word::Repeat::with_dom(3, [Word::new(4, [2, 2]).unwrap(); 1]).unwrap()
        );
        writes(repeat, [word(12, [2, 2])]);
        assert!(word::Repeat::with_cod(5, [w(12)]).is_err());
        assert!(word::Repeat::with_cod(0, [w(12)]).is_err());

        let extend = word::signed::Extend::with_cod(3, [w(8)]).unwrap();
        assert_eq!(
            extend,
            word::signed::Extend::with_dom(3, [Word::new(5, [2, 2]).unwrap(); 1]).unwrap()
        );
        writes(extend, [word(8, [2, 2])]);
        let extend = word::unsigned::Extend::with_cod(3, [w(8)]).unwrap();
        writes(extend, [word(8, [2, 2])]);
        assert!(word::signed::Extend::with_cod(8, [w(8)]).is_err());
        assert!(word::unsigned::Extend::with_cod(9, [w(8)]).is_err());
    }
}
