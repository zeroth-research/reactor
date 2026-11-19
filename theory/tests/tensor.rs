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
        let op = real(real::MatMul([2, 3], [3, 4]));
        let dom: Poly<sort::Any> = base::mono![
            sort::Real::new([2, 3]).into(),
            sort::Real::new([3, 4]).into()
        ]
        .into();
        let cod: Poly<sort::Any> = base::mono![sort::Real::new([2, 4]).into()].into();
        assert_eq!(op.dom(), dom);
        assert_eq!(op.cod(), cod);

        let op = int(int::Transpose([2, 3]));
        let cod: Poly<sort::Any> = base::mono![sort::Int::new([3, 2]).into()].into();
        assert_eq!(op.cod(), cod);

        let op = boolean(bool::Not([2, 3]));
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
        assert!(real::LRA::try_from(real::NRA::<Mat>::from(real::Add([2, 3]))).is_ok());
        assert!(real::LRA::try_from(real::NRA::<Mat>::from(real::Lt([2, 3]))).is_ok());
        assert!(real::LRA::try_from(real::NRA::<Mat>::from(real::Mul([2, 3]))).is_err());
        assert!(real::LRA::try_from(real::NRA::<Mat>::from(real::MatMul([2, 3], [3, 4]))).is_err());
        assert!(int::LIA::try_from(int::NIA::<Mat>::from(int::Mod([2, 3]))).is_err());
        assert!(int::LIA::try_from(int::NIA::<Mat>::from(int::ReLU([2, 3]))).is_ok());
    }

    #[test]
    fn logics_cast_to_and_from_any() {
        // operations wrap into their logic, and logics into `Any`
        let lra = LRA::Real(real::Add([2, 3]).into());
        assert!(matches!(
            Any::from(lra),
            Any::Real(real::NRA::Add(real::Add([2, 3])))
        ));
        let lia = LIA::Int(int::Sub([2, 3]).into());
        assert!(matches!(
            Any::from(lia),
            Any::Int(int::NIA::Sub(int::Sub([2, 3])))
        ));

        // `Any` casts down to the logics holding the operation only
        assert!(LRA::try_from(real(real::ReLU([2, 3]))).is_ok());
        assert!(LIA::try_from(real(real::ReLU([2, 3]))).is_err());
        assert!(LIA::try_from(int(int::ReLU([2, 3]))).is_ok());

        // boolean operations are in every logic
        let op = || boolean(bool::And([2, 3]));
        assert!(Prop::try_from(op()).is_ok());
        assert!(NRA::try_from(op()).is_ok());
        assert!(NIA::try_from(op()).is_ok());
        assert!(LRA::try_from(op()).is_ok());
        assert!(LIA::try_from(op()).is_ok());
        assert!(BV::try_from(op()).is_ok());

        // `Prop` holds the boolean operations only
        assert!(Prop::try_from(real(real::Add([2, 3]))).is_err());
    }

    #[test]
    fn linear_logics_hold_the_linear_operations_only() {
        // `Mul`, `Div`, `Mod` and `MatMul` are in neither linear logic
        assert!(LRA::try_from(real(real::MatMul([2, 3], [3, 4]))).is_err());
        assert!(LRA::try_from(real(real::Mul([2, 3]))).is_err());
        assert!(LRA::try_from(real(real::Div([2, 3]))).is_err());
        assert!(LIA::try_from(int(int::MatMul([2, 3], [3, 4]))).is_err());
        assert!(LIA::try_from(int(int::Mul([2, 3]))).is_err());
        assert!(LIA::try_from(int(int::Div([2, 3]))).is_err());
        assert!(LIA::try_from(int(int::Mod([2, 3]))).is_err());

        // but they are in the non-linear ones
        assert!(NRA::try_from(real(real::MatMul([2, 3], [3, 4]))).is_ok());
        assert!(NRA::try_from(real(real::Div([2, 3]))).is_ok());
        assert!(NIA::try_from(int(int::Mod([2, 3]))).is_ok());

        // comparisons are linear
        assert!(LRA::try_from(real(real::Lt([2, 3]))).is_ok());
        assert!(LIA::try_from(int(int::Ge([2, 3]))).is_ok());
    }

    #[test]
    fn non_linear_logics_hold_their_theory_only() {
        assert!(NRA::try_from(int(int::Add([2, 3]))).is_err());
        assert!(NIA::try_from(real(real::Add([2, 3]))).is_err());
        assert!(NRA::try_from(word(word::Add(8, [2, 3]))).is_err());
    }

    #[test]
    fn logics_embed_into_larger_logics() {
        let lra = LRA::Real(real::Sub([2, 3]).into());
        assert!(matches!(
            NRA::from(lra),
            NRA::Real(real::NRA::Sub(real::Sub([2, 3])))
        ));
        let lia = LIA::Int(int::Max([2, 3]).into());
        assert!(matches!(
            NIA::from(lia),
            NIA::Int(int::NIA::Max(int::Max([2, 3])))
        ));

        let prop = || Prop::Bool(bool::Or([2, 3]).into());
        assert!(matches!(
            NRA::from(prop()),
            NRA::Bool(bool::Prop::Or(bool::Or([2, 3])))
        ));
        assert!(matches!(NIA::from(prop()), NIA::Bool(_)));
        assert!(matches!(LRA::from(prop()), LRA::Bool(_)));
        assert!(matches!(LIA::from(prop()), LIA::Bool(_)));
        assert!(matches!(BV::from(prop()), BV::Bool(_)));
    }

    #[test]
    fn bv_holds_word_and_boolean_operations() {
        assert!(BV::try_from(word(word::signed::Div(8, [2, 3]))).is_ok());
        assert!(BV::try_from(word(word::unsigned::Lt(8, [2, 3]))).is_ok());
        assert!(BV::try_from(word(word::Concat(8, 4, [2, 3]))).is_ok());
        assert!(BV::try_from(int(int::Add([2, 3]))).is_err());
        assert!(BV::try_from(real(real::Add([2, 3]))).is_err());

        // word operations are in none of the other logics
        assert!(LIA::try_from(word(word::Add(8, [2, 3]))).is_err());
        assert!(Prop::try_from(word(word::Not(8, [2, 3]))).is_err());
    }

    #[test]
    fn word_ops_have_their_boundary_in_sort() {
        let op = word(word::Add(8, [2, 3]));
        let w: sort::Any = sort::Word::new_unchecked(8, [2, 3]).into();
        let dom: Poly<sort::Any> = base::mono![w, w].into();
        let cod: Poly<sort::Any> = base::mono![w].into();
        assert_eq!(op.dom(), dom);
        assert_eq!(op.cod(), cod);

        // comparisons write the booleans of `bool`
        let op = word(word::signed::Lt(8, [2, 3]));
        let cod: Poly<sort::Any> = base::mono![sort::Bool::new([2, 3]).into()].into();
        assert_eq!(op.dom(), dom);
        assert_eq!(op.cod(), cod);

        // signed and unsigned operations are distinct operations
        assert!(matches!(
            word::BV::<Mat>::from(word::signed::Div(8, [2, 3])),
            word::BV::SDiv(_)
        ));
        assert!(matches!(
            word::BV::<Mat>::from(word::unsigned::Div(8, [2, 3])),
            word::BV::UDiv(_)
        ));
    }

    #[test]
    fn comparisons_read_numbers_and_write_booleans() {
        let op = int(int::Lt([2, 3]));
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
        assert_eq!(And([2, 3]).dom(), two(b));
        assert_eq!(And([2, 3]).cod(), one(b));
        assert_eq!(Or([2, 3]).dom(), two(b));
        assert_eq!(Xor([2, 3]).cod(), one(b));
        assert_eq!(Not([2, 3]).dom(), one(b));
        assert_eq!(Not([2, 3]).cod(), one(b));
        assert_eq!(Eq([2, 3]).dom(), two(b));
        assert_eq!(Eq([2, 3]).cod(), one(b));
        assert_eq!(Ne([2, 3]).dom(), two(b));
        assert_eq!(Ne([2, 3]).cod(), one(b));
    }

    #[test]
    fn ops_display_their_name() {
        assert_eq!(And([1, 1]).to_string(), "And");
        assert_eq!(Not([1, 1]).to_string(), "Not");
        assert_eq!(Eq([1, 1]).to_string(), "Eq");
        assert_eq!(Ne([1, 1]).to_string(), "Ne");
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
        assert_eq!(Add([2, 3]).dom(), two(r));
        assert_eq!(Add([2, 3]).cod(), one(r));
        assert_eq!(Sub([2, 3]).dom(), two(r));
        assert_eq!(Mul([2, 3]).dom(), two(r));
        assert_eq!(Mul([2, 3]).cod(), one(r));
        assert_eq!(Div([2, 3]).dom(), two(r));
        assert_eq!(Div([2, 3]).cod(), one(r));
        assert_eq!(Mod([2, 3]).dom(), two(r));
        assert_eq!(Mod([2, 3]).cod(), one(r));
        assert_eq!(Min([2, 3]).dom(), two(r));
        assert_eq!(Max([2, 3]).cod(), one(r));
        assert_eq!(ReLU([2, 3]).dom(), one(r));
        assert_eq!(ReLU([2, 3]).cod(), one(r));
    }

    #[test]
    fn transpose_swaps_shape() {
        let t = Transpose([2, 3]);
        assert_eq!(t.dom(), one(Int::new([2, 3])));
        assert_eq!(t.cod(), one(Int::new([3, 2])));
    }

    #[test]
    fn matmul_contracts_inner_size() {
        let p = MatMul([2, 3], [3, 4]);
        let dom: Poly<Int> = mono![Int::new([2, 3]), Int::new([3, 4])].into();
        assert_eq!(p.dom(), dom);
        assert_eq!(p.cod(), one(Int::new([2, 4])));
    }

    #[test]
    fn ops_display_their_name() {
        assert_eq!(Add([1, 1]).to_string(), "Add");
        assert_eq!(Mul([1, 1]).to_string(), "Mul");
        assert_eq!(Div([1, 1]).to_string(), "Div");
        assert_eq!(Mod([1, 1]).to_string(), "Mod");
        assert_eq!(ReLU([1, 1]).to_string(), "ReLU");
        assert_eq!(Transpose([1, 1]).to_string(), "Transpose");
    }
    #[test]
    fn comparisons_write_booleans() {
        let x: Sort = Int::new([2, 3]).into();
        let b: Sort = Bool::new([2, 3]).into();
        let dom: Poly<Sort> = mono![x, x].into();
        let cod: Poly<Sort> = mono![b].into();
        for (d, c) in [
            (Lt([2, 3]).dom(), Lt([2, 3]).cod()),
            (Le([2, 3]).dom(), Le([2, 3]).cod()),
            (Gt([2, 3]).dom(), Gt([2, 3]).cod()),
            (Ge([2, 3]).dom(), Ge([2, 3]).cod()),
            (Eq([2, 3]).dom(), Eq([2, 3]).cod()),
            (Ne([2, 3]).dom(), Ne([2, 3]).cod()),
        ] {
            assert_eq!(d, dom);
            assert_eq!(c, cod);
        }
        assert_eq!(Le([1, 1]).to_string(), "Le");
        assert_eq!(Ne([1, 1]).to_string(), "Ne");
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
        assert_eq!(Add([2, 3]).dom(), two(r));
        assert_eq!(Add([2, 3]).cod(), one(r));
        assert_eq!(Sub([2, 3]).dom(), two(r));
        assert_eq!(Mul([2, 3]).dom(), two(r));
        assert_eq!(Mul([2, 3]).cod(), one(r));
        assert_eq!(Div([2, 3]).dom(), two(r));
        assert_eq!(Div([2, 3]).cod(), one(r));
        assert_eq!(Min([2, 3]).dom(), two(r));
        assert_eq!(Max([2, 3]).cod(), one(r));
        assert_eq!(ReLU([2, 3]).dom(), one(r));
        assert_eq!(ReLU([2, 3]).cod(), one(r));
    }

    #[test]
    fn transpose_swaps_shape() {
        let t = Transpose([2, 3]);
        assert_eq!(t.dom(), one(Real::new([2, 3])));
        assert_eq!(t.cod(), one(Real::new([3, 2])));
    }

    #[test]
    fn matmul_contracts_inner_size() {
        let p = MatMul([2, 3], [3, 4]);
        let dom: Poly<Real> = mono![Real::new([2, 3]), Real::new([3, 4])].into();
        assert_eq!(p.dom(), dom);
        assert_eq!(p.cod(), one(Real::new([2, 4])));
    }

    #[test]
    fn ops_display_their_name() {
        assert_eq!(Add([1, 1]).to_string(), "Add");
        assert_eq!(Mul([1, 1]).to_string(), "Mul");
        assert_eq!(Div([1, 1]).to_string(), "Div");
        assert_eq!(ReLU([1, 1]).to_string(), "ReLU");
        assert_eq!(Transpose([1, 1]).to_string(), "Transpose");
    }
    #[test]
    fn comparisons_write_booleans() {
        let x: Sort = Real::new([2, 3]).into();
        let b: Sort = Bool::new([2, 3]).into();
        let dom: Poly<Sort> = mono![x, x].into();
        let cod: Poly<Sort> = mono![b].into();
        for (d, c) in [
            (Lt([2, 3]).dom(), Lt([2, 3]).cod()),
            (Le([2, 3]).dom(), Le([2, 3]).cod()),
            (Gt([2, 3]).dom(), Gt([2, 3]).cod()),
            (Ge([2, 3]).dom(), Ge([2, 3]).cod()),
            (Eq([2, 3]).dom(), Eq([2, 3]).cod()),
            (Ne([2, 3]).dom(), Ne([2, 3]).cod()),
        ] {
            assert_eq!(d, dom);
            assert_eq!(c, cod);
        }
        assert_eq!(Le([1, 1]).to_string(), "Le");
        assert_eq!(Ne([1, 1]).to_string(), "Ne");
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
        assert_eq!(Add(8, [2, 3]).dom(), two(x));
        assert_eq!(Add(8, [2, 3]).cod(), one(x));
        assert_eq!(Xnor(8, [2, 3]).cod(), one(x));
        assert_eq!(Neg(8, [2, 3]).dom(), one(x));
        assert_eq!(Not(8, [2, 3]).cod(), one(x));
        assert_eq!(RotateLeft(3, 8, [2, 3]).dom(), one(x));
        assert_eq!(RotateRight(3, 8, [2, 3]).cod(), one(x));
        assert_eq!(signed::Div(8, [2, 3]).dom(), two(x));
        assert_eq!(signed::Mod(8, [2, 3]).cod(), one(x));
        assert_eq!(unsigned::Rem(8, [2, 3]).cod(), one(x));
        assert_eq!(unsigned::Shr(8, [2, 3]).dom(), two(x));
    }

    #[test]
    fn comparisons_write_booleans() {
        let x = w(8, [2, 3]);
        assert_eq!(Eq(8, [2, 3]).dom(), two(x));
        assert_eq!(Eq(8, [2, 3]).cod(), one(b([2, 3])));
        assert_eq!(Ne(8, [2, 3]).cod(), one(b([2, 3])));
        assert_eq!(signed::Lt(8, [2, 3]).dom(), two(x));
        assert_eq!(signed::Lt(8, [2, 3]).cod(), one(b([2, 3])));
        assert_eq!(unsigned::Ge(8, [2, 3]).cod(), one(b([2, 3])));
    }

    #[test]
    fn width_changing_boundaries() {
        assert_eq!(Extract(7, 4, 16, [2, 3]).dom(), one(w(16, [2, 3])));
        assert_eq!(Extract(7, 4, 16, [2, 3]).cod(), one(w(4, [2, 3])));
        assert_eq!(Extract(0, 0, 8, [1, 1]).cod(), one(w(1, [1, 1])));

        let dom: Poly<Sort> = mono![w(8, [2, 3]), w(4, [2, 3])].into();
        assert_eq!(Concat(8, 4, [2, 3]).dom(), dom);
        assert_eq!(Concat(8, 4, [2, 3]).cod(), one(w(12, [2, 3])));

        assert_eq!(unsigned::Extend(8, 8, [2, 3]).cod(), one(w(16, [2, 3])));
        assert_eq!(signed::Extend(24, 8, [2, 3]).cod(), one(w(32, [2, 3])));
        assert_eq!(Repeat(3, 4, [2, 3]).cod(), one(w(12, [2, 3])));
    }

    #[test]
    fn ops_display_their_name_and_indices() {
        assert_eq!(Add(8, [1, 1]).to_string(), "Add");
        assert_eq!(Ne(8, [1, 1]).to_string(), "Ne");
        assert_eq!(Extract(7, 4, 8, [1, 1]).to_string(), "Extract[7:4]");
        assert_eq!(RotateLeft(3, 8, [1, 1]).to_string(), "RotateLeft(3)");
        assert_eq!(Repeat(2, 8, [1, 1]).to_string(), "Repeat(2)");
        assert_eq!(signed::Div(8, [1, 1]).to_string(), "SDiv");
        assert_eq!(unsigned::Div(8, [1, 1]).to_string(), "UDiv");
        assert_eq!(signed::Shr(8, [1, 1]).to_string(), "AShr");
        assert_eq!(unsigned::Shr(8, [1, 1]).to_string(), "LShr");
        assert_eq!(signed::Ge(8, [1, 1]).to_string(), "SGe");
        assert_eq!(signed::Extend(8, 8, [1, 1]).to_string(), "SignExtend(8)");
        assert_eq!(unsigned::Extend(8, 8, [1, 1]).to_string(), "ZeroExtend(8)");
    }
}
