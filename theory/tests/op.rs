//! Tests for `Op`, the operations of all theories, and `Any`, the sorts
//! of all theories.

use base::boundary::{Boundary, Poly};
use base::differentiable::T;
use base::mono;
use theory::Any;
use theory::zero::Real0;
use theory::{Combinatorial, Differential, Matrix, Sequential, matrix, scalar};

/// A matrix carrying its shape only, enough to compute boundaries.
#[derive(Debug)]
struct Mat([usize; 2]);

impl Matrix for Mat {
    fn shape(&self) -> [usize; 2] {
        self.0
    }
}

impl std::fmt::Display for Mat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Mat({:?})", self.0)
    }
}

#[test]
fn zero_is_the_one_of_all_theories() {
    // the matrix sorts hold Real0, which comes out as the one of all theories
    assert_eq!(Any::from(matrix::Any::from(Real0)), Any::Zero(Real0));
    assert_eq!(Any::from(Real0), Any::Zero(Real0));
}

#[test]
fn the_other_sorts_keep_their_kind() {
    let r = matrix::real::Real::new_unchecked([2, 3]);
    assert_eq!(Any::from(matrix::Any::from(r)), Any::Matrix(r.into()));
    let e = scalar::Event;
    assert_eq!(Any::from(scalar::Any::from(e)), Any::Scalar(e.into()));
}

#[test]
fn tangents_are_taken_within_each_kind() {
    let r = matrix::real::Real::new_unchecked([2, 3]);
    assert_eq!(
        T(Any::from(matrix::Any::from(r))),
        Any::from(matrix::Any::from(T(r)))
    );
    // the discrete sorts of both kinds have the same tangent, Real0
    let i = matrix::int::Int::new_unchecked([2, 3]);
    assert_eq!(T(Any::from(matrix::Any::from(i))), Any::Zero(Real0));
    let e = scalar::Event;
    assert_eq!(T(Any::from(scalar::Any::from(e))), Any::Zero(Real0));
    assert_eq!(T(Any::Zero(Real0)), Any::Zero(Real0));
}

#[test]
fn the_operations_of_both_kinds_lift_their_boundary() {
    // a matrix operation
    let r = matrix::real::Real::new_unchecked([2, 2]);
    let add: theory::Op<Mat> = matrix::Op::from(matrix::real::NRA::from(
        matrix::real::Add::with_dom([r, r]).unwrap(),
    ))
    .into();
    let real = Any::from(matrix::Any::from(r));
    assert_eq!(add.dom(), mono![real.clone(), real.clone()].into());
    assert_eq!(add.cod(), mono![real].into());

    // a scalar operation, whose comparison writes a boolean of the matrix
    // theories, of the shape [1, 1]
    let ne: theory::Op<Mat> = scalar::Op::from(scalar::event::Op::from(scalar::event::Ne)).into();
    let event = Any::from(scalar::Event);
    assert_eq!(ne.dom(), mono![event.clone(), event].into());
    assert_eq!(
        ne.cod(),
        mono![Any::from(matrix::bool::Bool::new_unchecked([1, 1]))].into()
    );
    assert_eq!(ne.to_string(), "Ne");

    // the zero rate of change of a discrete matrix value writes the one Real0
    let zero: theory::Op<Mat> = matrix::Op::Zero(theory::zero::Zero).into();
    assert_eq!(zero.dom(), Poly::unit());
    assert_eq!(zero.cod(), mono![Any::Zero(Real0)].into());
}

/// The sorts of all kinds that the structural operations are tested at.
fn sorts() -> [Any; 4] {
    [
        matrix::Any::from(matrix::real::Real::new_unchecked([2, 3])).into(),
        scalar::Any::from(scalar::Enum::new(["idle", "busy"]).unwrap()).into(),
        scalar::Any::from(scalar::Event).into(),
        Any::Zero(Real0),
    ]
}

#[test]
fn havoc_writes_a_value_of_any_sort_reading_nothing() {
    for s in sorts() {
        let op = theory::Op::<Mat>::havoc(&s);
        assert_eq!(op.dom(), Poly::unit());
        assert_eq!(op.cod(), mono![s].into());
        assert_eq!(op.to_string(), "Havoc");
    }
}

#[test]
fn skip_is_the_identity_of_any_sort() {
    for s in sorts() {
        let op = theory::Op::<Mat>::skip(&s);
        assert_eq!(op.dom(), mono![s.clone()].into());
        assert_eq!(op.cod(), mono![s].into());
        assert_eq!(op.to_string(), "Skip");
    }
}

#[test]
fn the_structural_operations_belong_to_no_kind() {
    for s in sorts() {
        assert!(matches!(theory::Op::<Mat>::havoc(&s), theory::Op::Havoc(_)));
        assert!(matches!(theory::Op::<Mat>::skip(&s), theory::Op::Skip(_)));
        assert!(matches!(theory::Op::<Mat>::zero(&s), theory::Op::Zero(_)));
    }
}

#[test]
fn zero_writes_the_zero_rate_of_change_of_its_sort() {
    // of a real, of degree 1, and of Real0, the tangent of the discrete sorts
    let rate = Any::from(matrix::Any::from(T(matrix::real::Real::new_unchecked([
        2, 3,
    ]))));
    for s in [
        rate,
        T(Any::from(matrix::Any::from(
            matrix::int::Int::new_unchecked([2, 3]),
        ))),
        T(Any::from(scalar::Any::from(scalar::Event))),
    ] {
        let op = theory::Op::<Mat>::zero(&s);
        assert_eq!(op.dom(), Poly::unit());
        assert_eq!(op.cod(), mono![s].into());
        assert_eq!(op.to_string(), "Zero");
    }
}

#[test]
fn zero_never_fails_even_for_a_sort_that_is_no_rate_of_change() {
    let event = Any::from(scalar::Any::from(scalar::Event));
    let op = theory::Op::<Mat>::zero(&event);
    assert_eq!(op.cod(), mono![event].into());
}
