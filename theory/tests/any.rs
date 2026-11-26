//! Tests for `Any`, the operations of all theories, and `sort::Any`, the sorts
//! of all theories.

use base::boundary::{Boundary, Poly};
use base::differentiable::T;
use base::mono;
use theory::sort::{Any, Zero};
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
    // the matrix sorts hold Zero, which comes out as the one of all theories
    assert_eq!(Any::from(matrix::sort::Any::from(Zero)), Any::Zero(Zero));
    assert_eq!(Any::from(Zero), Any::Zero(Zero));
}

#[test]
fn the_other_sorts_keep_their_kind() {
    let r = matrix::sort::Real::new([2, 3]);
    assert_eq!(Any::from(matrix::sort::Any::from(r)), Any::Matrix(r.into()));
    let e = scalar::sort::Event;
    assert_eq!(Any::from(scalar::sort::Any::from(e)), Any::Scalar(e.into()));
}

#[test]
fn tangents_are_taken_within_each_kind() {
    let r = matrix::sort::Real::new([2, 3]);
    assert_eq!(
        T(Any::from(matrix::sort::Any::from(r))),
        Any::from(matrix::sort::Any::from(T(r)))
    );
    // the discrete sorts of both kinds have the same tangent, Zero
    let i = matrix::sort::Int::new([2, 3]);
    assert_eq!(T(Any::from(matrix::sort::Any::from(i))), Any::Zero(Zero));
    let e = scalar::sort::Event;
    assert_eq!(T(Any::from(scalar::sort::Any::from(e))), Any::Zero(Zero));
    assert_eq!(T(Any::Zero(Zero)), Any::Zero(Zero));
}

#[test]
fn the_operations_of_both_kinds_lift_their_boundary() {
    // a matrix operation
    let r = matrix::sort::Real::new([2, 2]);
    let add: theory::Any<Mat> = matrix::Any::from(matrix::real::NRA::from(
        matrix::real::Add::with_dom([r, r]).unwrap(),
    ))
    .into();
    let real = Any::from(matrix::sort::Any::from(r));
    assert_eq!(add.dom(), mono![real.clone(), real.clone()].into());
    assert_eq!(add.cod(), mono![real].into());

    // a scalar operation, whose comparison writes a boolean of the matrix
    // theories, of the shape [1, 1]
    let ne: theory::Any<Mat> =
        scalar::Any::from(scalar::event::Any::from(scalar::event::Ne)).into();
    let event = Any::from(scalar::sort::Event);
    assert_eq!(ne.dom(), mono![event.clone(), event].into());
    assert_eq!(
        ne.cod(),
        mono![Any::from(matrix::sort::Bool::new([1, 1]))].into()
    );
    assert_eq!(ne.to_string(), "Ne");

    // the zero rate of change of a discrete matrix value writes the one Zero
    let zero: theory::Any<Mat> = matrix::Any::Zero(theory::zero::Zero).into();
    assert_eq!(zero.dom(), Poly::unit());
    assert_eq!(zero.cod(), mono![Any::Zero(Zero)].into());
}

/// The sorts of all kinds that the structural operations are tested at.
fn sorts() -> [Any; 4] {
    [
        matrix::sort::Any::from(matrix::sort::Real::new([2, 3])).into(),
        scalar::sort::Any::from(scalar::sort::Enum::new(["idle", "busy"]).unwrap()).into(),
        scalar::sort::Any::from(scalar::sort::Event).into(),
        Any::Zero(Zero),
    ]
}

#[test]
fn havoc_writes_a_value_of_any_sort_reading_nothing() {
    for s in sorts() {
        let op = theory::Any::<Mat>::havoc(&s);
        assert_eq!(op.dom(), Poly::unit());
        assert_eq!(op.cod(), mono![s].into());
        assert_eq!(op.to_string(), "Havoc");
    }
}

#[test]
fn skip_is_the_identity_of_any_sort() {
    for s in sorts() {
        let op = theory::Any::<Mat>::skip(&s);
        assert_eq!(op.dom(), mono![s.clone()].into());
        assert_eq!(op.cod(), mono![s].into());
        assert_eq!(op.to_string(), "Skip");
    }
}

#[test]
fn the_structural_operations_belong_to_no_kind() {
    for s in sorts() {
        assert!(matches!(
            theory::Any::<Mat>::havoc(&s),
            theory::Any::Havoc(_)
        ));
        assert!(matches!(theory::Any::<Mat>::skip(&s), theory::Any::Skip(_)));
        assert!(matches!(theory::Any::<Mat>::zero(&s), theory::Any::Zero(_)));
    }
}

#[test]
fn zero_writes_the_zero_rate_of_change_of_its_sort() {
    // of a real, of degree 1, and of Zero, the tangent of the discrete sorts
    let rate = Any::from(matrix::sort::Any::from(T(matrix::sort::Real::new([2, 3]))));
    for s in [
        rate,
        T(Any::from(matrix::sort::Any::from(matrix::sort::Int::new(
            [2, 3],
        )))),
        T(Any::from(scalar::sort::Any::from(scalar::sort::Event))),
    ] {
        let op = theory::Any::<Mat>::zero(&s);
        assert_eq!(op.dom(), Poly::unit());
        assert_eq!(op.cod(), mono![s].into());
        assert_eq!(op.to_string(), "Zero");
    }
}

#[test]
fn zero_never_fails_even_for_a_sort_that_is_no_rate_of_change() {
    let event = Any::from(scalar::sort::Any::from(scalar::sort::Event));
    let op = theory::Any::<Mat>::zero(&event);
    assert_eq!(op.cod(), mono![event].into());
}
