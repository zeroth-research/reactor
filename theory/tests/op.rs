//! Tests for `Op`, the operations of all theories, and `Any`, the sorts
//! of all theories.

use base::boundary::{Boundary, Poly};
use base::differentiable::T;
use base::mono;
use theory::zero::Real0;
use theory::{Any, Combinatorial, Differential, Matrix, Reals, Sequential};
use theory::{bool::Bool, r#enum::Enum, event, event::Event, int::Int, real, real::Real};

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

type Op = theory::Op<Mat>;

#[test]
fn each_sort_is_a_variant_of_any() {
    let r = Real::new_unchecked([2, 3]);
    assert_eq!(Any::from(r), Any::Real(r));
    assert_eq!(Any::from(Event), Any::Event(Event));
    let e = Enum::new(["idle", "busy"]).unwrap();
    assert_eq!(Any::from(e.clone()), Any::Enum(e));
    assert_eq!(Any::from(Real0), Any::Zero(Real0));
}

#[test]
fn the_subsets_hold_the_matrix_sorts_only() {
    let r = Real::new_unchecked([2, 3]);
    assert_eq!(Reals::try_from(Any::Real(r)).ok(), Some(Reals::Real(r)));
    assert!(Reals::try_from(Any::Event(Event)).is_err());
    // the subsets are `Copy`, as their sorts are, unlike `Any`
    let s = Reals::from(r);
    let t = s;
    assert_eq!(s, t);
}

#[test]
fn tangents_are_taken_within_any() {
    let r = Real::new_unchecked([2, 3]);
    assert_eq!(T(Any::from(r)), Any::from(T(r)));
    // the discrete sorts, matrix or scalar, have the same tangent, Real0
    assert_eq!(T(Any::from(Int::new_unchecked([2, 3]))), Any::Zero(Real0));
    assert_eq!(T(Any::from(Event)), Any::Zero(Real0));
    assert_eq!(T(Any::Zero(Real0)), Any::Zero(Real0));
}

#[test]
fn the_operations_of_every_theory_lift_their_boundary() {
    // a matrix operation
    let r = Real::new_unchecked([2, 2]);
    let add: Op = real::NRA::from(real::Add::with_dom([r, r]).unwrap()).into();
    let real = Any::from(r);
    assert_eq!(add.dom(), mono![real.clone(), real.clone()].into());
    assert_eq!(add.cod(), mono![real].into());

    // a scalar operation, whose comparison writes a boolean of the shape [1, 1]
    let ne: Op = event::Op::from(event::Ne).into();
    let event = Any::from(Event);
    assert_eq!(ne.dom(), mono![event.clone(), event].into());
    assert_eq!(
        ne.cod(),
        mono![Any::from(Bool::new_unchecked([1, 1]))].into()
    );
    assert_eq!(ne.to_string(), "Ne");

    // the zero rate of change of a discrete value writes the one Real0
    let zero = Op::Zero(theory::zero::Zero);
    assert_eq!(zero.dom(), Poly::unit());
    assert_eq!(zero.cod(), mono![Any::Zero(Real0)].into());
}

/// The sorts of every kind that the structural operations are tested at.
fn sorts() -> [Any; 4] {
    [
        Real::new_unchecked([2, 3]).into(),
        Enum::new(["idle", "busy"]).unwrap().into(),
        Event.into(),
        Any::Zero(Real0),
    ]
}

#[test]
fn havoc_writes_a_value_of_any_sort_reading_nothing() {
    for s in sorts() {
        let op = Op::havoc(&s);
        assert!(matches!(op, Op::Havoc(_)));
        assert_eq!(op.dom(), Poly::unit());
        assert_eq!(op.cod(), mono![s].into());
        assert_eq!(op.to_string(), "Havoc");
    }
}

#[test]
fn skip_is_the_identity_of_any_sort() {
    for s in sorts() {
        let op = Op::skip(&s);
        assert!(matches!(op, Op::Skip(_)));
        assert_eq!(op.dom(), mono![s.clone()].into());
        assert_eq!(op.cod(), mono![s].into());
        assert_eq!(op.to_string(), "Skip");
    }
}

#[test]
fn zero_writes_the_zero_rate_of_change_of_a_tangent_sort() {
    // of a real, of degree 1, the zero matrix of that degree
    let rate = Any::from(T(Real::new_unchecked([2, 3])));
    let op = Op::zero(&rate);
    assert!(matches!(op, Op::Real(real::NRA::Zero(_))));
    assert_eq!(op.cod(), mono![rate].into());
    // of Real0, the tangent of the discrete sorts, matrix or scalar, its only value
    for s in [
        T(Any::from(Int::new_unchecked([2, 3]))),
        T(Any::from(Event)),
    ] {
        let op = Op::zero(&s);
        assert!(matches!(op, Op::Zero(_)));
        assert_eq!(op.dom(), Poly::unit());
        assert_eq!(op.cod(), mono![s].into());
        assert_eq!(op.to_string(), "Zero");
    }
}

#[test]
#[should_panic(expected = "only the tangent sorts")]
fn zero_panics_for_a_sort_that_is_no_rate_of_change() {
    Op::zero(&Any::from(Event));
}
