use base::semiring::{Probability, Semiring};

#[test]
fn f64_units() {
    assert_eq!(f64::zero() + 0.25, 0.25);
    assert_eq!(f64::one() * 0.25, 0.25);
    assert_eq!(f64::zero() * 0.25, 0.0);
}

#[test]
fn f64_complement() {
    assert_eq!(0.25f64.complement(), 0.75);
    assert_eq!(f64::one().complement(), f64::zero());
    assert_eq!(f64::zero().complement(), f64::one());
}

#[test]
fn f64_probabilities_are_the_unit_interval() {
    for p in [0.0, 0.5, 1.0] {
        assert!(p.is_probability());
    }
    for p in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
        assert!(!p.is_probability());
    }
}
