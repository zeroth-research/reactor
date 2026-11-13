//! Tests for the catch-all op enums (`Any` and its `Combinatorial`,
//! `Sequential`, and `Differential` subsets): casts and display.

use theory::any::{Any, Combinatorial, Differential, Sequential, Sort};
use theory::bv::BV;
use theory::lia::LIA;
use theory::lra::LRA;

// Casts up the hierarchy (Into) wrap the op in the right variant.

#[test]
fn base_theories_cast_up() {
    assert!(matches!(LRA::Add().into(), Combinatorial::LRA(LRA::Add())));
    assert!(matches!(LRA::Add().into(), Sequential::LRA(LRA::Add())));
    assert!(matches!(LRA::Add().into(), Differential::LRA(LRA::Add())));
    assert!(matches!(LIA::Add().into(), Combinatorial::LIA(LIA::Add())));
    assert!(matches!(LIA::Add().into(), Sequential::LIA(LIA::Add())));
    assert!(matches!(BV::And().into(), Combinatorial::BV(BV::And())));
    assert!(matches!(BV::And().into(), Sequential::BV(BV::And())));
}

#[test]
fn catch_alls_cast_up() {
    let comb: Sequential = Combinatorial::LRA(LRA::Add()).into();
    assert!(matches!(comb, Sequential::LRA(LRA::Add())));
    let t = Sort::Real([1, 1]);
    assert!(matches!(
        Combinatorial::HAVOC(t).into(),
        Sequential::HAVOC(_)
    ));
    assert!(matches!(Sequential::SKIP(t).into(), Any::SKIP(_)));
    assert!(matches!(Differential::ZERO(t).into(), Any::ZERO(_)));
}

// Casts down the hierarchy (TryInto) recover the op or reject it.

#[test]
fn display_is_stable() {
    // Elements print their name, member ops delegate to the base theory.
    let t = Sort::Real([1, 1]);
    assert_eq!(Any::HAVOC(t).to_string(), "HAVOC");
    assert_eq!(Sequential::SKIP(t).to_string(), "SKIP");
    assert_eq!(Differential::ZERO(t).to_string(), "ZERO");
    assert_eq!(Any::LRA(LRA::Add()).to_string(), "Add");
    assert_eq!(Sequential::LIA(LIA::ReLU()).to_string(), "ReLU");
    assert_eq!(Combinatorial::BV(BV::MatMul()).to_string(), "MatMul");
    // Base-theory special cases keep their formats.
    assert_eq!(
        LRA::Uninterpreted("f".to_string()).to_string(),
        "Uninterpreted(f)"
    );
    assert_eq!(
        BV::BitSelect { high: 7, low: 0 }.to_string(),
        "BitSelect[7:0]"
    );
    assert_eq!(BV::Extend { extra: 8 }.to_string(), "Extend(+8)");
}

#[test]
fn any_casts_down_or_rejects() {
    let t = Sort::Real([1, 1]);
    assert!(matches!(Any::SKIP(t).try_into(), Ok(Sequential::SKIP(_))));
    assert!(matches!(
        Any::HAVOC(t).try_into(),
        Ok(Combinatorial::HAVOC(_))
    ));
    assert!(matches!(Any::ZERO(t).try_into(), Ok(Differential::ZERO(_))));
    // ZERO is differential-only; SKIP is not combinatorial or differential.
    assert!(Sequential::try_from(Any::ZERO(t)).is_err());
    assert!(Combinatorial::try_from(Any::SKIP(t)).is_err());
    assert!(Differential::try_from(Any::SKIP(t)).is_err());
}
