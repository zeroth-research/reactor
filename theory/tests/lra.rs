//! Tests for the theory of linear real arithmetic.

#![cfg(feature = "torch")]

use theory::lra::Sort;

fn real(r: usize, c: usize) -> Sort {
    Sort::Real([r, c])
}

fn bool_t(r: usize, c: usize) -> Sort {
    Sort::Bool([r, c])
}

#[test]
fn type_kind_and_shape() {
    assert!(real(2, 3).is_real() && !real(2, 3).is_bool());
    assert_eq!(real(2, 3).shape(), &[2, 3]);
    assert!(bool_t(1, 1).is_bool() && !bool_t(1, 1).is_real());
}
