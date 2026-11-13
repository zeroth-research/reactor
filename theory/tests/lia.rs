//! Tests for the theory of linear integer arithmetic.

#![cfg(feature = "torch")]

use theory::lia::Sort;

fn int(r: usize, c: usize) -> Sort {
    Sort::Int([r, c])
}

fn bool_t(r: usize, c: usize) -> Sort {
    Sort::Bool([r, c])
}

#[test]
fn type_kind_and_shape() {
    assert!(int(2, 3).is_int() && !int(2, 3).is_bool());
    assert_eq!(int(2, 3).shape(), &[2, 3]);
    assert!(bool_t(1, 1).is_bool() && !bool_t(1, 1).is_int());
}
