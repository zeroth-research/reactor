//! Tests for the `Tangent` trait: the tangent former `T` on sorts.

use base::tangent::{T, Tangent};

/// A minimal sort: graded reals, and booleans with the trivial tangent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sort {
    Real(u8),
    Bool,
    Zero,
}

impl Tangent for Sort {
    fn tangent(&self) -> Self {
        match self {
            Sort::Real(rank) => Sort::Real(rank + 1),
            Sort::Bool | Sort::Zero => Sort::Zero,
        }
    }
}

#[test]
fn tangents() {
    // reals are graded by their rank
    assert_eq!(T(Sort::Real(0)), Sort::Real(1));
    assert_eq!(T(T(Sort::Real(0))), Sort::Real(2));

    // booleans have the trivial tangent, a fixed point of `T`
    assert_eq!(T(Sort::Bool), Sort::Zero);
    assert_eq!(T(Sort::Zero), Sort::Zero);

    // the function is the method
    assert_eq!(T(Sort::Real(0)), Sort::Real(0).tangent());
}
