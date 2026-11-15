//! Tests for the `Tangent` trait: the tangent former `T` on sorts.

use base::tangent::{Differentiable, T, Tangent};

/// A minimal sort: graded reals, and booleans with the trivial tangent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sort {
    Real(u8),
    Bool,
    Zero,
}

impl Tangent for Sort {
    fn of(s: Sort) -> Self {
        match s {
            Sort::Real(rank) => Sort::Real(rank + 1),
            Sort::Bool | Sort::Zero => Sort::Zero,
        }
    }
}

#[test]
fn tangents() {
    // reals are graded by their rank; as with `From`, the tangent sort is
    // named, since it cannot be inferred from the base sort alone
    let t: Sort = T(Sort::Real(0));
    assert_eq!(t, Sort::Real(1));
    let tt: Sort = T(t);
    assert_eq!(tt, Sort::Real(2));

    // booleans have the trivial tangent, a fixed point of `T`
    let t: Sort = T(Sort::Bool);
    assert_eq!(t, Sort::Zero);
    let t: Sort = T(Sort::Zero);
    assert_eq!(t, Sort::Zero);

    // the function is the tangent sort's constructor, as with `From`, and
    // the base sort's method, as with `Into`
    let t: Sort = T(Sort::Real(0));
    assert_eq!(t, Sort::of(Sort::Real(0)));
    let u: Sort = Sort::Real(0).tangent();
    assert_eq!(t, u);
}
