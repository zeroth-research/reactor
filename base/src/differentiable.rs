//! Differentiable sorts: the sort-level part of the differential structure.

/// Sorts with a tangent sort: `T(s)`, the sort of rates of change of values
/// of sort `s`, is a [`Self::Tangent`](Differentiable::Tangent).
///
/// Only the *object part* of a tangent structure — the action of `T` on
/// sorts — which is all the reactive layer needs to mint derivative wires.
/// The structure maps (zero section, bundle addition, ...) are signature
/// generators, and their axioms belong to theories, not here.
///
/// The tangent sort is an associated type, as `Output` is for
/// [`Add`](std::ops::Add): `T` is a function on sorts, so each sort has
/// exactly one tangent sort, and it is inferred from the base sort, as in
/// `T(T(s))` or `T(s).into()`. Sorts closed under `T`, such as graded
/// reals, are their own tangent sort. Discrete sorts have a *trivial*
/// tangent, not a missing one: `T` maps them to an inhabited singleton sort
/// (`Zero`), whose only writer is the `zero` generator, and which may be a
/// type of its own (`type Tangent = Zero`).
pub trait Differentiable {
    /// The tangent sort `T(Self)`: `T` as in the tangent bundle `TM`.
    type Tangent;

    /// The tangent sort `T(self)`.
    fn tangent(self) -> Self::Tangent;
}

/// The tangent of a borrowed sort: that of a copy of the sort, so that a
/// sort need not be moved to take its tangent.
impl<S: Differentiable + Copy> Differentiable for &S {
    type Tangent = S::Tangent;

    fn tangent(self) -> S::Tangent {
        (*self).tangent()
    }
}

/// `T` as a function on sorts: `T(s)` is `s.tangent()`.
#[allow(non_snake_case)]
pub fn T<S: Differentiable>(s: S) -> S::Tangent {
    s.tangent()
}
