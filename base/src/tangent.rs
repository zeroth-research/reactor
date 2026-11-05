//! Tangents: the sort-level part of the differential structure.

/// Sorts closed under the tangent former: `T(s)` is the sort of rates of
/// change of values of sort `s`.
///
/// Only the *object part* of a tangent structure — the action of `T` on
/// sorts — which is all the reactive layer needs to mint derivative wires.
/// The structure maps (zero section, bundle addition, ...) are signature
/// generators, and their axioms belong to theories, not here.
///
/// Discrete sorts have a *trivial* tangent, not a missing one: `T` maps
/// them to an inhabited singleton sort (`Zero`), whose only writer is the
/// `zero` generator.
pub trait Tangent {
    /// The tangent sort `T(self)`: `T` as in the tangent bundle `TM`.
    fn tangent(&self) -> Self;
}

/// `T` as a function on sorts: `T(s)` is `s.tangent()`.
#[allow(non_snake_case)]
pub fn T<S: Tangent>(s: S) -> S {
    s.tangent()
}
