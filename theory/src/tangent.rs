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

#[cfg(test)]
mod tests {
    use super::*;

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
}
