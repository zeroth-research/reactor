//! Tangents: the sort-level part of the differential structure.

/// Tangent sorts: a sort type `U` implements `Tangent<S>` when `T(s)`, the
/// sort of rates of change of values of sort `s: S`, is a `U`.
///
/// Only the *object part* of a tangent structure — the action of `T` on
/// sorts — which is all the reactive layer needs to mint derivative wires.
/// The structure maps (zero section, bundle addition, ...) are signature
/// generators, and their axioms belong to theories, not here.
///
/// The trait follows [`From`]: it is implemented by the tangent sort, so
/// that `U::of(s)` reads "the `U` that is the tangent of `s`", and
/// [`Differentiable`] is its counterpart on the base sort, `s.tangent()`,
/// as [`Into`] is to `From`. The base sort defaults to `Self`, for sorts closed under `T`,
/// such as graded reals. Discrete sorts have a *trivial* tangent, not a
/// missing one: `T` maps them to an inhabited singleton sort (`Zero`),
/// whose only writer is the `zero` generator, and which may be a type of
/// its own (`impl Tangent<Int> for Zero`).
pub trait Tangent<S = Self> {
    /// The tangent sort `T(s)`: `T` as in the tangent bundle `TM`.
    fn of(s: S) -> Self;
}

/// The counterpart of [`Tangent`] on the base sort, as [`Into`] is to
/// [`From`]: implemented for every `S` whose tangent sort `U` implements
/// `Tangent<S>`, and not meant to be implemented directly: `s.tangent()`
/// is `U::of(s)`.
pub trait Differentiable<U> {
    /// The tangent sort `T(self)`.
    fn tangent(self) -> U;
}

impl<S, U: Tangent<S>> Differentiable<U> for S {
    fn tangent(self) -> U {
        U::of(self)
    }
}

/// `T` as a function on sorts: `T(s)` is `U::of(s)`.
#[allow(non_snake_case)]
pub fn T<S, U: Tangent<S>>(s: S) -> U {
    U::of(s)
}
