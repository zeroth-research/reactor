//! Semirings: the algebra of weights.
//!
//! The weights of a probabilistic choice are composed in sequence by
//! multiplication and in parallel by addition, both with their units. That is
//! the structure of a *semiring*; the probabilities `[0, 1]` sit inside the
//! semiring of non-negative reals as the weights that are at most `1`, and
//! [`Probability`] adds what a probabilistic choice needs on top: the order
//! and the complement `1 − p`.

use std::ops::{Add, Mul};

/// A semiring `(S, +, ·, 0, 1)`: `+` is associative and commutative with
/// unit `0`, `·` is associative with unit `1`, `·` distributes over `+` on
/// both sides, and `0` annihilates under `·`.
///
/// The laws are not checked; implementors are trusted to satisfy them (up to
/// rounding, for floating-point types).
pub trait Semiring: Sized + Clone + PartialEq + Add<Output = Self> + Mul<Output = Self> {
    /// The unit of `+`, and the annihilator of `·`.
    fn zero() -> Self;

    /// The unit of `·`.
    fn one() -> Self;
}

/// The weights of probabilistic choice: a semiring ordered so that the
/// probabilities are the interval `[0, 1]`, with the complement `1 − p` of
/// each probability `p`.
pub trait Probability: Semiring + PartialOrd {
    /// The complement `1 − p` of a probability `p`: the weight of the other
    /// branch of a choice that takes the first with weight `p`. Only
    /// specified on [probabilities](Self::is_probability).
    fn complement(&self) -> Self;

    /// Whether the weight is a probability, i.e., lies in `[0, 1]`. Weights
    /// incomparable with `0` or `1` (such as a floating-point `NaN`) are not.
    fn is_probability(&self) -> bool {
        Self::zero() <= *self && *self <= Self::one()
    }
}

impl Semiring for f64 {
    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }
}

impl Probability for f64 {
    fn complement(&self) -> Self {
        1.0 - self
    }
}
