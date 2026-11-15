/*!
# The singleton sort

Defines the sort [`Zero`], with exactly one value. It is the trivial
tangent of discrete sorts, such as integers: their rates of change carry
no information, so `T` maps them to `Zero` rather than leaving them
without a tangent. Each discrete sort states this in its own
`impl Differentiable`, with `type Tangent = Zero`.
*/

use base::differentiable::Differentiable;
use std::fmt;

/// The singleton sort, with exactly one value: the trivial tangent of
/// discrete sorts. Its own tangent is itself.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Zero;

impl fmt::Display for Zero {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Zero")
    }
}

impl Differentiable for Zero {
    type Tangent = Zero;

    fn tangent(self) -> Zero {
        Zero
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base::differentiable::T;

    #[test]
    fn tangent_is_itself() {
        assert_eq!(T(Zero), Zero);
    }

    #[test]
    fn display() {
        assert_eq!(Zero.to_string(), "Zero");
    }
}
