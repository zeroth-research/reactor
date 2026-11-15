/*!
# The singleton sort

Defines the sort [`Zero`], with exactly one value. It is the trivial
tangent of discrete sorts, such as integers: their rates of change carry
no information, so `T` maps them to `Zero` rather than leaving them
without a tangent. Each discrete sort states this with its own
`impl Tangent<S> for Zero`.
*/

use base::tangent::Tangent;
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

impl Tangent for Zero {
    fn of(_: Zero) -> Zero {
        Zero
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base::tangent::T;

    #[test]
    fn tangent_is_itself() {
        let t: Zero = T(Zero);
        assert_eq!(t, Zero);
        assert_eq!(Zero::of(Zero), Zero);
    }

    #[test]
    fn display() {
        assert_eq!(Zero.to_string(), "Zero");
    }
}
