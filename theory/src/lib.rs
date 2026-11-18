pub mod op;
pub mod sort;
pub mod tensor;
#[cfg(feature = "torch")]
pub mod torch;

pub use tensor::{PyTensor, Tensor};

pub trait Combinatorial<S> {
    /// The operation `I → range` that chooses an arbitrary value of
    /// `range`.
    fn havoc(range: &S) -> Self;
}

pub trait Sequential<S> {
    /// The operation `range → range` that copies its value.
    fn skip(range: &S) -> Self;
}

pub trait Differential<S> {
    /// The operation `I → range` that writes the zero rate of change, where
    /// `range` is the sort of the derivative wire, not of its variable.
    fn zero(range: &S) -> Self;
}
