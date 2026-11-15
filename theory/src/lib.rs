pub mod any;
pub mod bv;
pub mod int;
pub mod lia;
pub mod lra;
pub mod real;
pub mod tensor;
pub mod zero;

pub use tensor::PyTensor;

pub trait Combinatorial<S> {
    // Returns the unary itype that chooses an element within the range
    fn havoc(range: &S) -> Self;
}

pub trait Sequential<S> {
    // Returns the unary itype that copies a value within the range
    fn skip(range: &S) -> Self;
}

pub trait Differential<S> {
    // Returns the unary itype that indicates zero derivative
    fn zero(range: &S) -> Self;
}
