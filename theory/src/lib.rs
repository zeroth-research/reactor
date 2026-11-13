pub mod any;
pub mod bv;
pub mod lia;
pub mod lra;
pub mod matrix;
pub mod tensor;

use base::boundary::Boundary;
pub use matrix::{Affine, Matrix};
pub use tensor::PyTensor;

// The distinguished operations over the sorts `S`. They are unary: `havoc` and `zero`
// write a single wire of their range (`I -> range`), and `skip` copies a single wire
// (`range -> range`).

pub trait Combinatorial<S>: Boundary<S> {
    // Returns the unary itype that chooses an element within the range
    fn havoc(range: &S) -> Self;
}

pub trait Sequential<S>: Boundary<S> {
    // Returns the unary itype that copies a value within the range
    fn skip(range: &S) -> Self;
}

pub trait Differential<S>: Boundary<S> {
    // Returns the unary itype that indicates zero derivative, the range is the sort
    // of the derivative (i.e., a tangent sort)
    fn zero(range: &S) -> Self;
}
