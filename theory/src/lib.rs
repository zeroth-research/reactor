pub mod matrix;
pub mod sort;
#[cfg(feature = "torch")]
pub mod torch;
pub mod zero;

pub use matrix::Matrix;

/// The failure of a partial constructor, with its message: raised in Python
/// as a `ValueError`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

#[cfg(feature = "pyo3")]
impl From<Error> for pyo3::PyErr {
    fn from(e: Error) -> Self {
        pyo3::exceptions::PyValueError::new_err(e.0)
    }
}

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
