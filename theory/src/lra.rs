/*!
# Linear integer arithmetic

Defines the theory [`LRA`] of linear integer arithmetic over matrices,
mixing integer and boolean matrices in a single signature.

A [`Sort`] value is either `Int(rows, cols)` or `Bool(rows, cols)`.
`Type` converts to and from [`int::IntType`] and [`bool::PropType`]
so that integer and propositional terms embed directly into `RLA`. The
operations in [`LRA`] are:

- [`LRA::Real`] — a matrix literal whose sort (real or boolean) is taken
  from the write wire; the tensor's element kind must match that sort.
- [`LRA::And`], [`LRA::Or`], [`LRA::Xor`], [`LRA::Not`]
  — boolean operations on the boolean fragment of `Type`.
- [`LRA::Le`], [`LRA::Lt`], [`LRA::Ge`], [`LRA::Gt`], [`LRA::Eq`], [`LRA::Ne`]
  — pointwise integer comparisons producing a scalar `Bool(1,1)`.
- [`LRA::Ite`] — if-then-else: reads a boolean guard and two same-typed branches.
- [`LRA::Linear`]`(A, B)` — the affine map `x ↦ A·x + B`, with `A` and
  `B` constant integer matrices of compatible shapes.
- [`LRA::ReLU`] — the shape-preserving rectified-linear map on integer matrices.
*/

use crate::*;
#[cfg(feature = "pyo3")]
use pyo3::pyclass;
use std::fmt;

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub enum Sort {
    Real([usize; 2]),
    Bool([usize; 2]),
}

impl Sort {
    pub fn is_bool(&self) -> bool {
        matches!(self, Sort::Bool(..))
    }

    pub fn is_real(&self) -> bool {
        matches!(self, Sort::Real(..))
    }

    pub fn shape(&self) -> &[usize; 2] {
        match self {
            Sort::Bool(shape) | Sort::Real(shape) => shape,
        }
    }
}

impl fmt::Display for Sort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sort::Real([i, j]) => write!(f, "Real({i}, {j})"),
            Sort::Bool([i, j]) => write!(f, "Bool({i}, {j})"),
        }
    }
}

#[derive(Clone, Debug, strum::Display)]
#[cfg_attr(feature = "pyo3", pyclass(frozen))]
pub enum LRA {
    // constant matrix literal; its sort (Real or Bool) is taken from the write wire
    #[strum(to_string = "{0}")]
    Real(crate::PyTensor),
    #[strum(to_string = "{0}")]
    Bool(crate::PyTensor),
    // boolean operations
    And(),
    Or(),
    Xor(),
    Not(),
    // integer comparisons
    Le(),
    Lt(),
    Ge(),
    Gt(),
    Eq(),
    Ne(),
    // linear / matrix operations
    // A*x + B where `A` and `B` are constants
    #[strum(to_string = "Linear({0}, {1})")]
    Linear(crate::PyTensor, crate::PyTensor),
    Add(),
    Sub(),
    // XXX: should these be in RLA?
    ReLU(),
    Argmax(),
    Min(),
    Max(),
    // matrix operations
    Transpose(),
    // control flow
    Ite(),
    Id(),
    #[strum(to_string = "Uninterpreted({0})")]
    Uninterpreted(String),
    BoolZerograd([usize; 2]), // unstable
    RealZerograd([usize; 2]), // unstable
    AnyBool([usize; 2]),
    AnyReal([usize; 2]),
}

impl Sequential<Sort> for LRA {
    fn skip(_range: &Sort) -> Self {
        LRA::Id()
    }
}

impl Combinatorial<Sort> for LRA {
    fn havoc(range: &Sort) -> Self {
        match range {
            Sort::Bool(shape) => LRA::AnyBool(*shape),
            Sort::Real(shape) => LRA::AnyReal(*shape),
        }
    }
}

impl Differential<Sort> for LRA {
    fn zero(range: &Sort) -> Self {
        match range {
            Sort::Bool(shape) => LRA::BoolZerograd(*shape),
            Sort::Real(shape) => LRA::RealZerograd(*shape),
        }
    }
}

#[cfg(all(test, feature = "torch"))]
mod tests {
    use super::*;

    fn real(r: usize, c: usize) -> Sort {
        Sort::Real([r, c])
    }

    fn bool_t(r: usize, c: usize) -> Sort {
        Sort::Bool([r, c])
    }

    #[test]
    fn type_kind_and_shape() {
        assert!(real(2, 3).is_real() && !real(2, 3).is_bool());
        assert_eq!(real(2, 3).shape(), &[2, 3]);
        assert!(bool_t(1, 1).is_bool() && !bool_t(1, 1).is_real());
    }
}
