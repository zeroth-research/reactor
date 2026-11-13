/*!
# Linear integer arithmetic

Defines the theory [`LIA`] of linear integer arithmetic over matrices,
mixing integer and boolean matrices in a single signature.

A [`Sort`] value is either `Int(rows, cols)` or `Bool(rows, cols)`.
`Type` converts to and from [`int::IntType`] and [`bool::PropType`]
so that integer and propositional terms embed directly into `LIA`. The
operations in [`LIA`] are:

- [`LIA::Int`] — a matrix literal whose sort (integer or boolean) is taken
  from the write wire; the tensor's element kind must match that sort.
- [`LIA::And`], [`LIA::Or`], [`LIA::Xor`], [`LIA::Not`]
  — boolean operations on the boolean fragment of `Type`.
- [`LIA::Le`], [`LIA::Lt`], [`LIA::Ge`], [`LIA::Gt`], [`LIA::Eq`], [`LIA::Ne`]
  — pointwise integer comparisons producing a scalar `Bool(1,1)`.
- [`LIA::Ite`] — if-then-else: reads a boolean guard and two same-typed branches.
- [`LIA::Linear`]`(A, B)` — the affine map `x ↦ A·x + B`, with `A` and
  `B` constant integer matrices of compatible shapes.
- [`LIA::ReLU`] — the shape-preserving rectified-linear map on integer matrices.
*/

use crate::*;
#[cfg(feature = "pyo3")]
use pyo3::pyclass;
use std::fmt;

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub enum Sort {
    Int([usize; 2]),
    Bool([usize; 2]),
}

impl Sort {
    pub fn is_bool(&self) -> bool {
        matches!(self, Sort::Bool(..))
    }

    pub fn is_int(&self) -> bool {
        matches!(self, Sort::Int(..))
    }

    pub fn shape(&self) -> &[usize; 2] {
        match self {
            Sort::Bool(shape) | Sort::Int(shape) => shape,
        }
    }
}

impl fmt::Display for Sort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sort::Int([i, j]) => write!(f, "Int({i}, {j})"),
            Sort::Bool([i, j]) => write!(f, "Bool({i}, {j})"),
        }
    }
}

#[derive(Clone, Debug, strum::Display)]
#[cfg_attr(feature = "pyo3", pyclass(frozen))]
pub enum LIA {
    // constant matrix literal; its sort (Int or Bool) is taken from the write wire
    #[strum(to_string = "{0}")]
    Int(crate::PyTensor),
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
    Linear(crate::PyTensor, crate::PyTensor),
    Add(),
    Sub(),
    // XXX: should these be in LIA?
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
    AnyInt([usize; 2]),
    AnyBool([usize; 2]),
}

impl Sequential<Sort> for LIA {
    fn skip(_range: &Sort) -> Self {
        LIA::Id()
    }
}

impl Combinatorial<Sort> for LIA {
    fn havoc(range: &Sort) -> Self {
        match range {
            Sort::Int(shape) => LIA::AnyInt(*shape),
            Sort::Bool(shape) => LIA::AnyBool(*shape),
        }
    }
}
