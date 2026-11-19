/*!
# Boolean algebra

Defines one type per operation on the sort [`Bool`] of boolean matrices. Every
operation reads and writes boolean matrices, and carries enough
information to compute its [`Boundary`]:

- [`Const`] — a constant boolean matrix literal.
- [`And`], [`Or`], [`Xor`] — pointwise conjunction, disjunction and
  exclusive disjunction.
- [`Not`] — pointwise negation.
- [`Eq`](struct@Eq), [`Ne`] — pointwise equality and disequality.
- [`Ite`] — the pointwise choice `if c then x else y`, by a boolean matrix `c`.
*/

use super::sort::Bool;
use crate::Tensor;
use base::boundary::{Boundary, Poly};
use base::mono;
use derive_more::From;
use pyo3::prelude::*;
use std::fmt;

/// A constant boolean matrix literal, held in a tensor of type `T`.
#[derive(Debug)]
pub struct Const<T>(pub T);

/// Pointwise conjunction `x ∧ y` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[pyo3::pyclass(frozen, eq, str)]
pub struct And(pub [usize; 2]);

/// Pointwise disjunction `x ∨ y` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[pyo3::pyclass(frozen, eq, str)]
pub struct Or(pub [usize; 2]);

/// Pointwise exclusive disjunction `x ⊕ y` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[pyo3::pyclass(frozen, eq, str)]
pub struct Xor(pub [usize; 2]);

/// Pointwise negation `¬x` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[pyo3::pyclass(frozen, eq, str)]
pub struct Not(pub [usize; 2]);

/// Pointwise equality `x = y` of `[m, n]` matrices: the negation of their
/// exclusive disjunction.
///
/// SMT-LIB2: `(= x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[pyo3::pyclass(frozen, eq, str)]
pub struct Eq(pub [usize; 2]);

/// Pointwise disequality `x ≠ y` of `[m, n]` matrices: their exclusive
/// disjunction.
///
/// SMT-LIB2: `(distinct x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[pyo3::pyclass(frozen, eq, str)]
pub struct Ne(pub [usize; 2]);

/// Pointwise choice `if c then x else y` between `[m, n]` boolean matrices
/// `x` and `y`, by the boolean matrix `c`, read first.
///
/// SMT-LIB2: `(ite c x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[pyo3::pyclass(frozen, eq, str)]
pub struct Ite(pub [usize; 2]);

/// The operations of propositional logic, the boolean operations of
/// SMT-LIB2's `Core` theory, with their boundary in [`Bool`].
#[derive(Debug, From)]
pub enum Prop<T> {
    Const(Const<T>),
    And(And),
    Or(Or),
    Xor(Xor),
    Not(Not),
    Eq(Eq),
    Ne(Ne),
    Ite(Ite),
}

impl<T> Clone for Prop<T>
where
    Const<T>: Clone,
{
    fn clone(&self) -> Self {
        match self {
            Prop::Const(op) => Prop::Const(op.clone()),
            Prop::And(op) => Prop::And(*op),
            Prop::Or(op) => Prop::Or(*op),
            Prop::Xor(op) => Prop::Xor(*op),
            Prop::Not(op) => Prop::Not(*op),
            Prop::Eq(op) => Prop::Eq(*op),
            Prop::Ne(op) => Prop::Ne(*op),
            Prop::Ite(op) => Prop::Ite(*op),
        }
    }
}

impl<T: fmt::Display> fmt::Display for Prop<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Prop::Const(op) => op.fmt(f),
            Prop::And(op) => op.fmt(f),
            Prop::Or(op) => op.fmt(f),
            Prop::Xor(op) => op.fmt(f),
            Prop::Not(op) => op.fmt(f),
            Prop::Eq(op) => op.fmt(f),
            Prop::Ne(op) => op.fmt(f),
            Prop::Ite(op) => op.fmt(f),
        }
    }
}

impl<T: Tensor> Boundary<Bool> for Prop<T> {
    fn dom(&self) -> Poly<Bool> {
        match self {
            Prop::Const(op) => op.dom(),
            Prop::And(op) => op.dom(),
            Prop::Or(op) => op.dom(),
            Prop::Xor(op) => op.dom(),
            Prop::Not(op) => op.dom(),
            Prop::Eq(op) => op.dom(),
            Prop::Ne(op) => op.dom(),
            Prop::Ite(op) => op.dom(),
        }
    }

    fn cod(&self) -> Poly<Bool> {
        match self {
            Prop::Const(op) => op.cod(),
            Prop::And(op) => op.cod(),
            Prop::Or(op) => op.cod(),
            Prop::Xor(op) => op.cod(),
            Prop::Not(op) => op.cod(),
            Prop::Eq(op) => op.cod(),
            Prop::Ne(op) => op.cod(),
            Prop::Ite(op) => op.cod(),
        }
    }
}

impl<T: Tensor> Boundary<Bool> for Const<T> {
    fn dom(&self) -> Poly<Bool> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool::new(self.0.shape())].into()
    }
}

impl Boundary<Bool> for And {
    fn dom(&self) -> Poly<Bool> {
        mono![Bool::new(self.0); 2].into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool::new(self.0)].into()
    }
}

impl Boundary<Bool> for Or {
    fn dom(&self) -> Poly<Bool> {
        mono![Bool::new(self.0); 2].into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool::new(self.0)].into()
    }
}

impl Boundary<Bool> for Xor {
    fn dom(&self) -> Poly<Bool> {
        mono![Bool::new(self.0); 2].into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool::new(self.0)].into()
    }
}

impl Boundary<Bool> for Not {
    fn dom(&self) -> Poly<Bool> {
        mono![Bool::new(self.0)].into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool::new(self.0)].into()
    }
}

impl Boundary<Bool> for Eq {
    fn dom(&self) -> Poly<Bool> {
        mono![Bool::new(self.0); 2].into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool::new(self.0)].into()
    }
}

impl Boundary<Bool> for Ne {
    fn dom(&self) -> Poly<Bool> {
        mono![Bool::new(self.0); 2].into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool::new(self.0)].into()
    }
}

impl<T: fmt::Display> fmt::Display for Const<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Display for And {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "And")
    }
}

impl fmt::Display for Or {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Or")
    }
}

impl fmt::Display for Xor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Xor")
    }
}

impl fmt::Display for Not {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Not")
    }
}

impl fmt::Display for Eq {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Eq")
    }
}

impl fmt::Display for Ne {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ne")
    }
}

impl Boundary<Bool> for Ite {
    fn dom(&self) -> Poly<Bool> {
        mono![Bool::new(self.0); 3].into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool::new(self.0)].into()
    }
}

impl fmt::Display for Ite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ite")
    }
}

/// Extracts an operation of propositional logic from a Python object,
/// trying each operation in turn: those carrying no data through their own
/// classes, and [`Const`] through the extraction that the tensor type `T`
/// provides.
impl<'py, T> pyo3::FromPyObject<'py> for Prop<T>
where
    Const<T>: pyo3::FromPyObject<'py>,
{
    fn extract_bound(ob: &pyo3::Bound<'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use pyo3::types::PyAnyMethods;

        if let Ok(op) = ob.extract::<And>() {
            return Ok(Prop::And(op));
        }
        if let Ok(op) = ob.extract::<Or>() {
            return Ok(Prop::Or(op));
        }
        if let Ok(op) = ob.extract::<Xor>() {
            return Ok(Prop::Xor(op));
        }
        if let Ok(op) = ob.extract::<Not>() {
            return Ok(Prop::Not(op));
        }
        if let Ok(op) = ob.extract::<Eq>() {
            return Ok(Prop::Eq(op));
        }
        if let Ok(op) = ob.extract::<Ne>() {
            return Ok(Prop::Ne(op));
        }
        if let Ok(op) = ob.extract::<Ite>() {
            return Ok(Prop::Ite(op));
        }
        if let Ok(op) = ob.extract::<Const<T>>() {
            return Ok(Prop::Const(op));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "expected an operation of propositional logic, got {}",
            ob.get_type()
        )))
    }
}

// The Python constructors of the operations carrying no tensor: each takes
// its sizes positionally, in the order of its fields, the shape as a list.
#[pymethods]
impl And {
    #[new]
    #[pyo3(signature = (shape, /))]
    fn new(shape: [usize; 2]) -> Self {
        And(shape)
    }

    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[getter]
    fn shape(&self) -> [usize; 2] {
        self.0
    }
}

#[pymethods]
impl Or {
    #[new]
    #[pyo3(signature = (shape, /))]
    fn new(shape: [usize; 2]) -> Self {
        Or(shape)
    }

    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[getter]
    fn shape(&self) -> [usize; 2] {
        self.0
    }
}

#[pymethods]
impl Xor {
    #[new]
    #[pyo3(signature = (shape, /))]
    fn new(shape: [usize; 2]) -> Self {
        Xor(shape)
    }

    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[getter]
    fn shape(&self) -> [usize; 2] {
        self.0
    }
}

#[pymethods]
impl Not {
    #[new]
    #[pyo3(signature = (shape, /))]
    fn new(shape: [usize; 2]) -> Self {
        Not(shape)
    }

    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[getter]
    fn shape(&self) -> [usize; 2] {
        self.0
    }
}

#[pymethods]
impl Eq {
    #[new]
    #[pyo3(signature = (shape, /))]
    fn new(shape: [usize; 2]) -> Self {
        Eq(shape)
    }

    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[getter]
    fn shape(&self) -> [usize; 2] {
        self.0
    }
}

#[pymethods]
impl Ne {
    #[new]
    #[pyo3(signature = (shape, /))]
    fn new(shape: [usize; 2]) -> Self {
        Ne(shape)
    }

    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[getter]
    fn shape(&self) -> [usize; 2] {
        self.0
    }
}

#[pymethods]
impl Ite {
    #[new]
    #[pyo3(signature = (shape, /))]
    fn new(shape: [usize; 2]) -> Self {
        Ite(shape)
    }

    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[getter]
    fn shape(&self) -> [usize; 2] {
        self.0
    }
}
