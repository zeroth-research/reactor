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
- [`All`] — the conjunction of all the entries of a matrix, as a `[1, 1]`
  matrix: what reduces a pointwise comparison to the condition of an
  [`Ite`].
- [`Ite`] — the choice `if c then x else y` between whole matrices, by a
  `[1, 1]` boolean matrix `c`.
- [`Implies`] — pointwise implication.
- [`Get`] — one entry of a matrix.
*/

use super::sort::Bool;
use crate::Tensor;
use base::boundary::{Boundary, Poly};
use base::mono;
use derive_more::From;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use std::fmt;

/// A constant boolean matrix literal, held in a tensor of type `T`.
#[derive(Debug)]
pub struct Const<T> {
    tensor: T,
}

impl<T> Const<T> {
    /// The constant held in `tensor`.
    pub fn new(tensor: T) -> Self {
        Const { tensor }
    }

    /// The tensor holding the constant.
    pub fn tensor(&self) -> &T {
        &self.tensor
    }
}

/// Pointwise conjunction `x ∧ y` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, from_py_object, eq, str))]
pub struct And {
    shape: [usize; 2],
}

/// Pointwise disjunction `x ∨ y` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, from_py_object, eq, str))]
pub struct Or {
    shape: [usize; 2],
}

/// Pointwise exclusive disjunction `x ⊕ y` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, from_py_object, eq, str))]
pub struct Xor {
    shape: [usize; 2],
}

/// Pointwise negation `¬x` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, from_py_object, eq, str))]
pub struct Not {
    shape: [usize; 2],
}

/// Pointwise equality `x = y` of `[m, n]` matrices: the negation of their
/// exclusive disjunction.
///
/// SMT-LIB2: `(= x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, from_py_object, eq, str))]
pub struct Eq {
    shape: [usize; 2],
}

/// Pointwise disequality `x ≠ y` of `[m, n]` matrices: their exclusive
/// disjunction.
///
/// SMT-LIB2: `(distinct x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, from_py_object, eq, str))]
pub struct Ne {
    shape: [usize; 2],
}

/// The conjunction `⋀ᵢⱼ xᵢⱼ` of all the entries of an `[m, n]` matrix,
/// written as a `[1, 1]` matrix: true iff every entry is.
///
/// SMT-LIB2: `(and x₀₀ … xₘₙ)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, from_py_object, eq, str))]
pub struct All {
    shape: [usize; 2],
}

/// The choice `if c then x else y` between `[m, n]` boolean matrices `x`
/// and `y`, by the `[1, 1]` boolean matrix `c`, read first: one of the two
/// whole matrices, not entry by entry.
///
/// SMT-LIB2: `(ite c x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, from_py_object, eq, str))]
pub struct Ite {
    shape: [usize; 2],
}

/// Pointwise implication `x → y` of `[m, n]` matrices: the disjunction of
/// `¬x` and `y`.
///
/// SMT-LIB2: `(=> x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, from_py_object, eq, str))]
pub struct Implies {
    shape: [usize; 2],
}

/// The entry in row `row` and column `col` of an `[m, n]` boolean matrix,
/// written as a `[1, 1]` matrix.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, from_py_object, eq, str))]
pub struct Get {
    row: usize,
    col: usize,
    shape: [usize; 2],
}

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
    All(All),
    Ite(Ite),
    Implies(Implies),
    Get(Get),
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
            Prop::All(op) => Prop::All(*op),
            Prop::Ite(op) => Prop::Ite(*op),
            Prop::Implies(op) => Prop::Implies(*op),
            Prop::Get(op) => Prop::Get(*op),
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
            Prop::All(op) => op.fmt(f),
            Prop::Ite(op) => op.fmt(f),
            Prop::Implies(op) => op.fmt(f),
            Prop::Get(op) => op.fmt(f),
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
            Prop::All(op) => op.dom(),
            Prop::Ite(op) => op.dom(),
            Prop::Implies(op) => op.dom(),
            Prop::Get(op) => op.dom(),
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
            Prop::All(op) => op.cod(),
            Prop::Ite(op) => op.cod(),
            Prop::Implies(op) => op.cod(),
            Prop::Get(op) => op.cod(),
        }
    }
}

impl<T: Tensor> Boundary<Bool> for Const<T> {
    fn dom(&self) -> Poly<Bool> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool::new(self.tensor.shape())].into()
    }
}

impl Boundary<Bool> for And {
    fn dom(&self) -> Poly<Bool> {
        mono![Bool::new(self.shape); 2].into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool::new(self.shape)].into()
    }
}

impl Boundary<Bool> for Or {
    fn dom(&self) -> Poly<Bool> {
        mono![Bool::new(self.shape); 2].into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool::new(self.shape)].into()
    }
}

impl Boundary<Bool> for Xor {
    fn dom(&self) -> Poly<Bool> {
        mono![Bool::new(self.shape); 2].into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool::new(self.shape)].into()
    }
}

impl Boundary<Bool> for Not {
    fn dom(&self) -> Poly<Bool> {
        mono![Bool::new(self.shape)].into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool::new(self.shape)].into()
    }
}

impl Boundary<Bool> for Eq {
    fn dom(&self) -> Poly<Bool> {
        mono![Bool::new(self.shape); 2].into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool::new(self.shape)].into()
    }
}

impl Boundary<Bool> for Ne {
    fn dom(&self) -> Poly<Bool> {
        mono![Bool::new(self.shape); 2].into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool::new(self.shape)].into()
    }
}

/// Takes a matrix of shape `[m, n]` to one of shape `[1, 1]`.
impl Boundary<Bool> for All {
    fn dom(&self) -> Poly<Bool> {
        mono![Bool::new(self.shape)].into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool::new([1, 1])].into()
    }
}

impl<T: fmt::Display> fmt::Display for Const<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.tensor)
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

impl fmt::Display for All {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "All")
    }
}

/// Reads the `[1, 1]` condition, then the two `[m, n]` branches.
impl Boundary<Bool> for Ite {
    fn dom(&self) -> Poly<Bool> {
        mono![
            Bool::new([1, 1]),
            Bool::new(self.shape),
            Bool::new(self.shape)
        ]
        .into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool::new(self.shape)].into()
    }
}

impl fmt::Display for Ite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ite")
    }
}

impl Boundary<Bool> for Implies {
    fn dom(&self) -> Poly<Bool> {
        mono![Bool::new(self.shape); 2].into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool::new(self.shape)].into()
    }
}

impl fmt::Display for Implies {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Implies")
    }
}

impl Boundary<Bool> for Get {
    fn dom(&self) -> Poly<Bool> {
        mono![Bool::new(self.shape)].into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool::new([1, 1])].into()
    }
}

impl fmt::Display for Get {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Get[{}][{}]", self.row, self.col)
    }
}

/// Extracts an operation of propositional logic from a Python object,
/// trying each operation in turn: those carrying no data through their own
/// classes, and [`Const`] through the extraction that the tensor type `T`
/// provides.
#[cfg(feature = "pyo3")]
impl<'a, 'py, T> pyo3::FromPyObject<'a, 'py> for Prop<T>
where
    Const<T>: pyo3::FromPyObject<'a, 'py>,
{
    type Error = pyo3::PyErr;

    fn extract(ob: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
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
        if let Ok(op) = ob.extract::<All>() {
            return Ok(Prop::All(op));
        }
        if let Ok(op) = ob.extract::<Ite>() {
            return Ok(Prop::Ite(op));
        }
        if let Ok(op) = ob.extract::<Implies>() {
            return Ok(Prop::Implies(op));
        }
        if let Ok(op) = ob.extract::<Get>() {
            return Ok(Prop::Get(op));
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
#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl And {
    #[cfg_attr(feature = "pyo3", new)]
    #[cfg_attr(feature = "pyo3", pyo3(signature = (shape, /)))]
    pub fn new(shape: [usize; 2]) -> Self {
        And { shape }
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Or {
    #[cfg_attr(feature = "pyo3", new)]
    #[cfg_attr(feature = "pyo3", pyo3(signature = (shape, /)))]
    pub fn new(shape: [usize; 2]) -> Self {
        Or { shape }
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Xor {
    #[cfg_attr(feature = "pyo3", new)]
    #[cfg_attr(feature = "pyo3", pyo3(signature = (shape, /)))]
    pub fn new(shape: [usize; 2]) -> Self {
        Xor { shape }
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Not {
    #[cfg_attr(feature = "pyo3", new)]
    #[cfg_attr(feature = "pyo3", pyo3(signature = (shape, /)))]
    pub fn new(shape: [usize; 2]) -> Self {
        Not { shape }
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Eq {
    #[cfg_attr(feature = "pyo3", new)]
    #[cfg_attr(feature = "pyo3", pyo3(signature = (shape, /)))]
    pub fn new(shape: [usize; 2]) -> Self {
        Eq { shape }
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Ne {
    #[cfg_attr(feature = "pyo3", new)]
    #[cfg_attr(feature = "pyo3", pyo3(signature = (shape, /)))]
    pub fn new(shape: [usize; 2]) -> Self {
        Ne { shape }
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl All {
    /// The conjunction of the entries of `[m, n]` matrices, `shape`.
    #[cfg_attr(feature = "pyo3", new)]
    #[cfg_attr(feature = "pyo3", pyo3(signature = (shape, /)))]
    pub fn new(shape: [usize; 2]) -> Self {
        All { shape }
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    /// The shape `[m, n]` of the matrix read.
    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Ite {
    #[cfg_attr(feature = "pyo3", new)]
    #[cfg_attr(feature = "pyo3", pyo3(signature = (shape, /)))]
    pub fn new(shape: [usize; 2]) -> Self {
        Ite { shape }
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Implies {
    #[cfg_attr(feature = "pyo3", new)]
    #[cfg_attr(feature = "pyo3", pyo3(signature = (shape, /)))]
    pub fn new(shape: [usize; 2]) -> Self {
        Implies { shape }
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Get {
    /// Fails unless `row` and `col` index an entry of an `[m, n]` matrix.
    #[cfg_attr(feature = "pyo3", new)]
    #[cfg_attr(feature = "pyo3", pyo3(signature = (row, col, shape, /)))]
    pub fn new(row: usize, col: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
        super::check_entry(row, col, shape)?;
        Ok(Get { row, col, shape })
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str, &'static str) {
        ("row", "col", "shape")
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn row(&self) -> usize {
        self.row
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn col(&self) -> usize {
        self.col
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}
