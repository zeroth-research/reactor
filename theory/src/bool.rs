/*!
# Boolean algebra

Defines one type per operation on the sort [`Bool`] of boolean matrices. Every
operation reads and writes boolean matrices, and carries enough
information to compute its [`Boundary`]:

- [`Truth`] — a constant boolean matrix literal.
- [`And`], [`Or`], [`Xor`] — pointwise conjunction, disjunction and
  exclusive disjunction.
- [`Not`] — pointwise negation.
- [`Eq`](struct@Eq), [`Ne`] — pointwise equality and disequality.
- [`Ite`] — the pointwise choice `if c then x else y`, by a boolean matrix `c`.
- [`broadcast::Ite`] — the choice of the whole matrices, by a scalar boolean
  `c` of the shape `[1, 1]`.
- [`reduce::And`], [`reduce::Or`] — the conjunction and disjunction of the
  entries of a matrix along an axis: all and any.
*/

use crate::Matrix;
use crate::zero::Real0;
#[cfg(feature = "pyo3")]
use base::boundary::Mono;
use base::boundary::{Boundary, Poly};
use base::differentiable::Differentiable;
use base::mono;
use derive_more::{Display, From};
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use std::fmt;

/// The sort of boolean matrices with the given `[rows, cols]` shape.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, get_all, module = "reactor.bool")
)]
pub struct Bool {
    shape: [usize; 2],
}

impl Bool {
    /// Boolean matrices of the given shape, as [`Bool::new`], but panics if
    /// `shape` is empty.
    pub fn new_unchecked(shape: [usize; 2]) -> Self {
        Self::new(shape).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

/// Python matches a `Bool` by `shape`.
#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Bool {
    /// Boolean matrices of the given shape; fails unless the shape has at
    /// least one row and one column.
    #[cfg_attr(feature = "pyo3", new)]
    pub fn new(shape: [usize; 2]) -> Result<Self, crate::Error> {
        crate::matrix::check_shape(shape)?;
        Ok(Bool { shape })
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    /// The tangent sort `T(self)`: the sort of the rate of change of its
    /// values.
    #[cfg(feature = "pyo3")]
    #[pyo3(name = "tangent")]
    fn tangent(&self) -> Real0 {
        Differentiable::tangent(self)
    }
}

impl fmt::Display for Bool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [i, j] = self.shape;
        write!(f, "Bool([{i}, {j}])")
    }
}

/// Booleans are discrete, so their tangent is trivial: the singleton sort
/// [`Real0`].
impl Differentiable for Bool {
    type Tangent = Real0;

    fn tangent(self) -> Real0 {
        Real0
    }
}

/// A constant boolean matrix literal, held in a tensor of type `T`.
#[derive(Clone, Copy, Debug, Display)]
#[display("{tensor}")]
pub struct Truth<T> {
    tensor: T,
}

impl<T: Matrix> Truth<T> {
    /// The constant held in `tensor`, writing the boolean matrix of its
    /// shape; fails unless it has at least one row and one column.
    pub fn new(tensor: T) -> Result<Self, crate::Error> {
        crate::matrix::check_shape(tensor.shape())?;
        Ok(Truth { tensor })
    }

    /// The constant held in `tensor`, writing the boolean matrix of `cod`; fails
    /// unless `tensor` is of its shape.
    pub fn with_cod(tensor: T, cod: [Bool; 1]) -> Result<Self, crate::Error> {
        let [x] = cod;
        if tensor.shape() != x.shape() {
            return Err(crate::unexpected_cod(
                &format!("1 sort of the shape {:?} of the constant", tensor.shape()),
                &cod,
            ));
        }
        Ok(Truth { tensor })
    }
}

impl<T> Truth<T> {
    /// The tensor holding the constant.
    pub fn tensor(&self) -> &T {
        &self.tensor
    }
}

/// Pointwise conjunction `x ∧ y` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
#[display("And")]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.bool")
)]
pub struct And {
    shape: [usize; 2],
}

/// Pointwise disjunction `x ∨ y` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
#[display("Or")]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.bool")
)]
pub struct Or {
    shape: [usize; 2],
}

/// Pointwise exclusive disjunction `x ⊕ y` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
#[display("Xor")]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.bool")
)]
pub struct Xor {
    shape: [usize; 2],
}

/// Pointwise negation `¬x` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
#[display("Not")]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.bool")
)]
pub struct Not {
    shape: [usize; 2],
}

/// Pointwise equality `x = y` of `[m, n]` matrices: the negation of their
/// exclusive disjunction.
///
/// SMT-LIB2: `(= x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
#[display("Eq")]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.bool")
)]
pub struct Eq {
    shape: [usize; 2],
}

/// Pointwise disequality `x ≠ y` of `[m, n]` matrices: their exclusive
/// disjunction.
///
/// SMT-LIB2: `(distinct x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
#[display("Ne")]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.bool")
)]
pub struct Ne {
    shape: [usize; 2],
}

/// Pointwise choice `if c then x else y` between `[m, n]` boolean matrices
/// `x` and `y`, by the boolean matrix `c`, read first.
///
/// SMT-LIB2: `(ite c x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
#[display("Ite")]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.bool")
)]
pub struct Ite {
    shape: [usize; 2],
}

/// The operations of propositional logic, the boolean operations of
/// SMT-LIB2's `Core` theory, with their boundary in [`Bool`].
#[derive(Clone, Copy, Debug, From, Display)]
pub enum Prop<T> {
    Truth(Truth<T>),
    And(And),
    Or(Or),
    Xor(Xor),
    Not(Not),
    Eq(Eq),
    Ne(Ne),
    Ite(Ite),
    BroadcastIte(broadcast::Ite),
    ReduceAnd(reduce::And),
    ReduceOr(reduce::Or),
}

impl<T: Matrix> Boundary<Bool> for Prop<T> {
    fn dom(&self) -> Poly<Bool> {
        match self {
            Prop::Truth(op) => op.dom(),
            Prop::And(op) => op.dom(),
            Prop::Or(op) => op.dom(),
            Prop::Xor(op) => op.dom(),
            Prop::Not(op) => op.dom(),
            Prop::Eq(op) => op.dom(),
            Prop::Ne(op) => op.dom(),
            Prop::Ite(op) => op.dom(),
            Prop::BroadcastIte(op) => op.dom(),
            Prop::ReduceAnd(op) => op.dom(),
            Prop::ReduceOr(op) => op.dom(),
        }
    }

    fn cod(&self) -> Poly<Bool> {
        match self {
            Prop::Truth(op) => op.cod(),
            Prop::And(op) => op.cod(),
            Prop::Or(op) => op.cod(),
            Prop::Xor(op) => op.cod(),
            Prop::Not(op) => op.cod(),
            Prop::Eq(op) => op.cod(),
            Prop::Ne(op) => op.cod(),
            Prop::Ite(op) => op.cod(),
            Prop::BroadcastIte(op) => op.cod(),
            Prop::ReduceAnd(op) => op.cod(),
            Prop::ReduceOr(op) => op.cod(),
        }
    }
}

impl<T: Matrix> Boundary<Bool> for Truth<T> {
    fn dom(&self) -> Poly<Bool> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool::new_unchecked(self.tensor.shape())].into()
    }
}

impl Boundary<Bool> for And {
    fn dom(&self) -> Poly<Bool> {
        mono![Bool::new_unchecked(self.shape); 2].into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool::new_unchecked(self.shape)].into()
    }
}

impl Boundary<Bool> for Or {
    fn dom(&self) -> Poly<Bool> {
        mono![Bool::new_unchecked(self.shape); 2].into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool::new_unchecked(self.shape)].into()
    }
}

impl Boundary<Bool> for Xor {
    fn dom(&self) -> Poly<Bool> {
        mono![Bool::new_unchecked(self.shape); 2].into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool::new_unchecked(self.shape)].into()
    }
}

impl Boundary<Bool> for Not {
    fn dom(&self) -> Poly<Bool> {
        mono![Bool::new_unchecked(self.shape)].into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool::new_unchecked(self.shape)].into()
    }
}

impl Boundary<Bool> for Eq {
    fn dom(&self) -> Poly<Bool> {
        mono![Bool::new_unchecked(self.shape); 2].into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool::new_unchecked(self.shape)].into()
    }
}

impl Boundary<Bool> for Ne {
    fn dom(&self) -> Poly<Bool> {
        mono![Bool::new_unchecked(self.shape); 2].into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool::new_unchecked(self.shape)].into()
    }
}

impl Boundary<Bool> for Ite {
    fn dom(&self) -> Poly<Bool> {
        mono![Bool::new_unchecked(self.shape); 3].into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool::new_unchecked(self.shape)].into()
    }
}

/// Extracts an operation of propositional logic from a Python object,
/// trying each operation in turn: those carrying no data through their own
/// classes, and [`Truth`] through the extraction that the tensor type `T`
/// provides.
#[cfg(feature = "pyo3")]
impl<'a, 'py, T> pyo3::FromPyObject<'a, 'py> for Prop<T>
where
    Truth<T>: pyo3::FromPyObject<'a, 'py>,
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
        if let Ok(op) = ob.extract::<Ite>() {
            return Ok(Prop::Ite(op));
        }
        if let Ok(op) = ob.extract::<broadcast::Ite>() {
            return Ok(Prop::BroadcastIte(op));
        }
        if let Ok(op) = ob.extract::<reduce::And>() {
            return Ok(Prop::ReduceAnd(op));
        }
        if let Ok(op) = ob.extract::<reduce::Or>() {
            return Ok(Prop::ReduceOr(op));
        }
        if let Ok(op) = ob.extract::<Truth<T>>() {
            return Ok(Prop::Truth(op));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "expected an operation of propositional logic, got {}",
            ob.get_type()
        )))
    }
}

impl And {
    /// The operation reading the boolean matrices of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Bool; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(And { shape: s.shape() })
    }

    /// The operation writing the boolean matrix of `cod`.
    pub fn with_cod(cod: [Bool; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(And { shape: s.shape() })
    }
}

// The Python constructors of the operations carrying no tensor: each takes
// its sizes positionally, in the order of its fields, the shape as a list.
#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl And {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Bool; 2], cod: Option<Bool>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads, as a tuple.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom(&self) -> Mono<Bool> {
        Mono::try_from(Boundary::dom(self)).unwrap()
    }

    /// The sorts it writes, as a tuple.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod(&self) -> Mono<Bool> {
        Mono::try_from(Boundary::cod(self)).unwrap()
    }
}

impl Or {
    /// The operation reading the boolean matrices of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Bool; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Or { shape: s.shape() })
    }

    /// The operation writing the boolean matrix of `cod`.
    pub fn with_cod(cod: [Bool; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Or { shape: s.shape() })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Or {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Bool; 2], cod: Option<Bool>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads, as a tuple.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom(&self) -> Mono<Bool> {
        Mono::try_from(Boundary::dom(self)).unwrap()
    }

    /// The sorts it writes, as a tuple.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod(&self) -> Mono<Bool> {
        Mono::try_from(Boundary::cod(self)).unwrap()
    }
}

impl Xor {
    /// The operation reading the boolean matrices of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Bool; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Xor { shape: s.shape() })
    }

    /// The operation writing the boolean matrix of `cod`.
    pub fn with_cod(cod: [Bool; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Xor { shape: s.shape() })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Xor {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Bool; 2], cod: Option<Bool>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads, as a tuple.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom(&self) -> Mono<Bool> {
        Mono::try_from(Boundary::dom(self)).unwrap()
    }

    /// The sorts it writes, as a tuple.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod(&self) -> Mono<Bool> {
        Mono::try_from(Boundary::cod(self)).unwrap()
    }
}

impl Not {
    /// The operation reading the boolean matrix of `dom`.
    pub fn with_dom(dom: [Bool; 1]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Not { shape: s.shape() })
    }

    /// The operation writing the boolean matrix of `cod`.
    pub fn with_cod(cod: [Bool; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Not { shape: s.shape() })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Not {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: Bool, cod: Option<Bool>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom([dom]), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads, as a tuple.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom(&self) -> Mono<Bool> {
        Mono::try_from(Boundary::dom(self)).unwrap()
    }

    /// The sorts it writes, as a tuple.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod(&self) -> Mono<Bool> {
        Mono::try_from(Boundary::cod(self)).unwrap()
    }
}

impl Eq {
    /// The operation reading the boolean matrices of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Bool; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Eq { shape: s.shape() })
    }

    /// The operation writing the boolean matrix of `cod`.
    pub fn with_cod(cod: [Bool; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Eq { shape: s.shape() })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Eq {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Bool; 2], cod: Option<Bool>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads, as a tuple.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom(&self) -> Mono<Bool> {
        Mono::try_from(Boundary::dom(self)).unwrap()
    }

    /// The sorts it writes, as a tuple.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod(&self) -> Mono<Bool> {
        Mono::try_from(Boundary::cod(self)).unwrap()
    }
}

impl Ne {
    /// The operation reading the boolean matrices of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Bool; 2]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Ne { shape: s.shape() })
    }

    /// The operation writing the boolean matrix of `cod`.
    pub fn with_cod(cod: [Bool; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Ne { shape: s.shape() })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Ne {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Bool; 2], cod: Option<Bool>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads, as a tuple.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom(&self) -> Mono<Bool> {
        Mono::try_from(Boundary::dom(self)).unwrap()
    }

    /// The sorts it writes, as a tuple.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod(&self) -> Mono<Bool> {
        Mono::try_from(Boundary::cod(self)).unwrap()
    }
}

impl Ite {
    /// The operation reading the boolean matrices of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Bool; 3]) -> Result<Self, crate::Error> {
        let s = crate::same_sort(dom)?;
        Ok(Ite { shape: s.shape() })
    }

    /// The operation writing the boolean matrix of `cod`.
    pub fn with_cod(cod: [Bool; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Ite { shape: s.shape() })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Ite {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Bool; 3], cod: Option<Bool>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads, as a tuple.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom(&self) -> Mono<Bool> {
        Mono::try_from(Boundary::dom(self)).unwrap()
    }

    /// The sorts it writes, as a tuple.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod(&self) -> Mono<Bool> {
        Mono::try_from(Boundary::cod(self)).unwrap()
    }
}

/// The operations of the theory that broadcast a scalar operand, of the
/// shape `[1, 1]`, to the shape of the others; the others are in the parent
/// module.
pub mod broadcast {
    use super::*;

    /// Choice `if c then x else y` between whole `[m, n]` boolean matrices `x` and
    /// `y`, by the scalar boolean `c`, of the shape `[1, 1]`, read first: the
    /// pointwise [`super::Ite`] by `c` broadcast to `[m, n]`.
    ///
    /// SMT-LIB2: `(ite c x y)`, by the only entry of `c`.
    #[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
    #[display("Ite")]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.bool.broadcast")
    )]
    pub struct Ite {
        shape: [usize; 2],
    }

    impl Boundary<Bool> for Ite {
        fn dom(&self) -> Poly<Bool> {
            mono![
                Bool::new_unchecked([1, 1]),
                Bool::new_unchecked(self.shape),
                Bool::new_unchecked(self.shape)
            ]
            .into()
        }

        fn cod(&self) -> Poly<Bool> {
            mono![Bool::new_unchecked(self.shape)].into()
        }
    }

    impl Ite {
        /// The choice reading the scalar boolean condition and the branches of
        /// `dom`; fails unless the condition is of the shape `[1, 1]` and the
        /// branches of the same sort.
        pub fn with_dom(dom: [Bool; 3]) -> Result<Self, crate::Error> {
            let [c, x, y] = dom;
            if c.shape() != [1, 1] || x != y {
                return Err(crate::unexpected_dom(
                    "a condition Bool([1, 1]) and two branches Bool of the same sort",
                    &dom,
                ));
            }
            Ok(Ite { shape: x.shape() })
        }

        /// The choice writing the boolean matrix of `cod`.
        pub fn with_cod(cod: [Bool; 1]) -> Result<Self, crate::Error> {
            let [s] = cod;
            Ok(Ite { shape: s.shape() })
        }
    }

    #[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
    impl Ite {
        /// Python builds it from the parameters that its boundary does not determine,
        /// and its domain `dom`; its codomain `cod`, when given, must be the one it
        /// writes.
        #[cfg(feature = "pyo3")]
        #[new]
        #[pyo3(signature = (dom, cod = None))]
        fn py_new(dom: [Bool; 3], cod: Option<Bool>) -> Result<Self, crate::Error> {
            crate::check_cod(Self::with_dom(dom), cod)
        }

        /// The parameters that its boundary does not determine, then its domain, when it
        /// reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str) {
            ("dom", "cod")
        }

        /// The sorts it reads, as a tuple.
        #[cfg(feature = "pyo3")]
        #[getter(dom)]
        fn py_dom(&self) -> Mono<Bool> {
            Mono::try_from(Boundary::dom(self)).unwrap()
        }

        /// The sorts it writes, as a tuple.
        #[cfg(feature = "pyo3")]
        #[getter(cod)]
        fn py_cod(&self) -> Mono<Bool> {
            Mono::try_from(Boundary::cod(self)).unwrap()
        }
    }
}

/// The operations of the theory that reduce a matrix along an axis to a
/// single row or column, by an associative and commutative operation on its
/// entries; the pointwise ones are in the parent module.
///
/// The `axis` is that of the entries reduced, as in NumPy: along `0`, the
/// rows of an `[m, n]` matrix reduce to one, of the shape `[1, n]`, and
/// along `1` its columns to one, of the shape `[m, 1]`. The result keeps both
/// dimensions, so that it remains a matrix.
pub mod reduce {
    use super::*;

    /// The shape of an `[m, n]` matrix reduced along `axis`, which must be 0
    /// or 1.
    fn reduced(shape: [usize; 2], axis: usize) -> [usize; 2] {
        let mut shape = shape;
        shape[axis] = 1;
        shape
    }

    /// Fails unless `axis` is an axis of a matrix: 0 or 1.
    fn check_axis(axis: usize) -> Result<(), crate::Error> {
        if axis > 1 {
            return Err(crate::Error(format!(
                "expected the axis 0 or 1 of a matrix, got {axis}"
            )));
        }
        Ok(())
    }

    /// Conjunction of the entries of an `[m, n]` boolean matrix along `axis`:
    /// whether all of them hold, in each column along 0, writing a `[1, n]`
    /// matrix, and in each row along 1, writing an `[m, 1]` matrix.
    ///
    /// SMT-LIB2: `(and x₁ … xₖ)` of the entries reduced.
    #[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
    #[display("And")]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.bool.reduce")
    )]
    pub struct And {
        shape: [usize; 2],
        axis: usize,
    }

    /// Disjunction of the entries of an `[m, n]` boolean matrix along `axis`:
    /// whether any of them holds, in each column along 0, writing a `[1, n]`
    /// matrix, and in each row along 1, writing an `[m, 1]` matrix.
    ///
    /// SMT-LIB2: `(or x₁ … xₖ)` of the entries reduced.
    #[derive(Clone, Copy, PartialEq, Debug, Eq, Display)]
    #[display("Or")]
    #[cfg_attr(
        feature = "pyo3",
        pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.bool.reduce")
    )]
    pub struct Or {
        shape: [usize; 2],
        axis: usize,
    }

    impl And {
        /// The reduction along `axis` of the boolean matrix of `dom`;
        /// fails unless `axis` is 0 or 1. Its codomain does not
        /// determine it, as it loses the size of the axis.
        pub fn with_dom(axis: usize, dom: [Bool; 1]) -> Result<Self, crate::Error> {
            check_axis(axis)?;
            let [s] = dom;
            Ok(And {
                shape: s.shape(),
                axis,
            })
        }
    }

    impl Boundary<Bool> for And {
        fn dom(&self) -> Poly<Bool> {
            mono![Bool::new_unchecked(self.shape)].into()
        }

        fn cod(&self) -> Poly<Bool> {
            mono![Bool::new_unchecked(reduced(self.shape, self.axis))].into()
        }
    }

    #[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
    impl And {
        /// Python builds it from its axis, which its boundary does not
        /// determine, and its domain `dom`; its codomain `cod`, when
        /// given, must be the one it writes.
        #[cfg(feature = "pyo3")]
        #[new]
        #[pyo3(signature = (axis, dom, cod = None))]
        fn py_new(axis: usize, dom: Bool, cod: Option<Bool>) -> Result<Self, crate::Error> {
            crate::check_cod(Self::with_dom(axis, [dom]), cod)
        }

        /// The parameters that its boundary does not determine, then its
        /// domain, when it reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str, &'static str) {
            ("axis", "dom", "cod")
        }

        /// The sorts it reads, as a tuple.
        #[cfg(feature = "pyo3")]
        #[getter(dom)]
        fn py_dom(&self) -> Mono<Bool> {
            Mono::try_from(Boundary::dom(self)).unwrap()
        }

        /// The sorts it writes, as a tuple.
        #[cfg(feature = "pyo3")]
        #[getter(cod)]
        fn py_cod(&self) -> Mono<Bool> {
            Mono::try_from(Boundary::cod(self)).unwrap()
        }

        /// The axis along which it reduces: 0 for the rows, 1 for the
        /// columns.
        #[cfg_attr(feature = "pyo3", getter)]
        pub fn axis(&self) -> usize {
            self.axis
        }
    }

    impl Or {
        /// The reduction along `axis` of the boolean matrix of `dom`;
        /// fails unless `axis` is 0 or 1. Its codomain does not
        /// determine it, as it loses the size of the axis.
        pub fn with_dom(axis: usize, dom: [Bool; 1]) -> Result<Self, crate::Error> {
            check_axis(axis)?;
            let [s] = dom;
            Ok(Or {
                shape: s.shape(),
                axis,
            })
        }
    }

    impl Boundary<Bool> for Or {
        fn dom(&self) -> Poly<Bool> {
            mono![Bool::new_unchecked(self.shape)].into()
        }

        fn cod(&self) -> Poly<Bool> {
            mono![Bool::new_unchecked(reduced(self.shape, self.axis))].into()
        }
    }

    #[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
    impl Or {
        /// Python builds it from its axis, which its boundary does not
        /// determine, and its domain `dom`; its codomain `cod`, when
        /// given, must be the one it writes.
        #[cfg(feature = "pyo3")]
        #[new]
        #[pyo3(signature = (axis, dom, cod = None))]
        fn py_new(axis: usize, dom: Bool, cod: Option<Bool>) -> Result<Self, crate::Error> {
            crate::check_cod(Self::with_dom(axis, [dom]), cod)
        }

        /// The parameters that its boundary does not determine, then its
        /// domain, when it reads anything, and its codomain.
        #[cfg(feature = "pyo3")]
        #[classattr]
        fn __match_args__() -> (&'static str, &'static str, &'static str) {
            ("axis", "dom", "cod")
        }

        /// The sorts it reads, as a tuple.
        #[cfg(feature = "pyo3")]
        #[getter(dom)]
        fn py_dom(&self) -> Mono<Bool> {
            Mono::try_from(Boundary::dom(self)).unwrap()
        }

        /// The sorts it writes, as a tuple.
        #[cfg(feature = "pyo3")]
        #[getter(cod)]
        fn py_cod(&self) -> Mono<Bool> {
            Mono::try_from(Boundary::cod(self)).unwrap()
        }

        /// The axis along which it reduces: 0 for the rows, 1 for the
        /// columns.
        #[cfg_attr(feature = "pyo3", getter)]
        pub fn axis(&self) -> usize {
            self.axis
        }
    }
}
