/*!
# Boolean algebra

Defines one type per operation on the sort [`Bool`] of boolean matrices. Every
operation reads and writes boolean matrices, and carries enough
information to compute its [`Boundary`]:

- [`Const`] — a constant boolean matrix literal.
- [`And`], [`Or`], [`Xor`] — pointwise conjunction, disjunction and
  exclusive disjunction.
- [`Not`] — pointwise negation.
- [`Id`] — the identity.
- [`Ite`] — if-then-else, on a single boolean.
- [`Havoc`] — an arbitrary matrix, chosen nondeterministically.
- [`Uninterpreted`] — a named value of unknown meaning.
- [`Zero`] — the only value of the sort [`Zero`](crate::sort::Zero).

[`Zero`] writes no booleans, but the rate of change of discrete values,
of the sort [`Zero`](crate::sort::Zero), which all theories share, as
they share [`Bool`]. It is here so that every logic holds it, as each
holds [`Prop`]; so the boundary of `Prop` is in [`Sort`], the sorts of
the theory, rather than in [`Bool`].
*/

use crate::Tensor;
use crate::sort::Bool;
use base::boundary::{Boundary, Poly};
use base::mono;
use derive_more::{Display, From};
use std::fmt;

/// The sorts of the theory: the booleans of SMT-LIB2's `Core` theory,
/// together with [`Zero`](crate::sort::Zero), which [`Zero`] writes. It is
/// the subset [`sort::Prop`](crate::sort::Prop) of the sorts of all
/// theories.
pub use crate::sort::Prop as Sort;

/// A constant boolean matrix literal, held in a tensor of type `T`.
#[derive(Clone, Debug)]
pub struct Const<T>(pub T);

/// Pointwise conjunction `x ∧ y` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, eq, str, module = "reactor.op.bool")
)]
pub struct And(pub usize, pub usize);

/// Pointwise disjunction `x ∨ y` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, eq, str, module = "reactor.op.bool")
)]
pub struct Or(pub usize, pub usize);

/// Pointwise exclusive disjunction `x ⊕ y` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, eq, str, module = "reactor.op.bool")
)]
pub struct Xor(pub usize, pub usize);

/// Pointwise negation `¬x` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, eq, str, module = "reactor.op.bool")
)]
pub struct Not(pub usize, pub usize);

/// The identity `x ↦ x` on `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, eq, str, module = "reactor.op.bool")
)]
pub struct Id(pub usize, pub usize);

/// If-then-else `if c then x else y` of `[m, n]` matrices, on a single
/// boolean `c`, which picks either matrix whole.
///
/// SMT-LIB2: `(ite c x y)`, of the Core theory, with the same `c` for every
/// entry.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, eq, str, module = "reactor.op.bool")
)]
pub struct Ite(pub usize, pub usize);

/// An arbitrary boolean matrix of shape `[m, n]`, chosen nondeterministically.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, eq, str, module = "reactor.op.bool")
)]
pub struct Havoc(pub usize, pub usize);

/// A boolean matrix of shape `[m, n]` of unknown meaning, named by the given
/// symbol, as `Uninterpreted(symbol, m, n)`: e.g., the result of a call that
/// the frontend cannot translate.
#[derive(Clone, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, eq, str, module = "reactor.op.bool")
)]
pub struct Uninterpreted(pub String, pub usize, pub usize);

/// The only value of the sort [`Zero`](crate::sort::Zero): the rate of
/// change of discrete values.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, eq, str, module = "reactor.op.bool")
)]
pub struct Zero;

/// The operations of propositional logic, the boolean operations of
/// SMT-LIB2's `Core` theory, and [`Zero`], with their boundary in [`Sort`].
#[derive(Clone, Debug, From, Display)]
pub enum Prop<T> {
    Const(Const<T>),
    And(And),
    Or(Or),
    Xor(Xor),
    Not(Not),
    Id(Id),
    Ite(Ite),
    Havoc(Havoc),
    Uninterpreted(Uninterpreted),
    Zero(Zero),
}

impl<T: Tensor> Boundary<Sort> for Prop<T> {
    fn dom(&self) -> Poly<Sort> {
        match self {
            Prop::Const(op) => op.dom().map(Sort::from),
            Prop::And(op) => op.dom().map(Sort::from),
            Prop::Or(op) => op.dom().map(Sort::from),
            Prop::Xor(op) => op.dom().map(Sort::from),
            Prop::Not(op) => op.dom().map(Sort::from),
            Prop::Id(op) => op.dom().map(Sort::from),
            Prop::Ite(op) => op.dom().map(Sort::from),
            Prop::Havoc(op) => op.dom().map(Sort::from),
            Prop::Uninterpreted(op) => op.dom().map(Sort::from),
            Prop::Zero(op) => op.dom(),
        }
    }

    fn cod(&self) -> Poly<Sort> {
        match self {
            Prop::Const(op) => op.cod().map(Sort::from),
            Prop::And(op) => op.cod().map(Sort::from),
            Prop::Or(op) => op.cod().map(Sort::from),
            Prop::Xor(op) => op.cod().map(Sort::from),
            Prop::Not(op) => op.cod().map(Sort::from),
            Prop::Id(op) => op.cod().map(Sort::from),
            Prop::Ite(op) => op.cod().map(Sort::from),
            Prop::Havoc(op) => op.cod().map(Sort::from),
            Prop::Uninterpreted(op) => op.cod().map(Sort::from),
            Prop::Zero(op) => op.cod(),
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
        mono![Bool::new([self.0, self.1]); 2].into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool::new([self.0, self.1])].into()
    }
}

impl Boundary<Bool> for Or {
    fn dom(&self) -> Poly<Bool> {
        mono![Bool::new([self.0, self.1]); 2].into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool::new([self.0, self.1])].into()
    }
}

impl Boundary<Bool> for Xor {
    fn dom(&self) -> Poly<Bool> {
        mono![Bool::new([self.0, self.1]); 2].into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool::new([self.0, self.1])].into()
    }
}

impl Boundary<Bool> for Not {
    fn dom(&self) -> Poly<Bool> {
        mono![Bool::new([self.0, self.1])].into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool::new([self.0, self.1])].into()
    }
}

impl Boundary<Bool> for Id {
    fn dom(&self) -> Poly<Bool> {
        mono![Bool::new([self.0, self.1])].into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool::new([self.0, self.1])].into()
    }
}

/// Takes a boolean of shape `[1, 1]` and two matrices of shape `[m, n]` to
/// a matrix of shape `[m, n]`.
impl Boundary<Bool> for Ite {
    fn dom(&self) -> Poly<Bool> {
        let x = Bool::new([self.0, self.1]);
        mono![Bool::new([1, 1]), x, x].into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool::new([self.0, self.1])].into()
    }
}

impl Boundary<Bool> for Havoc {
    fn dom(&self) -> Poly<Bool> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool::new([self.0, self.1])].into()
    }
}

impl Boundary<Bool> for Uninterpreted {
    fn dom(&self) -> Poly<Bool> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool::new([self.1, self.2])].into()
    }
}

impl Boundary<Sort> for Zero {
    fn dom(&self) -> Poly<Sort> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![crate::sort::Zero.into()].into()
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

impl fmt::Display for Id {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Id")
    }
}

impl fmt::Display for Ite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ite")
    }
}

impl fmt::Display for Havoc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Havoc")
    }
}

impl fmt::Display for Uninterpreted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Uninterpreted({})", self.0)
    }
}

impl fmt::Display for Zero {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Zero")
    }
}

/// Extracts an operation of propositional logic from a Python object,
/// trying each operation in turn: those carrying no data through their own
/// classes, and [`Const`] through the extraction that the tensor type `T`
/// provides.
#[cfg(feature = "pyo3")]
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
        if let Ok(op) = ob.extract::<Id>() {
            return Ok(Prop::Id(op));
        }
        if let Ok(op) = ob.extract::<Ite>() {
            return Ok(Prop::Ite(op));
        }
        if let Ok(op) = ob.extract::<Havoc>() {
            return Ok(Prop::Havoc(op));
        }
        if let Ok(op) = ob.extract::<Uninterpreted>() {
            return Ok(Prop::Uninterpreted(op));
        }
        if let Ok(op) = ob.extract::<Zero>() {
            return Ok(Prop::Zero(op));
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

/// Converts an operation of propositional logic into a Python object:
/// those carrying no data into their own classes, and [`Const`] through
/// the conversion that the tensor type `T` provides.
#[cfg(feature = "pyo3")]
impl<'py, T> pyo3::IntoPyObject<'py> for Prop<T>
where
    Const<T>: pyo3::IntoPyObject<'py>,
{
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, pyo3::PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> pyo3::PyResult<Self::Output> {
        use pyo3::IntoPyObjectExt;

        match self {
            Prop::Const(op) => op.into_bound_py_any(py),
            Prop::And(op) => op.into_bound_py_any(py),
            Prop::Or(op) => op.into_bound_py_any(py),
            Prop::Xor(op) => op.into_bound_py_any(py),
            Prop::Not(op) => op.into_bound_py_any(py),
            Prop::Id(op) => op.into_bound_py_any(py),
            Prop::Ite(op) => op.into_bound_py_any(py),
            Prop::Havoc(op) => op.into_bound_py_any(py),
            Prop::Uninterpreted(op) => op.into_bound_py_any(py),
            Prop::Zero(op) => op.into_bound_py_any(py),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(s: Bool) -> Poly<Bool> {
        mono![s].into()
    }

    fn two(s: Bool) -> Poly<Bool> {
        mono![s, s].into()
    }

    #[test]
    fn pointwise_boundaries() {
        let b = Bool::new([2, 3]);
        assert_eq!(And(2, 3).dom(), two(b));
        assert_eq!(And(2, 3).cod(), one(b));
        assert_eq!(Or(2, 3).dom(), two(b));
        assert_eq!(Xor(2, 3).cod(), one(b));
        assert_eq!(Not(2, 3).dom(), one(b));
        assert_eq!(Not(2, 3).cod(), one(b));
    }

    #[test]
    fn id_keeps_sort() {
        let x = Bool::new([2, 3]);
        assert_eq!(Id(2, 3).dom(), one(x));
        assert_eq!(Id(2, 3).cod(), one(x));
    }

    #[test]
    fn ite_reads_a_single_boolean() {
        let x = Bool::new([2, 3]);
        let dom: Poly<Bool> = mono![Bool::new([1, 1]), x, x].into();
        assert_eq!(Ite(2, 3).dom(), dom);
        assert_eq!(Ite(2, 3).cod(), one(x));
    }

    #[test]
    fn havoc_reads_nothing() {
        assert_eq!(Havoc(2, 3).dom(), Poly::unit());
        assert_eq!(Havoc(2, 3).cod(), one(Bool::new([2, 3])));
    }

    #[test]
    fn uninterpreted_reads_nothing() {
        let u = Uninterpreted("f".to_string(), 2, 3);
        assert_eq!(u.dom(), Poly::unit());
        assert_eq!(u.cod(), one(Bool::new([2, 3])));
    }

    #[test]
    fn zero_writes_the_only_value_of_zero() {
        assert_eq!(Zero.dom(), Poly::unit());
        let cod: Poly<Sort> = mono![crate::sort::Zero.into()].into();
        assert_eq!(Zero.cod(), cod);
    }

    #[test]
    fn ops_display_their_name() {
        assert_eq!(And(1, 1).to_string(), "And");
        assert_eq!(Not(1, 1).to_string(), "Not");
        assert_eq!(Id(1, 1).to_string(), "Id");
        assert_eq!(Ite(1, 1).to_string(), "Ite");
        assert_eq!(Havoc(1, 1).to_string(), "Havoc");
        assert_eq!(
            Uninterpreted("f".to_string(), 1, 1).to_string(),
            "Uninterpreted(f)"
        );
        assert_eq!(Zero.to_string(), "Zero");
    }
}
