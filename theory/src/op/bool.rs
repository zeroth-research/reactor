/*!
# Boolean algebra

Defines one type per operation on the sort [`Bool`] of boolean matrices. Every
operation reads and writes boolean matrices, and carries enough
information to compute its [`Boundary`]:

- [`Const`] — a constant boolean matrix literal.
- [`And`], [`Or`], [`Xor`] — pointwise conjunction, disjunction and
  exclusive disjunction.
- [`Not`] — pointwise negation.
*/

use crate::Tensor;
use crate::sort::Bool;
use base::boundary::{Boundary, Poly};
use base::mono;
use derive_more::From;
use std::fmt;

/// A constant boolean matrix literal, held in a tensor of type `T`.
#[derive(Clone, Debug)]
pub struct Const<T>(pub T);

/// Pointwise conjunction `x ∧ y` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct And(pub usize, pub usize);

/// Pointwise disjunction `x ∨ y` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct Or(pub usize, pub usize);

/// Pointwise exclusive disjunction `x ⊕ y` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct Xor(pub usize, pub usize);

/// Pointwise negation `¬x` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct Not(pub usize, pub usize);

/// The operations of propositional logic, the boolean operations of
/// SMT-LIB2's `Core` theory, with their boundary in [`Bool`].
#[derive(Clone, Debug, From)]
pub enum Prop<T> {
    Const(Const<T>),
    And(And),
    Or(Or),
    Xor(Xor),
    Not(Not),
}

impl<T: Tensor> Boundary<Bool> for Prop<T> {
    fn dom(&self) -> Poly<Bool> {
        match self {
            Prop::Const(op) => op.dom(),
            Prop::And(op) => op.dom(),
            Prop::Or(op) => op.dom(),
            Prop::Xor(op) => op.dom(),
            Prop::Not(op) => op.dom(),
        }
    }

    fn cod(&self) -> Poly<Bool> {
        match self {
            Prop::Const(op) => op.cod(),
            Prop::And(op) => op.cod(),
            Prop::Or(op) => op.cod(),
            Prop::Xor(op) => op.cod(),
            Prop::Not(op) => op.cod(),
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
        if let Ok(op) = ob.extract::<Const<T>>() {
            return Ok(Prop::Const(op));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "expected an operation of propositional logic, got {}",
            ob.get_type()
        )))
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
    fn ops_display_their_name() {
        assert_eq!(And(1, 1).to_string(), "And");
        assert_eq!(Not(1, 1).to_string(), "Not");
    }
}
