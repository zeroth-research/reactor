/*!
# Booleans

Defines one type per operation on the sort [`Bool`] of scalar booleans, which
the comparisons of the scalar theories write, and the conditions of their
choices read: the boolean operations of SMT-LIB2's `Core` theory, as those of
[`crate::matrix::bool`] on boolean matrices:

- [`True`], [`False`] — the constants, reading nothing.
- [`And`], [`Or`], [`Xor`] — conjunction, disjunction and exclusive
  disjunction.
- [`Not`] — negation.
- [`Eq`](struct@Eq), [`Ne`] — equality and disequality.
- [`Ite`] — the choice `if c then x else y`, by a boolean `c`.

The sort has no parameters, so neither do the operations: each is a unit
value, determined by its type alone, as `And` or `True`, and its boundary
is fixed.
*/

use super::sort::Bool;
use base::boundary::{Boundary, Poly};
use base::mono;
use derive_more::From;
use std::fmt;

/// The constant `true`, reading nothing.
///
/// SMT-LIB2: `true`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
pub struct True;

impl Boundary<Bool> for True {
    fn dom(&self) -> Poly<Bool> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool].into()
    }
}

impl fmt::Display for True {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "True")
    }
}

/// The constant `false`, reading nothing.
///
/// SMT-LIB2: `false`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
pub struct False;

impl Boundary<Bool> for False {
    fn dom(&self) -> Poly<Bool> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool].into()
    }
}

impl fmt::Display for False {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "False")
    }
}

/// Conjunction `x ∧ y` of two booleans.
///
/// SMT-LIB2: `(and x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
pub struct And;

impl Boundary<Bool> for And {
    fn dom(&self) -> Poly<Bool> {
        mono![Bool; 2].into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool].into()
    }
}

impl fmt::Display for And {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "And")
    }
}

/// Disjunction `x ∨ y` of two booleans.
///
/// SMT-LIB2: `(or x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
pub struct Or;

impl Boundary<Bool> for Or {
    fn dom(&self) -> Poly<Bool> {
        mono![Bool; 2].into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool].into()
    }
}

impl fmt::Display for Or {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Or")
    }
}

/// Exclusive disjunction `x ⊕ y` of two booleans.
///
/// SMT-LIB2: `(xor x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
pub struct Xor;

impl Boundary<Bool> for Xor {
    fn dom(&self) -> Poly<Bool> {
        mono![Bool; 2].into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool].into()
    }
}

impl fmt::Display for Xor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Xor")
    }
}

/// Negation `¬x` of a boolean.
///
/// SMT-LIB2: `(not x)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
pub struct Not;

impl Boundary<Bool> for Not {
    fn dom(&self) -> Poly<Bool> {
        mono![Bool].into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool].into()
    }
}

impl fmt::Display for Not {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Not")
    }
}

/// Equality `x = y` of two booleans: the negation of their exclusive
/// disjunction.
///
/// SMT-LIB2: `(= x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
pub struct Eq;

impl Boundary<Bool> for Eq {
    fn dom(&self) -> Poly<Bool> {
        mono![Bool; 2].into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool].into()
    }
}

impl fmt::Display for Eq {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Eq")
    }
}

/// Disequality `x ≠ y` of two booleans: their exclusive disjunction.
///
/// SMT-LIB2: `(distinct x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
pub struct Ne;

impl Boundary<Bool> for Ne {
    fn dom(&self) -> Poly<Bool> {
        mono![Bool; 2].into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool].into()
    }
}

impl fmt::Display for Ne {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ne")
    }
}

/// Choice `if c then x else y` between two booleans `x` and `y`, by the
/// boolean `c`, read first.
///
/// SMT-LIB2: `(ite c x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
pub struct Ite;

impl Boundary<Bool> for Ite {
    fn dom(&self) -> Poly<Bool> {
        mono![Bool; 3].into()
    }

    fn cod(&self) -> Poly<Bool> {
        mono![Bool].into()
    }
}

impl fmt::Display for Ite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ite")
    }
}

/// The operations on scalar booleans, those of propositional logic, with their
/// boundary in [`Bool`].
#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash, From)]
pub enum Any {
    True(True),
    False(False),
    And(And),
    Or(Or),
    Xor(Xor),
    Not(Not),
    Eq(Eq),
    Ne(Ne),
    Ite(Ite),
}

impl Boundary<Bool> for Any {
    fn dom(&self) -> Poly<Bool> {
        match self {
            Any::True(op) => op.dom(),
            Any::False(op) => op.dom(),
            Any::And(op) => op.dom(),
            Any::Or(op) => op.dom(),
            Any::Xor(op) => op.dom(),
            Any::Not(op) => op.dom(),
            Any::Eq(op) => op.dom(),
            Any::Ne(op) => op.dom(),
            Any::Ite(op) => op.dom(),
        }
    }

    fn cod(&self) -> Poly<Bool> {
        match self {
            Any::True(op) => op.cod(),
            Any::False(op) => op.cod(),
            Any::And(op) => op.cod(),
            Any::Or(op) => op.cod(),
            Any::Xor(op) => op.cod(),
            Any::Not(op) => op.cod(),
            Any::Eq(op) => op.cod(),
            Any::Ne(op) => op.cod(),
            Any::Ite(op) => op.cod(),
        }
    }
}

impl fmt::Display for Any {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Any::True(op) => op.fmt(f),
            Any::False(op) => op.fmt(f),
            Any::And(op) => op.fmt(f),
            Any::Or(op) => op.fmt(f),
            Any::Xor(op) => op.fmt(f),
            Any::Not(op) => op.fmt(f),
            Any::Eq(op) => op.fmt(f),
            Any::Ne(op) => op.fmt(f),
            Any::Ite(op) => op.fmt(f),
        }
    }
}
