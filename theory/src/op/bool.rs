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

use crate::sort::Bool;
use base::boundary::{Boundary, Poly};
use base::mono;
use std::fmt;

/// A constant boolean matrix literal.
#[derive(Clone, Debug)]
pub struct Const(pub crate::PyTensor);

/// Pointwise conjunction `x ∧ y` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct And(pub usize, pub usize);

/// Pointwise disjunction `x ∨ y` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Or(pub usize, pub usize);

/// Pointwise exclusive disjunction `x ⊕ y` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Xor(pub usize, pub usize);

/// Pointwise negation `¬x` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Not(pub usize, pub usize);

impl Boundary<Bool> for Const {
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

impl fmt::Display for Const {
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
