/*!
# Real arithmetic

Defines one type per operation on the sort [`Real`] of real matrices. Every operation reads and writes real matrices, and carries enough
information to compute its [`Boundary`]:

- [`Const`] — a constant real matrix literal.
- [`Add`], [`Sub`] — pointwise addition and subtraction.
- [`Linear`]`(A, B)` — the affine map `x ↦ A·x + B`, with `A` and `B`
  constant real matrices of compatible shapes.
- [`ReLU`] — the shape-preserving rectified-linear map `x ↦ max(x, 0)`.
- [`Min`], [`Max`] — pointwise minimum and maximum.
- [`Transpose`] — matrix transposition.
- [`MatMul`] — the matrix product `x·y`.
*/

use crate::sort::Real;
use base::boundary::{Boundary, Poly};
use base::mono;
use std::fmt;

/// A constant real matrix literal.
#[derive(Clone, Debug)]
pub struct Const(pub crate::PyTensor);

/// Pointwise addition `x + y` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Add(pub usize, pub usize);

/// Pointwise subtraction `x - y` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Sub(pub usize, pub usize);

/// The affine map `x ↦ A·x + B`, where `A` and `B` are constants.
#[derive(Clone, Debug)]
pub struct Linear(pub crate::PyTensor, pub crate::PyTensor);

/// The rectified-linear map `x ↦ max(x, 0)`, applied pointwise to
/// `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct ReLU(pub usize, pub usize);

/// Pointwise minimum `min(x, y)` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Min(pub usize, pub usize);

/// Pointwise maximum `max(x, y)` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Max(pub usize, pub usize);

/// Transposition `xᵀ` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Transpose(pub usize, pub usize);

/// The matrix product `x·y` of an `[m, n]` matrix `x` and an `[n, k]`
/// matrix `y`, with the sizes given as `MatMul(m, n, k)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct MatMul(pub usize, pub usize, pub usize);

impl Boundary<Real> for Const {
    fn dom(&self) -> Poly<Real> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new(self.0.shape())].into()
    }
}

impl Boundary<Real> for Add {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1]); 2].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1])].into()
    }
}

impl Boundary<Real> for Sub {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1]); 2].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1])].into()
    }
}

/// With `A` of shape `[m, n]` and `B` of shape `[m, k]`, the map takes `x`
/// of shape `[n, k]` to a matrix of shape `[m, k]`.
impl Boundary<Real> for Linear {
    fn dom(&self) -> Poly<Real> {
        let [_, n] = self.0.shape();
        let [_, k] = self.1.shape();
        mono![Real::new([n, k])].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new(self.1.shape())].into()
    }
}

impl Boundary<Real> for ReLU {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1])].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1])].into()
    }
}

impl Boundary<Real> for Min {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1]); 2].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1])].into()
    }
}

impl Boundary<Real> for Max {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1]); 2].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1])].into()
    }
}

/// Takes a matrix of shape `[m, n]` to one of shape `[n, m]`.
impl Boundary<Real> for Transpose {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1])].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new([self.1, self.0])].into()
    }
}

/// Takes matrices of shapes `[m, n]` and `[n, k]` to one of shape `[m, k]`.
impl Boundary<Real> for MatMul {
    fn dom(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.1]), Real::new([self.1, self.2])].into()
    }

    fn cod(&self) -> Poly<Real> {
        mono![Real::new([self.0, self.2])].into()
    }
}

impl fmt::Display for Const {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Display for Linear {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Linear({}, {})", self.0, self.1)
    }
}

impl fmt::Display for Add {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Add")
    }
}

impl fmt::Display for Sub {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sub")
    }
}

impl fmt::Display for ReLU {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ReLU")
    }
}

impl fmt::Display for Min {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Min")
    }
}

impl fmt::Display for Max {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Max")
    }
}

impl fmt::Display for Transpose {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Transpose")
    }
}

impl fmt::Display for MatMul {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MatMul")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(s: Real) -> Poly<Real> {
        mono![s].into()
    }

    fn two(s: Real) -> Poly<Real> {
        mono![s, s].into()
    }

    #[test]
    fn pointwise_boundaries() {
        let r = Real::new([2, 3]);
        assert_eq!(Add(2, 3).dom(), two(r));
        assert_eq!(Add(2, 3).cod(), one(r));
        assert_eq!(Sub(2, 3).dom(), two(r));
        assert_eq!(Min(2, 3).dom(), two(r));
        assert_eq!(Max(2, 3).cod(), one(r));
        assert_eq!(ReLU(2, 3).dom(), one(r));
        assert_eq!(ReLU(2, 3).cod(), one(r));
    }

    #[test]
    fn transpose_swaps_shape() {
        let t = Transpose(2, 3);
        assert_eq!(t.dom(), one(Real::new([2, 3])));
        assert_eq!(t.cod(), one(Real::new([3, 2])));
    }

    #[test]
    fn matmul_contracts_inner_size() {
        let p = MatMul(2, 3, 4);
        let dom: Poly<Real> = mono![Real::new([2, 3]), Real::new([3, 4])].into();
        assert_eq!(p.dom(), dom);
        assert_eq!(p.cod(), one(Real::new([2, 4])));
    }

    #[test]
    fn ops_display_their_name() {
        assert_eq!(Add(1, 1).to_string(), "Add");
        assert_eq!(ReLU(1, 1).to_string(), "ReLU");
        assert_eq!(Transpose(1, 1).to_string(), "Transpose");
    }
}
