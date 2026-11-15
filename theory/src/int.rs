/*!
# Integer arithmetic

Defines the sort [`Int`] of integer matrices, whose tangent is the
singleton sort [`Zero`], and one type per operation on them. Every
operation reads and writes integer matrices, and carries enough
information to compute its [`Boundary`]:

- [`Const`] — a constant integer matrix literal.
- [`Add`], [`Sub`] — pointwise addition and subtraction.
- [`Linear`]`(A, B)` — the affine map `x ↦ A·x + B`, with `A` and `B`
  constant integer matrices of compatible shapes.
- [`ReLU`] — the shape-preserving rectified-linear map `x ↦ max(x, 0)`.
- [`Min`], [`Max`] — pointwise minimum and maximum.
- [`Transpose`] — matrix transposition.
- [`MatMul`] — the matrix product `x·y`.
*/

use crate::zero::Zero;
use base::boundary::{Boundary, Poly};
use base::mono;
use base::tangent::Tangent;
use std::fmt;

/// The sort of integer matrices with the given `[rows, cols]` shape.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Int {
    shape: [usize; 2],
}

impl Int {
    pub fn new(shape: [usize; 2]) -> Self {
        Int { shape }
    }

    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl fmt::Display for Int {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [i, j] = self.shape;
        write!(f, "Int({i}, {j})")
    }
}

/// Integers are discrete, so their tangent is trivial: the singleton sort
/// [`Zero`].
impl Tangent<Int> for Zero {
    fn of(_: Int) -> Zero {
        Zero
    }
}

/// A constant integer matrix literal.
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

impl Boundary<Int> for Const {
    fn dom(&self) -> Poly<Int> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new(self.0.shape())].into()
    }
}

impl Boundary<Int> for Add {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1]); 2].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1])].into()
    }
}

impl Boundary<Int> for Sub {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1]); 2].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1])].into()
    }
}

/// With `A` of shape `[m, n]` and `B` of shape `[m, k]`, the map takes `x`
/// of shape `[n, k]` to a matrix of shape `[m, k]`.
impl Boundary<Int> for Linear {
    fn dom(&self) -> Poly<Int> {
        let [_, n] = self.0.shape();
        let [_, k] = self.1.shape();
        mono![Int::new([n, k])].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new(self.1.shape())].into()
    }
}

impl Boundary<Int> for ReLU {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1])].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1])].into()
    }
}

impl Boundary<Int> for Min {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1]); 2].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1])].into()
    }
}

impl Boundary<Int> for Max {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1]); 2].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1])].into()
    }
}

/// Takes a matrix of shape `[m, n]` to one of shape `[n, m]`.
impl Boundary<Int> for Transpose {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1])].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new([self.1, self.0])].into()
    }
}

/// Takes matrices of shapes `[m, n]` and `[n, k]` to one of shape `[m, k]`.
impl Boundary<Int> for MatMul {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1]), Int::new([self.1, self.2])].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.2])].into()
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
    use base::tangent::T;

    fn one(s: Int) -> Poly<Int> {
        mono![s].into()
    }

    fn two(s: Int) -> Poly<Int> {
        mono![s, s].into()
    }

    #[test]
    fn sort_shape_and_display() {
        let r = Int::new([2, 3]);
        assert_eq!(r.shape(), [2, 3]);
        assert_eq!(r.to_string(), "Int(2, 3)");
    }

    #[test]
    fn tangent_is_trivial() {
        let t: Zero = T(Int::new([2, 3]));
        assert_eq!(t, Zero);
        assert_eq!(Zero::of(Int::new([2, 3])), Zero);
    }

    #[test]
    fn pointwise_boundaries() {
        let r = Int::new([2, 3]);
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
        assert_eq!(t.dom(), one(Int::new([2, 3])));
        assert_eq!(t.cod(), one(Int::new([3, 2])));
    }

    #[test]
    fn matmul_contracts_inner_size() {
        let p = MatMul(2, 3, 4);
        let dom: Poly<Int> = mono![Int::new([2, 3]), Int::new([3, 4])].into();
        assert_eq!(p.dom(), dom);
        assert_eq!(p.cod(), one(Int::new([2, 4])));
    }

    #[test]
    fn ops_display_their_name() {
        assert_eq!(Add(1, 1).to_string(), "Add");
        assert_eq!(ReLU(1, 1).to_string(), "ReLU");
        assert_eq!(Transpose(1, 1).to_string(), "Transpose");
    }
}
