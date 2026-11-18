/*!
# Integer arithmetic

Defines one type per operation on the sort [`Int`] of integer matrices. Every
operation reads and writes integer matrices, and carries enough
information to compute its [`Boundary`]:

- [`Const`] — a constant integer matrix literal.
- [`Add`], [`Sub`] — pointwise addition and subtraction.
- [`Mul`] — pointwise multiplication `x ⊙ y`.
- [`Div`], [`Mod`] — pointwise Euclidean division and remainder, as
  SMT-LIB2's `div` and `mod`.
- [`Linear`]`(A, B)` — the affine map `x ↦ A·x + B`, with `A` and `B`
  constant integer matrices of compatible shapes.
- [`ReLU`] — the shape-preserving rectified-linear map `x ↦ max(x, 0)`.
- [`Min`], [`Max`] — pointwise minimum and maximum.
- [`Transpose`] — matrix transposition.
- [`MatMul`] — the matrix product `x·y`.
- [`Lt`], [`Le`], [`Gt`], [`Ge`], [`Eq`](struct@Eq), [`Ne`] — pointwise
  comparisons, which write boolean matrices of the same shape.

The comparisons are those of SMT-LIB2's `Ints` theory, `<`, `<=`, `>` and
`>=`, together with `=` and `distinct` of the Core theory, lifted pointwise
to matrices. As they write booleans, their boundary is in [`Sort`], the
sorts of the theory, rather than in [`Int`].
*/

use super::sort::{Bool, Int};
use crate::Tensor;
use base::boundary::{Boundary, Poly};
use base::mono;
use derive_more::From;
use std::fmt;
use subenum::subenum;

/// The sorts of the theory, as in the signature of SMT-LIB2's `Ints`:
/// int matrices, and the booleans of the Core theory, which comparisons
/// write, together with [`Zero`](crate::tensor::sort::Zero). It is the subset
/// [`tensor::sort::Ints`](crate::tensor::sort::Ints) of the sorts of all theories.
pub use super::sort::Ints as Sort;

/// A constant integer matrix literal, held in a tensor of type `T`.
#[derive(Clone, Debug)]
pub struct Const<T>(pub T);

/// Pointwise addition `x + y` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct Add(pub usize, pub usize);

/// Pointwise subtraction `x - y` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct Sub(pub usize, pub usize);

/// Pointwise multiplication `x ⊙ y` of `[m, n]` matrices: the Hadamard
/// product, not the matrix product [`MatMul`].
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct Mul(pub usize, pub usize);

/// Pointwise Euclidean division `x div y` of `[m, n]` matrices: the
/// quotient `q` such that `x = q·y + r` with `0 ≤ r < |y|`. It rounds down
/// for positive `y` and up for negative `y`, unlike Rust's `/`, which
/// rounds towards zero. As in SMT-LIB2, `x div 0` is unspecified.
///
/// SMT-LIB2: `(div x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct Div(pub usize, pub usize);

/// Pointwise Euclidean remainder `x mod y` of `[m, n]` matrices: the `r`
/// such that `x = (x div y)·y + r` with `0 ≤ r < |y|`, so never negative,
/// unlike Rust's `%`. As in SMT-LIB2, `x mod 0` is unspecified.
///
/// SMT-LIB2: `(mod x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct Mod(pub usize, pub usize);

/// The affine map `x ↦ A·x + B`, where `A` and `B` are constants.
#[derive(Clone, Debug)]
pub struct Linear<T>(pub T, pub T);

/// The rectified-linear map `x ↦ max(x, 0)`, applied pointwise to
/// `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct ReLU(pub usize, pub usize);

/// Pointwise minimum `min(x, y)` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct Min(pub usize, pub usize);

/// Pointwise maximum `max(x, y)` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct Max(pub usize, pub usize);

/// Transposition `xᵀ` of `[m, n]` matrices.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct Transpose(pub usize, pub usize);

/// The matrix product `x·y` of an `[m, n]` matrix `x` and an `[n, k]`
/// matrix `y`, with the sizes given as `MatMul(m, n, k)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct MatMul(pub usize, pub usize, pub usize);

/// Pointwise comparison `x < y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(< x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct Lt(pub usize, pub usize);

/// Pointwise comparison `x ≤ y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(<= x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct Le(pub usize, pub usize);

/// Pointwise comparison `x > y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(> x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct Gt(pub usize, pub usize);

/// Pointwise comparison `x ≥ y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(>= x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct Ge(pub usize, pub usize);

/// Pointwise comparison `x = y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(= x y)`, of the Core theory.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct Eq(pub usize, pub usize);

/// Pointwise comparison `x ≠ y` of `[m, n]` matrices, writing an `[m, n]`
/// boolean matrix.
///
/// SMT-LIB2: `(distinct x y)`, of the Core theory.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, eq, str))]
pub struct Ne(pub usize, pub usize);

/// The operations of the theory of integer arithmetic, SMT-LIB2's `Ints`,
/// with their boundary in [`Sort`].
///
/// `subenum` defines [`LIA`], the linear ones: all but `Mul`, `Div`, `Mod`
/// and `MatMul`, which multiply or divide two variables. It also generates
/// the casts between them: `From` up to `NIA`, and `TryFrom` down from it.
#[subenum(LIA)]
#[derive(Clone, Debug, From)]
pub enum NIA<T> {
    #[subenum(LIA)]
    Const(Const<T>),
    #[subenum(LIA)]
    Add(Add),
    #[subenum(LIA)]
    Sub(Sub),
    Mul(Mul),
    Div(Div),
    Mod(Mod),
    #[subenum(LIA)]
    Linear(Linear<T>),
    #[subenum(LIA)]
    ReLU(ReLU),
    #[subenum(LIA)]
    Min(Min),
    #[subenum(LIA)]
    Max(Max),
    #[subenum(LIA)]
    Transpose(Transpose),
    MatMul(MatMul),
    #[subenum(LIA)]
    Lt(Lt),
    #[subenum(LIA)]
    Le(Le),
    #[subenum(LIA)]
    Gt(Gt),
    #[subenum(LIA)]
    Ge(Ge),
    #[subenum(LIA)]
    Eq(Eq),
    #[subenum(LIA)]
    Ne(Ne),
}

impl<T: Tensor> Boundary<Sort> for NIA<T> {
    fn dom(&self) -> Poly<Sort> {
        match self {
            NIA::Const(op) => op.dom().map(Sort::from),
            NIA::Add(op) => op.dom().map(Sort::from),
            NIA::Sub(op) => op.dom().map(Sort::from),
            NIA::Mul(op) => op.dom().map(Sort::from),
            NIA::Div(op) => op.dom().map(Sort::from),
            NIA::Mod(op) => op.dom().map(Sort::from),
            NIA::Linear(op) => op.dom().map(Sort::from),
            NIA::ReLU(op) => op.dom().map(Sort::from),
            NIA::Min(op) => op.dom().map(Sort::from),
            NIA::Max(op) => op.dom().map(Sort::from),
            NIA::Transpose(op) => op.dom().map(Sort::from),
            NIA::MatMul(op) => op.dom().map(Sort::from),
            NIA::Lt(op) => op.dom(),
            NIA::Le(op) => op.dom(),
            NIA::Gt(op) => op.dom(),
            NIA::Ge(op) => op.dom(),
            NIA::Eq(op) => op.dom(),
            NIA::Ne(op) => op.dom(),
        }
    }

    fn cod(&self) -> Poly<Sort> {
        match self {
            NIA::Const(op) => op.cod().map(Sort::from),
            NIA::Add(op) => op.cod().map(Sort::from),
            NIA::Sub(op) => op.cod().map(Sort::from),
            NIA::Mul(op) => op.cod().map(Sort::from),
            NIA::Div(op) => op.cod().map(Sort::from),
            NIA::Mod(op) => op.cod().map(Sort::from),
            NIA::Linear(op) => op.cod().map(Sort::from),
            NIA::ReLU(op) => op.cod().map(Sort::from),
            NIA::Min(op) => op.cod().map(Sort::from),
            NIA::Max(op) => op.cod().map(Sort::from),
            NIA::Transpose(op) => op.cod().map(Sort::from),
            NIA::MatMul(op) => op.cod().map(Sort::from),
            NIA::Lt(op) => op.cod(),
            NIA::Le(op) => op.cod(),
            NIA::Gt(op) => op.cod(),
            NIA::Ge(op) => op.cod(),
            NIA::Eq(op) => op.cod(),
            NIA::Ne(op) => op.cod(),
        }
    }
}
impl<T: Tensor> Boundary<Sort> for LIA<T> {
    fn dom(&self) -> Poly<Sort> {
        match self {
            LIA::Const(op) => op.dom().map(Sort::from),
            LIA::Add(op) => op.dom().map(Sort::from),
            LIA::Sub(op) => op.dom().map(Sort::from),
            LIA::Linear(op) => op.dom().map(Sort::from),
            LIA::ReLU(op) => op.dom().map(Sort::from),
            LIA::Min(op) => op.dom().map(Sort::from),
            LIA::Max(op) => op.dom().map(Sort::from),
            LIA::Transpose(op) => op.dom().map(Sort::from),
            LIA::Lt(op) => op.dom(),
            LIA::Le(op) => op.dom(),
            LIA::Gt(op) => op.dom(),
            LIA::Ge(op) => op.dom(),
            LIA::Eq(op) => op.dom(),
            LIA::Ne(op) => op.dom(),
        }
    }

    fn cod(&self) -> Poly<Sort> {
        match self {
            LIA::Const(op) => op.cod().map(Sort::from),
            LIA::Add(op) => op.cod().map(Sort::from),
            LIA::Sub(op) => op.cod().map(Sort::from),
            LIA::Linear(op) => op.cod().map(Sort::from),
            LIA::ReLU(op) => op.cod().map(Sort::from),
            LIA::Min(op) => op.cod().map(Sort::from),
            LIA::Max(op) => op.cod().map(Sort::from),
            LIA::Transpose(op) => op.cod().map(Sort::from),
            LIA::Lt(op) => op.cod(),
            LIA::Le(op) => op.cod(),
            LIA::Gt(op) => op.cod(),
            LIA::Ge(op) => op.cod(),
            LIA::Eq(op) => op.cod(),
            LIA::Ne(op) => op.cod(),
        }
    }
}

impl<T: Tensor> Boundary<Int> for Const<T> {
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

impl Boundary<Int> for Mul {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1]); 2].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1])].into()
    }
}

impl Boundary<Int> for Div {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1]); 2].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1])].into()
    }
}

impl Boundary<Int> for Mod {
    fn dom(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1]); 2].into()
    }

    fn cod(&self) -> Poly<Int> {
        mono![Int::new([self.0, self.1])].into()
    }
}

/// With `A` of shape `[m, n]` and `B` of shape `[m, k]`, the map takes `x`
/// of shape `[n, k]` to a matrix of shape `[m, k]`.
impl<T: Tensor> Boundary<Int> for Linear<T> {
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

impl Boundary<Sort> for Lt {
    fn dom(&self) -> Poly<Sort> {
        mono![Int::new([self.0, self.1]).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new([self.0, self.1]).into()].into()
    }
}

impl Boundary<Sort> for Le {
    fn dom(&self) -> Poly<Sort> {
        mono![Int::new([self.0, self.1]).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new([self.0, self.1]).into()].into()
    }
}

impl Boundary<Sort> for Gt {
    fn dom(&self) -> Poly<Sort> {
        mono![Int::new([self.0, self.1]).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new([self.0, self.1]).into()].into()
    }
}

impl Boundary<Sort> for Ge {
    fn dom(&self) -> Poly<Sort> {
        mono![Int::new([self.0, self.1]).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new([self.0, self.1]).into()].into()
    }
}

impl Boundary<Sort> for Eq {
    fn dom(&self) -> Poly<Sort> {
        mono![Int::new([self.0, self.1]).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new([self.0, self.1]).into()].into()
    }
}

impl Boundary<Sort> for Ne {
    fn dom(&self) -> Poly<Sort> {
        mono![Int::new([self.0, self.1]).into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new([self.0, self.1]).into()].into()
    }
}

impl<T: fmt::Display> fmt::Display for Const<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<T: fmt::Display> fmt::Display for Linear<T> {
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

impl fmt::Display for Mul {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Mul")
    }
}

impl fmt::Display for Div {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Div")
    }
}

impl fmt::Display for Mod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Mod")
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

impl fmt::Display for Lt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Lt")
    }
}

impl fmt::Display for Le {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Le")
    }
}

impl fmt::Display for Gt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Gt")
    }
}

impl fmt::Display for Ge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ge")
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

/// Extracts an operation of integer arithmetic from a Python object, trying
/// each operation in turn: those carrying no data through their own
/// classes, and [`Const`] and [`Linear`] through the extraction that the
/// tensor type `T` provides.
#[cfg(feature = "pyo3")]
impl<'py, T> pyo3::FromPyObject<'py> for NIA<T>
where
    Const<T>: pyo3::FromPyObject<'py>,
    Linear<T>: pyo3::FromPyObject<'py>,
{
    fn extract_bound(ob: &pyo3::Bound<'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use pyo3::types::PyAnyMethods;

        if let Ok(op) = ob.extract::<Add>() {
            return Ok(NIA::Add(op));
        }
        if let Ok(op) = ob.extract::<Sub>() {
            return Ok(NIA::Sub(op));
        }
        if let Ok(op) = ob.extract::<Mul>() {
            return Ok(NIA::Mul(op));
        }
        if let Ok(op) = ob.extract::<Div>() {
            return Ok(NIA::Div(op));
        }
        if let Ok(op) = ob.extract::<Mod>() {
            return Ok(NIA::Mod(op));
        }
        if let Ok(op) = ob.extract::<ReLU>() {
            return Ok(NIA::ReLU(op));
        }
        if let Ok(op) = ob.extract::<Min>() {
            return Ok(NIA::Min(op));
        }
        if let Ok(op) = ob.extract::<Max>() {
            return Ok(NIA::Max(op));
        }
        if let Ok(op) = ob.extract::<Transpose>() {
            return Ok(NIA::Transpose(op));
        }
        if let Ok(op) = ob.extract::<MatMul>() {
            return Ok(NIA::MatMul(op));
        }
        if let Ok(op) = ob.extract::<Lt>() {
            return Ok(NIA::Lt(op));
        }
        if let Ok(op) = ob.extract::<Le>() {
            return Ok(NIA::Le(op));
        }
        if let Ok(op) = ob.extract::<Gt>() {
            return Ok(NIA::Gt(op));
        }
        if let Ok(op) = ob.extract::<Ge>() {
            return Ok(NIA::Ge(op));
        }
        if let Ok(op) = ob.extract::<Eq>() {
            return Ok(NIA::Eq(op));
        }
        if let Ok(op) = ob.extract::<Ne>() {
            return Ok(NIA::Ne(op));
        }
        if let Ok(op) = ob.extract::<Const<T>>() {
            return Ok(NIA::Const(op));
        }
        if let Ok(op) = ob.extract::<Linear<T>>() {
            return Ok(NIA::Linear(op));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "expected an operation of integer arithmetic, got {}",
            ob.get_type()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(s: Int) -> Poly<Int> {
        mono![s].into()
    }

    fn two(s: Int) -> Poly<Int> {
        mono![s, s].into()
    }

    #[test]
    fn pointwise_boundaries() {
        let r = Int::new([2, 3]);
        assert_eq!(Add(2, 3).dom(), two(r));
        assert_eq!(Add(2, 3).cod(), one(r));
        assert_eq!(Sub(2, 3).dom(), two(r));
        assert_eq!(Mul(2, 3).dom(), two(r));
        assert_eq!(Mul(2, 3).cod(), one(r));
        assert_eq!(Div(2, 3).dom(), two(r));
        assert_eq!(Div(2, 3).cod(), one(r));
        assert_eq!(Mod(2, 3).dom(), two(r));
        assert_eq!(Mod(2, 3).cod(), one(r));
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
        assert_eq!(Mul(1, 1).to_string(), "Mul");
        assert_eq!(Div(1, 1).to_string(), "Div");
        assert_eq!(Mod(1, 1).to_string(), "Mod");
        assert_eq!(ReLU(1, 1).to_string(), "ReLU");
        assert_eq!(Transpose(1, 1).to_string(), "Transpose");
    }
    #[test]
    fn comparisons_write_booleans() {
        let x: Sort = Int::new([2, 3]).into();
        let b: Sort = Bool::new([2, 3]).into();
        let dom: Poly<Sort> = mono![x, x].into();
        let cod: Poly<Sort> = mono![b].into();
        for (d, c) in [
            (Lt(2, 3).dom(), Lt(2, 3).cod()),
            (Le(2, 3).dom(), Le(2, 3).cod()),
            (Gt(2, 3).dom(), Gt(2, 3).cod()),
            (Ge(2, 3).dom(), Ge(2, 3).cod()),
            (Eq(2, 3).dom(), Eq(2, 3).cod()),
            (Ne(2, 3).dom(), Ne(2, 3).cod()),
        ] {
            assert_eq!(d, dom);
            assert_eq!(c, cod);
        }
        assert_eq!(Le(1, 1).to_string(), "Le");
        assert_eq!(Ne(1, 1).to_string(), "Ne");
    }
}
