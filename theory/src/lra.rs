/*!
# Linear real arithmetic

Defines the theory [`LRA`] of linear real arithmetic over matrices,
mixing real and boolean matrices in a single signature.

A [`Sort`] value is either a real matrix `Real { shape, rank }` or a boolean
matrix `Bool { shape }`, or `Zero { shape }`, the sort of the derivatives of
booleans (see [`Tangent::tangent`]). The `rank` of a real is its differential grade:
`0` for values, `d >= 1` for their `d`-th derivatives.

The operations carry their parameters (the shapes of the matrices, and the ranks
of the derivatives), so that each operation has a fixed [`Boundary`]: e.g., the
boundary of `Add(2, 3)` is `Real(2, 3) ⊗ Real(2, 3) -> Real(2, 3)`. The operations
that apply to several kinds of values come in one variant per kind, e.g.,
[`LRA::IdReal`] and [`LRA::IdBool`]. The operations in [`LRA`] are:

- [`LRA::Real`], [`LRA::Bool`] — real and boolean matrix literals.
- [`LRA::And`], [`LRA::Or`], [`LRA::Xor`], [`LRA::Not`]
  — boolean operations.
- [`LRA::Le`], [`LRA::Lt`], [`LRA::Ge`], [`LRA::Gt`], [`LRA::Eq`], [`LRA::Ne`]
  — pointwise real comparisons producing a boolean matrix.
- [`LRA::IteReal`], [`LRA::IteBool`] — if-then-else: reads a boolean guard and
  two same-typed branches.
- [`LRA::Linear`]`(A, B, batch)` — the affine map `x ↦ A·x + B`, with `A` and
  `B` constant real matrices of compatible shapes, of `x` with `batch` columns.
- [`LRA::ReLU`] — the shape-preserving rectified-linear map on real matrices.
- [`LRA::TAdd`], [`LRA::TSub`], [`LRA::TLinear`], [`LRA::TIte`], [`LRA::TId`]
  — the operations on the derivatives of reals (of the rank `>= 1`).
- [`LRA::RealZerograd`], [`LRA::BoolZerograd`] — the zero derivatives.

`LRA` implements [`Theory`]; [`Theory::check`] checks that the sorts of the
read and written wires are the domain and the codomain of the operation.

## Examples

```
use theory::Theory;
use theory::lra::{LRA, Sort};
use base::tangent::Tangent;

// Wires carry a sort (derivatives carry the tangent sorts).
let ok = |s| Ok::<_, String>(s);

// Pointwise less-than on scalars: Real(1,1), Real(1,1) -> Bool(1,1).
let i = Sort::real([1, 1]);
let b = Sort::Bool { shape: [1, 1] };
assert!(LRA::Lt(1, 1).check([i, i].map(ok), [b].map(ok)).is_ok());

// ReLU preserves shape and stays in the real fragment.
let m = Sort::real([3, 4]);
assert!(LRA::ReLU(3, 4).check([m].map(ok), [m].map(ok)).is_ok());
assert!(LRA::ReLU(3, 4).check([b].map(ok), [b].map(ok)).is_err());

// Derivatives have their own operations.
let dm = m.tangent();
assert!(LRA::TAdd(3, 4, 1).check([dm, dm].map(ok), [dm].map(ok)).is_ok());
assert!(LRA::Add(3, 4).check([dm, dm].map(ok), [dm].map(ok)).is_err());
```
*/

use crate::*;
use base::boundary::{Boundary, Poly};
use base::tangent::Tangent;
#[cfg(feature = "pyo3")]
use pyo3::pyclass;
use std::fmt;

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub enum Sort {
    // a real matrix; `rank` is the differential grade: `0` for values,
    // `d >= 1` for their `d`-th derivatives
    Real { shape: [usize; 2], rank: u8 },
    Bool { shape: [usize; 2] },
    // the tangent of `Bool` (it has only the zero derivative)
    Zero { shape: [usize; 2] },
}

impl Sort {
    /// The sort of real values (of rank `0`) of the given shape
    pub fn real(shape: [usize; 2]) -> Self {
        Sort::Real { shape, rank: 0 }
    }

    pub fn is_bool(&self) -> bool {
        matches!(self, Sort::Bool { .. })
    }

    pub fn is_real(&self) -> bool {
        matches!(self, Sort::Real { .. })
    }

    /// Is this sort a tangent (i.e., a sort of derivatives)?
    pub fn is_tangent(&self) -> bool {
        matches!(self, Sort::Real { rank: 1.., .. } | Sort::Zero { .. })
    }

    pub fn shape(&self) -> &[usize; 2] {
        match self {
            Sort::Bool { shape } | Sort::Real { shape, .. } | Sort::Zero { shape } => shape,
        }
    }
}

impl fmt::Display for Sort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sort::Real {
                shape: [i, j],
                rank: 0,
            } => write!(f, "Real({i}, {j})"),
            Sort::Real {
                shape: [i, j],
                rank,
            } => write!(f, "TReal({i}, {j}; {rank})"),
            Sort::Bool { shape: [i, j] } => write!(f, "Bool({i}, {j})"),
            Sort::Zero { shape: [i, j] } => write!(f, "Zero({i}, {j})"),
        }
    }
}

impl Tangent for Sort {
    fn tangent(&self) -> Self {
        match self {
            Sort::Real { shape, rank } => Sort::Real {
                shape: *shape,
                rank: rank + 1,
            },
            Sort::Bool { shape } | Sort::Zero { shape } => Sort::Zero { shape: *shape },
        }
    }
}

/// The operations of linear real arithmetic. The parameters `(rows, cols)` are the shape
/// of the matrices the operation applies to (e.g., of its operands), and `rank` is the rank
/// of the derivatives (at least `1`).
#[derive(Clone, Debug, strum::IntoStaticStr)]
#[cfg_attr(feature = "pyo3", pyclass(frozen))]
pub enum LRA {
    // real and boolean matrix literals (of the shape of the tensor)
    Real(crate::PyTensor),
    Bool(crate::PyTensor),
    // boolean operations
    And(usize, usize),
    Or(usize, usize),
    Xor(usize, usize),
    Not(usize, usize),
    // real comparisons
    Le(usize, usize),
    Lt(usize, usize),
    Ge(usize, usize),
    Gt(usize, usize),
    Eq(usize, usize),
    Ne(usize, usize),
    // linear / matrix operations
    // A*x + B where `A` and `B` are constants, and `x` has `batch` columns
    Linear(crate::PyTensor, crate::PyTensor, usize),
    Add(usize, usize),
    Sub(usize, usize),
    // XXX: should these be in RLA?
    ReLU(usize, usize),
    Argmax(usize, usize),
    Min(usize, usize),
    Max(usize, usize),
    // matrix operations
    TransposeReal(usize, usize),
    TransposeBool(usize, usize),
    // control flow
    IteReal(usize, usize),
    IteBool(usize, usize),
    IdReal(usize, usize),
    IdBool(usize, usize),
    // an uninterpreted value (written, used by the builder for untraced calls)
    UninterpretedReal(String, usize, usize),
    UninterpretedBool(String, usize, usize),
    AnyReal(usize, usize),
    AnyBool(usize, usize),
    // operations on derivatives of reals of the rank `rank`
    TAdd(usize, usize, u8),
    TSub(usize, usize, u8),
    TLinear(crate::PyTensor, crate::PyTensor, usize, u8),
    TIte(usize, usize, u8),
    TId(usize, usize, u8),
    // the zero derivatives
    RealZerograd(usize, usize, u8), // unstable
    BoolZerograd(usize, usize),     // unstable
    // discarding a value (read, nothing written): of a real, a boolean, a derivative of a
    // real of the rank `rank`, and a zero derivative of a boolean
    DiscardReal(usize, usize),
    DiscardBool(usize, usize),
    TDiscard(usize, usize, u8),
    DiscardZero(usize, usize),
}

/// Prints the name of the operation, the tensors of the literals and of the affine maps,
/// and the names of the uninterpreted values (the other parameters are given by the wires)
impl fmt::Display for LRA {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name: &'static str = self.into();
        match self {
            LRA::Real(t) | LRA::Bool(t) => write!(f, "{t}"),
            LRA::Linear(a, b, ..) | LRA::TLinear(a, b, ..) => write!(f, "{name}({a}, {b})"),
            LRA::UninterpretedReal(x, ..) | LRA::UninterpretedBool(x, ..) => {
                write!(f, "{name}({x})")
            }
            _ => f.write_str(name),
        }
    }
}

impl Sequential for LRA {
    fn skip(range: &Self::Sort) -> Self {
        match *range {
            Sort::Real {
                shape: [r, c],
                rank: 0,
            } => LRA::IdReal(r, c),
            Sort::Real {
                shape: [r, c],
                rank,
            } => LRA::TId(r, c, rank),
            // a zero derivative cannot be copied: the check of the term with it fails
            Sort::Bool { shape: [r, c] } | Sort::Zero { shape: [r, c] } => LRA::IdBool(r, c),
        }
    }
}

impl Combinatorial for LRA {
    fn havoc(range: &Self::Sort) -> Self {
        // a derivative cannot be havoced: the check of the term with a tangent fails
        match *range {
            Sort::Bool { shape: [r, c] } | Sort::Zero { shape: [r, c] } => LRA::AnyBool(r, c),
            Sort::Real { shape: [r, c], .. } => LRA::AnyReal(r, c),
        }
    }
}

impl Differential for LRA {
    fn zero(range: &Self::Sort) -> Self {
        // the range is the sort of the derivative; values have no zero derivative,
        // so the check of the term with them fails
        match *range {
            Sort::Bool { shape: [r, c] } | Sort::Zero { shape: [r, c] } => LRA::BoolZerograd(r, c),
            Sort::Real {
                shape: [r, c],
                rank,
            } => LRA::RealZerograd(r, c, rank),
        }
    }
}

impl Theory for LRA {
    type Sort = Sort;
    const NAME: &'static str = "LRA";

    fn check<R, W, E: fmt::Display>(&self, read: R, write: W) -> Result<(), String>
    where
        R: IntoIterator<Item = Result<Sort, E>>,
        W: IntoIterator<Item = Result<Sort, E>>,
    {
        try_dom(self)
            .and_then(|dom| check_wires(dom, try_cod(self)?, read, write))
            .map_err(|e| format!("{self:?}: {e}"))
    }

    fn discard(domain: &Sort) -> Self {
        match *domain {
            Sort::Real {
                shape: [r, c],
                rank: 0,
            } => LRA::DiscardReal(r, c),
            Sort::Real {
                shape: [r, c],
                rank,
            } => LRA::TDiscard(r, c, rank),
            Sort::Bool { shape: [r, c] } => LRA::DiscardBool(r, c),
            Sort::Zero { shape: [r, c] } => LRA::DiscardZero(r, c),
        }
    }
}

impl Boundary<Sort> for LRA {
    /// # Panics
    ///
    /// If `A` of [`LRA::Linear`] or [`LRA::TLinear`] is not a 2D tensor, or the rank of a
    /// derivative is `0` ([`Theory::check`] reports these as errors).
    fn dom(&self) -> Poly<Sort> {
        try_dom(self).unwrap_or_else(|e| panic!("{self:?}: {e}"))
    }

    /// # Panics
    ///
    /// If the tensor of a literal or `A` of [`LRA::Linear`] or [`LRA::TLinear`] is not 2D,
    /// `B` of these is not a column vector, or the rank of a derivative is `0`
    /// ([`Theory::check`] reports these as errors).
    fn cod(&self) -> Poly<Sort> {
        try_cod(self).unwrap_or_else(|e| panic!("{self:?}: {e}"))
    }
}

/// The sort of the derivatives of the rank `rank` of the reals of the shape `[r, c]`, or an
/// error if `rank` is `0`
fn treal(r: usize, c: usize, rank: u8) -> Result<Sort, String> {
    match rank {
        0 => Err("the rank of a derivative must be at least 1".to_string()),
        rank => Ok(Sort::Real {
            shape: [r, c],
            rank,
        }),
    }
}

/// The domain of `op`: the sorts of its inputs, or an error if `op` is ill-formed
fn try_dom(op: &LRA) -> Result<Poly<Sort>, String> {
    let real = |r, c| Sort::real([r, c]);
    let bool = |r, c| Sort::Bool { shape: [r, c] };
    Ok(match *op {
        LRA::Real(_) | LRA::Bool(_) => Poly::unit(),
        LRA::And(r, c) | LRA::Or(r, c) | LRA::Xor(r, c) => prod([bool(r, c), bool(r, c)]),
        LRA::Not(r, c) => prod([bool(r, c)]),
        LRA::Le(r, c)
        | LRA::Lt(r, c)
        | LRA::Ge(r, c)
        | LRA::Gt(r, c)
        | LRA::Eq(r, c)
        | LRA::Ne(r, c) => prod([real(r, c), real(r, c)]),
        // Y = A·X + B  where X=[in,batch], A=[out,in], B=[out,1] (or empty), Y=[out,batch]
        LRA::Linear(ref a, _, batch) => prod([real(linear_inputs(a)?, batch)]),
        // `Min` and `Max` are pointwise, as in the evaluator
        LRA::Add(r, c) | LRA::Sub(r, c) | LRA::Min(r, c) | LRA::Max(r, c) => {
            prod([real(r, c), real(r, c)])
        }
        LRA::ReLU(r, c) | LRA::Argmax(r, c) | LRA::TransposeReal(r, c) => prod([real(r, c)]),
        LRA::TransposeBool(r, c) => prod([bool(r, c)]),
        LRA::IteReal(r, c) => prod([bool(1, 1), real(r, c), real(r, c)]),
        LRA::IteBool(r, c) => prod([bool(1, 1), bool(r, c), bool(r, c)]),
        LRA::IdReal(r, c) => prod([real(r, c)]),
        LRA::IdBool(r, c) => prod([bool(r, c)]),
        LRA::UninterpretedReal(..)
        | LRA::AnyReal(..)
        | LRA::UninterpretedBool(..)
        | LRA::AnyBool(..) => Poly::unit(),
        LRA::TAdd(r, c, rank) | LRA::TSub(r, c, rank) => {
            let t = treal(r, c, rank)?;
            prod([t, t])
        }
        LRA::TLinear(ref a, _, batch, rank) => prod([treal(linear_inputs(a)?, batch, rank)?]),
        LRA::TIte(r, c, rank) => {
            let t = treal(r, c, rank)?;
            prod([bool(1, 1), t, t])
        }
        LRA::TId(r, c, rank) => prod([treal(r, c, rank)?]),
        LRA::RealZerograd(..) | LRA::BoolZerograd(..) => Poly::unit(),
        LRA::DiscardReal(r, c) => prod([real(r, c)]),
        LRA::DiscardBool(r, c) => prod([bool(r, c)]),
        LRA::TDiscard(r, c, rank) => prod([treal(r, c, rank)?]),
        LRA::DiscardZero(r, c) => prod([bool(r, c).tangent()]),
    })
}

/// The codomain of `op`: the sort of its output, or an error if `op` is ill-formed
fn try_cod(op: &LRA) -> Result<Poly<Sort>, String> {
    let real = |r, c| Sort::real([r, c]);
    let bool = |r, c| Sort::Bool { shape: [r, c] };
    let sort = match *op {
        LRA::Real(ref cm) => {
            let [r, c] = shape_2d(cm)?;
            real(r, c)
        }
        LRA::Bool(ref cm) => {
            let [r, c] = shape_2d(cm)?;
            bool(r, c)
        }
        LRA::And(r, c) | LRA::Or(r, c) | LRA::Xor(r, c) | LRA::Not(r, c) => bool(r, c),
        LRA::Le(r, c)
        | LRA::Lt(r, c)
        | LRA::Ge(r, c)
        | LRA::Gt(r, c)
        | LRA::Eq(r, c)
        | LRA::Ne(r, c) => bool(r, c),
        // Y = A·X + B  where X=[in,batch], A=[out,in], B=[out,1] (or empty), Y=[out,batch]
        LRA::Linear(ref a, ref b, batch) => real(linear_outputs(a, b)?, batch),
        // `Min` and `Max` are pointwise, as in the evaluator
        LRA::Add(r, c) | LRA::Sub(r, c) | LRA::Min(r, c) | LRA::Max(r, c) | LRA::ReLU(r, c) => {
            real(r, c)
        }
        // the index into the flattened matrix, as in the evaluator and the Python builder
        LRA::Argmax(..) => real(1, 1),
        LRA::TransposeReal(r, c) => real(c, r),
        LRA::TransposeBool(r, c) => bool(c, r),
        LRA::IteReal(r, c) | LRA::IdReal(r, c) => real(r, c),
        LRA::IteBool(r, c) | LRA::IdBool(r, c) => bool(r, c),
        LRA::UninterpretedReal(_, r, c) | LRA::AnyReal(r, c) => real(r, c),
        LRA::UninterpretedBool(_, r, c) | LRA::AnyBool(r, c) => bool(r, c),
        LRA::TAdd(r, c, rank)
        | LRA::TSub(r, c, rank)
        | LRA::TIte(r, c, rank)
        | LRA::TId(r, c, rank)
        | LRA::RealZerograd(r, c, rank) => treal(r, c, rank)?,
        LRA::TLinear(ref a, ref b, batch, rank) => treal(linear_outputs(a, b)?, batch, rank)?,
        LRA::BoolZerograd(r, c) => bool(r, c).tangent(),
        // nothing is written
        LRA::DiscardReal(..) | LRA::DiscardBool(..) | LRA::TDiscard(..) | LRA::DiscardZero(..) => {
            return Ok(Poly::unit());
        }
    };
    Ok(prod([sort]))
}

#[cfg(all(test, feature = "torch"))]
mod tests {
    use super::*;

    fn real(r: usize, c: usize) -> Sort {
        Sort::real([r, c])
    }

    fn bool_t(r: usize, c: usize) -> Sort {
        Sort::Bool { shape: [r, c] }
    }

    fn ok(s: Sort) -> Result<Sort, String> {
        Ok(s)
    }

    fn real_tensor(rows: i64, cols: i64) -> crate::PyTensor {
        tch::Tensor::zeros([rows, cols], (tch::Kind::Float, tch::Device::Cpu)).into()
    }

    #[test]
    fn type_kind_and_shape() {
        assert!(real(2, 3).is_real() && !real(2, 3).is_bool());
        assert_eq!(real(2, 3).shape(), &[2, 3]);
        assert!(bool_t(1, 1).is_bool() && !bool_t(1, 1).is_real());
    }

    #[test]
    fn const_real_ok() {
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[0.0f64, 1.0], [2.0, 3.0]]).into();
        assert!(
            LRA::Real(cm)
                .check([].map(ok), [real(2, 2)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn const_real_covector_write_fails() {
        // a real constant writes a value (0-form), never a derivative:
        // a covector (1-form) write wire must be rejected
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[0.0f64]]).into();
        assert!(
            LRA::Real(cm)
                .check([].map(ok), [real(1, 1).tangent()].map(ok))
                .is_err()
        );
    }

    #[test]
    fn const_real_bool_write_fails() {
        assert!(
            LRA::Real(tch::Tensor::from_slice2(&[[0.0f64]]).into())
                .check([].map(ok), [bool_t(1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn const_real_wrong_rows_fails() {
        assert!(
            LRA::Real(tch::Tensor::from_slice2(&[[0.0f64]]).into())
                .check([].map(ok), [real(2, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn const_real_with_read_fails() {
        let t = real(1, 1);
        assert!(
            LRA::Real(tch::Tensor::from_slice2(&[[0.0f64]]).into())
                .check([t].map(ok), [t].map(ok))
                .is_err()
        );
    }

    #[test]
    fn const_bool_ok() {
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[true, false], [false, true]]).into();
        assert!(
            LRA::Bool(cm)
                .check([].map(ok), [bool_t(2, 2)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn const_bool_real_write_fails() {
        assert!(
            LRA::Bool(tch::Tensor::from_slice2(&[[true]]).into())
                .check([].map(ok), [real(1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn not_ok() {
        let b = bool_t(2, 3);
        assert!(LRA::Not(2, 3).check([b].map(ok), [b].map(ok)).is_ok());
        // the shape is given by the operation
        assert!(LRA::Not(3, 2).check([b].map(ok), [b].map(ok)).is_err());
    }

    #[test]
    fn not_real_input_fails() {
        let t = real(1, 1);
        assert!(LRA::Not(1, 1).check([t].map(ok), [t].map(ok)).is_err());
    }

    #[test]
    fn and_ok() {
        let b = bool_t(2, 2);
        assert!(LRA::And(2, 2).check([b, b].map(ok), [b].map(ok)).is_ok());
    }

    #[test]
    fn or_ok() {
        let b = bool_t(1, 1);
        assert!(LRA::Or(1, 1).check([b, b].map(ok), [b].map(ok)).is_ok());
    }

    #[test]
    fn xor_ok() {
        let b = bool_t(3, 1);
        assert!(LRA::Xor(3, 1).check([b, b].map(ok), [b].map(ok)).is_ok());
    }

    #[test]
    fn and_real_output_fails() {
        let b = bool_t(1, 1);
        assert!(
            LRA::And(1, 1)
                .check([b, b].map(ok), [real(1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn and_type_mismatch_fails() {
        assert!(
            LRA::And(1, 1)
                .check([bool_t(1, 1), bool_t(1, 2)].map(ok), [bool_t(1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn lt_ok() {
        assert!(
            LRA::Lt(1, 1)
                .check([real(1, 1), real(1, 1)].map(ok), [bool_t(1, 1)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn le_ok() {
        assert!(
            LRA::Le(2, 3)
                .check([real(2, 3), real(2, 3)].map(ok), [bool_t(2, 3)].map(ok))
                .is_ok()
        );

        assert!(
            LRA::Le(2, 3)
                .check([real(3, 3), real(2, 3)].map(ok), [bool_t(2, 3)].map(ok))
                .is_err()
        );

        assert!(
            LRA::Le(2, 3)
                .check([real(2, 3), real(2, 3)].map(ok), [bool_t(3, 3)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn eq_ok() {
        assert!(
            LRA::Eq(2, 2)
                .check([real(2, 2), real(2, 2)].map(ok), [bool_t(2, 2)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn cmp_non_bool_output_fails() {
        let t = real(1, 1);
        assert!(LRA::Lt(1, 1).check([t, t].map(ok), [t].map(ok)).is_err());
    }

    #[test]
    fn cmp_input_mismatch_fails() {
        assert!(
            LRA::Eq(1, 1)
                .check([real(1, 1), real(1, 2)].map(ok), [bool_t(1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn add_ok() {
        let t = real(3, 4);
        assert!(LRA::Add(3, 4).check([t, t].map(ok), [t].map(ok)).is_ok());
    }

    #[test]
    fn add_shape_mismatch_fails() {
        assert!(
            LRA::Add(1, 2)
                .check([real(1, 2), real(2, 1)].map(ok), [real(1, 2)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn add_bool_fails() {
        let b = bool_t(1, 1);
        assert!(LRA::Add(1, 1).check([b, b].map(ok), [b].map(ok)).is_err());
    }

    #[test]
    fn relu_ok() {
        let t = real(3, 4);
        assert!(LRA::ReLU(3, 4).check([t].map(ok), [t].map(ok)).is_ok());
    }

    #[test]
    fn relu_bool_fails() {
        let b = bool_t(1, 1);
        assert!(LRA::ReLU(1, 1).check([b].map(ok), [b].map(ok)).is_err());
    }

    #[test]
    fn argmax_ok() {
        assert!(
            LRA::Argmax(3, 4)
                .check([real(3, 4)].map(ok), [real(1, 1)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn argmax_matrix_output_fails() {
        assert!(
            LRA::Argmax(3, 4)
                .check([real(3, 4)].map(ok), [real(3, 4)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn min_ok() {
        // pointwise
        let t = real(4, 1);
        assert!(LRA::Min(4, 1).check([t, t].map(ok), [t].map(ok)).is_ok());
        assert!(
            LRA::Max(4, 1)
                .check([t].map(ok), [real(1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn linear_ok() {
        // A=[2,3] maps 3 features to 2; X=[3,4] is 3 features × 4 batch items.
        // Convention: Y = A·X  →  Y=[2,4].
        assert!(
            LRA::Linear(real_tensor(2, 3), real_tensor(0, 0), 4)
                .check([real(3, 4)].map(ok), [real(2, 4)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn linear_with_bias_ok() {
        // A=[2,3], b=[2,1] column bias, X=[3,1] single sample → Y=[2,1].
        assert!(
            LRA::Linear(real_tensor(2, 3), real_tensor(2, 1), 1)
                .check([real(3, 1)].map(ok), [real(2, 1)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn linear_dim_mismatch_fails() {
        // A=[2,3] but X has 4 rows — inner dimension mismatch.
        let linear = |batch| LRA::Linear(real_tensor(2, 3), real_tensor(0, 0), batch);
        assert!(
            linear(1)
                .check([real(4, 1)].map(ok), [real(2, 1)].map(ok))
                .is_err()
        );
        // the batch is given by the operation
        assert!(
            linear(2)
                .check([real(3, 1)].map(ok), [real(2, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn transpose_ok() {
        assert!(
            LRA::TransposeReal(3, 4)
                .check([real(3, 4)].map(ok), [real(4, 3)].map(ok))
                .is_ok()
        );
        assert!(
            LRA::TransposeBool(3, 4)
                .check([bool_t(3, 4)].map(ok), [bool_t(4, 3)].map(ok))
                .is_ok()
        );
        assert!(
            LRA::TransposeReal(3, 4)
                .check([bool_t(3, 4)].map(ok), [bool_t(4, 3)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn transpose_wrong_shape_fails() {
        assert!(
            LRA::TransposeReal(3, 4)
                .check([real(3, 4)].map(ok), [real(3, 4)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn ite_ok() {
        let t = real(3, 4);
        assert!(
            LRA::IteReal(3, 4)
                .check([bool_t(1, 1), t, t].map(ok), [t].map(ok))
                .is_ok()
        );
        let b = bool_t(3, 4);
        assert!(
            LRA::IteBool(3, 4)
                .check([bool_t(1, 1), b, b].map(ok), [b].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn ite_non_bool_guard_fails() {
        let t = real(1, 1);
        assert!(
            LRA::IteReal(1, 1)
                .check([t, t, t].map(ok), [t].map(ok))
                .is_err()
        );
    }

    #[test]
    fn ite_arm_mismatch_fails() {
        assert!(
            LRA::IteReal(1, 1)
                .check(
                    [bool_t(1, 1), real(1, 1), real(1, 2)].map(ok),
                    [real(1, 1)].map(ok)
                )
                .is_err()
        );
    }

    #[test]
    fn id_ok() {
        let t = real(4, 4);
        assert!(LRA::IdReal(4, 4).check([t].map(ok), [t].map(ok)).is_ok());
        let b = bool_t(4, 4);
        assert!(LRA::IdBool(4, 4).check([b].map(ok), [b].map(ok)).is_ok());
        assert!(LRA::IdReal(4, 4).check([b].map(ok), [b].map(ok)).is_err());
    }

    #[test]
    fn id_type_mismatch_fails() {
        assert!(
            LRA::IdReal(1, 1)
                .check([real(1, 1)].map(ok), [real(2, 2)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn id_arity_mismatch_fails() {
        let t = real(1, 1);
        let id = || LRA::IdReal(1, 1);
        assert!(id().check([t, t].map(ok), [t, t].map(ok)).is_err());
        assert!(id().check([t, t].map(ok), [t].map(ok)).is_err());
        assert!(id().check([t].map(ok), [t, t].map(ok)).is_err());
    }

    #[test]
    fn havoc_ok() {
        assert!(
            LRA::AnyReal(2, 1)
                .check([].map(ok), [real(2, 1)].map(ok))
                .is_ok()
        );
        assert!(
            LRA::AnyBool(1, 1)
                .check([].map(ok), [bool_t(1, 1)].map(ok))
                .is_ok()
        );
        // `havoc` is given the sort of the written wire
        assert!(matches!(LRA::havoc(&real(2, 1)), LRA::AnyReal(2, 1)));
        assert!(matches!(LRA::havoc(&bool_t(2, 1)), LRA::AnyBool(2, 1)));
    }

    #[test]
    fn havoc_read_fails() {
        let t = real(1, 1);
        assert!(LRA::AnyReal(1, 1).check([t].map(ok), [t].map(ok)).is_err());
    }

    #[test]
    fn havoc_arity_mismatch_fails() {
        assert!(
            LRA::AnyReal(2, 1)
                .check([].map(ok), [real(2, 1), bool_t(1, 1)].map(ok))
                .is_err()
        );
        assert!(LRA::AnyReal(1, 1).check([].map(ok), [].map(ok)).is_err());
        // the write sort must match the op's declared range
        assert!(
            LRA::AnyReal(2, 1)
                .check([].map(ok), [real(1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn skip_ok() {
        // `skip` copies a value, or a derivative, of the given sort
        let (r, dr) = (real(2, 1), real(2, 1).tangent());
        assert!(matches!(LRA::skip(&r), LRA::IdReal(2, 1)));
        assert!(matches!(LRA::skip(&dr), LRA::TId(2, 1, 1)));
        assert!(matches!(LRA::skip(&bool_t(2, 1)), LRA::IdBool(2, 1)));
        assert!(LRA::skip(&r).check([r].map(ok), [r].map(ok)).is_ok());
        assert!(LRA::skip(&dr).check([dr].map(ok), [dr].map(ok)).is_ok());
        // a zero derivative cannot be copied
        let z = bool_t(2, 1).tangent();
        assert!(LRA::skip(&z).check([z].map(ok), [z].map(ok)).is_err());
    }

    #[test]
    fn zero_ok() {
        // the zero derivative is written to the derivative (a tangent), not to a value
        let zero = |op: LRA, s: Sort| op.check([].map(ok), [s].map(ok));
        let (r, b) = (real(2, 1), bool_t(2, 1));
        assert!(zero(LRA::RealZerograd(2, 1, 1), r.tangent()).is_ok());
        assert!(zero(LRA::RealZerograd(2, 1, 1), r).is_err());
        assert!(zero(LRA::RealZerograd(2, 1, 1), r.tangent().tangent()).is_err());
        assert!(zero(LRA::RealZerograd(2, 1, 2), r.tangent().tangent()).is_ok());
        assert!(zero(LRA::BoolZerograd(2, 1), b.tangent()).is_ok());
        assert!(zero(LRA::BoolZerograd(2, 1), b).is_err());
        // `zero` is given the sort of the derivative
        assert!(matches!(
            LRA::zero(&r.tangent()),
            LRA::RealZerograd(2, 1, 1)
        ));
        assert!(matches!(LRA::zero(&b.tangent()), LRA::BoolZerograd(2, 1)));
        // values have no zero derivative
        assert!(zero(LRA::zero(&r), r).is_err());
    }

    #[test]
    fn derivatives() {
        let (r, b) = (real(2, 1), bool_t(2, 1));
        assert_eq!(
            r.tangent(),
            Sort::Real {
                shape: [2, 1],
                rank: 1
            }
        );
        assert_eq!(
            r.tangent().tangent(),
            Sort::Real {
                shape: [2, 1],
                rank: 2
            }
        );
        assert_eq!(b.tangent(), Sort::Zero { shape: [2, 1] });
        assert_eq!(b.tangent().tangent(), Sort::Zero { shape: [2, 1] });
        // the operations on derivatives (of the same rank)
        let dr = r.tangent();
        let g = bool_t(1, 1);
        assert!(
            LRA::TAdd(2, 1, 1)
                .check([dr, dr].map(ok), [dr].map(ok))
                .is_ok()
        );
        assert!(
            LRA::TSub(2, 1, 1)
                .check([dr, dr].map(ok), [dr].map(ok))
                .is_ok()
        );
        assert!(LRA::TId(2, 1, 1).check([dr].map(ok), [dr].map(ok)).is_ok());
        assert!(
            LRA::TIte(2, 1, 1)
                .check([g, dr, dr].map(ok), [dr].map(ok))
                .is_ok()
        );
        let ddr = dr.tangent();
        assert!(
            LRA::TAdd(2, 1, 2)
                .check([ddr, ddr].map(ok), [ddr].map(ok))
                .is_ok()
        );
        // ... do not apply to values, nor to derivatives of other ranks
        assert!(
            LRA::TAdd(2, 1, 1)
                .check([r, r].map(ok), [r].map(ok))
                .is_err()
        );
        assert!(
            LRA::TAdd(2, 1, 1)
                .check([dr, r].map(ok), [dr].map(ok))
                .is_err()
        );
        assert!(
            LRA::TAdd(2, 1, 1)
                .check([dr, dr].map(ok), [ddr].map(ok))
                .is_err()
        );
        // the operations on values do not apply to derivatives
        assert!(
            LRA::Add(2, 1)
                .check([dr, dr].map(ok), [dr].map(ok))
                .is_err()
        );
        assert!(LRA::ReLU(2, 1).check([dr].map(ok), [dr].map(ok)).is_err());
        assert!(LRA::Lt(2, 1).check([dr, dr].map(ok), [b].map(ok)).is_err());
        assert!(LRA::IdReal(2, 1).check([dr].map(ok), [dr].map(ok)).is_err());
        // `dx = x` mixes a derivative and a value
        assert!(LRA::IdReal(2, 1).check([r].map(ok), [dr].map(ok)).is_err());
        assert!(LRA::TId(2, 1, 1).check([r].map(ok), [dr].map(ok)).is_err());
        // boolean operations and guards are on values only
        assert!(
            LRA::Not(2, 1)
                .check([b.tangent()].map(ok), [b.tangent()].map(ok))
                .is_err()
        );
        assert!(
            LRA::TIte(2, 1, 1)
                .check([g.tangent(), dr, dr].map(ok), [dr].map(ok))
                .is_err()
        );
    }

    #[test]
    fn zero_arity_mismatch_fails() {
        let t = real(1, 1).tangent();
        assert!(
            LRA::RealZerograd(1, 1, 1)
                .check([t].map(ok), [t].map(ok))
                .is_err()
        );
        assert!(
            LRA::RealZerograd(1, 1, 1)
                .check([].map(ok), [t, t].map(ok))
                .is_err()
        );
        assert!(
            LRA::RealZerograd(1, 1, 1)
                .check([].map(ok), [].map(ok))
                .is_err()
        );
    }

    // Formats the boundary of `op` as `dom -> cod`
    fn show(op: LRA) -> String {
        format!("{} -> {}", op.dom(), op.cod())
    }

    #[test]
    fn boundaries() {
        assert_eq!(
            show(LRA::Add(2, 3)),
            "Real(2, 3) ⊗ Real(2, 3) -> Real(2, 3)"
        );
        assert_eq!(show(LRA::Lt(2, 3)), "Real(2, 3) ⊗ Real(2, 3) -> Bool(2, 3)");
        assert_eq!(
            show(LRA::Linear(real_tensor(2, 3), real_tensor(0, 0), 4)),
            "Real(3, 4) -> Real(2, 4)"
        );
        assert_eq!(show(LRA::Argmax(2, 3)), "Real(2, 3) -> Real(1, 1)");
        assert_eq!(
            show(LRA::And(2, 3)),
            "Bool(2, 3) ⊗ Bool(2, 3) -> Bool(2, 3)"
        );
        assert_eq!(show(LRA::Real(real_tensor(2, 3))), "I -> Real(2, 3)");
        assert_eq!(show(LRA::AnyReal(2, 3)), "I -> Real(2, 3)");
        assert_eq!(show(LRA::TransposeReal(2, 3)), "Real(2, 3) -> Real(3, 2)");
        assert_eq!(show(LRA::TransposeBool(2, 3)), "Bool(2, 3) -> Bool(3, 2)");
        assert_eq!(
            show(LRA::IteReal(2, 3)),
            "Bool(1, 1) ⊗ Real(2, 3) ⊗ Real(2, 3) -> Real(2, 3)"
        );
        assert_eq!(show(LRA::IdBool(2, 3)), "Bool(2, 3) -> Bool(2, 3)");
        assert_eq!(
            show(LRA::UninterpretedBool("f".into(), 2, 3)),
            "I -> Bool(2, 3)"
        );
        // derivatives
        assert_eq!(
            show(LRA::TAdd(2, 3, 1)),
            "TReal(2, 3; 1) ⊗ TReal(2, 3; 1) -> TReal(2, 3; 1)"
        );
        assert_eq!(
            show(LRA::TLinear(real_tensor(2, 3), real_tensor(2, 1), 1, 2)),
            "TReal(3, 1; 2) -> TReal(2, 1; 2)"
        );
        assert_eq!(
            show(LRA::TIte(2, 3, 1)),
            "Bool(1, 1) ⊗ TReal(2, 3; 1) ⊗ TReal(2, 3; 1) -> TReal(2, 3; 1)"
        );
        assert_eq!(show(LRA::RealZerograd(2, 3, 1)), "I -> TReal(2, 3; 1)");
        assert_eq!(show(LRA::BoolZerograd(2, 3)), "I -> Zero(2, 3)");
        // the operations with the same boundaries
        assert_eq!(show(LRA::Add(2, 3)), show(LRA::Max(2, 3)));
        assert_ne!(show(LRA::AnyReal(2, 3)), show(LRA::AnyBool(2, 3)));
    }

    #[test]
    #[should_panic(expected = "the rank of a derivative must be at least 1")]
    fn boundary_ill_formed() {
        let _ = LRA::TAdd(2, 3, 0).dom();
    }

    #[test]
    fn display() {
        assert_eq!(LRA::Add(2, 3).to_string(), "Add");
        assert_eq!(LRA::IdReal(2, 3).to_string(), "IdReal");
        assert_eq!(LRA::TAdd(2, 3, 1).to_string(), "TAdd");
        assert_eq!(
            LRA::UninterpretedReal("f".into(), 2, 3).to_string(),
            "UninterpretedReal(f)"
        );
    }

    #[test]
    fn check_errors() {
        assert_eq!(
            LRA::Add(1, 1).check([real(1, 1), real(1, 2)].map(ok), [real(1, 1)].map(ok)),
            Err(
                "Add(1, 1): expected `Real(1, 1) ⊗ Real(1, 1) -> Real(1, 1)`, \
                 got `Real(1, 1) ⊗ Real(1, 2) -> Real(1, 1)`"
                    .to_string()
            )
        );
        // derivatives and values do not mix
        let dr = real(1, 1).tangent();
        assert_eq!(
            LRA::TAdd(1, 1, 1).check([dr, real(1, 1)].map(ok), [dr].map(ok)),
            Err(
                "TAdd(1, 1, 1): expected `TReal(1, 1; 1) ⊗ TReal(1, 1; 1) -> TReal(1, 1; 1)`, \
                 got `TReal(1, 1; 1) ⊗ Real(1, 1) -> TReal(1, 1; 1)`"
                    .to_string()
            )
        );
        let b = bool_t(1, 1);
        assert_eq!(
            LRA::Add(1, 1).check([b, b].map(ok), [b].map(ok)),
            Err(
                "Add(1, 1): expected `Real(1, 1) ⊗ Real(1, 1) -> Real(1, 1)`, \
                 got `Bool(1, 1) ⊗ Bool(1, 1) -> Bool(1, 1)`"
                    .to_string()
            )
        );
        // a nullary write
        assert_eq!(
            LRA::AnyReal(1, 1).check([].map(ok), [].map(ok)),
            Err("AnyReal(1, 1): expected `I -> Real(1, 1)`, got `I -> I`".to_string())
        );
        // errors of wires are propagated
        assert_eq!(
            LRA::Not(1, 1).check([Err("bad wire")], [b].map(Ok)),
            Err("Not(1, 1): bad wire".to_string())
        );
    }

    #[test]
    fn check_ill_formed_ops() {
        // ill-formed tensors are reported, not panicking
        let t: crate::PyTensor = tch::Tensor::from_slice(&[0.0f64, 1.0]).into();
        let res = LRA::Real(t).check([].map(ok), [real(1, 2)].map(ok));
        assert!(res.unwrap_err().ends_with("expected a 2D tensor, got 1D"));
        let res = LRA::Linear(real_tensor(2, 3), real_tensor(2, 2), 1)
            .check([real(3, 1)].map(ok), [real(2, 1)].map(ok));
        assert!(
            res.unwrap_err()
                .ends_with("`B` must be a column vector [2x1]")
        );
        // a derivative has the rank at least 1
        let t = real(1, 1);
        let res = LRA::TAdd(1, 1, 0).check([t, t].map(ok), [t].map(ok));
        assert!(
            res.unwrap_err()
                .ends_with("the rank of a derivative must be at least 1")
        );
    }

    #[test]
    fn check_uninterpreted() {
        let f = |r, c| LRA::UninterpretedReal("f".into(), r, c);
        let t = real(2, 3);
        // a single write of the given sort
        assert!(f(2, 3).check([].map(ok), [t].map(ok)).is_ok());
        assert!(f(3, 2).check([].map(ok), [t].map(ok)).is_err());
        assert!(f(2, 3).check([].map(ok), [t.tangent()].map(ok)).is_err());
        assert!(f(2, 3).check([t].map(ok), [].map(ok)).is_err());
        assert!(f(2, 3).check([t].map(ok), [t].map(ok)).is_err());
        assert!(f(2, 3).check([].map(ok), [t, t].map(ok)).is_err());
        assert!(f(2, 3).check([].map(ok), [].map(ok)).is_err());
        let b = bool_t(2, 3);
        let g = LRA::UninterpretedBool("g".into(), 2, 3);
        assert!(g.check([].map(ok), [b].map(ok)).is_ok());
        assert!(f(2, 3).check([].map(ok), [b].map(ok)).is_err());
    }

    #[test]
    fn linear_derivatives() {
        // `2 * d(t)` folds to a Linear over derivatives
        let lin = |rank| LRA::TLinear(real_tensor(2, 3), real_tensor(2, 1), 1, rank);
        let (x, y) = (real(3, 1), real(2, 1));
        assert!(
            lin(1)
                .check([x.tangent()].map(ok), [y.tangent()].map(ok))
                .is_ok()
        );
        assert!(lin(1).check([x.tangent()].map(ok), [y].map(ok)).is_err());
        assert!(lin(1).check([x].map(ok), [y.tangent()].map(ok)).is_err());
        assert!(lin(1).check([x].map(ok), [y].map(ok)).is_err());
        assert!(
            lin(2)
                .check(
                    [x.tangent().tangent()].map(ok),
                    [y.tangent().tangent()].map(ok)
                )
                .is_ok()
        );
        // `Linear` is on values only
        let linear = LRA::Linear(real_tensor(2, 3), real_tensor(2, 1), 1);
        assert!(
            linear
                .check([x.tangent()].map(ok), [y.tangent()].map(ok))
                .is_err()
        );
    }

    #[test]
    fn discard_ok() {
        // `discard` reads one wire of the given sort, a value or a derivative, and writes none
        let (r, b) = (real(2, 1), bool_t(2, 1));
        let (dr, db) = (r.tangent(), b.tangent());
        assert!(matches!(LRA::discard(&r), LRA::DiscardReal(2, 1)));
        assert!(matches!(LRA::discard(&b), LRA::DiscardBool(2, 1)));
        assert!(matches!(LRA::discard(&dr), LRA::TDiscard(2, 1, 1)));
        assert!(matches!(LRA::discard(&db), LRA::DiscardZero(2, 1)));
        for s in [r, b, dr, db] {
            assert!(LRA::discard(&s).check([s].map(ok), [].map(ok)).is_ok());
            assert_eq!(LRA::discard(&s).cod(), Poly::unit());
        }
    }

    #[test]
    fn discard_write_or_mismatch_fails() {
        let (r, b) = (real(2, 1), bool_t(2, 1));
        let discard = LRA::discard(&r);
        // nothing is written
        assert!(discard.check([r].map(ok), [r].map(ok)).is_err());
        // exactly one wire of the domain is read
        assert!(discard.check([].map(ok), [].map(ok)).is_err());
        assert!(discard.check([r, r].map(ok), [].map(ok)).is_err());
        assert!(discard.check([b].map(ok), [].map(ok)).is_err());
        assert!(discard.check([r.tangent()].map(ok), [].map(ok)).is_err());
        // the rank of a discarded derivative is at least 1
        assert!(
            LRA::TDiscard(2, 1, 0)
                .check([r].map(ok), [].map(ok))
                .is_err()
        );
    }
}
