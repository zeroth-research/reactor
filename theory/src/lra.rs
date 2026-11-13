/*!
# Linear real arithmetic

Defines the theory [`LRA`] of linear real arithmetic over matrices,
mixing real and boolean matrices in a single signature.

A [`Sort`] value is either a real matrix `Real(shape, degree)` or a boolean
matrix `Bool(shape)`, or `Zero(shape)`, the sort of the derivatives of
booleans (see [`Tangent::tangent`]). The `degree` of a real is `0` for values,
and `d >= 1` for their `d`-th derivatives.

The operations carry their parameters (the shapes of the matrices, and the degrees
of the derivatives), so that each operation has a fixed [`Boundary`]: e.g., the
boundary of `Add(2, 3)` is `Real(2, 3) ⊗ Real(2, 3) -> Real(2, 3)`. The parameters
that can be ill-formed are checked on their creation (the tensors of the literals are
[`Matrix`]es, the affine maps are [`Affine`], and the degrees of the derivatives are
[`NonZeroU8`]), so the boundary of every operation exists. The operations
that apply to several kinds of values come in one variant per kind, e.g.,
[`LRA::IdReal`] and [`LRA::IdBool`]. The operations in [`LRA`] are:

- [`LRA::Real`], [`LRA::Bool`] — real and boolean matrix literals.
- [`LRA::And`], [`LRA::Or`], [`LRA::Xor`], [`LRA::Not`]
  — boolean operations.
- [`LRA::Le`], [`LRA::Lt`], [`LRA::Ge`], [`LRA::Gt`], [`LRA::Eq`], [`LRA::Ne`]
  — pointwise real comparisons producing a boolean matrix.
- [`LRA::IteReal`], [`LRA::IteBool`] — if-then-else: reads a boolean guard and
  two same-typed branches.
- [`LRA::Linear`]`(affine, batch)` — the affine map `x ↦ A·x + B` (see [`Affine`]),
  with `A` and `B` constant real matrices, of `x` with `batch` columns.
- [`LRA::ReLU`] — the shape-preserving rectified-linear map on real matrices.
- [`LRA::TAdd`], [`LRA::TSub`], [`LRA::TLinear`], [`LRA::TIte`], [`LRA::TId`]
  — the operations on the derivatives of reals (of the degree `>= 1`).
- [`LRA::RealZerograd`], [`LRA::BoolZerograd`] — the zero derivatives.

An operation applies to the wires whose sorts are its boundary: the read wires
have the sorts of its domain, and the written wires the sorts of its codomain.

## Examples

```
use base::boundary::Boundary;
use theory::lra::LRA;
use std::num::NonZeroU8;

let boundary = |op: LRA| format!("{} -> {}", op.dom(), op.cod());

// Pointwise less-than on scalars.
assert_eq!(boundary(LRA::Lt(1, 1)), "Real(1, 1) ⊗ Real(1, 1) -> Bool(1, 1)");

// ReLU preserves shape and stays in the real fragment.
assert_eq!(boundary(LRA::ReLU(3, 4)), "Real(3, 4) -> Real(3, 4)");

// Derivatives have their own operations.
assert_eq!(
    boundary(LRA::TAdd(3, 4, NonZeroU8::MIN)),
    "Real(3, 4; 1) ⊗ Real(3, 4; 1) -> Real(3, 4; 1)"
);
```
*/

use crate::*;
use base::boundary::{Boundary, Mono, Poly};
use base::tangent::Tangent;
#[cfg(feature = "pyo3")]
use pyo3::pyclass;
use std::fmt;
use std::num::NonZeroU8;

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub enum Sort {
    // a real of the given shape and degree: a value at degree 0, its derivative
    // of that order (the tangent of the degree below) at degree >= 1
    Real([usize; 2], u8),
    Bool([usize; 2]),
    // the tangent of `Bool` (it has only the zero derivative)
    Zero([usize; 2]),
}

impl Sort {
    pub fn is_bool(&self) -> bool {
        matches!(self, Sort::Bool(..))
    }

    /// Is this sort a real value (i.e., a real of degree 0)?
    pub fn is_real(&self) -> bool {
        matches!(self, Sort::Real(_, 0))
    }

    /// Is this sort a tangent (i.e., a sort of derivatives)?
    pub fn is_tangent(&self) -> bool {
        matches!(self, Sort::Real(_, 1..) | Sort::Zero(..))
    }

    pub fn shape(&self) -> &[usize; 2] {
        match self {
            Sort::Bool(shape) | Sort::Real(shape, _) | Sort::Zero(shape) => shape,
        }
    }
}

impl fmt::Display for Sort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sort::Real([i, j], 0) => write!(f, "Real({i}, {j})"),
            Sort::Real([i, j], d) => write!(f, "Real({i}, {j}; {d})"),
            Sort::Bool([i, j]) => write!(f, "Bool({i}, {j})"),
            Sort::Zero([i, j]) => write!(f, "Zero({i}, {j})"),
        }
    }
}

impl Tangent for Sort {
    fn tangent(&self) -> Self {
        match self {
            Sort::Real(shape, d) => Sort::Real(*shape, d + 1),
            Sort::Bool(shape) | Sort::Zero(shape) => Sort::Zero(*shape),
        }
    }
}

/// The operations of linear real arithmetic. The parameters `(rows, cols)` are the shape
/// of the matrices the operation applies to (e.g., of its operands), and `degree` is the
/// degree of the derivatives (at least `1`, hence [`NonZeroU8`]).
#[derive(Clone, Debug, strum::IntoStaticStr)]
#[cfg_attr(feature = "pyo3", pyclass(frozen))]
pub enum LRA {
    // real and boolean matrix literals (of the shape of the tensor)
    Real(Matrix),
    Bool(Matrix),
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
    Linear(Affine, usize),
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
    // operations on derivatives of reals of the degree `degree`
    TAdd(usize, usize, NonZeroU8),
    TSub(usize, usize, NonZeroU8),
    TLinear(Affine, usize, NonZeroU8),
    TIte(usize, usize, NonZeroU8),
    TId(usize, usize, NonZeroU8),
    // the zero derivatives
    RealZerograd(usize, usize, NonZeroU8), // unstable
    BoolZerograd(usize, usize),            // unstable
}

/// Prints the name of the operation, the tensors of the literals and of the affine maps,
/// and the names of the uninterpreted values (the other parameters are given by the wires)
impl fmt::Display for LRA {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name: &'static str = self.into();
        match self {
            LRA::Real(t) | LRA::Bool(t) => write!(f, "{t}"),
            LRA::Linear(m, ..) | LRA::TLinear(m, ..) => write!(f, "{name}({}, {})", m.a(), m.b()),
            LRA::UninterpretedReal(x, ..) | LRA::UninterpretedBool(x, ..) => {
                write!(f, "{name}({x})")
            }
            _ => f.write_str(name),
        }
    }
}

impl Sequential<Sort> for LRA {
    fn skip(range: &Sort) -> Self {
        match *range {
            Sort::Real([r, c], degree) => match NonZeroU8::new(degree) {
                None => LRA::IdReal(r, c),
                Some(degree) => LRA::TId(r, c, degree),
            },
            // a zero derivative cannot be copied: the check of the term with it fails
            Sort::Bool([r, c]) | Sort::Zero([r, c]) => LRA::IdBool(r, c),
        }
    }
}

impl Combinatorial<Sort> for LRA {
    fn havoc(range: &Sort) -> Self {
        // a derivative cannot be havoced: the check of the term with a tangent fails
        match *range {
            Sort::Bool([r, c]) | Sort::Zero([r, c]) => LRA::AnyBool(r, c),
            Sort::Real([r, c], _) => LRA::AnyReal(r, c),
        }
    }
}

impl Differential<Sort> for LRA {
    fn zero(range: &Sort) -> Self {
        // the range is the sort of the derivative; values have no zero derivative: the
        // check of the term with a value fails (the zero of a real value is the zero of its
        // first derivative)
        match *range {
            Sort::Bool([r, c]) | Sort::Zero([r, c]) => LRA::BoolZerograd(r, c),
            Sort::Real([r, c], degree) => {
                LRA::RealZerograd(r, c, NonZeroU8::new(degree).unwrap_or(NonZeroU8::MIN))
            }
        }
    }
}

impl Boundary<Sort> for LRA {
    fn dom(&self) -> Poly<Sort> {
        let real = |r, c| Sort::Real([r, c], 0);
        let bool = |r, c| Sort::Bool([r, c]);
        let sorts = match *self {
            LRA::Real(_) | LRA::Bool(_) => vec![],
            LRA::And(r, c) | LRA::Or(r, c) | LRA::Xor(r, c) => vec![bool(r, c), bool(r, c)],
            LRA::Not(r, c) => vec![bool(r, c)],
            LRA::Le(r, c)
            | LRA::Lt(r, c)
            | LRA::Ge(r, c)
            | LRA::Gt(r, c)
            | LRA::Eq(r, c)
            | LRA::Ne(r, c) => vec![real(r, c), real(r, c)],
            // Y = A·X + B  where X=[in,batch], A=[out,in], B=[out,1] (or empty), Y=[out,batch]
            LRA::Linear(ref m, batch) => vec![real(m.inputs(), batch)],
            // `Min` and `Max` are pointwise, as in the evaluator
            LRA::Add(r, c) | LRA::Sub(r, c) | LRA::Min(r, c) | LRA::Max(r, c) => {
                vec![real(r, c), real(r, c)]
            }
            LRA::ReLU(r, c) | LRA::Argmax(r, c) | LRA::TransposeReal(r, c) => vec![real(r, c)],
            LRA::TransposeBool(r, c) => vec![bool(r, c)],
            LRA::IteReal(r, c) => vec![bool(1, 1), real(r, c), real(r, c)],
            LRA::IteBool(r, c) => vec![bool(1, 1), bool(r, c), bool(r, c)],
            LRA::IdReal(r, c) => vec![real(r, c)],
            LRA::IdBool(r, c) => vec![bool(r, c)],
            LRA::UninterpretedReal(..)
            | LRA::AnyReal(..)
            | LRA::UninterpretedBool(..)
            | LRA::AnyBool(..) => vec![],
            LRA::TAdd(r, c, degree) | LRA::TSub(r, c, degree) => {
                let t = treal(r, c, degree);
                vec![t, t]
            }
            LRA::TLinear(ref m, batch, degree) => vec![treal(m.inputs(), batch, degree)],
            LRA::TIte(r, c, degree) => {
                let t = treal(r, c, degree);
                vec![bool(1, 1), t, t]
            }
            LRA::TId(r, c, degree) => vec![treal(r, c, degree)],
            LRA::RealZerograd(..) | LRA::BoolZerograd(..) => vec![],
        };
        sorts.into_iter().product::<Mono<Sort>>().into()
    }

    fn cod(&self) -> Poly<Sort> {
        let real = |r, c| Sort::Real([r, c], 0);
        let bool = |r, c| Sort::Bool([r, c]);
        let sort = match *self {
            LRA::Real(ref m) => {
                let [r, c] = m.shape();
                real(r, c)
            }
            LRA::Bool(ref m) => {
                let [r, c] = m.shape();
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
            LRA::Linear(ref m, batch) => real(m.outputs(), batch),
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
            LRA::TAdd(r, c, degree)
            | LRA::TSub(r, c, degree)
            | LRA::TIte(r, c, degree)
            | LRA::TId(r, c, degree)
            | LRA::RealZerograd(r, c, degree) => treal(r, c, degree),
            LRA::TLinear(ref m, batch, degree) => treal(m.outputs(), batch, degree),
            LRA::BoolZerograd(r, c) => bool(r, c).tangent(),
        };
        [sort].into_iter().product::<Mono<Sort>>().into()
    }
}

/// The sort of the derivatives of the degree `degree` of the reals of the shape `[r, c]`
fn treal(r: usize, c: usize, degree: NonZeroU8) -> Sort {
    Sort::Real([r, c], degree.get())
}

#[cfg(all(test, feature = "torch"))]
mod tests {
    use super::*;

    fn real(r: usize, c: usize) -> Sort {
        Sort::Real([r, c], 0)
    }

    fn bool_t(r: usize, c: usize) -> Sort {
        Sort::Bool([r, c])
    }

    fn real_tensor(rows: i64, cols: i64) -> crate::PyTensor {
        tch::Tensor::zeros([rows, cols], (tch::Kind::Float, tch::Device::Cpu)).into()
    }

    fn matrix(t: crate::PyTensor) -> Matrix {
        Matrix::new(t).unwrap()
    }

    fn affine(a: crate::PyTensor, b: crate::PyTensor) -> Affine {
        Affine::new(a, b).unwrap()
    }

    fn deg(degree: u8) -> NonZeroU8 {
        NonZeroU8::new(degree).unwrap()
    }

    // Formats the boundary of `op` as `dom -> cod`
    fn show(op: LRA) -> String {
        format!("{} -> {}", op.dom(), op.cod())
    }

    #[test]
    fn type_kind_and_shape() {
        assert!(real(2, 3).is_real() && !real(2, 3).is_bool());
        assert_eq!(real(2, 3).shape(), &[2, 3]);
        assert!(bool_t(1, 1).is_bool() && !bool_t(1, 1).is_real());
    }

    #[test]
    fn literals() {
        // a literal writes a matrix of the shape of its tensor; a real literal writes a
        // value, never a derivative
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[0.0f64, 1.0], [2.0, 3.0]]).into();
        assert_eq!(show(LRA::Real(matrix(cm))), "I -> Real(2, 2)");
        assert_eq!(
            show(LRA::Real(matrix(real_tensor(2, 3)))),
            "I -> Real(2, 3)"
        );
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[true, false], [false, true]]).into();
        assert_eq!(show(LRA::Bool(matrix(cm))), "I -> Bool(2, 2)");
    }

    #[test]
    fn boolean_ops() {
        let bin = |s| format!("{s} ⊗ {s} -> {s}");
        for op in [LRA::And(2, 3), LRA::Or(2, 3), LRA::Xor(2, 3)] {
            assert_eq!(show(op), bin("Bool(2, 3)"));
        }
        assert_eq!(show(LRA::Not(2, 3)), "Bool(2, 3) -> Bool(2, 3)");
        // the shape is given by the operation
        assert_eq!(show(LRA::Not(3, 2)), "Bool(3, 2) -> Bool(3, 2)");
    }

    #[test]
    fn comparisons() {
        for op in [
            LRA::Le(2, 3),
            LRA::Lt(2, 3),
            LRA::Ge(2, 3),
            LRA::Gt(2, 3),
            LRA::Eq(2, 3),
            LRA::Ne(2, 3),
        ] {
            assert_eq!(show(op), "Real(2, 3) ⊗ Real(2, 3) -> Bool(2, 3)");
        }
    }

    #[test]
    fn arithmetic() {
        // `Min` and `Max` are pointwise
        for op in [
            LRA::Add(2, 3),
            LRA::Sub(2, 3),
            LRA::Min(2, 3),
            LRA::Max(2, 3),
        ] {
            assert_eq!(show(op), "Real(2, 3) ⊗ Real(2, 3) -> Real(2, 3)");
        }
        assert_eq!(show(LRA::ReLU(3, 4)), "Real(3, 4) -> Real(3, 4)");
        // the index into the flattened matrix
        assert_eq!(show(LRA::Argmax(3, 4)), "Real(3, 4) -> Real(1, 1)");
    }

    #[test]
    fn linear() {
        // A=[2,3] maps 3 features to 2; X=[3,4] is 3 features × 4 batch items.
        // Convention: Y = A·X  →  Y=[2,4].
        let linear = |b, batch| LRA::Linear(affine(real_tensor(2, 3), b), batch);
        assert_eq!(
            show(linear(real_tensor(0, 0), 4)),
            "Real(3, 4) -> Real(2, 4)"
        );
        // A=[2,3], b=[2,1] column bias, X=[3,1] single sample → Y=[2,1].
        assert_eq!(
            show(linear(real_tensor(2, 1), 1)),
            "Real(3, 1) -> Real(2, 1)"
        );
        // the batch is given by the operation
        assert_eq!(
            show(linear(real_tensor(0, 0), 2)),
            "Real(3, 2) -> Real(2, 2)"
        );
    }

    #[test]
    fn transpose() {
        assert_eq!(show(LRA::TransposeReal(3, 4)), "Real(3, 4) -> Real(4, 3)");
        assert_eq!(show(LRA::TransposeBool(3, 4)), "Bool(3, 4) -> Bool(4, 3)");
    }

    #[test]
    fn ite() {
        // a single boolean guard and two branches of the sort of the result
        assert_eq!(
            show(LRA::IteReal(3, 4)),
            "Bool(1, 1) ⊗ Real(3, 4) ⊗ Real(3, 4) -> Real(3, 4)"
        );
        assert_eq!(
            show(LRA::IteBool(3, 4)),
            "Bool(1, 1) ⊗ Bool(3, 4) ⊗ Bool(3, 4) -> Bool(3, 4)"
        );
    }

    #[test]
    fn id() {
        assert_eq!(show(LRA::IdReal(4, 4)), "Real(4, 4) -> Real(4, 4)");
        assert_eq!(show(LRA::IdBool(4, 4)), "Bool(4, 4) -> Bool(4, 4)");
    }

    #[test]
    fn uninterpreted() {
        // a single write of the given sort
        assert_eq!(
            show(LRA::UninterpretedReal("f".into(), 2, 3)),
            "I -> Real(2, 3)"
        );
        assert_eq!(
            show(LRA::UninterpretedBool("g".into(), 2, 3)),
            "I -> Bool(2, 3)"
        );
    }

    #[test]
    fn havoc() {
        assert_eq!(show(LRA::AnyReal(2, 1)), "I -> Real(2, 1)");
        assert_eq!(show(LRA::AnyBool(2, 1)), "I -> Bool(2, 1)");
        // `havoc` is given the sort of the written wire
        assert!(matches!(LRA::havoc(&real(2, 1)), LRA::AnyReal(2, 1)));
        assert!(matches!(LRA::havoc(&bool_t(2, 1)), LRA::AnyBool(2, 1)));
    }

    #[test]
    fn skip() {
        // `skip` copies a value, or a derivative, of the given sort
        let (r, dr) = (real(2, 1), real(2, 1).tangent());
        assert!(matches!(LRA::skip(&r), LRA::IdReal(2, 1)));
        assert!(matches!(LRA::skip(&dr), LRA::TId(2, 1, d) if d.get() == 1));
        assert!(matches!(LRA::skip(&bool_t(2, 1)), LRA::IdBool(2, 1)));
        assert_eq!(show(LRA::skip(&dr)), "Real(2, 1; 1) -> Real(2, 1; 1)");
        // a zero derivative cannot be copied: `skip` of it copies a boolean
        let z = bool_t(2, 1).tangent();
        assert_eq!(show(LRA::skip(&z)), "Bool(2, 1) -> Bool(2, 1)");
    }

    #[test]
    fn zero() {
        // the zero derivative is written to the derivative (a tangent), not to a value
        assert_eq!(show(LRA::RealZerograd(2, 1, deg(1))), "I -> Real(2, 1; 1)");
        assert_eq!(show(LRA::RealZerograd(2, 1, deg(2))), "I -> Real(2, 1; 2)");
        assert_eq!(show(LRA::BoolZerograd(2, 1)), "I -> Zero(2, 1)");
        // `zero` is given the sort of the derivative
        let (r, b) = (real(2, 1), bool_t(2, 1));
        assert!(matches!(
            LRA::zero(&r.tangent()),
            LRA::RealZerograd(2, 1, d) if d.get() == 1
        ));
        assert!(matches!(LRA::zero(&b.tangent()), LRA::BoolZerograd(2, 1)));
        // values have no zero derivative: the zero of a value is the zero of its first
        // derivative
        assert_eq!(show(LRA::zero(&r)), "I -> Real(2, 1; 1)");
        assert_eq!(show(LRA::zero(&b)), "I -> Zero(2, 1)");
    }

    #[test]
    fn derivatives() {
        let (r, b) = (real(2, 1), bool_t(2, 1));
        assert_eq!(r.tangent(), Sort::Real([2, 1], 1));
        assert_eq!(r.tangent().tangent(), Sort::Real([2, 1], 2));
        assert_eq!(b.tangent(), Sort::Zero([2, 1]));
        assert_eq!(b.tangent().tangent(), Sort::Zero([2, 1]));
        // the operations on derivatives read and write derivatives of their degree (the
        // operations on values read and write values only, see above)
        let bin = |s| format!("{s} ⊗ {s} -> {s}");
        assert_eq!(show(LRA::TAdd(2, 1, deg(1))), bin("Real(2, 1; 1)"));
        assert_eq!(show(LRA::TSub(2, 1, deg(1))), bin("Real(2, 1; 1)"));
        assert_eq!(show(LRA::TAdd(2, 1, deg(2))), bin("Real(2, 1; 2)"));
        assert_eq!(
            show(LRA::TId(2, 1, deg(1))),
            "Real(2, 1; 1) -> Real(2, 1; 1)"
        );
        // the guard is a value
        assert_eq!(
            show(LRA::TIte(2, 1, deg(1))),
            "Bool(1, 1) ⊗ Real(2, 1; 1) ⊗ Real(2, 1; 1) -> Real(2, 1; 1)"
        );
        // `2 * d(t)` folds to a Linear over derivatives
        let lin =
            |degree| LRA::TLinear(affine(real_tensor(2, 3), real_tensor(2, 1)), 1, deg(degree));
        assert_eq!(show(lin(1)), "Real(3, 1; 1) -> Real(2, 1; 1)");
        assert_eq!(show(lin(2)), "Real(3, 1; 2) -> Real(2, 1; 2)");
    }

    #[test]
    fn ill_formed_parameters() {
        // the parameters are checked on their creation (so the boundaries always exist)
        let t: crate::PyTensor = tch::Tensor::from_slice(&[0.0f64, 1.0]).into();
        assert!(Matrix::new(t).is_err());
        assert!(Affine::new(real_tensor(2, 3), real_tensor(2, 2)).is_err());
        assert!(NonZeroU8::new(0).is_none());
    }

    #[test]
    fn display() {
        assert_eq!(LRA::Add(2, 3).to_string(), "Add");
        assert_eq!(LRA::IdReal(2, 3).to_string(), "IdReal");
        assert_eq!(LRA::TAdd(2, 3, deg(1)).to_string(), "TAdd");
        assert_eq!(
            LRA::UninterpretedReal("f".into(), 2, 3).to_string(),
            "UninterpretedReal(f)"
        );
    }
}
