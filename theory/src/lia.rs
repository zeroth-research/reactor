/*!
# Linear integer arithmetic

Defines the theory [`LIA`] of linear integer arithmetic over matrices,
mixing integer and boolean matrices in a single signature.

A [`Sort`] value is either `Int(rows, cols)` or `Bool(rows, cols)`,
or `Zero(rows, cols)`, the sort of their (zero) derivatives.

The operations carry their parameters (the shapes of the matrices), so that each
operation has a fixed [`Boundary`]: e.g., the boundary of `Add(2, 3)` is
`Int(2, 3) ⊗ Int(2, 3) -> Int(2, 3)`. The parameters that can be ill-formed are checked
on their creation (the tensors of the literals are [`Matrix`]es, and the affine maps are
[`Affine`]), so the boundary of every operation exists. The operations that apply to several kinds of
values come in one variant per kind, e.g., [`LIA::IdInt`] and [`LIA::IdBool`].
The operations in [`LIA`] are:

- [`LIA::Int`], [`LIA::Bool`] — integer and boolean matrix literals.
- [`LIA::And`], [`LIA::Or`], [`LIA::Xor`], [`LIA::Not`]
  — boolean operations.
- [`LIA::Le`], [`LIA::Lt`], [`LIA::Ge`], [`LIA::Gt`], [`LIA::Eq`], [`LIA::Ne`]
  — pointwise integer comparisons producing a boolean matrix.
- [`LIA::IteInt`], [`LIA::IteBool`] — if-then-else: reads a boolean guard and
  two same-typed branches.
- [`LIA::Linear`]`(affine, batch)` — the affine map `x ↦ A·x + B` (see [`Affine`]),
  with `A` and `B` constant integer matrices, of `x` with `batch` columns.
- [`LIA::ReLU`] — the shape-preserving rectified-linear map on integer matrices.

An operation applies to the wires whose sorts are its boundary: the read wires
have the sorts of its domain, and the written wires the sorts of its codomain.

## Examples

```
use base::boundary::Boundary;
use theory::lia::LIA;

let boundary = |op: LIA| format!("{} -> {}", op.dom(), op.cod());

// Pointwise less-than on scalars.
assert_eq!(boundary(LIA::Lt(1, 1)), "Int(1, 1) ⊗ Int(1, 1) -> Bool(1, 1)");

// ReLU preserves shape and stays in the integer fragment.
assert_eq!(boundary(LIA::ReLU(3, 4)), "Int(3, 4) -> Int(3, 4)");
```
*/

use crate::*;
use base::boundary::{Boundary, Mono, Poly};
use base::tangent::Tangent;
#[cfg(feature = "pyo3")]
use pyo3::pyclass;
use std::fmt;

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub enum Sort {
    Int([usize; 2]),
    Bool([usize; 2]),
    // the tangent of `Int` and `Bool` (they have only the zero derivative)
    Zero([usize; 2]),
}

impl Sort {
    pub fn is_bool(&self) -> bool {
        matches!(self, Sort::Bool(..))
    }

    pub fn is_int(&self) -> bool {
        matches!(self, Sort::Int(..))
    }

    pub fn shape(&self) -> &[usize; 2] {
        match self {
            Sort::Bool(shape) | Sort::Int(shape) | Sort::Zero(shape) => shape,
        }
    }
}

impl fmt::Display for Sort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sort::Int([i, j]) => write!(f, "Int({i}, {j})"),
            Sort::Bool([i, j]) => write!(f, "Bool({i}, {j})"),
            Sort::Zero([i, j]) => write!(f, "Zero({i}, {j})"),
        }
    }
}

impl Tangent for Sort {
    fn tangent(&self) -> Self {
        Sort::Zero(*self.shape())
    }
}

/// The operations of linear integer arithmetic. The parameters `(rows, cols)` are the
/// shape of the matrices the operation applies to (e.g., of its operands).
#[derive(Clone, Debug, strum::IntoStaticStr)]
#[cfg_attr(feature = "pyo3", pyclass(frozen))]
pub enum LIA {
    // integer and boolean matrix literals (of the shape of the tensor)
    Int(Matrix),
    Bool(Matrix),
    // boolean operations
    And(usize, usize),
    Or(usize, usize),
    Xor(usize, usize),
    Not(usize, usize),
    // integer comparisons
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
    // XXX: should these be in LIA?
    ReLU(usize, usize),
    Argmax(usize, usize),
    Min(usize, usize),
    Max(usize, usize),
    // matrix operations
    TransposeInt(usize, usize),
    TransposeBool(usize, usize),
    // control flow
    IteInt(usize, usize),
    IteBool(usize, usize),
    IdInt(usize, usize),
    IdBool(usize, usize),
    // an uninterpreted value (written, used by the builder for untraced calls)
    UninterpretedInt(String, usize, usize),
    UninterpretedBool(String, usize, usize),
    AnyInt(usize, usize),
    AnyBool(usize, usize),
}

/// Prints the name of the operation, the tensors of the literals and of the affine maps,
/// and the names of the uninterpreted values (the other parameters are given by the wires)
impl fmt::Display for LIA {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name: &'static str = self.into();
        match self {
            LIA::Int(t) | LIA::Bool(t) => write!(f, "{t}"),
            LIA::Linear(m, _) => write!(f, "{name}({}, {})", m.a(), m.b()),
            LIA::UninterpretedInt(x, ..) | LIA::UninterpretedBool(x, ..) => {
                write!(f, "{name}({x})")
            }
            _ => f.write_str(name),
        }
    }
}

impl Sequential<Sort> for LIA {
    fn skip(range: &Sort) -> Self {
        // a zero derivative cannot be copied: the check of the term with it fails
        match *range {
            Sort::Int([r, c]) | Sort::Zero([r, c]) => LIA::IdInt(r, c),
            Sort::Bool([r, c]) => LIA::IdBool(r, c),
        }
    }
}

impl Combinatorial<Sort> for LIA {
    fn havoc(range: &Sort) -> Self {
        // a derivative cannot be havoced: the check of the term with `Zero` fails
        match *range {
            Sort::Int([r, c]) | Sort::Zero([r, c]) => LIA::AnyInt(r, c),
            Sort::Bool([r, c]) => LIA::AnyBool(r, c),
        }
    }
}

impl Boundary<Sort> for LIA {
    fn dom(&self) -> Poly<Sort> {
        let int = |r, c| Sort::Int([r, c]);
        let bool = |r, c| Sort::Bool([r, c]);
        let sorts = match *self {
            LIA::Int(_) | LIA::Bool(_) => vec![],
            LIA::And(r, c) | LIA::Or(r, c) | LIA::Xor(r, c) => vec![bool(r, c), bool(r, c)],
            LIA::Not(r, c) => vec![bool(r, c)],
            LIA::Le(r, c)
            | LIA::Lt(r, c)
            | LIA::Ge(r, c)
            | LIA::Gt(r, c)
            | LIA::Eq(r, c)
            | LIA::Ne(r, c) => vec![int(r, c), int(r, c)],
            // Y = A·X + B  where X=[in,batch], A=[out,in], B=[out,1] (or empty), Y=[out,batch]
            LIA::Linear(ref m, batch) => vec![int(m.inputs(), batch)],
            // `Min` and `Max` are pointwise, as in the evaluator
            LIA::Add(r, c) | LIA::Sub(r, c) | LIA::Min(r, c) | LIA::Max(r, c) => {
                vec![int(r, c), int(r, c)]
            }
            LIA::ReLU(r, c) | LIA::Argmax(r, c) | LIA::TransposeInt(r, c) => vec![int(r, c)],
            LIA::TransposeBool(r, c) => vec![bool(r, c)],
            LIA::IteInt(r, c) => vec![bool(1, 1), int(r, c), int(r, c)],
            LIA::IteBool(r, c) => vec![bool(1, 1), bool(r, c), bool(r, c)],
            LIA::IdInt(r, c) => vec![int(r, c)],
            LIA::IdBool(r, c) => vec![bool(r, c)],
            LIA::UninterpretedInt(..)
            | LIA::AnyInt(..)
            | LIA::UninterpretedBool(..)
            | LIA::AnyBool(..) => vec![],
        };
        sorts.into_iter().product::<Mono<Sort>>().into()
    }

    fn cod(&self) -> Poly<Sort> {
        let int = |r, c| Sort::Int([r, c]);
        let bool = |r, c| Sort::Bool([r, c]);
        let sort = match *self {
            LIA::Int(ref m) => {
                let [r, c] = m.shape();
                int(r, c)
            }
            LIA::Bool(ref m) => {
                let [r, c] = m.shape();
                bool(r, c)
            }
            LIA::And(r, c) | LIA::Or(r, c) | LIA::Xor(r, c) | LIA::Not(r, c) => bool(r, c),
            LIA::Le(r, c)
            | LIA::Lt(r, c)
            | LIA::Ge(r, c)
            | LIA::Gt(r, c)
            | LIA::Eq(r, c)
            | LIA::Ne(r, c) => bool(r, c),
            // Y = A·X + B  where X=[in,batch], A=[out,in], B=[out,1] (or empty), Y=[out,batch]
            LIA::Linear(ref m, batch) => int(m.outputs(), batch),
            // `Min` and `Max` are pointwise, as in the evaluator
            LIA::Add(r, c) | LIA::Sub(r, c) | LIA::Min(r, c) | LIA::Max(r, c) | LIA::ReLU(r, c) => {
                int(r, c)
            }
            // the index into the flattened matrix, as in the evaluator and the Python builder
            LIA::Argmax(..) => int(1, 1),
            LIA::TransposeInt(r, c) => int(c, r),
            LIA::TransposeBool(r, c) => bool(c, r),
            LIA::IteInt(r, c) | LIA::IdInt(r, c) => int(r, c),
            LIA::IteBool(r, c) | LIA::IdBool(r, c) => bool(r, c),
            LIA::UninterpretedInt(_, r, c) | LIA::AnyInt(r, c) => int(r, c),
            LIA::UninterpretedBool(_, r, c) | LIA::AnyBool(r, c) => bool(r, c),
        };
        [sort].into_iter().product::<Mono<Sort>>().into()
    }
}

#[cfg(all(test, feature = "torch"))]
mod tests {
    use super::*;

    fn int(r: usize, c: usize) -> Sort {
        Sort::Int([r, c])
    }

    fn bool_t(r: usize, c: usize) -> Sort {
        Sort::Bool([r, c])
    }

    fn int_tensor(rows: i64, cols: i64) -> crate::PyTensor {
        tch::Tensor::zeros([rows, cols], (tch::Kind::Int64, tch::Device::Cpu)).into()
    }

    fn matrix(t: crate::PyTensor) -> Matrix {
        Matrix::new(t).unwrap()
    }

    fn affine(a: crate::PyTensor, b: crate::PyTensor) -> Affine {
        Affine::new(a, b).unwrap()
    }

    // Formats the boundary of `op` as `dom -> cod`
    fn show(op: LIA) -> String {
        format!("{} -> {}", op.dom(), op.cod())
    }

    #[test]
    fn type_kind_and_shape() {
        assert!(int(2, 3).is_int() && !int(2, 3).is_bool());
        assert_eq!(int(2, 3).shape(), &[2, 3]);
        assert!(bool_t(1, 1).is_bool() && !bool_t(1, 1).is_int());
    }

    #[test]
    fn boundary_ops() {
        let bin = |s| format!("{s} ⊗ {s} -> {s}");
        for op in [LIA::And(2, 3), LIA::Or(2, 3), LIA::Xor(2, 3)] {
            assert_eq!(show(op), bin("Bool(2, 3)"));
        }
        // `Min` and `Max` are pointwise
        for op in [
            LIA::Add(2, 3),
            LIA::Sub(2, 3),
            LIA::Min(2, 3),
            LIA::Max(2, 3),
        ] {
            assert_eq!(show(op), bin("Int(2, 3)"));
        }
        for op in [
            LIA::Le(2, 3),
            LIA::Lt(2, 3),
            LIA::Ge(2, 3),
            LIA::Gt(2, 3),
            LIA::Eq(2, 3),
            LIA::Ne(2, 3),
        ] {
            assert_eq!(show(op), "Int(2, 3) ⊗ Int(2, 3) -> Bool(2, 3)");
        }
        assert_eq!(show(LIA::Not(2, 3)), "Bool(2, 3) -> Bool(2, 3)");
        assert_eq!(show(LIA::ReLU(2, 3)), "Int(2, 3) -> Int(2, 3)");
        // the index into the flattened matrix
        assert_eq!(show(LIA::Argmax(2, 3)), "Int(2, 3) -> Int(1, 1)");
        assert_eq!(show(LIA::TransposeInt(2, 3)), "Int(2, 3) -> Int(3, 2)");
        assert_eq!(show(LIA::TransposeBool(2, 3)), "Bool(2, 3) -> Bool(3, 2)");
        // a single boolean guard and two branches of the sort of the result
        assert_eq!(
            show(LIA::IteInt(2, 3)),
            "Bool(1, 1) ⊗ Int(2, 3) ⊗ Int(2, 3) -> Int(2, 3)"
        );
        assert_eq!(
            show(LIA::IteBool(2, 3)),
            "Bool(1, 1) ⊗ Bool(2, 3) ⊗ Bool(2, 3) -> Bool(2, 3)"
        );
        assert_eq!(show(LIA::IdInt(2, 3)), "Int(2, 3) -> Int(2, 3)");
        assert_eq!(show(LIA::IdBool(2, 3)), "Bool(2, 3) -> Bool(2, 3)");
        // a single write of the given sort
        assert_eq!(
            show(LIA::UninterpretedInt("f".into(), 2, 3)),
            "I -> Int(2, 3)"
        );
        assert_eq!(
            show(LIA::UninterpretedBool("g".into(), 2, 3)),
            "I -> Bool(2, 3)"
        );
        assert_eq!(show(LIA::AnyInt(2, 3)), "I -> Int(2, 3)");
        assert_eq!(show(LIA::AnyBool(2, 3)), "I -> Bool(2, 3)");
    }

    #[test]
    fn boundary_const() {
        // a literal writes a matrix of the shape of its tensor
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[0i64, 1], [2, 3]]).into();
        assert_eq!(show(LIA::Int(matrix(cm))), "I -> Int(2, 2)");
        assert_eq!(show(LIA::Int(matrix(int_tensor(2, 3)))), "I -> Int(2, 3)");
        let b: crate::PyTensor = tch::Tensor::from_slice2(&[[true, false]]).into();
        assert_eq!(show(LIA::Bool(matrix(b.clone()))), "I -> Bool(1, 2)");
        // the sort is given by the variant, not by the tensor
        assert_eq!(show(LIA::Int(matrix(b))), "I -> Int(1, 2)");
        assert_eq!(show(LIA::Bool(matrix(int_tensor(2, 3)))), "I -> Bool(2, 3)");
        // the same boundaries
        assert_eq!(
            show(LIA::AnyInt(2, 3)),
            show(LIA::Int(matrix(int_tensor(2, 3))))
        );
    }

    #[test]
    fn boundary_linear() {
        // A=[2,3] maps 3 features to 2; X=[3,4] is 3 features × 4 batch items.
        // Convention: Y = A·X + B  →  Y=[2,4].
        assert_eq!(
            show(LIA::Linear(affine(int_tensor(2, 3), int_tensor(2, 1)), 4)),
            "Int(3, 4) -> Int(2, 4)"
        );
        // A=[2,3] without a bias, X=[3,1] single sample → Y=[2,1].
        assert_eq!(
            show(LIA::Linear(affine(int_tensor(2, 3), int_tensor(0, 0)), 1)),
            "Int(3, 1) -> Int(2, 1)"
        );
    }

    #[test]
    fn skip_and_havoc() {
        let (i, b) = (int(2, 3), bool_t(2, 3));
        assert_eq!(show(LIA::skip(&i)), "Int(2, 3) -> Int(2, 3)");
        assert_eq!(show(LIA::skip(&b)), "Bool(2, 3) -> Bool(2, 3)");
        assert_eq!(show(LIA::havoc(&i)), "I -> Int(2, 3)");
        assert_eq!(show(LIA::havoc(&b)), "I -> Bool(2, 3)");
        // derivatives cannot be copied or havoced: `skip` and `havoc` of them are on values
        let z = i.tangent();
        assert_eq!(show(LIA::skip(&z)), "Int(2, 3) -> Int(2, 3)");
        assert_eq!(show(LIA::havoc(&z)), "I -> Int(2, 3)");
    }

    #[test]
    fn derivatives() {
        // the derivatives of integers and booleans are zero; the operations of LIA are on
        // values only (see `boundary_ops`)
        let z = int(2, 3).tangent();
        assert_eq!(z, Sort::Zero([2, 3]));
        assert_eq!(bool_t(2, 3).tangent(), z);
        assert_eq!(z.tangent(), z);
    }

    #[test]
    fn ill_formed_parameters() {
        // the parameters are checked on their creation (so the boundaries always exist)
        let t: crate::PyTensor = tch::Tensor::from_slice(&[0i64, 1]).into();
        assert!(Matrix::new(t).is_err());
        assert!(Affine::new(int_tensor(2, 3), int_tensor(2, 2)).is_err());
    }

    #[test]
    fn display() {
        assert_eq!(LIA::Add(2, 3).to_string(), "Add");
        assert_eq!(LIA::IdBool(2, 3).to_string(), "IdBool");
        assert_eq!(
            LIA::UninterpretedInt("f".into(), 2, 3).to_string(),
            "UninterpretedInt(f)"
        );
    }
}
