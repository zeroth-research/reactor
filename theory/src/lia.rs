/*!
# Linear integer arithmetic

Defines the theory [`LIA`] of linear integer arithmetic over matrices,
mixing integer and boolean matrices in a single signature.

A [`Sort`] value is either `Int(rows, cols)` or `Bool(rows, cols)`,
or `Zero(rows, cols)`, the sort of their (zero) derivatives.

The operations carry their parameters (the shapes of the matrices), so that each
operation has a fixed [`Boundary`]: e.g., the boundary of `Add(2, 3)` is
`Int(2, 3) ⊗ Int(2, 3) -> Int(2, 3)`. The operations that apply to several kinds of
values come in one variant per kind, e.g., [`LIA::IdInt`] and [`LIA::IdBool`].
The operations in [`LIA`] are:

- [`LIA::Int`], [`LIA::Bool`] — integer and boolean matrix literals.
- [`LIA::And`], [`LIA::Or`], [`LIA::Xor`], [`LIA::Not`]
  — boolean operations.
- [`LIA::Le`], [`LIA::Lt`], [`LIA::Ge`], [`LIA::Gt`], [`LIA::Eq`], [`LIA::Ne`]
  — pointwise integer comparisons producing a boolean matrix.
- [`LIA::IteInt`], [`LIA::IteBool`] — if-then-else: reads a boolean guard and
  two same-typed branches.
- [`LIA::Linear`]`(A, B, batch)` — the affine map `x ↦ A·x + B`, with `A` and
  `B` constant integer matrices of compatible shapes, of `x` with `batch` columns.
- [`LIA::ReLU`] — the shape-preserving rectified-linear map on integer matrices.

`LIA` implements [`Theory`]; [`Theory::check`] checks that the sorts of the
read and written wires are the domain and the codomain of the operation.

## Examples

```
use theory::Theory;
use theory::lia::{LIA, Sort};

// Wires carry a sort.
let ok = |s| Ok::<_, String>(s);

// Pointwise less-than on scalars: Int(1,1), Int(1,1) -> Bool(1,1).
let i = Sort::Int([1, 1]);
let b = Sort::Bool([1, 1]);
assert!(LIA::Lt(1, 1).check([i, i].map(ok), [b].map(ok)).is_ok());

// ReLU preserves shape and stays in the integer fragment.
let m = Sort::Int([3, 4]);
assert!(LIA::ReLU(3, 4).check([m].map(ok), [m].map(ok)).is_ok());
assert!(LIA::ReLU(3, 4).check([b].map(ok), [b].map(ok)).is_err());
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
    Int(crate::PyTensor),
    Bool(crate::PyTensor),
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
    Linear(crate::PyTensor, crate::PyTensor, usize),
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
    // discarding a value (read, nothing written): of an integer, a boolean, and a zero
    // derivative
    DiscardInt(usize, usize),
    DiscardBool(usize, usize),
    DiscardZero(usize, usize),
}

/// Prints the name of the operation, the tensors of the literals and of the affine maps,
/// and the names of the uninterpreted values (the other parameters are given by the wires)
impl fmt::Display for LIA {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name: &'static str = self.into();
        match self {
            LIA::Int(t) | LIA::Bool(t) => write!(f, "{t}"),
            LIA::Linear(a, b, _) => write!(f, "{name}({a}, {b})"),
            LIA::UninterpretedInt(x, ..) | LIA::UninterpretedBool(x, ..) => {
                write!(f, "{name}({x})")
            }
            _ => f.write_str(name),
        }
    }
}

impl Sequential for LIA {
    fn skip(range: &Self::Sort) -> Self {
        // a zero derivative cannot be copied: the check of the term with it fails
        match *range {
            Sort::Int([r, c]) | Sort::Zero([r, c]) => LIA::IdInt(r, c),
            Sort::Bool([r, c]) => LIA::IdBool(r, c),
        }
    }
}

impl Combinatorial for LIA {
    fn havoc(range: &Sort) -> Self {
        // a derivative cannot be havoced: the check of the term with `Zero` fails
        match *range {
            Sort::Int([r, c]) | Sort::Zero([r, c]) => LIA::AnyInt(r, c),
            Sort::Bool([r, c]) => LIA::AnyBool(r, c),
        }
    }
}

impl Theory for LIA {
    type Sort = Sort;
    const NAME: &'static str = "LIA";

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
            Sort::Int([r, c]) => LIA::DiscardInt(r, c),
            Sort::Bool([r, c]) => LIA::DiscardBool(r, c),
            Sort::Zero([r, c]) => LIA::DiscardZero(r, c),
        }
    }
}

impl Boundary<Sort> for LIA {
    /// # Panics
    ///
    /// If `A` of [`LIA::Linear`] is not a 2D tensor ([`Theory::check`] reports this as an
    /// error).
    fn dom(&self) -> Poly<Sort> {
        try_dom(self).unwrap_or_else(|e| panic!("{self:?}: {e}"))
    }

    /// # Panics
    ///
    /// If the tensor of a literal or `A` of [`LIA::Linear`] is not 2D, or `B` of
    /// [`LIA::Linear`] is not a column vector ([`Theory::check`] reports these as errors).
    fn cod(&self) -> Poly<Sort> {
        try_cod(self).unwrap_or_else(|e| panic!("{self:?}: {e}"))
    }
}

/// The domain of `op`: the sorts of its inputs, or an error if `op` is ill-formed
fn try_dom(op: &LIA) -> Result<Poly<Sort>, String> {
    let int = |r, c| Sort::Int([r, c]);
    let bool = |r, c| Sort::Bool([r, c]);
    Ok(match *op {
        LIA::Int(_) | LIA::Bool(_) => Poly::unit(),
        LIA::And(r, c) | LIA::Or(r, c) | LIA::Xor(r, c) => prod([bool(r, c), bool(r, c)]),
        LIA::Not(r, c) => prod([bool(r, c)]),
        LIA::Le(r, c)
        | LIA::Lt(r, c)
        | LIA::Ge(r, c)
        | LIA::Gt(r, c)
        | LIA::Eq(r, c)
        | LIA::Ne(r, c) => prod([int(r, c), int(r, c)]),
        // Y = A·X + B  where X=[in,batch], A=[out,in], B=[out,1] (or empty), Y=[out,batch]
        LIA::Linear(ref a, _, batch) => prod([int(linear_inputs(a)?, batch)]),
        // `Min` and `Max` are pointwise, as in the evaluator
        LIA::Add(r, c) | LIA::Sub(r, c) | LIA::Min(r, c) | LIA::Max(r, c) => {
            prod([int(r, c), int(r, c)])
        }
        LIA::ReLU(r, c) | LIA::Argmax(r, c) | LIA::TransposeInt(r, c) => prod([int(r, c)]),
        LIA::TransposeBool(r, c) => prod([bool(r, c)]),
        LIA::IteInt(r, c) => prod([bool(1, 1), int(r, c), int(r, c)]),
        LIA::IteBool(r, c) => prod([bool(1, 1), bool(r, c), bool(r, c)]),
        LIA::IdInt(r, c) => prod([int(r, c)]),
        LIA::IdBool(r, c) => prod([bool(r, c)]),
        LIA::UninterpretedInt(..)
        | LIA::AnyInt(..)
        | LIA::UninterpretedBool(..)
        | LIA::AnyBool(..) => Poly::unit(),
        LIA::DiscardInt(r, c) => prod([int(r, c)]),
        LIA::DiscardBool(r, c) => prod([bool(r, c)]),
        LIA::DiscardZero(r, c) => prod([Sort::Zero([r, c])]),
    })
}

/// The codomain of `op`: the sort of its output, or an error if `op` is ill-formed
fn try_cod(op: &LIA) -> Result<Poly<Sort>, String> {
    let int = |r, c| Sort::Int([r, c]);
    let bool = |r, c| Sort::Bool([r, c]);
    let sort = match *op {
        LIA::Int(ref cm) => {
            let [r, c] = shape_2d(cm)?;
            int(r, c)
        }
        LIA::Bool(ref cm) => {
            let [r, c] = shape_2d(cm)?;
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
        LIA::Linear(ref a, ref b, batch) => int(linear_outputs(a, b)?, batch),
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
        // nothing is written
        LIA::DiscardInt(..) | LIA::DiscardBool(..) | LIA::DiscardZero(..) => {
            return Ok(Poly::unit());
        }
    };
    Ok(prod([sort]))
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

    fn ok(s: Sort) -> Result<Sort, String> {
        Ok(s)
    }

    #[test]
    fn type_kind_and_shape() {
        assert!(int(2, 3).is_int() && !int(2, 3).is_bool());
        assert_eq!(int(2, 3).shape(), &[2, 3]);
        assert!(bool_t(1, 1).is_bool() && !bool_t(1, 1).is_int());
    }

    #[test]
    fn const_int_ok() {
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[0i64, 1], [2, 3]]).into();
        assert!(LIA::Int(cm).check([].map(ok), [int(2, 2)].map(ok)).is_ok());
    }

    #[test]
    fn const_int_bool_write_fails() {
        assert!(
            LIA::Int(tch::Tensor::from_slice2(&[[0i64]]).into())
                .check([].map(ok), [bool_t(1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn const_int_wrong_rows_fails() {
        assert!(
            LIA::Int(tch::Tensor::from_slice2(&[[0i64]]).into())
                .check([].map(ok), [int(2, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn const_int_with_read_fails() {
        let t = int(1, 1);
        assert!(
            LIA::Int(tch::Tensor::from_slice2(&[[0i64]]).into())
                .check([t].map(ok), [t].map(ok))
                .is_err()
        );
    }

    #[test]
    fn const_bool_ok() {
        let cm: crate::PyTensor = tch::Tensor::from_slice2(&[[true, false], [false, true]]).into();
        assert!(
            LIA::Bool(cm)
                .check([].map(ok), [bool_t(2, 2)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn const_bool_int_write_fails() {
        assert!(
            LIA::Bool(tch::Tensor::from_slice2(&[[true]]).into())
                .check([].map(ok), [int(1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn not_ok() {
        let b = bool_t(2, 3);
        assert!(LIA::Not(2, 3).check([b].map(ok), [b].map(ok)).is_ok());
    }

    #[test]
    fn not_int_input_fails() {
        let t = int(1, 1);
        assert!(LIA::Not(1, 1).check([t].map(ok), [t].map(ok)).is_err());
    }

    #[test]
    fn and_ok() {
        let b = bool_t(2, 2);
        assert!(LIA::And(2, 2).check([b, b].map(ok), [b].map(ok)).is_ok());
    }

    #[test]
    fn or_ok() {
        let b = bool_t(1, 1);
        assert!(LIA::Or(1, 1).check([b, b].map(ok), [b].map(ok)).is_ok());
    }

    #[test]
    fn xor_ok() {
        let b = bool_t(3, 1);
        assert!(LIA::Xor(3, 1).check([b, b].map(ok), [b].map(ok)).is_ok());
    }

    #[test]
    fn and_int_output_fails() {
        let b = bool_t(1, 1);
        assert!(
            LIA::And(1, 1)
                .check([b, b].map(ok), [int(1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn and_type_mismatch_fails() {
        assert!(
            LIA::And(1, 1)
                .check([bool_t(1, 1), bool_t(1, 2)].map(ok), [bool_t(1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn lt_ok() {
        assert!(
            LIA::Lt(1, 1)
                .check([int(1, 1), int(1, 1)].map(ok), [bool_t(1, 1)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn le_ok() {
        let res = LIA::Le(2, 3).check([int(2, 3), int(2, 3)].map(ok), [bool_t(2, 3)].map(ok));
        assert!(res.is_ok(), "result: {:?}", res);
    }

    #[test]
    fn eq_ok() {
        assert!(
            LIA::Eq(3, 2)
                .check([int(3, 2), int(3, 2)].map(ok), [bool_t(3, 2)].map(ok))
                .is_ok()
        );
        assert!(
            LIA::Eq(3, 2)
                .check([int(3, 3), int(3, 2)].map(ok), [bool_t(3, 2)].map(ok))
                .is_err()
        );
        assert!(
            LIA::Eq(3, 2)
                .check([int(3, 2), int(3, 2)].map(ok), [bool_t(3, 3)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn cmp_non_bool_output_fails() {
        let t = int(1, 1);
        assert!(LIA::Lt(1, 1).check([t, t].map(ok), [t].map(ok)).is_err());
    }

    #[test]
    fn cmp_input_mismatch_fails() {
        assert!(
            LIA::Eq(1, 1)
                .check([int(1, 1), int(1, 2)].map(ok), [bool_t(1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn add_ok() {
        let t = int(3, 4);
        assert!(LIA::Add(3, 4).check([t, t].map(ok), [t].map(ok)).is_ok());
    }

    #[test]
    fn add_shape_mismatch_fails() {
        assert!(
            LIA::Add(1, 2)
                .check([int(1, 2), int(2, 1)].map(ok), [int(1, 2)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn add_bool_fails() {
        let b = bool_t(1, 1);
        assert!(LIA::Add(1, 1).check([b, b].map(ok), [b].map(ok)).is_err());
    }

    #[test]
    fn relu_ok() {
        let t = int(3, 4);
        assert!(LIA::ReLU(3, 4).check([t].map(ok), [t].map(ok)).is_ok());
    }

    #[test]
    fn relu_bool_fails() {
        let b = bool_t(1, 1);
        assert!(LIA::ReLU(1, 1).check([b].map(ok), [b].map(ok)).is_err());
    }

    #[test]
    fn argmax_ok() {
        assert!(
            LIA::Argmax(3, 4)
                .check([int(3, 4)].map(ok), [int(1, 1)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn argmax_matrix_output_fails() {
        assert!(
            LIA::Argmax(3, 4)
                .check([int(3, 4)].map(ok), [int(3, 4)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn min_ok() {
        // pointwise
        let t = int(4, 1);
        assert!(LIA::Min(4, 1).check([t, t].map(ok), [t].map(ok)).is_ok());
        assert!(
            LIA::Max(4, 1)
                .check([t].map(ok), [int(1, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn linear_ok() {
        // A=[2,3] maps 3 features to 2; X=[3,4] is 3 features × 4 batch items.
        // Convention: Y = A·X  →  Y=[2,4].
        let a: crate::PyTensor =
            tch::Tensor::zeros([2, 3], (tch::Kind::Int64, tch::Device::Cpu)).into();
        let b: crate::PyTensor =
            tch::Tensor::zeros([0, 0], (tch::Kind::Int64, tch::Device::Cpu)).into();
        assert!(
            LIA::Linear(a, b, 4)
                .check([int(3, 4)].map(ok), [int(2, 4)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn linear_with_bias_ok() {
        // A=[2,3], b=[2,1] column bias, X=[3,1] single sample → Y=[2,1].
        let a: crate::PyTensor =
            tch::Tensor::zeros([2, 3], (tch::Kind::Int64, tch::Device::Cpu)).into();
        let b: crate::PyTensor =
            tch::Tensor::zeros([2, 1], (tch::Kind::Int64, tch::Device::Cpu)).into();
        assert!(
            LIA::Linear(a, b, 1)
                .check([int(3, 1)].map(ok), [int(2, 1)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn linear_dim_mismatch_fails() {
        // A=[2,3] but X has 4 rows — inner dimension mismatch.
        let a: crate::PyTensor =
            tch::Tensor::zeros([2, 3], (tch::Kind::Int64, tch::Device::Cpu)).into();
        let b: crate::PyTensor =
            tch::Tensor::zeros([0, 0], (tch::Kind::Int64, tch::Device::Cpu)).into();
        assert!(
            LIA::Linear(a, b, 1)
                .check([int(4, 1)].map(ok), [int(2, 1)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn transpose_ok() {
        assert!(
            LIA::TransposeInt(3, 4)
                .check([int(3, 4)].map(ok), [int(4, 3)].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn transpose_wrong_shape_fails() {
        assert!(
            LIA::TransposeInt(3, 4)
                .check([int(3, 4)].map(ok), [int(3, 4)].map(ok))
                .is_err()
        );
    }

    #[test]
    fn ite_ok() {
        let t = int(3, 4);
        assert!(
            LIA::IteInt(3, 4)
                .check([bool_t(1, 1), t, t].map(ok), [t].map(ok))
                .is_ok()
        );
    }

    #[test]
    fn ite_non_bool_guard_fails() {
        let t = int(1, 1);
        assert!(
            LIA::IteInt(1, 1)
                .check([t, t, t].map(ok), [t].map(ok))
                .is_err()
        );
    }

    #[test]
    fn ite_arm_mismatch_fails() {
        assert!(
            LIA::IteInt(1, 1)
                .check(
                    [bool_t(1, 1), int(1, 1), int(1, 2)].map(ok),
                    [int(1, 1)].map(ok)
                )
                .is_err()
        );
    }

    #[test]
    fn id_ok() {
        let t = int(4, 4);
        assert!(LIA::IdInt(4, 4).check([t].map(ok), [t].map(ok)).is_ok());
        let b = bool_t(4, 4);
        assert!(LIA::IdBool(4, 4).check([b].map(ok), [b].map(ok)).is_ok());
        assert!(LIA::IdInt(4, 4).check([b].map(ok), [b].map(ok)).is_err());
    }

    #[test]
    fn id_type_mismatch_fails() {
        assert!(
            LIA::IdInt(1, 1)
                .check([int(1, 1)].map(ok), [int(2, 2)].map(ok))
                .is_err()
        );
    }

    // Formats the boundary of `op` as `dom -> cod`
    fn show(op: LIA) -> String {
        format!("{} -> {}", op.dom(), op.cod())
    }

    fn int_tensor(rows: i64, cols: i64) -> crate::PyTensor {
        tch::Tensor::zeros([rows, cols], (tch::Kind::Int64, tch::Device::Cpu)).into()
    }

    #[test]
    fn boundary_ops() {
        let bin = |s| format!("{s} ⊗ {s} -> {s}");
        for op in [LIA::And(2, 3), LIA::Or(2, 3), LIA::Xor(2, 3)] {
            assert_eq!(show(op), bin("Bool(2, 3)"));
        }
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
        assert_eq!(show(LIA::Argmax(2, 3)), "Int(2, 3) -> Int(1, 1)");
        assert_eq!(show(LIA::TransposeInt(2, 3)), "Int(2, 3) -> Int(3, 2)");
        assert_eq!(show(LIA::TransposeBool(2, 3)), "Bool(2, 3) -> Bool(3, 2)");
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
        assert_eq!(
            show(LIA::UninterpretedInt("f".into(), 2, 3)),
            "I -> Int(2, 3)"
        );
        assert_eq!(show(LIA::AnyInt(2, 3)), "I -> Int(2, 3)");
        assert_eq!(show(LIA::AnyBool(2, 3)), "I -> Bool(2, 3)");
    }

    #[test]
    fn boundary_const() {
        assert_eq!(show(LIA::Int(int_tensor(2, 3))), "I -> Int(2, 3)");
        let b: crate::PyTensor = tch::Tensor::from_slice2(&[[true, false]]).into();
        assert_eq!(show(LIA::Bool(b.clone())), "I -> Bool(1, 2)");
        // the sort is given by the variant, not by the tensor
        assert_eq!(show(LIA::Int(b)), "I -> Int(1, 2)");
        assert_eq!(show(LIA::Bool(int_tensor(2, 3))), "I -> Bool(2, 3)");
        // the same boundaries
        assert_eq!(show(LIA::AnyInt(2, 3)), show(LIA::Int(int_tensor(2, 3))));
    }

    #[test]
    fn boundary_linear() {
        // A=[2,3] maps 3 features to 2
        assert_eq!(
            show(LIA::Linear(int_tensor(2, 3), int_tensor(2, 1), 4)),
            "Int(3, 4) -> Int(2, 4)"
        );
        assert_eq!(
            show(LIA::Linear(int_tensor(2, 3), int_tensor(0, 0), 1)),
            "Int(3, 1) -> Int(2, 1)"
        );
    }

    #[test]
    #[should_panic(expected = "expected a 2D tensor, got 1D")]
    fn boundary_const_not_2d() {
        let t: crate::PyTensor = tch::Tensor::from_slice(&[0i64, 1]).into();
        let _ = show(LIA::Int(t));
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

    #[test]
    fn check_errors() {
        assert_eq!(
            LIA::Lt(1, 1).check([int(1, 1), int(1, 2)].map(ok), [bool_t(1, 1)].map(ok)),
            Err("Lt(1, 1): expected `Int(1, 1) ⊗ Int(1, 1) -> Bool(1, 1)`, \
                 got `Int(1, 1) ⊗ Int(1, 2) -> Bool(1, 1)`"
                .to_string())
        );
        let b = bool_t(1, 1);
        assert_eq!(
            LIA::Not(1, 1).check([b, b].map(ok), [b].map(ok)),
            Err("Not(1, 1): expected `Bool(1, 1) -> Bool(1, 1)`, \
                 got `Bool(1, 1) ⊗ Bool(1, 1) -> Bool(1, 1)`"
                .to_string())
        );
        // errors of wires are propagated
        assert_eq!(
            LIA::Not(1, 1).check([Err("bad wire")], [b].map(Ok)),
            Err("Not(1, 1): bad wire".to_string())
        );
    }

    #[test]
    fn check_ill_formed_ops() {
        // ill-formed tensors are reported, not panicking
        let t: crate::PyTensor = tch::Tensor::from_slice(&[0i64, 1]).into();
        let res = LIA::Int(t).check([].map(ok), [int(1, 2)].map(ok));
        assert!(res.unwrap_err().ends_with("expected a 2D tensor, got 1D"));
        let res = LIA::Linear(int_tensor(2, 3), int_tensor(2, 2), 1)
            .check([int(3, 1)].map(ok), [int(2, 1)].map(ok));
        assert!(
            res.unwrap_err()
                .ends_with("`B` must be a column vector [2x1]")
        );
    }

    #[test]
    fn check_uninterpreted() {
        let f = || LIA::UninterpretedInt("f".into(), 2, 3);
        let t = int(2, 3);
        // a single write of the given sort
        assert!(f().check([].map(ok), [t].map(ok)).is_ok());
        assert!(f().check([].map(ok), [bool_t(2, 3)].map(ok)).is_err());
        assert!(f().check([t].map(ok), [].map(ok)).is_err());
        assert!(f().check([t].map(ok), [t].map(ok)).is_err());
        assert!(f().check([].map(ok), [t, t].map(ok)).is_err());
        assert!(f().check([].map(ok), [].map(ok)).is_err());
        let g = LIA::UninterpretedBool("g".into(), 1, 1);
        assert!(g.check([].map(ok), [bool_t(1, 1)].map(ok)).is_ok());
    }

    #[test]
    fn check_skip_and_havoc() {
        let (i, b) = (int(2, 3), bool_t(2, 3));
        assert!(LIA::skip(&i).check([i].map(ok), [i].map(ok)).is_ok());
        assert!(LIA::skip(&b).check([b].map(ok), [b].map(ok)).is_ok());
        assert!(LIA::havoc(&i).check([].map(ok), [i].map(ok)).is_ok());
        assert!(LIA::havoc(&b).check([].map(ok), [b].map(ok)).is_ok());
        // derivatives cannot be copied or havoced
        let z = i.tangent();
        assert!(LIA::skip(&z).check([z].map(ok), [z].map(ok)).is_err());
        assert!(LIA::havoc(&z).check([].map(ok), [z].map(ok)).is_err());
    }

    #[test]
    fn check_derivatives() {
        // the derivatives of integers and booleans are zero
        let z = int(2, 3).tangent();
        assert_eq!(z, Sort::Zero([2, 3]));
        assert_eq!(bool_t(2, 3).tangent(), z);
        assert_eq!(z.tangent(), z);
        // operations on values do not apply to derivatives
        assert!(LIA::Add(2, 3).check([z, z].map(ok), [z].map(ok)).is_err());
        assert!(LIA::Not(2, 3).check([z].map(ok), [z].map(ok)).is_err());
        assert!(LIA::AnyInt(2, 3).check([].map(ok), [z].map(ok)).is_err());
        assert!(
            LIA::IdInt(2, 3)
                .check([int(2, 3)].map(ok), [z].map(ok))
                .is_err()
        );
    }

    #[test]
    fn discard_ok() {
        // `discard` reads one wire of the given sort and writes none
        let (i, b, z) = (int(2, 1), bool_t(2, 1), int(2, 1).tangent());
        assert!(matches!(LIA::discard(&i), LIA::DiscardInt(2, 1)));
        assert!(matches!(LIA::discard(&b), LIA::DiscardBool(2, 1)));
        assert!(matches!(LIA::discard(&z), LIA::DiscardZero(2, 1)));
        for s in [i, b, z] {
            assert!(LIA::discard(&s).check([s].map(ok), [].map(ok)).is_ok());
            assert_eq!(LIA::discard(&s).cod(), Poly::unit());
        }
    }

    #[test]
    fn discard_write_or_mismatch_fails() {
        let (i, b) = (int(2, 1), bool_t(2, 1));
        let discard = LIA::discard(&i);
        assert!(discard.check([i].map(ok), [i].map(ok)).is_err());
        assert!(discard.check([].map(ok), [].map(ok)).is_err());
        assert!(discard.check([b].map(ok), [].map(ok)).is_err());
        assert!(discard.check([int(1, 1)].map(ok), [].map(ok)).is_err());
    }
}
