/*!
# Floating-point numbers

Defines one type per operation on the sort [`Float`] of matrices of IEEE 754
floating-point numbers, of any format `(exponent, significand)`, together with
[`RoundingMode`], the rounding modes that the rounding operations carry, and the
local sort [`Sort`] of the theory, holding floats and the booleans that
comparisons write. Each operation is an operation of the SMT-LIB2 theory
`FloatingPoint`, lifted pointwise to matrices: it applies the SMT-LIB2
operation independently to each entry, with the same semantics, and its
documentation gives the SMT-LIB2 term it computes on one entry.

Every operation carries enough information to compute its [`Boundary`]: the
sort of the floats it reads or writes and, if it rounds, its rounding mode,
which SMT-LIB2 passes as a first argument and here is a parameter of the
operation, fixed when it is built. The operations are those that read and
write floats alone: the conversions from and to reals and bit-vectors are not
included.

| Operation | SMT-LIB2 |
|---|---|
| [`Const`] | `(fp s e m)` |
| [`Ite`] | `ite` |
| [`Convert`] | `((_ to_fp eb sb) rm x)` |
| [`PlusInfinity`] | `(_ +oo eb sb)` |
| [`MinusInfinity`] | `(_ -oo eb sb)` |
| [`PlusZero`] | `(_ +zero eb sb)` |
| [`MinusZero`] | `(_ -zero eb sb)` |
| [`NaN`] | `(_ NaN eb sb)` |
| [`Abs`] | `(fp.abs x)` |
| [`Neg`] | `(fp.neg x)` |
| [`Sqrt`] | `(fp.sqrt rm x)` |
| [`RoundToIntegral`] | `(fp.roundToIntegral rm x)` |
| [`Add`] | `(fp.add rm x y)` |
| [`Sub`] | `(fp.sub rm x y)` |
| [`Mul`] | `(fp.mul rm x y)` |
| [`Div`] | `(fp.div rm x y)` |
| [`Fma`] | `(fp.fma rm x y z)` |
| [`Rem`] | `(fp.rem x y)` |
| [`Min`] | `(fp.min x y)` |
| [`Max`] | `(fp.max x y)` |
| [`Le`] | `(fp.leq x y)` |
| [`Lt`] | `(fp.lt x y)` |
| [`Ge`] | `(fp.geq x y)` |
| [`Gt`] | `(fp.gt x y)` |
| [`Eq`](struct@Eq) | `(fp.eq x y)` |
| [`IsNormal`] | `(fp.isNormal x)` |
| [`IsSubnormal`] | `(fp.isSubnormal x)` |
| [`IsZero`] | `(fp.isZero x)` |
| [`IsInfinite`] | `(fp.isInfinite x)` |
| [`IsNaN`] | `(fp.isNaN x)` |
| [`IsNegative`] | `(fp.isNegative x)` |
| [`IsPositive`] | `(fp.isPositive x)` |

The only equality is [`Eq`](struct@Eq), `fp.eq`, which follows IEEE 754: the
equality `=` and `distinct` of the SMT-LIB2 Core theory, which tell `-0`
from `+0` and take NaN to be equal to itself, are not included. The
comparisons and the predicates `fp.is…` write booleans, as in SMT-LIB2.
*/

use super::sort::{Bool, Float};
use crate::Tensor;
use base::boundary::{Boundary, Poly};
use base::mono;
use derive_more::From;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use std::fmt;

/// The sorts of the theory, as in the signature of SMT-LIB2's FloatingPoint
/// on floats alone: floats, and the booleans of the Core theory, which
/// comparisons write, together with [`Zero`](crate::tensor::sort::Zero), the
/// tangent of both. It is the subset
/// [`tensor::sort::Floats`](crate::tensor::sort::Floats) of the sorts of all
/// theories.
pub use super::sort::Floats as Sort;

/// The rounding modes of IEEE 754, which the rounding operations carry, named
/// as SMT-LIB2 abbreviates them.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(
        frozen,
        from_py_object,
        eq,
        eq_int,
        str,
        module = "reactor.tensor.float"
    )
)]
pub enum RoundingMode {
    /// To the nearest, ties to the even significand: `roundNearestTiesToEven`.
    RNE,
    /// To the nearest, ties away from zero: `roundNearestTiesToAway`.
    RNA,
    /// Toward positive infinity: `roundTowardPositive`.
    RTP,
    /// Toward negative infinity: `roundTowardNegative`.
    RTN,
    /// Toward zero: `roundTowardZero`.
    RTZ,
}

impl fmt::Display for RoundingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            RoundingMode::RNE => "RNE",
            RoundingMode::RNA => "RNA",
            RoundingMode::RTP => "RTP",
            RoundingMode::RTN => "RTN",
            RoundingMode::RTZ => "RTZ",
        };
        f.write_str(name)
    }
}

/// A constant matrix literal of floats of the sort `sort`, held in a tensor of
/// type `T`.
///
/// SMT-LIB2: a literal `(fp s e m)`.
#[derive(Debug)]
pub struct Const<T> {
    tensor: T,
    sort: Float,
}

impl<T: Tensor> Const<T> {
    /// The constant held in `tensor`, writing the floats of `cod`; fails
    /// unless `tensor` is of its shape.
    pub fn with_cod(tensor: T, cod: [Float; 1]) -> Result<Self, crate::Error> {
        let [sort] = cod;
        if tensor.shape() != sort.shape() {
            return Err(crate::tensor::unexpected_cod(
                &format!("1 sort of the shape {:?} of the constant", tensor.shape()),
                &cod,
            ));
        }
        Ok(Const { tensor, sort })
    }
}

impl<T> Const<T> {
    /// The tensor holding the constant.
    pub fn tensor(&self) -> &T {
        &self.tensor
    }
}

impl<T: Tensor> Boundary<Sort> for Const<T> {
    fn dom(&self) -> Poly<Sort> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![self.sort.into()].into()
    }
}

impl<T: fmt::Display> fmt::Display for Const<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.tensor)
    }
}

/// The pointwise choice `if c then x else y`, by a boolean matrix `c`, between
/// floats of the sort `sort`.
///
/// SMT-LIB2: `(ite c x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.float")
)]
pub struct Ite {
    sort: Float,
}

impl Ite {
    /// The choice reading the boolean condition and the branches of `dom`;
    /// fails unless the branches are of the same sort, and the condition of
    /// their shape.
    pub fn with_dom(dom: [Sort; 3]) -> Result<Self, crate::Error> {
        match dom {
            [Sort::Bool(c), Sort::Float(x), Sort::Float(y)] if x == y && c.shape() == x.shape() => {
                Ok(Ite { sort: x })
            }
            _ => Err(crate::tensor::unexpected_dom(
                "a condition Bool and two branches Float of its shape",
                &dom,
            )),
        }
    }

    /// The choice writing the floats of `cod`.
    pub fn with_cod(cod: [Float; 1]) -> Result<Self, crate::Error> {
        let [sort] = cod;
        Ok(Ite { sort })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Ite {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Sort; 3], cod: Option<Float>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }
}

impl Boundary<Sort> for Ite {
    fn dom(&self) -> Poly<Sort> {
        mono![
            Bool::new(self.sort.shape()).into(),
            self.sort.into(),
            self.sort.into()
        ]
        .into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![self.sort.into()].into()
    }
}

impl fmt::Display for Ite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ite")
    }
}

/// The pointwise conversion of floats of the sort `from` to the format
/// `(exponent, significand)` of the sort `to`, of the same shape, in the
/// rounding mode `rm`: IEEE 754's `convertFormat`.
///
/// SMT-LIB2: `((_ to_fp eb sb) rm x)`, of a float `x`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.float")
)]
pub struct Convert {
    rm: RoundingMode,
    from: Float,
    to: Float,
}

impl Convert {
    /// The conversion to the format `(exponent, significand)`, in the rounding
    /// mode `rm`, of the floats of `dom`; fails unless the format is one of
    /// [`Float`].
    pub fn with_dom(
        rm: RoundingMode,
        exponent: usize,
        significand: usize,
        dom: [Float; 1],
    ) -> Result<Self, crate::Error> {
        let [from] = dom;
        let to = Float::new(exponent, significand, from.shape())?;
        Ok(Convert { rm, from, to })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Convert {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (rm, exponent, significand, /, dom, cod = None))]
    fn py_new(
        rm: RoundingMode,
        exponent: usize,
        significand: usize,
        dom: Float,
        cod: Option<Float>,
    ) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom(rm, exponent, significand, [dom]), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (
        &'static str,
        &'static str,
        &'static str,
        &'static str,
        &'static str,
    ) {
        ("rm", "exponent", "significand", "dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }

    /// The rounding mode.
    #[cfg_attr(feature = "pyo3", getter)]
    pub fn rm(&self) -> RoundingMode {
        self.rm
    }

    /// The bits of exponent of the floats it writes.
    #[cfg_attr(feature = "pyo3", getter)]
    pub fn exponent(&self) -> usize {
        self.to.exponent()
    }

    /// The bits of significand of the floats it writes.
    #[cfg_attr(feature = "pyo3", getter)]
    pub fn significand(&self) -> usize {
        self.to.significand()
    }
}

impl Boundary<Sort> for Convert {
    fn dom(&self) -> Poly<Sort> {
        mono![self.from.into()].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![self.to.into()].into()
    }
}

impl fmt::Display for Convert {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Convert({}, {}, {})",
            self.rm,
            self.to.exponent(),
            self.to.significand()
        )
    }
}

/// The positive infinity of every entry, of floats of the sort `sort`.
///
/// SMT-LIB2: `(_ +oo eb sb)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.float")
)]
pub struct PlusInfinity {
    sort: Float,
}

/// The negative infinity of every entry, of floats of the sort `sort`.
///
/// SMT-LIB2: `(_ -oo eb sb)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.float")
)]
pub struct MinusInfinity {
    sort: Float,
}

/// The positive zero of every entry, of floats of the sort `sort`.
///
/// SMT-LIB2: `(_ +zero eb sb)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.float")
)]
pub struct PlusZero {
    sort: Float,
}

/// The negative zero of every entry, of floats of the sort `sort`.
///
/// SMT-LIB2: `(_ -zero eb sb)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.float")
)]
pub struct MinusZero {
    sort: Float,
}

/// Not a number of every entry, of floats of the sort `sort`.
///
/// SMT-LIB2: `(_ NaN eb sb)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.float")
)]
pub struct NaN {
    sort: Float,
}

/// Pointwise absolute value `|x|`, of floats of the sort `sort`.
///
/// SMT-LIB2: `(fp.abs x)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.float")
)]
pub struct Abs {
    sort: Float,
}

/// Pointwise negation `-x`, of floats of the sort `sort`.
///
/// SMT-LIB2: `(fp.neg x)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.float")
)]
pub struct Neg {
    sort: Float,
}

/// Pointwise square root `√x`, rounded, in the rounding mode `rm`, of floats of the sort `sort`.
///
/// SMT-LIB2: `(fp.sqrt rm x)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.float")
)]
pub struct Sqrt {
    rm: RoundingMode,
    sort: Float,
}

/// Pointwise rounding of `x` to an integral value, in the rounding mode `rm`, of floats of the sort `sort`.
///
/// SMT-LIB2: `(fp.roundToIntegral rm x)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.float")
)]
pub struct RoundToIntegral {
    rm: RoundingMode,
    sort: Float,
}

/// Pointwise addition `x + y`, rounded, in the rounding mode `rm`, of floats of the sort `sort`.
///
/// SMT-LIB2: `(fp.add rm x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.float")
)]
pub struct Add {
    rm: RoundingMode,
    sort: Float,
}

/// Pointwise subtraction `x - y`, rounded, in the rounding mode `rm`, of floats of the sort `sort`.
///
/// SMT-LIB2: `(fp.sub rm x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.float")
)]
pub struct Sub {
    rm: RoundingMode,
    sort: Float,
}

/// Pointwise multiplication `x · y`, rounded, in the rounding mode `rm`, of floats of the sort `sort`.
///
/// SMT-LIB2: `(fp.mul rm x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.float")
)]
pub struct Mul {
    rm: RoundingMode,
    sort: Float,
}

/// Pointwise division `x / y`, rounded, in the rounding mode `rm`, of floats of the sort `sort`.
///
/// SMT-LIB2: `(fp.div rm x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.float")
)]
pub struct Div {
    rm: RoundingMode,
    sort: Float,
}

/// Pointwise fused multiplication and addition `x · y + z`, rounded once, in the rounding mode `rm`, of floats of the sort `sort`.
///
/// SMT-LIB2: `(fp.fma rm x y z)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.float")
)]
pub struct Fma {
    rm: RoundingMode,
    sort: Float,
}

/// Pointwise remainder `x - y · n`, where `n` is the integer nearest to `x / y`, ties to even, of floats of the sort `sort`.
///
/// SMT-LIB2: `(fp.rem x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.float")
)]
pub struct Rem {
    sort: Float,
}

/// Pointwise minimum, of floats of the sort `sort`.
///
/// SMT-LIB2: `(fp.min x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.float")
)]
pub struct Min {
    sort: Float,
}

/// Pointwise maximum, of floats of the sort `sort`.
///
/// SMT-LIB2: `(fp.max x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.float")
)]
pub struct Max {
    sort: Float,
}

/// Pointwise comparison `x ≤ y`, false when either is NaN, of floats of the sort `sort`. Writes booleans.
///
/// SMT-LIB2: `(fp.leq x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.float")
)]
pub struct Le {
    sort: Float,
}

/// Pointwise comparison `x < y`, false when either is NaN, of floats of the sort `sort`. Writes booleans.
///
/// SMT-LIB2: `(fp.lt x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.float")
)]
pub struct Lt {
    sort: Float,
}

/// Pointwise comparison `x ≥ y`, false when either is NaN, of floats of the sort `sort`. Writes booleans.
///
/// SMT-LIB2: `(fp.geq x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.float")
)]
pub struct Ge {
    sort: Float,
}

/// Pointwise comparison `x > y`, false when either is NaN, of floats of the sort `sort`. Writes booleans.
///
/// SMT-LIB2: `(fp.gt x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.float")
)]
pub struct Gt {
    sort: Float,
}

/// Pointwise IEEE 754 equality `x = y`: `-0` and `+0` are equal, and NaN is equal to nothing, itself included, of floats of the sort `sort`. Writes booleans.
///
/// SMT-LIB2: `(fp.eq x y)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.float")
)]
pub struct Eq {
    sort: Float,
}

/// Whether `x` is a normal number, of floats of the sort `sort`. Writes booleans.
///
/// SMT-LIB2: `(fp.isNormal x)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.float")
)]
pub struct IsNormal {
    sort: Float,
}

/// Whether `x` is a subnormal number, of floats of the sort `sort`. Writes booleans.
///
/// SMT-LIB2: `(fp.isSubnormal x)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.float")
)]
pub struct IsSubnormal {
    sort: Float,
}

/// Whether `x` is a zero, of either sign, of floats of the sort `sort`. Writes booleans.
///
/// SMT-LIB2: `(fp.isZero x)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.float")
)]
pub struct IsZero {
    sort: Float,
}

/// Whether `x` is an infinity, of either sign, of floats of the sort `sort`. Writes booleans.
///
/// SMT-LIB2: `(fp.isInfinite x)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.float")
)]
pub struct IsInfinite {
    sort: Float,
}

/// Whether `x` is NaN, of floats of the sort `sort`. Writes booleans.
///
/// SMT-LIB2: `(fp.isNaN x)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.float")
)]
pub struct IsNaN {
    sort: Float,
}

/// Whether `x` is negative, `-0` included, NaN excluded, of floats of the sort `sort`. Writes booleans.
///
/// SMT-LIB2: `(fp.isNegative x)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.float")
)]
pub struct IsNegative {
    sort: Float,
}

/// Whether `x` is positive, `+0` included, NaN excluded, of floats of the sort `sort`. Writes booleans.
///
/// SMT-LIB2: `(fp.isPositive x)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor.float")
)]
pub struct IsPositive {
    sort: Float,
}

/// The operations of the theory of floating-point numbers, with their
/// boundary in [`Sort`]: those of the SMT-LIB2 theory `FloatingPoint` that
/// read and write floats alone.
#[derive(Debug, From)]
pub enum FP<T> {
    Const(Const<T>),
    Ite(Ite),
    Convert(Convert),
    PlusInfinity(PlusInfinity),
    MinusInfinity(MinusInfinity),
    PlusZero(PlusZero),
    MinusZero(MinusZero),
    NaN(NaN),
    Abs(Abs),
    Neg(Neg),
    Sqrt(Sqrt),
    RoundToIntegral(RoundToIntegral),
    Add(Add),
    Sub(Sub),
    Mul(Mul),
    Div(Div),
    Fma(Fma),
    Rem(Rem),
    Min(Min),
    Max(Max),
    Le(Le),
    Lt(Lt),
    Ge(Ge),
    Gt(Gt),
    Eq(Eq),
    IsNormal(IsNormal),
    IsSubnormal(IsSubnormal),
    IsZero(IsZero),
    IsInfinite(IsInfinite),
    IsNaN(IsNaN),
    IsNegative(IsNegative),
    IsPositive(IsPositive),
}

impl<T> Clone for FP<T>
where
    Const<T>: Clone,
{
    fn clone(&self) -> Self {
        match self {
            FP::Const(op) => FP::Const(op.clone()),
            FP::Ite(op) => FP::Ite(*op),
            FP::Convert(op) => FP::Convert(*op),
            FP::PlusInfinity(op) => FP::PlusInfinity(*op),
            FP::MinusInfinity(op) => FP::MinusInfinity(*op),
            FP::PlusZero(op) => FP::PlusZero(*op),
            FP::MinusZero(op) => FP::MinusZero(*op),
            FP::NaN(op) => FP::NaN(*op),
            FP::Abs(op) => FP::Abs(*op),
            FP::Neg(op) => FP::Neg(*op),
            FP::Sqrt(op) => FP::Sqrt(*op),
            FP::RoundToIntegral(op) => FP::RoundToIntegral(*op),
            FP::Add(op) => FP::Add(*op),
            FP::Sub(op) => FP::Sub(*op),
            FP::Mul(op) => FP::Mul(*op),
            FP::Div(op) => FP::Div(*op),
            FP::Fma(op) => FP::Fma(*op),
            FP::Rem(op) => FP::Rem(*op),
            FP::Min(op) => FP::Min(*op),
            FP::Max(op) => FP::Max(*op),
            FP::Le(op) => FP::Le(*op),
            FP::Lt(op) => FP::Lt(*op),
            FP::Ge(op) => FP::Ge(*op),
            FP::Gt(op) => FP::Gt(*op),
            FP::Eq(op) => FP::Eq(*op),
            FP::IsNormal(op) => FP::IsNormal(*op),
            FP::IsSubnormal(op) => FP::IsSubnormal(*op),
            FP::IsZero(op) => FP::IsZero(*op),
            FP::IsInfinite(op) => FP::IsInfinite(*op),
            FP::IsNaN(op) => FP::IsNaN(*op),
            FP::IsNegative(op) => FP::IsNegative(*op),
            FP::IsPositive(op) => FP::IsPositive(*op),
        }
    }
}

impl<T: fmt::Display> fmt::Display for FP<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FP::Const(op) => op.fmt(f),
            FP::Ite(op) => op.fmt(f),
            FP::Convert(op) => op.fmt(f),
            FP::PlusInfinity(op) => op.fmt(f),
            FP::MinusInfinity(op) => op.fmt(f),
            FP::PlusZero(op) => op.fmt(f),
            FP::MinusZero(op) => op.fmt(f),
            FP::NaN(op) => op.fmt(f),
            FP::Abs(op) => op.fmt(f),
            FP::Neg(op) => op.fmt(f),
            FP::Sqrt(op) => op.fmt(f),
            FP::RoundToIntegral(op) => op.fmt(f),
            FP::Add(op) => op.fmt(f),
            FP::Sub(op) => op.fmt(f),
            FP::Mul(op) => op.fmt(f),
            FP::Div(op) => op.fmt(f),
            FP::Fma(op) => op.fmt(f),
            FP::Rem(op) => op.fmt(f),
            FP::Min(op) => op.fmt(f),
            FP::Max(op) => op.fmt(f),
            FP::Le(op) => op.fmt(f),
            FP::Lt(op) => op.fmt(f),
            FP::Ge(op) => op.fmt(f),
            FP::Gt(op) => op.fmt(f),
            FP::Eq(op) => op.fmt(f),
            FP::IsNormal(op) => op.fmt(f),
            FP::IsSubnormal(op) => op.fmt(f),
            FP::IsZero(op) => op.fmt(f),
            FP::IsInfinite(op) => op.fmt(f),
            FP::IsNaN(op) => op.fmt(f),
            FP::IsNegative(op) => op.fmt(f),
            FP::IsPositive(op) => op.fmt(f),
        }
    }
}

impl<T: Tensor> Boundary<Sort> for FP<T> {
    fn dom(&self) -> Poly<Sort> {
        match self {
            FP::Const(op) => op.dom(),
            FP::Ite(op) => op.dom(),
            FP::Convert(op) => op.dom(),
            FP::PlusInfinity(op) => op.dom(),
            FP::MinusInfinity(op) => op.dom(),
            FP::PlusZero(op) => op.dom(),
            FP::MinusZero(op) => op.dom(),
            FP::NaN(op) => op.dom(),
            FP::Abs(op) => op.dom(),
            FP::Neg(op) => op.dom(),
            FP::Sqrt(op) => op.dom(),
            FP::RoundToIntegral(op) => op.dom(),
            FP::Add(op) => op.dom(),
            FP::Sub(op) => op.dom(),
            FP::Mul(op) => op.dom(),
            FP::Div(op) => op.dom(),
            FP::Fma(op) => op.dom(),
            FP::Rem(op) => op.dom(),
            FP::Min(op) => op.dom(),
            FP::Max(op) => op.dom(),
            FP::Le(op) => op.dom(),
            FP::Lt(op) => op.dom(),
            FP::Ge(op) => op.dom(),
            FP::Gt(op) => op.dom(),
            FP::Eq(op) => op.dom(),
            FP::IsNormal(op) => op.dom(),
            FP::IsSubnormal(op) => op.dom(),
            FP::IsZero(op) => op.dom(),
            FP::IsInfinite(op) => op.dom(),
            FP::IsNaN(op) => op.dom(),
            FP::IsNegative(op) => op.dom(),
            FP::IsPositive(op) => op.dom(),
        }
    }

    fn cod(&self) -> Poly<Sort> {
        match self {
            FP::Const(op) => op.cod(),
            FP::Ite(op) => op.cod(),
            FP::Convert(op) => op.cod(),
            FP::PlusInfinity(op) => op.cod(),
            FP::MinusInfinity(op) => op.cod(),
            FP::PlusZero(op) => op.cod(),
            FP::MinusZero(op) => op.cod(),
            FP::NaN(op) => op.cod(),
            FP::Abs(op) => op.cod(),
            FP::Neg(op) => op.cod(),
            FP::Sqrt(op) => op.cod(),
            FP::RoundToIntegral(op) => op.cod(),
            FP::Add(op) => op.cod(),
            FP::Sub(op) => op.cod(),
            FP::Mul(op) => op.cod(),
            FP::Div(op) => op.cod(),
            FP::Fma(op) => op.cod(),
            FP::Rem(op) => op.cod(),
            FP::Min(op) => op.cod(),
            FP::Max(op) => op.cod(),
            FP::Le(op) => op.cod(),
            FP::Lt(op) => op.cod(),
            FP::Ge(op) => op.cod(),
            FP::Gt(op) => op.cod(),
            FP::Eq(op) => op.cod(),
            FP::IsNormal(op) => op.cod(),
            FP::IsSubnormal(op) => op.cod(),
            FP::IsZero(op) => op.cod(),
            FP::IsInfinite(op) => op.cod(),
            FP::IsNaN(op) => op.cod(),
            FP::IsNegative(op) => op.cod(),
            FP::IsPositive(op) => op.cod(),
        }
    }
}

/// Extracts an operation of floating-point arithmetic from a Python object,
/// trying each operation in turn: those carrying no data through their own
/// classes, and [`Const`] through the extraction that the tensor type `T`
/// provides.
#[cfg(feature = "pyo3")]
impl<'a, 'py, T> pyo3::FromPyObject<'a, 'py> for FP<T>
where
    Const<T>: pyo3::FromPyObject<'a, 'py>,
{
    type Error = pyo3::PyErr;

    fn extract(ob: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use pyo3::types::PyAnyMethods;

        if let Ok(op) = ob.extract::<Ite>() {
            return Ok(FP::Ite(op));
        }
        if let Ok(op) = ob.extract::<Convert>() {
            return Ok(FP::Convert(op));
        }
        if let Ok(op) = ob.extract::<PlusInfinity>() {
            return Ok(FP::PlusInfinity(op));
        }
        if let Ok(op) = ob.extract::<MinusInfinity>() {
            return Ok(FP::MinusInfinity(op));
        }
        if let Ok(op) = ob.extract::<PlusZero>() {
            return Ok(FP::PlusZero(op));
        }
        if let Ok(op) = ob.extract::<MinusZero>() {
            return Ok(FP::MinusZero(op));
        }
        if let Ok(op) = ob.extract::<NaN>() {
            return Ok(FP::NaN(op));
        }
        if let Ok(op) = ob.extract::<Abs>() {
            return Ok(FP::Abs(op));
        }
        if let Ok(op) = ob.extract::<Neg>() {
            return Ok(FP::Neg(op));
        }
        if let Ok(op) = ob.extract::<Sqrt>() {
            return Ok(FP::Sqrt(op));
        }
        if let Ok(op) = ob.extract::<RoundToIntegral>() {
            return Ok(FP::RoundToIntegral(op));
        }
        if let Ok(op) = ob.extract::<Add>() {
            return Ok(FP::Add(op));
        }
        if let Ok(op) = ob.extract::<Sub>() {
            return Ok(FP::Sub(op));
        }
        if let Ok(op) = ob.extract::<Mul>() {
            return Ok(FP::Mul(op));
        }
        if let Ok(op) = ob.extract::<Div>() {
            return Ok(FP::Div(op));
        }
        if let Ok(op) = ob.extract::<Fma>() {
            return Ok(FP::Fma(op));
        }
        if let Ok(op) = ob.extract::<Rem>() {
            return Ok(FP::Rem(op));
        }
        if let Ok(op) = ob.extract::<Min>() {
            return Ok(FP::Min(op));
        }
        if let Ok(op) = ob.extract::<Max>() {
            return Ok(FP::Max(op));
        }
        if let Ok(op) = ob.extract::<Le>() {
            return Ok(FP::Le(op));
        }
        if let Ok(op) = ob.extract::<Lt>() {
            return Ok(FP::Lt(op));
        }
        if let Ok(op) = ob.extract::<Ge>() {
            return Ok(FP::Ge(op));
        }
        if let Ok(op) = ob.extract::<Gt>() {
            return Ok(FP::Gt(op));
        }
        if let Ok(op) = ob.extract::<Eq>() {
            return Ok(FP::Eq(op));
        }
        if let Ok(op) = ob.extract::<IsNormal>() {
            return Ok(FP::IsNormal(op));
        }
        if let Ok(op) = ob.extract::<IsSubnormal>() {
            return Ok(FP::IsSubnormal(op));
        }
        if let Ok(op) = ob.extract::<IsZero>() {
            return Ok(FP::IsZero(op));
        }
        if let Ok(op) = ob.extract::<IsInfinite>() {
            return Ok(FP::IsInfinite(op));
        }
        if let Ok(op) = ob.extract::<IsNaN>() {
            return Ok(FP::IsNaN(op));
        }
        if let Ok(op) = ob.extract::<IsNegative>() {
            return Ok(FP::IsNegative(op));
        }
        if let Ok(op) = ob.extract::<IsPositive>() {
            return Ok(FP::IsPositive(op));
        }
        if let Ok(op) = ob.extract::<Const<T>>() {
            return Ok(FP::Const(op));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "expected an operation of floating-point arithmetic, got {}",
            ob.get_type()
        )))
    }
}

impl Boundary<Sort> for PlusInfinity {
    fn dom(&self) -> Poly<Sort> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![self.sort.into()].into()
    }
}

impl Boundary<Sort> for MinusInfinity {
    fn dom(&self) -> Poly<Sort> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![self.sort.into()].into()
    }
}

impl Boundary<Sort> for PlusZero {
    fn dom(&self) -> Poly<Sort> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![self.sort.into()].into()
    }
}

impl Boundary<Sort> for MinusZero {
    fn dom(&self) -> Poly<Sort> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![self.sort.into()].into()
    }
}

impl Boundary<Sort> for NaN {
    fn dom(&self) -> Poly<Sort> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![self.sort.into()].into()
    }
}

impl Boundary<Sort> for Abs {
    fn dom(&self) -> Poly<Sort> {
        mono![self.sort.into()].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![self.sort.into()].into()
    }
}

impl Boundary<Sort> for Neg {
    fn dom(&self) -> Poly<Sort> {
        mono![self.sort.into()].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![self.sort.into()].into()
    }
}

impl Boundary<Sort> for Sqrt {
    fn dom(&self) -> Poly<Sort> {
        mono![self.sort.into()].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![self.sort.into()].into()
    }
}

impl Boundary<Sort> for RoundToIntegral {
    fn dom(&self) -> Poly<Sort> {
        mono![self.sort.into()].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![self.sort.into()].into()
    }
}

impl Boundary<Sort> for Add {
    fn dom(&self) -> Poly<Sort> {
        mono![self.sort.into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![self.sort.into()].into()
    }
}

impl Boundary<Sort> for Sub {
    fn dom(&self) -> Poly<Sort> {
        mono![self.sort.into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![self.sort.into()].into()
    }
}

impl Boundary<Sort> for Mul {
    fn dom(&self) -> Poly<Sort> {
        mono![self.sort.into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![self.sort.into()].into()
    }
}

impl Boundary<Sort> for Div {
    fn dom(&self) -> Poly<Sort> {
        mono![self.sort.into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![self.sort.into()].into()
    }
}

impl Boundary<Sort> for Fma {
    fn dom(&self) -> Poly<Sort> {
        mono![self.sort.into(); 3].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![self.sort.into()].into()
    }
}

impl Boundary<Sort> for Rem {
    fn dom(&self) -> Poly<Sort> {
        mono![self.sort.into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![self.sort.into()].into()
    }
}

impl Boundary<Sort> for Min {
    fn dom(&self) -> Poly<Sort> {
        mono![self.sort.into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![self.sort.into()].into()
    }
}

impl Boundary<Sort> for Max {
    fn dom(&self) -> Poly<Sort> {
        mono![self.sort.into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![self.sort.into()].into()
    }
}

impl Boundary<Sort> for Le {
    fn dom(&self) -> Poly<Sort> {
        mono![self.sort.into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new(self.sort.shape()).into()].into()
    }
}

impl Boundary<Sort> for Lt {
    fn dom(&self) -> Poly<Sort> {
        mono![self.sort.into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new(self.sort.shape()).into()].into()
    }
}

impl Boundary<Sort> for Ge {
    fn dom(&self) -> Poly<Sort> {
        mono![self.sort.into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new(self.sort.shape()).into()].into()
    }
}

impl Boundary<Sort> for Gt {
    fn dom(&self) -> Poly<Sort> {
        mono![self.sort.into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new(self.sort.shape()).into()].into()
    }
}

impl Boundary<Sort> for Eq {
    fn dom(&self) -> Poly<Sort> {
        mono![self.sort.into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new(self.sort.shape()).into()].into()
    }
}

impl Boundary<Sort> for IsNormal {
    fn dom(&self) -> Poly<Sort> {
        mono![self.sort.into()].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new(self.sort.shape()).into()].into()
    }
}

impl Boundary<Sort> for IsSubnormal {
    fn dom(&self) -> Poly<Sort> {
        mono![self.sort.into()].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new(self.sort.shape()).into()].into()
    }
}

impl Boundary<Sort> for IsZero {
    fn dom(&self) -> Poly<Sort> {
        mono![self.sort.into()].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new(self.sort.shape()).into()].into()
    }
}

impl Boundary<Sort> for IsInfinite {
    fn dom(&self) -> Poly<Sort> {
        mono![self.sort.into()].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new(self.sort.shape()).into()].into()
    }
}

impl Boundary<Sort> for IsNaN {
    fn dom(&self) -> Poly<Sort> {
        mono![self.sort.into()].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new(self.sort.shape()).into()].into()
    }
}

impl Boundary<Sort> for IsNegative {
    fn dom(&self) -> Poly<Sort> {
        mono![self.sort.into()].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new(self.sort.shape()).into()].into()
    }
}

impl Boundary<Sort> for IsPositive {
    fn dom(&self) -> Poly<Sort> {
        mono![self.sort.into()].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new(self.sort.shape()).into()].into()
    }
}

impl fmt::Display for PlusInfinity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PlusInfinity")
    }
}

impl fmt::Display for MinusInfinity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MinusInfinity")
    }
}

impl fmt::Display for PlusZero {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PlusZero")
    }
}

impl fmt::Display for MinusZero {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MinusZero")
    }
}

impl fmt::Display for NaN {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NaN")
    }
}

impl fmt::Display for Abs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Abs")
    }
}

impl fmt::Display for Neg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Neg")
    }
}

impl fmt::Display for Sqrt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sqrt({})", self.rm)
    }
}

impl fmt::Display for RoundToIntegral {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RoundToIntegral({})", self.rm)
    }
}

impl fmt::Display for Add {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Add({})", self.rm)
    }
}

impl fmt::Display for Sub {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sub({})", self.rm)
    }
}

impl fmt::Display for Mul {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Mul({})", self.rm)
    }
}

impl fmt::Display for Div {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Div({})", self.rm)
    }
}

impl fmt::Display for Fma {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fma({})", self.rm)
    }
}

impl fmt::Display for Rem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Rem")
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

impl fmt::Display for Le {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Le")
    }
}

impl fmt::Display for Lt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Lt")
    }
}

impl fmt::Display for Ge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ge")
    }
}

impl fmt::Display for Gt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Gt")
    }
}

impl fmt::Display for Eq {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Eq")
    }
}

impl fmt::Display for IsNormal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IsNormal")
    }
}

impl fmt::Display for IsSubnormal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IsSubnormal")
    }
}

impl fmt::Display for IsZero {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IsZero")
    }
}

impl fmt::Display for IsInfinite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IsInfinite")
    }
}

impl fmt::Display for IsNaN {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IsNaN")
    }
}

impl fmt::Display for IsNegative {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IsNegative")
    }
}

impl fmt::Display for IsPositive {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IsPositive")
    }
}

impl PlusInfinity {
    /// The operation writing the floats of `cod`.
    pub fn with_cod(cod: [Float; 1]) -> Result<Self, crate::Error> {
        let [sort] = cod;
        Ok(PlusInfinity { sort })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl PlusInfinity {
    /// Python builds it from its codomain `cod`.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (cod))]
    fn py_new(cod: Float) -> Result<Self, crate::Error> {
        Self::with_cod([cod])
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("cod",)
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }
}

impl MinusInfinity {
    /// The operation writing the floats of `cod`.
    pub fn with_cod(cod: [Float; 1]) -> Result<Self, crate::Error> {
        let [sort] = cod;
        Ok(MinusInfinity { sort })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl MinusInfinity {
    /// Python builds it from its codomain `cod`.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (cod))]
    fn py_new(cod: Float) -> Result<Self, crate::Error> {
        Self::with_cod([cod])
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("cod",)
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }
}

impl PlusZero {
    /// The operation writing the floats of `cod`.
    pub fn with_cod(cod: [Float; 1]) -> Result<Self, crate::Error> {
        let [sort] = cod;
        Ok(PlusZero { sort })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl PlusZero {
    /// Python builds it from its codomain `cod`.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (cod))]
    fn py_new(cod: Float) -> Result<Self, crate::Error> {
        Self::with_cod([cod])
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("cod",)
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }
}

impl MinusZero {
    /// The operation writing the floats of `cod`.
    pub fn with_cod(cod: [Float; 1]) -> Result<Self, crate::Error> {
        let [sort] = cod;
        Ok(MinusZero { sort })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl MinusZero {
    /// Python builds it from its codomain `cod`.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (cod))]
    fn py_new(cod: Float) -> Result<Self, crate::Error> {
        Self::with_cod([cod])
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("cod",)
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }
}

impl NaN {
    /// The operation writing the floats of `cod`.
    pub fn with_cod(cod: [Float; 1]) -> Result<Self, crate::Error> {
        let [sort] = cod;
        Ok(NaN { sort })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl NaN {
    /// Python builds it from its codomain `cod`.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (cod))]
    fn py_new(cod: Float) -> Result<Self, crate::Error> {
        Self::with_cod([cod])
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("cod",)
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }
}

impl Abs {
    /// The operation reading the floats of `dom`.
    pub fn with_dom(dom: [Float; 1]) -> Result<Self, crate::Error> {
        let [sort] = dom;
        Ok(Abs { sort })
    }

    /// The operation writing the floats of `cod`.
    pub fn with_cod(cod: [Float; 1]) -> Result<Self, crate::Error> {
        let [sort] = cod;
        Ok(Abs { sort })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Abs {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: Float, cod: Option<Float>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom([dom]), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }
}

impl Neg {
    /// The operation reading the floats of `dom`.
    pub fn with_dom(dom: [Float; 1]) -> Result<Self, crate::Error> {
        let [sort] = dom;
        Ok(Neg { sort })
    }

    /// The operation writing the floats of `cod`.
    pub fn with_cod(cod: [Float; 1]) -> Result<Self, crate::Error> {
        let [sort] = cod;
        Ok(Neg { sort })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Neg {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: Float, cod: Option<Float>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom([dom]), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }
}

impl Sqrt {
    /// The operation reading the floats of `dom`.
    pub fn with_dom(rm: RoundingMode, dom: [Float; 1]) -> Result<Self, crate::Error> {
        let [sort] = dom;
        Ok(Sqrt { rm, sort })
    }

    /// The operation writing the floats of `cod`.
    pub fn with_cod(rm: RoundingMode, cod: [Float; 1]) -> Result<Self, crate::Error> {
        let [sort] = cod;
        Ok(Sqrt { rm, sort })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Sqrt {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (rm, /, dom, cod = None))]
    fn py_new(rm: RoundingMode, dom: Float, cod: Option<Float>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom(rm, [dom]), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str, &'static str) {
        ("rm", "dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }

    /// The rounding mode.
    #[cfg_attr(feature = "pyo3", getter)]
    pub fn rm(&self) -> RoundingMode {
        self.rm
    }
}

impl RoundToIntegral {
    /// The operation reading the floats of `dom`.
    pub fn with_dom(rm: RoundingMode, dom: [Float; 1]) -> Result<Self, crate::Error> {
        let [sort] = dom;
        Ok(RoundToIntegral { rm, sort })
    }

    /// The operation writing the floats of `cod`.
    pub fn with_cod(rm: RoundingMode, cod: [Float; 1]) -> Result<Self, crate::Error> {
        let [sort] = cod;
        Ok(RoundToIntegral { rm, sort })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl RoundToIntegral {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (rm, /, dom, cod = None))]
    fn py_new(rm: RoundingMode, dom: Float, cod: Option<Float>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom(rm, [dom]), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str, &'static str) {
        ("rm", "dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }

    /// The rounding mode.
    #[cfg_attr(feature = "pyo3", getter)]
    pub fn rm(&self) -> RoundingMode {
        self.rm
    }
}

impl Add {
    /// The operation reading the floats of `dom`; fails unless they are equal.
    pub fn with_dom(rm: RoundingMode, dom: [Float; 2]) -> Result<Self, crate::Error> {
        let sort = crate::tensor::same_sort(dom)?;
        Ok(Add { rm, sort })
    }

    /// The operation writing the floats of `cod`.
    pub fn with_cod(rm: RoundingMode, cod: [Float; 1]) -> Result<Self, crate::Error> {
        let [sort] = cod;
        Ok(Add { rm, sort })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Add {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (rm, /, dom, cod = None))]
    fn py_new(rm: RoundingMode, dom: [Float; 2], cod: Option<Float>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom(rm, dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str, &'static str) {
        ("rm", "dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }

    /// The rounding mode.
    #[cfg_attr(feature = "pyo3", getter)]
    pub fn rm(&self) -> RoundingMode {
        self.rm
    }
}

impl Sub {
    /// The operation reading the floats of `dom`; fails unless they are equal.
    pub fn with_dom(rm: RoundingMode, dom: [Float; 2]) -> Result<Self, crate::Error> {
        let sort = crate::tensor::same_sort(dom)?;
        Ok(Sub { rm, sort })
    }

    /// The operation writing the floats of `cod`.
    pub fn with_cod(rm: RoundingMode, cod: [Float; 1]) -> Result<Self, crate::Error> {
        let [sort] = cod;
        Ok(Sub { rm, sort })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Sub {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (rm, /, dom, cod = None))]
    fn py_new(rm: RoundingMode, dom: [Float; 2], cod: Option<Float>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom(rm, dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str, &'static str) {
        ("rm", "dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }

    /// The rounding mode.
    #[cfg_attr(feature = "pyo3", getter)]
    pub fn rm(&self) -> RoundingMode {
        self.rm
    }
}

impl Mul {
    /// The operation reading the floats of `dom`; fails unless they are equal.
    pub fn with_dom(rm: RoundingMode, dom: [Float; 2]) -> Result<Self, crate::Error> {
        let sort = crate::tensor::same_sort(dom)?;
        Ok(Mul { rm, sort })
    }

    /// The operation writing the floats of `cod`.
    pub fn with_cod(rm: RoundingMode, cod: [Float; 1]) -> Result<Self, crate::Error> {
        let [sort] = cod;
        Ok(Mul { rm, sort })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Mul {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (rm, /, dom, cod = None))]
    fn py_new(rm: RoundingMode, dom: [Float; 2], cod: Option<Float>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom(rm, dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str, &'static str) {
        ("rm", "dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }

    /// The rounding mode.
    #[cfg_attr(feature = "pyo3", getter)]
    pub fn rm(&self) -> RoundingMode {
        self.rm
    }
}

impl Div {
    /// The operation reading the floats of `dom`; fails unless they are equal.
    pub fn with_dom(rm: RoundingMode, dom: [Float; 2]) -> Result<Self, crate::Error> {
        let sort = crate::tensor::same_sort(dom)?;
        Ok(Div { rm, sort })
    }

    /// The operation writing the floats of `cod`.
    pub fn with_cod(rm: RoundingMode, cod: [Float; 1]) -> Result<Self, crate::Error> {
        let [sort] = cod;
        Ok(Div { rm, sort })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Div {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (rm, /, dom, cod = None))]
    fn py_new(rm: RoundingMode, dom: [Float; 2], cod: Option<Float>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom(rm, dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str, &'static str) {
        ("rm", "dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }

    /// The rounding mode.
    #[cfg_attr(feature = "pyo3", getter)]
    pub fn rm(&self) -> RoundingMode {
        self.rm
    }
}

impl Fma {
    /// The operation reading the floats of `dom`; fails unless they are equal.
    pub fn with_dom(rm: RoundingMode, dom: [Float; 3]) -> Result<Self, crate::Error> {
        let sort = crate::tensor::same_sort(dom)?;
        Ok(Fma { rm, sort })
    }

    /// The operation writing the floats of `cod`.
    pub fn with_cod(rm: RoundingMode, cod: [Float; 1]) -> Result<Self, crate::Error> {
        let [sort] = cod;
        Ok(Fma { rm, sort })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Fma {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (rm, /, dom, cod = None))]
    fn py_new(rm: RoundingMode, dom: [Float; 3], cod: Option<Float>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom(rm, dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str, &'static str) {
        ("rm", "dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }

    /// The rounding mode.
    #[cfg_attr(feature = "pyo3", getter)]
    pub fn rm(&self) -> RoundingMode {
        self.rm
    }
}

impl Rem {
    /// The operation reading the floats of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Float; 2]) -> Result<Self, crate::Error> {
        let sort = crate::tensor::same_sort(dom)?;
        Ok(Rem { sort })
    }

    /// The operation writing the floats of `cod`.
    pub fn with_cod(cod: [Float; 1]) -> Result<Self, crate::Error> {
        let [sort] = cod;
        Ok(Rem { sort })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Rem {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Float; 2], cod: Option<Float>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }
}

impl Min {
    /// The operation reading the floats of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Float; 2]) -> Result<Self, crate::Error> {
        let sort = crate::tensor::same_sort(dom)?;
        Ok(Min { sort })
    }

    /// The operation writing the floats of `cod`.
    pub fn with_cod(cod: [Float; 1]) -> Result<Self, crate::Error> {
        let [sort] = cod;
        Ok(Min { sort })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Min {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Float; 2], cod: Option<Float>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }
}

impl Max {
    /// The operation reading the floats of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Float; 2]) -> Result<Self, crate::Error> {
        let sort = crate::tensor::same_sort(dom)?;
        Ok(Max { sort })
    }

    /// The operation writing the floats of `cod`.
    pub fn with_cod(cod: [Float; 1]) -> Result<Self, crate::Error> {
        let [sort] = cod;
        Ok(Max { sort })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Max {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Float; 2], cod: Option<Float>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }
}

impl Le {
    /// The operation reading the floats of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Float; 2]) -> Result<Self, crate::Error> {
        let sort = crate::tensor::same_sort(dom)?;
        Ok(Le { sort })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Le {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Float; 2], cod: Option<Bool>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }
}

impl Lt {
    /// The operation reading the floats of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Float; 2]) -> Result<Self, crate::Error> {
        let sort = crate::tensor::same_sort(dom)?;
        Ok(Lt { sort })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Lt {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Float; 2], cod: Option<Bool>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }
}

impl Ge {
    /// The operation reading the floats of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Float; 2]) -> Result<Self, crate::Error> {
        let sort = crate::tensor::same_sort(dom)?;
        Ok(Ge { sort })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Ge {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Float; 2], cod: Option<Bool>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }
}

impl Gt {
    /// The operation reading the floats of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Float; 2]) -> Result<Self, crate::Error> {
        let sort = crate::tensor::same_sort(dom)?;
        Ok(Gt { sort })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Gt {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Float; 2], cod: Option<Bool>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }
}

impl Eq {
    /// The operation reading the floats of `dom`; fails unless they are equal.
    pub fn with_dom(dom: [Float; 2]) -> Result<Self, crate::Error> {
        let sort = crate::tensor::same_sort(dom)?;
        Ok(Eq { sort })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Eq {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Float; 2], cod: Option<Bool>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom(dom), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }
}

impl IsNormal {
    /// The operation reading the floats of `dom`.
    pub fn with_dom(dom: [Float; 1]) -> Result<Self, crate::Error> {
        let [sort] = dom;
        Ok(IsNormal { sort })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl IsNormal {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: Float, cod: Option<Bool>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom([dom]), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }
}

impl IsSubnormal {
    /// The operation reading the floats of `dom`.
    pub fn with_dom(dom: [Float; 1]) -> Result<Self, crate::Error> {
        let [sort] = dom;
        Ok(IsSubnormal { sort })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl IsSubnormal {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: Float, cod: Option<Bool>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom([dom]), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }
}

impl IsZero {
    /// The operation reading the floats of `dom`.
    pub fn with_dom(dom: [Float; 1]) -> Result<Self, crate::Error> {
        let [sort] = dom;
        Ok(IsZero { sort })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl IsZero {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: Float, cod: Option<Bool>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom([dom]), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }
}

impl IsInfinite {
    /// The operation reading the floats of `dom`.
    pub fn with_dom(dom: [Float; 1]) -> Result<Self, crate::Error> {
        let [sort] = dom;
        Ok(IsInfinite { sort })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl IsInfinite {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: Float, cod: Option<Bool>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom([dom]), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }
}

impl IsNaN {
    /// The operation reading the floats of `dom`.
    pub fn with_dom(dom: [Float; 1]) -> Result<Self, crate::Error> {
        let [sort] = dom;
        Ok(IsNaN { sort })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl IsNaN {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: Float, cod: Option<Bool>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom([dom]), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }
}

impl IsNegative {
    /// The operation reading the floats of `dom`.
    pub fn with_dom(dom: [Float; 1]) -> Result<Self, crate::Error> {
        let [sort] = dom;
        Ok(IsNegative { sort })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl IsNegative {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: Float, cod: Option<Bool>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom([dom]), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }
}

impl IsPositive {
    /// The operation reading the floats of `dom`.
    pub fn with_dom(dom: [Float; 1]) -> Result<Self, crate::Error> {
        let [sort] = dom;
        Ok(IsPositive { sort })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl IsPositive {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: Float, cod: Option<Bool>) -> Result<Self, crate::Error> {
        crate::tensor::check_cod(Self::with_dom([dom]), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::tensor::py_sorts(py, Boundary::cod(self))
    }
}
