/*!
# Tensors

The tensor operations of each theory, one type per operation, in a module per
theory: [`real`] and [`int`] for arithmetic, [`bool`](mod@bool) for
propositional logic, and [`word`] for fixed-size bit-vectors. Each
operation reads and writes the sorts of [`sort`], and computes its
own boundary.

The operations are organised in two levels. Each theory module gathers its
operations in an enum named after the logic of that theory alone:
[`real::NRA`], [`int::NIA`], [`bool::Prop`] and [`word::BV`]. The
arithmetic ones mark their linear operations with a subenum named after
the linear logic: [`real::LRA`] and [`int::LIA`].
Here, [`Any`] holds the operations of all theories, one variant per
theory, and each logic picks the theories it combines, restricting
arithmetic to its linear operations where the logic is linear:

| Logic | Operations | SMT-LIB2 logic |
|---|---|---|
| [`Prop`] | [`bool::Prop`] | none: the `Core` theory alone |
| [`NRA`] | [`real::NRA`], [`bool::Prop`] | `QF_NRA` |
| [`LRA`] | [`real::LRA`], [`bool::Prop`] | `QF_LRA` |
| [`NIA`] | [`int::NIA`], [`bool::Prop`] | `QF_NIA` |
| [`LIA`] | [`int::LIA`], [`bool::Prop`] | `QF_LIA` |
| [`BV`] | [`word::BV`], [`bool::Prop`] | `QF_BV` |

The operations that carry data, the constants and the coefficients of the
affine maps, hold it in matrices of a type `T` implementing [`Tensor`], over
which every operation, theory enum and logic is generic. With the `torch`
feature, `tch::Tensor` implements it, converting to and from Python
`torch.Tensor` objects (see the `torch` module).
*/

pub mod bool;
pub mod int;
pub mod real;
pub mod sort;
pub mod word;
pub mod zero;

use base::boundary::{Boundary, Poly};
use derive_more::From;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use std::fmt;
use subenum::subenum;
pub use zero::Zero;

/// A matrix of values, as held by the operations that carry data: the
/// constants, and the coefficients of `Linear`. The operations are generic
/// over it, so that any tensor representation can back them.
pub trait Tensor {
    /// The `[rows, cols]` shape of the matrix.
    fn shape(&self) -> [usize; 2];
}

/// The operations of all theories: a value of `Any` is an operation of one
/// of them, on the sorts in [`sort::Any`], with its data held in tensors of
/// type `T`; or one of the structural operations [`Havoc`], [`Skip`] and
/// [`Zero`], which belong to no theory and so to no logic.
///
/// `subenum` defines the logics that take whole theories: [`Prop`],
/// propositional logic; [`NRA`] and [`NIA`], non-linear real and integer
/// arithmetic; and [`BV`], the logic `QF_BV` of fixed-size bit-vectors. It
/// also generates their casts: `From` up to `Any`, and `TryFrom` down from
/// it. The linear logics [`LRA`] and [`LIA`] restrict a theory to its
/// linear operations, which `subenum` cannot express, so they are defined
/// on their own, with the same casts.
///
/// `Any` is cloneable when the constants and affine maps of its tensor type
/// are: `Clone` is implemented by hand rather than derived, as a derive would
/// require `T: Clone`, which tensor types such as `tch::Tensor` are not.
#[subenum(NRA, NIA, Prop, BV)]
#[derive(Debug, From)]
pub enum Any<T> {
    #[subenum(NRA)]
    Real(real::NRA<T>),
    #[subenum(NIA)]
    Int(int::NIA<T>),
    #[subenum(NRA, NIA, Prop, BV)]
    Bool(bool::Prop<T>),
    #[subenum(BV)]
    Word(word::BV<T>),
    Havoc(Havoc),
    Skip(Skip),
    Zero(Zero),
}

/// Linear real arithmetic: the linear operations of [`real`], with the
/// boolean operations.
#[derive(Debug, From)]
pub enum LRA<T> {
    Real(real::LRA<T>),
    Bool(bool::Prop<T>),
}

/// Linear integer arithmetic: the linear operations of [`int`], with the
/// boolean operations.
#[derive(Debug, From)]
pub enum LIA<T> {
    Int(int::LIA<T>),
    Bool(bool::Prop<T>),
}

/// Writes an arbitrary value of the given sort, reading nothing: the
/// nondeterministic choice of a value.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor")
)]
pub struct Havoc {
    sort: sort::Any,
}

/// Writes the value it reads, of the given sort: the identity.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.tensor")
)]
pub struct Skip {
    sort: sort::Any,
}

/// Fails unless `a` and `b` are the shapes `[m, n]` and `[m, k]` of the
/// matrices `A` and `B` of an affine map `x ↦ A·x + B`.
pub(crate) fn check_affine(a: [usize; 2], b: [usize; 2]) -> Result<(), crate::Error> {
    if a[0] != b[0] {
        return Err(crate::Error(format!(
            "expected A of shape [m, n] and B of shape [m, k], got {a:?} and {b:?}"
        )));
    }
    Ok(())
}

/// The failure of a `with_dom` constructor: `dom` is not the domain of an
/// instance of the operation, which reads the sorts described by `expected`.
pub(crate) fn unexpected_dom<S: fmt::Display>(expected: &str, dom: &[S]) -> crate::Error {
    unexpected("domain", expected, dom)
}

/// The failure of a `with_cod` constructor: `cod` is not the codomain of an
/// instance of the operation, which writes the sorts described by `expected`.
pub(crate) fn unexpected_cod<S: fmt::Display>(expected: &str, cod: &[S]) -> crate::Error {
    unexpected("codomain", expected, cod)
}

fn unexpected<S: fmt::Display>(boundary: &str, expected: &str, sorts: &[S]) -> crate::Error {
    let sorts: Vec<String> = sorts.iter().map(ToString::to_string).collect();
    crate::Error(format!(
        "expected a {boundary} of {expected}, got [{}]",
        sorts.join(", ")
    ))
}

/// Fails unless the sorts of `dom`, at least one, are all equal, and
/// returns that sort: the domain of an operation whose inputs share it.
pub(crate) fn same_sort<S: Copy + PartialEq + fmt::Display, const N: usize>(
    dom: [S; N],
) -> Result<S, crate::Error> {
    let s = dom[0];
    if dom.iter().any(|t| *t != s) {
        return Err(unexpected_dom(&format!("{N} equal sorts"), &dom));
    }
    Ok(s)
}

/// The operation a Python constructor builds from its domain, `op`, checked
/// against its codomain `cod` when given: that must be the codomain of `op`.
#[cfg(feature = "pyo3")]
pub(crate) fn check_cod<Op, S, C>(
    op: Result<Op, crate::Error>,
    cod: Option<C>,
) -> Result<Op, crate::Error>
where
    Op: Boundary<S>,
    S: Clone + PartialEq + fmt::Display,
    C: Into<S> + fmt::Display,
{
    let op = op?;
    if let Some(c) = cod {
        let expected: Poly<S> = base::mono![c.into()].into();
        if op.cod() != expected {
            return Err(crate::Error(format!(
                "the operation reading this domain writes {}, not {expected}",
                op.cod()
            )));
        }
    }
    Ok(op)
}

/// The sorts of the domain or codomain `poly` of an operation, as Python
/// presents them: the sort alone when there is one, else their list.
///
/// Panics unless `poly` is a single product of sorts, as the boundaries of
/// the operations are.
#[cfg(feature = "pyo3")]
pub(crate) fn py_sorts<'py, S: Into<sort::Any>>(
    py: Python<'py>,
    poly: Poly<S>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut monos = poly.into_iter();
    let (Some(mono), None) = (monos.next(), monos.next()) else {
        panic!("the boundary of an operation is a single product of sorts");
    };
    let mut sorts: Vec<sort::Any> = mono.into_iter().map(Into::into).collect();
    if sorts.len() == 1 {
        sorts.pop().unwrap().into_pyobject(py)
    } else {
        Ok(sorts.into_pyobject(py)?.into_any())
    }
}

impl Boundary<sort::Any> for Havoc {
    fn dom(&self) -> Poly<sort::Any> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<sort::Any> {
        base::mono![self.sort].into()
    }
}

impl Boundary<sort::Any> for Skip {
    fn dom(&self) -> Poly<sort::Any> {
        base::mono![self.sort].into()
    }

    fn cod(&self) -> Poly<sort::Any> {
        base::mono![self.sort].into()
    }
}

impl fmt::Display for Havoc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Havoc")
    }
}

impl fmt::Display for Skip {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Skip")
    }
}

impl Havoc {
    /// The choice of a value of the sort of `cod`.
    pub fn with_cod(cod: [sort::Any; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Havoc { sort: s })
    }

    pub fn new(sort: sort::Any) -> Self {
        Havoc { sort }
    }
}

impl Skip {
    /// The identity reading the sort of `dom`.
    pub fn with_dom(dom: [sort::Any; 1]) -> Result<Self, crate::Error> {
        let [s] = dom;
        Ok(Skip { sort: s })
    }

    /// The identity writing the sort of `cod`.
    pub fn with_cod(cod: [sort::Any; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Skip { sort: s })
    }

    pub fn new(sort: sort::Any) -> Self {
        Skip { sort }
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Havoc {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its codomain `cod`.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (cod))]
    fn py_new(cod: sort::Any) -> Result<Self, crate::Error> {
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

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn sort(&self) -> sort::Any {
        self.sort
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Skip {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its domain `dom`; its codomain `cod`, when given, must be the one it
    /// writes.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: sort::Any, cod: Option<sort::Any>) -> Result<Self, crate::Error> {
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

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn sort(&self) -> sort::Any {
        self.sort
    }
}

impl<T> crate::Combinatorial<sort::Any> for Any<T> {
    fn havoc(range: &sort::Any) -> Self {
        Any::Havoc(Havoc::new(*range))
    }
}

impl<T> crate::Sequential<sort::Any> for Any<T> {
    fn skip(range: &sort::Any) -> Self {
        Any::Skip(Skip::new(*range))
    }
}

/// The zero rate of change of a value of a tangent sort: the zero matrix of
/// a real sort, of its degree, and the only value of the singleton sort,
/// the tangent of the discrete sorts.
impl<T> crate::Differential<sort::Any> for Any<T> {
    fn zero(range: &sort::Any) -> Self {
        match range {
            sort::Any::Real(r) => Any::Real(real::Zero::with_degree(r.shape(), r.degree()).into()),
            sort::Any::Zero(_) => Any::Zero(Zero),
            _ => panic!("only the tangent sorts, reals and Zero, have a zero: {range}"),
        }
    }
}

impl<T> Clone for Any<T>
where
    real::NRA<T>: Clone,
    int::NIA<T>: Clone,
    bool::Prop<T>: Clone,
    word::BV<T>: Clone,
{
    fn clone(&self) -> Self {
        match self {
            Any::Real(op) => Any::Real(op.clone()),
            Any::Int(op) => Any::Int(op.clone()),
            Any::Bool(op) => Any::Bool(op.clone()),
            Any::Word(op) => Any::Word(op.clone()),
            Any::Havoc(op) => Any::Havoc(*op),
            Any::Skip(op) => Any::Skip(*op),
            Any::Zero(op) => Any::Zero(*op),
        }
    }
}

impl<T> Clone for NRA<T>
where
    real::NRA<T>: Clone,
    bool::Prop<T>: Clone,
{
    fn clone(&self) -> Self {
        match self {
            NRA::Real(op) => NRA::Real(op.clone()),
            NRA::Bool(op) => NRA::Bool(op.clone()),
        }
    }
}

impl<T> Clone for NIA<T>
where
    int::NIA<T>: Clone,
    bool::Prop<T>: Clone,
{
    fn clone(&self) -> Self {
        match self {
            NIA::Int(op) => NIA::Int(op.clone()),
            NIA::Bool(op) => NIA::Bool(op.clone()),
        }
    }
}

impl<T> Clone for Prop<T>
where
    bool::Prop<T>: Clone,
{
    fn clone(&self) -> Self {
        match self {
            Prop::Bool(op) => Prop::Bool(op.clone()),
        }
    }
}

impl<T> Clone for BV<T>
where
    bool::Prop<T>: Clone,
    word::BV<T>: Clone,
{
    fn clone(&self) -> Self {
        match self {
            BV::Bool(op) => BV::Bool(op.clone()),
            BV::Word(op) => BV::Word(op.clone()),
        }
    }
}

impl<T> Clone for LRA<T>
where
    real::LRA<T>: Clone,
    bool::Prop<T>: Clone,
{
    fn clone(&self) -> Self {
        match self {
            LRA::Real(op) => LRA::Real(op.clone()),
            LRA::Bool(op) => LRA::Bool(op.clone()),
        }
    }
}

impl<T> Clone for LIA<T>
where
    int::LIA<T>: Clone,
    bool::Prop<T>: Clone,
{
    fn clone(&self) -> Self {
        match self {
            LIA::Int(op) => LIA::Int(op.clone()),
            LIA::Bool(op) => LIA::Bool(op.clone()),
        }
    }
}

impl<T: fmt::Display> fmt::Display for Any<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Any::Real(op) => op.fmt(f),
            Any::Int(op) => op.fmt(f),
            Any::Bool(op) => op.fmt(f),
            Any::Word(op) => op.fmt(f),
            Any::Havoc(op) => op.fmt(f),
            Any::Skip(op) => op.fmt(f),
            Any::Zero(op) => op.fmt(f),
        }
    }
}

impl<T: Tensor> Boundary<sort::Any> for Any<T> {
    fn dom(&self) -> Poly<sort::Any> {
        match self {
            Any::Real(op) => op.dom().map(sort::Any::from),
            Any::Int(op) => op.dom().map(sort::Any::from),
            Any::Bool(op) => op.dom().map(sort::Any::from),
            Any::Word(op) => op.dom().map(sort::Any::from),
            Any::Havoc(op) => op.dom(),
            Any::Skip(op) => op.dom(),
            Any::Zero(op) => op.dom().map(sort::Any::from),
        }
    }

    fn cod(&self) -> Poly<sort::Any> {
        match self {
            Any::Real(op) => op.cod().map(sort::Any::from),
            Any::Int(op) => op.cod().map(sort::Any::from),
            Any::Bool(op) => op.cod().map(sort::Any::from),
            Any::Word(op) => op.cod().map(sort::Any::from),
            Any::Havoc(op) => op.cod(),
            Any::Skip(op) => op.cod(),
            Any::Zero(op) => op.cod().map(sort::Any::from),
        }
    }
}

/// Extracts an operation of any theory from a Python object, trying each
/// theory in turn through its own extraction: real, integer, boolean, and
/// then bit-vector operations. The operations of each theory are distinct
/// Python classes, so at most one theory accepts a given object.
#[cfg(feature = "pyo3")]
impl<'a, 'py, T> pyo3::FromPyObject<'a, 'py> for Any<T>
where
    real::NRA<T>: pyo3::FromPyObject<'a, 'py>,
    int::NIA<T>: pyo3::FromPyObject<'a, 'py>,
    bool::Prop<T>: pyo3::FromPyObject<'a, 'py>,
    word::BV<T>: pyo3::FromPyObject<'a, 'py>,
{
    type Error = pyo3::PyErr;

    fn extract(ob: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use pyo3::types::PyAnyMethods;

        if let Ok(op) = ob.extract::<real::NRA<T>>() {
            return Ok(Any::Real(op));
        }
        if let Ok(op) = ob.extract::<int::NIA<T>>() {
            return Ok(Any::Int(op));
        }
        if let Ok(op) = ob.extract::<bool::Prop<T>>() {
            return Ok(Any::Bool(op));
        }
        if let Ok(op) = ob.extract::<word::BV<T>>() {
            return Ok(Any::Word(op));
        }
        if let Ok(op) = ob.extract::<Havoc>() {
            return Ok(Any::Havoc(op));
        }
        if let Ok(op) = ob.extract::<Skip>() {
            return Ok(Any::Skip(op));
        }
        if let Ok(op) = ob.extract::<Zero>() {
            return Ok(Any::Zero(op));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "expected an operation of real, integer, boolean or bit-vector \
             arithmetic, got {}",
            ob.get_type()
        )))
    }
}

impl<T> From<LRA<T>> for Any<T> {
    fn from(op: LRA<T>) -> Self {
        match op {
            LRA::Real(op) => Any::Real(op.into()),
            LRA::Bool(op) => Any::Bool(op),
        }
    }
}

impl<T> TryFrom<Any<T>> for LRA<T> {
    type Error = String;

    fn try_from(op: Any<T>) -> Result<Self, Self::Error> {
        match op {
            Any::Real(op) => op
                .try_into()
                .map(LRA::Real)
                .map_err(|_| "LRA holds the linear real operations only".to_string()),
            Any::Bool(op) => Ok(LRA::Bool(op)),
            _ => Err("LRA holds real and boolean operations only".to_string()),
        }
    }
}

impl<T> From<LIA<T>> for Any<T> {
    fn from(op: LIA<T>) -> Self {
        match op {
            LIA::Int(op) => Any::Int(op.into()),
            LIA::Bool(op) => Any::Bool(op),
        }
    }
}

impl<T> TryFrom<Any<T>> for LIA<T> {
    type Error = String;

    fn try_from(op: Any<T>) -> Result<Self, Self::Error> {
        match op {
            Any::Int(op) => op
                .try_into()
                .map(LIA::Int)
                .map_err(|_| "LIA holds the linear integer operations only".to_string()),
            Any::Bool(op) => Ok(LIA::Bool(op)),
            _ => Err("LIA holds integer and boolean operations only".to_string()),
        }
    }
}

// The embeddings between logics, which `subenum` cannot express: it casts
// between `Any` and each logic, but not between two of them. `Prop` embeds
// into every other logic, and each linear logic into its non-linear one.
// Their inverses are absent, as nothing casts down between logics.
impl<T> From<Prop<T>> for NRA<T> {
    fn from(Prop::Bool(op): Prop<T>) -> Self {
        NRA::Bool(op)
    }
}

impl<T> From<Prop<T>> for NIA<T> {
    fn from(Prop::Bool(op): Prop<T>) -> Self {
        NIA::Bool(op)
    }
}

impl<T> From<Prop<T>> for LRA<T> {
    fn from(Prop::Bool(op): Prop<T>) -> Self {
        LRA::Bool(op)
    }
}

impl<T> From<Prop<T>> for LIA<T> {
    fn from(Prop::Bool(op): Prop<T>) -> Self {
        LIA::Bool(op)
    }
}

impl<T> From<Prop<T>> for BV<T> {
    fn from(Prop::Bool(op): Prop<T>) -> Self {
        BV::Bool(op)
    }
}

impl<T> From<LRA<T>> for NRA<T> {
    fn from(op: LRA<T>) -> Self {
        match op {
            LRA::Real(op) => NRA::Real(op.into()),
            LRA::Bool(op) => NRA::Bool(op),
        }
    }
}

impl<T> From<LIA<T>> for NIA<T> {
    fn from(op: LIA<T>) -> Self {
        match op {
            LIA::Int(op) => NIA::Int(op.into()),
            LIA::Bool(op) => NIA::Bool(op),
        }
    }
}
