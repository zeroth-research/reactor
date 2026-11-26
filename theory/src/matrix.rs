/*!
# Matrices

The matrix operations of each theory, one type per operation, in a module per
theory: [`real`] and [`int`] for arithmetic, [`bool`](mod@bool) for
propositional logic, and [`word`] for fixed-size bit-vectors. Each
operation reads and writes the sorts of [`Any`], and computes its
own boundary.

The operations are organised in two levels. Each theory module gathers its
operations in an enum named after the logic of that theory alone:
[`real::NRA`], [`int::NIA`], [`bool::Prop`] and [`word::BV`]. The
arithmetic ones mark their linear operations with a subenum named after
the linear logic: [`real::LRA`] and [`int::LIA`].
Here, [`Op`] holds the operations of all theories, one variant per
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
affine maps, hold it in matrices of a type `T` implementing [`Matrix`], over
which every operation, theory enum and logic is generic. With the `torch`
feature, `tch::Tensor` implements it, converting to and from Python
`torch.Tensor` objects (see the `torch` module).

## Sorts

[`Any`] holds the sorts of all theories, just as [`Op`] holds the
operations of all theories. `subenum` defines the sorts of each theory as a
subset of them, and generates the casts between `Any` and each subset:
`From` up to `Any`, and `TryFrom` down from it.

### Naming the subsets

The subsets are named after the SMT-LIB2 *theories*, and not after the
*logics*, such as [`LRA`] or [`BV`], which name subsets of the
operations. A theory fixes the sorts, and each logic restricts its
operations: the `L` of `LRA` stands for linear, a restriction on the
operations, so `LRA` and a non-linear `NRA` would share the same sorts.

Each subset is the plural of the sort its theory adds, as in SMT-LIB2's
`Reals` and `Ints`, so that the singular names an individual sort and the
plural the sorts of a theory:

| Subset | Sorts | SMT-LIB2 theory |
|---|---|---|
| [`Bools`] | [`Bool`], [`Real0`] | `Core` |
| [`Reals`] | [`Real`], [`Bool`], [`Real0`] | `Reals` |
| [`Ints`] | [`Int`], [`Bool`], [`Real0`] | `Ints` |
| [`Nums`] | [`Real`], [`Int`], [`Bool`], [`Real0`] | `Reals_Ints` |
| [`Words`] | [`Word`], [`Bool`], [`Real0`] | `FixedSizeBitVectors` |

[`Nums`], the sorts of both kinds of numbers, shortens `Reals_Ints`, and
[`Words`] follows our sort [`Word`] rather than SMT-LIB2's longer name.
[`Bools`] follows our sort [`Bool`] too: the theory of booleans is
SMT-LIB2's `Core`, a name too generic for its sorts, and the name of
propositional logic, [`Prop`], is taken by its operations.

Every subset includes [`Bool`], because every SMT-LIB2 theory includes the
Core theory, whose booleans are the sort that comparisons write; [`Bools`]
holds exactly the sorts that the others share.

Every subset also includes [`Real0`], which is not an SMT-LIB2 sort, but
the trivial tangent of the discrete sorts [`Bool`], [`Int`] and [`Word`],
and its own tangent. With it, each subset holds the tangents of its sorts,
and so is closed under taking tangents, as [`Any`] is.

The sort of each theory is defined, with its operations, in the module of
the theory: [`Real`] in [`real`], [`Int`] in [`int`], [`Bool`] in
[`bool`](mod@bool) and [`Word`] in [`word`]; their unions are defined here.
*/

pub mod bool;
pub mod int;
pub mod real;
pub mod word;

use self::{bool::Bool, int::Int, real::Real, word::Word};
use crate::zero::{self, Real0};
use base::boundary::{Boundary, Poly};
use base::differentiable::Differentiable;
use derive_more::From;
#[cfg(feature = "pyo3")]
use pyo3::{exceptions::PyTypeError, prelude::*};
use std::fmt;
use subenum::subenum;

/// A matrix of values, as held by the operations that carry data: the
/// constants, and the coefficients of `Affine`. The operations are generic
/// over it, so that any representation of matrices, such as torch tensors, can
/// back them.
pub trait Matrix {
    /// The `[rows, cols]` shape of the matrix.
    fn shape(&self) -> [usize; 2];
}

/// The sorts of all theories: a value of `Any` is a sort of one of them.
#[subenum(Bools, Reals, Ints, Nums, Words)]
#[derive(Clone, Copy, PartialEq, Debug, Eq, From)]
pub enum Any {
    #[subenum(Reals, Nums)]
    Real(Real),
    #[subenum(Ints, Nums)]
    Int(Int),
    #[subenum(Bools, Reals, Ints, Nums, Words)]
    Bool(Bool),
    #[subenum(Words)]
    Word(Word),
    #[subenum(Bools, Reals, Ints, Nums, Words)]
    Zero(Real0),
}

/// Each sort's tangent, taken within `Any`.
impl Differentiable for Any {
    type Tangent = Any;

    fn tangent(self) -> Any {
        match self {
            Any::Real(r) => r.tangent().into(),
            Any::Int(i) => i.tangent().into(),
            Any::Bool(b) => b.tangent().into(),
            Any::Word(w) => w.tangent().into(),
            Any::Zero(z) => z.tangent().into(),
        }
    }
}

impl fmt::Display for Any {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Any::Real(r) => r.fmt(f),
            Any::Int(i) => i.fmt(f),
            Any::Bool(b) => b.fmt(f),
            Any::Word(w) => w.fmt(f),
            Any::Zero(z) => z.fmt(f),
        }
    }
}

/// Each subset displays its sorts as [`Any`] does.
macro_rules! display_subset {
    ($($subset:ident),*) => {$(
        impl fmt::Display for $subset {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                Any::from(*self).fmt(f)
            }
        }
    )*};
}

display_subset!(Bools, Reals, Ints, Nums);

impl fmt::Display for Words {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Words::Word(w) => w.fmt(f),
            Words::Bool(b) => b.fmt(f),
            Words::Zero(z) => z.fmt(f),
        }
    }
}

/// All the sorts of words are discrete, so their tangent is trivial: the singleton
/// sort [`Real0`].
impl Differentiable for Words {
    type Tangent = Real0;

    fn tangent(self) -> Real0 {
        Real0
    }
}

/// Extracts a sort of any theory from an instance of its class.
#[cfg(feature = "pyo3")]
impl<'a, 'py> FromPyObject<'a, 'py> for Any {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(s) = ob.extract::<Real>() {
            return Ok(Any::Real(s));
        }
        if let Ok(s) = ob.extract::<Int>() {
            return Ok(Any::Int(s));
        }
        if let Ok(s) = ob.extract::<Bool>() {
            return Ok(Any::Bool(s));
        }
        if let Ok(s) = ob.extract::<Word>() {
            return Ok(Any::Word(s));
        }
        if let Ok(s) = ob.extract::<Real0>() {
            return Ok(Any::Zero(s));
        }
        Err(PyTypeError::new_err(format!(
            "expected a sort: Real, Int, Bool, Word or Real0, got {}",
            ob.get_type()
        )))
    }
}

/// Extracts a sort of the theories of each subset from an instance of its
/// class, as a sort of [`Any`] that the subset holds: the Python constructor
/// of `Ite` reads them.
#[cfg(feature = "pyo3")]
macro_rules! extract_subset {
    ($($subset:ident),*) => {$(
        impl<'a, 'py> FromPyObject<'a, 'py> for $subset {
            type Error = PyErr;

            fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
                let sort = ob.extract::<Any>()?;
                $subset::try_from(sort).map_err(|_| {
                    PyTypeError::new_err(format!(
                        "expected a sort of {}, got {sort}",
                        stringify!($subset)
                    ))
                })
            }
        }
    )*};
}

#[cfg(feature = "pyo3")]
extract_subset!(Reals, Ints, Words);

/// Converts a sort into an instance of its class.
#[cfg(feature = "pyo3")]
impl<'py> IntoPyObject<'py> for Any {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        match self {
            Any::Real(s) => s.into_pyobject(py).map(Bound::into_any),
            Any::Int(s) => s.into_pyobject(py).map(Bound::into_any),
            Any::Bool(s) => s.into_pyobject(py).map(Bound::into_any),
            Any::Word(s) => s.into_pyobject(py).map(Bound::into_any),
            Any::Zero(s) => s.into_pyobject(py).map(Bound::into_any),
        }
    }
}

/// The operations of all theories: a value of `Op` is an operation of one
/// of them, on the sorts in [`Any`], with its data held in matrices of
/// type `T`; or one of the structural operations [`Havoc`], [`Skip`] and
/// [`Zero`](crate::zero::Zero), which belong to no theory and so to no logic.
///
/// `subenum` defines the logics that take whole theories: [`Prop`],
/// propositional logic; [`NRA`] and [`NIA`], non-linear real and integer
/// arithmetic; and [`BV`], the logic `QF_BV` of fixed-size bit-vectors. It
/// also generates their casts: `From` up to `Op`, and `TryFrom` down from
/// it. The linear logics [`LRA`] and [`LIA`] restrict a theory to its
/// linear operations, which `subenum` cannot express, so they are defined
/// on their own, with the same casts.
///
/// `Op` is cloneable when the constants and affine maps of its matrix type
/// are: `Clone` is implemented by hand rather than derived, as a derive would
/// require `T: Clone`, which matrix types such as `tch::Tensor` are not.
#[subenum(NRA, NIA, Prop, BV)]
#[derive(Debug, From)]
pub enum Op<T> {
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
    Zero(zero::Zero),
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
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.matrix")
)]
pub struct Havoc {
    sort: Any,
}

/// Writes the value it reads, of the given sort: the identity.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.matrix")
)]
pub struct Skip {
    sort: Any,
}

/// Fails unless `shape` has at least one row and one column: a matrix of no
/// entries has exactly one value, the vector of no coordinates, so that its
/// sort would be another copy of [`Real0`](crate::zero::Real0).
pub(crate) fn check_shape(shape: [usize; 2]) -> Result<(), crate::Error> {
    if shape.contains(&0) {
        return Err(crate::Error(format!(
            "matrices have at least one row and one column, got the shape {shape:?}"
        )));
    }
    Ok(())
}

/// Fails unless `a` and `b` are the shapes `[m, n]` and `[m, k]` of the
/// matrices `A` and `B` of an affine map `x ↦ A·x + B`, none of them empty.
pub(crate) fn check_affine(a: [usize; 2], b: [usize; 2]) -> Result<(), crate::Error> {
    check_shape(a)?;
    check_shape(b)?;
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
pub(crate) fn py_sorts<'py, S: Into<Any>>(
    py: Python<'py>,
    poly: Poly<S>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut monos = poly.into_iter();
    let (Some(mono), None) = (monos.next(), monos.next()) else {
        panic!("the boundary of an operation is a single product of sorts");
    };
    let mut sorts: Vec<Any> = mono.into_iter().map(Into::into).collect();
    if sorts.len() == 1 {
        sorts.pop().unwrap().into_pyobject(py)
    } else {
        Ok(sorts.into_pyobject(py)?.into_any())
    }
}

impl Boundary<Any> for Havoc {
    fn dom(&self) -> Poly<Any> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Any> {
        base::mono![self.sort].into()
    }
}

impl Boundary<Any> for Skip {
    fn dom(&self) -> Poly<Any> {
        base::mono![self.sort].into()
    }

    fn cod(&self) -> Poly<Any> {
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
    pub fn with_cod(cod: [Any; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Havoc { sort: s })
    }
}

impl Skip {
    /// The identity reading the sort of `dom`.
    pub fn with_dom(dom: [Any; 1]) -> Result<Self, crate::Error> {
        let [s] = dom;
        Ok(Skip { sort: s })
    }

    /// The identity writing the sort of `cod`.
    pub fn with_cod(cod: [Any; 1]) -> Result<Self, crate::Error> {
        let [s] = cod;
        Ok(Skip { sort: s })
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Havoc {
    /// Python builds it from the parameters that its boundary does not determine,
    /// and its codomain `cod`.
    #[cfg(feature = "pyo3")]
    #[new]
    #[pyo3(signature = (cod))]
    fn py_new(cod: Any) -> Result<Self, crate::Error> {
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
        crate::matrix::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn sort(&self) -> Any {
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
    fn py_new(dom: Any, cod: Option<Any>) -> Result<Self, crate::Error> {
        crate::matrix::check_cod(Self::with_dom([dom]), cod)
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
        crate::matrix::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
        crate::matrix::py_sorts(py, Boundary::cod(self))
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn sort(&self) -> Any {
        self.sort
    }
}

impl<T> crate::Combinatorial<Any> for Op<T> {
    fn havoc(range: &Any) -> Self {
        Op::Havoc(Havoc { sort: *range })
    }
}

impl<T> crate::Sequential<Any> for Op<T> {
    fn skip(range: &Any) -> Self {
        Op::Skip(Skip { sort: *range })
    }
}

/// The zero rate of change of a value of a tangent sort: the zero matrix of
/// a real sort, of its degree, and the only value of the singleton sort,
/// the tangent of the discrete sorts.
impl<T> crate::Differential<Any> for Op<T> {
    fn zero(range: &Any) -> Self {
        match range {
            Any::Real(r) => Op::Real(
                real::Zero::with_cod([*r])
                    .expect("the zero of a real writes it")
                    .into(),
            ),
            Any::Zero(_) => Op::Zero(zero::Zero),
            _ => panic!("only the tangent sorts, reals and Zero, have a zero: {range}"),
        }
    }
}

impl<T> Clone for Op<T>
where
    real::NRA<T>: Clone,
    int::NIA<T>: Clone,
    bool::Prop<T>: Clone,
    word::BV<T>: Clone,
{
    fn clone(&self) -> Self {
        match self {
            Op::Real(op) => Op::Real(op.clone()),
            Op::Int(op) => Op::Int(op.clone()),
            Op::Bool(op) => Op::Bool(op.clone()),
            Op::Word(op) => Op::Word(op.clone()),
            Op::Havoc(op) => Op::Havoc(*op),
            Op::Skip(op) => Op::Skip(*op),
            Op::Zero(op) => Op::Zero(*op),
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

impl<T: fmt::Display> fmt::Display for Op<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::Real(op) => op.fmt(f),
            Op::Int(op) => op.fmt(f),
            Op::Bool(op) => op.fmt(f),
            Op::Word(op) => op.fmt(f),
            Op::Havoc(op) => op.fmt(f),
            Op::Skip(op) => op.fmt(f),
            Op::Zero(op) => op.fmt(f),
        }
    }
}

impl<T: Matrix> Boundary<Any> for Op<T> {
    fn dom(&self) -> Poly<Any> {
        match self {
            Op::Real(op) => op.dom().map(Any::from),
            Op::Int(op) => op.dom().map(Any::from),
            Op::Bool(op) => op.dom().map(Any::from),
            Op::Word(op) => op.dom().map(Any::from),
            Op::Havoc(op) => op.dom(),
            Op::Skip(op) => op.dom(),
            Op::Zero(op) => op.dom().map(Any::from),
        }
    }

    fn cod(&self) -> Poly<Any> {
        match self {
            Op::Real(op) => op.cod().map(Any::from),
            Op::Int(op) => op.cod().map(Any::from),
            Op::Bool(op) => op.cod().map(Any::from),
            Op::Word(op) => op.cod().map(Any::from),
            Op::Havoc(op) => op.cod(),
            Op::Skip(op) => op.cod(),
            Op::Zero(op) => op.cod().map(Any::from),
        }
    }
}

/// Extracts an operation of any theory from a Python object, trying each
/// theory in turn through its own extraction: real, integer, boolean, and
/// then bit-vector operations. The operations of each theory are distinct
/// Python classes, so at most one theory accepts a given object.
#[cfg(feature = "pyo3")]
impl<'a, 'py, T> pyo3::FromPyObject<'a, 'py> for Op<T>
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
            return Ok(Op::Real(op));
        }
        if let Ok(op) = ob.extract::<int::NIA<T>>() {
            return Ok(Op::Int(op));
        }
        if let Ok(op) = ob.extract::<bool::Prop<T>>() {
            return Ok(Op::Bool(op));
        }
        if let Ok(op) = ob.extract::<word::BV<T>>() {
            return Ok(Op::Word(op));
        }
        if let Ok(op) = ob.extract::<Havoc>() {
            return Ok(Op::Havoc(op));
        }
        if let Ok(op) = ob.extract::<Skip>() {
            return Ok(Op::Skip(op));
        }
        if let Ok(op) = ob.extract::<zero::Zero>() {
            return Ok(Op::Zero(op));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "expected an operation of real, integer, boolean or bit-vector \
             arithmetic, got {}",
            ob.get_type()
        )))
    }
}

impl<T> From<LRA<T>> for Op<T> {
    fn from(op: LRA<T>) -> Self {
        match op {
            LRA::Real(op) => Op::Real(op.into()),
            LRA::Bool(op) => Op::Bool(op),
        }
    }
}

impl<T> TryFrom<Op<T>> for LRA<T> {
    type Error = String;

    fn try_from(op: Op<T>) -> Result<Self, Self::Error> {
        match op {
            Op::Real(op) => op
                .try_into()
                .map(LRA::Real)
                .map_err(|_| "LRA holds the linear real operations only".to_string()),
            Op::Bool(op) => Ok(LRA::Bool(op)),
            _ => Err("LRA holds real and boolean operations only".to_string()),
        }
    }
}

impl<T> From<LIA<T>> for Op<T> {
    fn from(op: LIA<T>) -> Self {
        match op {
            LIA::Int(op) => Op::Int(op.into()),
            LIA::Bool(op) => Op::Bool(op),
        }
    }
}

impl<T> TryFrom<Op<T>> for LIA<T> {
    type Error = String;

    fn try_from(op: Op<T>) -> Result<Self, Self::Error> {
        match op {
            Op::Int(op) => op
                .try_into()
                .map(LIA::Int)
                .map_err(|_| "LIA holds the linear integer operations only".to_string()),
            Op::Bool(op) => Ok(LIA::Bool(op)),
            _ => Err("LIA holds integer and boolean operations only".to_string()),
        }
    }
}

// The embeddings between logics, which `subenum` cannot express: it casts
// between `Op` and each logic, but not between two of them. `Prop` embeds
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
