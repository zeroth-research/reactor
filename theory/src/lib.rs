/*!
# Theories

The operations of each theory, one type per operation, in a module per
theory: [`real`] and [`int`] for arithmetic, [`bool`](mod@bool) for
propositional logic, and [`word`] for fixed-size bit-vectors, whose
operations act on matrices; and `enum` for the enumerations of
symbolic constants, and [`event`] for the events, booleans whose change
signals them, each its own sort, whose operations act on values of no shape and read and write
the booleans of the shape `[1, 1]`. Each module defines the sort of its
theory, with its operations, which read and write the sorts of [`Any`] and
compute their own boundary.

The operations are organised in two levels. Each theory module gathers its
operations in an enum named after the logic of that theory alone:
[`real::NRA`], [`int::NIA`], [`bool::Prop`], [`word::BV`] and `enum::Op`,
while the events have their one operation, [`event::Xor`]. The arithmetic ones mark their linear operations with a
subenum named after the linear logic: [`real::LRA`] and [`int::LIA`].
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

The structural operations, [`Havoc`], [`Skip`] and [`zero::Zero`], belong
to no theory and so to no logic.

The operations that carry data, the constants and the coefficients of the
affine maps, hold it in matrices of a type `T` implementing [`Matrix`], over
which every operation, theory enum and logic is generic. With the `torch`
feature, `torch::Tensor`, a `tch::Tensor` whose clone shares its storage,
implements it, converting to and from Python `torch.Tensor` objects (see
the `torch` module).

## Sorts

[`Any`] holds the sorts of all theories, just as [`Op`] holds the
operations of all theories. `subenum` defines the sorts of the matrix
theories as subsets of them, and generates the casts between `Any` and each
subset: `From` up to `Any`, and `TryFrom` down from it.

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
| [`Bools`] | [`Bool`], [`Event`], [`Real0`] | `Core`, an event a `Bool` |
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
Core theory, whose booleans are the sort that comparisons write. [`Bools`]
holds the events too, as SMT-LIB2 encodes an event as a `Bool`, so that
their operation, [`event::Xor`], reads and writes the sorts of [`Bools`].

Every subset also includes [`Real0`], which is not an SMT-LIB2 sort, but
the trivial tangent of the discrete sorts [`Bool`], [`Int`] and [`Word`],
and its own tangent. With it, each subset holds the tangents of its sorts,
and so is closed under taking tangents, as [`Any`] is.

The subsets hold sorts that are `Copy`, and so are `Copy` too; [`Any`] is
not, as an [`Enum`] holds the names of its values.
*/

pub mod bool;
pub mod r#enum;
pub mod event;
pub mod int;
pub mod matrix;
pub mod real;
#[cfg(feature = "torch")]
pub mod torch;
pub mod word;
pub mod zero;

pub use matrix::Matrix;

use self::{bool::Bool, r#enum::Enum, event::Event, int::Int, real::Real, word::Word};
#[cfg(feature = "pyo3")]
use base::boundary::Mono;
use base::boundary::{Boundary, Poly};
use base::differentiable::Differentiable;
use derive_more::{Display, From};
#[cfg(feature = "pyo3")]
use pyo3::{exceptions::PyTypeError, prelude::*};
use std::fmt;
use subenum::subenum;
use zero::Real0;

/// The sorts of all theories: a value of `Any` is a sort of one of them, or
/// [`Real0`], which they all share.
#[subenum(
    Bools(derive(Copy)),
    Reals(derive(Copy)),
    Ints(derive(Copy)),
    Nums(derive(Copy)),
    Words(derive(Copy))
)]
#[derive(Clone, PartialEq, Debug, Eq, From, Display)]
pub enum Any {
    #[subenum(Reals, Nums)]
    Real(Real),
    #[subenum(Ints, Nums)]
    Int(Int),
    #[subenum(Bools, Reals, Ints, Nums, Words)]
    Bool(Bool),
    #[subenum(Words)]
    Word(Word),
    Enum(Enum),
    #[subenum(Bools)]
    Event(Event),
    #[subenum(Bools, Reals, Ints, Nums, Words)]
    Zero(Real0),
}

/// Each sort's tangent, taken within `Any`: a real of one degree higher, and
/// [`Real0`] for the discrete sorts and itself.
impl Differentiable for Any {
    type Tangent = Any;

    fn tangent(self) -> Any {
        match self {
            Any::Real(r) => r.tangent().into(),
            Any::Int(i) => i.tangent().into(),
            Any::Bool(b) => b.tangent().into(),
            Any::Word(w) => w.tangent().into(),
            Any::Enum(e) => e.tangent().into(),
            Any::Event(e) => e.tangent().into(),
            Any::Zero(z) => z.tangent().into(),
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

/// Extracts a sort from an instance of its class.
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
        if let Ok(s) = ob.extract::<Enum>() {
            return Ok(Any::Enum(s));
        }
        if let Ok(s) = ob.extract::<Event>() {
            return Ok(Any::Event(s));
        }
        if let Ok(s) = ob.extract::<Real0>() {
            return Ok(Any::Zero(s));
        }
        Err(PyTypeError::new_err(format!(
            "expected a sort: Real, Int, Bool, Word, Enum, Event or Real0, got {}",
            ob.get_type()
        )))
    }
}

/// Extracts a sort of the theories of each subset from an instance of its
/// class, as a sort of [`Any`] that the subset holds: the Python constructor
/// of `Ite`, of the matrix theories, and of `event::Xor` read them. Converts
/// it back into its class, as [`Any`] does: the getters `dom` and `cod` of the
/// operations give the sorts of the subset of their theory.
#[cfg(feature = "pyo3")]
macro_rules! py_subset {
    ($($subset:ident),*) => {$(
        impl<'py> IntoPyObject<'py> for $subset {
            type Target = PyAny;
            type Output = Bound<'py, PyAny>;
            type Error = PyErr;

            fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
                Any::from(self).into_pyobject(py)
            }
        }

        impl<'a, 'py> FromPyObject<'a, 'py> for $subset {
            type Error = PyErr;

            fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
                let sort = ob.extract::<Any>()?;
                $subset::try_from(sort.clone()).map_err(|_| {
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
py_subset!(Bools, Reals, Ints, Words);

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
            Any::Enum(s) => s.into_pyobject(py).map(Bound::into_any),
            Any::Event(s) => s.into_pyobject(py).map(Bound::into_any),
            Any::Zero(s) => s.into_pyobject(py).map(Bound::into_any),
        }
    }
}

/// The operations of all theories: a value of `Op` is an operation of one
/// of them, on the sorts in [`Any`], with the data of the matrix theories
/// held in matrices of type `T`; or one of the structural operations
/// [`Havoc`], [`Skip`] and [`zero::Zero`], which belong to no theory and so
/// to no logic.
///
/// `subenum` defines the logics that take whole theories: [`Prop`],
/// propositional logic; [`NRA`] and [`NIA`], non-linear real and integer
/// arithmetic; and [`BV`], the logic `QF_BV` of fixed-size bit-vectors. It
/// also generates their casts: `From` up to `Op`, and `TryFrom` down from
/// it. The linear logics [`LRA`] and [`LIA`] restrict a theory to its
/// linear operations, which `subenum` cannot express, so they are defined
/// on their own, with the same casts.
///
/// `Op` is cloneable when its matrix type `T` is, as are the operations and
/// logics of each theory.
#[subenum(
    NRA(derive(Copy)),
    NIA(derive(Copy)),
    Prop(derive(Copy)),
    BV(derive(Copy))
)]
#[derive(Clone, Debug, From, Display)]
pub enum Op<T> {
    #[subenum(NRA)]
    Real(real::NRA<T>),
    #[subenum(NIA)]
    Int(int::NIA<T>),
    #[subenum(NRA, NIA, Prop, BV)]
    Bool(bool::Prop<T>),
    #[subenum(BV)]
    Word(word::BV<T>),
    Enum(r#enum::Op),
    Event(event::Xor),
    Havoc(Havoc),
    Skip(Skip),
    Zero(zero::Zero),
}

/// Linear real arithmetic: the linear operations of [`real`], with the
/// boolean operations.
#[derive(Clone, Copy, Debug, From)]
pub enum LRA<T> {
    Real(real::LRA<T>),
    Bool(bool::Prop<T>),
}

/// Linear integer arithmetic: the linear operations of [`int`], with the
/// boolean operations.
#[derive(Clone, Copy, Debug, From)]
pub enum LIA<T> {
    Int(int::LIA<T>),
    Bool(bool::Prop<T>),
}

/// Writes an arbitrary value of the given sort, of any theory, reading
/// nothing: the nondeterministic choice of a value.
#[derive(Clone, PartialEq, Debug, Eq, Display)]
#[display("Havoc")]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor")
)]
pub struct Havoc {
    sort: Any,
}

/// Writes the value it reads, of the given sort, of any theory: the identity.
#[derive(Clone, PartialEq, Debug, Eq, Display)]
#[display("Skip")]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor")
)]
pub struct Skip {
    sort: Any,
}

/// The failure of a `with_dom` constructor: `dom` is not the domain of an
/// instance of the operation, which reads the sorts described by `expected`.
pub(crate) fn unexpected_dom<S: fmt::Display>(expected: &str, dom: &[S]) -> Error {
    unexpected("domain", expected, dom)
}

/// The failure of a `with_cod` constructor: `cod` is not the codomain of an
/// instance of the operation, which writes the sorts described by `expected`.
pub(crate) fn unexpected_cod<S: fmt::Display>(expected: &str, cod: &[S]) -> Error {
    unexpected("codomain", expected, cod)
}

fn unexpected<S: fmt::Display>(boundary: &str, expected: &str, sorts: &[S]) -> Error {
    let sorts: Vec<String> = sorts.iter().map(ToString::to_string).collect();
    Error(format!(
        "expected a {boundary} of {expected}, got [{}]",
        sorts.join(", ")
    ))
}

/// Fails unless the sorts of `dom`, at least one, are all equal, and
/// returns that sort: the domain of an operation whose inputs share it.
pub(crate) fn same_sort<S: Copy + PartialEq + fmt::Display, const N: usize>(
    dom: [S; N],
) -> Result<S, Error> {
    let s = dom[0];
    if dom.iter().any(|t| *t != s) {
        return Err(unexpected_dom(&format!("{N} equal sorts"), &dom));
    }
    Ok(s)
}

/// The operation a Python constructor builds from its domain, `op`, checked
/// against its codomain `cod` when given: that must be the codomain of `op`.
#[cfg(feature = "pyo3")]
pub(crate) fn check_cod<Op, S, C>(op: Result<Op, Error>, cod: Option<C>) -> Result<Op, Error>
where
    Op: Boundary<S>,
    S: Clone + PartialEq + fmt::Display,
    C: Into<S> + fmt::Display,
{
    let op = op?;
    if let Some(c) = cod {
        let expected: Poly<S> = base::mono![c.into()].into();
        if op.cod() != expected {
            return Err(Error(format!(
                "the operation reading this domain writes {}, not {expected}",
                op.cod()
            )));
        }
    }
    Ok(op)
}

impl Boundary<Any> for Havoc {
    fn dom(&self) -> Poly<Any> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Any> {
        base::mono![self.sort.clone()].into()
    }
}

impl Boundary<Any> for Skip {
    fn dom(&self) -> Poly<Any> {
        base::mono![self.sort.clone()].into()
    }

    fn cod(&self) -> Poly<Any> {
        base::mono![self.sort.clone()].into()
    }
}

impl Havoc {
    /// The choice of a value of the sort of `cod`.
    pub fn with_cod(cod: [Any; 1]) -> Result<Self, Error> {
        let [s] = cod;
        Ok(Havoc { sort: s })
    }
}

impl Skip {
    /// The identity reading the sort of `dom`.
    pub fn with_dom(dom: [Any; 1]) -> Result<Self, Error> {
        let [s] = dom;
        Ok(Skip { sort: s })
    }

    /// The identity writing the sort of `cod`.
    pub fn with_cod(cod: [Any; 1]) -> Result<Self, Error> {
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
    fn py_new(cod: Any) -> Result<Self, Error> {
        Self::with_cod([cod])
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("cod",)
    }

    /// The sorts it writes, as a tuple.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod(&self) -> Mono<Any> {
        Mono::try_from(Boundary::cod(self)).unwrap()
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
    fn py_new(dom: Any, cod: Option<Any>) -> Result<Self, Error> {
        check_cod(Self::with_dom([dom]), cod)
    }

    /// The parameters that its boundary does not determine, then its domain, when it
    /// reads anything, and its codomain.
    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads, as a tuple.
    #[cfg(feature = "pyo3")]
    #[getter(dom)]
    fn py_dom(&self) -> Mono<Any> {
        Mono::try_from(Boundary::dom(self)).unwrap()
    }

    /// The sorts it writes, as a tuple.
    #[cfg(feature = "pyo3")]
    #[getter(cod)]
    fn py_cod(&self) -> Mono<Any> {
        Mono::try_from(Boundary::cod(self)).unwrap()
    }
}

/// The nondeterministic choice of a value of any sort.
impl<T: Matrix> Combinatorial<Any> for Op<T> {
    fn havoc(range: &Any) -> Self {
        Op::Havoc(Havoc {
            sort: range.clone(),
        })
    }
}

/// The identity of any sort.
impl<T: Matrix> Sequential<Any> for Op<T> {
    fn skip(range: &Any) -> Self {
        Op::Skip(Skip {
            sort: range.clone(),
        })
    }
}

/// The zero rate of change of a value of a tangent sort: the zero matrix of
/// a real sort, of its degree, and the only value of the singleton sort,
/// the tangent of the discrete sorts.
///
/// Panics for any other sort, which is no rate of change.
impl<T: Matrix> Differential<Any> for Op<T> {
    fn zero(range: &Any) -> Self {
        match range {
            Any::Real(r) => Op::Real(
                real::Zero::with_cod([*r])
                    .expect("the zero of a real writes it")
                    .into(),
            ),
            Any::Zero(_) => Op::Zero(zero::Zero),
            _ => panic!("only the tangent sorts, reals and Real0, have a zero: {range}"),
        }
    }
}

/// The boundary of each operation, lifted into the sorts of all theories.
impl<T: Matrix> Boundary<Any> for Op<T> {
    fn dom(&self) -> Poly<Any> {
        match self {
            Op::Real(op) => op.dom().map(Any::from),
            Op::Int(op) => op.dom().map(Any::from),
            Op::Bool(op) => op.dom().map(Any::from),
            Op::Word(op) => op.dom().map(Any::from),
            Op::Enum(op) => op.dom(),
            Op::Event(op) => op.dom().map(Any::from),
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
            Op::Enum(op) => op.cod(),
            Op::Event(op) => op.cod().map(Any::from),
            Op::Havoc(op) => op.cod(),
            Op::Skip(op) => op.cod(),
            Op::Zero(op) => op.cod().map(Any::from),
        }
    }
}

/// Extracts an operation from a Python object, trying each theory in turn
/// through its own extraction: real, integer, boolean, bit-vector,
/// enumeration and event operations, and then the structural ones. The operations of each theory
/// are distinct Python classes, so at most one theory accepts a given
/// object.
#[cfg(feature = "pyo3")]
impl<'a, 'py, T> FromPyObject<'a, 'py> for Op<T>
where
    real::NRA<T>: FromPyObject<'a, 'py>,
    int::NIA<T>: FromPyObject<'a, 'py>,
    bool::Prop<T>: FromPyObject<'a, 'py>,
    word::BV<T>: FromPyObject<'a, 'py>,
{
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
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
        if let Ok(op) = ob.extract::<r#enum::Op>() {
            return Ok(Op::Enum(op));
        }
        if let Ok(op) = ob.extract::<event::Xor>() {
            return Ok(Op::Event(op));
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
        Err(PyTypeError::new_err(format!(
            "expected an operation of a theory, or a structural one, got {}",
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

/// The failure of a partial constructor, with its message: raised in Python
/// as a `ValueError`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

#[cfg(feature = "pyo3")]
impl From<Error> for pyo3::PyErr {
    fn from(e: Error) -> Self {
        pyo3::exceptions::PyValueError::new_err(e.0)
    }
}

/// The instructions of a theory that choose a value; as the other traits of
/// the structural instructions, it carries their boundary, which the terms
/// built of them check against their wires.
pub trait Combinatorial<S>: Boundary<S> {
    /// The itype that chooses an element within the range: reading nothing,
    /// and writing one value of `range`.
    fn havoc(range: &S) -> Self;
}

pub trait Sequential<S>: Boundary<S> {
    /// The itype that copies a value within the range: reading one value of
    /// `range`, and writing one.
    fn skip(range: &S) -> Self;
}

pub trait Differential<S>: Boundary<S> {
    /// The itype that indicates zero derivative: reading nothing, and writing
    /// one value of `range`.
    fn zero(range: &S) -> Self;
}
