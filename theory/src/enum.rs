/*!
# Enumerations

Defines the sort [`Enum`] of enumerations of symbolic constants, and one
type per operation on it, as in nuXmv, whose symbolic constants admit only
equality and choice. Every operation carries the enumerations it reads or
writes, and so enough information to compute its [`Boundary`]:

- [`Sym`] — a symbolic constant of an enumeration.
- [`Into`] — a value of an enumeration, as one of another that holds all its
  values.
- [`Eq`](struct@Eq), [`Ne`] — equality and disequality of two values of an
  enumeration, writing a [`Bool`] of the shape `[1, 1]`.
- [`Ite`] — the choice `if c then x else y` between two values of an
  enumeration, by a [`Bool`] `c` of the shape `[1, 1]`.

SMT-LIB2 has no anonymous enumerations: it declares an enumeration as a
datatype of nullary constructors, `(declare-datatypes ((E 0)) (((idle)
(busy))))`, whose constructors are its constants.

As in SMT-LIB2, the comparisons and the choice read values of a single
enumeration. Values of different enumerations meet through [`Into`], which
keeps the name of a value, as SMT-LIB2's `to_real` keeps the value of an
integer: a value of `{a, b}` and one of `{b, c}` compare as values of `{a,
b, c}`, equal when both are `b`, as in nuXmv. In SMT-LIB2, `Into` is the
function defined by cases from the datatype of its domain to that of its
codomain, or the identity when all the symbolic constants of a model are of
one datatype, each enumeration a constraint of membership.
*/

pub use crate::Any as Sort;
use crate::bool::Bool;
use crate::zero::Real0;
use base::boundary::{Boundary, Poly};
use base::differentiable::Differentiable;
use base::mono;
use derive_more::From;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use std::fmt;
// `Into`, the operation, shadows the trait of the prelude, whose `into` this
// module still calls.
use std::convert::Into as _;

/// An enumeration: the sort whose values are the symbolic constants it names,
/// as nuXmv's symbolic enumeration types, such as `{idle, busy, done}`.
///
/// An enumeration is anonymous: it is identified by its values alone, as a
/// set, so that two enumerations of the same values, in whatever order, are
/// the same sort. It holds them sorted and without repetitions.
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.enum")
)]
pub struct Enum {
    /// The values, strictly increasing: sorted, without repetitions, and never
    /// empty. Membership, inclusion and union rely on it.
    values: Vec<String>,
}

impl Enum {
    /// The enumeration of the symbolic constants `values`, each given once:
    /// their order does not matter. Fails when there are none, as a sort with
    /// no values cannot be written, or when one is repeated.
    pub fn new<I>(values: I) -> Result<Self, crate::Error>
    where
        I: IntoIterator,
        I::Item: std::convert::Into<String>,
    {
        let mut values: Vec<String> = values.into_iter().map(std::convert::Into::into).collect();
        values.sort();
        if values.is_empty() {
            return Err(crate::Error(
                "an enumeration must have at least one value".to_string(),
            ));
        }
        if let Some(w) = values.windows(2).find(|w| w[0] == w[1]) {
            return Err(crate::Error(format!(
                "the value {:?} of an enumeration is repeated",
                w[0]
            )));
        }
        Ok(Enum { values })
    }

    /// The symbolic constants of the enumeration, sorted.
    pub fn values(&self) -> &[String] {
        &self.values
    }

    /// Whether `value` is one of the symbolic constants of the enumeration.
    pub fn contains(&self, value: &str) -> bool {
        self.values
            .binary_search_by(|v| v.as_str().cmp(value))
            .is_ok()
    }

    /// The number of its values.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether `other` holds all the values of `self`: a single walk through
    /// both, sorted.
    pub fn is_subset(&self, other: &Enum) -> bool {
        let mut theirs = other.values.iter();
        self.values
            .iter()
            .all(|v| theirs.by_ref().find(|w| *w >= v) == Some(v))
    }

    /// The enumeration of the values of both `self` and `other`: the merge of
    /// both, sorted, keeping one of the values they share.
    pub fn union(&self, other: &Enum) -> Enum {
        let (mut xs, mut ys) = (
            self.values.iter().peekable(),
            other.values.iter().peekable(),
        );
        let mut values = Vec::with_capacity(self.len() + other.len());
        loop {
            let next = match (xs.peek(), ys.peek()) {
                (Some(x), Some(y)) if x < y => xs.next(),
                (Some(x), Some(y)) if y < x => ys.next(),
                (Some(_), Some(_)) => {
                    ys.next();
                    xs.next()
                }
                (Some(_), None) => xs.next(),
                (None, Some(_)) => ys.next(),
                (None, None) => break,
            };
            values.extend(next.cloned());
        }
        Enum { values }
    }

    /// Never: an enumeration has at least one value.
    pub fn is_empty(&self) -> bool {
        false
    }
}

/// Python builds it from its values, any iterable of distinct strings, and
/// matches it by them, sorted; `|` is the union of enumerations, and `in` the
/// membership of a value.
#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Enum {
    /// A string is not taken for the iterable of its characters: the values
    /// of an enumeration are names, never letters.
    #[cfg(feature = "pyo3")]
    #[new]
    fn py_new(values: &Bound<'_, PyAny>) -> PyResult<Self> {
        if values.is_instance_of::<pyo3::types::PyString>() {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "expected an iterable of the values of the enumeration, got a str",
            ));
        }
        let values = values
            .try_iter()?
            .map(|v| v?.extract::<String>())
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self::new(values)?)
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("values",)
    }

    /// The symbolic constants of the enumeration, sorted.
    #[cfg(feature = "pyo3")]
    #[getter(values)]
    fn py_values(&self) -> Vec<String> {
        self.values.clone()
    }

    /// The tangent sort `T(self)`: the sort of the rate of change of its
    /// values.
    #[cfg(feature = "pyo3")]
    #[pyo3(name = "tangent")]
    fn py_tangent(&self) -> Real0 {
        Differentiable::tangent(self)
    }

    /// The union `self | other`: the enumeration of the values of both.
    #[cfg(feature = "pyo3")]
    fn __or__(&self, other: &Self) -> Self {
        self.union(other)
    }

    /// The membership `value in self`: whether `value` is one of its
    /// symbolic constants; never for a value that is not a string.
    #[cfg(feature = "pyo3")]
    fn __contains__(&self, value: &Bound<'_, PyAny>) -> bool {
        value
            .extract::<&str>()
            .is_ok_and(|value| self.contains(value))
    }
}

/// Writes it as it is built, with its values sorted: `Enum(["busy", "idle"])`.
impl fmt::Display for Enum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Enum([")?;
        for (i, v) in self.values.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{v:?}")?;
        }
        write!(f, "])")
    }
}

/// The enumerations are discrete, so their tangent is trivial: the singleton
/// sort [`Real0`].
impl Differentiable for Enum {
    type Tangent = Real0;

    fn tangent(self) -> Real0 {
        Real0
    }
}

/// The tangent of a borrowed enumeration, which is not `Copy`: that of the
/// enumeration.
impl Differentiable for &Enum {
    type Tangent = Real0;

    fn tangent(self) -> Real0 {
        Real0
    }
}

/// A symbolic constant of an enumeration: one of its values.
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.enum")
)]
pub struct Sym {
    value: String,
    sort: Enum,
}

impl Sym {
    /// The constant `value`, writing the enumeration of `cod`; fails unless
    /// `value` is one of its values.
    pub fn with_cod(value: &str, cod: [Enum; 1]) -> Result<Self, crate::Error> {
        let [sort] = cod;
        if !sort.contains(value) {
            return Err(crate::unexpected_cod(
                &format!("1 enumeration of the value {value:?}"),
                &[sort],
            ));
        }
        Ok(Sym {
            value: value.to_string(),
            sort,
        })
    }
}

/// A value of the enumeration `dom`, as the value of the same name of the
/// enumeration `cod`, which holds all the values of `dom`: total, and the
/// identity on names.
///
/// SMT-LIB2: the function from the datatype of `dom` to that of `cod` mapping
/// each constructor to the one of its name.
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.enum")
)]
pub struct Into {
    dom: Enum,
    cod: Enum,
}

impl Into {
    /// The conversion reading the enumeration `dom` and writing `cod`; fails
    /// unless `cod` holds all the values of `dom`.
    pub fn new(dom: Enum, cod: Enum) -> Result<Self, crate::Error> {
        if !dom.is_subset(&cod) {
            return Err(crate::Error(format!(
                "expected a codomain holding all the values of {dom}, got {cod}"
            )));
        }
        Ok(Into { dom, cod })
    }
}

/// Equality `x = y` of two values of an enumeration.
///
/// SMT-LIB2: `(= x y)`.
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.enum")
)]
pub struct Eq {
    sort: Enum,
}

/// Disequality `x ≠ y` of two values of an enumeration.
///
/// SMT-LIB2: `(distinct x y)`.
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.enum")
)]
pub struct Ne {
    sort: Enum,
}

/// Choice `if c then x else y` between two values `x` and `y` of an
/// enumeration, by the boolean `c`, of the shape `[1, 1]`, read first.
///
/// SMT-LIB2: `(ite c x y)`.
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(frozen, from_py_object, eq, str, module = "reactor.enum")
)]
pub struct Ite {
    sort: Enum,
}

/// The two enumerations of `dom`, which must be the same.
fn same_enum(dom: [Enum; 2]) -> Result<Enum, crate::Error> {
    let [x, y] = dom;
    if x != y {
        return Err(crate::unexpected_dom("2 equal enumerations", &[x, y]));
    }
    Ok(x)
}

impl Eq {
    /// The comparison reading the enumerations of `dom`; fails unless they are
    /// the same. Its codomain, a [`Bool`], does not determine it.
    pub fn with_dom(dom: [Enum; 2]) -> Result<Self, crate::Error> {
        Ok(Eq {
            sort: same_enum(dom)?,
        })
    }
}

impl Ne {
    /// The comparison reading the enumerations of `dom`; fails unless they are
    /// the same. Its codomain, a [`Bool`], does not determine it.
    pub fn with_dom(dom: [Enum; 2]) -> Result<Self, crate::Error> {
        Ok(Ne {
            sort: same_enum(dom)?,
        })
    }
}

impl Ite {
    /// The choice reading the boolean condition and the branches of `dom`;
    /// fails unless the condition is a boolean of the shape `[1, 1]` and the
    /// branches of the same enumeration.
    pub fn with_dom(dom: [Sort; 3]) -> Result<Self, crate::Error> {
        match dom {
            [Sort::Bool(c), Sort::Enum(x), Sort::Enum(y)] if c.shape() == [1, 1] && x == y => {
                Ok(Ite { sort: x })
            }
            _ => Err(crate::unexpected_dom(
                "a condition Bool([1, 1]) and two branches of the same enumeration",
                &dom,
            )),
        }
    }

    /// The choice writing the enumeration of `cod`.
    pub fn with_cod(cod: [Enum; 1]) -> Result<Self, crate::Error> {
        let [sort] = cod;
        Ok(Ite { sort })
    }
}

/// The operations on enumerations, with their boundary in [`Sort`].
#[derive(Clone, PartialEq, Debug, Eq, Hash, From)]
pub enum Op {
    Sym(Sym),
    Into(Into),
    Eq(Eq),
    Ne(Ne),
    Ite(Ite),
}

impl Boundary<Sort> for Sym {
    fn dom(&self) -> Poly<Sort> {
        Poly::unit()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![self.sort.clone().into()].into()
    }
}

impl Boundary<Sort> for Into {
    fn dom(&self) -> Poly<Sort> {
        mono![self.dom.clone().into()].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![self.cod.clone().into()].into()
    }
}

impl Boundary<Sort> for Eq {
    fn dom(&self) -> Poly<Sort> {
        mono![self.sort.clone().into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new_unchecked([1, 1]).into()].into()
    }
}

impl Boundary<Sort> for Ne {
    fn dom(&self) -> Poly<Sort> {
        mono![self.sort.clone().into(); 2].into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![Bool::new_unchecked([1, 1]).into()].into()
    }
}

impl Boundary<Sort> for Ite {
    fn dom(&self) -> Poly<Sort> {
        mono![
            Bool::new_unchecked([1, 1]).into(),
            self.sort.clone().into(),
            self.sort.clone().into()
        ]
        .into()
    }

    fn cod(&self) -> Poly<Sort> {
        mono![self.sort.clone().into()].into()
    }
}

impl Boundary<Sort> for Op {
    fn dom(&self) -> Poly<Sort> {
        match self {
            Op::Sym(op) => op.dom(),
            Op::Into(op) => op.dom(),
            Op::Eq(op) => op.dom(),
            Op::Ne(op) => op.dom(),
            Op::Ite(op) => op.dom(),
        }
    }

    fn cod(&self) -> Poly<Sort> {
        match self {
            Op::Sym(op) => op.cod(),
            Op::Into(op) => op.cod(),
            Op::Eq(op) => op.cod(),
            Op::Ne(op) => op.cod(),
            Op::Ite(op) => op.cod(),
        }
    }
}

/// Writes the symbolic constant itself.
impl fmt::Display for Sym {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl fmt::Display for Into {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Into")
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

impl fmt::Display for Ite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ite")
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::Sym(op) => op.fmt(f),
            Op::Into(op) => op.fmt(f),
            Op::Eq(op) => op.fmt(f),
            Op::Ne(op) => op.fmt(f),
            Op::Ite(op) => op.fmt(f),
        }
    }
}

// The Python constructors of the operations: each is built from the
// parameters its sorts do not determine, then its domain, when it reads
// anything, a single sort bare and several as a list, and optionally its
// codomain, which must agree; and matches by them the same way.

#[cfg(feature = "pyo3")]
#[pymethods]
impl Sym {
    /// Python builds it from its value, and its codomain `cod`, the
    /// enumeration holding it.
    #[new]
    fn py_new(value: &str, cod: Enum) -> Result<Self, crate::Error> {
        Self::with_cod(value, [cod])
    }

    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("value", "cod")
    }

    /// The symbolic constant.
    #[getter]
    fn value(&self) -> &str {
        &self.value
    }

    /// The enumeration it writes.
    #[getter(cod)]
    fn py_cod(&self) -> Enum {
        self.sort.clone()
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl Into {
    /// Python builds it from its domain `dom` and its codomain `cod`, which
    /// its domain does not determine: an enumeration holding all its values.
    #[new]
    fn py_new(dom: Enum, cod: Enum) -> Result<Self, crate::Error> {
        Self::new(dom, cod)
    }

    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[getter(dom)]
    fn py_dom<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[getter(cod)]
    fn py_cod<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_sorts(py, Boundary::cod(self))
    }
}

// The constructors of the comparisons, each reading two values of an
// enumeration.

#[cfg(feature = "pyo3")]
#[pymethods]
impl Eq {
    /// Python builds it from its domain `dom`, two values of the same
    /// enumeration; its codomain `cod`, when given, must be the one it
    /// writes.
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Enum; 2], cod: Option<Sort>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
    }

    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[getter(dom)]
    fn py_dom<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[getter(cod)]
    fn py_cod<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_sorts(py, Boundary::cod(self))
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl Ne {
    /// Python builds it from its domain `dom`, two values of the same
    /// enumeration; its codomain `cod`, when given, must be the one it
    /// writes.
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Enum; 2], cod: Option<Sort>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
    }

    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[getter(dom)]
    fn py_dom<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[getter(cod)]
    fn py_cod<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_sorts(py, Boundary::cod(self))
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl Ite {
    /// Python builds it from its domain `dom`, the condition, a boolean of
    /// the shape `[1, 1]`, and two branches of the same enumeration; its
    /// codomain `cod`, when given, must be the one it writes.
    #[new]
    #[pyo3(signature = (dom, cod = None))]
    fn py_new(dom: [Sort; 3], cod: Option<Sort>) -> Result<Self, crate::Error> {
        crate::check_cod(Self::with_dom(dom), cod)
    }

    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[getter(dom)]
    fn py_dom<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_sorts(py, Boundary::dom(self))
    }

    /// The sort it writes.
    #[getter(cod)]
    fn py_cod<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_sorts(py, Boundary::cod(self))
    }
}

/// Extracts an operation on enumerations from an instance of its class.
#[cfg(feature = "pyo3")]
impl<'a, 'py> FromPyObject<'a, 'py> for Op {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(op) = ob.extract::<Sym>() {
            return Ok(Op::Sym(op));
        }
        if let Ok(op) = ob.extract::<Into>() {
            return Ok(Op::Into(op));
        }
        if let Ok(op) = ob.extract::<Eq>() {
            return Ok(Op::Eq(op));
        }
        if let Ok(op) = ob.extract::<Ne>() {
            return Ok(Op::Ne(op));
        }
        if let Ok(op) = ob.extract::<Ite>() {
            return Ok(Op::Ite(op));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "expected an operation on enumerations, got {}",
            ob.get_type()
        )))
    }
}

/// Converts an operation on enumerations into an instance of its class.
#[cfg(feature = "pyo3")]
impl<'py> IntoPyObject<'py> for Op {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        match self {
            Op::Sym(op) => op.into_pyobject(py).map(Bound::into_any),
            Op::Into(op) => op.into_pyobject(py).map(Bound::into_any),
            Op::Eq(op) => op.into_pyobject(py).map(Bound::into_any),
            Op::Ne(op) => op.into_pyobject(py).map(Bound::into_any),
            Op::Ite(op) => op.into_pyobject(py).map(Bound::into_any),
        }
    }
}
