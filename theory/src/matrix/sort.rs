/*!
# Sorts

The sorts of all theories, and the sorts of each theory.

[`Any`] holds the sorts of all theories, just as [`crate::matrix::Any`] holds the
operations of all theories. `subenum` defines the sorts of each theory as a
subset of them, and generates the casts between `Any` and each subset:
`From` up to `Any`, and `TryFrom` down from it.

## Naming the subsets

The subsets are named after the SMT-LIB2 *theories*, and not after the
*logics*, such as [`crate::matrix::LRA`] or [`crate::matrix::BV`], which name subsets of the
operations. A theory fixes the sorts, and each logic restricts its
operations: the `L` of `LRA` stands for linear, a restriction on the
operations, so `LRA` and a non-linear `NRA` would share the same sorts.

Each subset is the plural of the sort its theory adds, as in SMT-LIB2's
`Reals` and `Ints`, so that the singular names an individual sort and the
plural the sorts of a theory:

| Subset | Sorts | SMT-LIB2 theory |
|---|---|---|
| [`Prop`] | [`Bool`], [`Zero`] | `Core` |
| [`Reals`] | [`Real`], [`Bool`], [`Zero`] | `Reals` |
| [`Ints`] | [`Int`], [`Bool`], [`Zero`] | `Ints` |
| [`Nums`] | [`Real`], [`Int`], [`Bool`], [`Zero`] | `Reals_Ints` |
| [`Words`] | [`Word`], [`Bool`], [`Zero`] | `FixedSizeBitVectors` |

[`Nums`], the sorts of both kinds of numbers, shortens `Reals_Ints`, and
[`Words`] follows our sort [`Word`] rather than SMT-LIB2's longer name.
[`Prop`] is the exception to the plurals: the theory of booleans is
SMT-LIB2's `Core`, a name too generic for its sorts, so it takes the name
of propositional logic, whose operations are [`crate::matrix::Prop`].

Every subset includes [`Bool`], because every SMT-LIB2 theory includes the
Core theory, whose booleans are the sort that comparisons write; [`Prop`]
holds exactly the sorts that the others share.

Every subset also includes [`Zero`], which is not an SMT-LIB2 sort, but
the trivial tangent of the discrete sorts [`Bool`], [`Int`] and [`Word`],
and its own tangent. With it, each subset holds the tangents of its sorts,
and so is closed under taking tangents, as [`Any`] is.

The sorts are defined here, and the operations on them in the modules of
their theories: [`super::real`], [`super::int`], [`super::bool`] and
[`super::word`].
*/

use crate::sort::Zero;
use base::differentiable::Differentiable;
use derive_more::From;
use std::fmt;
use subenum::subenum;

#[cfg(feature = "pyo3")]
use pyo3::{exceptions::PyTypeError, prelude::*};

/// The sort of real matrices with the given `[rows, cols]` shape, graded by
/// their `degree`: values of degree `d` are `d`-th order rates of change,
/// and plain reals have degree 0.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(
        frozen,
        from_py_object,
        eq,
        str,
        get_all,
        module = "reactor.matrix.sort"
    )
)]
pub struct Real {
    degree: u8,
    shape: [usize; 2],
}

impl Real {
    /// Real matrices of the given shape, of the given `degree`: the
    /// `degree`-th order rates of change of plain ones. Outside this crate,
    /// they are written with the tangent, `T(Real::new(shape))`.
    pub(crate) fn with_degree(shape: [usize; 2], degree: u8) -> Self {
        Real { degree, shape }
    }

    pub fn degree(&self) -> u8 {
        self.degree
    }

    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

/// Python matches a `Real` by `shape`.
#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Real {
    /// Plain real matrices of the given shape, of degree 0; those of a
    /// higher degree are their tangents, `T(Real(shape))`.
    #[cfg_attr(feature = "pyo3", new)]
    pub fn new(shape: [usize; 2]) -> Self {
        Real { degree: 0, shape }
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    /// The tangent sort `T(self)`: the sort of the rate of change of its
    /// values.
    #[cfg(feature = "pyo3")]
    #[pyo3(name = "tangent")]
    fn tangent(&self) -> Real {
        Differentiable::tangent(self)
    }
}

impl fmt::Display for Real {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [i, j] = self.shape;
        match self.degree {
            0 => write!(f, "Real([{i}, {j}])"),
            1 => write!(f, "T(Real([{i}, {j}]))"),
            d => write!(f, "T^{d}(Real([{i}, {j}]))"),
        }
    }
}

/// The rate of change of a real matrix is a real matrix of the same shape,
/// one degree higher.
impl Differentiable for Real {
    type Tangent = Real;

    fn tangent(self) -> Real {
        Real {
            degree: self
                .degree
                .checked_add(1)
                .expect("real sort degree out of range"),
            shape: self.shape,
        }
    }
}

/// The sort of integer matrices with the given `[rows, cols]` shape.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(
        frozen,
        from_py_object,
        eq,
        str,
        get_all,
        module = "reactor.matrix.sort"
    )
)]
pub struct Int {
    shape: [usize; 2],
}

impl Int {
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

/// Python matches an `Int` by `shape`.
#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Int {
    #[cfg_attr(feature = "pyo3", new)]
    pub fn new(shape: [usize; 2]) -> Self {
        Int { shape }
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    /// The tangent sort `T(self)`: the sort of the rate of change of its
    /// values.
    #[cfg(feature = "pyo3")]
    #[pyo3(name = "tangent")]
    fn tangent(&self) -> Zero {
        Differentiable::tangent(self)
    }
}

impl fmt::Display for Int {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [i, j] = self.shape;
        write!(f, "Int([{i}, {j}])")
    }
}

/// Integers are discrete, so their tangent is trivial: the singleton sort
/// [`Zero`].
impl Differentiable for Int {
    type Tangent = Zero;

    fn tangent(self) -> Zero {
        Zero
    }
}

/// The sort of boolean matrices with the given `[rows, cols]` shape.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(
        frozen,
        from_py_object,
        eq,
        str,
        get_all,
        module = "reactor.matrix.sort"
    )
)]
pub struct Bool {
    shape: [usize; 2],
}

impl Bool {
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

/// Python matches a `Bool` by `shape`.
#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Bool {
    #[cfg_attr(feature = "pyo3", new)]
    pub fn new(shape: [usize; 2]) -> Self {
        Bool { shape }
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    /// The tangent sort `T(self)`: the sort of the rate of change of its
    /// values.
    #[cfg(feature = "pyo3")]
    #[pyo3(name = "tangent")]
    fn tangent(&self) -> Zero {
        Differentiable::tangent(self)
    }
}

impl fmt::Display for Bool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [i, j] = self.shape;
        write!(f, "Bool([{i}, {j}])")
    }
}

/// Booleans are discrete, so their tangent is trivial: the singleton sort
/// [`Zero`].
impl Differentiable for Bool {
    type Tangent = Zero;

    fn tangent(self) -> Zero {
        Zero
    }
}

/// The sort of matrices of `width`-bit words with the given `[rows, cols]`
/// shape: entrywise, the SMT-LIB2 sort `(_ BitVec width)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(
        frozen,
        from_py_object,
        eq,
        str,
        get_all,
        module = "reactor.matrix.sort"
    )
)]
pub struct Word {
    width: usize,
    shape: [usize; 2],
}

/// Fails unless `width` is positive: as in SMT-LIB2, words have at least one
/// bit.
pub(crate) fn check_width(width: usize) -> Result<(), crate::Error> {
    if width == 0 {
        return Err(crate::Error("words have at least one bit".to_string()));
    }
    Ok(())
}

impl Word {
    /// Matrices of `width`-bit words, as [`Word::new`], but panics if
    /// `width` is zero.
    pub fn new_unchecked(width: usize, shape: [usize; 2]) -> Self {
        Self::new(width, shape).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

/// Python matches a `Word` by `width` and `shape`.
#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl Word {
    /// Matrices of `width`-bit words; as in SMT-LIB2, `width` must be
    /// positive, or this returns an error.
    #[cfg_attr(feature = "pyo3", new)]
    pub fn new(width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
        check_width(width)?;
        Ok(Word { width, shape })
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("width", "shape")
    }

    /// The tangent sort `T(self)`: the sort of the rate of change of its
    /// values.
    #[cfg(feature = "pyo3")]
    #[pyo3(name = "tangent")]
    fn tangent(&self) -> Zero {
        Differentiable::tangent(self)
    }
}

impl fmt::Display for Word {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [i, j] = self.shape;
        write!(f, "Word{}([{i}, {j}])", self.width)
    }
}

/// Words are discrete, so their tangent is trivial: the singleton sort
/// [`Zero`].
impl Differentiable for Word {
    type Tangent = Zero;

    fn tangent(self) -> Zero {
        Zero
    }
}

/// The sorts of all theories: a value of `Any` is a sort of one of them.
#[subenum(Prop, Reals, Ints, Nums, Words)]
#[derive(Clone, Copy, PartialEq, Debug, Eq, From)]
pub enum Any {
    #[subenum(Reals, Nums)]
    Real(Real),
    #[subenum(Ints, Nums)]
    Int(Int),
    #[subenum(Prop, Reals, Ints, Nums, Words)]
    Bool(Bool),
    #[subenum(Words)]
    Word(Word),
    #[subenum(Prop, Reals, Ints, Nums, Words)]
    Zero(Zero),
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

display_subset!(Prop, Reals, Ints, Nums);

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
/// sort [`Zero`].
impl Differentiable for Words {
    type Tangent = Zero;

    fn tangent(self) -> Zero {
        Zero
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
        if let Ok(s) = ob.extract::<Zero>() {
            return Ok(Any::Zero(s));
        }
        Err(PyTypeError::new_err(format!(
            "expected a sort: Real, Int, Bool, Word or Zero, got {}",
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
