/*!
# Casts between the theories' sorts

Every operation of a theory reads the sorts of that theory. A cast reads the
sort of one theory and writes that of another, pointwise and keeping the
shape, and so belongs to no theory, and to no logic of one:

| Operation | Reads | Writes | SMT-LIB2 |
|---|---|---|---|
| [`ToReal`] | `Int([m, n])` | `Real([m, n])`, the same number | `to_real` |
| [`ToInt`] | `Real([m, n])` | `Int([m, n])`, the floor | `to_int` |
| [`UbvToInt`] | `Word(width, [m, n])` | `Int([m, n])`, the unsigned value | `ubv_to_int` |

A formula mixing the theories (an integer invariant of a module of words,
say) joins them with these.
*/

use super::sort::{Any, Int, Real, Word, check_width};
use base::boundary::{Boundary, Poly};
use base::mono;
use derive_more::From;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use std::fmt;

/// Pointwise the integers of an `[m, n]` matrix as reals.
///
/// SMT-LIB2: `(to_real x)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, from_py_object, eq, str))]
pub struct ToReal {
    shape: [usize; 2],
}

/// Pointwise the floor of the reals of an `[m, n]` matrix, as integers.
///
/// SMT-LIB2: `(to_int x)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, from_py_object, eq, str))]
pub struct ToInt {
    shape: [usize; 2],
}

/// Pointwise the unsigned value of the `width`-bit words of an `[m, n]`
/// matrix, as integers, as `UbvToInt(width, [m, n])`.
///
/// SMT-LIB2: `(ubv_to_int x)`.
#[derive(Clone, Copy, PartialEq, Debug, Eq)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(frozen, from_py_object, eq, str))]
pub struct UbvToInt {
    width: usize,
    shape: [usize; 2],
}

/// The casts, with their boundary in [`Any`].
#[derive(Clone, Copy, PartialEq, Debug, Eq, From)]
pub enum Cast {
    ToReal(ToReal),
    ToInt(ToInt),
    UbvToInt(UbvToInt),
}

impl fmt::Display for Cast {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Cast::ToReal(op) => op.fmt(f),
            Cast::ToInt(op) => op.fmt(f),
            Cast::UbvToInt(op) => op.fmt(f),
        }
    }
}

impl Boundary<Any> for Cast {
    fn dom(&self) -> Poly<Any> {
        match self {
            Cast::ToReal(op) => op.dom(),
            Cast::ToInt(op) => op.dom(),
            Cast::UbvToInt(op) => op.dom(),
        }
    }

    fn cod(&self) -> Poly<Any> {
        match self {
            Cast::ToReal(op) => op.cod(),
            Cast::ToInt(op) => op.cod(),
            Cast::UbvToInt(op) => op.cod(),
        }
    }
}

impl Boundary<Any> for ToReal {
    fn dom(&self) -> Poly<Any> {
        mono![Int::new(self.shape).into()].into()
    }

    fn cod(&self) -> Poly<Any> {
        mono![Real::new(self.shape).into()].into()
    }
}

impl Boundary<Any> for ToInt {
    fn dom(&self) -> Poly<Any> {
        mono![Real::new(self.shape).into()].into()
    }

    fn cod(&self) -> Poly<Any> {
        mono![Int::new(self.shape).into()].into()
    }
}

impl Boundary<Any> for UbvToInt {
    fn dom(&self) -> Poly<Any> {
        mono![Word::new_unchecked(self.width, self.shape).into()].into()
    }

    fn cod(&self) -> Poly<Any> {
        mono![Int::new(self.shape).into()].into()
    }
}

impl fmt::Display for ToReal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ToReal")
    }
}

impl fmt::Display for ToInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ToInt")
    }
}

impl fmt::Display for UbvToInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "UbvToInt")
    }
}

/// Extracts a cast from a Python object, trying each cast's class in turn.
#[cfg(feature = "pyo3")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for Cast {
    type Error = pyo3::PyErr;

    fn extract(ob: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use pyo3::types::PyAnyMethods;

        if let Ok(op) = ob.extract::<ToReal>() {
            return Ok(Cast::ToReal(op));
        }
        if let Ok(op) = ob.extract::<ToInt>() {
            return Ok(Cast::ToInt(op));
        }
        if let Ok(op) = ob.extract::<UbvToInt>() {
            return Ok(Cast::UbvToInt(op));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "expected a cast, got {}",
            ob.get_type()
        )))
    }
}

/// Converts a cast into the Python class of its kind.
#[cfg(feature = "pyo3")]
impl<'py> pyo3::IntoPyObject<'py> for Cast {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, pyo3::PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> pyo3::PyResult<Self::Output> {
        match self {
            Cast::ToReal(op) => op.into_pyobject(py).map(pyo3::Bound::into_any),
            Cast::ToInt(op) => op.into_pyobject(py).map(pyo3::Bound::into_any),
            Cast::UbvToInt(op) => op.into_pyobject(py).map(pyo3::Bound::into_any),
        }
    }
}

// The Python constructors: each takes its sizes positionally, in the order of
// its fields, the shape as a list.
#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl ToReal {
    #[cfg_attr(feature = "pyo3", new)]
    #[cfg_attr(feature = "pyo3", pyo3(signature = (shape, /)))]
    pub fn new(shape: [usize; 2]) -> Self {
        ToReal { shape }
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl ToInt {
    #[cfg_attr(feature = "pyo3", new)]
    #[cfg_attr(feature = "pyo3", pyo3(signature = (shape, /)))]
    pub fn new(shape: [usize; 2]) -> Self {
        ToInt { shape }
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str,) {
        ("shape",)
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

#[cfg_attr(feature = "pyo3", cfg_eval::cfg_eval, pymethods)]
impl UbvToInt {
    /// Fails unless `width` is positive.
    #[cfg_attr(feature = "pyo3", new)]
    #[cfg_attr(feature = "pyo3", pyo3(signature = (width, shape, /)))]
    pub fn new(width: usize, shape: [usize; 2]) -> Result<Self, crate::Error> {
        check_width(width)?;
        Ok(UbvToInt { width, shape })
    }

    #[cfg(feature = "pyo3")]
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("width", "shape")
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn width(&self) -> usize {
        self.width
    }

    #[cfg_attr(feature = "pyo3", getter)]
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}
