/*!
# JAX

The JAX backing of the operations of [`crate::tensor`]: [`Array`], a
`jax.Array` held as its Python object, implements [`crate::Tensor`], so that
the operations carrying tensors may hold JAX arrays, as
`tensor::real::Const<jax::Array>`.

A `jax.Array` is immutable, so an [`Array`] reads its shape and the kind of
its elements once, when it is held: they cannot change afterwards, and
telling them needs no Python. It holds the array behind an `Arc`, as
cloning a Python object needs the GIL, and cloning an `Arc` does not.

An object is a `jax.Array` only if `jax` is imported, as no array of it can
exist otherwise: this module imports nothing until it is given one.

As for [`crate::torch`], the constants are Python classes of their own, one
per theory, as an array alone does not tell the theory of its constant:
[`real::Const`], [`int::Const`], [`bool::Const`] and [`word::Const`]. Each
holds, in a private field, an [`Array`] of the element kind of its theory,
checked on construction, and [`word::Const`] its width too. So are the
pointwise products with a constant, [`real::Scale`] and [`int::Scale`]. The
affine maps [`real::Linear`] and [`int::Linear`] are classes too, holding
the [`Array`]s `A` and `B` of shapes `[m, n]` and `[m, k]`, checked on
construction. The real ones take a degree, as their torch counterparts do. The operations of [`crate::tensor`] extract from the class
of their theory, sharing its arrays. An [`Array`] is `Sync`, so unlike the
torch classes, these need no `unsafe` to be Python classes.
*/

pub mod bool;
pub mod int;
pub mod real;
pub mod word;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::fmt;
use std::sync::Arc;

/// A `jax.Array` holding a matrix, with its shape and the kind of its
/// elements, read when it is held.
#[derive(Clone, Debug)]
pub struct Array {
    array: Arc<Py<PyAny>>,
    shape: [usize; 2],
    dtype: String,
    kind: Kind,
}

/// The kinds of elements that the theories tell apart.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Bool,
    Int,
    Float,
    Other,
}

impl Array {
    /// Whether `ob` is a `jax.Array`: never unless `jax` is imported, nor
    /// if `jax` has no `Array`, as when its import is blocked.
    pub fn is_array(ob: &Bound<'_, PyAny>) -> PyResult<bool> {
        let modules = ob.py().import("sys")?.getattr("modules")?;
        let Ok(array) = modules.get_item("jax").and_then(|jax| jax.getattr("Array")) else {
            return Ok(false);
        };
        ob.is_instance(&array)
    }

    /// Holds the `jax.Array` `ob`. Fails with a `TypeError` unless it is
    /// one, and with a `ValueError` unless it is a matrix.
    pub fn new(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        if !Self::is_array(ob)? {
            return Err(PyTypeError::new_err(format!(
                "expected a jax.Array, got {}",
                ob.get_type()
            )));
        }
        let shape: Vec<usize> = ob.getattr("shape")?.extract()?;
        let [rows, cols] = shape[..] else {
            let message = format!("expected a matrix, got an array of shape {shape:?}");
            return Err(crate::Error(message).into());
        };
        let dtype = ob.getattr("dtype")?;
        Ok(Array {
            array: Arc::new(ob.clone().unbind()),
            shape: [rows, cols],
            dtype: dtype.str()?.to_string(),
            kind: kind(&dtype)?,
        })
    }

    /// Whether the elements are floating-point numbers.
    pub fn is_floating_point(&self) -> bool {
        self.kind == Kind::Float
    }

    /// Whether the elements are integers, signed or not.
    pub fn is_integer(&self) -> bool {
        self.kind == Kind::Int
    }

    /// Whether the elements are booleans.
    pub fn is_bool(&self) -> bool {
        self.kind == Kind::Bool
    }

    /// The name of the type of the elements, as `float32`.
    pub fn dtype(&self) -> &str {
        &self.dtype
    }

    /// The `jax.Array` itself.
    pub fn bind<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.array.bind(py).clone()
    }
}

/// The kind of the elements of the `dtype` of a `jax.Array`.
fn kind(dtype: &Bound<'_, PyAny>) -> PyResult<Kind> {
    let jnp = dtype.py().import("jax.numpy")?;
    let is = |kind: &str| -> PyResult<bool> {
        jnp.call_method1("issubdtype", (dtype, jnp.getattr(kind)?))?
            .extract()
    };
    Ok(if is("bool_")? {
        Kind::Bool
    } else if is("integer")? {
        Kind::Int
    } else if is("floating")? {
        Kind::Float
    } else {
        Kind::Other
    })
}

impl crate::Tensor for Array {
    /// The `[rows, cols]` shape of the matrix, as read when it was held.
    fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

/// Prints the array as Python does, which needs the GIL.
impl fmt::Display for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let printed =
            Python::attach(|py| -> PyResult<String> { Ok(self.array.bind(py).str()?.to_string()) });
        match printed {
            Ok(array) => f.write_str(&array),
            Err(_) => write!(f, "<jax.Array of {}>", self.dtype),
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Array {
    type Error = PyErr;

    /// Fails as [`Array::new`] does.
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Array::new(&ob)
    }
}

impl<'py> IntoPyObject<'py> for Array {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    /// The `jax.Array` itself.
    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        Ok(self.bind(py))
    }
}

/// Fails unless `is` holds of the elements of each of `arrays`, which are
/// the `elements` of a theory, as `"floating-point"`.
fn check_elements<const N: usize>(
    arrays: [&Array; N],
    is: fn(&Array) -> bool,
    elements: &str,
) -> Result<(), crate::Error> {
    if arrays.iter().all(|a| is(a)) {
        return Ok(());
    }
    let dtypes: Vec<&str> = arrays.iter().map(|a| a.dtype()).collect();
    let arrays = if N == 1 { "an array" } else { "arrays" };
    Err(crate::Error(format!(
        "expected {arrays} of {elements} elements, got {}",
        dtypes.join(" and ")
    )))
}

/// Fails unless `a` and `b` are the matrices of an affine map `x ↦ A·x + B`:
/// `A` of shape `[m, n]` and `B` of shape `[m, k]`.
fn check_affine(a: &Array, b: &Array) -> Result<(), crate::Error> {
    crate::tensor::check_affine(a.shape, b.shape)
}

/// Converts an operation of any theory into its Python class, through the
/// conversion of its theory; the structural operations are classes of their
/// own.
impl<'py> IntoPyObject<'py> for crate::tensor::Any<Array> {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        use crate::tensor::Any;

        match self {
            Any::Real(op) => op.into_pyobject(py),
            Any::Int(op) => op.into_pyobject(py),
            Any::Bool(op) => op.into_pyobject(py),
            Any::Word(op) => op.into_pyobject(py),
            Any::Havoc(op) => op.into_pyobject(py).map(Bound::into_any),
            Any::Skip(op) => op.into_pyobject(py).map(Bound::into_any),
            Any::Cast(op) => op.into_pyobject(py).map(Bound::into_any),
            Any::Zero(op) => op.into_pyobject(py).map(Bound::into_any),
        }
    }
}
