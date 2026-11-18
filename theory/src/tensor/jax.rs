//! The JAX arrays, with the `jax` feature.

use super::{Builtin, Error, Kind};
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;
use std::fmt;
use std::sync::Arc;

/// A JAX array holding a matrix, held as its Python `jax.Array`, which is
/// immutable: its shape and kind are read once, when it is held.
#[derive(Clone, Debug)]
pub struct Jax {
    // An `Arc`, as a `Py` is `Clone` only with the GIL held.
    array: Arc<Py<PyAny>>,
    shape: [usize; 2],
    kind: Kind,
}

impl Jax {
    /// Whether `ob` is a `jax.Array`; never unless `jax` is imported, as no
    /// array of it can exist then, nor if `jax` has no `Array`, as when its
    /// import is blocked.
    pub(super) fn is_array(ob: &Bound<'_, PyAny>) -> PyResult<bool> {
        let modules = ob.py().import("sys")?.getattr("modules")?;
        let Ok(array) = modules.get_item("jax").and_then(|jax| jax.getattr("Array")) else {
            return Ok(false);
        };
        ob.is_instance(&array)
    }

    /// The `jax.Array` `ob`, which fails unless it is a matrix.
    pub(super) fn new(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let shape: Vec<usize> = ob.getattr("shape")?.extract()?;
        let [rows, cols] = shape[..] else {
            return Err(Error::NotAMatrix(shape).into());
        };
        Ok(Jax {
            array: Arc::new(ob.clone().unbind()),
            shape: [rows, cols],
            kind: kind(ob)?,
        })
    }

    /// The `[rows, cols]` shape of the matrix.
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    /// The kind of the elements.
    pub fn kind(&self) -> Kind {
        self.kind.clone()
    }

    /// The `jax.Array`.
    pub fn array<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.array.bind(py).clone()
    }

    /// A copy of the matrix in Rust.
    pub fn to_builtin(&self, py: Python<'_>) -> PyResult<Builtin> {
        self.array(py).call_method0("tolist")?.extract()
    }
}

/// The kind of the elements of the `jax.Array` `ob`, from its `dtype`.
fn kind(ob: &Bound<'_, PyAny>) -> PyResult<Kind> {
    let jnp = ob.py().import("jax.numpy")?;
    let dtype = ob.getattr("dtype")?;
    let is = |kind: &str| -> PyResult<bool> {
        jnp.call_method1("issubdtype", (&dtype, jnp.getattr(kind)?))?
            .extract()
    };
    Ok(if is("bool_")? {
        Kind::Bool
    } else if is("integer")? {
        Kind::Int
    } else if is("floating")? {
        Kind::Float
    } else {
        Kind::Other(dtype.str()?.to_string())
    })
}

impl<'py> IntoPyObject<'py> for Jax {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    /// The `jax.Array` itself.
    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        Ok(self.array(py))
    }
}

impl fmt::Display for Jax {
    /// Displays as a [`Builtin`] copy of the array, which needs the GIL.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match Python::with_gil(|py| self.to_builtin(py)) {
            Ok(matrix) => matrix.fmt(f),
            Err(_) => write!(f, "<jax.Array of {}>", self.kind),
        }
    }
}
