/*!
# Mixed

The tensors of the other backings together: [`Tensor`] is a torch tensor
or a JAX array, so that the operations of one module, or even one affine
map, may hold either, as `tensor::Any<mixed::Tensor>`. It needs both the
`torch` and the `jax` features. A [`Tensor`] extracts from a
`torch.Tensor`, held as a [`tch::Tensor`] sharing its storage, or from a
`jax.Array`, held as a [`crate::jax::Array`], checked to be a matrix
either way, so that its shape, and so the boundary of its operation, is
known.

As for [`crate::torch`] and [`crate::jax`], the constants are Python
classes of their own, one per theory: [`real::Const`], [`int::Const`],
[`bool::Const`] and [`word::Const`], each holding, in a private field, a
[`Tensor`] of the element kind of its theory, checked on construction, and
[`word::Const`] its width too. So are the pointwise products with a
constant, [`real::Scale`] and [`int::Scale`]. The affine maps
[`real::Linear`] and [`int::Linear`] are classes too, holding the
[`Tensor`]s `A` and `B` of shapes `[m, n]` and `[m, k]`, checked on
construction. The real ones take a degree, as their torch counterparts do. The operations of [`crate::tensor`] extract from the class of
their theory, sharing its tensors. A [`tch::Tensor`] is not `Sync`, which a
Python class must be, so each class implements `Sync` itself, as only its
own module touches the tensor.
*/

pub mod bool;
pub mod int;
pub mod real;
pub mod word;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::fmt;

/// A matrix that Python passes to an operation: a `torch.Tensor` or a
/// `jax.Array`.
#[derive(Debug)]
pub enum Tensor {
    /// A `torch.Tensor`, sharing its storage.
    Torch(tch::Tensor),
    /// A `jax.Array`, holding the array itself.
    Jax(crate::jax::Array),
}

impl Tensor {
    /// Whether the elements are floating-point numbers.
    pub fn is_floating_point(&self) -> bool {
        match self {
            Tensor::Torch(t) => t.is_floating_point(),
            Tensor::Jax(a) => a.is_floating_point(),
        }
    }

    /// Whether the elements are integers, signed or not.
    pub fn is_integer(&self) -> bool {
        match self {
            Tensor::Torch(t) => crate::torch::is_integer(t.kind()),
            Tensor::Jax(a) => a.is_integer(),
        }
    }

    /// Whether the elements are booleans.
    pub fn is_bool(&self) -> bool {
        match self {
            Tensor::Torch(t) => t.kind() == tch::Kind::Bool,
            Tensor::Jax(a) => a.is_bool(),
        }
    }

    /// The name of the type of the elements, in the terms of the backing:
    /// as `Float` for torch and `float32` for JAX.
    pub fn dtype(&self) -> String {
        match self {
            Tensor::Torch(t) => format!("{:?}", t.kind()),
            Tensor::Jax(a) => a.dtype().to_string(),
        }
    }
}

impl crate::Tensor for Tensor {
    fn shape(&self) -> [usize; 2] {
        match self {
            Tensor::Torch(t) => crate::Tensor::shape(t),
            Tensor::Jax(a) => crate::Tensor::shape(a),
        }
    }
}

/// A clone shares the tensor of the original: a shallow clone of a torch
/// tensor, as libtorch counts its references, and the same JAX array.
impl Clone for Tensor {
    fn clone(&self) -> Self {
        match self {
            Tensor::Torch(t) => Tensor::Torch(t.shallow_clone()),
            Tensor::Jax(a) => Tensor::Jax(a.clone()),
        }
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Tensor::Torch(t) => fmt::Display::fmt(t, f),
            Tensor::Jax(a) => fmt::Display::fmt(a, f),
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Tensor {
    type Error = PyErr;

    /// Fails with a `TypeError` unless `ob` is a `torch.Tensor` or a
    /// `jax.Array`, and with a `ValueError` unless it is a matrix.
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if crate::jax::Array::is_array(&ob)? {
            return Ok(Tensor::Jax(crate::jax::Array::new(&ob)?));
        }
        if let Ok(tensor) = ob.extract::<pyo3_tch::PyTensor>() {
            crate::torch::check_matrix(&tensor)?;
            return Ok(Tensor::Torch(tensor.0));
        }
        Err(PyTypeError::new_err(format!(
            "expected a torch.Tensor or a jax.Array, got {}",
            ob.get_type()
        )))
    }
}

impl<'py> IntoPyObject<'py> for Tensor {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    /// The `torch.Tensor`, sharing the storage, or the `jax.Array` itself.
    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        match self {
            Tensor::Torch(t) => pyo3_tch::PyTensor(t).into_pyobject(py),
            Tensor::Jax(a) => a.into_pyobject(py),
        }
    }
}

/// Fails unless `is` holds of the elements of each of `tensors`, which are
/// the `elements` of a theory, as `"floating-point"`.
fn check_elements<const N: usize>(
    tensors: [&Tensor; N],
    is: fn(&Tensor) -> bool,
    elements: &str,
) -> Result<(), crate::Error> {
    if tensors.iter().all(|t| is(t)) {
        return Ok(());
    }
    let dtypes: Vec<String> = tensors.iter().map(|t| t.dtype()).collect();
    let tensors = if N == 1 { "a tensor" } else { "tensors" };
    Err(crate::Error(format!(
        "expected {tensors} of {elements} elements, got {}",
        dtypes.join(" and ")
    )))
}

/// Fails unless `a` and `b` are the matrices of an affine map `x ↦ A·x + B`:
/// `A` of shape `[m, n]` and `B` of shape `[m, k]`.
fn check_affine(a: &Tensor, b: &Tensor) -> Result<(), crate::Error> {
    use crate::Tensor as _;

    crate::tensor::check_affine(a.shape(), b.shape())
}

/// Converts an operation of any theory into its Python class, through the
/// conversion of its theory; the structural operations are classes of their
/// own.
impl<'py> IntoPyObject<'py> for crate::tensor::Any<Tensor> {
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
