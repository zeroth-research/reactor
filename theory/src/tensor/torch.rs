//! The torch tensors, with the `torch` feature.

// The conversions from and into Python are mostly taken from the `tch-pyo3`
// crate: https://github.com/LaurentMazare/tch-rs/blob/main/pyo3-tch/src/lib.rs
//
// The used code comes with the following LICENSE (see
// https://github.com/LaurentMazare/tch-rs?tab=readme-ov-file#license):

// Permission is hereby granted, free of charge, to any
// person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the
// Software without restriction, including without
// limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software
// is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice
// shall be included in all copies or substantial portions
// of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
// ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
// TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
// PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
// SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

use super::{Error, Kind, Value, fmt_matrix};
use pyo3::prelude::*;
use pyo3_tch::{tch, wrap_tch_err};
use std::fmt;

/// A torch tensor holding a matrix. A Python `torch.Tensor` converts into
/// one and back sharing its storage, so that a change to the tensor in
/// Python, as training makes, shows in the operations holding it.
#[derive(Debug)]
pub struct Torch(tch::Tensor);

// SAFETY: libtorch makes a tensor safe to share between threads, counting
// its references atomically, and the conversions from and into Python hold
// the GIL.
unsafe impl Sync for Torch {}

impl Clone for Torch {
    /// Shares the storage of the tensor.
    fn clone(&self) -> Self {
        Torch(self.0.shallow_clone())
    }
}

impl Torch {
    /// The matrix `tensor`, which fails unless it has two dimensions.
    pub fn new(tensor: tch::Tensor) -> Result<Self, Error> {
        match tensor.size()[..] {
            [_, _] => Ok(Torch(tensor)),
            ref size => Err(Error::NotAMatrix(
                size.iter().map(|&d| d as usize).collect(),
            )),
        }
    }

    /// The tensor.
    pub fn tensor(&self) -> &tch::Tensor {
        &self.0
    }

    /// The `[rows, cols]` shape of the matrix.
    pub fn shape(&self) -> [usize; 2] {
        let [rows, cols] = self.0.size()[..] else {
            unreachable!("a `Torch` holds a matrix")
        };
        [rows as usize, cols as usize]
    }

    /// The kind of the elements.
    pub fn kind(&self) -> Kind {
        match self.0.kind() {
            tch::Kind::Bool => Kind::Bool,
            tch::Kind::Uint8
            | tch::Kind::Int8
            | tch::Kind::Int16
            | tch::Kind::Int
            | tch::Kind::Int64 => Kind::Int,
            _ if self.0.is_floating_point() => Kind::Float,
            kind => Kind::Other(format!("{kind:?}")),
        }
    }

    /// The element in row `i` and column `j`, unless the elements are of
    /// another kind than [`Kind::Bool`], [`Kind::Int`] and [`Kind::Float`].
    fn value(&self, kind: &Kind, i: usize, j: usize) -> Value {
        let index = [i as i64, j as i64];
        match kind {
            Kind::Bool => Value::Bool(self.0.int64_value(&index) != 0),
            Kind::Int => Value::Int(self.0.int64_value(&index)),
            _ => Value::Float(self.0.double_value(&index)),
        }
    }

    /// The tensor of the Python object `ob`: none unless `ob` is a
    /// `torch.Tensor`, and an error unless it is a matrix.
    pub(super) fn from_python(ob: &Bound<'_, PyAny>) -> PyResult<Option<Self>> {
        let ptr = ob.as_ptr() as *mut tch::python::CPyObject;
        // SAFETY: `ob` is a live Python object, and the GIL is held.
        let tensor = unsafe { tch::Tensor::pyobject_unpack(ptr) }.map_err(wrap_tch_err)?;
        tensor
            .map(|t| Torch::new(t).map_err(PyErr::from))
            .transpose()
    }
}

impl<'py> IntoPyObject<'py> for Torch {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    /// The `torch.Tensor` sharing the storage of the tensor.
    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        let ptr = self.0.pyobject_wrap().map_err(wrap_tch_err)?;
        // SAFETY: `pyobject_wrap` returns a new reference to a Python object.
        Ok(unsafe { Bound::from_owned_ptr(py, ptr as *mut pyo3::ffi::PyObject) })
    }
}

impl fmt::Display for Torch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind() {
            Kind::Other(name) => write!(f, "<tensor of {name}>"),
            kind => fmt_matrix(f, self.shape(), |i, j| self.value(&kind, i, j)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn torch(tensor: tch::Tensor) -> Torch {
        Torch::new(tensor).unwrap()
    }

    #[test]
    fn small_tensors_print_entirely_on_one_line() {
        let t = torch(tch::Tensor::from_slice(&[1.5f64, 2.0, 3.0]).reshape([1, 3]));
        assert_eq!(t.to_string(), "[[1.5, 2, 3]]");
        let b = torch(tch::Tensor::from_slice(&[true, false]).reshape([2, 1]));
        assert_eq!(b.to_string(), "[[true], [false]]");
    }

    #[test]
    fn large_tensors_abbreviate_with_dots() {
        let t =
            torch(tch::Tensor::arange(24, (tch::Kind::Int64, tch::Device::Cpu)).reshape([2, 12]));
        assert_eq!(
            t.to_string(),
            "[[0, 1, 2, ..., 9, 10, 11], [12, 13, 14, ..., 21, 22, 23]]"
        );
    }

    #[test]
    fn tensors_are_matrices_of_a_kind() {
        let t = tch::Tensor::zeros([2, 3], (tch::Kind::Float, tch::Device::Cpu));
        assert_eq!(torch(t).kind(), Kind::Float);
        let t = tch::Tensor::zeros([2, 3], (tch::Kind::Int, tch::Device::Cpu));
        assert_eq!(torch(t).kind(), Kind::Int);
        let t = tch::Tensor::zeros([4], (tch::Kind::Float, tch::Device::Cpu));
        assert_eq!(Torch::new(t).unwrap_err(), Error::NotAMatrix(vec![4]));
    }
}
