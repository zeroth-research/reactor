/*!
# Constant matrices

The constant matrices carried by the operations: the literals ([`Matrix`]) and the affine
maps ([`Affine`]). They are checked on their creation (e.g., a [`Matrix`] is a 2D tensor),
so that the operations that carry them always have a boundary. In Python, the checks
raise a `ValueError` when the operation is created.
*/

use crate::PyTensor;
#[cfg(feature = "pyo3")]
use pyo3::{exceptions::PyValueError, prelude::*};
use std::fmt;

/// The `[rows, cols]` shape of the 2D tensor `t`, or an error if `t` is not 2D
pub(crate) fn shape_2d(t: &PyTensor) -> Result<[usize; 2], String> {
    match t.size().as_slice() {
        &[rows, cols] => Ok([rows as usize, cols as usize]),
        size => Err(format!("expected a 2D tensor, got {}D", size.len())),
    }
}

/// A constant matrix: a 2D tensor. In Python, it is the tensor itself.
#[derive(Clone)]
pub struct Matrix {
    tensor: PyTensor,
    shape: [usize; 2],
}

impl Matrix {
    /// The matrix of the tensor `tensor`, or an error if `tensor` is not 2D
    pub fn new(tensor: PyTensor) -> Result<Self, String> {
        let shape = shape_2d(&tensor)?;
        Ok(Self { tensor, shape })
    }

    pub fn tensor(&self) -> &PyTensor {
        &self.tensor
    }

    /// The `[rows, cols]` shape of the matrix
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl TryFrom<PyTensor> for Matrix {
    type Error = String;

    fn try_from(tensor: PyTensor) -> Result<Self, String> {
        Self::new(tensor)
    }
}

/// Prints the tensor
impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.tensor(), f)
    }
}

/// Prints the tensor (the shape is that of the tensor)
impl fmt::Debug for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.tensor(), f)
    }
}

#[cfg(feature = "pyo3")]
impl<'py> FromPyObject<'py> for Matrix {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        Self::new(ob.extract()?).map_err(PyValueError::new_err)
    }
}

#[cfg(feature = "pyo3")]
impl<'py> IntoPyObject<'py> for Matrix {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        self.tensor().clone().into_pyobject(py)
    }
}

/// The constants of the affine map `x ↦ A·x + B`: the matrix `A` of the shape
/// `[outputs, inputs]`, and `B`, either empty (no bias) or a column vector `[outputs, 1]`
#[derive(Clone, Debug)]
#[cfg_attr(feature = "pyo3", pyclass(frozen))]
pub struct Affine {
    a: Matrix,
    b: PyTensor,
}

impl Affine {
    /// The affine map `x ↦ a·x + b`, or an error if `a` is not 2D, or `b` is neither empty
    /// nor a column vector of the rows of `a`
    pub fn new(a: PyTensor, b: PyTensor) -> Result<Self, String> {
        let a = Matrix::new(a).map_err(|e| format!("`A`: {e}"))?;
        let [out, _] = a.shape();
        if b.numel() != 0 && shape_2d(&b) != Ok([out, 1]) {
            return Err(format!("`B` must be empty or a column vector [{out}x1]"));
        }
        Ok(Self { a, b })
    }

    pub fn a(&self) -> &PyTensor {
        self.a.tensor()
    }

    pub fn b(&self) -> &PyTensor {
        &self.b
    }

    /// The number of the inputs (the columns of `A`)
    pub fn inputs(&self) -> usize {
        self.a.shape()[1]
    }

    /// The number of the outputs (the rows of `A`)
    pub fn outputs(&self) -> usize {
        self.a.shape()[0]
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl Affine {
    #[new]
    fn py_new(a: PyTensor, b: PyTensor) -> PyResult<Self> {
        Self::new(a, b).map_err(PyValueError::new_err)
    }

    #[getter(a)]
    fn py_a(&self) -> PyTensor {
        self.a().clone()
    }

    #[getter(b)]
    fn py_b(&self) -> PyTensor {
        self.b().clone()
    }

    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("a", "b")
    }

    fn __repr__(&self) -> String {
        format!("Affine({}, {})", self.a(), self.b())
    }
}

#[cfg(all(test, feature = "torch"))]
mod tests {
    use super::*;

    fn tensor(size: &[i64]) -> PyTensor {
        tch::Tensor::zeros(size, (tch::Kind::Float, tch::Device::Cpu)).into()
    }

    #[test]
    fn matrix() {
        assert_eq!(Matrix::new(tensor(&[2, 3])).unwrap().shape(), [2, 3]);
        assert_eq!(Matrix::new(tensor(&[0, 3])).unwrap().shape(), [0, 3]);
        assert_eq!(
            Matrix::new(tensor(&[2])).err().unwrap(),
            "expected a 2D tensor, got 1D"
        );
        assert_eq!(
            Matrix::new(tensor(&[1, 2, 3])).err().unwrap(),
            "expected a 2D tensor, got 3D"
        );
    }

    #[test]
    fn affine() {
        let affine = Affine::new(tensor(&[2, 3]), tensor(&[2, 1])).unwrap();
        assert_eq!((affine.inputs(), affine.outputs()), (3, 2));
        // no bias
        assert!(Affine::new(tensor(&[2, 3]), tensor(&[0, 0])).is_ok());
        assert!(Affine::new(tensor(&[2, 3]), tensor(&[0])).is_ok());
        assert_eq!(
            Affine::new(tensor(&[3]), tensor(&[0, 0])).err().unwrap(),
            "`A`: expected a 2D tensor, got 1D"
        );
        for b in [&[2, 2][..], &[1, 1], &[2]] {
            assert_eq!(
                Affine::new(tensor(&[2, 3]), tensor(b)).err().unwrap(),
                "`B` must be empty or a column vector [2x1]"
            );
        }
    }
}
