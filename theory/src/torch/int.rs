//! The Python classes of the operations of integer arithmetic carrying
//! tensors.

use crate::tensor::int;
use crate::tensor::sort::Int;
use pyo3::prelude::*;
use pyo3_tch::PyTensor;

/// The affine map `x ↦ A·x + B` of integer matrices: torch tensors of
/// integer elements.
#[pyclass(frozen, name = "Linear", module = "reactor.tensor.int")]
pub struct Linear(tch::Tensor, tch::Tensor);

// SAFETY: the tensors are private to this module, which only reads their
// shape and element kind and takes shallow clones of them; libtorch makes
// these thread-safe, counting the references to a tensor atomically.
unsafe impl Sync for Linear {}

#[pymethods]
impl Linear {
    /// The map of the tensors `a` and `b`, reading `dom`; its codomain `cod`,
    /// when given, must be the one it writes. Fails when `a` and `b` are not
    /// `torch.Tensor`s of integer elements, of shapes `[m, n]` and `[m, k]`,
    /// or `dom` not of shape `[n, k]`.
    #[new]
    #[pyo3(signature = (a, b, /, dom, cod = None))]
    fn new(a: PyTensor, b: PyTensor, dom: Int, cod: Option<Int>) -> Result<Self, crate::Error> {
        if !super::is_integer(a.kind()) || !super::is_integer(b.kind()) {
            return Err(crate::Error(format!(
                "expected tensors of integer elements, got {:?} and {:?}",
                a.kind(),
                b.kind()
            )));
        }
        super::check_affine(&a, &b)?;
        let op = crate::tensor::check_cod(int::Linear::with_dom(a.0, b.0, [dom]), cod)?;
        Ok(Linear(op.a().shallow_clone(), op.b().shallow_clone()))
    }

    #[getter]
    fn a(&self) -> PyTensor {
        PyTensor(self.0.shallow_clone())
    }

    #[getter]
    fn b(&self) -> PyTensor {
        PyTensor(self.1.shallow_clone())
    }

    /// The tensors it carries, then its domain, when it reads anything, and
    /// its codomain.
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str, &'static str, &'static str) {
        ("a", "b", "dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[getter]
    fn dom<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let op: crate::tensor::int::Linear<tch::Tensor> = slf.extract()?;
        crate::tensor::py_sorts(slf.py(), base::boundary::Boundary::dom(&op))
    }

    /// The sort it writes.
    #[getter]
    fn cod<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let op: crate::tensor::int::Linear<tch::Tensor> = slf.extract()?;
        crate::tensor::py_sorts(slf.py(), base::boundary::Boundary::cod(&op))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for crate::tensor::int::Linear<tch::Tensor> {
    type Error = PyErr;

    /// Fails when `ob` is not an [`Linear`].
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let linear = ob.cast::<Linear>()?.get();
        Ok(crate::tensor::int::Linear::new(
            linear.0.shallow_clone(),
            linear.1.shallow_clone(),
        )?)
    }
}

/// A constant integer matrix: a `torch.Tensor` of integer elements.
#[pyclass(frozen, name = "Const", module = "reactor.tensor.int")]
pub struct Const(tch::Tensor);

// SAFETY: the tensor is private to this module, which only reads its
// shape and element kind and takes shallow clones of it; libtorch makes
// these thread-safe, counting the references to a tensor atomically.
unsafe impl Sync for Const {}

#[pymethods]
impl Const {
    /// The constant held in `tensor`, writing `cod`. Fails when `tensor` is
    /// not a `torch.Tensor` of two dimensions and integer elements, or `cod`
    /// not of its shape.
    #[new]
    #[pyo3(signature = (tensor, /, cod))]
    fn new(tensor: PyTensor, cod: Int) -> Result<Self, crate::Error> {
        super::check_matrix(&tensor)?;
        if !super::is_integer(tensor.kind()) {
            return Err(crate::Error(format!(
                "expected a tensor of integer elements, got {:?}",
                tensor.kind()
            )));
        }
        let op = int::Const::with_cod(tensor.0, [cod])?;
        Ok(Const(op.tensor().shallow_clone()))
    }

    #[getter]
    fn tensor(&self) -> PyTensor {
        PyTensor(self.0.shallow_clone())
    }

    /// The tensors it carries, then its domain, when it reads anything, and
    /// its codomain.
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str) {
        ("tensor", "cod")
    }

    /// The sort it writes.
    #[getter]
    fn cod<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let op: crate::tensor::int::Const<tch::Tensor> = slf.extract()?;
        crate::tensor::py_sorts(slf.py(), base::boundary::Boundary::cod(&op))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for crate::tensor::int::Const<tch::Tensor> {
    type Error = PyErr;

    /// Fails when `ob` is not an [`Const`].
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let constant = ob.cast::<Const>()?.get();
        Ok(crate::tensor::int::Const::new(constant.0.shallow_clone()))
    }
}

/// A clone shares the tensor of the original: `tch::Tensor` is not `Clone`,
/// so this takes a shallow clone of it, as libtorch counts its references.
impl Clone for crate::tensor::int::Const<tch::Tensor> {
    fn clone(&self) -> Self {
        crate::tensor::int::Const::new(self.tensor().shallow_clone())
    }
}

/// The pointwise product `x ↦ C ⊙ x` with a constant integer matrix `C`: a
/// `torch.Tensor` of integer elements.
#[pyclass(frozen, name = "Scale", module = "reactor.tensor.int")]
pub struct Scale(tch::Tensor);

// SAFETY: the tensor is private to this module, which only reads its
// shape and element kind and takes shallow clones of it; libtorch makes
// these thread-safe, counting the references to a tensor atomically.
unsafe impl Sync for Scale {}

#[pymethods]
impl Scale {
    /// The product with the constant held in `tensor`, reading `dom`; its
    /// codomain `cod`, when given, must be the one it writes. Fails when
    /// `tensor` is not a `torch.Tensor` of two dimensions and integer
    /// elements, or `dom` not of its shape.
    #[new]
    #[pyo3(signature = (tensor, /, dom, cod = None))]
    fn new(tensor: PyTensor, dom: Int, cod: Option<Int>) -> Result<Self, crate::Error> {
        super::check_matrix(&tensor)?;
        if !super::is_integer(tensor.kind()) {
            return Err(crate::Error(format!(
                "expected a tensor of integer elements, got {:?}",
                tensor.kind()
            )));
        }
        let op = crate::tensor::check_cod(int::Scale::with_dom(tensor.0, [dom]), cod)?;
        Ok(Scale(op.tensor().shallow_clone()))
    }

    #[getter]
    fn tensor(&self) -> PyTensor {
        PyTensor(self.0.shallow_clone())
    }

    /// The tensors it carries, then its domain, when it reads anything, and
    /// its codomain.
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str, &'static str) {
        ("tensor", "dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[getter]
    fn dom<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let op: crate::tensor::int::Scale<tch::Tensor> = slf.extract()?;
        crate::tensor::py_sorts(slf.py(), base::boundary::Boundary::dom(&op))
    }

    /// The sort it writes.
    #[getter]
    fn cod<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let op: crate::tensor::int::Scale<tch::Tensor> = slf.extract()?;
        crate::tensor::py_sorts(slf.py(), base::boundary::Boundary::cod(&op))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for crate::tensor::int::Scale<tch::Tensor> {
    type Error = PyErr;

    /// Fails when `ob` is not an [`Scale`].
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let scale = ob.cast::<Scale>()?.get();
        Ok(crate::tensor::int::Scale::new(scale.0.shallow_clone()))
    }
}

/// A clone shares the tensor of the original: `tch::Tensor` is not `Clone`,
/// so this takes a shallow clone of it, as libtorch counts its references.
impl Clone for crate::tensor::int::Scale<tch::Tensor> {
    fn clone(&self) -> Self {
        crate::tensor::int::Scale::new(self.tensor().shallow_clone())
    }
}

/// A clone shares the tensor of the original: `tch::Tensor` is not `Clone`,
/// so this takes a shallow clone of it, as libtorch counts its references.
impl Clone for crate::tensor::int::Linear<tch::Tensor> {
    fn clone(&self) -> Self {
        crate::tensor::int::Linear::new(self.a().shallow_clone(), self.b().shallow_clone())
            .expect("a clone has the shapes of the original")
    }
}

/// Converts an operation of the theory into its Python class: the
/// operations carrying tensors into the classes of this module, holding the
/// same tensors, and the others into their own classes.
impl<'py> IntoPyObject<'py> for crate::tensor::int::NIA<tch::Tensor> {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        use crate::tensor::int::NIA;

        match self {
            NIA::Const(op) => Const(op.tensor().shallow_clone())
                .into_pyobject(py)
                .map(Bound::into_any),
            NIA::Scale(op) => Scale(op.tensor().shallow_clone())
                .into_pyobject(py)
                .map(Bound::into_any),
            NIA::Add(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Sub(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Mul(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Div(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Mod(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Linear(op) => Linear(op.a().shallow_clone(), op.b().shallow_clone())
                .into_pyobject(py)
                .map(Bound::into_any),
            NIA::ReLU(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Neg(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Abs(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Argmax(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Min(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Max(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Transpose(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::MatMul(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Lt(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Ite(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Le(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Gt(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Ge(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Eq(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Ne(op) => op.into_pyobject(py).map(Bound::into_any),
        }
    }
}
