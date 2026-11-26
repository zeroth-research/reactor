//! The Python classes of the operations of integer arithmetic carrying
//! tensors.

use crate::int;
use crate::int::Int;
use pyo3::prelude::*;
use pyo3_tch::PyTensor;

/// The linear map `x ↦ A·x` of integer matrices: `A` a torch tensor of
/// integer elements, held with the sort the map reads, as `A` does not
/// determine its columns.
#[pyclass(frozen, name = "Linear", module = "reactor.int")]
pub struct Linear(tch::Tensor, Int);

// SAFETY: the tensor is private to this module, which only reads its
// shape and element kind and takes shallow clones of it; libtorch makes
// these thread-safe, counting the references to a tensor atomically.
unsafe impl Sync for Linear {}

#[pymethods]
impl Linear {
    /// The map of the tensor `a`, reading `dom`; its codomain `cod`, when
    /// given, must be the one it writes. As `a` does not determine the
    /// columns of `dom`, either may be omitted but not both. Fails when `a`
    /// is not a `torch.Tensor` of two dimensions and integer elements, of
    /// shape `[m, n]`, or `dom` not of shape `[n, k]`, or `cod` not of shape
    /// `[m, k]`.
    #[new]
    #[pyo3(signature = (a, /, dom = None, cod = None))]
    fn new(a: PyTensor, dom: Option<Int>, cod: Option<Int>) -> Result<Self, crate::Error> {
        super::check_matrix(&a)?;
        if !super::is_integer(a.kind()) {
            return Err(crate::Error(format!(
                "expected a tensor of integer elements, got {:?}",
                a.kind()
            )));
        }
        let op = match (dom, cod) {
            (Some(dom), cod) => crate::check_cod(int::Linear::with_dom(a.0, [dom]), cod)?,
            (None, Some(cod)) => int::Linear::with_cod(a.0, [cod])?,
            (None, None) => {
                return Err(crate::Error(
                    "expected a domain or a codomain, as A does not determine the columns".into(),
                ));
            }
        };
        let dom = super::sole(base::boundary::Boundary::dom(&op));
        Ok(Linear(op.a().shallow_clone(), dom))
    }

    #[getter]
    fn a(&self) -> PyTensor {
        PyTensor(self.0.shallow_clone())
    }

    /// The tensors it carries, then its domain, when it reads anything, and
    /// its codomain.
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str, &'static str) {
        ("a", "dom", "cod")
    }

    /// The sorts it reads: the sort alone when it reads one, else their list.
    #[getter]
    fn dom<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let op: crate::int::Linear<tch::Tensor> = slf.extract()?;
        crate::py_sorts(slf.py(), base::boundary::Boundary::dom(&op))
    }

    /// The sort it writes.
    #[getter]
    fn cod<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let op: crate::int::Linear<tch::Tensor> = slf.extract()?;
        crate::py_sorts(slf.py(), base::boundary::Boundary::cod(&op))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for crate::int::Linear<tch::Tensor> {
    type Error = PyErr;

    /// Fails when `ob` is not a [`Linear`].
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let linear = ob.cast::<Linear>()?.get();
        Ok(crate::int::Linear::with_dom(
            linear.0.shallow_clone(),
            [linear.1],
        )?)
    }
}

/// A clone shares the tensor of the original: `tch::Tensor` is not `Clone`,
/// so this takes a shallow clone of it, as libtorch counts its references.
impl Clone for crate::int::Linear<tch::Tensor> {
    fn clone(&self) -> Self {
        let dom = super::sole(base::boundary::Boundary::dom(self));
        crate::int::Linear::with_dom(self.a().shallow_clone(), [dom])
            .expect("a clone has the shapes of the original")
    }
}

/// The affine map `x ↦ A·x + B` of integer matrices: torch tensors of
/// integer elements.
#[pyclass(frozen, name = "Affine", module = "reactor.int")]
pub struct Affine(tch::Tensor, tch::Tensor);

// SAFETY: the tensors are private to this module, which only reads their
// shape and element kind and takes shallow clones of them; libtorch makes
// these thread-safe, counting the references to a tensor atomically.
unsafe impl Sync for Affine {}

#[pymethods]
impl Affine {
    /// The map of the tensors `a` and `b`, reading `dom`, which they determine
    /// and may be omitted; its codomain `cod`, when given, must be the one it
    /// writes. Fails when `a` and `b` are not `torch.Tensor`s of integer
    /// elements, of shapes `[m, n]` and `[m, k]`, or `dom` not of shape
    /// `[n, k]`.
    #[new]
    #[pyo3(signature = (a, b, /, dom = None, cod = None))]
    fn new(
        a: PyTensor,
        b: PyTensor,
        dom: Option<Int>,
        cod: Option<Int>,
    ) -> Result<Self, crate::Error> {
        if !super::is_integer(a.kind()) || !super::is_integer(b.kind()) {
            return Err(crate::Error(format!(
                "expected tensors of integer elements, got {:?} and {:?}",
                a.kind(),
                b.kind()
            )));
        }
        super::check_affine(&a, &b)?;
        let dom = dom.unwrap_or_else(|| {
            let ([_, n], [_, k]) = (crate::Matrix::shape(&a.0), crate::Matrix::shape(&b.0));
            Int::new_unchecked([n, k])
        });
        let op = crate::check_cod(int::Affine::with_dom(a.0, b.0, [dom]), cod)?;
        Ok(Affine(op.a().shallow_clone(), op.b().shallow_clone()))
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
        let op: crate::int::Affine<tch::Tensor> = slf.extract()?;
        crate::py_sorts(slf.py(), base::boundary::Boundary::dom(&op))
    }

    /// The sort it writes.
    #[getter]
    fn cod<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let op: crate::int::Affine<tch::Tensor> = slf.extract()?;
        crate::py_sorts(slf.py(), base::boundary::Boundary::cod(&op))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for crate::int::Affine<tch::Tensor> {
    type Error = PyErr;

    /// Fails when `ob` is not an [`Affine`].
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let affine = ob.cast::<Affine>()?.get();
        let cod = Int::new_unchecked(crate::Matrix::shape(&affine.1));
        Ok(crate::int::Affine::with_cod(
            affine.0.shallow_clone(),
            affine.1.shallow_clone(),
            [cod],
        )?)
    }
}

/// A constant integer matrix: a `torch.Tensor` of integer elements.
#[pyclass(frozen, name = "Num", module = "reactor.int")]
pub struct Num(tch::Tensor);

// SAFETY: the tensor is private to this module, which only reads its
// shape and element kind and takes shallow clones of it; libtorch makes
// these thread-safe, counting the references to a tensor atomically.
unsafe impl Sync for Num {}

#[pymethods]
impl Num {
    /// The constant held in `tensor`, writing `cod`, which it determines and may be
    /// omitted. Fails when `tensor` is not a `torch.Tensor` of two dimensions and
    /// integer elements, or `cod` not of its shape.
    #[new]
    #[pyo3(signature = (tensor, /, cod = None))]
    fn new(tensor: PyTensor, cod: Option<Int>) -> Result<Self, crate::Error> {
        super::check_matrix(&tensor)?;
        if !super::is_integer(tensor.kind()) {
            return Err(crate::Error(format!(
                "expected a tensor of integer elements, got {:?}",
                tensor.kind()
            )));
        }
        let cod = match cod {
            Some(cod) => cod,
            None => Int::new(crate::Matrix::shape(&tensor.0))?,
        };
        let op = int::Num::with_cod(tensor.0, [cod])?;
        Ok(Num(op.tensor().shallow_clone()))
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
        let op: crate::int::Num<tch::Tensor> = slf.extract()?;
        crate::py_sorts(slf.py(), base::boundary::Boundary::cod(&op))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for crate::int::Num<tch::Tensor> {
    type Error = PyErr;

    /// Fails when `ob` is not an [`Num`].
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let constant = ob.cast::<Num>()?.get();
        let cod = Int::new_unchecked(crate::Matrix::shape(&constant.0));
        Ok(crate::int::Num::with_cod(
            constant.0.shallow_clone(),
            [cod],
        )?)
    }
}

/// A clone shares the tensor of the original: `tch::Tensor` is not `Clone`,
/// so this takes a shallow clone of it, as libtorch counts its references.
impl Clone for crate::int::Num<tch::Tensor> {
    fn clone(&self) -> Self {
        let cod = Int::new_unchecked(crate::Matrix::shape(self.tensor()));
        crate::int::Num::with_cod(self.tensor().shallow_clone(), [cod])
            .expect("a clone has the shape of the original")
    }
}

/// The pointwise product `x ↦ C ⊙ x` with a constant integer matrix `C`: a
/// `torch.Tensor` of integer elements.
#[pyclass(frozen, name = "Scale", module = "reactor.int")]
pub struct Scale(tch::Tensor);

// SAFETY: the tensor is private to this module, which only reads its
// shape and element kind and takes shallow clones of it; libtorch makes
// these thread-safe, counting the references to a tensor atomically.
unsafe impl Sync for Scale {}

#[pymethods]
impl Scale {
    /// The product with the constant held in `tensor`, reading `dom`, which
    /// it determines and may be omitted; its codomain `cod`, when given, must
    /// be the one it writes. Fails when `tensor` is not a `torch.Tensor` of
    /// two dimensions and integer elements, or `dom` not of its shape.
    #[new]
    #[pyo3(signature = (tensor, /, dom = None, cod = None))]
    fn new(tensor: PyTensor, dom: Option<Int>, cod: Option<Int>) -> Result<Self, crate::Error> {
        super::check_matrix(&tensor)?;
        if !super::is_integer(tensor.kind()) {
            return Err(crate::Error(format!(
                "expected a tensor of integer elements, got {:?}",
                tensor.kind()
            )));
        }
        let dom = match dom {
            Some(dom) => dom,
            None => Int::new(crate::Matrix::shape(&tensor.0))?,
        };
        let op = crate::check_cod(int::Scale::with_dom(tensor.0, [dom]), cod)?;
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
        let op: crate::int::Scale<tch::Tensor> = slf.extract()?;
        crate::py_sorts(slf.py(), base::boundary::Boundary::dom(&op))
    }

    /// The sort it writes.
    #[getter]
    fn cod<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let op: crate::int::Scale<tch::Tensor> = slf.extract()?;
        crate::py_sorts(slf.py(), base::boundary::Boundary::cod(&op))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for crate::int::Scale<tch::Tensor> {
    type Error = PyErr;

    /// Fails when `ob` is not an [`Scale`].
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let scale = ob.cast::<Scale>()?.get();
        let dom = Int::new_unchecked(crate::Matrix::shape(&scale.0));
        Ok(crate::int::Scale::with_dom(scale.0.shallow_clone(), [dom])?)
    }
}

/// A clone shares the tensor of the original: `tch::Tensor` is not `Clone`,
/// so this takes a shallow clone of it, as libtorch counts its references.
impl Clone for crate::int::Scale<tch::Tensor> {
    fn clone(&self) -> Self {
        let dom = Int::new_unchecked(crate::Matrix::shape(self.tensor()));
        crate::int::Scale::with_dom(self.tensor().shallow_clone(), [dom])
            .expect("a clone has the shape of the original")
    }
}

/// A clone shares the tensor of the original: `tch::Tensor` is not `Clone`,
/// so this takes a shallow clone of it, as libtorch counts its references.
impl Clone for crate::int::Affine<tch::Tensor> {
    fn clone(&self) -> Self {
        let cod = Int::new_unchecked(crate::Matrix::shape(self.b()));
        crate::int::Affine::with_cod(self.a().shallow_clone(), self.b().shallow_clone(), [cod])
            .expect("a clone has the shapes of the original")
    }
}

/// Converts an operation of the theory into its Python class: the
/// operations carrying tensors into the classes of this module, holding the
/// same tensors, and the others into their own classes.
impl<'py> IntoPyObject<'py> for crate::int::NIA<tch::Tensor> {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        use crate::int::NIA;

        match self {
            NIA::Num(op) => Num(op.tensor().shallow_clone())
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
            NIA::Linear(op) => Linear(
                op.a().shallow_clone(),
                super::sole(base::boundary::Boundary::dom(&op)),
            )
            .into_pyobject(py)
            .map(Bound::into_any),
            NIA::Affine(op) => Affine(op.a().shallow_clone(), op.b().shallow_clone())
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
            NIA::BroadcastIte(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Le(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Gt(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Ge(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Eq(op) => op.into_pyobject(py).map(Bound::into_any),
            NIA::Ne(op) => op.into_pyobject(py).map(Bound::into_any),
        }
    }
}
