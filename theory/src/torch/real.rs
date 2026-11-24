//! The Python classes of the operations of real arithmetic carrying tensors.

use crate::tensor::real;
use crate::tensor::sort::Real;
use pyo3::prelude::*;
use pyo3_tch::PyTensor;

/// The linear map `x ↦ A·x` of real matrices: `A` a torch tensor of
/// floating-point elements, held with the sort the map reads, as `A` does not
/// determine its columns.
#[pyclass(frozen, name = "Linear", module = "reactor.tensor.real")]
pub struct Linear(tch::Tensor, Real);

// SAFETY: the tensor is private to this module, which only reads its
// shape and element kind and takes shallow clones of it; libtorch makes
// these thread-safe, counting the references to a tensor atomically.
unsafe impl Sync for Linear {}

#[pymethods]
impl Linear {
    /// The map of the tensor `a`, reading `dom`; its codomain `cod`, when
    /// given, must be the one it writes. As `a` does not determine the
    /// columns of `dom`, either may be omitted but not both. Fails when `a`
    /// is not a `torch.Tensor` of two dimensions and floating-point elements,
    /// of shape `[m, n]`, or `dom` not of shape `[n, k]`, or `cod` not of
    /// shape `[m, k]`.
    #[new]
    #[pyo3(signature = (a, /, dom = None, cod = None))]
    fn new(a: PyTensor, dom: Option<Real>, cod: Option<Real>) -> Result<Self, crate::Error> {
        super::check_matrix(&a)?;
        if !a.is_floating_point() {
            return Err(crate::Error(format!(
                "expected a tensor of floating-point elements, got {:?}",
                a.kind()
            )));
        }
        let op = match (dom, cod) {
            (Some(dom), cod) => crate::tensor::check_cod(real::Linear::with_dom(a.0, [dom]), cod)?,
            (None, Some(cod)) => real::Linear::with_cod(a.0, [cod])?,
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

    #[getter]
    fn degree(&self) -> u8 {
        self.1.degree()
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
        let op: crate::tensor::real::Linear<tch::Tensor> = slf.extract()?;
        crate::tensor::py_sorts(slf.py(), base::boundary::Boundary::dom(&op))
    }

    /// The sort it writes.
    #[getter]
    fn cod<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let op: crate::tensor::real::Linear<tch::Tensor> = slf.extract()?;
        crate::tensor::py_sorts(slf.py(), base::boundary::Boundary::cod(&op))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for crate::tensor::real::Linear<tch::Tensor> {
    type Error = PyErr;

    /// Fails when `ob` is not a [`Linear`].
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let linear = ob.cast::<Linear>()?.get();
        Ok(crate::tensor::real::Linear::with_dom(
            linear.0.shallow_clone(),
            [linear.1],
        )?)
    }
}

/// A clone shares the tensor of the original: `tch::Tensor` is not `Clone`,
/// so this takes a shallow clone of it, as libtorch counts its references.
impl Clone for crate::tensor::real::Linear<tch::Tensor> {
    fn clone(&self) -> Self {
        let dom = super::sole(base::boundary::Boundary::dom(self));
        crate::tensor::real::Linear::with_dom(self.a().shallow_clone(), [dom])
            .expect("a clone has the sort of the original")
    }
}

/// The affine map `x ↦ A·x + B` of real matrices: torch tensors of
/// floating-point elements.
#[pyclass(frozen, name = "Affine", module = "reactor.tensor.real")]
pub struct Affine(tch::Tensor, tch::Tensor, u8);

// SAFETY: the tensors are private to this module, which only reads their
// shape and element kind and takes shallow clones of them; libtorch makes
// these thread-safe, counting the references to a tensor atomically.
unsafe impl Sync for Affine {}

#[pymethods]
impl Affine {
    /// The map of the tensors `a` and `b`, reading `dom`, which they determine
    /// but for its degree: when omitted, it is of degree 0. Its codomain `cod`,
    /// when given, must be the one it writes. Fails when `a` and `b` are not
    /// `torch.Tensor`s of floating-point elements, of shapes `[m, n]` and
    /// `[m, k]`, or `dom` not of shape `[n, k]`.
    #[new]
    #[pyo3(signature = (a, b, /, dom = None, cod = None))]
    fn new(
        a: PyTensor,
        b: PyTensor,
        dom: Option<Real>,
        cod: Option<Real>,
    ) -> Result<Self, crate::Error> {
        if !a.is_floating_point() || !b.is_floating_point() {
            return Err(crate::Error(format!(
                "expected tensors of floating-point elements, got {:?} and {:?}",
                a.kind(),
                b.kind()
            )));
        }
        super::check_affine(&a, &b)?;
        let dom = dom.unwrap_or_else(|| {
            let ([_, n], [_, k]) = (crate::Tensor::shape(&a.0), crate::Tensor::shape(&b.0));
            Real::new([n, k])
        });
        let op = crate::tensor::check_cod(real::Affine::with_dom(a.0, b.0, [dom]), cod)?;
        Ok(Affine(
            op.a().shallow_clone(),
            op.b().shallow_clone(),
            op.degree(),
        ))
    }

    #[getter]
    fn a(&self) -> PyTensor {
        PyTensor(self.0.shallow_clone())
    }

    #[getter]
    fn b(&self) -> PyTensor {
        PyTensor(self.1.shallow_clone())
    }

    #[getter]
    fn degree(&self) -> u8 {
        self.2
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
        let op: crate::tensor::real::Affine<tch::Tensor> = slf.extract()?;
        crate::tensor::py_sorts(slf.py(), base::boundary::Boundary::dom(&op))
    }

    /// The sort it writes.
    #[getter]
    fn cod<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let op: crate::tensor::real::Affine<tch::Tensor> = slf.extract()?;
        crate::tensor::py_sorts(slf.py(), base::boundary::Boundary::cod(&op))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for crate::tensor::real::Affine<tch::Tensor> {
    type Error = PyErr;

    /// Fails when `ob` is not a [`Affine`].
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let affine = ob.cast::<Affine>()?.get();
        let cod = Real::with_degree(crate::Tensor::shape(&affine.1), affine.2);
        Ok(crate::tensor::real::Affine::with_cod(
            affine.0.shallow_clone(),
            affine.1.shallow_clone(),
            [cod],
        )?)
    }
}

/// A constant real matrix: a `torch.Tensor` of floating-point or integer
/// elements, kept as given, since integers are reals too.
#[pyclass(frozen, name = "Const", module = "reactor.tensor.real")]
pub struct Const(tch::Tensor, u8);

// SAFETY: the tensor is private to this module, which only reads its
// shape and element kind and takes shallow clones of it; libtorch makes
// these thread-safe, counting the references to a tensor atomically.
unsafe impl Sync for Const {}

#[pymethods]
impl Const {
    /// The constant held in `tensor`, writing `cod`, which it determines but for
    /// its degree: when omitted, it is of degree 0. Fails when `tensor` is not a
    /// `torch.Tensor` of two dimensions and floating-point or integer elements, or
    /// `cod` not of its shape.
    #[new]
    #[pyo3(signature = (tensor, /, cod = None))]
    fn new(tensor: PyTensor, cod: Option<Real>) -> Result<Self, crate::Error> {
        super::check_matrix(&tensor)?;
        if !tensor.is_floating_point() && !super::is_integer(tensor.kind()) {
            return Err(crate::Error(format!(
                "expected a tensor of floating-point or integer elements, got {:?}",
                tensor.kind()
            )));
        }
        let cod = cod.unwrap_or_else(|| Real::new(crate::Tensor::shape(&tensor.0)));
        let op = real::Const::with_cod(tensor.0, [cod])?;
        Ok(Const(op.tensor().shallow_clone(), op.degree()))
    }

    #[getter]
    fn tensor(&self) -> PyTensor {
        PyTensor(self.0.shallow_clone())
    }

    #[getter]
    fn degree(&self) -> u8 {
        self.1
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
        let op: crate::tensor::real::Const<tch::Tensor> = slf.extract()?;
        crate::tensor::py_sorts(slf.py(), base::boundary::Boundary::cod(&op))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for crate::tensor::real::Const<tch::Tensor> {
    type Error = PyErr;

    /// Fails when `ob` is not a [`Const`].
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let constant = ob.cast::<Const>()?.get();
        let cod = Real::with_degree(crate::Tensor::shape(&constant.0), constant.1);
        Ok(crate::tensor::real::Const::with_cod(
            constant.0.shallow_clone(),
            [cod],
        )?)
    }
}

/// A clone shares the tensor of the original: `tch::Tensor` is not `Clone`,
/// so this takes a shallow clone of it, as libtorch counts its references.
impl Clone for crate::tensor::real::Const<tch::Tensor> {
    fn clone(&self) -> Self {
        let cod = Real::with_degree(crate::Tensor::shape(self.tensor()), self.degree());
        crate::tensor::real::Const::with_cod(self.tensor().shallow_clone(), [cod])
            .expect("a clone has the sort of the original")
    }
}

/// The pointwise product `x ↦ C ⊙ x` with a constant real matrix `C`: a
/// `torch.Tensor` of floating-point elements.
#[pyclass(frozen, name = "Scale", module = "reactor.tensor.real")]
pub struct Scale(tch::Tensor, u8);

// SAFETY: the tensor is private to this module, which only reads its
// shape and element kind and takes shallow clones of it; libtorch makes
// these thread-safe, counting the references to a tensor atomically.
unsafe impl Sync for Scale {}

#[pymethods]
impl Scale {
    /// The product with the constant held in `tensor`, reading `dom`, which
    /// it determines but for its degree: when omitted, it is of degree 0. Its
    /// codomain `cod`, when given, must be the one it writes. Fails when
    /// `tensor` is not a `torch.Tensor` of two dimensions and floating-point
    /// elements, or `dom` not of its shape.
    #[new]
    #[pyo3(signature = (tensor, /, dom = None, cod = None))]
    fn new(tensor: PyTensor, dom: Option<Real>, cod: Option<Real>) -> Result<Self, crate::Error> {
        super::check_matrix(&tensor)?;
        if !tensor.is_floating_point() {
            return Err(crate::Error(format!(
                "expected a tensor of floating-point elements, got {:?}",
                tensor.kind()
            )));
        }
        let dom = dom.unwrap_or_else(|| Real::new(crate::Tensor::shape(&tensor.0)));
        let op = crate::tensor::check_cod(real::Scale::with_dom(tensor.0, [dom]), cod)?;
        Ok(Scale(op.tensor().shallow_clone(), op.degree()))
    }

    #[getter]
    fn tensor(&self) -> PyTensor {
        PyTensor(self.0.shallow_clone())
    }

    #[getter]
    fn degree(&self) -> u8 {
        self.1
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
        let op: crate::tensor::real::Scale<tch::Tensor> = slf.extract()?;
        crate::tensor::py_sorts(slf.py(), base::boundary::Boundary::dom(&op))
    }

    /// The sort it writes.
    #[getter]
    fn cod<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let op: crate::tensor::real::Scale<tch::Tensor> = slf.extract()?;
        crate::tensor::py_sorts(slf.py(), base::boundary::Boundary::cod(&op))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for crate::tensor::real::Scale<tch::Tensor> {
    type Error = PyErr;

    /// Fails when `ob` is not a [`Scale`].
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let scale = ob.cast::<Scale>()?.get();
        let dom = Real::with_degree(crate::Tensor::shape(&scale.0), scale.1);
        Ok(crate::tensor::real::Scale::with_dom(
            scale.0.shallow_clone(),
            [dom],
        )?)
    }
}

/// A clone shares the tensor of the original: `tch::Tensor` is not `Clone`,
/// so this takes a shallow clone of it, as libtorch counts its references.
impl Clone for crate::tensor::real::Scale<tch::Tensor> {
    fn clone(&self) -> Self {
        let dom = Real::with_degree(crate::Tensor::shape(self.tensor()), self.degree());
        crate::tensor::real::Scale::with_dom(self.tensor().shallow_clone(), [dom])
            .expect("a clone has the sort of the original")
    }
}

/// A clone shares the tensor of the original: `tch::Tensor` is not `Clone`,
/// so this takes a shallow clone of it, as libtorch counts its references.
impl Clone for crate::tensor::real::Affine<tch::Tensor> {
    fn clone(&self) -> Self {
        let cod = Real::with_degree(crate::Tensor::shape(self.b()), self.degree());
        crate::tensor::real::Affine::with_cod(
            self.a().shallow_clone(),
            self.b().shallow_clone(),
            [cod],
        )
        .expect("a clone has the shapes of the original")
    }
}

/// Converts an operation of the theory into its Python class: the
/// operations carrying tensors into the classes of this module, holding the
/// same tensors, and the others into their own classes.
impl<'py> IntoPyObject<'py> for crate::tensor::real::NRA<tch::Tensor> {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        use crate::tensor::real::NRA;

        match self {
            NRA::Const(op) => Const(op.tensor().shallow_clone(), op.degree())
                .into_pyobject(py)
                .map(Bound::into_any),
            NRA::Scale(op) => Scale(op.tensor().shallow_clone(), op.degree())
                .into_pyobject(py)
                .map(Bound::into_any),
            NRA::Add(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Sub(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Mul(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Div(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Linear(op) => Linear(
                op.a().shallow_clone(),
                super::sole(base::boundary::Boundary::dom(&op)),
            )
            .into_pyobject(py)
            .map(Bound::into_any),
            NRA::Affine(op) => Affine(op.a().shallow_clone(), op.b().shallow_clone(), op.degree())
                .into_pyobject(py)
                .map(Bound::into_any),
            NRA::ReLU(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Abs(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Neg(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Zero(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Argmax(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Min(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Max(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Transpose(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::MatMul(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Lt(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Ite(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Le(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Gt(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Ge(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Eq(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Ne(op) => op.into_pyobject(py).map(Bound::into_any),
        }
    }
}
