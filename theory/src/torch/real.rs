//! The Python classes of the operations of real arithmetic carrying tensors.

use crate::real;
use crate::real::Real;
use base::boundary::Mono;
use pyo3::prelude::*;
use pyo3_tch::PyTensor;

/// The linear map `x ↦ A·x` of real matrices: `A` a torch tensor of
/// floating-point elements, held with the sort the map reads, as `A` does not
/// determine its columns.
#[pyclass(frozen, name = "Linear", module = "reactor.real")]
pub struct Linear(super::Tensor, Real);

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
        let a = super::Tensor(a.0);
        super::check_matrix(&a)?;
        if !a.is_floating_point() {
            return Err(crate::Error(format!(
                "expected a tensor of floating-point elements, got {:?}",
                a.kind()
            )));
        }
        let op = match (dom, cod) {
            (Some(dom), cod) => crate::check_cod(real::Linear::with_dom(a, [dom]), cod)?,
            (None, Some(cod)) => real::Linear::with_cod(a, [cod])?,
            (None, None) => {
                return Err(crate::Error(
                    "expected a domain or a codomain, as A does not determine the columns".into(),
                ));
            }
        };
        let dom = super::sole(op.dom());
        Ok(Linear(op.a().clone(), dom))
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

    /// The sorts it reads, as a tuple.
    #[getter]
    fn dom(slf: &Bound<'_, Self>) -> PyResult<Mono<Real>> {
        let op: crate::real::Linear<super::Tensor> = slf.extract()?;
        Ok(op.dom())
    }

    /// The sorts it writes, as a tuple.
    #[getter]
    fn cod(slf: &Bound<'_, Self>) -> PyResult<Mono<Real>> {
        let op: crate::real::Linear<super::Tensor> = slf.extract()?;
        Ok(op.cod())
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for crate::real::Linear<super::Tensor> {
    type Error = PyErr;

    /// Fails when `ob` is not a [`Linear`].
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let linear = ob.cast::<Linear>()?.get();
        Ok(crate::real::Linear::with_dom(linear.0.clone(), [linear.1])?)
    }
}

/// The affine map `x ↦ A·x + B` of real matrices: torch tensors of
/// floating-point elements.
#[pyclass(frozen, name = "Affine", module = "reactor.real")]
pub struct Affine(super::Tensor, super::Tensor, u8);

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
        let a = super::Tensor(a.0);
        let b = super::Tensor(b.0);
        if !a.is_floating_point() || !b.is_floating_point() {
            return Err(crate::Error(format!(
                "expected tensors of floating-point elements, got {:?} and {:?}",
                a.kind(),
                b.kind()
            )));
        }
        super::check_affine(&a, &b)?;
        let dom = dom.unwrap_or_else(|| {
            let ([_, n], [_, k]) = (crate::Matrix::shape(&a), crate::Matrix::shape(&b));
            Real::new_unchecked([n, k])
        });
        let op = crate::check_cod(real::Affine::with_dom(a, b, [dom]), cod)?;
        Ok(Affine(
            op.a().clone(),
            op.b().clone(),
            super::sole(op.cod()).degree(),
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

    /// The tensors it carries, then its domain, when it reads anything, and
    /// its codomain.
    #[classattr]
    fn __match_args__() -> (&'static str, &'static str, &'static str, &'static str) {
        ("a", "b", "dom", "cod")
    }

    /// The sorts it reads, as a tuple.
    #[getter]
    fn dom(slf: &Bound<'_, Self>) -> PyResult<Mono<Real>> {
        let op: crate::real::Affine<super::Tensor> = slf.extract()?;
        Ok(op.dom())
    }

    /// The sorts it writes, as a tuple.
    #[getter]
    fn cod(slf: &Bound<'_, Self>) -> PyResult<Mono<Real>> {
        let op: crate::real::Affine<super::Tensor> = slf.extract()?;
        Ok(op.cod())
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for crate::real::Affine<super::Tensor> {
    type Error = PyErr;

    /// Fails when `ob` is not a [`Affine`].
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let affine = ob.cast::<Affine>()?.get();
        let cod = Real::with_degree(crate::Matrix::shape(&affine.1), affine.2);
        Ok(crate::real::Affine::with_cod(
            affine.0.clone(),
            affine.1.clone(),
            [cod],
        )?)
    }
}

/// A constant real matrix: a `torch.Tensor` of floating-point or integer
/// elements, kept as given, since integers are reals too.
#[pyclass(frozen, name = "Num", module = "reactor.real")]
pub struct Num(super::Tensor, u8);

// SAFETY: the tensor is private to this module, which only reads its
// shape and element kind and takes shallow clones of it; libtorch makes
// these thread-safe, counting the references to a tensor atomically.
unsafe impl Sync for Num {}

#[pymethods]
impl Num {
    /// The constant held in `tensor`, writing `cod`, which it determines but for
    /// its degree: when omitted, it is of degree 0. Fails when `tensor` is not a
    /// `torch.Tensor` of two dimensions and floating-point or integer elements, or
    /// `cod` not of its shape.
    #[new]
    #[pyo3(signature = (tensor, /, cod = None))]
    fn new(tensor: PyTensor, cod: Option<Real>) -> Result<Self, crate::Error> {
        let tensor = super::Tensor(tensor.0);
        super::check_matrix(&tensor)?;
        if !tensor.is_floating_point() && !super::is_integer(tensor.kind()) {
            return Err(crate::Error(format!(
                "expected a tensor of floating-point or integer elements, got {:?}",
                tensor.kind()
            )));
        }
        let cod = match cod {
            Some(cod) => cod,
            None => Real::new(crate::Matrix::shape(&tensor))?,
        };
        let op = real::Num::with_cod(tensor, [cod])?;
        Ok(Num(op.tensor().clone(), cod.degree()))
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

    /// The sorts it writes, as a tuple.
    #[getter]
    fn cod(slf: &Bound<'_, Self>) -> PyResult<Mono<Real>> {
        let op: crate::real::Num<super::Tensor> = slf.extract()?;
        Ok(op.cod())
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for crate::real::Num<super::Tensor> {
    type Error = PyErr;

    /// Fails when `ob` is not a [`Num`].
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let constant = ob.cast::<Num>()?.get();
        let cod = Real::with_degree(crate::Matrix::shape(&constant.0), constant.1);
        Ok(crate::real::Num::with_cod(constant.0.clone(), [cod])?)
    }
}

/// The pointwise product `x ↦ C ⊙ x` with a constant real matrix `C`: a
/// `torch.Tensor` of floating-point elements.
#[pyclass(frozen, name = "Scale", module = "reactor.real")]
pub struct Scale(super::Tensor, u8);

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
        let tensor = super::Tensor(tensor.0);
        super::check_matrix(&tensor)?;
        if !tensor.is_floating_point() {
            return Err(crate::Error(format!(
                "expected a tensor of floating-point elements, got {:?}",
                tensor.kind()
            )));
        }
        let dom = match dom {
            Some(dom) => dom,
            None => Real::new(crate::Matrix::shape(&tensor))?,
        };
        let op = crate::check_cod(real::Scale::with_dom(tensor, [dom]), cod)?;
        Ok(Scale(op.tensor().clone(), dom.degree()))
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

    /// The sorts it reads, as a tuple.
    #[getter]
    fn dom(slf: &Bound<'_, Self>) -> PyResult<Mono<Real>> {
        let op: crate::real::Scale<super::Tensor> = slf.extract()?;
        Ok(op.dom())
    }

    /// The sorts it writes, as a tuple.
    #[getter]
    fn cod(slf: &Bound<'_, Self>) -> PyResult<Mono<Real>> {
        let op: crate::real::Scale<super::Tensor> = slf.extract()?;
        Ok(op.cod())
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for crate::real::Scale<super::Tensor> {
    type Error = PyErr;

    /// Fails when `ob` is not a [`Scale`].
    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let scale = ob.cast::<Scale>()?.get();
        let dom = Real::with_degree(crate::Matrix::shape(&scale.0), scale.1);
        Ok(crate::real::Scale::with_dom(scale.0.clone(), [dom])?)
    }
}

/// Converts an operation of the theory into its Python class: the
/// operations carrying tensors into the classes of this module, holding the
/// same tensors, and the others into their own classes.
impl<'py> IntoPyObject<'py> for crate::real::NRA<super::Tensor> {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        use crate::real::NRA;

        match self {
            NRA::Num(op) => Num(op.tensor().clone(), super::sole(op.cod()).degree())
                .into_pyobject(py)
                .map(Bound::into_any),
            NRA::Scale(op) => Scale(op.tensor().clone(), super::sole(op.dom()).degree())
                .into_pyobject(py)
                .map(Bound::into_any),
            NRA::Add(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Sub(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Mul(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Div(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Linear(op) => Linear(op.a().clone(), super::sole(op.dom()))
                .into_pyobject(py)
                .map(Bound::into_any),
            NRA::Affine(op) => Affine(
                op.a().clone(),
                op.b().clone(),
                super::sole(op.cod()).degree(),
            )
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
            NRA::BroadcastIte(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Le(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Gt(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Ge(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Eq(op) => op.into_pyobject(py).map(Bound::into_any),
            NRA::Ne(op) => op.into_pyobject(py).map(Bound::into_any),
        }
    }
}
