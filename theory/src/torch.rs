/*!
# Torch

The torch backing of the operations of [`crate::op`]: [`tch::Tensor`]
implements [`crate::Tensor`], so that the operations carrying tensors hold
torch tensors directly, as `op::real::Const<tch::Tensor>`. The Python
`torch.Tensor` objects convert to and from them through
[`pyo3_tch::PyTensor`], sharing their storage.

The constants are Python classes of their own, one per theory, as a tensor
alone does not tell the theory of its constant: [`real::Const`],
[`int::Const`], [`bool::Const`] and [`word::Const`]. Each holds, in a
private field, a [`tch::Tensor`] of the element kind of its theory, checked
on construction, and [`word::Const`] its width too. A [`tch::Tensor`] is not
`Sync`, which a Python class must be, so each class implements `Sync`
itself, as only its own module touches the tensor. The constants of
[`crate::op`] extract from the class of their theory, sharing its tensor.

The affine maps are Python classes too, [`real::Linear`] and
[`int::Linear`], each holding, in private fields, the [`tch::Tensor`]s `A`
and `B` of its theory, of shapes `[m, n]` and `[m, k]`, checked on
construction, and implementing `Sync` in the same way. [`crate::op::real::Linear`]
and [`crate::op::int::Linear`] extract from the class of their theory, sharing its
tensors. So does [`real::TLinear`], the affine map on rates of change.

The operations convert into these classes too, sharing their tensors, and
both ways work with any tensor type that converts from and into a
[`tch::Tensor`], such as [`crate::PyTensor`], which is `Sync` and `Clone`
where a [`tch::Tensor`] is neither. The classes are in the Python modules
of their theories, `reactor.op.real` and so on, which [`add_classes`]
fills, and take their tensors apart in `match`, as the classes of the
other operations do.
*/

pub mod bool;
pub mod int;
pub mod real;
pub mod word;

use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;

impl crate::Tensor for tch::Tensor {
    /// The `[rows, cols]` shape of the tensor, which must be a matrix.
    fn shape(&self) -> [usize; 2] {
        let size = self.size();
        let [rows, cols] = size[..] else {
            panic!("expected a matrix, got a tensor of shape {size:?}")
        };
        [rows as usize, cols as usize]
    }
}

/// Whether `kind` is an integer element kind.
fn is_integer(kind: tch::Kind) -> bool {
    matches!(
        kind,
        tch::Kind::Uint8 | tch::Kind::Int8 | tch::Kind::Int16 | tch::Kind::Int | tch::Kind::Int64
    )
}

/// Fails unless `a` and `b` are the matrices of an affine map `x ↦ A·x + B`:
/// `A` of shape `[m, n]` and `B` of shape `[m, k]`.
fn check_affine(a: &tch::Tensor, b: &tch::Tensor) -> PyResult<()> {
    let (a, b) = (a.size(), b.size());
    if a.len() != 2 || b.len() != 2 || a[0] != b[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "expected A of shape [m, n] and B of shape [m, k], got {a:?} and {b:?}"
        )));
    }
    Ok(())
}

/// The Python `repr` of `tensor`, as torch shows it.
fn repr(py: Python<'_>, tensor: &tch::Tensor) -> PyResult<String> {
    let tensor = pyo3_tch::PyTensor(tensor.shallow_clone()).into_pyobject(py)?;
    Ok(tensor.repr()?.to_string())
}

/// `tensor` as an operation of [`crate::op`] displays it, as in the modules
/// of `reactor`.
fn display(tensor: &tch::Tensor) -> crate::PyTensor {
    tensor.shallow_clone().into()
}

/// Adds the classes of the operations carrying tensors to the modules of
/// their theories, `real`, `int`, `bool` and `word`, in `module`, which
/// [`crate::op::py::add_modules`] made.
pub fn add_classes(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let theory = |name: &str| -> PyResult<Bound<'_, PyModule>> {
        Ok(module.getattr(name)?.downcast_into::<PyModule>()?)
    };
    let real = theory("real")?;
    real.add_class::<real::Const>()?;
    real.add_class::<real::Linear>()?;
    real.add_class::<real::TLinear>()?;
    let int = theory("int")?;
    int.add_class::<int::Const>()?;
    int.add_class::<int::Linear>()?;
    theory("bool")?.add_class::<bool::Const>()?;
    theory("word")?.add_class::<word::Const>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op;
    use base::boundary::Boundary;
    use pyo3_tch::PyTensor;

    /// Starts Python and imports `torch` once for all tests, before any of
    /// them holds the GIL: tests run in parallel threads, and importing
    /// `torch` from several of them at once deadlocks the import.
    fn prepare_python_with_torch() {
        static TORCH: std::sync::Once = std::sync::Once::new();
        TORCH.call_once(|| {
            pyo3::prepare_freethreaded_python();
            Python::with_gil(|py| {
                py.import("torch").unwrap();
            });
        });
    }

    /// A tensor of `value`s of shape `[rows, cols]` and element `kind`.
    fn full(rows: i64, cols: i64, value: f64, kind: tch::Kind) -> tch::Tensor {
        tch::Tensor::full([rows, cols], value, (kind, tch::Device::Cpu))
    }

    /// The Python `torch.Tensor` of `tensor`, sharing it.
    fn python<'py>(py: Python<'py>, tensor: tch::Tensor) -> Bound<'py, PyAny> {
        PyTensor(tensor).into_pyobject(py).unwrap()
    }

    #[test]
    fn torch_tensors_back_the_operations() {
        assert_eq!(
            crate::Tensor::shape(&full(2, 3, 0.0, tch::Kind::Float)),
            [2, 3]
        );

        // `A` of shape `[4, 2]` and `B` of shape `[4, 3]` map `[2, 3]` to `[4, 3]`
        let a = full(4, 2, 0.0, tch::Kind::Float);
        let b = full(4, 3, 0.0, tch::Kind::Float);
        let linear = op::real::Linear(a, b);
        assert_eq!(linear.dom(), op::real::ReLU(2, 3).dom());
        assert_eq!(linear.cod(), op::real::ReLU(4, 3).cod());
    }

    #[test]
    fn constants_are_built_from_tensors_of_their_theory_only() {
        prepare_python_with_torch();
        Python::with_gil(|py| {
            let floats = python(py, full(2, 3, 0.0, tch::Kind::Float));
            let ints = python(py, full(2, 3, 0.0, tch::Kind::Int64));
            let bools = python(py, full(2, 3, 0.0, tch::Kind::Bool));

            let real = py.get_type::<real::Const>();
            let int = py.get_type::<int::Const>();
            let boolean = py.get_type::<bool::Const>();
            let word = py.get_type::<word::Const>();

            assert!(real.call1((&floats,)).is_ok());
            assert!(real.call1((&ints,)).is_err());
            assert!(int.call1((&ints,)).is_ok());
            assert!(int.call1((&bools,)).is_err());
            assert!(boolean.call1((&bools,)).is_ok());
            assert!(boolean.call1((&floats,)).is_err());
            assert!(word.call1((&ints, 8)).is_ok());
            assert!(word.call1((&floats, 8)).is_err());
            assert!(real.call1((42,)).is_err());
        });
    }

    #[test]
    fn constants_extract_from_the_class_of_their_theory() {
        prepare_python_with_torch();
        Python::with_gil(|py| {
            let floats = python(py, full(2, 3, 0.0, tch::Kind::Float));
            let constant = py.get_type::<real::Const>().call1((&floats,)).unwrap();
            let op = constant.extract::<op::real::Const<tch::Tensor>>().unwrap();
            assert_eq!(op.cod(), op::real::ReLU(2, 3).cod());

            // the tensor is that of the constant
            let t = floats.extract::<PyTensor>().unwrap();
            assert_eq!(op.0.data_ptr(), t.data_ptr());

            // the class decides the theory, not the tensor
            assert!(constant.extract::<op::int::Const<tch::Tensor>>().is_err());
            assert!(floats.extract::<op::real::Const<tch::Tensor>>().is_err());

            let ints = python(py, full(2, 3, 0.0, tch::Kind::Int64));
            let constant = py.get_type::<word::Const>().call1((&ints, 8)).unwrap();
            let op = constant.extract::<op::word::Const<tch::Tensor>>().unwrap();
            assert_eq!(op.1, 8);
        });
    }

    #[test]
    fn linear_maps_are_built_from_tensors_of_their_theory_and_shapes() {
        prepare_python_with_torch();
        Python::with_gil(|py| {
            let a = python(py, full(4, 2, 1.0, tch::Kind::Float));
            let b = python(py, full(4, 1, 1.0, tch::Kind::Float));
            let real = py.get_type::<real::Linear>();
            assert!(real.call1((&a, &b)).is_ok());

            // `B` must have the rows of `A`
            let wrong_rows = python(py, full(3, 1, 1.0, tch::Kind::Float));
            assert!(real.call1((&a, &wrong_rows)).is_err());

            // and both must be matrices of the theory
            let ints = python(py, full(4, 1, 1.0, tch::Kind::Int64));
            assert!(real.call1((&a, &ints)).is_err());
            let int = py.get_type::<int::Linear>();
            let ia = python(py, full(4, 2, 1.0, tch::Kind::Int64));
            assert!(int.call1((&ia, &ints)).is_ok());
            assert!(int.call1((&a, &b)).is_err());
        });
    }

    #[test]
    fn linear_maps_extract_from_the_class_of_their_theory() {
        prepare_python_with_torch();
        Python::with_gil(|py| {
            let a = python(py, full(4, 2, 1.0, tch::Kind::Float));
            let b = python(py, full(4, 3, 1.0, tch::Kind::Float));
            let class = py.get_type::<real::Linear>().call1((&a, &b)).unwrap();
            let linear = class.extract::<op::real::Linear<tch::Tensor>>().unwrap();
            assert_eq!(linear.dom(), op::real::ReLU(2, 3).dom());
            assert_eq!(linear.cod(), op::real::ReLU(4, 3).cod());

            // the tensors are those of the class
            let ta = a.extract::<PyTensor>().unwrap();
            assert_eq!(linear.0.data_ptr(), ta.data_ptr());

            // the class decides the theory
            assert!(class.extract::<op::int::Linear<tch::Tensor>>().is_err());
            let any = class.extract::<op::Any<tch::Tensor>>().unwrap();
            assert!(matches!(any, op::Any::Real(op::real::NRA::Linear(_))));
        });
    }

    #[test]
    fn any_extracts_constants_and_linear_maps_with_torch_tensors() {
        prepare_python_with_torch();
        Python::with_gil(|py| {
            let ints = python(py, full(2, 3, 0.0, tch::Kind::Int64));
            let constant = py.get_type::<int::Const>().call1((&ints,)).unwrap();
            let any = constant.extract::<op::Any<tch::Tensor>>().unwrap();
            assert!(matches!(any, op::Any::Int(op::int::NIA::Const(_))));

            let bools = python(py, full(2, 3, 0.0, tch::Kind::Bool));
            let constant = py.get_type::<bool::Const>().call1((&bools,)).unwrap();
            let any = constant.extract::<op::Any<tch::Tensor>>().unwrap();
            assert!(matches!(any, op::Any::Bool(op::bool::Prop::Const(_))));

            let a = python(py, full(4, 2, 1.0, tch::Kind::Int64));
            let b = python(py, full(4, 1, 1.0, tch::Kind::Int64));
            let linear = py.get_type::<int::Linear>().call1((&a, &b)).unwrap();
            let any = linear.extract::<op::Any<tch::Tensor>>().unwrap();
            assert!(matches!(any, op::Any::Int(op::int::NIA::Linear(_))));
        });
    }

    #[test]
    fn rate_linear_maps_are_built_as_linear_maps() {
        prepare_python_with_torch();
        Python::with_gil(|py| {
            let a = python(py, full(4, 2, 1.0, tch::Kind::Float));
            let b = python(py, full(4, 3, 1.0, tch::Kind::Float));
            let class = py.get_type::<real::TLinear>();
            let linear = class.call1((&a, &b, 1)).unwrap();
            let op = linear.extract::<op::real::TLinear<tch::Tensor>>().unwrap();
            assert_eq!(op.dom(), op::real::TAdd(2, 3, 1).cod());
            assert_eq!(op.cod(), op::real::TAdd(4, 3, 1).cod());
            assert_eq!(op.2, 1);

            let wrong_rows = python(py, full(3, 3, 1.0, tch::Kind::Float));
            assert!(class.call1((&a, &wrong_rows, 1)).is_err());
            let ints = python(py, full(4, 3, 1.0, tch::Kind::Int64));
            assert!(class.call1((&a, &ints, 1)).is_err());
        });
    }

    #[test]
    fn operations_convert_into_the_class_of_their_theory_sharing_tensors() {
        prepare_python_with_torch();
        Python::with_gil(|py| {
            // held in a `crate::PyTensor`, as the Python bindings hold them
            let tensor: crate::PyTensor = full(2, 3, 1.0, tch::Kind::Float).into();
            let pointer = tensor.data_ptr();
            let constant = op::Any::Real(op::real::Const(tensor).into());
            let class = constant.into_pyobject(py).unwrap();
            assert!(class.downcast::<real::Const>().is_ok());

            let back = class.extract::<op::Any<crate::PyTensor>>().unwrap();
            let op::Any::Real(op::real::NRA::Const(op::real::Const(tensor))) = back else {
                panic!("expected a real constant, got {back:?}");
            };
            assert_eq!(tensor.data_ptr(), pointer);

            // the operations carrying no data convert into their own classes
            let add = op::Any::<crate::PyTensor>::Int(op::int::Add(2, 3).into());
            let class = add.into_pyobject(py).unwrap();
            assert!(class.downcast::<op::int::Add>().is_ok());
        });
    }
}
