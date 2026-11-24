//! Tests for the torch backing of `tensor`.

#![cfg(feature = "torch")]

use base::boundary::Boundary;
use pyo3::prelude::*;
use pyo3_tch::PyTensor;
use theory::tensor;
use theory::torch::*;

/// Starts Python and imports `torch` once for all tests, before any of
/// them holds the GIL: tests run in parallel threads, and importing
/// `torch` from several of them at once deadlocks the import.
fn prepare_python_with_torch() {
    static TORCH: std::sync::Once = std::sync::Once::new();
    TORCH.call_once(|| {
        Python::initialize();
        Python::attach(|py| {
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
        theory::Tensor::shape(&full(2, 3, 0.0, tch::Kind::Float)),
        [2, 3]
    );

    // `A` of shape `[4, 2]` and `B` of shape `[4, 3]` map `[2, 3]` to `[4, 3]`
    let a = full(4, 2, 0.0, tch::Kind::Float);
    let b = full(4, 3, 0.0, tch::Kind::Float);
    let linear = tensor::real::Linear::new(a, b).unwrap();
    assert_eq!(linear.dom(), tensor::real::ReLU::new([2, 3]).dom());
    assert_eq!(linear.cod(), tensor::real::ReLU::new([4, 3]).cod());
}

#[test]
fn constants_are_built_from_tensors_of_their_theory_only() {
    prepare_python_with_torch();
    Python::attach(|py| {
        let floats = python(py, full(2, 3, 0.0, tch::Kind::Float));
        let ints = python(py, full(2, 3, 0.0, tch::Kind::Int64));
        let bools = python(py, full(2, 3, 0.0, tch::Kind::Bool));

        let real = py.get_type::<real::Const>();
        let int = py.get_type::<int::Const>();
        let boolean = py.get_type::<bool::Const>();
        let word = py.get_type::<word::Const>();

        assert!(real.call1((&floats,)).is_ok());
        // integers are reals too, kept as given
        assert!(real.call1((&ints,)).is_ok());
        assert!(real.call1((&bools,)).is_err());
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
    Python::attach(|py| {
        let floats = python(py, full(2, 3, 0.0, tch::Kind::Float));
        let constant = py.get_type::<real::Const>().call1((&floats,)).unwrap();
        let op = constant
            .extract::<tensor::real::Const<tch::Tensor>>()
            .unwrap();
        assert_eq!(op.cod(), tensor::real::ReLU::new([2, 3]).cod());

        // the tensor is that of the constant
        let t = floats.extract::<PyTensor>().unwrap();
        assert_eq!(op.tensor().data_ptr(), t.data_ptr());

        // the class decides the theory, not the tensor
        assert!(
            constant
                .extract::<tensor::int::Const<tch::Tensor>>()
                .is_err()
        );
        assert!(
            floats
                .extract::<tensor::real::Const<tch::Tensor>>()
                .is_err()
        );

        let ints = python(py, full(2, 3, 0.0, tch::Kind::Int64));
        let constant = py.get_type::<word::Const>().call1((&ints, 8)).unwrap();
        let op = constant
            .extract::<tensor::word::Const<tch::Tensor>>()
            .unwrap();
        assert_eq!(op.width(), 8);
    });
}

#[test]
fn linear_maps_are_built_from_tensors_of_their_theory_and_shapes() {
    prepare_python_with_torch();
    Python::attach(|py| {
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
    Python::attach(|py| {
        let a = python(py, full(4, 2, 1.0, tch::Kind::Float));
        let b = python(py, full(4, 3, 1.0, tch::Kind::Float));
        let class = py.get_type::<real::Linear>().call1((&a, &b)).unwrap();
        let linear = class
            .extract::<tensor::real::Linear<tch::Tensor>>()
            .unwrap();
        assert_eq!(linear.dom(), tensor::real::ReLU::new([2, 3]).dom());
        assert_eq!(linear.cod(), tensor::real::ReLU::new([4, 3]).cod());

        // the tensors are those of the class
        let ta = a.extract::<PyTensor>().unwrap();
        assert_eq!(linear.a().data_ptr(), ta.data_ptr());

        // the class decides the theory
        assert!(class.extract::<tensor::int::Linear<tch::Tensor>>().is_err());
        let any = class.extract::<tensor::Any<tch::Tensor>>().unwrap();
        assert!(matches!(
            any,
            tensor::Any::Real(tensor::real::NRAT::Linear(_))
        ));
    });
}

#[test]
fn any_extracts_constants_and_linear_maps_with_torch_tensors() {
    prepare_python_with_torch();
    Python::attach(|py| {
        let ints = python(py, full(2, 3, 0.0, tch::Kind::Int64));
        let constant = py.get_type::<int::Const>().call1((&ints,)).unwrap();
        let any = constant.extract::<tensor::Any<tch::Tensor>>().unwrap();
        assert!(matches!(any, tensor::Any::Int(tensor::int::NIA::Const(_))));

        let scale = py.get_type::<int::Scale>().call1((&ints,)).unwrap();
        let any = scale.extract::<tensor::Any<tch::Tensor>>().unwrap();
        assert!(matches!(any, tensor::Any::Int(tensor::int::NIA::Scale(_))));
        // the class decides the operation and the theory
        assert!(scale.extract::<tensor::int::Const<tch::Tensor>>().is_err());
        let floats = python(py, full(2, 3, 0.0, tch::Kind::Float));
        assert!(py.get_type::<int::Scale>().call1((&floats,)).is_err());
        assert!(py.get_type::<real::Scale>().call1((&floats,)).is_ok());

        let bools = python(py, full(2, 3, 0.0, tch::Kind::Bool));
        let constant = py.get_type::<bool::Const>().call1((&bools,)).unwrap();
        let any = constant.extract::<tensor::Any<tch::Tensor>>().unwrap();
        assert!(matches!(
            any,
            tensor::Any::Bool(tensor::bool::Prop::Const(_))
        ));

        let a = python(py, full(4, 2, 1.0, tch::Kind::Int64));
        let b = python(py, full(4, 1, 1.0, tch::Kind::Int64));
        let linear = py.get_type::<int::Linear>().call1((&a, &b)).unwrap();
        let any = linear.extract::<tensor::Any<tch::Tensor>>().unwrap();
        assert!(matches!(any, tensor::Any::Int(tensor::int::NIA::Linear(_))));
    });
}

#[test]
fn any_extracts_the_operations_of_each_theory() {
    use tensor::{Any, bool, int, real, word};

    prepare_python_with_torch();
    Python::attach(|py| {
        let real_add = Bound::new(py, real::Add::new([2, 3])).unwrap();
        let op = real_add.extract::<Any<tch::Tensor>>().unwrap();
        assert!(matches!(
            op,
            Any::Real(real::NRAT::Add(op)) if op == real::Add::new([2, 3])
        ));

        let int_add = Bound::new(py, int::Add::new([2, 3])).unwrap();
        let op = int_add.extract::<Any<tch::Tensor>>().unwrap();
        assert!(matches!(
            op,
            Any::Int(int::NIA::Add(op)) if op == int::Add::new([2, 3])
        ));

        let and = Bound::new(py, bool::And::new([2, 3])).unwrap();
        let op = and.extract::<Any<tch::Tensor>>().unwrap();
        assert!(matches!(
            op,
            Any::Bool(bool::Prop::And(op)) if op == bool::And::new([2, 3])
        ));

        let sdiv = Bound::new(py, word::signed::Div::new(8, [2, 3]).unwrap()).unwrap();
        let op = sdiv.extract::<Any<tch::Tensor>>().unwrap();
        assert!(matches!(
            op,
            Any::Word(word::BV::SDiv(op)) if op == word::signed::Div::new(8, [2, 3]).unwrap()
        ));

        let not_an_operation = 42i64.into_pyobject(py).unwrap();
        assert!(not_an_operation.extract::<Any<tch::Tensor>>().is_err());
    });
}

#[test]
fn each_theory_extracts_its_own_operations_only() {
    use tensor::{bool, int, real, word};

    prepare_python_with_torch();
    Python::attach(|py| {
        let real_mul = Bound::new(py, real::Mul::new([2, 3])).unwrap();
        assert!(real_mul.extract::<real::NRA<tch::Tensor>>().is_ok());
        assert!(real_mul.extract::<int::NIA<tch::Tensor>>().is_err());

        let int_mod = Bound::new(py, int::Mod::new([2, 3])).unwrap();
        assert!(int_mod.extract::<int::NIA<tch::Tensor>>().is_ok());

        // `Not` is a different class in propositional logic and in QF_BV
        let not = Bound::new(py, bool::Not::new([2, 3])).unwrap();
        assert!(not.extract::<bool::Prop<tch::Tensor>>().is_ok());
        assert!(not.extract::<word::BV<tch::Tensor>>().is_err());
    });
}

#[test]
fn constants_are_built_from_matrices_only() {
    prepare_python_with_torch();
    Python::attach(|py| {
        let tensor =
            |size: &[i64], kind| python(py, tch::Tensor::zeros(size, (kind, tch::Device::Cpu)));
        let real = py.get_type::<real::Const>();
        let int = py.get_type::<int::Const>();
        let boolean = py.get_type::<bool::Const>();
        let word = py.get_type::<word::Const>();

        for size in [&[][..], &[3], &[2, 2, 2]] {
            assert!(real.call1((tensor(size, tch::Kind::Float),)).is_err());
            assert!(int.call1((tensor(size, tch::Kind::Int64),)).is_err());
            assert!(boolean.call1((tensor(size, tch::Kind::Bool),)).is_err());
            assert!(word.call1((tensor(size, tch::Kind::Int64), 8)).is_err());
        }

        // matrices may be empty
        assert!(real.call1((tensor(&[0, 3], tch::Kind::Float),)).is_ok());
        assert!(word.call1((tensor(&[2, 0], tch::Kind::Int64), 8)).is_ok());
    });
}
