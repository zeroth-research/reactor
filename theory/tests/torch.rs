//! Tests for the torch backing of `tensor`.

#![cfg(feature = "torch")]

use base::boundary::Boundary;
use pyo3::prelude::*;
use pyo3_tch::PyTensor;
use theory::tensor;
use theory::tensor::sort::{Bool, Int, Real, Word};
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

/// The Python object of the sort `s`.
fn sort<'py>(py: Python<'py>, s: impl Into<tensor::sort::Any>) -> Bound<'py, PyAny> {
    s.into().into_pyobject(py).unwrap()
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
    let affine = tensor::real::Affine::with_dom(a, b, [Real::new([2, 3])]).unwrap();
    assert_eq!(
        affine.dom(),
        tensor::real::ReLU::with_dom([Real::new([2, 3]); 1])
            .unwrap()
            .dom()
    );
    assert_eq!(
        affine.cod(),
        tensor::real::ReLU::with_dom([Real::new([4, 3]); 1])
            .unwrap()
            .cod()
    );
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

        let reals = sort(py, Real::new([2, 3]));
        let integers = sort(py, Int::new([2, 3]));
        let booleans = sort(py, Bool::new([2, 3]));
        let words = sort(py, Word::new_unchecked(8, [2, 3]));

        assert!(real.call1((&floats, &reals)).is_ok());
        // integers are reals too, kept as given
        assert!(real.call1((&ints, &reals)).is_ok());
        assert!(real.call1((&bools, &reals)).is_err());
        assert!(int.call1((&ints, &integers)).is_ok());
        assert!(int.call1((&bools, &integers)).is_err());
        assert!(boolean.call1((&bools, &booleans)).is_ok());
        assert!(boolean.call1((&floats, &booleans)).is_err());
        assert!(word.call1((&ints, &words)).is_ok());
        assert!(word.call1((&floats, &words)).is_err());
        assert!(real.call1((42, &reals)).is_err());

        // a constant writes its codomain, which is of the shape of its tensor, and of
        // degree 0 for reals unless given; words need it, for their width
        let plain = real.call1((&floats,)).unwrap();
        assert_eq!(
            plain.getattr("cod").unwrap().extract::<Real>().unwrap(),
            Real::new([2, 3])
        );
        assert!(int.call1((&ints,)).is_ok());
        assert!(boolean.call1((&bools,)).is_ok());
        assert!(word.call1((&ints,)).is_err());
        assert!(real.call1((&floats, sort(py, Real::new([3, 2])))).is_err());
        assert!(
            word.call1((&ints, sort(py, Word::new_unchecked(8, [3, 2]))))
                .is_err()
        );
    });
}

#[test]
fn constants_extract_from_the_class_of_their_theory() {
    prepare_python_with_torch();
    Python::attach(|py| {
        let floats = python(py, full(2, 3, 0.0, tch::Kind::Float));
        let constant = py
            .get_type::<real::Const>()
            .call1((&floats, sort(py, Real::new([2, 3]))))
            .unwrap();
        let op = constant
            .extract::<tensor::real::Const<tch::Tensor>>()
            .unwrap();
        assert_eq!(
            op.cod(),
            tensor::real::ReLU::with_dom([Real::new([2, 3]); 1])
                .unwrap()
                .cod()
        );

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
        let constant = py
            .get_type::<word::Const>()
            .call1((&ints, sort(py, Word::new_unchecked(8, [2, 3]))))
            .unwrap();
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
        let real = py.get_type::<real::Affine>();
        let reals = sort(py, Real::new([2, 1]));
        assert!(real.call1((&a, &b, &reals)).is_ok());

        // `B` must have the rows of `A`
        let wrong_rows = python(py, full(3, 1, 1.0, tch::Kind::Float));
        assert!(real.call1((&a, &wrong_rows, &reals)).is_err());

        // and both must be matrices of the theory
        let ints = python(py, full(4, 1, 1.0, tch::Kind::Int64));
        assert!(real.call1((&a, &ints, &reals)).is_err());
        let int = py.get_type::<int::Affine>();
        let integers = sort(py, Int::new([2, 1]));
        let ia = python(py, full(4, 2, 1.0, tch::Kind::Int64));
        assert!(int.call1((&ia, &ints, &integers)).is_ok());
        assert!(int.call1((&a, &b, &integers)).is_err());

        // a map reads its domain, of the columns of `A` and of `B`, plain reals unless
        // given, and writes the codomain, if given, of the rows of `A`
        let plain = real.call1((&a, &b)).unwrap();
        assert_eq!(
            plain.getattr("dom").unwrap().extract::<Real>().unwrap(),
            Real::new([2, 1])
        );
        assert!(real.call1((&a, &b, sort(py, Real::new([2, 3])))).is_err());
        assert!(
            real.call1((&a, &b, &reals, sort(py, Real::new([4, 1]))))
                .is_ok()
        );
        assert!(
            real.call1((&a, &b, &reals, sort(py, Real::new([1, 4]))))
                .is_err()
        );
    });
}

#[test]
fn linear_maps_extract_from_the_class_of_their_theory() {
    prepare_python_with_torch();
    Python::attach(|py| {
        let a = python(py, full(4, 2, 1.0, tch::Kind::Float));
        let b = python(py, full(4, 3, 1.0, tch::Kind::Float));
        let class = py
            .get_type::<real::Affine>()
            .call1((&a, &b, sort(py, Real::new([2, 3]))))
            .unwrap();
        let linear = class
            .extract::<tensor::real::Affine<tch::Tensor>>()
            .unwrap();
        assert_eq!(
            linear.dom(),
            tensor::real::ReLU::with_dom([Real::new([2, 3]); 1])
                .unwrap()
                .dom()
        );
        assert_eq!(
            linear.cod(),
            tensor::real::ReLU::with_dom([Real::new([4, 3]); 1])
                .unwrap()
                .cod()
        );

        // the tensors are those of the class
        let ta = a.extract::<PyTensor>().unwrap();
        assert_eq!(linear.a().data_ptr(), ta.data_ptr());

        // the class decides the theory
        assert!(class.extract::<tensor::int::Affine<tch::Tensor>>().is_err());
        let any = class.extract::<tensor::Any<tch::Tensor>>().unwrap();
        assert!(matches!(
            any,
            tensor::Any::Real(tensor::real::NRA::Affine(_))
        ));
    });
}

#[test]
fn any_extracts_constants_and_linear_maps_with_torch_tensors() {
    prepare_python_with_torch();
    Python::attach(|py| {
        let ints = python(py, full(2, 3, 0.0, tch::Kind::Int64));
        let integers = sort(py, Int::new([2, 3]));
        let constant = py
            .get_type::<int::Const>()
            .call1((&ints, &integers))
            .unwrap();
        let any = constant.extract::<tensor::Any<tch::Tensor>>().unwrap();
        assert!(matches!(any, tensor::Any::Int(tensor::int::NIA::Const(_))));

        let scale = py
            .get_type::<int::Scale>()
            .call1((&ints, &integers))
            .unwrap();
        let any = scale.extract::<tensor::Any<tch::Tensor>>().unwrap();
        assert!(matches!(any, tensor::Any::Int(tensor::int::NIA::Scale(_))));
        // the class decides the operation and the theory
        assert!(scale.extract::<tensor::int::Const<tch::Tensor>>().is_err());
        let floats = python(py, full(2, 3, 0.0, tch::Kind::Float));
        let reals = sort(py, Real::new([2, 3]));
        assert!(
            py.get_type::<int::Scale>()
                .call1((&floats, &integers))
                .is_err()
        );
        assert!(
            py.get_type::<real::Scale>()
                .call1((&floats, &reals))
                .is_ok()
        );

        let bools = python(py, full(2, 3, 0.0, tch::Kind::Bool));
        let constant = py
            .get_type::<bool::Const>()
            .call1((&bools, sort(py, Bool::new([2, 3]))))
            .unwrap();
        let any = constant.extract::<tensor::Any<tch::Tensor>>().unwrap();
        assert!(matches!(
            any,
            tensor::Any::Bool(tensor::bool::Prop::Const(_))
        ));

        let a = python(py, full(4, 2, 1.0, tch::Kind::Int64));
        let b = python(py, full(4, 1, 1.0, tch::Kind::Int64));
        let linear = py
            .get_type::<int::Affine>()
            .call1((&a, &b, sort(py, Int::new([2, 1]))))
            .unwrap();
        let any = linear.extract::<tensor::Any<tch::Tensor>>().unwrap();
        assert!(matches!(any, tensor::Any::Int(tensor::int::NIA::Affine(_))));
    });
}

#[test]
fn any_extracts_the_operations_of_each_theory() {
    use tensor::{Any, bool, int, real, word};

    prepare_python_with_torch();
    Python::attach(|py| {
        let real_add =
            Bound::new(py, real::Add::with_dom([Real::new([2, 3]); 2]).unwrap()).unwrap();
        let op = real_add.extract::<Any<tch::Tensor>>().unwrap();
        assert!(matches!(
            op,
            Any::Real(real::NRA::Add(op)) if op == real::Add::with_dom([Real::new([2, 3]); 2]).unwrap()
        ));

        let int_add = Bound::new(py, int::Add::with_dom([Int::new([2, 3]); 2]).unwrap()).unwrap();
        let op = int_add.extract::<Any<tch::Tensor>>().unwrap();
        assert!(matches!(
            op,
            Any::Int(int::NIA::Add(op)) if op == int::Add::with_dom([Int::new([2, 3]); 2]).unwrap()
        ));

        let and = Bound::new(py, bool::And::with_dom([Bool::new([2, 3]); 2]).unwrap()).unwrap();
        let op = and.extract::<Any<tch::Tensor>>().unwrap();
        assert!(matches!(
            op,
            Any::Bool(bool::Prop::And(op)) if op == bool::And::with_dom([Bool::new([2, 3]); 2]).unwrap()
        ));

        let sdiv = Bound::new(
            py,
            word::signed::Div::with_dom([Word::new(8, [2, 3]).unwrap(); 2]).unwrap(),
        )
        .unwrap();
        let op = sdiv.extract::<Any<tch::Tensor>>().unwrap();
        assert!(matches!(
            op,
            Any::Word(word::BV::SDiv(op)) if op == word::signed::Div::with_dom([Word::new(8, [2, 3]).unwrap(); 2]).unwrap()
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
        let real_mul =
            Bound::new(py, real::Mul::with_dom([Real::new([2, 3]); 2]).unwrap()).unwrap();
        assert!(real_mul.extract::<real::NRA<tch::Tensor>>().is_ok());
        assert!(real_mul.extract::<int::NIA<tch::Tensor>>().is_err());

        let int_mod = Bound::new(py, int::Mod::with_dom([Int::new([2, 3]); 2]).unwrap()).unwrap();
        assert!(int_mod.extract::<int::NIA<tch::Tensor>>().is_ok());

        // `Not` is a different class in propositional logic and in QF_BV
        let not = Bound::new(py, bool::Not::with_dom([Bool::new([2, 3]); 1]).unwrap()).unwrap();
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

        // a codomain of a matrix, so that only the tensor is wrong
        let reals = sort(py, Real::new([1, 1]));
        let integers = sort(py, Int::new([1, 1]));
        let booleans = sort(py, Bool::new([1, 1]));
        let words = sort(py, Word::new_unchecked(8, [1, 1]));
        for size in [&[][..], &[3], &[2, 2, 2]] {
            assert!(
                real.call1((tensor(size, tch::Kind::Float), &reals))
                    .is_err()
            );
            assert!(
                int.call1((tensor(size, tch::Kind::Int64), &integers))
                    .is_err()
            );
            assert!(
                boolean
                    .call1((tensor(size, tch::Kind::Bool), &booleans))
                    .is_err()
            );
            assert!(
                word.call1((tensor(size, tch::Kind::Int64), &words))
                    .is_err()
            );
        }

        // matrices may be empty
        let empty_reals = sort(py, Real::new([0, 3]));
        let empty_words = sort(py, Word::new_unchecked(8, [2, 0]));
        assert!(
            real.call1((tensor(&[0, 3], tch::Kind::Float), &empty_reals))
                .is_ok()
        );
        assert!(
            word.call1((tensor(&[2, 0], tch::Kind::Int64), &empty_words))
                .is_ok()
        );
    });
}

#[test]
fn float_constants_are_built_from_floats_of_their_format() {
    prepare_python_with_torch();
    Python::attach(|py| {
        let float = py.get_type::<theory::torch::float::Const>();
        let singles = python(py, full(2, 3, 1.5, tch::Kind::Float));
        let doubles = python(py, full(2, 3, 1.5, tch::Kind::Double));
        let ints = python(py, full(2, 3, 1.0, tch::Kind::Int64));
        let single = tensor::sort::Float::new(8, 24, [2, 3]).unwrap();

        // the format of the elements is that of the sort, which they determine
        assert!(float.call1((&singles, sort(py, single))).is_ok());
        let inferred = float.call1((&doubles,)).unwrap();
        assert_eq!(
            inferred
                .getattr("cod")
                .unwrap()
                .extract::<tensor::sort::Float>()
                .unwrap(),
            tensor::sort::Float::new(11, 53, [2, 3]).unwrap()
        );
        assert!(float.call1((&doubles, sort(py, single))).is_err());
        assert!(float.call1((&ints,)).is_err());
        assert!(
            float
                .call1((
                    &singles,
                    sort(py, tensor::sort::Float::new(8, 24, [3, 2]).unwrap())
                ))
                .is_err()
        );
        // a format torch has no floats of
        assert!(
            float
                .call1((
                    &singles,
                    sort(py, tensor::sort::Float::new(4, 4, [2, 3]).unwrap())
                ))
                .is_err()
        );

        // the constant extracts, sharing its tensor, through its theory and `Any`
        let constant = float.call1((&singles,)).unwrap();
        let op = constant
            .extract::<tensor::float::Const<tch::Tensor>>()
            .unwrap();
        let cod: base::boundary::Poly<tensor::float::Sort> = base::mono![single.into()].into();
        assert_eq!(op.cod(), cod);
        let t = singles.extract::<PyTensor>().unwrap();
        assert_eq!(op.tensor().data_ptr(), t.data_ptr());
        let any = constant.extract::<tensor::Any<tch::Tensor>>().unwrap();
        assert!(matches!(
            any,
            tensor::Any::Float(tensor::float::FP::Const(_))
        ));
    });
}

#[test]
fn float_operations_extract_through_any() {
    prepare_python_with_torch();
    Python::attach(|py| {
        let x = tensor::sort::Float::new(8, 24, [2, 3]).unwrap();
        let rm = tensor::float::RoundingMode::RTN;
        let add = Bound::new(py, tensor::float::Add::with_dom(rm, [x; 2]).unwrap()).unwrap();
        let any = add.extract::<tensor::Any<tch::Tensor>>().unwrap();
        assert!(matches!(any, tensor::Any::Float(tensor::float::FP::Add(op)) if op.rm() == rm));
        // and back to its class
        let back = any.into_pyobject(py).unwrap();
        assert!(back.cast::<tensor::float::Add>().is_ok());
    });
}
