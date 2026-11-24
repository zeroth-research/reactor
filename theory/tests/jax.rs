//! Tests for the JAX backing of `tensor`, and its Python classes. They need JAX, as the `jax` dependency group of the
//! Python package installs it, and pass without testing anything when it
//! is missing, as the Python tests of JAX skip.

#![cfg(feature = "jax")]

use base::boundary::{Boundary, Poly};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use std::ffi::CString;
use theory::jax::{Array, bool, int, real, word};
use theory::tensor;
use theory::tensor::sort;

/// Starts Python and imports `jax.numpy` once for all tests, before any of
/// them holds the GIL, as importing from several threads at once may
/// deadlock. Whether JAX is installed.
fn jax() -> bool {
    static JAX: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *JAX.get_or_init(|| {
        Python::initialize();
        Python::attach(|py| {
            // with `torch`, the binary links libtorch, and a process that
            // links it but imports JAX alone crashes as it exits
            #[cfg(feature = "torch")]
            py.import("torch").unwrap();
            match py.import("jax.numpy") {
                Ok(_) => true,
                Err(e) => {
                    eprintln!("not testing JAX, which is not installed: {e}");
                    false
                }
            }
        })
    })
}

/// The value of the Python expression `expr`, in which `jnp` is
/// `jax.numpy` and `np` is `numpy`.
fn eval<'py>(py: Python<'py>, expr: &str) -> Bound<'py, PyAny> {
    let modules = [
        ("jnp", py.import("jax.numpy").unwrap()),
        ("np", py.import("numpy").unwrap()),
    ];
    let locals = modules.into_py_dict(py).unwrap();
    let expr = CString::new(expr).unwrap();
    py.eval(&expr, None, Some(&locals)).unwrap()
}

/// The JAX array of the Python expression `expr`, held.
fn array(py: Python<'_>, expr: &str) -> Array {
    Array::new(&eval(py, expr)).unwrap()
}

#[test]
fn jax_arrays_back_the_operations() {
    if !jax() {
        return;
    }
    Python::attach(|py| {
        let a = array(py, "jnp.ones((4, 2))");
        assert_eq!(theory::Tensor::shape(&a), [4, 2]);

        // `A` of shape `[4, 2]` and `B` of shape `[4, 3]` map `[2, 3]` to `[4, 3]`
        let linear = tensor::real::Linear::new(a, array(py, "jnp.zeros((4, 3))")).unwrap();
        assert_eq!(linear.dom(), tensor::real::ReLU::new([2, 3]).dom());
        assert_eq!(linear.cod(), tensor::real::ReLU::new([4, 3]).cod());

        // and the operations of every theory, as `Any<jax::Array>`
        let ints = array(py, "jnp.zeros((2, 3), dtype=jnp.int32)");
        let constant = tensor::Any::Int(tensor::int::NIA::Const(tensor::int::Const::new(ints)));
        let int: sort::Any = sort::Int::new([2, 3]).into();
        assert_eq!(constant.cod(), Poly::from(base::mono![int]));
    });
}

#[test]
fn an_array_is_held_with_its_shape_and_kind() {
    if !jax() {
        return;
    }
    Python::attach(|py| {
        let floats = array(py, "jnp.array([[1.0, 2.0]])");
        assert!(floats.is_floating_point() && !floats.is_integer() && !floats.is_bool());
        assert_eq!(floats.dtype(), "float32");
        assert_eq!(floats.to_string(), "[[1. 2.]]");

        let ints = array(py, "jnp.zeros((1, 1), dtype=jnp.uint8)");
        assert!(ints.is_integer() && !ints.is_floating_point());
        let bools = array(py, "jnp.zeros((1, 1), dtype=bool)");
        assert!(bools.is_bool() && !bools.is_integer());

        // the array held is the Python object itself
        let ob = eval(py, "jnp.eye(2)");
        assert!(Array::new(&ob).unwrap().bind(py).is(&ob));

        let err = Array::new(&eval(py, "jnp.zeros(3)")).unwrap_err();
        assert!(err.is_instance_of::<PyValueError>(py));
        assert_eq!(
            err.value(py).to_string(),
            "expected a matrix, got an array of shape [3]"
        );

        // NumPy arrays are not JAX arrays
        let err = Array::new(&eval(py, "np.zeros((2, 2))")).unwrap_err();
        assert!(err.is_instance_of::<PyTypeError>(py));
    });
}

#[test]
fn constants_are_built_from_arrays_of_their_theory_only() {
    if !jax() {
        return;
    }
    Python::attach(|py| {
        let floats = eval(py, "jnp.zeros((2, 3))");
        let ints = eval(py, "jnp.zeros((2, 3), dtype=jnp.int32)");
        let bools = eval(py, "jnp.zeros((2, 3), dtype=bool)");

        let real = py.get_type::<real::Const>();
        let int = py.get_type::<int::Const>();
        let boolean = py.get_type::<bool::Const>();
        let word = py.get_type::<word::Const>();

        assert!(real.call1((&floats,)).is_ok());
        assert!(int.call1((&ints,)).is_ok());
        assert!(boolean.call1((&bools,)).is_ok());
        assert!(word.call1((&ints, 8)).is_ok());

        // integers are reals too, kept as given
        assert!(real.call1((&ints,)).is_ok());
        let err = real.call1((&bools,)).unwrap_err();
        assert_eq!(
            err.value(py).to_string(),
            "expected an array of floating-point or integer elements, got bool"
        );
        assert!(int.call1((&bools,)).is_err());
        assert!(boolean.call1((&floats,)).is_err());
        assert!(word.call1((&floats, 8)).is_err());

        let vector = eval(py, "jnp.zeros(3)");
        let err = real.call1((&vector,)).unwrap_err();
        assert!(err.is_instance_of::<PyValueError>(py));

        // only JAX arrays: the torch classes take torch tensors
        let numpy = eval(py, "np.zeros((2, 3))");
        let err = real.call1((&numpy,)).unwrap_err();
        assert!(err.is_instance_of::<PyTypeError>(py));
    });
}

#[test]
fn operations_hold_and_give_back_the_same_arrays() {
    if !jax() {
        return;
    }
    Python::attach(|py| {
        let floats = eval(py, "jnp.zeros((2, 3))");
        let constant = py.get_type::<real::Const>().call1((&floats,)).unwrap();
        assert!(constant.getattr("tensor").unwrap().is(&floats));

        let op = constant.extract::<tensor::real::Const<Array>>().unwrap();
        assert!(op.tensor().bind(py).is(&floats));
        assert_eq!(op.cod(), tensor::real::ReLU::new([2, 3]).cod());

        // the class decides the theory, and the operation converts back
        // into it, holding the same array
        let any = constant.extract::<tensor::Any<Array>>().unwrap();
        assert!(matches!(
            any,
            tensor::Any::Real(tensor::real::NRAT::Const(_))
        ));
        let back = any.into_pyobject(py).unwrap();
        assert!(back.is_instance_of::<real::Const>());
        assert!(back.getattr("tensor").unwrap().is(&floats));
    });
}

#[test]
fn linear_maps_are_built_from_arrays_of_their_theory_and_shapes() {
    if !jax() {
        return;
    }
    Python::attach(|py| {
        let a = eval(py, "jnp.ones((4, 2))");
        let b = eval(py, "jnp.ones((4, 1))");
        let real = py.get_type::<real::Linear>();
        let linear = real.call1((&a, &b)).unwrap();
        let linear = linear.extract::<tensor::real::Linear<Array>>().unwrap();
        assert_eq!(linear.dom(), tensor::real::ReLU::new([2, 1]).dom());

        // `B` must have the rows of `A`
        let err = real.call1((&a, eval(py, "jnp.ones((3, 1))"))).unwrap_err();
        assert_eq!(
            err.value(py).to_string(),
            "expected A of shape [m, n] and B of shape [m, k], got [4, 2] and [3, 1]"
        );

        // and both must be matrices of the theory
        let ints = eval(py, "jnp.ones((4, 1), dtype=jnp.int32)");
        assert!(real.call1((&a, &ints)).is_err());
        let int = py.get_type::<int::Linear>();
        let ia = eval(py, "jnp.ones((4, 2), dtype=jnp.int32)");
        assert!(int.call1((&ia, &ints)).is_ok());
        assert!(int.call1((&a, &b)).is_err());

        // a real map of rates of change, of degree 1, which `B` is of
        let rates = real.call1((&a, &b, 1)).unwrap();
        assert_eq!(rates.getattr("degree").unwrap().extract::<u8>().unwrap(), 1);
        let rates = rates.extract::<tensor::real::Linear<Array>>().unwrap();
        assert_eq!(rates.degree(), 1);
        assert_eq!(
            rates.cod(),
            tensor::real::ReLU::with_degree([4, 1], 1).cod()
        );
    });
}

#[test]
fn scales_and_constants_hold_arrays_of_their_theory_and_a_degree() {
    if !jax() {
        return;
    }
    Python::attach(|py| {
        let floats = eval(py, "jnp.ones((2, 3))");
        let ints = eval(py, "jnp.ones((2, 3), dtype=jnp.int32)");

        // the product with a constant, of reals of a degree, which it keeps
        let scale = py.get_type::<real::Scale>();
        assert!(scale.call1((&ints,)).is_err());
        let rates = scale.call1((&floats, 1)).unwrap();
        assert!(rates.getattr("tensor").unwrap().is(&floats));
        let any = rates.extract::<tensor::Any<Array>>().unwrap();
        let tensor::Any::Real(tensor::real::NRAT::Scale(op)) = &any else {
            panic!("a real scale extracts as one, got {any}");
        };
        assert_eq!(op.degree(), 1);
        assert_eq!(op.dom(), tensor::real::ReLU::with_degree([2, 3], 1).dom());
        let back = any.into_pyobject(py).unwrap();
        assert!(back.is_instance_of::<real::Scale>());
        assert_eq!(back.getattr("degree").unwrap().extract::<u8>().unwrap(), 1);

        // and of integers, which have no degree
        let scale = py.get_type::<int::Scale>();
        assert!(scale.call1((&floats,)).is_err());
        let any = scale
            .call1((&ints,))
            .unwrap()
            .extract::<tensor::Any<Array>>()
            .unwrap();
        assert!(matches!(any, tensor::Any::Int(tensor::int::NIA::Scale(_))));
        assert!(
            any.into_pyobject(py)
                .unwrap()
                .is_instance_of::<int::Scale>()
        );

        // a constant rate of change, as the `1` of `ẋ = 1`
        let constant = py.get_type::<real::Const>().call1((&floats, 1)).unwrap();
        let op = constant.extract::<tensor::real::Const<Array>>().unwrap();
        assert_eq!(op.cod(), tensor::real::ReLU::with_degree([2, 3], 1).cod());
    });
}
