//! Tests for the mixed backing of `tensor`, of torch tensors and JAX arrays
//! alike, and its Python classes. Its JAX half needs JAX, as the `jax`
//! dependency group of the Python package installs it, and is not tested
//! when it is missing, as the Python tests of JAX skip.

#![cfg(all(feature = "torch", feature = "jax"))]

use base::boundary::Boundary;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use pyo3_tch::PyTensor;
use std::ffi::CString;
use theory::mixed::{Tensor, bool, int, real, word};
use theory::tensor;

/// Starts Python and imports `torch`, then `jax.numpy`, once for all tests,
/// before any of them holds the GIL, as importing from several threads at
/// once may deadlock. Whether JAX is installed.
fn jax() -> bool {
    static JAX: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *JAX.get_or_init(|| {
        Python::initialize();
        Python::attach(|py| {
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

/// The value of the Python expression `expr`, in which `torch` is `torch`,
/// `jnp` is `jax.numpy` and `np` is `numpy`.
fn eval<'py>(py: Python<'py>, expr: &str) -> Bound<'py, PyAny> {
    let modules = [
        ("torch", py.import("torch").unwrap()),
        ("jnp", py.import("jax.numpy").unwrap()),
        ("np", py.import("numpy").unwrap()),
    ];
    let locals = modules.into_py_dict(py).unwrap();
    let expr = CString::new(expr).unwrap();
    py.eval(&expr, None, Some(&locals)).unwrap()
}

#[test]
fn tensors_extract_from_either_backing_as_matrices() {
    if !jax() {
        return;
    }
    Python::attach(|py| {
        // a torch tensor shares its storage
        let ob = eval(py, "torch.zeros(2, 3)");
        let Tensor::Torch(t) = ob.extract::<Tensor>().unwrap() else {
            panic!("expected a torch tensor");
        };
        let shared = ob.extract::<PyTensor>().unwrap();
        assert_eq!(t.data_ptr(), shared.data_ptr());
        assert_eq!(theory::Tensor::shape(&t), [2, 3]);

        // a JAX array is held itself
        let ob = eval(py, "jnp.zeros((2, 3))");
        let Tensor::Jax(a) = ob.extract::<Tensor>().unwrap() else {
            panic!("expected a JAX array");
        };
        assert!(a.bind(py).is(&ob));

        for vector in ["torch.zeros(3)", "jnp.zeros(3)"] {
            let err = eval(py, vector).extract::<Tensor>().unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py), "{vector}");
        }
        let err = eval(py, "np.zeros((2, 3))")
            .extract::<Tensor>()
            .unwrap_err();
        assert!(err.is_instance_of::<PyTypeError>(py));
        assert!(err.to_string().contains("a torch.Tensor or a jax.Array"));
    });
}

#[test]
fn constants_are_built_from_tensors_of_their_theory_only() {
    if !jax() {
        return;
    }
    Python::attach(|py| {
        let real = py.get_type::<real::Const>();
        let int = py.get_type::<int::Const>();
        let boolean = py.get_type::<bool::Const>();
        let word = py.get_type::<word::Const>();

        for (floats, ints, bools) in [
            (
                "torch.zeros(2, 3)",
                "torch.zeros(2, 3, dtype=torch.int64)",
                "torch.zeros(2, 3, dtype=torch.bool)",
            ),
            (
                "jnp.zeros((2, 3))",
                "jnp.zeros((2, 3), dtype=jnp.int32)",
                "jnp.zeros((2, 3), dtype=bool)",
            ),
        ] {
            let (floats, ints, bools) = (eval(py, floats), eval(py, ints), eval(py, bools));
            assert!(real.call1((&floats,)).is_ok());
            assert!(int.call1((&ints,)).is_ok());
            assert!(boolean.call1((&bools,)).is_ok());
            assert!(word.call1((&ints, 8)).is_ok());
            // integers are reals too, kept as given, but not booleans
            assert!(real.call1((&ints,)).is_ok());
            assert!(real.call1((&bools,)).is_err());
            assert!(int.call1((&bools,)).is_err());
            assert!(boolean.call1((&floats,)).is_err());
            assert!(word.call1((&floats, 8)).is_err());
        }

        // each backing names the kinds of elements in its own terms
        let scale = py.get_type::<real::Scale>();
        let err = scale.call1((eval(py, "jnp.zeros((1, 1), dtype=jnp.int32)"),));
        assert_eq!(
            err.unwrap_err().value(py).to_string(),
            "expected a tensor of floating-point elements, got int32"
        );
        let err = scale.call1((eval(py, "torch.zeros(1, 1, dtype=torch.int64)"),));
        assert_eq!(
            err.unwrap_err().value(py).to_string(),
            "expected a tensor of floating-point elements, got Int64"
        );
    });
}

/// The products with a constant hold either tensor, the real ones with a
/// degree, which they keep, and convert back into their class.
#[test]
fn scales_hold_either_tensor_and_reals_a_degree() {
    if !jax() {
        return;
    }
    Python::attach(|py| {
        for (floats, ints) in [
            ("torch.ones(2, 3)", "torch.ones(2, 3, dtype=torch.int64)"),
            ("jnp.ones((2, 3))", "jnp.ones((2, 3), dtype=jnp.int32)"),
        ] {
            let (floats, ints) = (eval(py, floats), eval(py, ints));

            let rates = py.get_type::<real::Scale>().call1((&floats, 1)).unwrap();
            let any = rates.extract::<tensor::Any<Tensor>>().unwrap();
            let tensor::Any::Real(tensor::real::NRAT::Scale(op)) = &any else {
                panic!("a real scale extracts as one, got {any}");
            };
            assert_eq!(op.dom(), tensor::real::ReLU::with_degree([2, 3], 1).dom());
            let back = any.into_pyobject(py).unwrap();
            assert!(back.is_instance_of::<real::Scale>());
            assert!(back.getattr("tensor").unwrap().is(&floats));
            assert_eq!(back.getattr("degree").unwrap().extract::<u8>().unwrap(), 1);

            let scale = py.get_type::<int::Scale>();
            assert!(scale.call1((&floats,)).is_err());
            let any = scale
                .call1((&ints,))
                .unwrap()
                .extract::<tensor::Any<Tensor>>()
                .unwrap();
            assert!(matches!(any, tensor::Any::Int(tensor::int::NIA::Scale(_))));
            assert!(
                any.into_pyobject(py)
                    .unwrap()
                    .is_instance_of::<int::Scale>()
            );
        }
    });
}

/// The operations of one module may hold torch tensors and JAX arrays
/// alike, even within one affine map.
#[test]
fn an_affine_map_holds_a_jax_array_and_a_torch_tensor() {
    if !jax() {
        return;
    }
    Python::attach(|py| {
        let a = eval(py, "jnp.eye(2)");
        let b = eval(py, "torch.ones(2, 1)");
        let class = py.get_type::<real::Linear>().call1((&a, &b)).unwrap();
        let linear = class.extract::<tensor::real::Linear<Tensor>>().unwrap();
        assert!(matches!(linear.a(), Tensor::Jax(_)));
        assert!(matches!(linear.b(), Tensor::Torch(_)));
        assert_eq!(linear.dom(), tensor::real::ReLU::new([2, 1]).dom());
        assert_eq!(linear.cod(), tensor::real::ReLU::new([2, 1]).cod());

        // `B` must have the rows of `A`, whatever their backings
        let wrong_rows = eval(py, "torch.ones(3, 1)");
        let err = py
            .get_type::<real::Linear>()
            .call1((&a, &wrong_rows))
            .unwrap_err();
        assert_eq!(
            err.value(py).to_string(),
            "expected A of shape [m, n] and B of shape [m, k], got [2, 2] and [3, 1]"
        );

        // the operation converts back into its class, holding the same array
        let any = class.extract::<tensor::Any<Tensor>>().unwrap();
        let back = any.into_pyobject(py).unwrap();
        assert!(back.is_instance_of::<real::Linear>());
        assert!(back.getattr("a").unwrap().is(&a));
    });
}
