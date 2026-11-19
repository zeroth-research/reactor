/*!
# The operations in Python

Each operation is a Python class, in a Python module per theory, as in
Rust: `reactor.op.real`, `reactor.op.int`, `reactor.op.bool` and
`reactor.op.word`. The classes of [`word`] are named after the variants of
[`word::BV`], as `SDiv` for [`word::signed::Div`].

An operation's constructor takes its fields in order, as
`real.Add(m, n)` or `word.Extract(i, j, w, m, n)`, with the names of the
operation's documentation, and `__match_args__` lists them, so that
`match` takes an operation apart as it is built, as in
`case real.Linear(a, b):`. Its fields are also read as attributes, as
`op.m`. Operations are immutable, and print as in Rust (`str`) or as their
constructor (`repr`); those carrying no data compare by value.

The operations carrying data, the constants and the affine maps, hold
[`Tensor`]s, which convert from and into their Python objects, such as a
`torch.Tensor`; their constructors check their tensors, as
[`real::Const::new`] does, raising a `TypeError` for elements of another
kind and a `ValueError` for tensors of another shape.

The enum of each theory, as [`real::NRA`], and [`Any`] convert from the
classes of their operations and into them; [`add_modules`] adds the modules
of the theories to a Python module.
*/

use super::Any;
use super::bool::{self, Prop};
use super::int::{self, NIA};
use super::real::{self, NRA};
use super::word::{self, BV};
use crate::Tensor;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// A field of an operation as its `repr` shows it.
trait Repr {
    fn repr(&self, py: Python<'_>) -> PyResult<String>;
}

impl Repr for usize {
    fn repr(&self, _: Python<'_>) -> PyResult<String> {
        Ok(self.to_string())
    }
}

impl Repr for u8 {
    fn repr(&self, _: Python<'_>) -> PyResult<String> {
        Ok(self.to_string())
    }
}

impl Repr for String {
    fn repr(&self, _: Python<'_>) -> PyResult<String> {
        Ok(format!("{self:?}"))
    }
}

/// A tensor as the `repr` of its Python object shows it, as
/// `tensor([[1.]])` for a `torch.Tensor`.
impl Repr for Tensor {
    fn repr(&self, py: Python<'_>) -> PyResult<String> {
        Ok(self.clone().into_pyobject(py)?.repr()?.to_string())
    }
}

/// The constructor of an operation from its fields: the struct itself, or,
/// for an operation that checks its fields, its `new`.
macro_rules! construct {
    ([$($i:tt $field:ident),*]) => {
        Ok(Self { $($i: $field),* })
    };
    ([$($i:tt $field:ident),*] $new:ident) => {
        Self::$new($($field),*).map_err(PyErr::from)
    };
}

/// The Python classes of the operations of a theory, from the variant and
/// the operation of each, with its fields: their index, the name of the
/// Python field, and the type; an operation that checks its fields is built
/// `via` its constructor. Also the conversions between the enum of the
/// theory and these classes, and the function `$add` that adds the classes
/// to a Python module.
macro_rules! theory {
    (
        $theory:ident, $enum:ident, $add:ident {
            $(
                $variant:ident = $op:ty { $($i:tt $field:ident: $type:ty),* }
                $(via $new:ident)?
            ),* $(,)?
        }
    ) => {
        $(
            #[pymethods]
            impl $op {
                #[new]
                fn py_new($($field: $type),*) -> PyResult<Self> {
                    construct!([$($i $field),*] $($new)?)
                }

                $(
                    #[getter]
                    fn $field(&self) -> $type {
                        self.$i.clone()
                    }
                )*

                #[classattr]
                fn __match_args__(py: Python<'_>) -> PyResult<Bound<'_, PyTuple>> {
                    PyTuple::new(py, Vec::<&str>::from([$(stringify!($field)),*]))
                }

                #[allow(unused_variables)] // `py`, for an operation without fields
                fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
                    let fields: Vec<String> = vec![$(Repr::repr(&self.$i, py)?),*];
                    Ok(format!(
                        concat!(stringify!($theory), ".", stringify!($variant), "({})"),
                        fields.join(", ")
                    ))
                }
            }
        )*

        impl<'py> FromPyObject<'py> for $enum {
            fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
                $(
                    if let Ok(op) = obj.downcast::<$op>() {
                        return Ok($enum::$variant(op.get().clone()));
                    }
                )*
                Err(PyTypeError::new_err(format!(
                    concat!("expected an operation of ", stringify!($theory), ", got {}"),
                    obj.get_type()
                )))
            }
        }

        impl<'py> IntoPyObject<'py> for $enum {
            type Target = PyAny;
            type Output = Bound<'py, PyAny>;
            type Error = PyErr;

            fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
                match self {
                    $($enum::$variant(op) => Bound::new(py, op).map(Bound::into_any),)*
                }
            }
        }

        /// The classes of the operations of the theory, to add to a
        /// Python module.
        fn $add(module: &Bound<'_, PyModule>) -> PyResult<()> {
            $(module.add_class::<$op>()?;)*
            Ok(())
        }
    };
}

theory! {
    real, NRA, add_real {
        Const = real::Const { 0 tensor: Tensor } via new,
        Add = real::Add { 0 m: usize, 1 n: usize },
        Sub = real::Sub { 0 m: usize, 1 n: usize },
        Mul = real::Mul { 0 m: usize, 1 n: usize },
        Div = real::Div { 0 m: usize, 1 n: usize },
        Linear = real::Linear { 0 a: Tensor, 1 b: Tensor } via new,
        ReLU = real::ReLU { 0 m: usize, 1 n: usize },
        Min = real::Min { 0 m: usize, 1 n: usize },
        Max = real::Max { 0 m: usize, 1 n: usize },
        Transpose = real::Transpose { 0 m: usize, 1 n: usize },
        MatMul = real::MatMul { 0 m: usize, 1 n: usize, 2 k: usize },
        Lt = real::Lt { 0 m: usize, 1 n: usize },
        Le = real::Le { 0 m: usize, 1 n: usize },
        Gt = real::Gt { 0 m: usize, 1 n: usize },
        Ge = real::Ge { 0 m: usize, 1 n: usize },
        Eq = real::Eq { 0 m: usize, 1 n: usize },
        Ne = real::Ne { 0 m: usize, 1 n: usize },
        Id = real::Id { 0 m: usize, 1 n: usize },
        Ite = real::Ite { 0 m: usize, 1 n: usize },
        Argmax = real::Argmax { 0 m: usize, 1 n: usize },
        Havoc = real::Havoc { 0 m: usize, 1 n: usize },
        Uninterpreted = real::Uninterpreted { 0 symbol: String, 1 m: usize, 2 n: usize },
        Zero = real::Zero { 0 m: usize, 1 n: usize, 2 d: u8 },
        TAdd = real::TAdd { 0 m: usize, 1 n: usize, 2 d: u8 },
        TSub = real::TSub { 0 m: usize, 1 n: usize, 2 d: u8 },
        TLinear = real::TLinear { 0 a: Tensor, 1 b: Tensor, 2 d: u8 } via new,
        TTranspose = real::TTranspose { 0 m: usize, 1 n: usize, 2 d: u8 },
        TIte = real::TIte { 0 m: usize, 1 n: usize, 2 d: u8 },
        TId = real::TId { 0 m: usize, 1 n: usize, 2 d: u8 },
    }
}

theory! {
    int, NIA, add_int {
        Const = int::Const { 0 tensor: Tensor } via new,
        Add = int::Add { 0 m: usize, 1 n: usize },
        Sub = int::Sub { 0 m: usize, 1 n: usize },
        Mul = int::Mul { 0 m: usize, 1 n: usize },
        Div = int::Div { 0 m: usize, 1 n: usize },
        Mod = int::Mod { 0 m: usize, 1 n: usize },
        Linear = int::Linear { 0 a: Tensor, 1 b: Tensor } via new,
        ReLU = int::ReLU { 0 m: usize, 1 n: usize },
        Min = int::Min { 0 m: usize, 1 n: usize },
        Max = int::Max { 0 m: usize, 1 n: usize },
        Transpose = int::Transpose { 0 m: usize, 1 n: usize },
        MatMul = int::MatMul { 0 m: usize, 1 n: usize, 2 k: usize },
        Lt = int::Lt { 0 m: usize, 1 n: usize },
        Le = int::Le { 0 m: usize, 1 n: usize },
        Gt = int::Gt { 0 m: usize, 1 n: usize },
        Ge = int::Ge { 0 m: usize, 1 n: usize },
        Eq = int::Eq { 0 m: usize, 1 n: usize },
        Ne = int::Ne { 0 m: usize, 1 n: usize },
        Id = int::Id { 0 m: usize, 1 n: usize },
        Ite = int::Ite { 0 m: usize, 1 n: usize },
        Argmax = int::Argmax { 0 m: usize, 1 n: usize },
        Havoc = int::Havoc { 0 m: usize, 1 n: usize },
        Uninterpreted = int::Uninterpreted { 0 symbol: String, 1 m: usize, 2 n: usize },
    }
}

theory! {
    bool, Prop, add_bool {
        Const = bool::Const { 0 tensor: Tensor } via new,
        And = bool::And { 0 m: usize, 1 n: usize },
        Or = bool::Or { 0 m: usize, 1 n: usize },
        Xor = bool::Xor { 0 m: usize, 1 n: usize },
        Not = bool::Not { 0 m: usize, 1 n: usize },
        Id = bool::Id { 0 m: usize, 1 n: usize },
        Ite = bool::Ite { 0 m: usize, 1 n: usize },
        Havoc = bool::Havoc { 0 m: usize, 1 n: usize },
        Uninterpreted = bool::Uninterpreted { 0 symbol: String, 1 m: usize, 2 n: usize },
        Zero = bool::Zero {  },
    }
}

theory! {
    word, BV, add_word {
        Const = word::Const { 0 tensor: Tensor, 1 width: usize } via new,
        Concat = word::Concat { 0 u: usize, 1 v: usize, 2 m: usize, 3 n: usize },
        Extract = word::Extract { 0 i: usize, 1 j: usize, 2 w: usize, 3 m: usize, 4 n: usize },
        Repeat = word::Repeat { 0 i: usize, 1 w: usize, 2 m: usize, 3 n: usize },
        Not = word::Not { 0 w: usize, 1 m: usize, 2 n: usize },
        And = word::And { 0 w: usize, 1 m: usize, 2 n: usize },
        Or = word::Or { 0 w: usize, 1 m: usize, 2 n: usize },
        Xor = word::Xor { 0 w: usize, 1 m: usize, 2 n: usize },
        Nand = word::Nand { 0 w: usize, 1 m: usize, 2 n: usize },
        Nor = word::Nor { 0 w: usize, 1 m: usize, 2 n: usize },
        Xnor = word::Xnor { 0 w: usize, 1 m: usize, 2 n: usize },
        Neg = word::Neg { 0 w: usize, 1 m: usize, 2 n: usize },
        Add = word::Add { 0 w: usize, 1 m: usize, 2 n: usize },
        Sub = word::Sub { 0 w: usize, 1 m: usize, 2 n: usize },
        Mul = word::Mul { 0 w: usize, 1 m: usize, 2 n: usize },
        Shl = word::Shl { 0 w: usize, 1 m: usize, 2 n: usize },
        RotateLeft = word::RotateLeft { 0 i: usize, 1 w: usize, 2 m: usize, 3 n: usize },
        RotateRight = word::RotateRight { 0 i: usize, 1 w: usize, 2 m: usize, 3 n: usize },
        Eq = word::Eq { 0 w: usize, 1 m: usize, 2 n: usize },
        Ne = word::Ne { 0 w: usize, 1 m: usize, 2 n: usize },
        SDiv = word::signed::Div { 0 w: usize, 1 m: usize, 2 n: usize },
        SRem = word::signed::Rem { 0 w: usize, 1 m: usize, 2 n: usize },
        SMod = word::signed::Mod { 0 w: usize, 1 m: usize, 2 n: usize },
        AShr = word::signed::Shr { 0 w: usize, 1 m: usize, 2 n: usize },
        SignExtend = word::signed::Extend { 0 i: usize, 1 w: usize, 2 m: usize, 3 n: usize },
        SLt = word::signed::Lt { 0 w: usize, 1 m: usize, 2 n: usize },
        SLe = word::signed::Le { 0 w: usize, 1 m: usize, 2 n: usize },
        SGt = word::signed::Gt { 0 w: usize, 1 m: usize, 2 n: usize },
        SGe = word::signed::Ge { 0 w: usize, 1 m: usize, 2 n: usize },
        UDiv = word::unsigned::Div { 0 w: usize, 1 m: usize, 2 n: usize },
        URem = word::unsigned::Rem { 0 w: usize, 1 m: usize, 2 n: usize },
        LShr = word::unsigned::Shr { 0 w: usize, 1 m: usize, 2 n: usize },
        ZeroExtend = word::unsigned::Extend { 0 i: usize, 1 w: usize, 2 m: usize, 3 n: usize },
        ULt = word::unsigned::Lt { 0 w: usize, 1 m: usize, 2 n: usize },
        ULe = word::unsigned::Le { 0 w: usize, 1 m: usize, 2 n: usize },
        UGt = word::unsigned::Gt { 0 w: usize, 1 m: usize, 2 n: usize },
        UGe = word::unsigned::Ge { 0 w: usize, 1 m: usize, 2 n: usize },
        Id = word::Id { 0 w: usize, 1 m: usize, 2 n: usize },
        Ite = word::Ite { 0 w: usize, 1 m: usize, 2 n: usize },
        Havoc = word::Havoc { 0 w: usize, 1 m: usize, 2 n: usize },
        Uninterpreted = word::Uninterpreted { 0 symbol: String, 1 w: usize, 2 m: usize, 3 n: usize },
    }
}

/// Adds the modules `real`, `int`, `bool` and `word` of the operations of
/// each theory to `module`, which Python is to know as `reactor.op`.
pub fn add_modules(module: &Bound<'_, PyModule>) -> PyResult<()> {
    type Add = fn(&Bound<'_, PyModule>) -> PyResult<()>;
    let theories: [(&str, Add); 4] = [
        ("real", add_real),
        ("int", add_int),
        ("bool", add_bool),
        ("word", add_word),
    ];
    for (name, add) in theories {
        let theory = PyModule::new(module.py(), &format!("reactor.op.{name}"))?;
        add(&theory)?;
        module.add(name, theory)?;
    }
    Ok(())
}

impl<'py> FromPyObject<'py> for Any {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(op) = obj.extract::<NRA>() {
            return Ok(Any::Real(op));
        }
        if let Ok(op) = obj.extract::<NIA>() {
            return Ok(Any::Int(op));
        }
        if let Ok(op) = obj.extract::<Prop>() {
            return Ok(Any::Bool(op));
        }
        if let Ok(op) = obj.extract::<BV>() {
            return Ok(Any::Word(op));
        }
        Err(PyTypeError::new_err(format!(
            "expected an operation of real, int, bool or word, got {}",
            obj.get_type()
        )))
    }
}

impl<'py> IntoPyObject<'py> for Any {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        match self {
            Any::Real(op) => op.into_pyobject(py),
            Any::Int(op) => op.into_pyobject(py),
            Any::Bool(op) => op.into_pyobject(py),
            Any::Word(op) => op.into_pyobject(py),
        }
    }
}

#[cfg(all(test, feature = "torch"))]
mod torch_tests {
    use super::*;
    use crate::tensor::TorchTensor;
    use base::boundary::Boundary;
    use pyo3_tch::tch;

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
        TorchTensor::new(tensor).unwrap().into_pyobject(py).unwrap()
    }

    /// The storage of a torch tensor.
    fn data_ptr(tensor: &Tensor) -> *mut std::ffi::c_void {
        let Tensor::Torch(t) = tensor else {
            panic!("expected a torch tensor, got {tensor:?}")
        };
        t.tensor().data_ptr()
    }

    #[test]
    fn torch_tensors_back_the_operations() {
        let a = TorchTensor::new(full(4, 2, 0.0, tch::Kind::Float)).unwrap();
        let b = TorchTensor::new(full(4, 3, 0.0, tch::Kind::Float)).unwrap();
        assert_eq!(a.shape(), [4, 2]);
        // `A` of shape `[4, 2]` and `B` of shape `[4, 3]` map `[2, 3]` to `[4, 3]`
        let linear = real::Linear::new(a.into(), b.into()).unwrap();
        assert_eq!(linear.dom(), real::ReLU(2, 3).dom());
        assert_eq!(linear.cod(), real::ReLU(4, 3).cod());
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
            let op = constant.extract::<real::Const>().unwrap();
            assert_eq!(op.cod(), real::ReLU(2, 3).cod());

            // the tensor is that of the constant
            let t = floats.extract::<Tensor>().unwrap();
            assert_eq!(data_ptr(&op.0), data_ptr(&t));

            // the class decides the theory, not the tensor
            assert!(constant.extract::<int::Const>().is_err());
            assert!(floats.extract::<real::Const>().is_err());

            let ints = python(py, full(2, 3, 0.0, tch::Kind::Int64));
            let constant = py.get_type::<word::Const>().call1((&ints, 8)).unwrap();
            let op = constant.extract::<word::Const>().unwrap();
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
    fn rate_linear_maps_are_built_as_linear_maps() {
        prepare_python_with_torch();
        Python::with_gil(|py| {
            let a = python(py, full(4, 2, 1.0, tch::Kind::Float));
            let b = python(py, full(4, 3, 1.0, tch::Kind::Float));
            let class = py.get_type::<real::TLinear>();
            let op = class.call1((&a, &b, 1)).unwrap();
            let op = op.extract::<real::TLinear>().unwrap();
            assert_eq!(op.dom(), real::TAdd(2, 3, 1).cod());
            assert_eq!(op.cod(), real::TAdd(4, 3, 1).cod());

            let wrong_rows = python(py, full(3, 3, 1.0, tch::Kind::Float));
            assert!(class.call1((&a, &wrong_rows, 1)).is_err());
            let ints = python(py, full(4, 3, 1.0, tch::Kind::Int64));
            assert!(class.call1((&a, &ints, 1)).is_err());
        });
    }

    #[test]
    fn any_converts_from_and_into_the_classes_sharing_tensors() {
        prepare_python_with_torch();
        Python::with_gil(|py| {
            let a = python(py, full(4, 2, 1.0, tch::Kind::Int64));
            let b = python(py, full(4, 1, 1.0, tch::Kind::Int64));
            let linear = py.get_type::<int::Linear>().call1((&a, &b)).unwrap();
            let any = linear.extract::<Any>().unwrap();
            let Any::Int(NIA::Linear(op)) = &any else {
                panic!("expected an integer linear map, got {any:?}")
            };
            assert_eq!(data_ptr(&op.0), data_ptr(&a.extract().unwrap()));

            let back = any.into_pyobject(py).unwrap();
            let op = back.downcast::<int::Linear>().unwrap().get();
            assert_eq!(data_ptr(&op.0), data_ptr(&a.extract().unwrap()));
        });
    }
}
