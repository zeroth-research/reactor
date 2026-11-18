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
constructor (`repr`).

The enum of each theory, [`real::NRA`], [`int::NIA`], [`bool::Prop`] and
[`word::BV`], and [`Any`], convert from the classes of its operations and
into them; [`add_modules`] adds the modules of the theories to a Python
module.
*/

use super::Any;
use super::bool::{self, Prop};
use super::int::{self, NIA};
use super::real::{self, NRA};
use super::word::{self, BV};
use crate::PyTensor;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// A field of an operation as its `repr` shows it.
trait Repr {
    fn repr(&self) -> String;
}

impl Repr for usize {
    fn repr(&self) -> String {
        self.to_string()
    }
}

impl Repr for u8 {
    fn repr(&self) -> String {
        self.to_string()
    }
}

impl Repr for String {
    fn repr(&self) -> String {
        format!("{self:?}")
    }
}

impl Repr for PyTensor {
    fn repr(&self) -> String {
        self.to_string()
    }
}

/// The Python classes of the operations of a theory, from the variant and
/// the operation of each, with its fields: their index, the name of the
/// Python field, and the type. Also the conversions between the enum of
/// the theory and these classes, and the function `$add` that adds the
/// classes to a Python module.
macro_rules! theory {
    (
        $theory:ident, $enum:ident, $add:ident {
            $($variant:ident = $op:ty { $($i:tt $field:ident: $type:ty),* }),* $(,)?
        }
    ) => {
        $(
            #[pymethods]
            impl $op {
                #[new]
                fn py_new($($field: $type),*) -> Self {
                    Self { $($i: $field),* }
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

                fn __repr__(&self) -> String {
                    let fields: Vec<String> = vec![$(Repr::repr(&self.$i)),*];
                    format!(
                        concat!(stringify!($theory), ".", stringify!($variant), "({})"),
                        fields.join(", ")
                    )
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
        Const = real::Const { 0 tensor: PyTensor },
        Add = real::Add { 0 m: usize, 1 n: usize },
        Sub = real::Sub { 0 m: usize, 1 n: usize },
        Mul = real::Mul { 0 m: usize, 1 n: usize },
        Div = real::Div { 0 m: usize, 1 n: usize },
        Linear = real::Linear { 0 a: PyTensor, 1 b: PyTensor },
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
        TLinear = real::TLinear { 0 a: PyTensor, 1 b: PyTensor, 2 d: u8 },
        TTranspose = real::TTranspose { 0 m: usize, 1 n: usize, 2 d: u8 },
        TIte = real::TIte { 0 m: usize, 1 n: usize, 2 d: u8 },
        TId = real::TId { 0 m: usize, 1 n: usize, 2 d: u8 },
    }
}

theory! {
    int, NIA, add_int {
        Const = int::Const { 0 tensor: PyTensor },
        Add = int::Add { 0 m: usize, 1 n: usize },
        Sub = int::Sub { 0 m: usize, 1 n: usize },
        Mul = int::Mul { 0 m: usize, 1 n: usize },
        Div = int::Div { 0 m: usize, 1 n: usize },
        Mod = int::Mod { 0 m: usize, 1 n: usize },
        Linear = int::Linear { 0 a: PyTensor, 1 b: PyTensor },
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
        Const = bool::Const { 0 tensor: PyTensor },
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
        Const = word::Const { 0 tensor: PyTensor, 1 w: usize },
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
