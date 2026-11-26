use pyo3::PyClass;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::borrow::Borrow;
use std::collections::HashSet;
use std::fmt::Debug;
use std::fmt::Write;

mod atom;
mod module;
mod term;
mod var;
mod wire;

use crate::atom::Atom;
use crate::module::Module;
use crate::term::Term;
use crate::var::Var;
use crate::wire::Wire;
use theory::matrix;

/// The operations of all theories, with their tensors held as `tch::Tensor`s.
pub(crate) type Op = theory::matrix::Op<tch::Tensor>;

#[pymodule]
fn reactor(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Wire>()?;
    m.add_class::<Var>()?;
    m.add_class::<Atom>()?;
    m.add_class::<Term>()?;
    m.add_class::<Module>()?;

    m.add_function(wrap_pyfunction!(var::x, m)?)?;
    m.add_function(wrap_pyfunction!(var::d, m)?)?;
    m.add_function(wrap_pyfunction!(tangent, m)?)?;

    // The sorts and operations, in namespaces mirroring the `theory` crate:
    // `zero` holds the singleton sort `Real0`, shared by all theories,
    // and the operation `Zero`, writing its only value; `matrix` holds the
    // structural operations, with a namespace per theory holding its sort and
    // its operations. The operations carrying tensors come from
    // `theory::torch`, the others from `theory::matrix`.
    let mut namespaces = Vec::new();
    {
        let zero = PyModule::new(py, "zero")?;
        zero.add_class::<theory::zero::Real0>()?;
        zero.add_class::<theory::zero::Zero>()?;
        m.add("zero", &zero)?;
        namespaces.push(("zero", zero));
    }
    {
        let matrix = PyModule::new(py, "matrix")?;
        matrix.add_class::<theory::matrix::Havoc>()?;
        matrix.add_class::<theory::matrix::Skip>()?;
        {
            let real = PyModule::new(py, "real")?;
            real.add_class::<theory::matrix::real::Real>()?;
            real.add_class::<theory::torch::real::Num>()?;
            real.add_class::<theory::torch::real::Scale>()?;
            real.add_class::<theory::torch::real::Linear>()?;
            real.add_class::<theory::torch::real::Affine>()?;
            real.add_class::<theory::matrix::real::Add>()?;
            real.add_class::<theory::matrix::real::Sub>()?;
            real.add_class::<theory::matrix::real::Mul>()?;
            real.add_class::<theory::matrix::real::Div>()?;
            real.add_class::<theory::matrix::real::ReLU>()?;
            real.add_class::<theory::matrix::real::Abs>()?;
            real.add_class::<theory::matrix::real::Neg>()?;
            real.add_class::<theory::matrix::real::Zero>()?;
            real.add_class::<theory::matrix::real::Argmax>()?;
            real.add_class::<theory::matrix::real::Min>()?;
            real.add_class::<theory::matrix::real::Max>()?;
            real.add_class::<theory::matrix::real::Transpose>()?;
            real.add_class::<theory::matrix::real::MatMul>()?;
            real.add_class::<theory::matrix::real::Lt>()?;
            real.add_class::<theory::matrix::real::Ite>()?;
            real.add_class::<theory::matrix::real::Le>()?;
            real.add_class::<theory::matrix::real::Gt>()?;
            real.add_class::<theory::matrix::real::Ge>()?;
            real.add_class::<theory::matrix::real::Eq>()?;
            real.add_class::<theory::matrix::real::Ne>()?;
            {
                let broadcast = PyModule::new(py, "broadcast")?;
                broadcast.add_class::<theory::matrix::real::broadcast::Ite>()?;
                real.add("broadcast", &broadcast)?;
                namespaces.push(("matrix.real.broadcast", broadcast));
            }
            matrix.add("real", &real)?;
            namespaces.push(("matrix.real", real));
        }
        {
            let int = PyModule::new(py, "int")?;
            int.add_class::<theory::matrix::int::Int>()?;
            int.add_class::<theory::torch::int::Num>()?;
            int.add_class::<theory::torch::int::Scale>()?;
            int.add_class::<theory::torch::int::Linear>()?;
            int.add_class::<theory::torch::int::Affine>()?;
            int.add_class::<theory::matrix::int::Add>()?;
            int.add_class::<theory::matrix::int::Sub>()?;
            int.add_class::<theory::matrix::int::Mul>()?;
            int.add_class::<theory::matrix::int::Div>()?;
            int.add_class::<theory::matrix::int::Mod>()?;
            int.add_class::<theory::matrix::int::ReLU>()?;
            int.add_class::<theory::matrix::int::Neg>()?;
            int.add_class::<theory::matrix::int::Abs>()?;
            int.add_class::<theory::matrix::int::Argmax>()?;
            int.add_class::<theory::matrix::int::Min>()?;
            int.add_class::<theory::matrix::int::Max>()?;
            int.add_class::<theory::matrix::int::Transpose>()?;
            int.add_class::<theory::matrix::int::MatMul>()?;
            int.add_class::<theory::matrix::int::Lt>()?;
            int.add_class::<theory::matrix::int::Ite>()?;
            int.add_class::<theory::matrix::int::Le>()?;
            int.add_class::<theory::matrix::int::Gt>()?;
            int.add_class::<theory::matrix::int::Ge>()?;
            int.add_class::<theory::matrix::int::Eq>()?;
            int.add_class::<theory::matrix::int::Ne>()?;
            {
                let broadcast = PyModule::new(py, "broadcast")?;
                broadcast.add_class::<theory::matrix::int::broadcast::Ite>()?;
                int.add("broadcast", &broadcast)?;
                namespaces.push(("matrix.int.broadcast", broadcast));
            }
            matrix.add("int", &int)?;
            namespaces.push(("matrix.int", int));
        }
        {
            let bool = PyModule::new(py, "bool")?;
            bool.add_class::<theory::matrix::bool::Bool>()?;
            bool.add_class::<theory::torch::bool::Truth>()?;
            bool.add_class::<theory::matrix::bool::And>()?;
            bool.add_class::<theory::matrix::bool::Or>()?;
            bool.add_class::<theory::matrix::bool::Xor>()?;
            bool.add_class::<theory::matrix::bool::Not>()?;
            bool.add_class::<theory::matrix::bool::Eq>()?;
            bool.add_class::<theory::matrix::bool::Ne>()?;
            bool.add_class::<theory::matrix::bool::Ite>()?;
            {
                let broadcast = PyModule::new(py, "broadcast")?;
                broadcast.add_class::<theory::matrix::bool::broadcast::Ite>()?;
                bool.add("broadcast", &broadcast)?;
                namespaces.push(("matrix.bool.broadcast", broadcast));
            }
            matrix.add("bool", &bool)?;
            namespaces.push(("matrix.bool", bool));
        }
        {
            let word = PyModule::new(py, "word")?;
            word.add_class::<theory::matrix::word::Word>()?;
            word.add_class::<theory::torch::word::Num>()?;
            word.add_class::<theory::matrix::word::Concat>()?;
            word.add_class::<theory::matrix::word::Extract>()?;
            word.add_class::<theory::matrix::word::Repeat>()?;
            word.add_class::<theory::matrix::word::Not>()?;
            word.add_class::<theory::matrix::word::And>()?;
            word.add_class::<theory::matrix::word::Or>()?;
            word.add_class::<theory::matrix::word::Xor>()?;
            word.add_class::<theory::matrix::word::Nand>()?;
            word.add_class::<theory::matrix::word::Nor>()?;
            word.add_class::<theory::matrix::word::Xnor>()?;
            word.add_class::<theory::matrix::word::Neg>()?;
            word.add_class::<theory::matrix::word::Ite>()?;
            word.add_class::<theory::matrix::word::Add>()?;
            word.add_class::<theory::matrix::word::Sub>()?;
            word.add_class::<theory::matrix::word::Mul>()?;
            word.add_class::<theory::matrix::word::Shl>()?;
            word.add_class::<theory::matrix::word::RotateLeft>()?;
            word.add_class::<theory::matrix::word::RotateRight>()?;
            word.add_class::<theory::matrix::word::Eq>()?;
            word.add_class::<theory::matrix::word::Ne>()?;
            {
                let signed = PyModule::new(py, "signed")?;
                signed.add_class::<theory::matrix::word::signed::Div>()?;
                signed.add_class::<theory::matrix::word::signed::Abs>()?;
                signed.add_class::<theory::matrix::word::signed::Rem>()?;
                signed.add_class::<theory::matrix::word::signed::Mod>()?;
                signed.add_class::<theory::matrix::word::signed::Shr>()?;
                signed.add_class::<theory::matrix::word::signed::Extend>()?;
                signed.add_class::<theory::matrix::word::signed::Lt>()?;
                signed.add_class::<theory::matrix::word::signed::Le>()?;
                signed.add_class::<theory::matrix::word::signed::Gt>()?;
                signed.add_class::<theory::matrix::word::signed::Ge>()?;
                word.add("signed", &signed)?;
                namespaces.push(("matrix.word.signed", signed));
            }
            {
                let unsigned = PyModule::new(py, "unsigned")?;
                unsigned.add_class::<theory::matrix::word::unsigned::Div>()?;
                unsigned.add_class::<theory::matrix::word::unsigned::Rem>()?;
                unsigned.add_class::<theory::matrix::word::unsigned::Shr>()?;
                unsigned.add_class::<theory::matrix::word::unsigned::Extend>()?;
                unsigned.add_class::<theory::matrix::word::unsigned::Lt>()?;
                unsigned.add_class::<theory::matrix::word::unsigned::Le>()?;
                unsigned.add_class::<theory::matrix::word::unsigned::Gt>()?;
                unsigned.add_class::<theory::matrix::word::unsigned::Ge>()?;
                word.add("unsigned", &unsigned)?;
                namespaces.push(("matrix.word.unsigned", unsigned));
            }
            {
                let broadcast = PyModule::new(py, "broadcast")?;
                broadcast.add_class::<theory::matrix::word::broadcast::Ite>()?;
                word.add("broadcast", &broadcast)?;
                namespaces.push(("matrix.word.broadcast", broadcast));
            }
            matrix.add("word", &word)?;
            namespaces.push(("matrix.word", word));
        }
        m.add("matrix", &matrix)?;
        namespaces.push(("matrix", matrix));
    }

    // Each namespace takes its dotted name in the package that holds this
    // extension, `reactor.matrix.real` for the extension `reactor.reactor`,
    // the name its classes give as their module, and is entered in
    // `sys.modules` under it, so that it can be imported by it, as
    // `import reactor.matrix.real`, not only reached as an attribute.
    let name = m.name()?.to_string();
    let package = name.rsplit_once('.').map_or(name.as_str(), |(p, _)| p);
    let modules = py.import("sys")?.getattr("modules")?;
    for (name, namespace) in namespaces {
        let dotted = format!("{package}.{name}");
        namespace.setattr("__name__", &dotted)?;
        modules.set_item(&dotted, namespace)?;
    }

    Ok(())
}

/// The tangent sort `T(sort)`: the sort of the rate of change of the values
/// of `sort`, one degree higher for a real, and the singleton
/// `zero.Real0()` for a discrete sort.
#[pyfunction]
#[pyo3(name = "T")]
fn tangent(sort: matrix::Any) -> matrix::Any {
    base::differentiable::T(sort)
}

fn try_iter_borrow<'py, P>(
    iter: &'py Bound<'py, PyAny>,
) -> PyResult<impl Iterator<Item = PyResult<PyRef<'py, P>>>>
where
    P: PyClass,
{
    let iter = iter
        .try_iter()?
        .map(|i| i?.extract::<PyRef<P>>().map_err(PyErr::from));
    Ok(iter)
}

fn try_var_iter_cloned(
    seq: &Bound<'_, PyAny>,
) -> PyResult<impl Iterator<Item = reactive::Var<matrix::Any>>> {
    // TODO: make base take result iterator to avoid unwrap
    let seq = try_iter_borrow::<Var>(seq)?;
    let seq = seq.into_iter().map(Result::unwrap);
    let seq = seq.map(|r| *r.base());
    Ok(seq)
}

fn try_term_iter_cloned<T>(
    seq: &Bound<'_, PyAny>,
) -> PyResult<impl Iterator<Item = dataflow::diagram::Term<T, matrix::Any>>>
where
    dataflow::diagram::Term<Op, matrix::Any>: TryInto<dataflow::diagram::Term<T, matrix::Any>>,
    <dataflow::diagram::Term<Op, matrix::Any> as TryInto<
        dataflow::diagram::Term<T, matrix::Any>,
    >>::Error: Debug,
{
    // TODO: make base take result iterator to avoid unwrap
    let seq = try_iter_borrow::<Term>(seq)?;
    let seq = seq.into_iter().map(Result::unwrap);
    let seq = seq.map(|r| r.base().clone().try_into().unwrap());
    Ok(seq)
}

fn try_wire_iter_cloned(
    seq: &Bound<'_, PyAny>,
) -> PyResult<impl Iterator<Item = dataflow::diagram::Wire<matrix::Any>>> {
    // TODO: make base take result iterator to avoid unwrap
    let seq = seq.try_iter()?.map(|i| -> PyResult<_> {
        let i = i?;
        if let Ok(wire) = i.extract::<PyRef<Wire>>() {
            return Ok(wire.base().clone());
        }
        // a variable stands for its latched wire in a wire position
        if let Ok(var) = i.extract::<PyRef<Var>>() {
            return Ok(dataflow::diagram::Wire::from(*var.base()));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "expected a Wire or a Variable",
        ))
    });
    let seq = seq.map(Result::unwrap);
    Ok(seq)
}

/// The python-facing `hide` argument: either a callable predicate over
/// variables (e.g. a lambda), or any iterable of variables (preferably a
/// set); `None` hides nothing.
pub(crate) enum Hide<'py> {
    Set(HashSet<reactive::Var<matrix::Any>>),
    Call(
        &'py Bound<'py, PyAny>,
        // errors raised by the callable, surfaced after the base call
        std::cell::RefCell<Option<PyErr>>,
    ),
}

impl<'py> Hide<'py> {
    pub(crate) fn try_from(hide: &'py Bound<'py, PyAny>) -> PyResult<Self> {
        if hide.is_callable() {
            Ok(Hide::Call(hide, Default::default()))
        } else if let Ok(iter) = try_var_iter_cloned(hide) {
            Ok(Hide::Set(iter.collect()))
        } else {
            Err(PyException::new_err(
                "hide must be either callable or iterable",
            ))
        }
    }

    /// The predicate handed to the base constructors.
    pub(crate) fn as_fn(&self) -> impl Fn(&reactive::Var<matrix::Any>) -> bool + '_ {
        move |var| match self {
            Hide::Set(set) => set.contains(var),
            Hide::Call(hide, err) => {
                if err.borrow().is_some() {
                    return false;
                }
                match hide.call1((Var::from(*var),)).and_then(|r| r.is_truthy()) {
                    Ok(hidden) => hidden,
                    Err(e) => {
                        *err.borrow_mut() = Some(e);
                        false
                    }
                }
            }
        }
    }

    /// Propagates any error the callable raised inside the predicate; must
    /// be checked before trusting the base result.
    pub(crate) fn err(self) -> PyResult<()> {
        match self {
            Hide::Call(_, err) => err.into_inner().map_or(Ok(()), Err),
            Hide::Set(_) => Ok(()),
        }
    }
}

/// Renders wires untyped and comma-separated. Accepts references to wires
/// and to variables alike: a `Var` borrows as its latched `Wire`.
pub(crate) fn wires_untyped_to_repr<'a, S, W, I>(wires: I) -> String
where
    S: Debug + 'a,
    W: Borrow<dataflow::diagram::Wire<S>> + 'a,
    I: IntoIterator<Item = &'a W>,
{
    let mut s = String::from("[");
    for (i, wire) in wires.into_iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        write!(s, "{:?}", wire.borrow().typed()).unwrap(); // we know it's infallible
    }
    s.push_str("]");
    s
}
