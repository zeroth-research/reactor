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
use theory::Any;

/// The operations of all theories, with their tensors held as `tch::Tensor`s.
pub(crate) type Op = theory::Op<tch::Tensor>;

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

    // The structural operations, which belong to no theory, at the top
    // level, as in the `theory` crate.
    m.add_class::<theory::Havoc>()?;
    m.add_class::<theory::Skip>()?;

    // The sorts and the other operations, in namespaces mirroring the
    // `theory` crate: `zero` holds the singleton sort `Real0`, shared by all
    // theories, and the operation `Zero`, writing its only value; `real`,
    // `int`, `bool`, `word`, `enum` and `event` each hold the sort of their
    // theory and its operations. The operations carrying tensors come from `theory::torch`,
    // the others from the module of their theory.
    let mut namespaces = Vec::new();
    {
        let zero = PyModule::new(py, "zero")?;
        zero.add_class::<theory::zero::Real0>()?;
        zero.add_class::<theory::zero::Zero>()?;
        m.add("zero", &zero)?;
        namespaces.push(("zero", zero));
    }
    {
        let real = PyModule::new(py, "real")?;
        real.add_class::<theory::real::Real>()?;
        real.add_class::<theory::torch::real::Num>()?;
        real.add_class::<theory::torch::real::Scale>()?;
        real.add_class::<theory::torch::real::Linear>()?;
        real.add_class::<theory::torch::real::Affine>()?;
        real.add_class::<theory::real::Add>()?;
        real.add_class::<theory::real::Sub>()?;
        real.add_class::<theory::real::Mul>()?;
        real.add_class::<theory::real::Div>()?;
        real.add_class::<theory::real::ReLU>()?;
        real.add_class::<theory::real::Abs>()?;
        real.add_class::<theory::real::Neg>()?;
        real.add_class::<theory::real::Zero>()?;
        real.add_class::<theory::real::Argmax>()?;
        real.add_class::<theory::real::Min>()?;
        real.add_class::<theory::real::Max>()?;
        real.add_class::<theory::real::Transpose>()?;
        real.add_class::<theory::real::MatMul>()?;
        real.add_class::<theory::real::Lt>()?;
        real.add_class::<theory::real::Ite>()?;
        real.add_class::<theory::real::Le>()?;
        real.add_class::<theory::real::Gt>()?;
        real.add_class::<theory::real::Ge>()?;
        real.add_class::<theory::real::Eq>()?;
        real.add_class::<theory::real::Ne>()?;
        {
            let broadcast = PyModule::new(py, "broadcast")?;
            broadcast.add_class::<theory::real::broadcast::Ite>()?;
            real.add("broadcast", &broadcast)?;
            namespaces.push(("real.broadcast", broadcast));
        }
        m.add("real", &real)?;
        namespaces.push(("real", real));
    }
    {
        let int = PyModule::new(py, "int")?;
        int.add_class::<theory::int::Int>()?;
        int.add_class::<theory::torch::int::Num>()?;
        int.add_class::<theory::torch::int::Scale>()?;
        int.add_class::<theory::torch::int::Linear>()?;
        int.add_class::<theory::torch::int::Affine>()?;
        int.add_class::<theory::int::Add>()?;
        int.add_class::<theory::int::Sub>()?;
        int.add_class::<theory::int::Mul>()?;
        int.add_class::<theory::int::Div>()?;
        int.add_class::<theory::int::Mod>()?;
        int.add_class::<theory::int::ReLU>()?;
        int.add_class::<theory::int::Neg>()?;
        int.add_class::<theory::int::Abs>()?;
        int.add_class::<theory::int::Argmax>()?;
        int.add_class::<theory::int::Min>()?;
        int.add_class::<theory::int::Max>()?;
        int.add_class::<theory::int::Transpose>()?;
        int.add_class::<theory::int::MatMul>()?;
        int.add_class::<theory::int::Lt>()?;
        int.add_class::<theory::int::Ite>()?;
        int.add_class::<theory::int::Le>()?;
        int.add_class::<theory::int::Gt>()?;
        int.add_class::<theory::int::Ge>()?;
        int.add_class::<theory::int::Eq>()?;
        int.add_class::<theory::int::Ne>()?;
        {
            let broadcast = PyModule::new(py, "broadcast")?;
            broadcast.add_class::<theory::int::broadcast::Ite>()?;
            int.add("broadcast", &broadcast)?;
            namespaces.push(("int.broadcast", broadcast));
        }
        m.add("int", &int)?;
        namespaces.push(("int", int));
    }
    {
        let bool = PyModule::new(py, "bool")?;
        bool.add_class::<theory::bool::Bool>()?;
        bool.add_class::<theory::torch::bool::Truth>()?;
        bool.add_class::<theory::bool::And>()?;
        bool.add_class::<theory::bool::Or>()?;
        bool.add_class::<theory::bool::Xor>()?;
        bool.add_class::<theory::bool::Not>()?;
        bool.add_class::<theory::bool::Eq>()?;
        bool.add_class::<theory::bool::Ne>()?;
        bool.add_class::<theory::bool::Ite>()?;
        {
            let broadcast = PyModule::new(py, "broadcast")?;
            broadcast.add_class::<theory::bool::broadcast::Ite>()?;
            bool.add("broadcast", &broadcast)?;
            namespaces.push(("bool.broadcast", broadcast));
        }
        m.add("bool", &bool)?;
        namespaces.push(("bool", bool));
    }
    {
        let word = PyModule::new(py, "word")?;
        word.add_class::<theory::word::Word>()?;
        word.add_class::<theory::torch::word::Num>()?;
        word.add_class::<theory::word::Concat>()?;
        word.add_class::<theory::word::Extract>()?;
        word.add_class::<theory::word::Repeat>()?;
        word.add_class::<theory::word::Not>()?;
        word.add_class::<theory::word::And>()?;
        word.add_class::<theory::word::Or>()?;
        word.add_class::<theory::word::Xor>()?;
        word.add_class::<theory::word::Nand>()?;
        word.add_class::<theory::word::Nor>()?;
        word.add_class::<theory::word::Xnor>()?;
        word.add_class::<theory::word::Neg>()?;
        word.add_class::<theory::word::Ite>()?;
        word.add_class::<theory::word::Add>()?;
        word.add_class::<theory::word::Sub>()?;
        word.add_class::<theory::word::Mul>()?;
        word.add_class::<theory::word::Shl>()?;
        word.add_class::<theory::word::RotateLeft>()?;
        word.add_class::<theory::word::RotateRight>()?;
        word.add_class::<theory::word::Eq>()?;
        word.add_class::<theory::word::Ne>()?;
        {
            let signed = PyModule::new(py, "signed")?;
            signed.add_class::<theory::word::signed::Div>()?;
            signed.add_class::<theory::word::signed::Abs>()?;
            signed.add_class::<theory::word::signed::Rem>()?;
            signed.add_class::<theory::word::signed::Mod>()?;
            signed.add_class::<theory::word::signed::Shr>()?;
            signed.add_class::<theory::word::signed::Extend>()?;
            signed.add_class::<theory::word::signed::Lt>()?;
            signed.add_class::<theory::word::signed::Le>()?;
            signed.add_class::<theory::word::signed::Gt>()?;
            signed.add_class::<theory::word::signed::Ge>()?;
            word.add("signed", &signed)?;
            namespaces.push(("word.signed", signed));
        }
        {
            let unsigned = PyModule::new(py, "unsigned")?;
            unsigned.add_class::<theory::word::unsigned::Div>()?;
            unsigned.add_class::<theory::word::unsigned::Rem>()?;
            unsigned.add_class::<theory::word::unsigned::Shr>()?;
            unsigned.add_class::<theory::word::unsigned::Extend>()?;
            unsigned.add_class::<theory::word::unsigned::Lt>()?;
            unsigned.add_class::<theory::word::unsigned::Le>()?;
            unsigned.add_class::<theory::word::unsigned::Gt>()?;
            unsigned.add_class::<theory::word::unsigned::Ge>()?;
            word.add("unsigned", &unsigned)?;
            namespaces.push(("word.unsigned", unsigned));
        }
        {
            let broadcast = PyModule::new(py, "broadcast")?;
            broadcast.add_class::<theory::word::broadcast::Ite>()?;
            word.add("broadcast", &broadcast)?;
            namespaces.push(("word.broadcast", broadcast));
        }
        m.add("word", &word)?;
        namespaces.push(("word", word));
    }
    {
        let r#enum = PyModule::new(py, "enum")?;
        r#enum.add_class::<theory::r#enum::Enum>()?;
        r#enum.add_class::<theory::r#enum::Sym>()?;
        r#enum.add_class::<theory::r#enum::Into>()?;
        r#enum.add_class::<theory::r#enum::Eq>()?;
        r#enum.add_class::<theory::r#enum::Ne>()?;
        r#enum.add_class::<theory::r#enum::Ite>()?;
        m.add("enum", &r#enum)?;
        namespaces.push(("enum", r#enum));
    }
    {
        let event = PyModule::new(py, "event")?;
        event.add_class::<theory::event::Event>()?;
        event.add_class::<theory::event::Toggle>()?;
        event.add_class::<theory::event::Eq>()?;
        event.add_class::<theory::event::Ne>()?;
        event.add_class::<theory::event::Ite>()?;
        m.add("event", &event)?;
        namespaces.push(("event", event));
    }

    // Each namespace takes its dotted name in the package that holds this
    // extension, `reactor.real` for the extension `reactor.reactor`,
    // the name its classes give as their module, and is entered in
    // `sys.modules` under it, so that it can be imported by it, as
    // `import reactor.real`, not only reached as an attribute.
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
fn tangent(sort: Any) -> Any {
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
) -> PyResult<impl Iterator<Item = reactive::Var<Any>>> {
    // TODO: make base take result iterator to avoid unwrap
    let seq = try_iter_borrow::<Var>(seq)?;
    let seq = seq.into_iter().map(Result::unwrap);
    let seq = seq.map(|r| r.base().clone());
    Ok(seq)
}

fn try_term_iter_cloned<T>(
    seq: &Bound<'_, PyAny>,
) -> PyResult<impl Iterator<Item = dataflow::diagram::Term<T, Any>>>
where
    dataflow::diagram::Term<Op, Any>: TryInto<dataflow::diagram::Term<T, Any>>,
    <dataflow::diagram::Term<Op, Any> as TryInto<dataflow::diagram::Term<T, Any>>>::Error: Debug,
{
    // TODO: make base take result iterator to avoid unwrap
    let seq = try_iter_borrow::<Term>(seq)?;
    let seq = seq.into_iter().map(Result::unwrap);
    let seq = seq.map(|r| r.base().clone().try_into().unwrap());
    Ok(seq)
}

fn try_wire_iter_cloned(
    seq: &Bound<'_, PyAny>,
) -> PyResult<impl Iterator<Item = dataflow::diagram::Wire<Any>>> {
    // TODO: make base take result iterator to avoid unwrap
    let seq = seq.try_iter()?.map(|i| -> PyResult<_> {
        let i = i?;
        if let Ok(wire) = i.extract::<PyRef<Wire>>() {
            return Ok(wire.base().clone());
        }
        // a variable stands for its latched wire in a wire position
        if let Ok(var) = i.extract::<PyRef<Var>>() {
            return Ok(dataflow::diagram::Wire::from(var.base().clone()));
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
    Set(HashSet<reactive::Var<Any>>),
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
    pub(crate) fn as_fn(&self) -> impl Fn(&reactive::Var<Any>) -> bool + '_ {
        move |var| match self {
            Hide::Set(set) => set.contains(var),
            Hide::Call(hide, err) => {
                if err.borrow().is_some() {
                    return false;
                }
                match hide
                    .call1((Var::from(var.clone()),))
                    .and_then(|r| r.is_truthy())
                {
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
