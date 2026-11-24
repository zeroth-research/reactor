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
use theory::tensor::sort;

/// The operations of all theories, with their tensors held as `tch::Tensor`s.
pub(crate) type Any = theory::tensor::Any<tch::Tensor>;

#[pymodule]
fn reactor(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Wire>()?;
    m.add_class::<Var>()?;
    m.add_class::<Atom>()?;
    m.add_class::<Term>()?;
    m.add_class::<Module>()?;

    m.add_function(wrap_pyfunction!(var::x, m)?)?;
    m.add_function(wrap_pyfunction!(var::d, m)?)?;

    // The operations and sorts, in namespaces mirroring the `theory` crate:
    // `tensor` holds the structural operations, with a namespace per theory
    // and for the sorts. The operations carrying tensors come from
    // `theory::torch`, the others from `theory::tensor`.
    let mut namespaces = Vec::new();
    {
        let tensor = PyModule::new(py, "tensor")?;
        tensor.add_class::<theory::tensor::Havoc>()?;
        tensor.add_class::<theory::tensor::Skip>()?;
        {
            let zero = PyModule::new(py, "zero")?;
            zero.add_class::<theory::tensor::zero::Zero>()?;
            tensor.add("zero", &zero)?;
            namespaces.push(("tensor.zero", zero));
        }
        {
            let real = PyModule::new(py, "real")?;
            real.add_class::<theory::torch::real::Const>()?;
            real.add_class::<theory::torch::real::Scale>()?;
            real.add_class::<theory::torch::real::Linear>()?;
            real.add_class::<theory::torch::real::Affine>()?;
            real.add_class::<theory::tensor::real::Add>()?;
            real.add_class::<theory::tensor::real::Sub>()?;
            real.add_class::<theory::tensor::real::Mul>()?;
            real.add_class::<theory::tensor::real::Div>()?;
            real.add_class::<theory::tensor::real::ReLU>()?;
            real.add_class::<theory::tensor::real::Abs>()?;
            real.add_class::<theory::tensor::real::Neg>()?;
            real.add_class::<theory::tensor::real::Zero>()?;
            real.add_class::<theory::tensor::real::Argmax>()?;
            real.add_class::<theory::tensor::real::Min>()?;
            real.add_class::<theory::tensor::real::Max>()?;
            real.add_class::<theory::tensor::real::Transpose>()?;
            real.add_class::<theory::tensor::real::MatMul>()?;
            real.add_class::<theory::tensor::real::Lt>()?;
            real.add_class::<theory::tensor::real::Ite>()?;
            real.add_class::<theory::tensor::real::Le>()?;
            real.add_class::<theory::tensor::real::Gt>()?;
            real.add_class::<theory::tensor::real::Ge>()?;
            real.add_class::<theory::tensor::real::Eq>()?;
            real.add_class::<theory::tensor::real::Ne>()?;
            tensor.add("real", &real)?;
            namespaces.push(("tensor.real", real));
        }
        {
            let int = PyModule::new(py, "int")?;
            int.add_class::<theory::torch::int::Const>()?;
            int.add_class::<theory::torch::int::Scale>()?;
            int.add_class::<theory::torch::int::Linear>()?;
            int.add_class::<theory::torch::int::Affine>()?;
            int.add_class::<theory::tensor::int::Add>()?;
            int.add_class::<theory::tensor::int::Sub>()?;
            int.add_class::<theory::tensor::int::Mul>()?;
            int.add_class::<theory::tensor::int::Div>()?;
            int.add_class::<theory::tensor::int::Mod>()?;
            int.add_class::<theory::tensor::int::ReLU>()?;
            int.add_class::<theory::tensor::int::Neg>()?;
            int.add_class::<theory::tensor::int::Abs>()?;
            int.add_class::<theory::tensor::int::Argmax>()?;
            int.add_class::<theory::tensor::int::Min>()?;
            int.add_class::<theory::tensor::int::Max>()?;
            int.add_class::<theory::tensor::int::Transpose>()?;
            int.add_class::<theory::tensor::int::MatMul>()?;
            int.add_class::<theory::tensor::int::Lt>()?;
            int.add_class::<theory::tensor::int::Ite>()?;
            int.add_class::<theory::tensor::int::Le>()?;
            int.add_class::<theory::tensor::int::Gt>()?;
            int.add_class::<theory::tensor::int::Ge>()?;
            int.add_class::<theory::tensor::int::Eq>()?;
            int.add_class::<theory::tensor::int::Ne>()?;
            tensor.add("int", &int)?;
            namespaces.push(("tensor.int", int));
        }
        {
            let bool = PyModule::new(py, "bool")?;
            bool.add_class::<theory::torch::bool::Const>()?;
            bool.add_class::<theory::tensor::bool::And>()?;
            bool.add_class::<theory::tensor::bool::Or>()?;
            bool.add_class::<theory::tensor::bool::Xor>()?;
            bool.add_class::<theory::tensor::bool::Not>()?;
            bool.add_class::<theory::tensor::bool::Eq>()?;
            bool.add_class::<theory::tensor::bool::Ne>()?;
            bool.add_class::<theory::tensor::bool::Ite>()?;
            tensor.add("bool", &bool)?;
            namespaces.push(("tensor.bool", bool));
        }
        {
            let word = PyModule::new(py, "word")?;
            word.add_class::<theory::torch::word::Const>()?;
            word.add_class::<theory::tensor::word::Concat>()?;
            word.add_class::<theory::tensor::word::Extract>()?;
            word.add_class::<theory::tensor::word::Repeat>()?;
            word.add_class::<theory::tensor::word::Not>()?;
            word.add_class::<theory::tensor::word::And>()?;
            word.add_class::<theory::tensor::word::Or>()?;
            word.add_class::<theory::tensor::word::Xor>()?;
            word.add_class::<theory::tensor::word::Nand>()?;
            word.add_class::<theory::tensor::word::Nor>()?;
            word.add_class::<theory::tensor::word::Xnor>()?;
            word.add_class::<theory::tensor::word::Neg>()?;
            word.add_class::<theory::tensor::word::Ite>()?;
            word.add_class::<theory::tensor::word::Add>()?;
            word.add_class::<theory::tensor::word::Sub>()?;
            word.add_class::<theory::tensor::word::Mul>()?;
            word.add_class::<theory::tensor::word::Shl>()?;
            word.add_class::<theory::tensor::word::RotateLeft>()?;
            word.add_class::<theory::tensor::word::RotateRight>()?;
            word.add_class::<theory::tensor::word::Eq>()?;
            word.add_class::<theory::tensor::word::Ne>()?;
            {
                let signed = PyModule::new(py, "signed")?;
                signed.add_class::<theory::tensor::word::signed::Div>()?;
                signed.add_class::<theory::tensor::word::signed::Abs>()?;
                signed.add_class::<theory::tensor::word::signed::Rem>()?;
                signed.add_class::<theory::tensor::word::signed::Mod>()?;
                signed.add_class::<theory::tensor::word::signed::Shr>()?;
                signed.add_class::<theory::tensor::word::signed::Extend>()?;
                signed.add_class::<theory::tensor::word::signed::Lt>()?;
                signed.add_class::<theory::tensor::word::signed::Le>()?;
                signed.add_class::<theory::tensor::word::signed::Gt>()?;
                signed.add_class::<theory::tensor::word::signed::Ge>()?;
                word.add("signed", &signed)?;
                namespaces.push(("tensor.word.signed", signed));
            }
            {
                let unsigned = PyModule::new(py, "unsigned")?;
                unsigned.add_class::<theory::tensor::word::unsigned::Div>()?;
                unsigned.add_class::<theory::tensor::word::unsigned::Rem>()?;
                unsigned.add_class::<theory::tensor::word::unsigned::Shr>()?;
                unsigned.add_class::<theory::tensor::word::unsigned::Extend>()?;
                unsigned.add_class::<theory::tensor::word::unsigned::Lt>()?;
                unsigned.add_class::<theory::tensor::word::unsigned::Le>()?;
                unsigned.add_class::<theory::tensor::word::unsigned::Gt>()?;
                unsigned.add_class::<theory::tensor::word::unsigned::Ge>()?;
                word.add("unsigned", &unsigned)?;
                namespaces.push(("tensor.word.unsigned", unsigned));
            }
            tensor.add("word", &word)?;
            namespaces.push(("tensor.word", word));
        }
        {
            let sort = PyModule::new(py, "sort")?;
            sort.add_class::<theory::tensor::sort::Real>()?;
            sort.add_class::<theory::tensor::sort::Int>()?;
            sort.add_class::<theory::tensor::sort::Bool>()?;
            sort.add_class::<theory::tensor::sort::Word>()?;
            sort.add_class::<theory::tensor::sort::Zero>()?;
            tensor.add("sort", &sort)?;
            namespaces.push(("tensor.sort", sort));
        }
        m.add("tensor", &tensor)?;
        namespaces.push(("tensor", tensor));
    }

    // Each namespace takes its dotted name, and is entered in `sys.modules`
    // under it, so that it can be imported by it, as
    // `import reactor.reactor.tensor.real`, not only reached as an attribute.
    let modules = py.import("sys")?.getattr("modules")?;
    for (name, namespace) in namespaces {
        let dotted = format!("{}.{}", m.name()?, name);
        namespace.setattr("__name__", &dotted)?;
        modules.set_item(&dotted, namespace)?;
    }

    Ok(())
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
) -> PyResult<impl Iterator<Item = reactive::Var<sort::Any>>> {
    // TODO: make base take result iterator to avoid unwrap
    let seq = try_iter_borrow::<Var>(seq)?;
    let seq = seq.into_iter().map(Result::unwrap);
    let seq = seq.map(|r| *r.base());
    Ok(seq)
}

fn try_term_iter_cloned<T>(
    seq: &Bound<'_, PyAny>,
) -> PyResult<impl Iterator<Item = dataflow::diagram::Term<T, sort::Any>>>
where
    dataflow::diagram::Term<Any, sort::Any>: TryInto<dataflow::diagram::Term<T, sort::Any>>,
    <dataflow::diagram::Term<Any, sort::Any> as TryInto<dataflow::diagram::Term<T, sort::Any>>>::Error: Debug,
{
    // TODO: make base take result iterator to avoid unwrap
    let seq = try_iter_borrow::<Term>(seq)?;
    let seq = seq.into_iter().map(Result::unwrap);
    let seq = seq.map(|r| r.base().clone().try_into().unwrap());
    Ok(seq)
}

fn try_wire_iter_cloned(
    seq: &Bound<'_, PyAny>,
) -> PyResult<impl Iterator<Item = dataflow::diagram::Wire<sort::Any>>> {
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
    Set(HashSet<reactive::Var<sort::Any>>),
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
    pub(crate) fn as_fn(&self) -> impl Fn(&reactive::Var<sort::Any>) -> bool + '_ {
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
