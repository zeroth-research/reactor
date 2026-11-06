use ::theory::bv::BV;
use ::theory::lia::LIA;
use ::theory::lra::LRA;
use pyo3::PyClass;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::collections::HashSet;
use std::fmt::Debug;
use std::fmt::Write;
use theory::Theory;
use theory::any::{Any, Combinatorial, Differential, Sequential, Sort};

mod atom;
mod block;
mod module;
mod term;
mod var;
mod wire;

use crate::atom::Atom;
use crate::block::Block;
use crate::module::Module;
use crate::term::Term;
use crate::var::{Var, View};
use crate::wire::Wire;

#[pymodule]
fn reactor(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Sort>()?;
    m.add_class::<Wire>()?;
    m.add_class::<Var>()?;
    m.add_class::<View>()?;
    m.add_class::<Atom>()?;
    m.add_class::<Term>()?;
    m.add_class::<Block>()?;
    m.add_class::<Module>()?;

    m.add_class::<Combinatorial>()?;
    m.add_class::<Sequential>()?;
    m.add_class::<Differential>()?;
    m.add_class::<LRA>()?;
    m.add_class::<LIA>()?;
    m.add_class::<BV>()?;

    m.add_function(wrap_pyfunction!(var::x, m)?)?;
    m.add_function(wrap_pyfunction!(var::d, m)?)?;

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

/// The variables of the python sequence.
fn try_var_iter_cloned(
    seq: &Bound<'_, PyAny>,
) -> PyResult<impl Iterator<Item = PyResult<reactive::Var<Sort>>>> {
    let seq = try_iter_borrow::<Var>(seq)?;
    Ok(seq.map(|var| var.map(|var| *var.base())))
}

/// The terms of the python sequence, over the theory `T`.
fn try_term_iter_cloned<T>(
    seq: &Bound<'_, PyAny>,
) -> PyResult<impl Iterator<Item = PyResult<dataflow::diagram::Term<T>>>>
where
    T: Theory,
    dataflow::diagram::Term<Any>: TryInto<dataflow::diagram::Term<T>>,
    <dataflow::diagram::Term<Any> as TryInto<dataflow::diagram::Term<T>>>::Error: Debug,
{
    let seq = try_iter_borrow::<Term>(seq)?;
    Ok(seq.map(|term| {
        let term = term?.base().clone();
        term.try_into()
            .map_err(|e| PyException::new_err(format!("{e:?}")))
    }))
}

/// The wires of the python sequence.
fn try_wire_iter_cloned(
    seq: &Bound<'_, PyAny>,
) -> PyResult<impl Iterator<Item = PyResult<base::Wire<Sort>>>> {
    let seq = try_iter_borrow::<Wire>(seq)?;
    Ok(seq.map(|wire| wire.map(|wire| *wire.base())))
}

/// The python-facing `hide` argument: either a callable predicate over
/// variables (e.g. a lambda), or any iterable of variables (preferably a
/// set); `None` hides nothing.
pub(crate) enum Hide<'py> {
    // the names of the variables
    Set(HashSet<u32>),
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
            let names = iter.map(|var| var.map(|var| var.name()));
            Ok(Hide::Set(names.collect::<PyResult<_>>()?))
        } else {
            Err(PyException::new_err(
                "hide must be either callable or iterable",
            ))
        }
    }

    /// The predicate handed to the base constructors; the callable is given
    /// the python variables.
    pub(crate) fn as_fn(&self) -> impl Fn(&reactive::Var<Sort>) -> bool + '_ {
        move |var| match self {
            Hide::Set(set) => set.contains(&var.name()),
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

/// Renders wires untyped and comma-separated.
pub(crate) fn wires_untyped_to_repr<'a, S, I>(wires: I) -> String
where
    S: Debug + 'a,
    I: IntoIterator<Item = &'a base::Wire<S>>,
{
    let mut s = String::from("[");
    for (i, wire) in wires.into_iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        write!(s, "{:?}", wire.typed()).unwrap(); // we know it's infallible
    }
    s.push_str("]");
    s
}
