use crate::block::Block;
use crate::var::{Var, View};
use crate::{try_iter_borrow, try_var_iter_cloned};
use pyo3::exceptions::{PyException, PyIndexError};
use pyo3::prelude::*;
use reactive::var::PREFIX;
use std::borrow::Cow;
use std::collections::HashMap;
use theory::any::{Combinatorial, Differential, Sequential, Sort};

#[pyclass(frozen)]
pub(crate) struct Atom {
    base: reactive::Atom<Combinatorial, Sequential, Differential>,
}

#[pymethods]
impl Atom {
    /// The atom controlling `ctrl`, awaiting `wait` and reading `read`, with
    /// the blocks `init`, `update` and `delay`. The atom binds its variables
    /// to the boundaries of its blocks by position, in the order of their
    /// names: `init` maps `X(wait)` to `X(ctrl)`, `update` maps `read`
    /// followed by `X(wait)` to `X(ctrl)`, and `delay` maps `read` followed
    /// by `d(wait)` to `d(ctrl)`.
    #[new]
    fn new(
        ctrl: &Bound<'_, PyAny>,
        wait: &Bound<'_, PyAny>,
        read: &Bound<'_, PyAny>,
        init: PyRef<'_, Block>,
        update: PyRef<'_, Block>,
        delay: PyRef<'_, Block>,
    ) -> PyResult<Self> {
        let base = reactive::Atom::new(
            interface(ctrl)?,
            interface(wait)?,
            interface(read)?,
            init.narrow()?,
            update.narrow()?,
            delay.narrow()?,
        );
        base.map(|base| Self { base }).map_err(PyException::new_err)
    }

    #[getter]
    fn read(slf: PyRef<'_, Self>) -> PyResult<AtomInterface> {
        Self::interface(slf, AtomInterfaceType::Read)
    }

    #[getter]
    fn ctrl(slf: PyRef<'_, Self>) -> PyResult<AtomInterface> {
        Self::interface(slf, AtomInterfaceType::Ctrl)
    }

    #[getter]
    fn wait(slf: PyRef<'_, Self>) -> PyResult<AtomInterface> {
        Self::interface(slf, AtomInterfaceType::Await)
    }

    #[getter]
    fn init(&self) -> Block {
        self.base.init().clone().into()
    }

    #[getter]
    fn update(&self) -> Block {
        self.base.update().clone().into()
    }

    #[getter]
    fn delay(&self) -> Block {
        self.base.delay().clone().into()
    }

    /// Renders the atom with variables named by `names`; variables
    /// without an entry print as `#{name}`. Don't rely on the fallback - it may change anytime
    fn show(&self, names: HashMap<Var, String>) -> String {
        let names = varnames(&names);
        self.base
            .with_varnames(|v| match names.get(&v.name()) {
                Some(varname) => Cow::Borrowed(*varname),
                None => Cow::Owned(format!("{PREFIX}{}", v.name())),
            })
            .to_string()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.base)
    }
}

impl Atom {
    pub(crate) fn base(&self) -> &reactive::Atom<Combinatorial, Sequential, Differential> {
        &self.base
    }

    fn interface(slf: PyRef<'_, Self>, interface: AtomInterfaceType) -> PyResult<AtomInterface> {
        let py = slf.py();
        let atom = slf.into_pyobject(py)?.unbind();
        Ok(AtomInterface { atom, interface })
    }
}

impl From<reactive::Atom<Combinatorial, Sequential, Differential>> for Atom {
    fn from(base: reactive::Atom<Combinatorial, Sequential, Differential>) -> Self {
        Self { base }
    }
}

/// The interface of the python variables of the sequence `vars`.
pub(crate) fn interface(vars: &Bound<'_, PyAny>) -> PyResult<reactive::AtomInterface<Sort>> {
    let vars = try_var_iter_cloned(vars)?.collect::<PyResult<Vec<_>>>()?;
    reactive::AtomInterface::try_from_iter(vars).map_err(PyException::new_err)
}

/// The names given to the variables, by the names of the variables.
pub(crate) fn varnames(names: &HashMap<Var, String>) -> HashMap<u32, &str> {
    names
        .iter()
        .map(|(var, varname)| (var.base().name(), varname.as_str()))
        .collect()
}

#[derive(Clone)]
enum AtomInterfaceType {
    Read,
    Ctrl,
    Await,
}

#[pyclass(sequence)]
struct AtomInterface {
    atom: Py<Atom>,
    interface: AtomInterfaceType,
}

impl AtomInterface {
    fn base(&self) -> &reactive::AtomInterface<theory::any::Sort> {
        let atom = &self.atom.get().base;
        match self.interface {
            AtomInterfaceType::Read => atom.read(),
            AtomInterfaceType::Await => atom.wait(),
            AtomInterfaceType::Ctrl => atom.ctrl(),
        }
    }
}

#[pymethods]
impl AtomInterface {
    fn __repr__(&self) -> String {
        interface_repr(self.base())
    }

    fn __getitem__(&self, index: usize) -> PyResult<Var> {
        self.base()
            .nth(index)
            .map(|v| Var::from(*v))
            .ok_or(PyIndexError::new_err("index out of bounds"))
    }

    fn __len__(&self) -> usize {
        self.base().len()
    }

    fn __eq__<'py>(&self, other: &Bound<'py, PyAny>) -> bool {
        interface_eq(self.base(), other)
    }

    fn __contains__(&self, item: &Bound<'_, PyAny>) -> bool {
        interface_contains(self.base(), item)
    }
}

/// Renders the variables of the interface, comma-separated.
pub(crate) fn interface_repr(interface: &reactive::ModuleInterface<theory::any::Sort>) -> String {
    let vars: Vec<String> = interface
        .iter()
        .map(|v| format!("{PREFIX}{}: {}", v.name(), v.dtype()))
        .collect();
    format!("[{}]", vars.join(", "))
}

/// Whether the interface lists exactly the python variables of `other`, in
/// order.
pub(crate) fn interface_eq(
    interface: &reactive::ModuleInterface<theory::any::Sort>,
    other: &Bound<'_, PyAny>,
) -> bool {
    let other = match try_iter_borrow::<Var>(other) {
        Ok(other) => other,
        Err(_) => return false,
    };

    let mut this = interface.iter();
    let mut other = other.into_iter();
    loop {
        match (this.next(), other.next()) {
            (Some(this), Some(Ok(other))) => {
                if this != other.base() {
                    return false;
                }
            }
            (None, None) => return true,
            _ => return false,
        }
    }
}

/// Whether the interface contains the python variable `item`, or the variable
/// of the view `item` (its next value or its derivative).
pub(crate) fn interface_contains(
    interface: &reactive::ModuleInterface<theory::any::Sort>,
    item: &Bound<'_, PyAny>,
) -> bool {
    if let Ok(var) = item.extract::<PyRef<Var>>() {
        return interface.contains(var.base());
    }
    if let Ok(view) = item.extract::<PyRef<View>>() {
        return interface.contains(view.var_ref().base());
    }
    false
}
