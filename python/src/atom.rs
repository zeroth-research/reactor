use crate::term::{Term, TermInterfaceType};
use crate::var::{Bindings, Var, name, named};
use crate::wire::Wire;
use crate::{try_iter_borrow, try_term_iter_cloned, try_var_iter_cloned, wires_untyped_to_repr};
use pyo3::exceptions::{PyException, PyIndexError};
use pyo3::prelude::*;
use std::borrow::Cow;
use std::collections::HashMap;
use theory::any::{Any, Combinatorial, Differential, Sequential};

#[pyclass(frozen)]
pub(crate) struct Atom {
    base: reactive::Atom<Combinatorial, Sequential, Differential>,
    // the python variables of the names of the atom
    bindings: Bindings,
}

#[pymethods]
impl Atom {
    #[new]
    #[pyo3(signature = (vars, init = None, delay = None, update = None))]
    fn new(
        vars: &Bound<'_, PyAny>,
        init: Option<&Bound<'_, PyAny>>,
        delay: Option<&Bound<'_, PyAny>>,
        update: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        match (init, delay, update) {
            (Some(init), None, Some(update)) => Self::sequential(vars, init, update),
            (Some(init), Some(delay), None) => Self::differential(vars, init, delay),
            (Some(init), Some(delay), Some(update)) => Self::hybrid(vars, init, update, delay),
            (None, None, Some(update)) => Self::jump(vars, update),
            (None, Some(delay), Some(update)) => Self::uninitialized(vars, update, delay),
            (Some(init), None, None) => Self::constant(vars, init),
            (None, Some(delay), None) => Self::flow(vars, delay),
            (None, None, None) => Self::hold(vars),
        }
    }

    #[staticmethod]
    fn sequential(
        vars: &Bound<'_, PyAny>,
        init: &Bound<'_, PyAny>,
        update: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let vars: Vec<_> = try_var_iter_cloned(vars)?.collect();
        let init = try_term_iter_cloned(&init)?;
        let update = try_term_iter_cloned(&update)?;

        match reactive::Atom::sequential(named(&vars), init, update) {
            Ok(base) => Ok(Self::from_base(base, Bindings::new(&vars))),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    fn differential(
        vars: &Bound<'_, PyAny>,
        init: &Bound<'_, PyAny>,
        delay: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let vars: Vec<_> = try_var_iter_cloned(vars)?.collect();
        let init = try_term_iter_cloned(&init)?;
        let delay = try_term_iter_cloned(&delay)?;

        match reactive::Atom::differential(named(&vars), init, delay) {
            Ok(base) => Ok(Self::from_base(base, Bindings::new(&vars))),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    fn hybrid(
        vars: &Bound<'_, PyAny>,
        init: &Bound<'_, PyAny>,
        update: &Bound<'_, PyAny>,
        delay: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let vars: Vec<_> = try_var_iter_cloned(vars)?.collect();
        let init = try_term_iter_cloned(&init)?;
        let update = try_term_iter_cloned(&update)?;
        let delay = try_term_iter_cloned(&delay)?;

        match reactive::Atom::hybrid(named(&vars), init, update, delay) {
            Ok(base) => Ok(Self::from_base(base, Bindings::new(&vars))),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    fn jump(vars: &Bound<'_, PyAny>, update: &Bound<'_, PyAny>) -> PyResult<Self> {
        let vars: Vec<_> = try_var_iter_cloned(vars)?.collect();
        let update = try_term_iter_cloned(&update)?;

        match reactive::Atom::jump(named(&vars), update) {
            Ok(base) => Ok(Self::from_base(base, Bindings::new(&vars))),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    fn uninitialized(
        vars: &Bound<'_, PyAny>,
        update: &Bound<'_, PyAny>,
        delay: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let vars: Vec<_> = try_var_iter_cloned(vars)?.collect();
        let update = try_term_iter_cloned(&update)?;
        let delay = try_term_iter_cloned(&delay)?;

        match reactive::Atom::uninitialized(named(&vars), update, delay) {
            Ok(base) => Ok(Self::from_base(base, Bindings::new(&vars))),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    fn constant(vars: &Bound<'_, PyAny>, init: &Bound<'_, PyAny>) -> PyResult<Self> {
        let vars: Vec<_> = try_var_iter_cloned(vars)?.collect();
        let init = try_term_iter_cloned(&init)?;

        match reactive::Atom::constant(named(&vars), init) {
            Ok(base) => Ok(Self::from_base(base, Bindings::new(&vars))),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    fn hold(vars: &Bound<'_, PyAny>) -> PyResult<Self> {
        let vars: Vec<_> = try_var_iter_cloned(vars)?.collect();

        match reactive::Atom::hold(named(&vars)) {
            Ok(base) => Ok(Self::from_base(base, Bindings::new(&vars))),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    fn flow(vars: &Bound<'_, PyAny>, delay: &Bound<'_, PyAny>) -> PyResult<Self> {
        let vars: Vec<_> = try_var_iter_cloned(vars)?.collect();
        let delay = try_term_iter_cloned(&delay)?;

        match reactive::Atom::flow(named(&vars), delay) {
            Ok(base) => Ok(Self::from_base(base, Bindings::new(&vars))),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    fn combinatorial(vars: &Bound<'_, PyAny>, assign: &Bound<'_, PyAny>) -> PyResult<Self> {
        let vars: Vec<_> = try_var_iter_cloned(vars)?.collect();
        let assign = try_term_iter_cloned::<Combinatorial>(&assign)?;

        match reactive::Atom::combinatorial(named(&vars), assign) {
            Ok(base) => Ok(Self::from_base(base, Bindings::new(&vars))),
            Err(msg) => Err(PyException::new_err(msg)),
        }
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
    fn init(slf: Bound<'_, Self>) -> AtomBlock {
        AtomBlock {
            atom: slf.unbind(),
            block: BlockType::Init,
        }
    }

    #[getter]
    fn update(slf: Bound<'_, Self>) -> AtomBlock {
        AtomBlock {
            atom: slf.unbind(),
            block: BlockType::Update,
        }
    }

    #[getter]
    fn delay(slf: Bound<'_, Self>) -> AtomBlock {
        AtomBlock {
            atom: slf.unbind(),
            block: BlockType::Delay,
        }
    }

    /// Renders the atom with variables named by `names`; variables
    /// without an entry print as `#{id}`. Don't rely on the fallback - it may change anytime
    fn show(&self, names: HashMap<Var, String>) -> String {
        let names: HashMap<u32, &str> = names
            .iter()
            .map(|(var, varname)| (name(var.base()).name(), varname.as_str()))
            .collect();
        self.base
            .with_varnames(|v| match names.get(&v.name()) {
                Some(varname) => Cow::Borrowed(*varname),
                None => Cow::Owned(format!("#{}", v.name())),
            })
            .to_string()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.base)
    }
}

impl Atom {
    pub(crate) fn from_base(
        base: reactive::Atom<Combinatorial, Sequential, Differential>,
        bindings: Bindings,
    ) -> Self {
        Self { base, bindings }
    }

    pub(crate) fn base(&self) -> &reactive::Atom<Combinatorial, Sequential, Differential> {
        &self.base
    }

    pub(crate) fn bindings(&self) -> &Bindings {
        &self.bindings
    }

    fn interface(slf: PyRef<'_, Self>, interface: AtomInterfaceType) -> PyResult<AtomInterface> {
        let py = slf.py();
        let atom = slf.into_pyobject(py)?.unbind();
        Ok(AtomInterface { atom, interface })
    }
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

    fn bindings(&self) -> &Bindings {
        &self.atom.get().bindings
    }
}
#[pymethods]
impl AtomInterface {
    fn __repr__(&self) -> String {
        interface_repr(self.base(), self.bindings())
    }

    fn __getitem__(&self, index: usize) -> PyResult<Var> {
        self.base()
            .nth(index)
            .map(|v| self.bindings().var(v))
            .ok_or(PyIndexError::new_err("index out of bounds"))
    }

    fn __len__(&self) -> usize {
        self.base().len()
    }

    fn __eq__<'py>(&self, other: &Bound<'py, PyAny>) -> bool {
        interface_eq(self.base(), other)
    }

    fn __contains__(&self, item: &Bound<'_, PyAny>) -> bool {
        interface_contains(self.base(), self.bindings(), item)
    }
}

/// Renders the latched wires of the variables of the interface, untyped and
/// comma-separated.
pub(crate) fn interface_repr(
    interface: &reactive::ModuleInterface<theory::any::Sort>,
    bindings: &Bindings,
) -> String {
    let vars: Vec<_> = interface.iter().map(|v| *bindings.wires(v)).collect();
    wires_untyped_to_repr(&vars)
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
                if *this != name(other.base()) {
                    return false;
                }
            }
            (None, None) => return true,
            _ => return false,
        }
    }
}

/// Whether the interface contains the python variable `item`, or the variable
/// of the wire `item` (latched, next, or derivative).
pub(crate) fn interface_contains(
    interface: &reactive::ModuleInterface<theory::any::Sort>,
    bindings: &Bindings,
    item: &Bound<'_, PyAny>,
) -> bool {
    if let Ok(var) = item.extract::<PyRef<Var>>() {
        return interface.contains(&name(var.base()));
    }
    if let Ok(wire) = item.extract::<PyRef<Wire>>() {
        let wire = wire.base();
        return interface.iter().map(|v| bindings.wires(v)).any(|var| {
            **var == *wire || reactive::var::X_ref(var) == wire || reactive::var::d_ref(var) == wire
        });
    }
    false
}

#[derive(Clone)]
enum BlockType {
    Init,
    Update,
    Delay,
}

#[pyclass(sequence, frozen)]
pub(crate) struct AtomBlock {
    atom: Py<Atom>,
    block: BlockType,
}

// impl AtomBlock {
//     fn base(&self) -> &dataflow::diagram::Block<Sequential> {
//         let atom = &self.atom.get().base;
//         match self.block {
//             BlockType::Init => atom.init(),
//             BlockType::Update => atom.update(),
//             BlockType::Delay => atom.delay(),
//         }
//     }
// }

#[pymethods]
impl AtomBlock {
    // requires display in base - do you need that now?
    // fn __str__(&self) -> String {
    //     self.base().to_string()
    // }

    fn __repr__(&self) -> String {
        let atom = &self.atom.get().base;
        match self.block {
            BlockType::Init => format!("{:?}", atom.init()),
            BlockType::Update => format!("{:?}", atom.update()),
            BlockType::Delay => format!("{:?}", atom.delay()),
        }
    }

    fn read(slf: Bound<'_, Self>) -> AtomBlockInterface {
        AtomBlockInterface {
            block: slf.unbind(),
            interface: TermInterfaceType::Read,
        }
    }

    fn write(slf: Bound<'_, Self>) -> AtomBlockInterface {
        AtomBlockInterface {
            block: slf.unbind(),
            interface: TermInterfaceType::Write,
        }
    }

    fn __getitem__(&self, index: usize) -> PyResult<Term> {
        let atom = &self.atom.get().base;
        let item: Option<dataflow::diagram::Term<Any>> = match self.block {
            BlockType::Init => atom.init().get(index).cloned().map(Into::into),
            BlockType::Update => atom.update().get(index).cloned().map(Into::into),
            BlockType::Delay => atom.delay().get(index).cloned().map(Into::into),
        };
        item.map(Into::into)
            .ok_or(PyIndexError::new_err("index out of bounds"))
    }

    fn __len__(&self) -> usize {
        let atom = &self.atom.get().base;
        match self.block {
            BlockType::Init => atom.init().len(),
            BlockType::Update => atom.update().len(),
            BlockType::Delay => atom.delay().len(),
        }
    }
}

#[pyclass(sequence)]
struct AtomBlockInterface {
    block: Py<AtomBlock>,
    interface: TermInterfaceType,
}

impl AtomBlockInterface {
    fn base(&self) -> &[base::Wire<theory::any::Sort>] {
        let atom = &self.block.get().atom.get().base;
        match (&self.block.get().block, &self.interface) {
            (BlockType::Init, TermInterfaceType::Read) => atom.init().read(),
            (BlockType::Init, TermInterfaceType::Write) => atom.init().write(),
            (BlockType::Update, TermInterfaceType::Read) => atom.update().read(),
            (BlockType::Update, TermInterfaceType::Write) => atom.update().write(),
            (BlockType::Delay, TermInterfaceType::Read) => atom.delay().read(),
            (BlockType::Delay, TermInterfaceType::Write) => atom.delay().write(),
        }
    }
}

#[pymethods]
impl AtomBlockInterface {
    fn __repr__(&self) -> String {
        wires_untyped_to_repr(self.base())
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        let other = match try_iter_borrow::<Wire>(other) {
            Ok(other) => other,
            Err(_) => return false,
        };

        let mut this = self.base().iter();
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

    fn __getitem__(&self, index: usize) -> PyResult<Wire> {
        self.base()
            .get(index)
            .map(|w| w.clone().into())
            .ok_or(PyIndexError::new_err("index out of bounds"))
    }

    fn __len__(&self) -> usize {
        self.base().len()
    }
}
