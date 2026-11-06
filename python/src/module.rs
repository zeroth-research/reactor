use crate::atom::{interface_contains, interface_eq, interface_repr, varnames};
use crate::*;
use pyo3::Py;
use pyo3::exceptions::{PyException, PyIndexError, PyTypeError};
use pyo3::types::PyTuple;
use reactive::var::PREFIX;
use std::borrow::Cow;
use std::collections::HashMap;
use theory::any::{Combinatorial, Differential, Sequential};

#[pyclass(subclass, frozen)]
pub(crate) struct Module {
    pub(crate) base: reactive::Module<Combinatorial, Sequential, Differential>,
}

type Base = reactive::Module<Combinatorial, Sequential, Differential>;

impl Module {
    /// Wraps the base module, or raises its error.
    fn wrap(module: Result<Base, String>) -> PyResult<Self> {
        module
            .map(|base| Self { base })
            .map_err(PyException::new_err)
    }
}

#[pymethods]
impl Module {
    /// A module of the atoms `args`, hiding the `hide` variables; or the
    /// parallel composition of the modules `args`, hiding the `hide`
    /// variables; or the empty module, given nothing.
    #[new]
    #[pyo3(signature = (*args, hide = None))]
    fn new(args: &Bound<'_, PyTuple>, hide: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        if args.is_empty() {
            if hide.is_some() {
                return Err(PyTypeError::new_err(
                    "an empty module has no variables to hide",
                ));
            }
            Self::wrap(Ok(reactive::Module::empty()))
        } else if args.get_item(0)?.downcast::<Atom>().is_ok() {
            // a sequence of atoms builds a single module
            let atoms: Vec<_> = try_iter_borrow::<Atom>(args)?.collect::<PyResult<_>>()?;
            let atoms = atoms.iter().map(|a| a.base().clone());
            if let Some(hide) = hide {
                let hide = Hide::try_from(hide)?;
                let module = reactive::Module::partially_observable(atoms, hide.as_fn());
                hide.err().and_then(|()| Self::wrap(module))
            } else {
                Self::wrap(reactive::Module::observable(atoms))
            }
        } else {
            // a sequence of modules composes in parallel
            Self::compose(args, hide)
        }
    }

    /// The parallel composition of a sequence of modules: all shared
    /// observable variables are coupled and the `hide` variables are hidden
    /// when provided.
    #[staticmethod]
    #[pyo3(signature = (*modules, hide = None))]
    fn compose(modules: &Bound<'_, PyTuple>, hide: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let modules: Vec<_> = try_iter_borrow::<Self>(modules)?.collect::<PyResult<_>>()?;
        let modules = modules.iter().map(|r| r.base.clone());
        if let Some(hide) = hide {
            let hide = Hide::try_from(hide)?;
            let module = reactive::Module::compose_hiding(modules, hide.as_fn());
            hide.err().and_then(|()| Self::wrap(module))
        } else {
            Self::wrap(reactive::Module::compose(modules))
        }
    }

    #[getter]
    fn extl(slf: PyRef<'_, Self>) -> PyResult<ModuleInterface> {
        Self::interface(slf, ModuleInterfaceType::Extl)
    }

    #[getter]
    fn intf(slf: PyRef<'_, Self>) -> PyResult<ModuleInterface> {
        Self::interface(slf, ModuleInterfaceType::Intf)
    }

    #[getter]
    fn prvt(slf: PyRef<'_, Self>) -> PyResult<ModuleInterface> {
        Self::interface(slf, ModuleInterfaceType::Prvt)
    }

    #[getter]
    fn obs(slf: PyRef<'_, Self>) -> PyResult<ModuleInterface> {
        Self::interface(slf, ModuleInterfaceType::Obs)
    }

    #[getter]
    fn ctrl(slf: PyRef<'_, Self>) -> PyResult<ModuleInterface> {
        Self::interface(slf, ModuleInterfaceType::Ctrl)
    }

    #[getter]
    fn atoms(slf: PyRef<'_, Self>) -> PyResult<ModuleAtoms> {
        let py = slf.py();
        let module = slf.into_pyobject(py)?.unbind();
        Ok(ModuleAtoms { module })
    }

    fn closed(&self) -> bool {
        self.base.is_closed()
    }

    fn open(&self) -> bool {
        self.base.is_open()
    }

    /// Renders the module with variables named by `names`; variables
    /// without an entry print as an untyped wire.
    fn with_varnames(&self, names: HashMap<Var, String>) -> String {
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

    fn __str__(&self) -> String {
        self.base
            .with_varnames(|v| Cow::Owned(format!("{PREFIX}{}", v.name())))
            .to_string()
    }

    fn __mul__(&self, other: PyRef<Module>) -> PyResult<Module> {
        let module = reactive::Module::compose([self.base.clone(), other.base.clone()]);
        Self::wrap(module)
    }

    fn hide(&self, hide: &Bound<'_, PyAny>) -> PyResult<Module> {
        let hide = Hide::try_from(hide)?;
        let module = self.base.clone().hide(hide.as_fn());
        hide.err()?;
        Self::wrap(module)
    }
}

impl Module {
    fn interface(
        slf: PyRef<'_, Self>,
        interface: ModuleInterfaceType,
    ) -> PyResult<ModuleInterface> {
        let py = slf.py();
        let module = slf.into_pyobject(py)?.unbind();
        Ok(ModuleInterface { module, interface })
    }
}

#[derive(Clone)]
enum ModuleInterfaceType {
    Extl,
    Intf,
    Prvt,
    Obs,
    Ctrl,
}
#[pyclass(sequence)]
struct ModuleInterface {
    module: Py<Module>,
    interface: ModuleInterfaceType,
}

impl ModuleInterface {
    fn base(&self) -> &reactive::ModuleInterface<Sort> {
        let module = &self.module.get().base;
        match self.interface {
            ModuleInterfaceType::Extl => module.extl(),
            ModuleInterfaceType::Intf => module.intf(),
            ModuleInterfaceType::Prvt => module.prvt(),
            ModuleInterfaceType::Obs => module.obs(),
            ModuleInterfaceType::Ctrl => module.ctrl(),
        }
    }
}
#[pymethods]
impl ModuleInterface {
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

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        interface_eq(self.base(), other)
    }

    fn __contains__(&self, item: &Bound<'_, PyAny>) -> bool {
        interface_contains(self.base(), item)
    }
}

#[pyclass(sequence)]
struct ModuleAtoms {
    module: Py<Module>,
}

#[pymethods]
impl ModuleAtoms {
    fn __getitem__(&self, index: usize) -> PyResult<Atom> {
        let atoms = self.module.get().base.atoms();
        let result = atoms.get(index).map(|atom| Atom::from(atom.clone()));
        result.ok_or(PyIndexError::new_err("index out of bounds"))
    }

    fn __len__(&self) -> usize {
        let module = &self.module.get().base;
        module.atoms().len()
    }
}
