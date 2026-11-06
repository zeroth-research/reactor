use crate::atom::{interface_contains, interface_eq, interface_repr};
use crate::var::{Bindings, named};
use crate::*;
use base::wire::PREFIX;
use pyo3::Py;
use pyo3::exceptions::{PyException, PyIndexError, PyTypeError};
use pyo3::types::{PyDict, PyTuple};
use std::borrow::Cow;
use std::collections::HashMap;
use theory::any::{Combinatorial, Differential, Sequential};

#[pyclass(subclass, frozen)]
pub(crate) struct Module {
    pub(crate) base: reactive::Module<Combinatorial, Sequential, Differential>,
    // the python variables of the names of the module
    bindings: Bindings,
}

type Base = reactive::Module<Combinatorial, Sequential, Differential>;

impl Module {
    /// Wraps the base module, or raises its error.
    fn wrap(module: Result<Base, String>, bindings: Bindings) -> PyResult<Self> {
        module
            .map(|base| Self { base, bindings })
            .map_err(PyException::new_err)
    }

    fn partially_observable<A>(
        atoms: A,
        hide: &Bound<'_, PyAny>,
        bindings: Bindings,
    ) -> PyResult<Self>
    where
        A: IntoIterator<Item = reactive::Atom<Combinatorial, Sequential, Differential>>,
    {
        let hide = Hide::try_from(hide)?;
        let module = reactive::Module::partially_observable(atoms, hide.as_fn(&bindings));
        hide.err().and_then(|()| Self::wrap(module, bindings))
    }

    fn observable<A>(atoms: A, bindings: Bindings) -> PyResult<Self>
    where
        A: IntoIterator<Item = reactive::Atom<Combinatorial, Sequential, Differential>>,
    {
        Self::wrap(reactive::Module::observable(atoms), bindings)
    }
}

#[pymethods]
impl Module {
    #[new]
    #[pyo3(signature = (*args, init = None, delay = None, update = None, vars = None, hide = None, **_kwargs))]
    fn new(
        args: &Bound<'_, PyTuple>,
        init: Option<&Bound<'_, PyAny>>,
        delay: Option<&Bound<'_, PyAny>>,
        update: Option<&Bound<'_, PyAny>>,
        vars: Option<&Bound<'_, PyAny>>,
        hide: Option<&Bound<'_, PyAny>>,
        // accepted but unused: subclasses extending Module pass their own
        // keyword arguments through this constructor
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        if args.len() > 0 {
            if init.is_some() || delay.is_some() || update.is_some() || vars.is_some() {
                Err(PyTypeError::new_err(
                    "positional arguments take atoms or modules",
                ))
            } else if args.get_item(0)?.downcast::<Atom>().is_ok() {
                // a sequence of atoms builds a single module
                let atoms: Vec<_> = try_iter_borrow::<Atom>(args)?.collect::<PyResult<_>>()?;
                let bindings = Bindings::merge(atoms.iter().map(|a| a.bindings()));
                let atoms = atoms.iter().map(|a| a.base().clone());
                if let Some(hide) = hide {
                    Self::partially_observable(atoms, hide, bindings)
                } else {
                    Self::observable(atoms, bindings)
                }
            } else {
                // sequence of modules composes in parallel
                Self::compose(args, hide)
            }
        } else if let Some(vars) = vars {
            Self::atomic(vars, init, delay, update, hide)
        } else if init.is_some() || delay.is_some() || update.is_some() {
            Err(PyTypeError::new_err(
                "missing `vars`? An atomic module needs some",
            ))
        } else {
            Self::wrap(Ok(reactive::Module::empty()), Bindings::default())
        }
    }

    /// Builds an atomic module over `vars`, dispatching on the given blocks:
    /// each combination of `init`, `delay`, and `update` selects the
    /// corresponding constructor, hiding the `hide` variables when given.
    #[staticmethod]
    #[pyo3(signature = (vars, init = None, delay = None, update = None, *, hide = None))]
    fn atomic(
        vars: &Bound<'_, PyAny>,
        init: Option<&Bound<'_, PyAny>>,
        delay: Option<&Bound<'_, PyAny>>,
        update: Option<&Bound<'_, PyAny>>,
        hide: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        match (init, delay, update) {
            (Some(init), None, Some(update)) => {
                if init.is(update) {
                    Self::combinatorial(vars, init, hide)
                } else {
                    Self::sequential(vars, init, update, hide)
                }
            }
            (Some(init), Some(delay), None) => Self::differential(vars, init, delay, hide),
            (Some(init), Some(delay), Some(update)) => {
                Self::hybrid(vars, init, update, delay, hide)
            }
            (None, None, Some(update)) => Self::jump(vars, update, hide),
            (None, Some(delay), Some(update)) => Self::uninitialized(vars, update, delay, hide),
            (None, Some(delay), None) => Self::flow(vars, delay, hide),
            (Some(init), None, None) => Self::constant(vars, init, hide),
            (None, None, None) => Self::hold(vars, hide),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (vars,init, update, *, hide = None))]
    fn sequential(
        vars: &Bound<'_, PyAny>,
        init: &Bound<'_, PyAny>,
        update: &Bound<'_, PyAny>,
        hide: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let init = try_term_iter_cloned(&init)?;
        let update = try_term_iter_cloned(&update)?;
        let vars: Vec<_> = try_var_iter_cloned(vars)?.collect();

        if let Some(hide) = hide {
            let atom = reactive::Atom::sequential(named(&vars), init, update)
                .map_err(PyException::new_err)?;
            Self::partially_observable([atom], hide, Bindings::new(&vars))
        } else {
            let module = reactive::Module::sequential(named(&vars), init, update);
            Self::wrap(module, Bindings::new(&vars))
        }
    }

    #[staticmethod]
    #[pyo3(signature = (vars, assign, *, hide = None))]
    fn combinatorial(
        vars: &Bound<'_, PyAny>,
        assign: &Bound<'_, PyAny>,
        hide: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let assign = try_term_iter_cloned::<Combinatorial>(&assign)?;
        let vars: Vec<_> = try_var_iter_cloned(vars)?.collect();
        if let Some(hide) = hide {
            let atom = reactive::Atom::combinatorial(named(&vars), assign)
                .map_err(PyException::new_err)?;
            Self::partially_observable([atom], hide, Bindings::new(&vars))
        } else {
            let module = reactive::Module::combinatorial(named(&vars), assign);
            Self::wrap(module, Bindings::new(&vars))
        }
    }

    #[staticmethod]
    #[pyo3(signature = (vars, init, delay, *, hide = None))]
    fn differential(
        vars: &Bound<'_, PyAny>,
        init: &Bound<'_, PyAny>,
        delay: &Bound<'_, PyAny>,
        hide: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let init = try_term_iter_cloned(&init)?;
        let delay = try_term_iter_cloned(&delay)?;
        let vars: Vec<_> = try_var_iter_cloned(vars)?.collect();
        if let Some(hide) = hide {
            let atom = reactive::Atom::differential(named(&vars), init, delay)
                .map_err(PyException::new_err)?;
            Self::partially_observable([atom], hide, Bindings::new(&vars))
        } else {
            let module = reactive::Module::differential(named(&vars), init, delay);
            Self::wrap(module, Bindings::new(&vars))
        }
    }

    #[staticmethod]
    #[pyo3(signature = (vars, init, update, delay, *, hide = None))]
    fn hybrid(
        vars: &Bound<'_, PyAny>,
        init: &Bound<'_, PyAny>,
        update: &Bound<'_, PyAny>,
        delay: &Bound<'_, PyAny>,
        hide: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let init = try_term_iter_cloned(&init)?;
        let update = try_term_iter_cloned(&update)?;
        let delay = try_term_iter_cloned(&delay)?;
        let vars: Vec<_> = try_var_iter_cloned(vars)?.collect();
        if let Some(hide) = hide {
            let atom = reactive::Atom::hybrid(named(&vars), init, update, delay)
                .map_err(PyException::new_err)?;
            Self::partially_observable([atom], hide, Bindings::new(&vars))
        } else {
            let module = reactive::Module::hybrid(named(&vars), init, update, delay);
            Self::wrap(module, Bindings::new(&vars))
        }
    }

    #[staticmethod]
    #[pyo3(signature = (vars, update, *, hide = None))]
    fn jump(
        vars: &Bound<'_, PyAny>,
        update: &Bound<'_, PyAny>,
        hide: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let update = try_term_iter_cloned(&update)?;
        let vars: Vec<_> = try_var_iter_cloned(vars)?.collect();
        if let Some(hide) = hide {
            let atom = reactive::Atom::jump(named(&vars), update).map_err(PyException::new_err)?;
            Self::partially_observable([atom], hide, Bindings::new(&vars))
        } else {
            let module = reactive::Module::jump(named(&vars), update);
            Self::wrap(module, Bindings::new(&vars))
        }
    }

    #[staticmethod]
    #[pyo3(signature = (vars, update, delay, *, hide = None))]
    fn uninitialized(
        vars: &Bound<'_, PyAny>,
        update: &Bound<'_, PyAny>,
        delay: &Bound<'_, PyAny>,
        hide: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let update = try_term_iter_cloned(&update)?;
        let delay = try_term_iter_cloned(&delay)?;
        let vars: Vec<_> = try_var_iter_cloned(vars)?.collect();
        if let Some(hide) = hide {
            let atom = reactive::Atom::uninitialized(named(&vars), update, delay)
                .map_err(PyException::new_err)?;
            Self::partially_observable([atom], hide, Bindings::new(&vars))
        } else {
            let module = reactive::Module::uninitialized(named(&vars), update, delay);
            Self::wrap(module, Bindings::new(&vars))
        }
    }

    #[staticmethod]
    #[pyo3(signature = (vars, *, hide = None))]
    fn hold(vars: &Bound<'_, PyAny>, hide: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let vars: Vec<_> = try_var_iter_cloned(vars)?.collect();
        if let Some(hide) = hide {
            let atom = reactive::Atom::hold(named(&vars)).map_err(PyException::new_err)?;
            Self::partially_observable([atom], hide, Bindings::new(&vars))
        } else {
            let module = reactive::Module::hold(named(&vars));
            Self::wrap(module, Bindings::new(&vars))
        }
    }

    #[staticmethod]
    #[pyo3(signature = (vars, delay, *, hide = None))]
    fn flow(
        vars: &Bound<'_, PyAny>,
        delay: &Bound<'_, PyAny>,
        hide: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let delay = try_term_iter_cloned(&delay)?;
        let vars: Vec<_> = try_var_iter_cloned(vars)?.collect();
        if let Some(hide) = hide {
            let atom = reactive::Atom::flow(named(&vars), delay).map_err(PyException::new_err)?;
            Self::partially_observable([atom], hide, Bindings::new(&vars))
        } else {
            let module = reactive::Module::flow(named(&vars), delay);
            Self::wrap(module, Bindings::new(&vars))
        }
    }

    #[staticmethod]
    #[pyo3(signature = (vars, init, *, hide = None))]
    fn constant(
        vars: &Bound<'_, PyAny>,
        init: &Bound<'_, PyAny>,
        hide: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let init = try_term_iter_cloned(&init)?;
        let vars: Vec<_> = try_var_iter_cloned(vars)?.collect();
        if let Some(hide) = hide {
            let atom =
                reactive::Atom::constant(named(&vars), init).map_err(PyException::new_err)?;
            Self::partially_observable([atom], hide, Bindings::new(&vars))
        } else {
            let module = reactive::Module::constant(named(&vars), init);
            Self::wrap(module, Bindings::new(&vars))
        }
    }

    /// The parallel composition of a sequence of modules: all shared
    /// observable variables are coupled and the `hide` variables are hidden
    /// when provided.
    #[staticmethod]
    #[pyo3(signature = (*modules, hide = None))]
    fn compose(modules: &Bound<'_, PyTuple>, hide: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let modules: Vec<_> = try_iter_borrow::<Self>(modules)?.collect::<PyResult<_>>()?;
        let bindings = Bindings::merge(modules.iter().map(|r| &r.bindings));
        let modules = modules.iter().map(|r| r.base.clone());
        if let Some(hide) = hide {
            let hide = Hide::try_from(hide)?;
            let module = reactive::Module::compose_hiding(modules, hide.as_fn(&bindings));
            hide.err().and_then(|()| Self::wrap(module, bindings))
        } else {
            Self::wrap(reactive::Module::compose(modules), bindings)
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
        let names: HashMap<u32, &str> = names
            .iter()
            .map(|(var, varname)| (var::name(var.base()).name(), varname.as_str()))
            .collect();
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
        Self::wrap(module, Bindings::merge([&self.bindings, &other.bindings]))
    }

    fn hide(&self, hide: &Bound<'_, PyAny>) -> PyResult<Module> {
        let hide = Hide::try_from(hide)?;
        let module = self.base.clone().hide(hide.as_fn(&self.bindings));
        hide.err()?;
        Self::wrap(module, self.bindings.clone())
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
    fn bindings(&self) -> &Bindings {
        &self.module.get().bindings
    }

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

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        interface_eq(self.base(), other)
    }

    fn __contains__(&self, item: &Bound<'_, PyAny>) -> bool {
        interface_contains(self.base(), self.bindings(), item)
    }
}

#[pyclass(sequence)]
struct ModuleAtoms {
    module: Py<Module>,
}

#[pymethods]
impl ModuleAtoms {
    fn __getitem__(&self, index: usize) -> PyResult<Atom> {
        let module = self.module.get();
        let atoms = module.base.atoms();
        let bindings = &module.bindings;
        let result = atoms
            .get(index)
            .map(|atom| Atom::from_base(atom.clone(), bindings.clone()));
        result.ok_or(PyIndexError::new_err("index out of bounds"))
    }

    fn __len__(&self) -> usize {
        let module = &self.module.get().base;
        module.atoms().len()
    }
}
