use crate::wire::Wire;
use pyo3::BoundObject;
use pyo3::basic::CompareOp;
use pyo3::prelude::*;
use pyo3::types::PyNotImplemented;
use std::collections::BTreeMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;
use theory::any::Sort;

#[pyclass(frozen)] // eq, hash, ord are implemented manually for compat with polymorphic __richcmp__
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub(crate) struct Var {
    base: reactive::Wire3<Sort>,
}

#[pymethods]
impl Var {
    #[new]
    pub(crate) fn new(dtype: Sort) -> Self {
        let base = reactive::Wire3::new(dtype);
        Self { base }
    }

    /// Attribute fallback to the latched wire, mirroring Rust's `Deref`:
    /// any attribute `Var` does not define itself is answered by the wire
    /// (`id`, `dtype`, ...).
    fn __getattr__(slf: PyRef<'_, Self>, name: &str) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let wire: Wire = base::Wire::from(*slf.base()).into();
        wire.into_pyobject(py)?.getattr(name).map(Into::into)
    }

    /// Comparison through the latched wire, consistent with the deref view:
    /// a variable compares as its latched wire, against variables and wires
    /// alike, under the wires' total order.
    fn __richcmp__<'py>(
        &self,
        other: &Bound<'py, PyAny>,
        op: CompareOp,
        py: Python<'py>,
    ) -> PyResult<Borrowed<'py, 'py, PyAny>> {
        let ordering = if let Ok(var) = other.extract::<PyRef<Var>>() {
            self.base.cmp(&var.base)
        } else if let Ok(wire) = other.extract::<PyRef<Wire>>() {
            let ltc: &base::Wire<Sort> = &self.base;
            ltc.cmp(wire.base())
        } else {
            return Ok(PyNotImplemented::get(py).into_any());
        };

        Ok(op.matches(ordering).into_pyobject(py)?.into_any())
    }

    /// Hashing matches equality: the latched wire is the bearing element,
    /// so a variable and its latched wire hash alike and are
    /// interchangeable as dictionary keys.
    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.base.hash(&mut hasher);
        hasher.finish()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.base.typed())
    }
}

impl Var {
    pub(crate) fn base(&self) -> &reactive::Wire3<Sort> {
        &self.base
    }
}

/// For keying purposes, a python `Var` is its base var: the borrowed view
/// hashes and compares identically (the wrapper's `Hash`/`Eq`/`Ord` all
/// delegate to the base), so maps keyed by `Var` can be probed with a
/// `&reactive::Wire3<Sort>` directly.
impl std::borrow::Borrow<reactive::Wire3<Sort>> for Var {
    fn borrow(&self) -> &reactive::Wire3<Sort> {
        &self.base
    }
}

impl From<reactive::Wire3<Sort>> for Var {
    fn from(base: reactive::Wire3<Sort>) -> Self {
        Self { base }
    }
}

/// The next wire of the variable.
#[pyfunction]
#[pyo3(name = "X")]
pub(crate) fn x(var: PyRef<'_, Var>) -> Wire {
    reactive::var::X(&var.base).into()
}

/// The derivative wire of the variable.
#[pyfunction]
pub(crate) fn d(var: PyRef<'_, Var>) -> Wire {
    reactive::var::d(&var.base).into()
}

/// The name of a python variable: the id of its latched wire, which no other
/// variable shares. Displaying a name as a wire id (`#id`) thus displays the
/// latched wire.
pub(crate) fn name(var: &reactive::Wire3<Sort>) -> reactive::Var<Sort> {
    let id = u32::try_from(var.id()).expect("wire ids exceed the range of names");
    reactive::Var::new(id, *var.dtype())
}

/// The python variables bound to their names, as the base constructors take
/// them.
pub(crate) fn named(
    vars: &[reactive::Wire3<Sort>],
) -> impl Iterator<Item = (reactive::Var<Sort>, &reactive::Wire3<Sort>)> {
    vars.iter().map(|var| (name(var), var))
}

/// The python variables an atom or a module is built from, by name. The base
/// knows the variables of atoms and modules by their names only; the
/// bindings give back the python variables, with their wires.
#[derive(Clone, Default)]
pub(crate) struct Bindings(Arc<BTreeMap<u32, reactive::Wire3<Sort>>>);

impl Bindings {
    pub(crate) fn new<'a, V>(vars: V) -> Self
    where
        V: IntoIterator<Item = &'a reactive::Wire3<Sort>>,
    {
        Self(Arc::new(
            vars.into_iter()
                .map(|var| (name(var).name(), *var))
                .collect(),
        ))
    }

    /// The bindings of all the given bindings.
    pub(crate) fn merge<'a, B>(all: B) -> Self
    where
        B: IntoIterator<Item = &'a Bindings>,
    {
        let vars = all.into_iter().flat_map(|b| b.0.values());
        Self::new(vars)
    }

    /// The wires of the variable of the name `var`.
    ///
    /// # Panics
    /// If no variable of the name is bound: the bindings of an atom or a
    /// module bind all the variables of its interfaces.
    pub(crate) fn wires(&self, var: &reactive::Var<Sort>) -> &reactive::Wire3<Sort> {
        self.0
            .get(&var.name())
            .unwrap_or_else(|| panic!("unbound variable #{}", var.name()))
    }

    /// The python variable of the name `var` (see [`Bindings::wires`]).
    pub(crate) fn var(&self, var: &reactive::Var<Sort>) -> Var {
        Var::from(*self.wires(var))
    }
}
