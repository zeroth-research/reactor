use base::tangent::Tangent;
use pyo3::prelude::*;
use reactive::var::PREFIX;
use std::sync::atomic::{AtomicU32, Ordering};
use theory::any::Sort;

/// The next name a fresh python variable takes.
static NEXT_NAME: AtomicU32 = AtomicU32::new(0);

/// A variable: a name, and the sort of the values it holds.
///
/// A variable holds no wires. The blocks of an atom take the values of its
/// variables at the positions of their boundaries; the variable itself
/// stands for its latched value, and [`x`] and [`d`] give its next value and
/// its derivative.
#[pyclass(frozen, eq, hash, ord)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct Var {
    base: reactive::Var<Sort>,
}

#[pymethods]
impl Var {
    /// A fresh variable of the sort `dtype`, under a name no other variable
    /// has.
    #[new]
    pub(crate) fn new(dtype: Sort) -> Self {
        let name = NEXT_NAME.fetch_add(1, Ordering::Relaxed);
        Self {
            base: reactive::Var::new(name, dtype),
        }
    }

    #[getter]
    fn name(&self) -> u32 {
        self.base.name()
    }

    #[getter]
    fn dtype(&self) -> Sort {
        *self.base.dtype()
    }

    fn __repr__(&self) -> String {
        format!("{PREFIX}{}: {}", self.base.name(), self.base.dtype())
    }
}

impl Var {
    pub(crate) fn base(&self) -> &reactive::Var<Sort> {
        &self.base
    }
}

impl From<reactive::Var<Sort>> for Var {
    fn from(base: reactive::Var<Sort>) -> Self {
        Self { base }
    }
}

/// The value of a variable a view stands for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum Kind {
    Next,
    Derivative,
}

/// A view of a variable: its next value, or its derivative. A view is no
/// wire: it names the value a block takes or gives at a position of its
/// boundary.
#[pyclass(frozen, eq, hash, ord)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct View {
    var: Var,
    kind: Kind,
}

#[pymethods]
impl View {
    /// The variable viewed.
    #[getter]
    fn var(&self) -> Var {
        self.var.clone()
    }

    /// The sort of the value viewed: the sort of the variable for the next
    /// value, its tangent for the derivative.
    #[getter]
    fn dtype(&self) -> Sort {
        let dtype = self.var.base.dtype();
        match self.kind {
            Kind::Next => *dtype,
            Kind::Derivative => dtype.tangent(),
        }
    }

    /// Whether the view is the next value of the variable.
    #[getter]
    fn next(&self) -> bool {
        self.kind == Kind::Next
    }

    /// Whether the view is the derivative of the variable.
    #[getter]
    fn derivative(&self) -> bool {
        self.kind == Kind::Derivative
    }

    fn __repr__(&self) -> String {
        let name = self.var.base.name();
        match self.kind {
            Kind::Next => format!("X({PREFIX}{name})"),
            Kind::Derivative => format!("d({PREFIX}{name})"),
        }
    }
}

/// The next value of the variable.
#[pyfunction]
#[pyo3(name = "X")]
pub(crate) fn x(var: Var) -> View {
    View {
        var,
        kind: Kind::Next,
    }
}

/// The derivative of the variable.
#[pyfunction]
pub(crate) fn d(var: Var) -> View {
    View {
        var,
        kind: Kind::Derivative,
    }
}

impl View {
    pub(crate) fn var_ref(&self) -> &Var {
        &self.var
    }
}
