use pyo3::prelude::*;
use theory::any::Sort;

#[pyclass(frozen, eq, hash, ord)]
#[derive(Debug, Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub(crate) struct Wire {
    base: dataflow::diagram::Wire<Sort>,
}

#[pymethods]
impl Wire {
    /// A fresh wire carrying values of the sort `dtype`; a derivative wire
    /// carries a tangent sort (e.g. a `Real` of degree >= 1), so the sort encodes the degree.
    #[new]
    pub(crate) fn new(dtype: Sort) -> Self {
        let base = dataflow::diagram::Wire::new(dtype);
        Self { base }
    }

    #[getter]
    fn id(&self) -> usize {
        self.base.id()
    }

    #[getter]
    fn dtype(&self) -> Sort {
        self.base.dtype().clone()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.base.typed())
    }
}

impl Wire {
    pub(crate) fn base(&self) -> &dataflow::diagram::Wire<Sort> {
        &self.base
    }
}

impl From<dataflow::diagram::Wire<Sort>> for Wire {
    fn from(base: dataflow::diagram::Wire<Sort>) -> Self {
        Self { base }
    }
}
