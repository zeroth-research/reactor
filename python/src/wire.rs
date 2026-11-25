use pyo3::prelude::*;
use theory::matrix::sort;

#[pyclass(module = "reactor", frozen, eq, hash, ord)]
#[derive(Debug, Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub(crate) struct Wire {
    base: dataflow::diagram::Wire<sort::Any>,
}

#[pymethods]
impl Wire {
    #[new]
    pub(crate) fn new(dtype: sort::Any) -> Self {
        let base = dataflow::diagram::Wire::new(dtype);
        Self { base }
    }

    #[getter]
    fn id(&self) -> usize {
        self.base.id()
    }

    #[getter]
    fn dtype(&self) -> sort::Any {
        self.base.dtype().clone()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.base.typed())
    }
}

impl Wire {
    pub(crate) fn base(&self) -> &dataflow::diagram::Wire<sort::Any> {
        &self.base
    }
}

impl From<dataflow::diagram::Wire<sort::Any>> for Wire {
    fn from(base: dataflow::diagram::Wire<sort::Any>) -> Self {
        Self { base }
    }
}
