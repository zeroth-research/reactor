use pyo3::prelude::*;
use theory::matrix;

#[pyclass(module = "reactor", frozen, eq, hash, ord)]
#[derive(Debug, Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub(crate) struct Wire {
    base: dataflow::diagram::Wire<matrix::Any>,
}

#[pymethods]
impl Wire {
    #[new]
    pub(crate) fn new(dtype: matrix::Any) -> Self {
        let base = dataflow::diagram::Wire::new(dtype);
        Self { base }
    }

    #[getter]
    fn id(&self) -> usize {
        self.base.id()
    }

    #[getter]
    fn dtype(&self) -> matrix::Any {
        self.base.dtype().clone()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.base.typed())
    }
}

impl Wire {
    pub(crate) fn base(&self) -> &dataflow::diagram::Wire<matrix::Any> {
        &self.base
    }
}

impl From<dataflow::diagram::Wire<matrix::Any>> for Wire {
    fn from(base: dataflow::diagram::Wire<matrix::Any>) -> Self {
        Self { base }
    }
}
