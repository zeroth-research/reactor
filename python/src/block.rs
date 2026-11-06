use crate::term::Term;
use crate::wire::Wire;
use crate::{try_term_iter_cloned, try_wire_iter_cloned};
use pyo3::exceptions::{PyException, PyIndexError};
use pyo3::prelude::*;
use theory::any::Any;

/// A block: terms computing the outputs `write` from the inputs `read`, both
/// in order (see `dataflow::diagram::Block::new`).
#[pyclass(frozen, sequence)]
#[derive(Clone)]
pub(crate) struct Block {
    base: dataflow::diagram::Block<Any>,
}

#[pymethods]
impl Block {
    /// The block with the inputs `read` and the outputs `write` of the
    /// terms. Every input is read (an input the block has no use for is
    /// discarded by a `DISCARD` term), and every output written exactly once.
    #[new]
    fn new(
        read: &Bound<'_, PyAny>,
        write: &Bound<'_, PyAny>,
        terms: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let read = try_wire_iter_cloned(read)?.collect::<PyResult<Vec<_>>>()?;
        let write = try_wire_iter_cloned(write)?.collect::<PyResult<Vec<_>>>()?;
        let terms = try_term_iter_cloned::<Any>(terms)?.collect::<PyResult<Vec<_>>>()?;
        dataflow::diagram::Block::new(read, write, terms)
            .map(|base| Self { base })
            .map_err(PyException::new_err)
    }

    /// The inputs of the block, in order.
    #[getter]
    fn read(&self) -> Vec<Wire> {
        self.base.read().iter().cloned().map(Wire::from).collect()
    }

    /// The outputs of the block, in order.
    #[getter]
    fn write(&self) -> Vec<Wire> {
        self.base.write().iter().cloned().map(Wire::from).collect()
    }

    fn __getitem__(&self, index: usize) -> PyResult<Term> {
        self.base
            .get(index)
            .cloned()
            .map(Term::from)
            .ok_or(PyIndexError::new_err("index out of bounds"))
    }

    fn __len__(&self) -> usize {
        self.base.len()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.base)
    }
}

impl Block {
    /// The block over the theory `T`, or the error naming the first term
    /// not in `T`.
    pub(crate) fn narrow<T>(&self) -> PyResult<dataflow::diagram::Block<T>>
    where
        T: theory::Theory<Sort = theory::any::Sort>,
        dataflow::diagram::Term<Any>: TryInto<dataflow::diagram::Term<T>, Error = String>,
    {
        let terms = self.base.iter().cloned().map(TryInto::try_into);
        let terms = terms.collect::<Result<Vec<_>, String>>();
        let read = self.base.read().to_vec();
        let write = self.base.write().to_vec();
        terms
            .and_then(|terms| dataflow::diagram::Block::new(read, write, terms))
            .map_err(PyException::new_err)
    }
}

impl<T> From<dataflow::diagram::Block<T>> for Block
where
    T: theory::Theory<Sort = theory::any::Sort> + Into<Any>,
{
    fn from(block: dataflow::diagram::Block<T>) -> Self {
        Self {
            base: dataflow::diagram::Block::convert(block),
        }
    }
}
