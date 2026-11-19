//! The terms, atoms and modules over the operations holding JAX arrays: the
//! classes `Term`, `Atom` and `Module` of the namespace `jax`.
// pyo3's `#[pymethods]`, expanded by the macros below, makes the slots of
// binary operators (`__eq__`, `__mul__`) unsafe functions that call unsafe
// functions outside `unsafe` blocks, which edition 2024 warns of.
#![allow(unsafe_op_in_unsafe_fn)]

use crate::var::Var;
use crate::wire::Wire;
use crate::{
    Hide, try_iter_borrow, try_var_iter_cloned, try_wire_iter_cloned, wires_untyped_to_repr,
};
use pyo3::exceptions::{PyException, PyIndexError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::borrow::Cow;
use std::collections::HashMap;
use theory::tensor::sort;

/// The operations of all theories, with their tensors held as `jax.Array`s.
pub(crate) type Any = theory::tensor::Any<theory::jax::Array>;

crate::term::term_classes!("reactor.jax");
crate::atom::atom_classes!();
crate::module::module_classes!();
