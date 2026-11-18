//! The logics, as Python classes in `reactor.op`: `NRA`, `LRA`, `NIA`,
//! `LIA`, `Prop` and `BV`, each of which names the logic of the same name in
//! `theory::op`. They select the theory of a frontend, as in
//! `builder_for(LIA)`, and tell which operations are in their logic, as in
//! `LRA.contains(op.real.Mul(2, 3))`, which is false.

use pyo3::prelude::*;
use theory::op;

macro_rules! logic {
    ($($(#[$doc:meta])* $logic:ident),* $(,)?) => {
        $(
            $(#[$doc])*
            #[allow(clippy::upper_case_acronyms)]
            #[pyclass(frozen, module = "reactor.op")]
            pub(crate) struct $logic;

            #[pymethods]
            impl $logic {
                /// Whether `op` is an operation of the logic.
                #[staticmethod]
                fn contains(op: crate::Any) -> bool {
                    op::$logic::<theory::PyTensor>::try_from(op).is_ok()
                }
            }
        )*

        /// The classes of the logics, to add to a Python module.
        pub(crate) fn add_classes(module: &Bound<'_, PyModule>) -> PyResult<()> {
            $(module.add_class::<$logic>()?;)*
            Ok(())
        }
    };
}

logic! {
    /// Non-linear real arithmetic, `QF_NRA`: the operations of `real`, with
    /// those of `bool`.
    NRA,
    /// Linear real arithmetic, `QF_LRA`: the linear operations of `real`,
    /// with those of `bool`.
    LRA,
    /// Non-linear integer arithmetic, `QF_NIA`: the operations of `int`, with
    /// those of `bool`.
    NIA,
    /// Linear integer arithmetic, `QF_LIA`: the linear operations of `int`,
    /// with those of `bool`.
    LIA,
    /// Propositional logic: the operations of `bool`.
    Prop,
    /// Fixed-size bit-vectors, `QF_BV`: the operations of `word`, with those
    /// of `bool`.
    BV,
}
