/*!
# SystemVerilog to reactive modules

Translates a synthesizable SystemVerilog subset into reactive modules over
the op set [`SvOp`]. The target semantics is 2-state, with time in steps:
one round of the module is one time step, in which the clocks and the
other inputs take new values, and a register changes only in a round
where the edge of its clock happens (see [`lower`]).

- [`theory`]: the sort [`Sort`] (a bit width) and the op set [`SvOp`].
- [`value`]: bit-vector values [`BitVec`] and the evaluator [`value::eval`].
- [`rm_sim`]: a round-by-round interpreter for modules over [`SvOp`].
- [`smt`]: sorts, values, ops and rounds of modules as SMT-LIB text; [`z3`]
  runs the solver on it; [`yosys`] translates a design into SMT-LIB
  independently, for comparison, and [`miter`] checks that both agree on
  the init round and on one time step, replaying counterexamples in both
  interpreters.
- [`ast`]: the abstract syntax of the accepted subset; [`frontend`] parses
  into it (evaluating [`constant`] expressions such as parameter values),
  [`elab`] builds the tree of module instances, [`typing`] checks each
  module, [`sizing`] makes the implicit resizes of IEEE 1800 explicit,
  [`ast_interp`] runs a hierarchy.
- [`lower`]: the translation of a hierarchy into a [`Module`]; [`print`]
  prints modules with names.
*/

// rustdoc needs it to check auto traits of `frontend::Design`, which holds
// the deeply nested syntax tree of `sv-parser`
#![recursion_limit = "256"]

pub mod ast;
pub mod ast_interp;
pub mod constant;
pub mod elab;
pub mod frontend;
pub mod lower;
pub mod miter;
pub mod print;
pub mod rm_sim;
pub mod sizing;
pub mod smt;
pub mod theory;
pub mod typing;
pub mod value;
pub mod yosys;
pub mod z3;

pub use crate::theory::{Sort, SvOp};
pub use crate::value::BitVec;

pub type Module = base::Module<SvOp, SvOp, SvOp>;
pub type Atom = base::Atom<SvOp, SvOp, SvOp>;
pub type Term = base::Term<SvOp>;
pub type Var = base::Var<Sort>;
pub type Wire = base::Wire<Sort>;
