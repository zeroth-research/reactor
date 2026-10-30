/*!
# SystemVerilog to reactive modules

Translates a synthesizable SystemVerilog subset into reactive modules over
the op set [`SvOp`]. The target semantics is 2-state and cycle-level: one
round of the module is one clock cycle.

- [`theory`]: the sort [`Sort`] (a bit width) and the op set [`SvOp`].
- [`value`]: bit-vector values [`BitVec`] and the evaluator [`value::eval`].
- [`rm_sim`]: a round-by-round interpreter for modules over [`SvOp`].
*/

pub mod rm_sim;
pub mod theory;
pub mod value;

pub use crate::theory::{Sort, SvOp};
pub use crate::value::BitVec;

pub type Module = base::Module<SvOp, SvOp, SvOp>;
pub type Atom = base::Atom<SvOp, SvOp, SvOp>;
pub type Term = base::Term<SvOp>;
pub type Var = base::Var<Sort>;
pub type Wire = base::Wire<Sort>;
