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

## The boundary of a design: blackboxes

Real designs instantiate modules whose source is not part of the design:
vendor primitives of an FPGA (clock generators, I/O cells, block RAMs),
hard macros of an ASIC, IP cores delivered as netlists. To translate such a
design, each of these modules is given as a *stub*: a module with its ports
(and parameters, if instances set them) but no body, marked
`(* blackbox *)`, which is how Yosys marks such modules too. Stubs usually
live in a file of their own, translated together with the design (the
command-line tool takes several files).

A blackbox is the boundary between the design and its environment. The
translation does not model its behaviour: the signals on its outputs take
any value in every time step, exactly like inputs of the top module (an
output used as a clock is an input clock), and nothing reads the signals on
its inputs. Its `inout` ports may be connected to `inout` ports (pins) of
the top module, which are then outside the design as well. This is a sound
over-approximation: every behaviour of the real cell is among those of the
translation, so a safety property proved of the translation holds for the
design with any implementation of the cell. What it loses is what the cell
guarantees (a clock generator's frequency, an I/O cell's sampling of its
pin), so a property that depends on that may fail on traces the real cell
never produces. The remedy is a refinement, not a change of the
translation: replace the stub by a model of the cell written in the
accepted subset, or compose the translation with a module of the
environment that drives the blackbox outputs. Every instance of a blackbox
is reported as a warning ([`lower::Translation::blackboxes`]), with the
signals it leaves free, so the abstraction is never silent; a missing module
is an error whose message points to stubs. See [`lower`] ("Blackboxes and
pins") for the translation and [`elab`] for the rules on connections.
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
