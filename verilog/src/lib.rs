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
  interpreters; [`iverilog`] simulates a design in Icarus Verilog, an
  oracle independent of everything here.
- [`ast`]: the abstract syntax of the accepted subset; [`frontend`] parses
  into it (evaluating [`constant`] expressions such as parameter values),
  [`elab`] builds the tree of module instances, [`typing`] checks each
  module, [`sizing`] makes the implicit resizes of IEEE 1800 explicit,
  [`ast_interp`] runs a hierarchy.
- [`lower`]: the translation of a hierarchy into a [`Module`]; [`print`](mod@print)
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

## Tests

`cargo test -p verilog` runs the quick tests (under a minute). The
planted-bug suite (`tests/planted.rs`: every mutant of every example,
compared by the miter in both directions) takes about two minutes and is
marked `#[ignore]`; the full run includes it:

```text
cargo test -p verilog -- --include-ignored
```

The comparisons with Yosys, z3 and iverilog are skipped with a message if
a tool is not found; with `V2RM_REQUIRE_TOOLS=1` they fail instead.
`UPDATE_GOLDEN=1` rewrites the golden files of `tests/examples.rs`.
*/

// rustdoc needs it to check auto traits of `frontend::Design`, which holds
// the deeply nested syntax tree of `sv-parser`
#![recursion_limit = "256"]

pub mod ast;
pub mod ast_interp;
pub mod constant;
pub mod elab;
pub mod frontend;
pub mod iverilog;
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

pub type Module = reactive::Module<SvOp, SvOp, SvOp, Sort>;
pub type Atom = reactive::Atom<SvOp, SvOp, SvOp, Sort>;
pub type Term = dataflow::diagram::Term<SvOp, Sort>;
pub type Var = reactive::Var<Sort>;
pub type Wire = dataflow::diagram::Wire<Sort>;
pub type Interface = reactive::Interface<Sort>;

/// A term of `op` writing `write` and reading `read`, after checking the
/// wire sorts against the op ([`SvOp::check`]). Every term of a translation
/// is built here, so a term whose sorts do not fit the op never exists.
pub fn term(
    op: SvOp,
    write: impl IntoIterator<Item = Wire>,
    read: impl IntoIterator<Item = Wire>,
) -> Result<Term, String> {
    let write: Vec<Wire> = write.into_iter().collect();
    let read: Vec<Wire> = read.into_iter().collect();
    let sorts = |ws: &[Wire]| ws.iter().map(|w| *w.dtype()).collect::<Vec<Sort>>();
    op.check(&sorts(&read), &sorts(&write))?;
    Term::function(op, write, read)
}

/// A new empty directory in the system's temporary directory, named
/// after `label`, removed with everything in it when dropped.
pub(crate) struct EmptyDir(pub(crate) std::path::PathBuf);

impl EmptyDir {
    pub(crate) fn new(label: &str) -> std::io::Result<Self> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        loop {
            let k = NEXT.fetch_add(1, Ordering::Relaxed);
            let name = format!("{label}-{}-{k}", std::process::id());
            let path = std::env::temp_dir().join(name);
            match std::fs::create_dir(&path) {
                Ok(()) => return Ok(Self(path)),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
    }
}

impl Drop for EmptyDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The binary of the tool `name`: the path in the environment variable
/// `var`, else `~/tools/oss-cad-suite/bin/NAME`, else `name` on the `PATH`
/// if it answers `NAME version_flag`.
fn find_tool(var: &str, name: &str, version_flag: &str) -> Option<std::path::PathBuf> {
    use std::path::{Path, PathBuf};
    let candidates = [
        std::env::var_os(var).map(PathBuf::from),
        std::env::var_os("HOME").map(|h| Path::new(&h).join("tools/oss-cad-suite/bin").join(name)),
    ];
    let found = candidates.into_iter().flatten().find(|p| p.exists());
    found.or_else(|| {
        let ok = std::process::Command::new(name)
            .arg(version_flag)
            .output()
            .is_ok();
        ok.then(|| PathBuf::from(name))
    })
}
