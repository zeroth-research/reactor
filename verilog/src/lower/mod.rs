/*!
# From the AST to a reactive module

Every signal becomes a variable of the width of its type: inputs are
external, outputs are interface variables, internal signals are private.
Each module is first sized ([`sizing`]): afterwards the operands of every
operator have one type and every assignment copies a value of its target's
width, so the terms below need no width rules of their own.

Every `assign` becomes a combinatorial atom that awaits the next values
`X(·)` of the signals it reads and writes the next values of its targets,
in every round. The expression becomes
one term per operator, with the source operators rewritten into [`SvOp`],
which has no `>`, `>=` and no logical operators:

| source | ops |
|---|---|
| `a > b`, `a >= b` | `ULt`/`SLt`, `ULe`/`SLe` with swapped operands |
| `<`, `<=` | `ULt`/`ULe`, or `SLt`/`SLe` for signed operands |
| `!a`, `a && b`, `a \|\| b` | `Not`/`And`/`Or` on `RedOr` of the operands (none for 1 bit) |
| `c ? a : b` | `Ite` on `RedOr c` (none for 1 bit) |
| `~^`, `~&`, `~\|` | `Not` of `Xor`, `RedAnd`, `RedOr` |
| `===`, `!==` | `Eq`, `Ne` |
| unary `+`, `$signed`, `$unsigned`, `signed'`, `unsigned'` | nothing |
| `/`, `%` | `UDiv`/`URem`, or `SDiv`/`SRem` for signed operands |
| `>>>` | `AShr` for a signed operand, else `LShr` |
| `{a, b, c}` | `Concat(a, Concat(b, c))` |
| `{n{e}}` | `n` copies of `e`'s wire, nested as above |
| `x[h:l]`, `x[c]`, `x[c +: w]`, … (constant) | `Extract`, nothing for all bits |
| `x[i]`, `x[i +: w]` (variable) | `Extract{w-1, 0}(LShr(x, i))`: bits beyond the top read 0 |
| `x[i -: w]` (variable, `w > 1`) | the same on `{x, (w-1)'0}`: bits below bit 0 read 0 too |
| `m[i]` (element of an array, variable index) | a tree of `Ite`s on the bits of `i`, from bit 0, over the elements `m[0]`, `m[1]`, … |
| target `m[i] = e` (element, variable index) | for each element `m[k]`: `m[k] := Ite(Eq(i, k), new, m[k])`, with `new` the element after the assignment |
| target `x[i] = e`, `x[i +: w] = e` (variable) | `x := Or(And(x, Not(Shl(ones, i))), Shl(ZExt(e), i))`, `ones` = `w` one bits |
| target `x[i -: w] = e` (variable) | the same on `{x, (w-1)'0}`, then `Extract` of the top bits |
| `N'(e)` | `SExt`/`ZExt` (as `e`'s signedness says) when widening, `Extract` when narrowing |
| an implicit resize (from sizing) | `SExt`/`ZExt` as the type it resizes to says, `Extract` when narrowing |
| `y = 'x;`, `q <= 'x;` (overwritten on every path, checked by typing) | nothing |
| `q = e;` in an `always_ff` (its value never read later in the block, nor by another block at the same clock edge, checked by typing) | as `q <= e;` |
| `(* parallel_case *) case …` (distinct constant labels, checked by sizing) | as without the attribute |

A read with a variable index may select bits outside its signal; they read
0. That is the standard's value for a 2-state signal and what Yosys builds;
for a 4-state `logic` or `reg` the standard gives x, which synthesis may
resolve to any value, so the translation is exact for such reads only
under synthesis tools that choose 0 (see [`typing::may_leave`]; the reads
are listed in [`Translation::out_of_range`] and the CLI warns about each).

## Time and clocks

One round of the module is one time step. An input clock is a 1-bit input
like any other: an external variable, whose level the environment chooses
freely in every round, as it chooses the other inputs. So rounds can
contain any interleaving of the edges of several clocks, edges of
different clocks in the same round, and rounds without any edge; a design
clocked by one clock that toggles in every round takes two rounds per
cycle. A derived clock is a register (or a copy or bit of one, see
[`typing::domains`]), e.g. a divider `slow <= !slow` clocked by `clk`: a
variable like any other register, which changes in the rounds where its
own clock has its edge. The edge of a clock `c` happens in a round when
its latched level `*c` and its next level `X(c)` are 0 and 1 (`posedge`),
or 1 and 0 (`negedge`). So an edge of `slow` happens in the same round as
the edge of `clk` that changes it. No edge happens in the first round, the
init round.

Every `always_ff` block becomes a sequential atom controlling the
registers it assigns. Its update first tests the edge of its clock,
`clk@pos := AND (NOT clk, X(clk))` (`clk@neg := AND (clk, NOT X(clk))`),
then computes the value after the edge, and finally sets each register `q`
to `X(q) := ITE (clk@pos, new, q)`: the new value if the edge happens,
else the old one. The value after the edge reads the *latched* values
`*x`, the values of the previous round, just before the edge. So the
update awaits only the next level of the clock (for a derived clock, after
the atom of its register), and a register closes a combinational loop
only through its clock: a register clocked by itself, directly or through
copies of it and other registers, is rejected as one (the edge test needs
the new level of the clock in the same round). The statements are executed symbolically
with a map from each register to the wire holding its value after the
edge so far, starting with `*q` (a register keeps its value on paths that
do not assign it). `q <= e` sets the entry of `q` to the wire of `e`, so
the last assignment wins; at `if`, both branches start from the same map
and the bits that differ are merged with `Ite`. The init action sets a
register with an initial value (`logic q = c;`, `initial q = c;`) to it,
`X(q) := c`, as hardware that keeps initial values does at power-on
(FPGAs load them with the configuration). Every other register gets the
value of an external variable `q@init`, `X(q) := ID X(q@init)`, so all
its initial values are possible.

Reading values just before the edge is what hardware does under two
timing assumptions, which this semantics makes and does not check:

- data inputs meet the setup and hold times of the registers reading
  them: an input does not change in the round where an edge samples it
  (it takes its new value in the round, but the edge sees the latched
  one). This cannot hold for a block's own clock, which changes at the
  very edge that runs the block, nor for the chain its clock comes from,
  so a block may not read them ([`typing::domains`]);
- at an edge, the registers of one clock (on either edge) and the signals
  computed from them change only after the edge, as their clock-to-output
  delay makes them do; so a register clocked by `negedge clk` reading one
  clocked by `posedge clk` sees its value from before that edge;
- the same holds between a clock and the clocks derived from it: an edge
  of a derived clock comes a clock-to-output delay after the edge of its
  source that causes it, at the same time as the other registers of the
  source change. A register on the derived clock samples their values
  from before that edge if the design meets its hold times on these paths,
  which timing analysis checks when the derived clock is declared as a
  generated clock. So a register clocked by `posedge slow` reading a
  counter clocked by `posedge clk` sees the counter's value from before
  the edge of `clk` that makes `slow` rise. (A simulator runs the blocks
  of `slow` after the updates of `clk`'s blocks and gives the new value
  instead; real hardware with met hold times does not.)

The two edges of one clock never happen in the same round. Registers of
one input clock and of the clocks derived from it are related; registers
of different input clocks are unrelated: an edge of one may come at any time
relative to the other, so a register sampling a value from another clock
domain in a round where that value changes could store any mix of old and
new bits (metastability). Such reads across clock domains are rejected for
now ([`typing::domains`]); so are clocks computed by logic (gated clocks)
and clock loops. An asynchronous reset from another domain on the
register of a derived clock makes edges of that clock at a time unrelated
to its source, so blocks on such an edge must not read other domains
either (see [`typing::domains`]).

## Asynchronous resets

A block `always_ff @(posedge clk or posedge rst) if (rst) q <= c; else …`
(or `negedge rst_n` with `if (!rst_n)`) has an asynchronous reset: a
flip-flop whose reset input acts at once, as a level, not at an edge. The
reset is active in a round when its next level `X(rst)` is 1 (`X(rst_n)`
is 0). While it is active, every register the reset branch assigns has its
reset value, whether or not a clock edge happens:

```text
X(q) := ITE (X(rst), c, ITE (clk@pos, new, q))     update
X(q) := ITE (X(rst), c, X(q@init))                 init
```

(with an initial value `v`, the init action is `ITE (X(rst), c, v)`)

so the atom also awaits the reset (and a register reset by a signal
computed from itself is a loop). The value after the edge, `new`, comes
from the body as written, `if (rst)` included, which reads the latched
reset like any other value. So a clock edge in the round where the reset
is released sees the reset still active and keeps the reset values:
releasing the reset takes effect at the next edge. This is the timing
assumption of resets: the release meets the recovery and removal times of
the registers (it does not change too close to an edge), just as data
inputs meet setup and hold. A reset asserted in a round where a clock has
an edge acts after that edge: the registers it resets end the round at
their reset values, while every block running at an edge of that round
reads their values from before the reset (the latched ones), as it reads
all registers; a reset asserted just before an edge in hardware is a reset
in an earlier round here. The same holds for registers clocked by an edge
that a reset makes (the reset of a derived clock's register, see
[`typing::domains`]): they change in the round of the reset, and blocks on
other clocks see the change from their next edge on. When the reset comes
from another clock domain, these registers and the reset registers
themselves change at a time unrelated to the clocks of the blocks reading
them, which could sample them during the change; this is assumed not to
happen (the reset is synchronized to the readers' clocks or asserted away
from their edges) and is not checked. Registers of the block that the reset branch
does not assign are not reset; they keep their value while the reset is
active, as plain registers whose value at an edge depends on the latched
reset. A reset must come from an input or a register, possibly through
copies (see [`typing::domains`]): a reset computed by logic could glitch
within a time step and reset registers, which rounds cannot express.

## Combinational blocks, instances and loops

Every `always_comb` block becomes a combinatorial atom controlling the
signals it assigns. Like an `assign`, it awaits the next values `X(x)`
of the signals it reads, except its own: the statements are executed
symbolically with a map from each of its signals to the wire holding its
current value (SSA, the intermediate values are printed `y@1`, `y@2`, …),
so a read of its own signal reads that wire. The map starts empty; typing
has made sure that every read and the end of the block find an entry on
every path (no stale reads, no latches). `if` and `case` merge entries as in
`always_ff`; an entry missing in a branch is missing after the merge.

Assignment targets may be selects and concatenations (`{c, s} = …`,
`y[3:0] = …`), so the map holds the value of each signal as segments: runs
of bits that are bits of some wire, or not assigned yet. An assignment
replaces the segments of the bits it assigns; the bits a target leaves
alone keep their segments (in `always_ff`, bits of `*q`, so they keep their
value). A read of bits of a signal of the block, and the final value,
concatenate the segments; a merge puts an `Ite` on each run of bits that
differs between the branches.

The elements of an array are signals like any other, so an array with
`N` elements is `N` variables `m[0]`, … (none for a ROM, whose elements are
constants). An element read with a variable index picks among them with a
tree of `Ite`s on the bits of the index, and an assignment to one updates
each element the index can pick, with `Ite` on whether the index is its
address (an index that is no address writes nothing). The block owning the
array therefore controls all its elements, and a block reading `m[i]`
reads or awaits all of them.

`case (s) l1, l2: a … default: d endcase` becomes a chain of `Ite`s on
`s == l1 | s == l2`, …, so the first matching item wins; without `default`,
no item matching leaves the map as it was.

A module with instances becomes the composition of a module made of its
own atoms and the translations of its instances (each translated on its
own, with fresh variables), with the module's internal signals hidden.
A port connected to a signal name is that signal's variable (so a clock
port is the clock of the instantiating module, and the instance's blocks
test its edges themselves); an input connected to another expression is a variable `u.a` driven by an extra
atom of the module, as if by `assign u.a = …` (so the expression is sized
as the right-hand side of an assignment to the port, see [`sizing`]; Yosys
sizes it by itself). Variables are named as
[`Inst::naming`] says, the initial value of a register at path `u.q` is the
external variable `u.q@init`, and intermediate values of an instance's
blocks are prefixed with its path.

The atoms are sorted so that each comes after the atoms controlling what it
awaits; a cycle is a combinational loop and is rejected. The check is per
atom, as the await relation of reactive modules is: an atom's update
is one action, so it waits for everything any of its terms reads. An
`always_comb` block driving several signals may therefore close a loop
that its data flow does not have (it drives `x` from `a` and `y` from `z`,
while another block drives `z` from `x`). Such designs are rejected, with
a message saying which block to split. Translating each signal of a block
as its own atom would accept them with the same meaning, at the cost of
duplicating shared subterms and of the one-to-one correspondence between
blocks and atoms. The check runs on names, not paths: a port connected to
a signal name is that signal. A loop through instances is found on the
atoms of the composed modules and reported with the signals on it.

## Blackboxes and pins

A blackbox (a module marked `(* blackbox *)`, with ports and parameters but
no body) is the boundary of the design: its behaviour belongs to the
environment, as a vendor cell (a PLL, an I/O cell) whose meaning the source
does not give. An instance of it adds no atoms and no variables of its own.
The signal connected to one of its outputs has no driver in the module, so
it is an external variable, exactly like an input of the top module: it
may take any value in every round (a sound over-approximation of every
behaviour of the cell), it is in no clock domain, and it may be an input
clock (the output of a PLL). Its inputs are read by nothing: what drives
them stays in the module as ordinary signals. A pin (an `inout` port of the
top module) is only connected to `inout` ports of blackboxes, so it lies
outside the design as well and is no variable. A model of a cell, if a
property needs one, is a refinement of this: a module composed with the
translation that drives the external variables of the blackbox outputs.

[`sizing`]: crate::sizing
[`SvOp`]: crate::SvOp
*/

use crate::ast::{self, Dir, Error, ErrorKind, Expr, ExprKind, Name, Span, Ty, Warning};
use crate::elab::{Inst, Path};
use crate::typing;
use crate::value::BitVec;
use crate::{Module, Var, Wire};
use std::collections::{BTreeMap, HashMap};

mod builder;
mod hierarchy;
mod order;
mod segments;

use hierarchy::{Out, lower_rec};

type Result<T> = std::result::Result<T, Error>;

/// A module with the names of its variables.
pub struct Translation {
    pub name: Name,
    pub module: Module,
    /// Every source signal, and `q@init` for every register `q` without an
    /// initial value, by name.
    pub vars: Vars,
    /// The name in the module of every signal path of the design (see
    /// [`Inst::naming`]): a port of an instance connected to a signal of the
    /// parent of the same width is that signal (`u.clk` → `clk`).
    pub paths: BTreeMap<Name, Name>,
    /// The input clocks of the whole design: the inputs of the top module
    /// that the `always_ff` blocks are clocked by, directly or through
    /// derived clocks, by name, sorted.
    pub clocks: Vec<Name>,
    /// The signals assigned in `always_ff` blocks, in source order.
    pub registers: Vec<Name>,
    /// The registers with an initial value (so without `q@init`), by
    /// path, in source order.
    pub initialized: Vec<Name>,
    /// The width of every register, by path.
    pub register_widths: BTreeMap<Name, usize>,
    /// The registers with an asynchronous reset (those the reset branch of
    /// their block assigns), by path, in source order.
    pub reset_registers: Vec<Name>,
    /// Inputs that nothing reads; they are not variables of the module.
    pub unread_inputs: Vec<Name>,
    /// The signals connected to outputs of blackboxes, by name (see
    /// [`Inst::blackbox_outputs`]): free in every round, like inputs. Those
    /// that something reads are external variables.
    pub blackbox_outputs: Vec<Name>,
    /// Reads with a variable index that may select bits outside their
    /// signal, located (with the instance path), sorted by position:
    /// those bits read 0, which for a 4-state signal is one choice for the
    /// standard's x (see [`typing::may_leave`]).
    pub out_of_range: Vec<Warning>,
    /// One located warning per instance of a blackbox (with the instance
    /// path), in order: the signals on its outputs take any value in every
    /// round and nothing reads its inputs, since its behaviour is not part
    /// of the design (see the module docs, "Blackboxes and pins").
    pub blackboxes: Vec<Warning>,
    /// Names of wires local to an atom: the intermediate values `y@1`,
    /// `y@2`, … of the signals of `always_comb` blocks, and the edge tests
    /// `clk@pos`, `clk@neg` of `always_ff` blocks (`clk@pos#2` for the
    /// second block of a module on the same edge, so that names are unique
    /// in the whole design).
    pub wire_names: HashMap<Wire, Name>,
    /// The module of every instance before it is composed into its parent,
    /// children before their parent; with the names of the whole design.
    pub instances: Vec<Part>,
}

/// The variables of a translation by name (a map from names to variables,
/// which it derefs to), and the name of each variable.
#[derive(Clone, Default)]
pub struct Vars {
    by_name: BTreeMap<Name, Var>,
    names: HashMap<Var, Name>,
}

impl Vars {
    /// The name of `var`, if it is one of the variables.
    pub fn name(&self, var: &Var) -> Option<&str> {
        self.names.get(var).map(String::as_str)
    }
}

impl From<BTreeMap<Name, Var>> for Vars {
    fn from(by_name: BTreeMap<Name, Var>) -> Self {
        let names = by_name.iter().map(|(n, v)| (*v, n.clone())).collect();
        Self { by_name, names }
    }
}

impl std::ops::Deref for Vars {
    type Target = BTreeMap<Name, Var>;

    fn deref(&self) -> &Self::Target {
        &self.by_name
    }
}

impl<'v> IntoIterator for &'v Vars {
    type Item = (&'v Name, &'v Var);
    type IntoIter = std::collections::btree_map::Iter<'v, Name, Var>;

    fn into_iter(self) -> Self::IntoIter {
        self.by_name.iter()
    }
}

/// The module of an instance, as it is composed into its parent.
pub struct Part {
    /// The instance path, e.g. `u.v`.
    pub path: Path,
    /// The module name with all overridable parameter values, as in an
    /// instantiation: `stage #(.W(4))`.
    pub title: String,
    pub module: Module,
}

/// The value of a literal or constant (after sizing, of the type of its
/// context).
fn literal(e: &Expr) -> Option<BitVec> {
    match &e.kind {
        ExprKind::Lit { value, .. } | ExprKind::Const { value, .. } => Some(value.clone()),
        _ => None,
    }
}

/// The name of the external variable holding the initial value of the
/// register `q`. Generated names contain `@`, which SystemVerilog
/// identifiers cannot, so they never collide with source names.
pub fn init_name(q: &str) -> Name {
    format!("{q}@init")
}

impl Translation {
    /// The name of a variable of the module.
    pub fn name(&self, var: &Var) -> &str {
        self.vars.name(var).unwrap_or("?")
    }

    /// The variable of a signal; panics if there is none.
    pub fn var(&self, name: &str) -> Var {
        self.vars[name]
    }

    /// The module printed with the signal names.
    pub fn print(&self) -> String {
        self.print_module(&self.module, &self.name)
    }

    /// The module of an instance, printed with the signal names of the
    /// whole design and titled like an instantiation: `stage #(.W(4)) s2`.
    pub fn print_part(&self, part: &Part) -> String {
        self.print_module(&part.module, &format!("{} {}", part.title, part.path))
    }

    fn print_module(&self, module: &Module, title: &str) -> String {
        crate::print::module(
            module,
            title,
            |v| self.name(v).to_string(),
            |w| self.wire_names.get(w).cloned(),
        )
    }
}

/// Checks and translates a module without instances.
pub fn lower(m: &ast::Module) -> Result<Translation> {
    lower_inst(&Inst::leaf(m.clone()))
}

/// Checks and translates an elaborated hierarchy.
pub fn lower_inst(inst: &Inst) -> Result<Translation> {
    let naming: BTreeMap<Name, Name> = inst.naming().into_iter().collect();
    let mut out = Out {
        naming: &naming,
        vars: BTreeMap::new(),
        registers: Vec::new(),
        initialized: Vec::new(),
        register_widths: BTreeMap::new(),
        reset_registers: Vec::new(),
        wire_names: HashMap::new(),
        instances: Vec::new(),
        out_of_range: Vec::new(),
        blackboxes: Vec::new(),
    };
    let module = lower_rec(inst, &Path::top(), &mut out)?;
    out.out_of_range
        .sort_by(|a, b| (a.span, &a.msg).cmp(&(b.span, &b.msg)));
    out.out_of_range.dedup();
    let clocks = typing::domains(inst)?.into_iter().collect();
    let m = &inst.module;
    let unread_inputs = m
        .ports
        .iter()
        .filter(|p| p.dir == Dir::Input)
        .map(|p| &p.decl.name)
        .filter(|x| !module.extl().contains(&out.vars[*x]))
        .cloned()
        .collect();
    Ok(Translation {
        name: m.name.clone(),
        module,
        vars: out.vars.into(),
        paths: naming.clone(),
        clocks,
        registers: out.registers,
        initialized: out.initialized,
        register_widths: out.register_widths,
        reset_registers: out.reset_registers,
        unread_inputs,
        blackbox_outputs: inst.blackbox_outputs(),
        out_of_range: out.out_of_range,
        blackboxes: out.blackboxes,
        wire_names: out.wire_names,
        instances: out.instances,
    })
}

/// The module name with its overridable parameters: `stage #(.W(4))`;
/// `int` values in decimal, others as sized literals.
fn title(m: &ast::Module) -> String {
    let params: Vec<String> = m
        .params
        .iter()
        .filter(|p| p.overridable)
        .map(|p| match p.ty {
            Ty::INT => format!(".{}({})", p.name, p.value.signed()),
            _ => format!(".{}({})", p.name, p.value),
        })
        .collect();
    if params.is_empty() {
        m.name.clone()
    } else {
        format!("{} #({})", m.name, params.join(", "))
    }
}

/// A bug of the translation, noticed at `span` if it concerns one place.
fn internal(span: Option<Span>, e: String) -> Error {
    let msg = format!("internal error: {e}");
    match span {
        Some(span) => Error::internal(span, msg),
        None => Error::unlocated(ErrorKind::Internal, msg),
    }
}
