/*!
# From the AST to a reactive module

Every signal becomes a variable of the width of its type: inputs are
external, outputs are interface variables, internal signals are private.

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
| `x[i]`, `x[i +: w]` (variable) | `Extract{w-1, 0}(LShr(x, i))` |
| `N'(e)` | `SExt`/`ZExt` (as `e`'s signedness says) when widening, `Extract` when narrowing |

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
and the bits that differ are merged with `Ite`. The init action sets
every register to an external variable `q@init`, so all initial states
are possible.

Reading values just before the edge is what hardware does under two
timing assumptions, which this semantics makes and does not check:

- data inputs meet the setup and hold times of the registers reading
  them: an input does not change in the round where an edge samples it
  (it takes its new value in the round, but the edge sees the latched
  one);
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
and clock loops.

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

so the atom also awaits the reset (and a register reset by a signal
computed from itself is a loop). The value after the edge, `new`, comes
from the body as written, `if (rst)` included, which reads the latched
reset like any other value. So a clock edge in the round where the reset
is released sees the reset still active and keeps the reset values:
releasing the reset takes effect at the next edge. This is the timing
assumption of resets: the release meets the recovery and removal times of
the registers (it does not change too close to an edge), just as data
inputs meet setup and hold. Registers of the block that the reset branch
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

`case (s) l1, l2: a … default: d endcase` becomes a chain of `Ite`s on
`s == l1 | s == l2`, …, so the first matching item wins; without `default`,
no item matching leaves the map as it was.

A module with instances becomes the composition of a module made of its
own atoms and the translations of its instances (each translated on its
own, with fresh variables), with the module's internal signals hidden.
A port connected to a signal name is that signal's variable (so a clock
port is the clock of the instantiating module, and the instance's blocks
test its edges themselves); an input connected to another expression is a variable `u.a` driven by an extra
atom of the module, as if by `assign u.a = …`. Variables are named as
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
*/

use crate::ast::{
    self, BinOp, Dir, Edge, Error, Expr, ExprKind, Item, Lvalue, LvalueKind, Name, Reset, Span,
    Stmt, StmtKind, Ty, UnOp,
};
use crate::elab::{self, Inst};
use crate::typing;
use crate::{Atom, Module, Sort, SvOp, Term, Var, Wire};
use base::var::X;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};

type Result<T> = std::result::Result<T, Error>;

/// A module with the names of its variables.
pub struct Translation {
    pub name: Name,
    pub module: Module,
    /// Every source signal, and `q@init` for every register `q`, by name.
    pub vars: BTreeMap<Name, Var>,
    /// The input clocks of the whole design: the inputs of the top module
    /// that the `always_ff` blocks are clocked by, directly or through
    /// derived clocks, by name, sorted.
    pub clocks: Vec<Name>,
    /// The signals assigned in `always_ff` blocks, in source order.
    pub registers: Vec<Name>,
    /// The registers with an asynchronous reset (those the reset branch of
    /// their block assigns), by path, in source order.
    pub reset_registers: Vec<Name>,
    /// Inputs that nothing reads; they are not variables of the module.
    pub unread_inputs: Vec<Name>,
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

/// The module of an instance, as it is composed into its parent.
pub struct Part {
    /// The instance path, e.g. `u.v`.
    pub path: Name,
    /// The module name with all overridable parameter values, as in an
    /// instantiation: `stage #(.W(4))`.
    pub title: String,
    pub module: Module,
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
        self.vars
            .iter()
            .find(|(_, v)| *v == var)
            .map_or("?", |(n, _)| n.as_str())
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
        reset_registers: Vec::new(),
        wire_names: HashMap::new(),
        instances: Vec::new(),
    };
    let module = lower_rec(inst, "", &mut out)?;
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
        vars: out.vars,
        clocks,
        registers: out.registers,
        reset_registers: out.reset_registers,
        unread_inputs,
        wire_names: out.wire_names,
        instances: out.instances,
    })
}

/// What the translation of a hierarchy collects.
struct Out<'n> {
    /// The name of every signal path (see [`Inst::naming`]).
    naming: &'n BTreeMap<Name, Name>,
    /// The variables by name.
    vars: BTreeMap<Name, Var>,
    /// The paths of the registers.
    registers: Vec<Name>,
    /// The paths of the registers with an asynchronous reset.
    reset_registers: Vec<Name>,
    wire_names: HashMap<Wire, Name>,
    instances: Vec<Part>,
}

impl Out<'_> {
    /// The variable named `name`, made on first use.
    fn var(&mut self, name: &str, width: usize) -> Var {
        *self
            .vars
            .entry(name.to_string())
            .or_insert_with(|| Var::new(Sort(width)))
    }
}

/// Translates the instance at path `prefix` (`""` or ending in `.`): its
/// own items become atoms, its instances modules, which are composed with
/// the signals of the module hidden.
fn lower_rec(inst: &Inst, prefix: &str, out: &mut Out) -> Result<Module> {
    let m = &inst.module;
    let ctx = |e: Error| elab::context(e, prefix);
    let mut env = typing::check(m).map_err(ctx)?;
    let registers: Vec<Name> = m.registers().into_iter().cloned().collect();

    // an input connected to an expression other than a name is a signal
    // `u.a` of the module, driven by an `assign`
    let mut connections = Vec::new();
    for i in m.instances() {
        for (c, f) in i.conns(Dir::Input) {
            let Some(e) = c
                .actual
                .as_ref()
                .filter(|e| !matches!(e.kind, ExprKind::Var(_)))
            else {
                continue;
            };
            let name = format!("{}.{}", i.name, c.port);
            let sig = typing::Sig {
                ty: f.ty,
                dir: None,
                span: c.span,
            };
            env.insert(name.clone(), sig);
            let lhs = Lvalue {
                kind: LvalueKind::Var(name),
                span: c.span,
            };
            connections.push(Item::Assign {
                lhs,
                rhs: e.clone(),
                span: c.span,
            });
        }
    }

    // in declaration order, so that variable ids (and printing) follow it
    let mut vars: BTreeMap<Name, Var> = BTreeMap::new();
    let decls = m.ports.iter().map(|p| &p.decl).chain(&m.decls);
    let locals = decls
        .map(|d| d.name.clone())
        .chain(connections.iter().flat_map(|c| c.targets()).cloned());
    for x in locals.collect::<Vec<_>>() {
        let name = out.naming[&format!("{prefix}{x}")].clone();
        vars.insert(x.clone(), out.var(&name, env[&x].ty.width));
    }
    for q in &registers {
        let path = format!("{prefix}{q}");
        vars.insert(init_name(q), out.var(&init_name(&path), env[q].ty.width));
        out.registers.push(path);
    }
    for (q, _) in m.items.iter().filter_map(reset_of).flat_map(|(_, v)| v) {
        out.reset_registers.push(format!("{prefix}{q}"));
    }

    // the await order is computed on the AST, so that a loop is reported
    // as such before any term is built
    let items = m
        .items
        .iter()
        .filter(|item| !matches!(item, Item::Instance(_)))
        .chain(&connections);
    // by name rather than path: ports connected to the same signal are one
    // variable (`.a(y), .y(y)` makes `assign y = a` a loop)
    let naming = out.naming;
    let canon = |x: &Name| {
        naming
            .get(&format!("{prefix}{x}"))
            .cloned()
            .unwrap_or_else(|| x.clone())
    };
    let pending: Vec<Pending> = items
        .map(|item| {
            let ctrl: Vec<Name> = item.targets().into_iter().cloned().collect();
            let awaits: BTreeSet<Name> = match item {
                // the init action awaits the initial values, the update
                // the next value of the clock (for the edge test); both
                // await the next value of an asynchronous reset
                Item::AlwaysFf { clock, .. } => ctrl
                    .iter()
                    .map(|q| init_name(q))
                    .chain([clock.clone()])
                    .chain(reset_of(item).map(|(r, _)| r.name.clone()))
                    .collect(),
                // `assign y = y` awaits itself, a block reads its own
                // signals from its local values
                Item::Assign { .. } => item.reads().into_iter().map(|(x, _)| x.clone()).collect(),
                Item::AlwaysComb { .. } => item
                    .reads()
                    .into_iter()
                    .map(|(x, _)| x)
                    .filter(|x| !ctrl.contains(x))
                    .cloned()
                    .collect(),
                Item::Instance(_) => unreachable!("not an atom"),
            };
            let deps = match item {
                Item::AlwaysComb { body, .. } if ctrl.len() > 1 => {
                    let mut deps: BTreeMap<Name, BTreeSet<Name>> = BTreeMap::new();
                    for (y, d) in dependencies(body) {
                        deps.entry(canon(&y))
                            .or_default()
                            .extend(d.iter().map(&canon));
                    }
                    Some(deps)
                }
                _ => None,
            };
            Pending {
                item,
                reset: reset_of(item).map(|(r, _)| canon(&r.name)),
                ctrl: ctrl.iter().map(&canon).collect(),
                awaits: awaits.iter().map(&canon).collect(),
                deps,
                span: item.span(),
            }
        })
        .collect();

    let mut atoms = Vec::new();
    // the number of edge tests named so far, per name
    let mut edges: BTreeMap<Name, usize> = BTreeMap::new();
    for item in await_order(pending).map_err(ctx)? {
        let span = item.span();
        let atom = match item {
            Item::Assign { lhs, rhs, .. } => {
                let mut b = Builder::new(&env, &vars, false);
                b.start(&item.targets(), |_| None);
                let (w, _) = b.expr(rhs)?;
                b.assign(lhs, w);
                b.finish_all(span)?;
                out.add_names(prefix, b.names);
                let scope: Vec<&Var> = b.used.iter().map(|n| &vars[n]).collect();
                Atom::combinatorial(scope, b.terms).map_err(|e| internal(span, e))?
            }
            Item::AlwaysFf {
                clock, edge, body, ..
            } => {
                let regs = item.targets();
                let reset = reset_of(item);
                let mut b = Builder::new(&env, &vars, true);
                let e = b.edge(clock, *edge, span)?;
                let base = format!("{clock}@{}", if *edge == Edge::Pos { "pos" } else { "neg" });
                let k = edges.entry(base.clone()).or_insert(0);
                *k += 1;
                let name = if *k == 1 { base } else { format!("{base}#{k}") };
                b.names.insert(e, name);
                b.start(&regs, |q| Some(*vars[q]));
                b.stmt(body)?;
                let active = reset
                    .as_ref()
                    .map(|(r, values)| (b.reset_level(r), values.as_slice()));
                b.finish_guarded(e, active, span)?;
                out.add_names(prefix, std::mem::take(&mut b.names));
                // the init action: the initial value, or the reset value
                // if the reset is active in the first round
                let mut init = Builder::new(&env, &vars, true);
                let active = reset.as_ref().map(|(r, _)| init.reset_level(r));
                for q in &regs {
                    let (q0, var) = (vars[&init_name(q)], vars[*q]);
                    init.used.insert(init_name(q));
                    let value = reset.iter().flat_map(|(_, v)| v).find(|(r, _)| r == q);
                    match (active, value) {
                        (Some(a), Some((_, c))) => {
                            let (c, _) = init.expr(c)?;
                            init.emit_to(SvOp::Ite, &[a, c, X(q0)], X(var), span)?;
                        }
                        _ => {
                            init.emit_to(SvOp::Id, &[X(q0)], X(var), span)?;
                        }
                    }
                }
                b.used.extend(init.used);
                let scope: Vec<&Var> = b.used.iter().map(|n| &vars[n]).collect();
                Atom::sequential(scope, init.terms, b.terms).map_err(|e| internal(span, e))?
            }
            Item::AlwaysComb { body, .. } => {
                let mut b = Builder::new(&env, &vars, false);
                b.start(&item.targets(), |_| None);
                b.stmt(body)?;
                b.finish_all(span)?;
                out.add_names(prefix, b.names);
                let scope: Vec<&Var> = b.used.iter().map(|n| &vars[n]).collect();
                Atom::combinatorial(scope, b.terms).map_err(|e| internal(span, e))?
            }
            Item::Instance(_) => unreachable!("not an atom"),
        };
        atoms.push(atom);
    }

    if inst.children.is_empty() {
        let private: BTreeSet<Var> = m.decls.iter().map(|d| vars[&d.name]).collect();
        return Module::partially_observable(atoms, |v| private.contains(v))
            .map_err(|e| internal(Span::default(), e));
    }

    // the module's own atoms observe everything, since its instances may
    // read and write its internal signals; hiding happens in the
    // composition
    let own =
        Module::partially_observable(atoms, |_| false).map_err(|e| internal(Span::default(), e))?;
    let mut modules = vec![(own, Span::default())];
    let mut hidden: BTreeSet<Var> = m.decls.iter().map(|d| vars[&d.name]).collect();
    hidden.extend(
        connections
            .iter()
            .flat_map(|c| c.targets())
            .map(|x| vars[x]),
    );
    for (u, child) in &inst.children {
        let i = m.instances().find(|i| i.name == *u).expect("an instance");
        let path = format!("{prefix}{u}.");
        let module = lower_rec(child, &path, out)?;
        out.instances.push(Part {
            path: format!("{prefix}{u}"),
            title: title(&child.module),
            module: module.clone(),
        });
        modules.push((module, i.span));
        // unconnected outputs are signals of the instance only
        for p in child.module.ports.iter().filter(|p| p.dir == Dir::Output) {
            let connected = i
                .conns
                .iter()
                .any(|c| c.port == p.decl.name && c.actual.is_some());
            if !connected {
                hidden.insert(out.vars[&format!("{path}{}", p.decl.name)]);
            }
        }
    }
    if let Some((span, msg)) = instance_loop(&modules, &out.vars) {
        return Err(ctx(Error::new(span, msg)));
    }
    Module::compose_hiding(modules.into_iter().map(|(m, _)| m), |v| hidden.contains(v))
        .map_err(|e| internal(Span::default(), e))
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

impl Out<'_> {
    /// Adds the names of intermediate values of an atom of the instance at
    /// `prefix`.
    fn add_names(&mut self, prefix: &str, names: HashMap<Wire, Name>) {
        self.wire_names
            .extend(names.into_iter().map(|(w, n)| (w, format!("{prefix}{n}"))));
    }
}

/// A combinational loop through the atoms of several modules (each of them
/// free of loops), with the position of the instance of the first module
/// on it that is not the first of `modules`.
fn instance_loop(modules: &[(Module, Span)], vars: &BTreeMap<Name, Var>) -> Option<(Span, String)> {
    let atoms: Vec<(&Atom, Span)> = modules
        .iter()
        .flat_map(|(m, span)| m.atoms().iter().map(move |a| (a, *span)))
        .collect();
    let name = |v: &Var| {
        vars.iter()
            .find(|(_, w)| *w == v)
            .map_or("?", |(n, _)| n.as_str())
    };
    // the variable by which atom `a` awaits atom `b`
    let via = |a: usize, b: usize| {
        atoms[a]
            .0
            .wait()
            .iter()
            .find(|v| atoms[b].0.ctrl().contains(v))
            .copied()
    };
    // depth-first search for a cycle
    let n = atoms.len();
    let mut state = vec![0u8; n]; // 0 new, 1 on the stack, 2 done
    let mut stack: Vec<usize> = Vec::new();
    fn dfs(
        a: usize,
        n: usize,
        via: &dyn Fn(usize, usize) -> Option<Var>,
        state: &mut [u8],
        stack: &mut Vec<usize>,
    ) -> Option<Vec<usize>> {
        state[a] = 1;
        stack.push(a);
        for b in 0..n {
            if via(a, b).is_none() {
                continue;
            }
            if state[b] == 1 {
                let start = stack.iter().position(|&x| x == b).unwrap();
                return Some(stack[start..].to_vec());
            }
            if state[b] == 0
                && let Some(c) = dfs(b, n, via, state, stack)
            {
                return Some(c);
            }
        }
        stack.pop();
        state[a] = 2;
        None
    }
    for a in 0..n {
        if state[a] == 0
            && let Some(cycle) = dfs(a, n, &via, &mut state, &mut stack)
        {
            // each atom awaits the next: the data flows backwards
            let mut signals: Vec<&str> = (0..cycle.len())
                .map(|k| name(&via(cycle[k], cycle[(k + 1) % cycle.len()]).unwrap()))
                .collect();
            signals.reverse();
            signals.push(signals[0]);
            let span = cycle
                .iter()
                .map(|&a| atoms[a].1)
                .find(|s| *s != Span::default())
                .unwrap_or_default();
            return Some((
                span,
                format!(
                    "combinational loop through instances: {}",
                    signals.join(" -> ")
                ),
            ));
        }
    }
    None
}

fn internal(span: Span, e: String) -> Error {
    Error::new(span, format!("internal error: {e}"))
}

/// An item before sorting, with the signals it controls and those it
/// awaits.
struct Pending<'m> {
    item: &'m Item,
    /// The asynchronous reset of an `always_ff` block, by canonical name.
    reset: Option<Name>,
    ctrl: Vec<Name>,
    awaits: BTreeSet<Name>,
    /// For an `always_comb` driving several signals, what each depends on.
    deps: Option<BTreeMap<Name, BTreeSet<Name>>>,
    span: Span,
}

impl Pending<'_> {
    /// A signal that `self` awaits and `other` controls.
    fn awaits_from<'p>(&self, other: &'p Pending) -> Option<&'p Name> {
        other.ctrl.iter().find(|x| self.awaits.contains(*x))
    }
}

/// Sorts the items so that each comes after the items controlling what it
/// awaits, keeping the source order where possible; a cycle is an error.
fn await_order<'m>(mut pending: Vec<Pending<'m>>) -> Result<Vec<&'m Item>> {
    let mut sorted = Vec::new();
    while !pending.is_empty() {
        let ready = |a: &Pending| !pending.iter().any(|b| a.awaits_from(b).is_some());
        match pending.iter().position(ready) {
            Some(i) => sorted.push(pending.remove(i).item),
            None => return Err(combinational_loop(&pending)),
        }
    }
    Ok(sorted)
}

/// An error naming a cycle among items none of which is ready, and a block
/// through which the cycle passes only because the block is one atom.
fn combinational_loop(pending: &[Pending]) -> Error {
    // every item awaits some other pending item: walk until one repeats,
    // recording the signal by which each item is entered
    let mut path: Vec<(usize, Option<&Name>)> = vec![(0, None)];
    loop {
        let at = path.last().unwrap().0;
        let (next, x) = pending
            .iter()
            .enumerate()
            .find_map(|(i, b)| pending[at].awaits_from(b).map(|x| (i, x)))
            .expect("not ready");
        if let Some(start) = path.iter().position(|&(i, _)| i == next) {
            // the items of the cycle, each with the signal it controls and
            // the one it awaits from the next item
            let items: Vec<usize> = path[start..].iter().map(|&(i, _)| i).collect();
            let mut signals: Vec<&Name> = vec![x];
            signals.extend(path[start + 1..].iter().map(|(_, x)| x.unwrap()));
            let cycle: Vec<&str> = signals.iter().chain([&x]).map(|x| x.as_str()).collect();
            let mut msg = format!("combinational loop: {}", cycle.join(" -> "));
            // a register atom awaits only its clock and its reset: the
            // loop goes through the edge test or the reset
            if let Some(k) = items
                .iter()
                .position(|&i| matches!(pending[i].item, Item::AlwaysFf { .. }))
            {
                let (i, awaited) = (items[k], signals[(k + 1) % signals.len()]);
                let what = if pending[i].reset.as_ref() == Some(awaited) {
                    "reset"
                } else {
                    "clock"
                };
                msg += &format!(
                    " (through the {what} of the `always_ff` at {})",
                    pending[i].span
                );
            }
            for (k, &i) in items.iter().enumerate() {
                let (ctrl, awaited) = (signals[k], signals[(k + 1) % signals.len()]);
                let Some(deps) = &pending[i].deps else {
                    continue;
                };
                if !deps[ctrl].contains(awaited) {
                    msg += &format!(
                        "; in the `always_comb` at {}, `{ctrl}` does not depend on `{awaited}`, \
                         but the block is translated as a whole: split it",
                        pending[i].span
                    );
                    break;
                }
            }
            return Error::new(pending[next].span, msg);
        }
        path.push((next, Some(x)));
    }
}

/// The asynchronous reset of an `always_ff` block and its reset values, if
/// it resets any register.
fn reset_of(item: &Item) -> Option<(&Reset, Vec<(&Name, &Expr)>)> {
    let Item::AlwaysFf {
        reset: Some(r),
        body,
        ..
    } = item
    else {
        return None;
    };
    let values = body.reset_values();
    (!values.is_empty()).then_some((r, values))
}

/// For each signal an `always_comb` block drives, the signals outside the
/// block on which its final value depends, through data or control.
fn dependencies(body: &Stmt) -> BTreeMap<Name, BTreeSet<Name>> {
    let mut deps = BTreeMap::new();
    depend(body, &BTreeSet::new(), &mut deps);
    deps
}

/// Updates `deps` (the current dependencies of the signals assigned so
/// far) by `s`, run under conditions depending on `ctrl`.
fn depend(s: &Stmt, ctrl: &BTreeSet<Name>, deps: &mut BTreeMap<Name, BTreeSet<Name>>) {
    let resolve = |e: &Expr, deps: &BTreeMap<Name, BTreeSet<Name>>| {
        let mut reads = Vec::new();
        e.reads(&mut reads);
        let mut out = BTreeSet::new();
        for (x, _) in reads {
            match deps.get(x) {
                Some(d) => out.extend(d.iter().cloned()),
                None => {
                    out.insert(x.clone());
                }
            }
        }
        out
    };
    // runs each branch under `ctrl` and joins the results; `fallthrough`
    // adds a path that runs no branch
    let join = |branches: &[&Stmt], ctrl, fallthrough, deps: &mut BTreeMap<_, _>| {
        let mut joined = if fallthrough {
            deps.clone()
        } else {
            BTreeMap::new()
        };
        for b in branches {
            let mut out = deps.clone();
            depend(b, ctrl, &mut out);
            for (y, d) in out {
                joined.entry(y).or_insert_with(BTreeSet::new).extend(d);
            }
        }
        *deps = joined;
    };
    match &s.kind {
        StmtKind::Block(stmts) => stmts.iter().for_each(|s| depend(s, ctrl, deps)),
        StmtKind::If(c, a, b) => {
            let mut inner = ctrl.clone();
            inner.extend(resolve(c, deps));
            join(&[a, b], &inner, false, deps);
        }
        StmtKind::Case {
            sel,
            items,
            default,
        } => {
            let mut inner = ctrl.clone();
            inner.extend(resolve(sel, deps));
            for l in items.iter().flat_map(|i| &i.labels) {
                inner.extend(resolve(l, deps));
            }
            let branches: Vec<&Stmt> = s.children();
            join(&branches, &inner, default.is_none(), deps);
        }
        StmtKind::NonBlocking { lhs, rhs } | StmtKind::Blocking { lhs, rhs } => {
            let mut d = resolve(rhs, deps);
            d.extend(ctrl.iter().cloned());
            // a select keeps the dependencies of the other bits
            let mut whole = Vec::new();
            let mut targets = Vec::new();
            lvalue_targets(lhs, &mut whole, &mut targets);
            for y in targets {
                let entry = deps.entry(y.clone()).or_default();
                if !whole.contains(&y) {
                    entry.extend(d.iter().cloned());
                } else {
                    *entry = d.clone();
                }
            }
        }
    }
}

/// Bits `width` bits of a wire starting at bit `lo`, or `None` for bits
/// of an `always_comb` signal that are not assigned yet.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Seg {
    width: usize,
    src: Option<(Wire, usize)>,
}

/// The value of a signal of the block so far: segments from the high bits
/// down, with adjacent segments of the same wire (or both unassigned)
/// joined, so that equal values have equal segments.
type Value = Vec<Seg>;

/// The value of all bits of `w`.
fn whole(w: Wire) -> Value {
    vec![Seg {
        width: w.dtype().width(),
        src: Some((w, 0)),
    }]
}

/// The wire of a value that is all bits of one wire.
fn as_wire(v: &Value) -> Option<Wire> {
    match v.as_slice() {
        [
            Seg {
                width,
                src: Some((w, 0)),
            },
        ] if *width == w.dtype().width() => Some(*w),
        _ => None,
    }
}

/// Joins adjacent segments of the same wire (or both unassigned).
fn coalesce(v: Value) -> Value {
    let mut out: Value = Vec::new();
    for seg in v {
        if let Some(last) = out.last_mut() {
            match (last.src, seg.src) {
                (None, None) => {
                    last.width += seg.width;
                    continue;
                }
                (Some((a, alo)), Some((b, blo))) if a == b && alo == blo + seg.width => {
                    *last = Seg {
                        width: last.width + seg.width,
                        src: Some((b, blo)),
                    };
                    continue;
                }
                _ => {}
            }
        }
        out.push(seg);
    }
    out
}

/// `v` split so that a segment boundary lies at each of the bit positions
/// `cuts` (counted from bit 0).
fn split(v: &Value, cuts: &BTreeSet<usize>) -> Value {
    let mut out = Vec::new();
    let mut top: usize = v.iter().map(|s| s.width).sum();
    for seg in v {
        let lo = top - seg.width;
        // the cuts strictly inside the segment, from the top down
        let mut hi = top;
        for &c in cuts.range(lo + 1..top).rev() {
            out.push(Seg {
                width: hi - c,
                src: seg.src.map(|(w, wlo)| (w, wlo + c - lo)),
            });
            hi = c;
        }
        out.push(Seg {
            width: hi - lo,
            src: seg.src,
        });
        top = lo;
    }
    out
}

/// The segment boundaries of `v`, counted from bit 0.
fn cuts(v: &Value) -> BTreeSet<usize> {
    let mut out = BTreeSet::new();
    let mut lo: usize = v.iter().map(|s| s.width).sum();
    for seg in v {
        lo -= seg.width;
        out.insert(lo);
    }
    out
}

/// Adds the signals of a target to `targets`, and those it assigns as a
/// whole to `whole`.
fn lvalue_targets<'l>(l: &'l Lvalue, whole: &mut Vec<&'l Name>, targets: &mut Vec<&'l Name>) {
    match &l.kind {
        LvalueKind::Var(y) => {
            whole.push(y);
            targets.push(y);
        }
        LvalueKind::Select(y, _) => targets.push(y),
        LvalueKind::Concat(ls) => ls.iter().for_each(|l| lvalue_targets(l, whole, targets)),
    }
}

/// Builds the terms of one atom.
struct Builder<'a> {
    env: &'a typing::Env,
    vars: &'a BTreeMap<Name, Var>,
    terms: Vec<Term>,
    /// The signals the atom reads or writes.
    used: BTreeSet<Name>,
    /// Whether signals are read latched (`*x`, in `always_ff`) or awaited
    /// (`X(x)`).
    latched: bool,
    /// The value of each signal the block drives so far: the value after
    /// the edge in `always_ff`; in `always_comb`, the current value, with
    /// unassigned segments for bits that may not be assigned yet.
    cur: Vec<(Name, Value)>,
    /// The names of the intermediate values in `always_comb`.
    names: HashMap<Wire, Name>,
    /// The number of intermediate values named so far, per signal.
    versions: BTreeMap<Name, usize>,
    /// The wires that [`finish`](Self::finish) renamed, to the variable
    /// wire they became.
    renamed: HashMap<Wire, Wire>,
}

impl<'a> Builder<'a> {
    fn new(env: &'a typing::Env, vars: &'a BTreeMap<Name, Var>, latched: bool) -> Self {
        Self {
            env,
            vars,
            terms: Vec::new(),
            used: BTreeSet::new(),
            latched,
            cur: Vec::new(),
            names: HashMap::new(),
            versions: BTreeMap::new(),
            renamed: HashMap::new(),
        }
    }

    /// Starts the block driving `targets`, each with all bits of
    /// `init(y)`, or all bits unassigned.
    fn start(&mut self, targets: &[&Name], init: impl Fn(&Name) -> Option<Wire>) {
        self.used.extend(targets.iter().map(|y| (*y).clone()));
        self.cur = targets
            .iter()
            .map(|y| {
                let value = match init(y) {
                    Some(w) => whole(w),
                    None => vec![Seg {
                        width: self.env[*y].ty.width,
                        src: None,
                    }],
                };
                ((*y).clone(), value)
            })
            .collect();
    }

    /// Lowers a statement of a block, updating [`cur`](Self::cur).
    fn stmt(&mut self, s: &Stmt) -> Result<()> {
        match &s.kind {
            StmtKind::Block(stmts) => stmts.iter().try_for_each(|s| self.stmt(s)),
            StmtKind::If(c, a, b) => {
                let (wc, tc) = self.expr(c)?;
                let wc = self.truth(wc, tc, c.span)?;
                let before = self.cur.clone();
                self.stmt(a)?;
                let then = std::mem::replace(&mut self.cur, before);
                self.stmt(b)?;
                let els = std::mem::take(&mut self.cur);
                self.cur = self.merge(wc, then, els, s.span)?;
                Ok(())
            }
            StmtKind::Case {
                sel,
                items,
                default,
            } => {
                // whether each item matches, from the values before the case
                let (ws, _) = self.expr(sel)?;
                let mut hits = Vec::new();
                for item in items {
                    let mut hit = None;
                    for l in &item.labels {
                        let (wl, _) = self.expr(l)?;
                        let eq = self.emit(SvOp::Eq, &[ws, wl], Sort(1), l.span)?;
                        hit = Some(match hit {
                            None => eq,
                            Some(h) => self.emit(SvOp::Or, &[h, eq], Sort(1), l.span)?,
                        });
                    }
                    hits.push(hit.expect("a label"));
                }
                let before = self.cur.clone();
                let mut outs = Vec::new();
                for item in items {
                    self.cur = before.clone();
                    self.stmt(&item.body)?;
                    outs.push(std::mem::take(&mut self.cur));
                }
                self.cur = before;
                if let Some(d) = default {
                    self.stmt(d)?;
                }
                let mut acc = std::mem::take(&mut self.cur);
                for (hit, out) in hits.into_iter().zip(outs).rev() {
                    acc = self.merge(hit, out, acc, s.span)?;
                }
                self.cur = acc;
                Ok(())
            }
            StmtKind::NonBlocking { lhs, rhs } | StmtKind::Blocking { lhs, rhs } => {
                let (w, _) = self.expr(rhs)?;
                self.assign(lhs, w);
                Ok(())
            }
        }
    }

    /// Assigns the value of `w` to the target `lhs`, the first piece taking
    /// the high bits.
    fn assign(&mut self, lhs: &Lvalue, w: Wire) {
        let mut top = w.dtype().width();
        for p in typing::pieces(self.env, lhs) {
            top -= p.width();
            self.write(&p.name, p.hi, p.lo, w, top);
        }
    }

    /// Makes the bits `hi` down to `lo` of the signal `y` hold the bits of
    /// `w` from `wlo` upwards.
    fn write(&mut self, y: &Name, hi: usize, lo: usize, w: Wire, wlo: usize) {
        let i = self.cur.iter().position(|(x, _)| x == y).expect("a target");
        let value = &self.cur[i].1;
        let cut = split(value, &BTreeSet::from([lo, hi + 1]));
        let mut out = Vec::new();
        let mut top: usize = cut.iter().map(|s| s.width).sum();
        for seg in cut {
            top -= seg.width;
            if top == lo {
                out.push(Seg {
                    width: hi + 1 - lo,
                    src: Some((w, wlo)),
                });
            }
            if top > hi || top < lo {
                out.push(seg);
            }
        }
        let value = coalesce(out);
        if let Some(w) = as_wire(&value) {
            self.name(y, w);
        }
        self.cur[i].1 = value;
    }

    /// The wire holding the bits `hi` down to `lo` of `value`; all of them
    /// must be assigned.
    fn build(&mut self, value: &Value, hi: usize, lo: usize, span: Span) -> Result<Wire> {
        let cut = split(value, &BTreeSet::from([lo, hi + 1]));
        let mut picked = Vec::new();
        let mut top: usize = cut.iter().map(|s| s.width).sum();
        for seg in cut {
            top -= seg.width;
            if lo <= top && top <= hi {
                picked.push(seg);
            }
        }
        let mut parts = Vec::new();
        for seg in coalesce(picked) {
            let (w, wlo) = seg
                .src
                .ok_or_else(|| internal(span, "read of unassigned bits".into()))?;
            let part = if wlo == 0 && seg.width == w.dtype().width() {
                w
            } else {
                let op = SvOp::Extract {
                    hi: wlo + seg.width - 1,
                    lo: wlo,
                };
                self.emit(op, &[w], Sort(seg.width), span)?
            };
            parts.push(part);
        }
        let mut acc = parts.pop().expect("a part");
        while let Some(part) = parts.pop() {
            let sort = Sort(part.dtype().width() + acc.dtype().width());
            acc = self.emit(SvOp::Concat, &[part, acc], sort, span)?;
        }
        Ok(acc)
    }

    /// The wire holding the bits `hi` down to `lo` of the signal `y` of the
    /// block. If they are a concatenation of several wires, it becomes
    /// their value, so that it is built only once.
    fn read_own(&mut self, y: &Name, hi: usize, lo: usize, span: Span) -> Result<Wire> {
        let i = self.cur.iter().position(|(x, _)| x == y).expect("a target");
        let value = self.cur[i].1.clone();
        let w = self.build(&value, hi, lo, span)?;
        if self
            .terms
            .last()
            .is_some_and(|t| t.write() == [w] && *t.itype() == SvOp::Concat)
        {
            self.write(y, hi, lo, w, 0);
        }
        Ok(w)
    }

    /// Names `w` as the next intermediate value of `y` in `always_comb`,
    /// unless it is not local or already has a name.
    fn name(&mut self, y: &Name, w: Wire) {
        if !self.latched
            && !self.names.contains_key(&w)
            && self.terms.iter().any(|t| t.write() == [w])
        {
            let k = self.versions.entry(y.clone()).or_insert(0);
            *k += 1;
            self.names.insert(w, format!("{y}@{k}"));
        }
    }

    /// The map after `if (c)` with the maps `then` and `els` after the two
    /// branches: `Ite` on each run of bits that are assigned in both and
    /// differ, unassigned where either is unassigned.
    fn merge(
        &mut self,
        c: Wire,
        then: Vec<(Name, Value)>,
        els: Vec<(Name, Value)>,
        span: Span,
    ) -> Result<Vec<(Name, Value)>> {
        let mut out = Vec::new();
        for ((y, a), (_, b)) in then.into_iter().zip(els) {
            if a == b {
                out.push((y, a));
                continue;
            }
            let mut at = cuts(&a);
            at.extend(cuts(&b));
            let (a, b) = (split(&a, &at), split(&b, &at));
            // the aligned segments from the top down, with an empty pair at
            // the end that closes the last run of differing bits
            let end = Seg {
                width: 0,
                src: None,
            };
            let mut value = Vec::new();
            let mut run: Vec<(Seg, Seg)> = Vec::new();
            for (sa, sb) in a.into_iter().zip(b).chain([(end, end)]) {
                let differ = sa.src.is_some() && sb.src.is_some() && sa != sb;
                if differ {
                    run.push((sa, sb));
                    continue;
                }
                if !run.is_empty() {
                    let (ra, rb): (Value, Value) = std::mem::take(&mut run).into_iter().unzip();
                    let width: usize = ra.iter().map(|s| s.width).sum();
                    let wa = self.build(&ra, width - 1, 0, span)?;
                    let wb = self.build(&rb, width - 1, 0, span)?;
                    let w = self.emit(SvOp::Ite, &[c, wa, wb], Sort(width), span)?;
                    value.extend(whole(w));
                }
                if sa.width > 0 {
                    let src = if sb.src.is_some() { sa.src } else { None };
                    value.push(Seg {
                        width: sa.width,
                        src,
                    });
                }
            }
            let value = coalesce(value);
            if let Some(w) = as_wire(&value) {
                self.name(&y, w);
            }
            out.push((y, value));
        }
        Ok(out)
    }

    /// Makes the next value `X(y)` of every signal of the block hold its
    /// final value.
    fn finish_all(&mut self, span: Span) -> Result<()> {
        // all final values are built before any wire is renamed, since one
        // may be built from bits of another
        let mut finals = Vec::new();
        for (y, value) in std::mem::take(&mut self.cur) {
            let width = self.env[&y].ty.width;
            finals.push((self.build(&value, width - 1, 0, span)?, y));
        }
        for (w, y) in finals {
            self.finish(w, X(self.vars[&y]), span)?;
        }
        Ok(())
    }

    /// The wire holding whether the edge `edge` of `clock` happens in the
    /// round: the latched level and the next one are 0 and 1 (`posedge`),
    /// or 1 and 0 (`negedge`).
    fn edge(&mut self, clock: &Name, edge: Edge, span: Span) -> Result<Wire> {
        self.used.insert(clock.clone());
        let var = self.vars[clock];
        let (before, after) = (*var, X(var));
        Ok(match edge {
            Edge::Pos => {
                let low = self.emit(SvOp::Not, &[before], Sort(1), span)?;
                self.emit(SvOp::And, &[low, after], Sort(1), span)?
            }
            Edge::Neg => {
                let low = self.emit(SvOp::Not, &[after], Sort(1), span)?;
                self.emit(SvOp::And, &[before, low], Sort(1), span)?
            }
        })
    }

    /// Makes the next value `X(q)` of every register of the block its
    /// final value if the edge `e` happens, else its latched value `*q`;
    /// with an asynchronous reset (`reset` = the wire saying whether it is
    /// active, and the reset values), a register with a reset value gets
    /// that value instead while the reset is active.
    fn finish_guarded(
        &mut self,
        e: Wire,
        reset: Option<(Wire, &[(&Name, &Expr)])>,
        span: Span,
    ) -> Result<()> {
        let mut finals = Vec::new();
        for (q, value) in std::mem::take(&mut self.cur) {
            let width = self.env[&q].ty.width;
            finals.push((self.build(&value, width - 1, 0, span)?, q, width));
        }
        for (w, q, width) in finals {
            let var = self.vars[&q];
            let value = reset.and_then(|(a, values)| {
                values.iter().find(|(r, _)| **r == q).map(|(_, c)| (a, *c))
            });
            match value {
                Some((a, c)) => {
                    let g = self.emit(SvOp::Ite, &[e, w, *var], Sort(width), span)?;
                    let (c, _) = self.expr(c)?;
                    self.emit_to(SvOp::Ite, &[a, c, g], X(var), span)?;
                }
                None => {
                    self.emit_to(SvOp::Ite, &[e, w, *var], X(var), span)?;
                }
            }
        }
        Ok(())
    }

    /// The wire saying whether the asynchronous reset `r` is active in
    /// this round: its next level `X(rst)`, or `NOT X(rst_n)` for an
    /// active-low reset.
    fn reset_level(&mut self, r: &Reset) -> Wire {
        self.used.insert(r.name.clone());
        let level = X(self.vars[&r.name]);
        match r.edge {
            Edge::Pos => level,
            Edge::Neg => {
                let low = Wire::scalar(Sort(1));
                let term = Term::function(SvOp::Not, [low], [level]);
                self.terms.push(term.expect("a 1-bit reset"));
                low
            }
        }
    }

    /// Makes `dest` hold the value of `w`: renames `w` to `dest` if a term
    /// of this atom writes `w` and none reads it, else adds `dest := w`.
    /// A wire already renamed (two signals with the same final value) is
    /// copied from the variable it became.
    fn finish(&mut self, w: Wire, dest: Wire, span: Span) -> Result<()> {
        if let Some(&r) = self.renamed.get(&w) {
            self.emit_to(SvOp::Id, &[r], dest, span)?;
            return Ok(());
        }
        let writer = self.terms.iter().position(|t| t.write() == [w]);
        let read = self.terms.iter().any(|t| t.read().contains(&w));
        match writer {
            Some(i) if !read => {
                let t = &self.terms[i];
                let op = t.itype().clone();
                let term = if t.read().is_empty() {
                    Term::constant(op, [dest])
                } else {
                    Term::function(op, [dest], t.read().iter().copied())
                };
                self.terms[i] = term.map_err(|e| internal(span, e))?;
                self.renamed.insert(w, dest);
                self.names.remove(&w);
            }
            _ => {
                self.emit_to(SvOp::Id, &[w], dest, span)?;
            }
        }
        Ok(())
    }

    /// Adds the term `w := op(reads)` with a fresh wire `w` of `sort`.
    fn emit(&mut self, op: SvOp, reads: &[Wire], sort: Sort, span: Span) -> Result<Wire> {
        self.emit_to(op, reads, Wire::scalar(sort), span)
    }

    /// Adds the term `dest := op(reads)`.
    fn emit_to(&mut self, op: SvOp, reads: &[Wire], dest: Wire, span: Span) -> Result<Wire> {
        let term = if reads.is_empty() {
            Term::constant(op, [dest])
        } else {
            Term::function(op, [dest], reads.iter().copied())
        };
        self.terms.push(term.map_err(|e| internal(span, e))?);
        Ok(dest)
    }

    /// The wire holding whether `w` (of type `t`) is non-zero.
    fn truth(&mut self, w: Wire, t: Ty, span: Span) -> Result<Wire> {
        if t.width == 1 {
            Ok(w)
        } else {
            self.emit(SvOp::RedOr, &[w], Sort(1), span)
        }
    }

    /// The wire holding the value of the signal `x`, or of its bits `hi`
    /// down to `lo` if given (else all of them, `None` in the result).
    fn read(
        &mut self,
        x: &Name,
        bits: Option<(usize, usize)>,
        span: Span,
    ) -> Result<(Wire, Option<(usize, usize)>)> {
        self.used.insert(x.clone());
        let var = self.vars[x];
        let own = self.cur.iter().any(|(y, _)| y == x);
        if self.latched {
            return Ok((*var, bits));
        }
        if !own {
            return Ok((X(var), bits));
        }
        let (hi, lo) = bits.unwrap_or((self.env[x].ty.width - 1, 0));
        Ok((self.read_own(x, hi, lo, span)?, None))
    }

    /// Lowers `e` and returns the wire holding its value and its type.
    fn expr(&mut self, e: &Expr) -> Result<(Wire, Ty)> {
        let span = e.span;
        match &e.kind {
            ExprKind::Lit { ty, value } | ExprKind::Const { ty, value, .. } => {
                let w = self.emit(SvOp::Lit(value.clone()), &[], Sort(ty.width), span)?;
                Ok((w, *ty))
            }
            ExprKind::Var(x) => {
                let t = typing::lookup(self.env, x, span)?.ty;
                Ok((self.read(x, None, span)?.0, t))
            }
            ExprKind::Unary(op, a) => {
                let (wa, ta) = self.expr(a)?;
                let t = typing::unary(*op, ta);
                let s = Sort(t.width);
                let w = match op {
                    UnOp::Plus => wa,
                    UnOp::BitNot => self.emit(SvOp::Not, &[wa], s, span)?,
                    UnOp::Neg => self.emit(SvOp::Neg, &[wa], s, span)?,
                    UnOp::LogNot => {
                        let b = self.truth(wa, ta, span)?;
                        self.emit(SvOp::Not, &[b], s, span)?
                    }
                    UnOp::RedAnd => self.emit(SvOp::RedAnd, &[wa], s, span)?,
                    UnOp::RedOr => self.emit(SvOp::RedOr, &[wa], s, span)?,
                    UnOp::RedXor => self.emit(SvOp::RedXor, &[wa], s, span)?,
                    UnOp::RedNand | UnOp::RedNor | UnOp::RedXnor => {
                        let red = match op {
                            UnOp::RedNand => SvOp::RedAnd,
                            UnOp::RedNor => SvOp::RedOr,
                            _ => SvOp::RedXor,
                        };
                        let r = self.emit(red, &[wa], s, span)?;
                        self.emit(SvOp::Not, &[r], s, span)?
                    }
                };
                Ok((w, t))
            }
            ExprKind::Binary(op, a, b) => {
                let (wa, ta) = self.expr(a)?;
                let (wb, tb) = self.expr(b)?;
                let t = typing::binary(*op, ta, tb, span)?;
                let s = Sort(t.width);
                let (lt, le, div, rem) = if ta.signed {
                    (SvOp::SLt, SvOp::SLe, SvOp::SDiv, SvOp::SRem)
                } else {
                    (SvOp::ULt, SvOp::ULe, SvOp::UDiv, SvOp::URem)
                };
                let w = match op {
                    BinOp::BitAnd => self.emit(SvOp::And, &[wa, wb], s, span)?,
                    BinOp::BitOr => self.emit(SvOp::Or, &[wa, wb], s, span)?,
                    BinOp::BitXor => self.emit(SvOp::Xor, &[wa, wb], s, span)?,
                    BinOp::BitXnor => {
                        let x = self.emit(SvOp::Xor, &[wa, wb], s, span)?;
                        self.emit(SvOp::Not, &[x], s, span)?
                    }
                    BinOp::Add => self.emit(SvOp::Add, &[wa, wb], s, span)?,
                    BinOp::Sub => self.emit(SvOp::Sub, &[wa, wb], s, span)?,
                    BinOp::Mul => self.emit(SvOp::Mul, &[wa, wb], s, span)?,
                    BinOp::Div => self.emit(div, &[wa, wb], s, span)?,
                    BinOp::Mod => self.emit(rem, &[wa, wb], s, span)?,
                    BinOp::Eq | BinOp::CaseEq => self.emit(SvOp::Eq, &[wa, wb], s, span)?,
                    BinOp::Ne | BinOp::CaseNe => self.emit(SvOp::Ne, &[wa, wb], s, span)?,
                    BinOp::Lt => self.emit(lt, &[wa, wb], s, span)?,
                    BinOp::Le => self.emit(le, &[wa, wb], s, span)?,
                    BinOp::Gt => self.emit(lt, &[wb, wa], s, span)?,
                    BinOp::Ge => self.emit(le, &[wb, wa], s, span)?,
                    BinOp::LogAnd | BinOp::LogOr => {
                        let x = self.truth(wa, ta, span)?;
                        let y = self.truth(wb, tb, span)?;
                        let op = if *op == BinOp::LogAnd {
                            SvOp::And
                        } else {
                            SvOp::Or
                        };
                        self.emit(op, &[x, y], s, span)?
                    }
                    BinOp::Shl | BinOp::AShl => self.emit(SvOp::Shl, &[wa, wb], s, span)?,
                    BinOp::Shr => self.emit(SvOp::LShr, &[wa, wb], s, span)?,
                    BinOp::AShr => {
                        let op = if ta.signed { SvOp::AShr } else { SvOp::LShr };
                        self.emit(op, &[wa, wb], s, span)?
                    }
                };
                Ok((w, t))
            }
            ExprKind::Cond(c, a, b) => {
                let (wc, tc) = self.expr(c)?;
                let wc = self.truth(wc, tc, span)?;
                let (wa, ta) = self.expr(a)?;
                let (wb, tb) = self.expr(b)?;
                let t = typing::cond(ta, tb, span)?;
                let w = self.emit(SvOp::Ite, &[wc, wa, wb], Sort(t.width), span)?;
                Ok((w, t))
            }
            ExprKind::Concat(es) => {
                let (w, ts) = self.concat(es, span)?;
                Ok((w, typing::concat(&ts)))
            }
            ExprKind::Repl(n, es) => {
                let (w, ts) = self.concat(es, span)?;
                let t = typing::repl(*n, typing::concat(&ts));
                let mut acc = w;
                for _ in 1..*n {
                    let sort = Sort(w.dtype().width() + acc.dtype().width());
                    acc = self.emit(SvOp::Concat, &[w, acc], sort, span)?;
                }
                Ok((acc, t))
            }
            ExprKind::Select(x, sel) => {
                let tx = typing::lookup(self.env, x, span)?.ty;
                let index = match sel.index() {
                    Some(i) => Some(self.expr(i)?),
                    None => None,
                };
                let t = typing::select(x, tx, sel, index.map(|i| i.1), span)?;
                let w = match index {
                    None => {
                        let (hi, lo) = sel.constant().flatten().expect("constant");
                        match self.read(x, Some((hi, lo)), span)? {
                            (w, Some((hi, lo))) if lo > 0 || hi + 1 < tx.width => {
                                let op = SvOp::Extract { hi, lo };
                                self.emit(op, &[w], Sort(t.width), span)?
                            }
                            (w, _) => w,
                        }
                    }
                    // the lowest selected bit is the index: `-:` has width 1
                    Some((wi, _)) => {
                        let (wx, _) = self.read(x, None, span)?;
                        let shifted = self.emit(SvOp::LShr, &[wx, wi], Sort(tx.width), span)?;
                        let op = SvOp::Extract {
                            hi: t.width - 1,
                            lo: 0,
                        };
                        self.emit(op, &[shifted], Sort(t.width), span)?
                    }
                };
                Ok((w, t))
            }
            ExprKind::Cast(c, a) => {
                let (wa, ta) = self.expr(a)?;
                let t = typing::cast(*c, ta, typing::context_free(a), span)?;
                let s = Sort(t.width);
                let w = match t.width.cmp(&ta.width) {
                    Ordering::Equal => wa,
                    Ordering::Less => {
                        let op = SvOp::Extract {
                            hi: t.width - 1,
                            lo: 0,
                        };
                        self.emit(op, &[wa], s, span)?
                    }
                    Ordering::Greater if ta.signed => {
                        self.emit(SvOp::SExt(t.width), &[wa], s, span)?
                    }
                    Ordering::Greater => self.emit(SvOp::ZExt(t.width), &[wa], s, span)?,
                };
                Ok((w, t))
            }
        }
    }

    /// The wire holding `{e1, e2, …}`, as nested binary `Concat`s
    /// `{e1, {e2, …}}`, and the types of the items.
    fn concat(&mut self, es: &[Expr], span: Span) -> Result<(Wire, Vec<Ty>)> {
        let mut parts = Vec::new();
        let mut ts = Vec::new();
        for e in es {
            let (w, t) = self.expr(e)?;
            parts.push(w);
            ts.push(t);
        }
        let mut acc = parts.pop().expect("an item");
        while let Some(part) = parts.pop() {
            let sort = Sort(part.dtype().width() + acc.dtype().width());
            acc = self.emit(SvOp::Concat, &[part, acc], sort, span)?;
        }
        Ok((acc, ts))
    }
}
