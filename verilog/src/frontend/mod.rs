/*!
# From source text to the AST

Parses with `sv-parser` into a [`Design`], and lowers the syntax tree of
one of its modules into an [`ast::Module`], rejecting everything outside
the accepted subset with a located error.

A module is lowered once for each set of parameter values: the values are
substituted while lowering, so the result has no parameters left. Ranges,
replication counts, select bounds, cast sizes and parameter values are
[`constant`] expressions: built into an [`Expr`] like every other
expression, then evaluated. In every expression, a parameter becomes a
constant of its type and an unsized literal a constant of type `int`
(`5`) or of 32 unsigned bits (`'hff`); `'0` and `'1` take the width of
their context. Operators on constants are kept, so that
[`sizing`](crate::sizing) evaluates them at the type of their context;
only `$clog2` is folded here, into an `int`. A concatenation item must not take its
width from an unsized literal (`{a, 1}`, `{a, N - 1}`; IEEE 1800 forbids
it), reached through the operands that take the width of their context
(the private `unsized_width`). Instances keep their parameter
values and connections; [`elab`](crate::elab) elaborates the modules they
name.

An unpacked array `logic [7:0] m [0:N-1]` (or `[N-1:0]`, `[N]`; one
dimension, addresses from 0, not as a port) becomes one signal per element,
`m[0]`, …, `m[N-1]` (see [`ast::Array`]). An element with a constant index
is that signal; one with a variable index is an [`ast::Elem`]. An array is
loaded from a file by `initial $readmemb("f", m);` or `$readmemh`, with `f`
relative to the directory of the source file and read here. A loaded array
that is never written is a ROM, whose elements are constants rather than
signals; every element needs a value. For an array that is also written,
the file gives the initial values of the elements it covers (a RAM
preload), and the others have no initial value. Arrays are resolved after
the whole module is read, since the `initial` may come after the uses.

Initial values ([`ast::Init`]) come from declarations (`logic q = c;`,
`output logic q = c`) and from `initial` blocks of blocking assignments
`q = c;` to whole signals or to elements with a constant index (`m[3] =
c;`). A value that is x in every bit (`'x`, `1'bx` for a 1-bit signal)
says that the initial value is unknown, which is what a signal without an
initial value has, so it adds nothing. A net declaration with a value,
`wire y = e;`, is no initial value but a continuous assignment, the same
as `assign y = e;`.

The macro `SYNTHESIS` is predefined (as `1`), as Yosys does, since we
translate what synthesis builds: code under `` `ifdef SYNTHESIS `` is read
and code under `` `ifndef SYNTHESIS `` (often simulation-only debug code) is
not. (Simulators such as iverilog do not define it.) `YOSYS`, which Yosys
also defines, is not: Yosys-specific code under `` `ifdef YOSYS `` is not
read. [`depends_on_macro`] tells whether a design depends on a macro.

Verilog spellings: `reg` is the same 2-state vector as `logic` and `bit`
(in the standard `reg` and `logic` are one 4-state type). A plain `always`
is classified by its event control alone: with edge events it is a clocked
block, the same as `always_ff`; with `@*` or `@(*)` a combinational block,
the same as `always_comb`; with a list of signals without edges (`@(a or
b)`, `@(a, b)`) also a combinational block, accepted by
[`typing`](crate::typing) only if the list names every signal the block
reads. The AST keeps the two kinds of block and records how each was
written ([`ast::Written`], `plain`), only for messages. A simulator runs an
`always @*` block for the first time when one of its signals changes,
while an `always_comb` also runs at time 0; the translation does not
depend on that (a combinational block computes its targets in every time
step), except that an `always @*` reading no signal never runs, so
[`typing`](crate::typing) rejects it. (`sv-parser` rejects `@ *` with a space; `@*` and `@ (*)` work.)

Attributes `(* … *)` are rejected, except `(* parallel_case *)` on a `case`
statement (kept in [`ast::StmtKind::Case`]; sizing
accepts it only when the labels are distinct constants, where it changes
nothing) and `(* blackbox *)` on a module. Others may change what synthesis
builds (`full_case` turns missing cases into don't-cares), so they are not
ignored.

A module marked `(* blackbox *)` is a stub of a cell outside the design
(see [`ast::Module::blackbox`]): it may declare parameters (also in its
body) but nothing else, and the widths of its ports must not depend on its
parameters, since the values an instance gives them have no effect. A
blackbox is never the top module. `inout` ports (*pins*) are accepted on
any module and kept apart from the other ports ([`ast::Module::pins`]);
elaboration and typing restrict their use.

A design may consist of several files ([`parse_design_files`]): they are
read as one text in the order given, so a macro defined in one file is
defined in the following ones (as Yosys and iverilog do). Positions count
the lines of all files; [`Design::locate`] gives the file and the position
in it. The files of `$readmemb`/`$readmemh` are found relative to the
directory of the file declaring the module.

`sv-parser` ignores operator precedence: a chain of binary operators comes
out nested to the right in source order (`a * b + c` as `a * (b + c)`), and
`?:` is nested the same way (`a + b ? c : d` as `a + (b ? c : d)`). The
frontend therefore flattens every chain back into its source order and
re-associates it with the precedence table of IEEE 1800-2017, Table 11-2
(`prec` in `frontend/expr.rs`). Parenthesized subexpressions are separate nodes and are
not affected.

[`ast::Module`]: crate::ast::Module
[`ast::Module::blackbox`]: crate::ast::Module::blackbox
[`ast::Module::pins`]: crate::ast::Module::pins
[`ast::Array`]: crate::ast::Array
[`ast::Elem`]: crate::ast::Elem
[`ast::Init`]: crate::ast::Init
[`ast::StmtKind::Case`]: crate::ast::StmtKind::Case
[`ast::Written`]: crate::ast::Written
[`constant`]: crate::constant
[`Expr`]: crate::ast::Expr
*/

use crate::ast::{Array, Const, Error, Init, Name, Override, Span};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use sv_parser as sv;
use sv_parser::{Iter, RefNode, RefNodes, SyntaxTree};

mod arrays;
mod decl;
mod design;
mod expr;
mod stmt;

pub use design::{
    Design, depends_on_macro, parse, parse_design, parse_design_file, parse_design_files,
    parse_file,
};

use arrays::Load;
use design::at;

type Result<T> = std::result::Result<T, Error>;

/// The largest integer accepted where one is needed (a range bound, a
/// replication count, a cast size, …): 2^16, the least limit on the width
/// of a vector that IEEE 1800 lets tools set (§6.9.1).
const LARGEST: usize = 1 << 16;

const VARIABLE_TARGET: &str = "a continuous assignment needs a constant select: an `assign` \
     drives fixed bits (assign the bit in an `always_comb` instead)";

struct Cx<'t> {
    tree: &'t SyntaxTree,
    lines: &'t [usize],
    /// The parameters declared so far, with their values.
    params: RefCell<BTreeMap<Name, Const>>,
    /// The parameter values given by the instance.
    overrides: &'t [Override],
    /// The overrides used so far.
    used: RefCell<BTreeSet<Name>>,
    /// The directory of the source file.
    dir: &'t Path,
    /// The arrays declared so far.
    arrays: RefCell<Vec<Array>>,
    /// The initial values so far.
    inits: RefCell<Vec<Init>>,
    /// The initial values that are x in every bit, so no initial values:
    /// the target, the size of the literal (none if unsized) and the
    /// position of the target.
    unknown: RefCell<Vec<(Name, Option<usize>, Span)>>,
    /// The arrays loaded from files so far.
    loads: RefCell<Vec<Load>>,
}

/// Whether the node contains any token; optional syntax that is absent or
/// empty contains none.
fn has_tokens<'a>(node: impl Into<RefNodes<'a>>) -> bool {
    Iter::new(node.into()).any(|n| matches!(n, RefNode::Locate(_)))
}

/// The name and the arguments of a call of a system task or function with
/// positional arguments only (`$f(a, b)`, also with empty arguments
/// `$f(a, , b)`), if it is one.
fn system_call(
    call: &sv::SystemTfCall,
) -> Option<(&sv::SystemTfIdentifier, Vec<&Option<sv::Expression>>)> {
    match call {
        sv::SystemTfCall::ArgOptionl(a) => {
            let (name, args) = &a.nodes;
            match args.as_ref().map(|a| &a.nodes.1) {
                Some(sv::ListOfArguments::Ordered(a)) if a.nodes.1.is_empty() => {
                    Some((name, a.nodes.0.contents()))
                }
                _ => None,
            }
        }
        sv::SystemTfCall::ArgExpression(a) => {
            let (name, args) = &a.nodes;
            let (args, clocking) = &args.nodes.1;
            if clocking.is_some() {
                return None;
            }
            Some((name, args.contents()))
        }
        sv::SystemTfCall::ArgDataType(_) => None,
    }
}

impl Cx<'_> {
    /// The position of the first token of the node.
    fn span<'a>(&self, node: impl Into<RefNodes<'a>>) -> Span {
        Iter::new(node.into())
            .find_map(|n| match n {
                RefNode::Locate(l) => self.tree.get_origin(l).map(|(_, off)| at(self.lines, off)),
                _ => None,
            })
            .unwrap_or_default()
    }

    /// The source text of the node, without trailing whitespace.
    fn text<'a>(&self, node: impl Into<RefNodes<'a>>) -> String {
        self.tree.get_str_trim(node).unwrap_or("").to_string()
    }

    fn err<'a, T>(&self, node: impl Into<RefNodes<'a>>, msg: impl Into<String>) -> Result<T> {
        Err(Error::new(self.span(node), msg))
    }

    /// Rejects the node as outside the subset, naming its syntactic kind.
    fn unsupported<'a, T>(&self, node: impl Into<RefNode<'a>>) -> Result<T> {
        let node: RefNode = node.into();
        let span = self.span(RefNodes(vec![node.clone()]));
        Err(Error::new(span, format!("unsupported: {node}")))
    }

    /// Rejects the node if it contains tokens, i.e. if the optional syntax
    /// `what` is present.
    fn absent<'a>(&self, node: impl Into<RefNodes<'a>> + Copy, what: &str) -> Result<()> {
        if has_tokens(node) {
            self.err(node, format!("{what} not supported"))
        } else {
            Ok(())
        }
    }

    /// Checks the attributes `attrs`, of which only `name` is accepted:
    /// rejects any other, and passes each `name` to `each` in order.
    fn attributes(
        &self,
        attrs: &[sv::AttributeInstance],
        name: &str,
        mut each: impl FnMut(&sv::AttrSpec) -> Result<()>,
    ) -> Result<()> {
        for a in attrs {
            for spec in a.nodes.1.contents() {
                if self.text(&spec.nodes.0) != name {
                    return self.err(
                        spec,
                        format!("attribute `{}` not supported", self.text(spec)),
                    );
                }
                each(spec)?;
            }
        }
        Ok(())
    }

    /// The statements of `begin … end`: no block name, and no declarations
    /// (rejected with `declarations`).
    fn seq_block<'b>(
        &self,
        b: &'b sv::SeqBlock,
        declarations: &str,
    ) -> Result<&'b [sv::StatementOrNull]> {
        let (_, label, decls, stmts, _, end_label) = &b.nodes;
        self.absent(label, "block name")?;
        self.absent(end_label, "block name")?;
        if let Some(d) = decls.first() {
            return self.err(d, declarations);
        }
        Ok(stmts)
    }

    fn identifier(&self, id: &sv::Identifier) -> Result<String> {
        match id {
            sv::Identifier::SimpleIdentifier(_) => Ok(self.text(id)),
            sv::Identifier::EscapedIdentifier(_) => {
                self.err(id, "escaped identifiers not supported")
            }
        }
    }
}
