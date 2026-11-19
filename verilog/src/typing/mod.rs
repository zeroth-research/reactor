/*!
# Well-formedness and the explicit-width fragment

[`checked`] runs every check of a module in order and gives a [`Checked`]
module, which the later phases (lowering, the reference interpreter,
printing the sized source) take: [`check`] (with the definite-assignment
rules of the blocks), then sizing, then [`explicit`].

[`check`] resolves names and checks that every signal has exactly one
driver, that assignments fit their kind of block, and the clocks and
resets of each module. The widths and signedness of expressions are left
to [`sizing`](crate::sizing), which makes every implicit resize of
IEEE 1800 explicit.

Sizing's result is in the *explicit-width fragment*, which [`explicit`] checks
and which lowering translates. The typing rules of the fragment are one
function per expression form ([`unary`], [`binary`], [`cond`]), called
again by lowering while it builds terms. They accept only expressions in
which no implicit resizing happens:

- operands of the arithmetic, bitwise and comparison operators, and the two
  arms of `?:`, have the same width and the same signedness;
- the right-hand side of an assignment has the width of the left-hand side;
- the condition of `?:` and the operands of `! && ||` and of the
  reductions are self-determined, so any width is fine, and so are the
  shift amount and the items of `{…}` and `{n{…}}`;
- resizing is explicit only: selects `x[…]` (always unsigned, like
  concatenations), `{…}`, `{n{…}}`, casts `N'(e)` and the resize nodes
  sizing inserts; `$signed`, `$unsigned`, `signed'` and `unsigned'` change
  the signedness only.

On this fragment the widths of IEEE 1800 (after propagation) equal the
self-determined widths, so the fragment has the standard's semantics. The
one place where this needs a restriction is a widening cast: `N'(e)` is
defined as assigning `e` to an `N`-bit variable, which evaluates `e` at `N`
bits (`16'(a + b)` keeps the carry of 8-bit `a + b`). So in the fragment the
operand of a widening cast must have a width independent of its context
([`context_free`]); after sizing, the operand of a cast is never narrower
than the cast. A narrowing cast evaluates its operand at the operand's own
width and truncates.

A constant select must be inside its signal. A read with a variable
(unsigned) index may select bits outside it, which read 0: the standard's
value for a 2-state signal, and what Yosys builds; for a 4-state `logic` or
`reg` the standard gives x, so such a read is exact only for synthesis
tools that resolve that x to 0 ([`select`], which sizing calls with the
type of the index, and [`may_leave`]).

The elements of an unpacked array are signals of their own (`m[3]`); an
element read with a variable index, `m[i]`, is accepted only if every value
of the index is an address of the array ([`element`]), like a select. An
array has one driver as a whole, like a vector.

Assignment targets are signals, selects of signals, elements of arrays
with a variable index, and concatenations of targets ([`pieces`]). An `assign` drives whole signals, so its selects are
constant. In a block, a select may have a variable (unsigned) index; then
any index value is fine: the bits of the select outside the signal are not
written, as the standard says, so `a[i] <= d` with `i` beyond the top bit
of `a` changes nothing. Since the bits it writes are not known statically,
such a select must be the only piece of its signal in a concatenation. The
same holds for an element with a variable index: an index that is no
address writes nothing.

Parameters have been substituted by the frontend: a parameter is a
constant of its type, an unsized decimal literal an `int`.

An instance drives the signals connected to its outputs, which must be
whole signals of the port's width (an output is no assignment, so nothing
is resized); the expression connected to an input is assigned to the port
(sized like any assignment). A clock port of an instance is connected to
the name of a 1-bit signal, which then counts as a clock of the module.

Only registers have initial values (`logic q = c;`, `initial q = c;`, a
file loaded into a written array): a signal computed by an `assign`, an
`always_comb` or an instance takes its value in every time step, so an
initial value would have no effect on it, and an input is set from
outside. An initial value is a constant, sized like an assignment to the
register. A signal with an initial value that nothing drives is a constant
(the frontend has made it an `assign` of the value).

An `always_comb` block must also be free of latches and of reads of stale
values (checked by [`check`]): every signal it drives is assigned on every path
through the block, and read only after it has been assigned on every path
to the read.
(A simulator gives such a read the value from the block's previous run,
which changes whenever any signal the block reads changes, while synthesis
gives the value assigned later in the block: the two disagree, and no
hardware element behaves like the first.) A plain `always @*` is a
combinational block with the same rules, and so is a plain `always` with
an event list without edges (`@(a or b)`) if the list names every signal
the block reads apart from its own targets: a simulator runs the block only
when a listed signal changes, while synthesis ignores the list, so a
missing signal would make them disagree. A plain `always` with edge events
is a clocked block with the rules of `always_ff`. An `always @*` must read
a signal other than its own targets: a simulator never runs one that reads
none (it runs the block only when a signal it reads changes), so its
targets would stay x.

Values are 2-state: an x is accepted only as a dead store, `y = 'x;` in an
`always_comb` or `q <= 'x;` in an `always_ff` (a literal that is x in every
bit, to a target with constant indices), whose bits every path assigns
again before they are read or the block ends (checked by [`check`]). Such an
assignment has no effect. (A live x would be a don't-care, which synthesis
may resolve to any value; that is not modelled.)
*/

use crate::ast::{Dir, Error, Module, Name, SignalId, Span, Ty};
use std::collections::BTreeMap;

mod check;
mod checked;
mod definite;
mod domains;
mod fragment;

// the rules the documentation refers to are public, the other helpers of
// lowering and sizing are not
pub use check::check;
pub use checked::{Checked, checked};
pub use definite::Unwritten;
pub use domains::domains;
pub use fragment::{
    ElemIndex, Piece, Variable, binary, cond, context_free, element, explicit, explicit_source,
    expr, may_leave, pieces, select, unary,
};
pub(crate) use fragment::{
    cast, concat, divisor, elem_type, leave_warning, pieces_in, repl, target_element, target_index,
};

type Result<T> = std::result::Result<T, Error>;

/// What a name refers to.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Sig {
    /// The signal, or the element of an array, the name denotes.
    pub id: SignalId,
    pub ty: Ty,
    /// `None` for an internal signal.
    pub dir: Option<Dir>,
    pub span: Span,
}

pub type Env = BTreeMap<Name, Sig>;

/// The signals of the module, or the first name declared twice (signals,
/// parameters and instances share one name space).
pub fn env(m: &Module) -> Result<Env> {
    // all declared names, in source order, with the signal if it is one
    let params = m.params.iter().map(|p| (&p.name, p.span, None));
    let insts = m.instances().map(|i| (&i.name, i.span, None));
    let arrays = m.arrays.iter().map(|a| (&a.name, a.span, None));
    let ports = m.ports.iter().map(|p| (&p.decl, Some(p.dir)));
    let decls = m.decls.iter().map(|d| (d, None));
    // the elements of an array are named after it, so only its name counts
    let sigs = ports.chain(decls).filter(|(d, _)| !d.is_element());
    let sigs = sigs.map(|(d, dir)| {
        let sig = Sig {
            id: d.id.clone(),
            ty: d.ty,
            dir,
            span: d.span,
        };
        (&d.name, d.span, Some(sig))
    });
    let elements = m
        .ports
        .iter()
        .map(|p| &p.decl)
        .chain(&m.decls)
        .filter(|d| d.is_element())
        .map(|d| {
            let sig = Sig {
                id: d.id.clone(),
                ty: d.ty,
                dir: None,
                span: d.span,
            };
            (d.name.clone(), sig)
        });
    let pins = m.pins.iter().map(|d| (&d.name, d.span, None));
    let mut names: Vec<_> = (params.chain(insts).chain(arrays).chain(pins))
        .chain(sigs)
        .collect();
    names.sort_by_key(|(_, span, _)| *span);
    let mut seen: BTreeMap<&Name, Span> = BTreeMap::new();
    let mut env = Env::new();
    for (name, span, sig) in names {
        if let Some(prev) = seen.insert(name, span) {
            return Err(Error::new(
                span,
                format!("`{name}` already declared at {prev}"),
            ));
        }
        if let Some(sig) = sig {
            env.insert(name.clone(), sig);
        }
    }
    env.extend(elements);
    Ok(env)
}

/// The array that the element `x` belongs to, or `x` itself for any other
/// name.
pub(crate) fn array_of<'e>(env: &'e Env, x: &'e str) -> &'e str {
    match env.get(x) {
        Some(Sig {
            id: SignalId {
                base,
                index: Some(_),
            },
            ..
        }) => base,
        _ => x,
    }
}

pub(crate) fn lookup<'e>(env: &'e Env, name: &str, span: Span) -> Result<&'e Sig> {
    env.get(name)
        .ok_or_else(|| Error::new(span, format!("`{name}` is not declared")))
}
