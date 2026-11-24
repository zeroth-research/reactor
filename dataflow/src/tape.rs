//! Tapes: one flat, normalized representation for both control flow and data
//! flow.
//!
//! **Status: draft for discussion.** This module defines the frozen
//! [`Tape`], its parts, their [`Boundary`](base::boundary::Boundary)
//! implementations and a well-formedness check, and a [`Builder`] that puts
//! tapes together from combinators and freezes them in normal form. See
//! [Not done yet](#not-done-yet) for what is missing.
//!
//! # Goals
//!
//! A tape is meant to be an intermediate representation (IR) for programs that
//! branch, both on data (`if x < y`) and on chance (a sampled boolean).
//! Three requirements follow:
//!
//! 1. **A normal form.** Two tapes built from equal programs, according to the
//!    structural laws below, should be the *same* data. This holds even if
//!    the normal form takes more memory (paths are expanded).
//! 2. **A flat, static memory layout.** A tape is a handful of `Vec`s of plain
//!    data, indexed by `u32`s: there is no `Box` and no recursive type. A tape
//!    is built once and never changed afterwards.
//! 3. **Boundaries everywhere.** The tape, each arm, each node and each
//!    route has a domain and a codomain, typed by the polynomials of
//!    [`base::boundary`]. [`Tape`], [`TapeParts`], [`Node`] and [`NodeKind`] implement
//!    [`Boundary`](base::boundary::Boundary) directly. An [`Arm`] or a
//!    [`Route`] is plain data that does not know its input summand or the
//!    codomain, so their boundaries are on the views [`ArmRef`] and
//!    [`RouteRef`], which borrow the tape. (Storing those types in every
//!    arm and route would duplicate the tape's `dom` and `cod`.)
//!
//! # Arms and outputs
//!
//! Take a tape `dom → cod`, with `dom = U₁ ⊕ … ⊕ Uₙ` and
//! `cod = V₁ ⊕ … ⊕ Vₘ`. A value of `dom` lies in exactly one summand `Uⱼ`, as
//! a value of a Rust enum is exactly one variant. So the tape is `n`
//! independent programs, one per input summand, like the arms of a `match`:
//!
//! ```text
//! match input {
//!     U₁(..) => { /* arm 0 */ }
//!     U₂(..) => { /* arm 1 */ }
//! }
//! ```
//!
//! **Arm** `j` is the program for input summand `Uⱼ`. It ends by choosing an
//! **output** `i`, that is, an output summand `Vᵢ`, and the values to put there.
//! Several paths, from one arm or from several arms, may end in the same
//! output: this is how tapes *join* (a `merge`). Splits and joins in the middle of
//! a program are not stored as such: only the paths from an input summand to
//! an output summand are.
//!
//! A [`Tape`] stores its arms one after the other. Each [`Arm`] has two
//! parts:
//!
//! - a **circuit**: the [`Node`]s that compute, from the arm's inputs, every
//!   value any path of the arm needs;
//! - a **routing DAG**: the [`Route`]s that send the value along the tapes.
//!   A split ([`Route::Split`]) divides the tape in two on a boolean computed
//!   by the circuit, as the `case` primitive does. An end
//!   ([`Route::End`]) stops the tape at an output summand and names the
//!   values that go there.
//!
//! # Circuits
//!
//! A circuit is a term graph (an acyclic hypergraph with an interface). Its
//! **values** ([`Value`]) are the vertices, and its **nodes** are the
//! hyperedges: a node applies one operation to some values and produces
//! `n_outs` new values. Values are numbered *per arm*, by position:
//!
//! - `v0 … v(k-1)` are the arm's inputs, one per sort of `Uⱼ` (`k = |Uⱼ|`);
//! - then come the outputs of the arm's nodes, in node order.
//!
//! A node's arguments are always values defined *before* it, so a circuit is
//! acyclic by construction: there are no loops or traces. There are no nodes
//! for the structure of strings. Copying a value is using it more than once,
//! discarding it is never using it, and swapping or passing it through is
//! referring to it at a different position in an argument list or in an end
//! node.
//!
//! The operations are opaque: the tape knows only their boundary and whether
//! they are deterministic (the [`Operation`] trait). This includes the
//! operations on booleans. The tape only knows *which* sort is the boolean
//! one, since splits read it (see [Booleans](#booleans-what-the-tape-assumes)).
//!
//! # Routing DAGs
//!
//! The routes of an arm form a DAG, with the root last and children
//! stored before their parents:
//!
//! - [`Route::Split`] reads a boolean value, its *condition*, and continues
//!   with its `hi` child if the condition is true, with its `lo` child if it
//!   is false. The condition can be any value of the boolean sort: an input,
//!   or the output of any operation, including the theory's own `and` or
//!   `or`.
//! - [`Route::End`] ends a path: it names an output and lists the values that
//!   go to that output summand, one per sort of `Vᵢ`.
//!
//! A route may have several parents, so equal sub-DAGs are stored once.
//! There is no sharing *across* arms, because values are numbered per
//! arm: `v0` in one arm and `v0` in another are different values.
//!
//! # Non-determinism and binders
//!
//! Some operations are not deterministic, for instance sampling a
//! distribution (`sample : D → B`, where the distribution is itself a value
//! of some opaque sort `D`). Two deterministic nodes with the same operation
//! and arguments are the same node. Two samples are not: they are two
//! independent draws. All operations are total, so a draw nobody uses can
//! always be dropped.
//!
//! A non-deterministic node ([`NodeKind::NonDet`]) is therefore *bound* by the
//! route where the draw happens: every route lists, in its `binds`, the
//! draws made when a run reaches it, before testing or leaving. In normal
//! form a draw is bound at the lowest route that dominates all its uses,
//! so a sample needed by only one branch is drawn only on that branch.
//!
//! Draws are named by their **level**, the number of draws made on the path
//! before them (de Bruijn levels). The level is part of the node: two branches
//! that make "the same" draw at the same level then share the node. This is
//! what makes `case ; (f ⊕ f) ; merge` and `f` the same tape, even when `f`
//! samples.
//!
//! Levels limit sharing in the routing DAG, but only a little. A route
//! that binds draws, itself or below it, must be reached after the same
//! number of draws on every path, or its draws would have no well-defined
//! level. Such a sub-DAG reached at two depths is stored twice, with
//! different levels. Routes that bind nothing are shared freely, such as
//! an end node reached both from a branch that sampled and from one that did not
//! (`r0` in the example below).
//!
//! # Example
//!
//! Take sorts `R`, `D` (an opaque distribution over booleans) and the boolean
//! sort `B`, and operations `gt0 : R → B`, `zero : I → R` (both
//! deterministic) and `sample : D → B` (non-deterministic). The tape
//! `(R ⊗ D) ⊕ I → R ⊕ I` of
//!
//! ```text
//! match input {
//!     inl (x, d) => if gt0(x) then (let b = sample(d);
//!                                   if b then inl x else inr ())
//!                   else inr ()
//!     inr ()     => inl zero()
//! }
//! ```
//!
//! is, as printed by its [`Display`](std::fmt::Display) impl:
//!
//! ```text
//! (R ⊗ D) ⊕ I → R ⊕ I
//! arm 0: R ⊗ D
//!   in v0: R, v1: D
//!   n0  v2 = gt0(v0)
//!   n1  v3 = sample@0(v1)
//!   r0  end 0 (v0)
//!   r1  end 1 ()
//!   r2  bind n1; split v3 ? r0 : r1
//!   r3  split v2 ? r2 : r1
//! arm 1: I
//!   in (none)
//!   n2  v0 = zero()
//!   r4  end 0 (v0)
//! ```
//!
//! Arm 0 splits on `gt0(x)` first and only then draws `b`. The end node `r1`
//! (to `I`) is shared by both splits. The join of the two arms into `R` is
//! the two end nodes `r0` and `r4` with output 0. In memory this is:
//!
//! ```text
//! arms      = [ Arm { nodes: 0..2, routes: 0..4 },
//!               Arm { nodes: 2..3, routes: 4..5 } ]
//! nodes     = [ Node { kind: Det(gt0),                  args: 0..1, n_outs: 1 },
//!               Node { kind: NonDet(sample, level: 0),  args: 1..2, n_outs: 1 },
//!               Node { kind: Det(zero),                 args: ∅,    n_outs: 1 } ]
//! args      = [ v0, v1 ]
//! routes    = [ End   { binds: ∅,    output: 0, outs: 0..1 },
//!               End   { binds: ∅,    output: 1, outs: ∅ },
//!               Split { binds: 0..1, cond: v3, hi: r0, lo: r1 },
//!               Split { binds: ∅,    cond: v2, hi: r2, lo: r1 },
//!               End   { binds: ∅,    output: 0, outs: 1..2 } ]
//! bound     = [ n1 ]
//! outs      = [ v0, v0 ]
//! ```
//!
//! (`a..b` is a [`Span`] into the named array.) The tests of this module
//! build this tape and a larger one, translated from a small program.
//!
//! # Memory layout
//!
//! Every variable-length list (a node's arguments, a route's binders, an end
//! node's values) lives in one shared array per kind of list. The owner keeps
//! a [`Span`] into it; this is the compressed-sparse-row pattern. The indices
//! ([`Value`], [`NodeId`], [`RouteId`]) are `u32` newtypes:
//!
//! | array       | element      | indexed by       | slices referenced by     |
//! |-------------|--------------|------------------|--------------------------|
//! | `arms`      | [`Arm`]      | arm index        | —                        |
//! | `nodes`     | [`Node`]     | [`NodeId`]       | [`Arm::nodes`]           |
//! | `args`      | [`Value`]    | —                | [`Node::args`]           |
//! | `routes`    | [`Route`]    | [`RouteId`]      | [`Arm::routes`]          |
//! | `bound`     | [`NodeId`]   | —                | [`Route::binds`]         |
//! | `outs`      | [`Value`]    | —                | [`Route::End::outs`]     |
//!
//! Node and route ids are global (indices into `nodes` and `routes`),
//! while values are local to an arm. Indices always point *backwards*: a
//! node's arguments are defined before it, and a route's children come
//! before it. So a cycle cannot even be written down, evaluation is a single
//! forward pass, and checking a tape is a simple, decidable pass over the
//! arrays.
//!
//! # Well-formedness and normal form
//!
//! [`Tape::new`] checks **well-formedness**: the conditions under which the
//! arrays describe a tape at all. These are listed on [`Tape::new`]: indices
//! in bounds and pointing backwards, sorts matching the boundaries,
//! conditions being booleans, and draws bound before use with consistent
//! levels.
//!
//! The **normal form** is a stronger, canonical shape, which
//! [`Builder::freeze`] produces. It is *not* checked by [`Tape::new`]:
//!
//! 1. *Circuits:* deterministic nodes are hash-consed (no two with the same
//!    operation and arguments), and so are draws (no two with the same
//!    operation, arguments and level); there are no dead nodes or unused
//!    draws.
//! 2. *Routing DAGs:* reduced: no split with `hi == lo`, and no two equal
//!    routes.
//! 3. *Binders:* each draw is bound at the lowest route dominating its uses.
//!    Draws bound at the same route are in the order of their first use in
//!    the traversal below.
//! 4. *Numbering:* nodes and routes are numbered in the order of a
//!    depth-first traversal from the root of each arm: at a split, its
//!    condition, then `hi`, then `lo`, then the split itself; at an end, its
//!    values, then the end itself. A node comes after its arguments. So two
//!    isomorphic arms become equal arrays.
//! 5. *Spans:* the empty list of a node or a route (its arguments, binders
//!    or values) is always [`Span::EMPTY`]. The spans of an arm are not
//!    lists but its place in the arrays, so an arm without nodes has an
//!    empty span that starts where its nodes would.
//!
//! Splits are **not ordered** yet, as in an ordered decision diagram: two
//! programs that test independent conditions in different orders give
//! different tapes, and a value already tested on a path may be tested
//! again. (Ordering needs a canonical order on conditions. That order is
//! hard to define when a condition depends on a draw, since a draw has no
//! canonical name until its level is known.)
//!
//! The normal form is canonical with respect to the structural laws: the
//! laws of tapes with `⊕` and `⊗`, and copy and discard of strings. It is not
//! canonical with respect to equations between the opaque operations, such as
//! `x + y = y + x`, nor, since booleans are opaque too, with respect to the
//! laws of boolean logic.
//!
//! # Building tapes
//!
//! A [`Builder`] owns *drafts*, tapes under construction, referred to by
//! [`Draft`] handles. Its combinators each create a new draft:
//!
//! | combinator | type | |
//! |---|---|---|
//! | [`lift(op)`](Builder::lift) | `M → N` | one operation |
//! | [`case(m)`](Builder::case) | `m ⊗ B → m ⊕ m` | split on a boolean |
//! | [`copy(m)`](Builder::copy) | `m → m ⊗ m` | |
//! | [`discard(m)`](Builder::discard) | `m → I` | |
//! | [`cross(a, b)`](Builder::cross) | `a ⊗ b → b ⊗ a` | |
//! | [`wiring(m, outputs)`](Builder::wiring) | `m → N` | any mix of the three above: output `k` carries input `outputs[k]` |
//! | [`id(p)`](Builder::id) | `p → p` | |
//! | [`swap(p, q)`](Builder::swap) | `p ⊕ q → q ⊕ p` | |
//! | [`merge(p)`](Builder::merge) | `p ⊕ p → p` | join tapes |
//! | [`initial(p)`](Builder::initial) | `0 → p` | with `sum`, the injections |
//! | [`sum(t, s)`](Builder::sum) | `t ⊕ s` | |
//! | [`whisker(l, t, r)`](Builder::whisker) | `l ⊗ t ⊗ r` | strings alongside a tape |
//! | [`compose(t, s)`](Builder::compose) | `t ; s` | |
//!
//! Drafts are kept in a cheaper, unnormalized form. Any draft can be frozen
//! at any time with [`Builder::freeze`], which returns a [`Tape`] in normal
//! form and leaves the draft usable. [`Builder::import`] brings a tape back
//! as a draft. The tests build the two example tapes of this module from
//! combinators, and check laws such as `id ; t = t`, associativity of `;`, and
//! `case ; (f ⊕ f) ; merge = f`, as equalities of frozen tapes.
//!
//! # Booleans: what the tape assumes
//!
//! The tape assumes as little as possible about booleans: the theory has a
//! boolean sort ([`TapeParts::boolean`]), and splits read values of it. Every
//! operation on booleans (`and`, `or`, `not`, `true`, `false`, `x < y`, …) is
//! an opaque [`Operation`], like any other.
//!
//! **What this costs.** The normal form cannot use the laws of boolean
//! logic, so programs that are equal because of those laws get different
//! tapes:
//!
//! - *Compound guards are not broken down.* `if a or b` is one split on the
//!   value `or(a, b)`, while `if a: … elif b: …` (with the same branches) is
//!   a split on `a` and one on `b`. In the [example](#example), writing the
//!   guard as `gt0(x) ∧ b` would give one split on an `and` value. The sample
//!   would then have to be drawn at the root, always, instead of only when
//!   `gt0(x)` holds.
//! - *Negation is not seen.* `if not b then P else Q` and
//!   `if b then Q else P` differ.
//! - *Constants are not seen.* A split on `true` is not resolved.
//! - *Boolean outputs are not simplified along a path.* A boolean that every
//!   path to an end node has already tested is still output as the computed value,
//!   not as a constant. A caller that branches on it again cannot resolve
//!   that split.
//!
//! None of this is unsound. The tapes are just less canonical.
//!
//! **Possible improvements**, from least to most assuming:
//!
//! 1. *Normalize in the front end.* The translation from the source language
//!    could lower compound guards itself: `if a or b` into nested `case`s, and
//!    `not` into swapped branches. This needs nothing from the theory, but it
//!    only covers guards written in the source, not booleans computed and
//!    tested later.
//! 2. *Let the theory classify its boolean operations*, through an optional
//!    method of [`Operation`] such as
//!    `fn as_boolean(&self) -> Option<BoolOp> { None }`, where
//!    `BoolOp = Const(bool) | Not | And | Or`. The builder would
//!    break down guards into splits on the remaining booleans (*atoms*), and
//!    resolve constants. A theory that classifies nothing keeps the current
//!    behaviour. Simplifying outputs along a path also requires the builder
//!    to *create* constants, so the theory would also need to provide
//!    `true` and `false`.
//! 3. *Built-in connectives*, as node kinds owned by the tape (`Const`,
//!    `Not`, `And`, `Or` in [`NodeKind`], as in an earlier draft). This makes
//!    the tape self-contained: the well-formedness check and a Lean or SMT
//!    mirror know these nodes without asking the theory. But a theory that
//!    has its own `and` then has two ways to write it, and the builder must
//!    always translate one into the other.
//!
//! Option 2 is likely the best compromise, if the canonicity of guards turns
//! out to matter: guards like `x < y or x < z` are common.
//!
//! # Cost
//!
//! The normal form expands paths: in a sequence of `k` conditionals, what
//! follows each one is copied into both of its branches, so the size can be
//! exponential in `k`. This was accepted on purpose, in exchange for the
//! canonical form.
//!
//! # Assumptions to revisit
//!
//! - **All operations are total**, sampling included. This is what makes
//!   computing every branch's values in one shared circuit sound. Partial
//!   operations (division by zero, failing observations) could later be
//!   supported with a third kind of route, `Fail`, reached when a
//!   definedness guard fails: tapes would then be partial, like
//!   subdistributions.
//! - **Sampling is opaque**: the tape only distinguishes deterministic from
//!   non-deterministic operations. Whether distributions should be
//!   transparent to tapes is still open.
//!
//! # Not done yet
//!
//! - More normalization: resolving a split on a value already tested on the
//!   path, and ordering independent splits (see
//!   [Well-formedness and normal form](#well-formedness-and-normal-form)).
//! - Checking the normal form (only well-formedness is checked).
//! - `⊗` of two tapes: it needs a convention for distributing
//!   `(A ⊕ B) ⊗ (C ⊕ D)`, and it may not be needed beyond whiskering.

mod build;
mod canon;
mod check;
mod display;
mod ids;
mod layout;
mod node;
mod route;
mod view;

pub use build::{Builder, Draft, DraftRef};
pub use ids::{NodeId, RouteId, Span, Value};
pub use layout::{Arm, Tape, TapeParts};
pub use node::{Node, NodeKind, Operation};
pub use route::Route;
pub use view::{ArmRef, NodeRef, RouteRef};
