"""The module as a `Relational.Module`: named variables, and atoms that
carry their functions, whose semantics is the functional encoding's.

The functional encoding states the module as `init` and `update` over
tuples, which is what `RM`, a `ReactiveTS`, is made of. This one states it
as ReactLib's reactive module (`Relational.Module`), so that the module and
its atoms can be taken apart, composed and reasoned about in Lean, and
proves that its semantics is that of `init` and `update`. In the namespace
`Reactive`:

- `Name`, one constructor per variable: `s<i>` the state variable `i`,
  `e<i>` the input `i` (`ctrl` and `extl` order, as the SMT names them);
  `Dom`, the type of each; `ctrlOf s` and `extlOf s`, the state `s` as the
  tuples `init` and `update` take.
- `atom<i>`, atom `i` of the module's order: an `Atom.ofFun` that awaits,
  reads and writes the variables the atom does, its functions the
  functional encoding's (`Atom<i>.init`/`update`, or the module's `init`
  and `update` when it has one atom) applied to them, a variable the atom
  neither reads nor awaits given as `default`. `atom<i>_init` and
  `atom<i>_update` say what its rounds are in terms of that function; that
  they hold says the atom reads and awaits no other variable.
- `module`, the `Relational.Module` of the atoms, and `wf`, that it is
  well-formed: its atoms are acyclic because each awaits only variables of
  atoms before it (`rank`).
- `initRound_iff`, `updateRound_iff`: the rounds of `module` are exactly
  those of `init` and `update`, the inputs read in the round before being
  those of the state before.
- `rts init_pre update_pre`, the `ReactiveTS` the certificate's `RM` is,
  and `run_traces`: a run of `module` whose inputs meet the preconditions
  is, on its state variables, one of its traces.
"""

from __future__ import annotations

from reactor import Var, X

from reactor_lean.codegen.context import LeanContext
from reactor_lean.codegen.encodings.functional import atom_namespace, awaited
from reactor_lean.codegen.terms import build_tuple, product_type
from reactor_lean.lowering.elements import product_accessor
from reactor_lean.lowering.sorts import dtype_to_lean_type

NAMESPACE = "Reactive"


class _Names:
    """The Lean name of each of the module's variables, by id."""

    def __init__(self, ctx: LeanContext):
        self.ctrl = [f"s{i}" for i in range(len(ctx.ctrl_latched))]
        self.extl = [f"e{i}" for i in range(len(ctx.extl_latched))]
        self.of: dict[int, str] = {
            v.id: n for v, n in zip(ctx.ctrl_latched, self.ctrl)}
        self.of.update({v.id: n for v, n in zip(ctx.extl_latched, self.extl)})
        self.all = self.ctrl + self.extl
        self.order = {n: k for k, n in enumerate(self.all)}

    def name(self, v: Var) -> str:
        try:
            return self.of[v.id]
        except KeyError:
            raise ValueError(f"variable {v} is not one of the module's") from None

    def finset(self, vs) -> str:
        """The `Finset Name` of `vs`, in the module's order."""
        names = sorted({self.name(v) for v in vs}, key=self.order.__getitem__)
        if not names:
            return "∅"
        return "{" + ", ".join(f".{n}" for n in names) + "}"


def _elem(val: str, name: str) -> str:
    """The value of `name` in the valuation `val`."""
    return f"({val} ⟨Name.{name}, by decide⟩)"


def _tuple_from(val: str, names: list[str], has: set[str]) -> str:
    """The tuple of `names` from the valuation `val`, which has the names
    `has`; a name it has not is `default`."""
    return build_tuple([_elem(val, n) if n in has else "default" for n in names])


def _state_tuple(state: str, names: list[str]) -> str:
    """The tuple of `names` in the state `state`, an argument as it is."""
    if len(names) == 1:
        return f"({state} .{names[0]})"
    return build_tuple([f"{state} .{n}" for n in names])


def _prod_ext(parts: list[str]) -> str:
    """The proof that two right-nested tuples are equal from the proofs that
    their components are."""
    if len(parts) == 1:
        return parts[0]
    return f"Prod.ext {parts[0]} ({_prod_ext(parts[1:])})"


def _atom(ctx: LeanContext, names: _Names, i: int, atom) -> list[str]:
    """`atom<i>`, its output valuation `atom<i>_out`, and the lemmas that
    say what its rounds are."""
    one = len(ctx.atoms) == 1
    a = f"atom{i}"
    # What the atom's functions are and give: the module's own `init` and
    # `update` for an atom of its own, in `ctrl` order; else the atom's, in
    # the atom's.
    if one:
        init_fn, update_fn = "_root_.init", "_root_.update"
        writes = list(ctx.ctrl_latched)
    else:
        ns = atom_namespace(i)
        init_fn, update_fn = f"_root_.{ns}.init", f"_root_.{ns}.update"
        writes = list(atom.ctrl)
    wnames = [names.name(v) for v in writes]
    waits = [] if one else [names.name(v) for v in awaited(ctx, atom)]
    reads = {names.name(v) for v in atom.read}
    awaits = {names.name(v) for v in atom.wait}
    out_ty = product_type([X(v) for v in writes])
    # The sets are named: Lean holds two copies of one `Finset` literal equal
    # by evaluating them, which takes it past its limits for a few elements.
    W = f"{a}_write"

    lines = [
        f"/-! ### Atom {i} -/",
        "",
        f"def {a}_await : Finset Name := {names.finset(atom.wait)}",
        f"def {a}_read : Finset Name := {names.finset(atom.read)}",
        f"def {W} : Finset Name := {names.finset(writes)}",
        "",
    ]
    # Its output, a valuation of what it writes.
    lines.append(f"/-- The valuation of what atom {i} writes that its "
                 "functions give. -/")
    lines.append(f"def {a}_out (o : {out_ty}) : Val Dom {W}")
    for k, n in enumerate(wnames):
        lines.append(f"  | ⟨.{n}, _⟩ => o{product_accessor(k, len(wnames))}")
    for n in names.all:
        if n not in wnames:
            lines.append(f"  | ⟨.{n}, h⟩ => absurd h (by decide)")
    lines.append("")
    lines += [
        f"theorem {a}_out_iff (s : State Dom) (o : {out_ty}) :",
        f"    s.restrict {W} = {a}_out o ↔ {_state_tuple('s', wnames)} = o := by",
        "  constructor",
        "  · intro h",
        "    exact " + _prod_ext(
            [f"(congrFun h ⟨.{n}, by decide⟩)" for n in wnames]),
        "  · rintro rfl",
        "    funext ⟨x, hx⟩",
        "    cases x <;> first | rfl | exact absurd hx (by decide)",
        "",
    ]

    # The atom: its functions are the encoding's, on what it reads and awaits.
    # `wait`, the next values of the variables of other atoms it awaits, when
    # it awaits any: from its valuation of them, and from a state.
    wait_val = [build_tuple([_elem("a", n) for n in waits])] if waits else []
    init_args = [_tuple_from("a", names.extl, awaits)] + wait_val
    update_args = [
        _tuple_from("r", names.ctrl, reads),
        _tuple_from("r", names.extl, reads),
        _tuple_from("a", names.extl, awaits),
    ] + wait_val
    lines += [
        f"/-- Atom {i}. -/",
        f"def {a} : Atom Dom := Atom.ofFun",
        f"  {a}_await {a}_read {W}",
        f"  (fun a => {a}_out ({init_fn} {' '.join(init_args)}))",
        f"  (fun r a => {a}_out ({update_fn} {' '.join(update_args)}))",
        "  (by decide)",
        "",
    ]

    # Its rounds, in terms of the encoding's functions.
    def wait_of(state: str) -> list[str]:
        return [_state_tuple(state, waits)] if waits else []

    init_call = " ".join([init_fn, "(extlOf s)"] + wait_of("s"))
    update_call = " ".join(
        [update_fn, "(ctrlOf s)", "(extlOf s)", "(extlOf s')"] + wait_of("s'"))
    lines += [
        f"theorem {a}_init (s : State Dom) :",
        f"    {a}.Init s ↔ {_state_tuple('s', wnames)} = {init_call} :=",
        f"  (Atom.ofFun_init s).trans ({a}_out_iff s _)",
        "",
        f"theorem {a}_update (s s' : State Dom) :",
        f"    {a}.Update s s' ↔ {_state_tuple("s'", wnames)} = {update_call} :=",
        f"  (Atom.ofFun_update s s').trans ({a}_out_iff s' _)",
        "",
    ]
    return lines


def _round(ctx: LeanContext, names: _Names, block: str) -> list[str]:
    """`initRound_iff` or `updateRound_iff`: the module's rounds are its
    atoms', which compose into the encoding's `init` or `update`."""
    n_atoms = len(ctx.atoms)
    lemmas = ", ".join(f"atom{i}_{block}" for i in range(n_atoms))
    if block == "init":
        name, rnd, binders, state = "initRound_iff", "InitRound", "(s : State Dom)", "s"
        rhs, args = "_root_.init (extlOf s)", "s"
        doc = "The initial rounds of `module` are those of `init`."
    else:
        name, rnd, binders, state = "updateRound_iff", "UpdateRound", "(s s' : State Dom)", "s'"
        rhs, args = "_root_.update (ctrlOf s) (extlOf s) (extlOf s')", "s s'"
        doc = ("The update rounds of `module` are those of `update`, the\n"
               "    inputs read in the round before being those of `s`.")
    lines = [
        f"/-- {doc} -/",
        f"theorem {name} {binders} :",
        f"    module.{rnd} {args} ↔ ctrlOf {state} = {rhs} := by",
        f"  rw [ctrlOf_eq {state}]",
        f"  simp only [Relational.Module.{rnd}, module, List.forall_mem_cons, List.not_mem_nil,",
        f"    IsEmpty.forall_iff, implies_true, and_true, {lemmas}]",
    ]
    if n_atoms > 1:
        hs = [f"h{i}" for i in range(n_atoms)]
        cs = [f"c{k}" for k in range(len(names.ctrl))]
        lines += [
            f"  rw [_root_.{block}]",
            "  constructor",
            f"  · rintro ⟨{', '.join(hs)}⟩",
            f"    simp only [{', '.join('← ' + h for h in hs)}]",
            "  · intro h",
            "    simp only [Prod.mk.injEq] at h",
            f"    obtain ⟨{', '.join(cs)}⟩ := h",
            f"    simp only [{', '.join(cs)}, Prod.mk.eta, and_self]",
        ]
    lines.append("")
    return lines


def to_lean_module(ctx: LeanContext) -> str:
    names = _Names(ctx)
    ctrl_ty = product_type(ctx.ctrl_latched)
    extl_ty = product_type(ctx.extl_latched)
    rank = {n: 0 for n in names.extl}
    for i, atom in enumerate(ctx.atoms):
        for v in atom.ctrl:
            rank[names.name(v)] = i + 1

    lines = [
        "noncomputable section",
        "",
        f"namespace {NAMESPACE}",
        "",
        "open Relational",
        "",
        "/-- The module's variables: `s<i>` its state variable `i`, `e<i>` its input `i`. -/",
        "inductive Name where",
        "  | " + " | ".join(names.all),
        "  deriving DecidableEq, Repr",
        "",
        "/-- The type of each variable's values. -/",
        "abbrev Dom : Name → Type",
    ]
    for v, n in zip(ctx.ctrl_latched + ctx.extl_latched, names.all):
        lines.append(f"  | .{n} => {dtype_to_lean_type(v)}")
    lines += [
        "",
        "/-- The state variables of `s`, as `init` and `update` take and give them. -/",
        f"def ctrlOf (s : State Dom) : {ctrl_ty} :=",
        f"  {_state_tuple('s', names.ctrl)}",
        "",
        f"theorem ctrlOf_eq (s : State Dom) : ctrlOf s = {_state_tuple('s', names.ctrl)} := rfl",
        "",
        "/-- The inputs of `s`, as `init` and `update` take them. -/",
        f"def extlOf (s : State Dom) : {extl_ty} :=",
        f"  {_state_tuple('s', names.extl)}",
        "",
    ]
    for i, atom in enumerate(ctx.atoms):
        lines += _atom(ctx, names, i, atom)

    module = ctx.module
    lines += [
        "/-! ### The module -/",
        "",
        "def module : Relational.Module Dom where",
        f"  intf := {names.finset(module.intf)}",
        f"  extl := {names.finset(module.extl)}",
        f"  priv := {names.finset(module.prvt)}",
        f"  atoms := [{', '.join(f'atom{i}' for i in range(len(ctx.atoms)))}]",
        "",
        "/-- A variable's place in the order the atoms run in: an input's is 0,",
        "    and atom `i` writes variables of place `i + 1`. -/",
        "def rank : Name → Nat",
    ]
    lines += [f"  | .{n} => {rank[n]}" for n in names.all]
    lines += [
        "",
        "theorem wf : module.WF where",
        "  intf_extl_disj := by decide",
        "  intf_priv_disj := by decide",
        "  extl_priv_disj := by decide",
        "  atoms_pairwise_disj := by decide",
        "  atoms_cov_ctrl := by decide",
        "  atoms_write_ctrl := by decide",
        "  atoms_read_vars := by decide",
        "  atoms_await_vars := by decide",
        "  -- each atom awaits only variables of a lower place than those it writes",
        "  atoms_acyclic := Atom.acyclic_of_rank rank (by decide)",
        "",
    ]
    lines += _round(ctx, names, "init")
    lines += _round(ctx, names, "update")
    lines += [
        "/-- The module as a `ReactiveTS` over `init` and `update`, its inputs",
        "    constrained by `init_pre` and `update_pre`: the certificate's `RM` is",
        "    `rts init_pre update_pre`. -/",
        f"def rts (init_pre update_pre : ({extl_ty}) × ({extl_ty}) → Prop) :",
        f"    ReactiveTS (({extl_ty}) × ({extl_ty})) ({ctrl_ty}) where",
        "  init e := _root_.init e.2",
        "  update x e := _root_.update x e.1 e.2",
        "  init_pre := init_pre",
        "  update_pre := update_pre",
        "",
        "theorem rts_init (init_pre update_pre e) :",
        "    (rts init_pre update_pre).init e = _root_.init e.2 := by",
        "  simp only [rts]",
        "",
        "theorem rts_update (init_pre update_pre x e) :",
        "    (rts init_pre update_pre).update x e = _root_.update x e.1 e.2 := by",
        "  simp only [rts]",
        "",
        "/-- A run of `module` whose inputs meet `init_pre` and `update_pre` is, on",
        "    its state variables, a trace of `rts init_pre update_pre`. -/",
        "theorem run_traces {init_pre update_pre} {ss : Cslib.ωSequence (State Dom)}",
        "    (h : module.Run ss) (h0 : ∃ el, init_pre (el, extlOf (ss 0)))",
        "    (hi : ∀ i, update_pre (extlOf (ss i), extlOf (ss (i + 1)))) :",
        "    (rts init_pre update_pre).traces (ss.map ctrlOf) := by",
        "  obtain ⟨el, h0⟩ := h0",
        "  refine h.reactiveTS_traces _ ctrlOf (fun s => (el, extlOf s))",
        "    (fun s s' => (extlOf s, extlOf s')) ?_ ?_ h0 hi",
        "  · intro s hs; rw [rts_init]; exact (initRound_iff s).mp hs",
        "  · intro s s' hs; rw [rts_update]; exact (updateRound_iff s s').mp hs",
        "",
        f"end {NAMESPACE}",
        "",
        "end",
    ]
    return "\n".join(lines)
