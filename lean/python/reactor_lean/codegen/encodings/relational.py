"""The two provable relational encodings: `Rel` (matrix) and `ScalarRel` (flat).

Both have one shape — a body def, a relation and a conjunction per slot
(`emit_rel_block`) — over their own type builder (`product_type` vs
`product_type_scalar`), body producer (`translate_terms_lean` vs
`translate_terms_scalar`) and projection (a product accessor vs a re-tupled
slice of the flattened element tuple).

What the two do *not* share is the closing theorem. `Rel` relates its
conjunction to the matrix `update`/`init` directly. `ScalarRel` relates its
own to `Scalar.update`/`Scalar.init`, and then bridges to the matrix form
through `Scalar.pack ∘ Scalar.unpack_ctrl` — the round trip the two state
types differ by.
"""

from dataclasses import dataclass
from typing import Callable

from reactor_lean.codegen.terms import (
    product_type,
    product_type_scalar,
    translate_terms_lean,
    translate_terms_scalar,
    reachable_terms,
)
from reactor_lean.lowering.sorts import dtype_to_lean_type
from reactor_lean.lowering.elements import product_accessor, flat_layout
from reactor_lean.codegen.context import LeanContext
from reactor_lean.codegen.encodings.common import (
    has_block,
    scalar_eq_name,
    scalar_bindings_with_recon,
    prepend_recon,
    flat_slice,
    effect_type,
)


# ══════════════════════════════════════════════════════════════════════════
# The shape both encodings have
# ══════════════════════════════════════════════════════════════════════════
# A per-slot body def (`effect_i` / `init_i`), a per-slot relation (`R_i` /
# `Init_i`) pinning one slot of the next state to that body, and a
# conjunction over them (`TransRel` / `InitCond`). Each body is proved equal
# to its slot of the functional counterpart (`*_eq`) by `rfl`: both sides
# come from the same body producer over the same reachable terms. The two
# encodings differ in what a state value is, which is what `RelSyntax`
# holds; the bodies are each encoding's own, handed over as text.


@dataclass(frozen=True)
class RelSyntax:
    """What a slot is in one encoding."""

    noncomputable: bool  # the bodies compute with reals
    slot_ty: Callable[[int], str]  # codomain of slot `i`'s body
    project: Callable[[str, int], str]  # slot `i` of a state expression
    # Extra `simp only` lemmas the `*_eq` proofs need. The scalar domain's
    # bodies come out of `Fin.cons`, the matrix domain's do not.
    eq_simp: "tuple[str, ...]" = ()


@dataclass(frozen=True)
class RelBlock:
    """One half of a relational encoding: the transition, or the init
    condition."""

    body_name: str  # `effect` / `init`
    rel_name: str  # `R` / `Init`
    conj_name: str  # `TransRel` / `InitCond`
    state_binders: str  # what quantifies the state(s): `(old new : S)`
    state_args: str  # and their names: `old new`
    target: str  # the state each slot is projected out of: `new`
    binders: str  # what every body def binds (leading space)
    args: str  # what every relation applies its body to (leading space)
    extl_binders: str  # the external inputs, carried by every declaration
    extl_args: str
    ref: str  # the functional counterpart each body is proved equal to
    ref_binders: str
    ref_args: str


def _eq_theorem(syn: RelSyntax, blk: RelBlock, i: int) -> "list[str]":
    """`body_i … = (ref …)[slot i]`, the theorem that ties one slot to its source.

    `rfl` closes it; the tactics after it are for the shapes where a 1×1
    `Matrix` meets a scalar and the equality is only propositional.
    """
    name = f"{blk.body_name}_{i}"
    lemmas = ", ".join([blk.ref, name, *syn.eq_simp])
    return [
        f"theorem {name}_eq : ∀ {blk.ref_binders},",
        f"    {name} {blk.ref_args} = "
        f"{syn.project(f'({blk.ref} {blk.ref_args})', i)} := by",
        f"  intro {blk.ref_args}",
        f"  simp only [{lemmas}]",
        "  try rfl",
        "  try (apply Prod.ext <;> funext i j <;> simp [Fin.fin_one_eq_zero])",
        "  try simp [Fin.fin_one_eq_zero]",
        "  try omega",
        "",
    ]


def emit_rel_block(
    syn: RelSyntax, blk: RelBlock, bodies: "list[str]"
) -> "list[str]":
    """Emit one block: the bodies, their `*_eq` theorems, the relations, the
    conjunction.

    `bodies` is the body producer's output, one per slot, already indented
    the way that producer indents.
    """
    slots = range(len(bodies))
    noncomp = "noncomputable " if syn.noncomputable else ""
    lines: "list[str]" = []

    for i, body in enumerate(bodies):
        lines.append(
            f"@[simp] {noncomp}def {blk.body_name}_{i}{blk.binders} "
            f": {syn.slot_ty(i)} :="
        )
        lines.append(body)
        lines.append("")

    for i in slots:
        lines.extend(_eq_theorem(syn, blk, i))

    # Every relation binds the state(s) and the inputs, even a module's
    # with none: an empty group is `(extl_n : Unit)`, not left out.
    for i in slots:
        lines.append(f"def {blk.rel_name}_{i} {blk.state_binders} "
                     f"{blk.extl_binders} : Prop :=")
        lines.append(f"  {syn.project(blk.target, i)} = "
                     f"{blk.body_name}_{i}{blk.args}")
        lines.append("")

    calls = [f"{blk.rel_name}_{i} {blk.state_args} {blk.extl_args}"
             for i in slots]
    lines.append(f"def {blk.conj_name} {blk.state_binders} "
                 f"{blk.extl_binders} : Prop :=")
    # A module with no state has nothing to conjoin.
    lines.append("  " + (" ∧\n  ".join(calls) if calls else "True"))
    lines.append("")

    return lines


# ══════════════════════════════════════════════════════════════════════════
# The two encodings
# ══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class _Domain:
    """The binders one relational encoding quantifies over."""

    state_ty: str
    update_binders: str
    update_args: str
    extl_binders: str
    extl_args: str
    extl_n_binder: str


def _domain(ctx: LeanContext, ty) -> _Domain:
    """The binders of a relational encoding, in one type builder.

    `ty` is the only thing the matrix and scalar domains disagree about here:
    `product_type` gives each wire its own `Matrix`, `product_type_scalar` one
    component per element. Both answer "Unit" for an empty group, so a module
    with no external inputs still gets the binder written out rather than
    dropped — the call sites are built from the same lists.
    """
    update_groups = ctx.update_groups
    extl_groups = update_groups[1:]

    def binders(groups):
        return " ".join(f"({p} : {ty(w)})" for p, w in groups)

    def args(groups):
        return " ".join(p for p, _ in groups)

    return _Domain(
        state_ty=ty(ctx.ctrl_next),
        update_binders=binders(update_groups),
        update_args=args(update_groups),
        extl_binders=binders(extl_groups),
        extl_args=args(extl_groups),
        extl_n_binder=f"(extl_n : {ty(ctx.extl_next)})",
    )


def _conj_eq_theorem(
    name: str,
    blk: RelBlock,
    n: int,
    rhs: str,
) -> "list[str]":
    """`conj … ↔ state = ref …`: the conjunction is the functional form.

    Unfolding the relations and rewriting each slot by its own `*_eq` leaves
    a conjunction of component equalities against a tuple equality, which is
    `Prod.ext_iff`; `tauto` finishes when the state is a single component and
    there is nothing to split.
    """
    simp = (
        [blk.conj_name]
        + [f"{blk.rel_name}_{i}" for i in range(n)]
        + [f"{blk.body_name}_{i}_eq" for i in range(n)]
        + ["Prod.ext_iff"]
    )
    return [
        f"theorem {name} : ∀ {blk.state_binders} {blk.extl_binders},",
        f"    {blk.conj_name} {blk.state_args} {blk.extl_args} ↔ {rhs} := by",
        f"  intro {blk.state_args} {blk.extl_args}",
        f"  simp only [{', '.join(simp)}]",
        "  try tauto",
        "",
    ]


def _update_block(dom: _Domain, ref: str) -> RelBlock:
    """The transition half: the state twice, the inputs alongside it.

    `ref` is the functional counterpart each slot is proved equal to —
    `update` in the matrix domain, `Scalar.update` in the flat one. It is the
    only thing the two encodings disagree about here; the body producer they
    disagree about is not a parameter of the block.
    """
    return RelBlock(
        body_name="effect",
        rel_name="R",
        conj_name="TransRel",
        state_binders=f"(old new : {dom.state_ty})",
        state_args="old new",
        target="new",
        binders=f" {dom.update_binders}",
        args=f" old {dom.extl_args}",
        extl_binders=dom.extl_binders,
        extl_args=dom.extl_args,
        ref=ref,
        ref_binders=dom.update_binders,
        ref_args=dom.update_args,
    )


def _init_block(dom: _Domain, ref: str) -> RelBlock:
    """The initial half: one state, and only the next-input group in scope."""
    return RelBlock(
        body_name="init",
        rel_name="Init",
        conj_name="InitCond",
        state_binders=f"(s : {dom.state_ty})",
        state_args="s",
        target="s",
        binders=f" {dom.extl_n_binder}",
        args=" extl_n",
        extl_binders=dom.extl_n_binder,
        extl_args="extl_n",
        ref=ref,
        ref_binders=dom.extl_n_binder,
        ref_args="extl_n",
    )


def _pack_roundtrip(var: str) -> "list[str]":
    """Tactics for ``var = Scalar.pack (Scalar.unpack_ctrl var)``.

    The functional and scalar states differ by exactly this round trip.
    Widest tactic first, as in `scalar.py`: a multi-element wire leaves a
    `match i, j with` on a symbolic `i` that only `fin_cases` clears, and
    once `i`/`j` are introduced a later `funext` can no longer fire.
    """
    return [
        f"    have hpack : {var} = Scalar.pack (Scalar.unpack_ctrl {var}) := by",
        "      simp only [Scalar.pack, Scalar.unpack_ctrl, ← Matrix_1_1_eq, Prod.eta]",
        "      try rfl",
        "      try (apply Prod.ext <;> funext i j <;> fin_cases i <;> fin_cases j <;> simp)",
        "      try (funext i j <;> fin_cases i <;> fin_cases j <;> simp)",
        "      try (apply Prod.ext <;> funext i j <;> simp [Fin.fin_one_eq_zero])",
        "      try (funext i j; simp [Fin.fin_one_eq_zero])",
    ]


def _pack_bridge_theorem(
    name: str, binders: str, intro: str, lhs: str, rhs: str, rewrites: "list[str]"
) -> "list[str]":
    """The same statement as `_conj_eq_theorem`, one domain up.

    `ScalarRel`'s conjunction is already known to be the scalar functional
    form (`*_scalar_eq`), and `scalar.py` proves that to be the matrix one
    (`update_scalar_eq` / `init_scalar_eq`); `rewrites` chains the two. What
    is left is the `pack ∘ unpack` round trip the two state types differ by
    -- forwards by rewriting with it, backwards by `simp` on its two
    definitions.
    """
    return [
        f"theorem {name} : ∀ {binders},",
        f"    {lhs} ↔ {rhs} := by",
        f"  intro {intro}",
        f"  rw [{', '.join(rewrites)}]",
        "  constructor",
        "  · intro h",
        *_pack_roundtrip("ctrl'"),
        "    rw [h] at hpack; exact hpack",
        "  · intro h",
        "    simp [Scalar.pack, Scalar.unpack_ctrl, h]",
        "",
    ]


def _unpack(param: str, wires, var: str = "") -> str:
    """`var`, read back as a matrix by `Scalar.unpack_<param>`.

    Left alone when the group is empty: `unpack_extl_l` of no wires is `Unit`
    to `Unit`, and `scalar.py` does not emit it. `var` defaults to the
    parameter's own name; the two init/trans theorems quantify the *next*
    ctrl state as `ctrl'` but still read it back with `unpack_ctrl`.
    """
    var = var or param
    return f"(Scalar.unpack_{param} {var})" if wires else var


def atom_to_lean_mat_rel(ctx: LeanContext) -> str:
    """Generate the matrix-domain relational encoding inside ``namespace Rel``."""
    n_ctrl = len(ctx.ctrl_next)
    dom = _domain(ctx, product_type)

    syn = RelSyntax(
        noncomputable=ctx.uses_real,
        slot_ty=lambda i: dtype_to_lean_type(ctx.ctrl_next[i]),
        project=lambda e, i: f"{e}{product_accessor(i, n_ctrl)}",
    )

    def _bodies(terms, bindings) -> "list[str]":
        return [
            translate_terms_lean(
                reachable_terms(terms, [w]), bindings, [w], ctx.constants
            )
            for w in ctx.ctrl_next
        ]

    lines = ["namespace Rel", ""]

    if has_block(ctx, "update"):
        blk = _update_block(dom, "update")
        lines += emit_rel_block(
            syn, blk, _bodies(ctx.atom.update, ctx.update_wire_names)
        )
        lines += _conj_eq_theorem(
            "TransRel_func_eq", blk, n_ctrl, f"new = update old {dom.extl_args}"
        )

    if has_block(ctx, "init"):
        blk = _init_block(dom, "init")
        lines += emit_rel_block(syn, blk, _bodies(ctx.atom.init, ctx.init_wire_names))
        lines += _conj_eq_theorem(
            "InitCond_func_eq", blk, n_ctrl, "s = init extl_n"
        )

    lines.append("end Rel")
    return "\n".join(lines)


def atom_to_lean_rel(ctx: LeanContext) -> str:
    """Generate the relational encoding inside ``namespace ScalarRel``."""
    # One `effect_i`/`R_i` per ctrl *wire*, but one state component per
    # *element*: `Scalar.update`'s codomain is the flattened tuple, so wire
    # `i`'s data sits at its own span in it, not at position `i`.
    layout = flat_layout(ctx.ctrl_next)
    n_ctrl = len(ctx.ctrl_next)
    dom = _domain(ctx, product_type_scalar)

    syn = RelSyntax(
        noncomputable=ctx.uses_real,
        slot_ty=lambda i: effect_type(ctx.ctrl_next[i]),
        project=lambda e, i: flat_slice(e, layout, ctx.ctrl_next[i]),
        eq_simp=("Fin.cons_zero", "Fin.cons_succ"),
    )

    def _bodies(terms, groups) -> "list[str]":
        recon, bindings, flat = scalar_bindings_with_recon(groups)
        return [
            prepend_recon(
                recon,
                translate_terms_scalar(
                    reachable_terms(terms, [w]),
                    bindings,
                    [w],
                    ctx.constants,
                    flat_slots=flat,
                ),
            )
            for w in ctx.ctrl_next
        ]

    lines = ["namespace ScalarRel", ""]

    # The matrix domain of the same module: what the two bridging theorems
    # quantify over, since they state the scalar relation in matrix terms.
    mat = _domain(ctx, product_type)
    new_unpack = _unpack("ctrl", ctx.ctrl_next, "ctrl'")

    if has_block(ctx, "update"):
        blk = _update_block(dom, "Scalar.update")
        lines += emit_rel_block(syn, blk, _bodies(ctx.atom.update, ctx.update_groups))
        lines += _conj_eq_theorem(
            "TransRel_scalar_eq", blk, n_ctrl,
            f"new = Scalar.update old {dom.extl_args}",
        )
        old_unpack = _unpack("ctrl", ctx.ctrl_latched)
        extl_unpacks = " ".join(_unpack(p, w) for p, w in ctx.update_groups[1:])
        lines += _pack_bridge_theorem(
            "TransRel_func_eq",
            f"(ctrl ctrl' : {product_type(ctx.ctrl_latched)}) {mat.extl_binders}",
            f"ctrl ctrl' {mat.extl_args}",
            f"TransRel {old_unpack} {new_unpack} {extl_unpacks}",
            f"ctrl' = update ctrl {mat.extl_args}",
            ["TransRel_scalar_eq", scalar_eq_name("update")],
        )

    if has_block(ctx, "init"):
        blk = _init_block(dom, "Scalar.init")
        lines += emit_rel_block(syn, blk, _bodies(ctx.atom.init, ctx.init_groups))
        lines += _conj_eq_theorem(
            "InitCond_scalar_eq", blk, n_ctrl, "s = Scalar.init extl_n"
        )
        extl_n_unpack = _unpack("extl_n", ctx.extl_next)
        lines += _pack_bridge_theorem(
            "InitCond_func_eq",
            f"(ctrl' : {mat.state_ty}) {mat.extl_n_binder}",
            "ctrl' extl_n",
            f"InitCond {new_unpack} {extl_n_unpack}",
            "ctrl' = init extl_n",
            ["InitCond_scalar_eq", scalar_eq_name("init")],
        )

    lines.append("end ScalarRel")
    return "\n".join(lines)
