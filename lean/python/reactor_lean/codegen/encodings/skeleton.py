"""The shape both relational encodings have, written once.

`Rel` (matrix domain) and `ScalarRel` (flat scalars) emit the same three
things in the same order: a per-slot body def (`effect_i` / `init_i`), a
per-slot relation (`R_i` / `Init_i`) pinning one slot of the next state to
that body, and a conjunction over them (`TransRel` / `InitCond`). Each
body is proved equal to its slot of the functional counterpart (`*_eq`) by
`rfl`: both sides come from the same body producer over the same reachable
terms, so they are syntactically equal.

The two differ in what a state value is -- a product of `Matrix`es, or one
component per element -- and that is what :class:`RelSyntax` holds: the
type of a slot, and how slot `i` is read out of a state. The bodies are
each encoding's own, handed to :func:`emit_rel_block` as text.
"""

from dataclasses import dataclass
from typing import Callable


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


def _decl(name: str, *binders: str) -> str:
    """`def name binders… : Prop :=`, skipping the binder groups that are
    empty: an encoding with no external inputs has declarations one binder
    group short of the others'."""
    parts = ["def", name, *[b for b in binders if b]]
    return f"{' '.join(parts)} : Prop :="


def _apply(head: str, *args: str) -> str:
    """`head args…`, skipping the argument groups that are empty."""
    return " ".join([head, *[a for a in args if a]])


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

    for i in slots:
        lines.append(_decl(f"{blk.rel_name}_{i}", blk.state_binders,
                           blk.extl_binders))
        lines.append(f"  {syn.project(blk.target, i)} = "
                     f"{blk.body_name}_{i}{blk.args}")
        lines.append("")

    calls = [_apply(f"{blk.rel_name}_{i}", blk.state_args, blk.extl_args)
             for i in slots]
    lines.append(_decl(blk.conj_name, blk.state_binders, blk.extl_binders))
    # A module with no state has nothing to conjoin.
    lines.append("  " + (" ∧\n  ".join(calls) if calls else "True"))
    lines.append("")

    return lines
