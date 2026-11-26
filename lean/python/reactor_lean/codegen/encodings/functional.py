"""Functional (init/update def) translation."""

from ..terms import translate_terms_lean, product_type
from ..context import LeanContext


def constants_block(ctx: LeanContext) -> str:
    """Render the top-level ``@[simp] def c0 : ...`` constant definitions."""
    defs = ctx.constants.defs()
    if not defs:
        return ""
    lines = ["/- Concrete constants -/", ""]
    lines.extend(defs)
    return "\n".join(lines)


def atom_to_lean_functional(ctx: LeanContext) -> str:
    """Generate the functional init/update definitions."""
    atom = ctx.atom

    init_body = translate_terms_lean(
        atom.init,
        ctx.init_wire_names,
        ctx.ctrl_next,
        ctx.constants,
    )
    update_body = translate_terms_lean(
        atom.update,
        ctx.update_wire_names,
        ctx.ctrl_next,
        ctx.constants,
    )

    lines = []
    cod = product_type(ctx.ctrl_next)
    noncomp = ctx.noncomputable

    def dom(groups) -> str:
        return " ".join(f"({p}: {product_type(w)})" for p, w in groups)

    lines.append(f"@[simp] {noncomp}def init {dom(ctx.init_groups)} : {cod} :=")
    lines.append(init_body)
    lines.append("")

    lines.append(f"@[simp] {noncomp}def update {dom(ctx.update_groups)} : {cod} :=")
    lines.append(update_body)
    lines.append("")

    return "\n".join(lines)
