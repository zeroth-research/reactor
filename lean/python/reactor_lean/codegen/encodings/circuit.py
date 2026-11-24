"""Circuit (Box algebra) translation."""

from reactor_lean.codegen.terms import product_type
from reactor_lean.codegen.boxes import (
    CircLayer,
    ty_list,
    native_to_vt,
    natives_to_vt,
    translate_terms_circ,
)
from reactor_lean.codegen.context import LeanContext

# The heartbeat budget of `init_circ_eq` and `update_circ_eq`; see where they
# are stated.
_CIRC_HEARTBEATS = 4000000

# Simp lemmas for reducing a single circuit layer
_LAYER_SIMP = [
    "Box.par",
    "ValTuple.split",
    "ValTuple.append",
    "ValTuple.append_split",
    "ValTuple.append_ite",
    "ValTuple.split_singleton_fst",
    "ValTuple.split_singleton_snd",
    "ValTuple.split_cons_fst_fst",
    "ValTuple.split_cons_fst_snd",
    "ValTuple.split_2_fst",
    "ValTuple.split_2_snd",
    "ValTuple.split_3_fst",
    "ValTuple.split_3_snd",
    "ValTuple.split_nil",
    "ValTuple.split_nil_snd",
    "Box.id",
    "Box.dup",
    "Box.swap",
    "Box.destr",
    "Box.const",
    "Box.not",
    "Box.and",
    "Box.or",
    "Box.implies",
    "Box.ite",
    "Box.add",
    "Box.sub",
    "Box.mul",
    "Box.neg",
    "Box.hadamard",
    "Box.div",
    "Box.mod",
    "Box.lt",
    "Box.le",
    "Box.gt",
    "Box.ge",
    "Box.eq",
    "Box.neq",
    "Box.min",
    "Box.max",
    "Box.nnLinear",
    "Box.linear",
    "Box.relu",
    "Box.get",
    "Box.intToReal",
    "Box.realToInt",
    "Box.bvToInt",
    "Box.bitToBool",
    "Box.boolToBit",
    "Box.argmax_1d",
    "Box.argmax",
    "ite_pair",
    "Matrix_1_1_lt_iff",
    "Matrix_1_1_le_iff",
    "↓Matrix.mk_apply",
]


def _emit_named_layers(
    ctx: LeanContext,
    block_name: str,
    circ_layers: "list[CircLayer]",
    dom: str,
    cod: str,
) -> "tuple[list[str], list[str]]":
    """Emit named @[simp] definitions for each layer and a composed definition."""
    lines: list[str] = []
    layer_names: list[str] = []
    noncomp = "noncomputable " if ctx.uses_real else ""

    for i, layer in enumerate(circ_layers):
        name = f"{block_name}_l{i}"
        in_tys = ", ".join(layer.in_tys)
        out_tys = ", ".join(layer.out_tys)
        lines.append(f"@[simp] {noncomp}def {name} : Box [{in_tys}] [{out_tys}] :=")
        lines.append(f"  {layer.body}")
        lines.append("")
        layer_names.append(name)

    lines.append(f"@[simp] {noncomp}def {block_name} : Box {dom} {cod} :=")
    lines.append(f"  {' ≫ '.join(layer_names)}")
    lines.append("")

    return lines, layer_names


def _simp_circ_macro() -> str:
    """Generate the simp_circ helper tactic macro."""
    layer_simp = ",\n    ".join(_LAYER_SIMP)
    lines = []
    lines.append("/-- Reduce one circuit layer: unfolds the given lemma,")
    lines.append("    then simplifies all Box/ValTuple plumbing. -/")
    lines.append(
        'macro "simp_circ" "[" ls:Lean.Parser.Tactic.simpLemma,* "]" : tactic =>'
    )
    lines.append("  `(tactic| simp only [$ls,*,")
    lines.append(f"    {layer_simp}])")
    return "\n".join(lines)


def _equiv_proof_tactic(
    ctx: LeanContext,
    intro_vars: "list[str]",
    layer_names: "list[str]",
    block_name: str,
) -> str:
    """Generate layer-by-layer proof tactic for equivalence theorem."""
    const_names = ", ".join(ctx.constants.names())

    proof = []
    proof.append(f"  intro {' '.join(intro_vars)}")
    proof.append(f"  simp_circ [Circ.{block_name}, Box.seq]")
    for name in layer_names:
        proof.append(f"  simp_circ [Circ.{name}]")
    final_simp = [block_name, "ite_pair"]
    if const_names:
        final_simp.append(const_names)
    proof.append(f"  simp [{', '.join(final_simp)}]")
    proof.append("  try exact List.ofFn_inj.mp rfl")
    proof.append("  try grind")
    proof.append("  try omega")
    proof.append("  try simp")
    proof.append("  try grind")
    proof.append(f"  try simp [{', '.join(final_simp)}]")
    proof.append("  try exact List.ofFn_inj.mp rfl")
    proof.append("  try omega")
    return "\n".join(proof)


def atom_to_lean_circuit(
    ctx: LeanContext,
) -> "tuple[str, list[str], list[str]]":
    """Generate the full Lean4 source for this module as a combinational circuit.

    Returns (source_string, init_layer_names, update_layer_names).
    """
    atom = ctx.atom

    init_inputs = ctx.extl_next
    init_outputs = ctx.ctrl_next
    init_layers = translate_terms_circ(
        atom.init,
        (init_inputs,),
        init_outputs,
        ctx.constants,
    )

    update_inputs = tuple(w for _, w in ctx.update_groups)
    update_outputs = ctx.ctrl_next
    update_layers = translate_terms_circ(
        atom.update,
        update_inputs,
        update_outputs,
        ctx.constants,
    )

    lines = ["namespace Circ"]
    init_layer_names: list[str] = []
    update_layer_names: list[str] = []

    if init_layers:
        init_dom = ty_list(init_inputs)
        init_cod = ty_list(init_outputs)
        layer_lines, init_layer_names = _emit_named_layers(
            ctx,
            "init",
            init_layers,
            init_dom,
            init_cod,
        )
        lines.extend(layer_lines)

    if update_layers:
        upd_dom = ty_list([w for ws in update_inputs for w in ws])
        upd_cod = ty_list(update_outputs)
        layer_lines, update_layer_names = _emit_named_layers(
            ctx,
            "update",
            update_layers,
            upd_dom,
            upd_cod,
        )
        lines.extend(layer_lines)

    lines.append("end Circ")
    return "\n".join(lines), init_layer_names, update_layer_names


def to_lean_equiv_theorems(
    ctx: LeanContext,
    init_layer_names: "list[str]",
    update_layer_names: "list[str]",
) -> str:
    """Generate theorems proving circuit ≡ functional."""
    lines: list[str] = []

    has_theorems = bool(init_layer_names or update_layer_names)
    if has_theorems:
        lines.append(_simp_circ_macro())
        lines.append("")

    n_ctrl = len(ctx.ctrl_next)

    if init_layer_names:
        n_extl_n = len(ctx.extl_next)
        init_binder = f"(extl_n : {product_type(ctx.extl_next)})"
        lhs_input = natives_to_vt([("extl_n", n_extl_n)])
        rhs_output = native_to_vt("r", n_ctrl)

        # One `simp_circ` per layer, and a layer's width grows with the
        # number of output-to-input paths through the block, so the proof's
        # cost is set by the module rather than by this file: a Petri net
        # with a dozen place-transition incidences exceeds the default
        # budget, and `System/Circ.lean` then fails to compile at all --
        # BUILD-FAIL, whatever the certificate, for a module that is otherwise
        # fine.
        # The same budget `scalar.py` raises for the same reason, and per
        # theorem rather than file-wide. Twice `scalar.py`'s: the dining
        # philosophers' `update_circ_eq` took 1M-2M heartbeats while the matrix type
        # was a function type and 2M-4M since it is Mathlib's `Matrix`.
        lines.append(f"set_option maxHeartbeats {_CIRC_HEARTBEATS} in")
        lines.append("set_option maxRecDepth 100000 in")
        lines.append(f"theorem init_circ_eq : ∀ {init_binder},")
        lines.append(f"    Circ.init.fn {lhs_input} =")
        lines.append("    let r := init extl_n")
        lines.append(f"    {rhs_output} := by")
        lines.append(
            _equiv_proof_tactic(
                ctx,
                ["extl_n"],
                init_layer_names,
                "init",
            )
        )
        lines.append("")

    if update_layer_names:
        update_params = [p for p, _ in ctx.update_groups]
        update_binders = " ".join(
            f"({p} : {product_type(w)})" for p, w in ctx.update_groups
        )
        lhs_input = natives_to_vt([(p, len(w)) for p, w in ctx.update_groups])
        rhs_output = native_to_vt("r", n_ctrl)

        lines.append(f"set_option maxHeartbeats {_CIRC_HEARTBEATS} in")
        lines.append("set_option maxRecDepth 100000 in")
        lines.append(f"theorem update_circ_eq : ∀ {update_binders},")
        lines.append(f"    Circ.update.fn {lhs_input} =")
        lines.append(f"    let r := update {' '.join(update_params)}")
        lines.append(f"    {rhs_output} := by")
        lines.append(
            _equiv_proof_tactic(
                ctx,
                update_params,
                update_layer_names,
                "update",
            )
        )
        lines.append("")

    return "\n".join(lines)
