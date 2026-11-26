"""Scalar encoding translation."""

from reactor import Wire

from ..terms import (
    argmax_variant,
    product_type,
    product_type_scalar,
    translate_terms_scalar,
    build_tuple,
    argmax_scalar_name,
)
from ...lowering.sorts import (
    dtype_to_lean_type,
    itype_name,
    lean_elem_slug,
    lean_elem_type,
    wire_shape,
)
from ...lowering.elements import mat_from_scalars, flat_layout
from ..context import LeanContext
from .common import (
    prepend_recon,
    scalar_bindings_with_recon,
    has_block,
    scalar_eq_name,
    scalar_unpack,
)


def _scalar_dom(param: str, wires: "list[Wire]") -> str:
    return f"({param}: {product_type_scalar(wires)})"


def _unpack_body(param: str, wires: "list[Wire]") -> str:
    """Body of ``unpack_<param>``: extract all flat elements from the wire."""
    layout = flat_layout(wires)
    return build_tuple([layout.wire_accessor(param, s) for s in layout.slots])


def _pack_body(var: str, wires: "list[Wire]") -> str:
    """Body of ``pack``: reconstruct each wire's matrix from flat scalar slots.

    The exact inverse of `_unpack_body`, and it has to be: `update_scalar_eq`
    and the `ScalarRel` bridges all rest on `pack (unpack x) = x`. Both walk
    the one layout, so the round trip is by construction rather than by two
    functions agreeing.
    """
    layout = flat_layout(wires)
    flat_accrs = layout.flat_accessors(var)
    parts = []
    for w in wires:
        offset, size = layout.span(w)
        parts.append(
            mat_from_scalars(flat_accrs[offset : offset + size], wire_shape(w))
        )
    return build_tuple(parts)


def _argmax_variants(ctx: LeanContext) -> "list[tuple[object, int]]":
    """(sort, n) for each distinct one-row Argmax input of the module, by
    element type and `n`: the ones `translate_terms_scalar` unrolls, and
    an `_argmax_scalar_def_lines` helper is emitted for."""
    seen: set[tuple[str, int]] = set()
    result: list[tuple[object, int]] = []
    for term in [*ctx.atom.init, *ctx.atom.update]:
        if (itype_name(term.itype) != "Argmax"
                or argmax_variant(term) != "argmax_1d"):
            continue
        in_wire = term.read[0]
        n = wire_shape(in_wire).total
        key = (lean_elem_type(in_wire.dtype), n)
        if key not in seen:
            seen.add(key)
            result.append((in_wire.dtype, n))
    return result


def _recon_wires(ctx: LeanContext) -> "list[Wire]":
    """One read wire per shape and element type that is more than 1×1:
    what `Scalar.update` may rebuild a `Matrix` of, and so what a
    `_recon_lemma_lines` lemma is proved for."""
    seen: set[tuple] = set()
    out: list[Wire] = []
    for _, wires in ctx.update_groups:
        for w in wires:
            key = (wire_shape(w), lean_elem_type(w.dtype))
            if not wire_shape(w).is_scalar and key not in seen:
                seen.add(key)
                out.append(w)
    return out


def _recon_name(w) -> str:
    shape = wire_shape(w)
    return f"recon_{lean_elem_slug(w.dtype)}_{shape.m}_{shape.n}"


def _recon_lemma_lines(w) -> "list[str]":
    """Lemma: rebuilding a wire from its own elements gives the wire back.

    `Scalar.update` reconstructs a `Matrix` from the unpacked scalars whenever a
    term needs the whole matrix (`matVecAffine`, say). In `update_scalar_eq`
    that reconstruction sits inside the argument, so the two sides differ by
    exactly `pack (unpack ctrl)` vs `ctrl` and `simp` cannot close the gap
    without this. Built by the same `mat_from_scalars` as the reconstruction
    itself, so the shapes match syntactically.

    The proofs rewrite with it in their first `simp only`, before anything
    else is unfolded: a later `simp` takes `(A * _m0) 0 0` apart into a sum
    over entries of the rebuilt matrix, and leaves no matrix to match.
    """
    shape = wire_shape(w)
    m, n = shape.m, shape.n
    slots = [f"c {i} {j}" for i in range(m) for j in range(n)]
    lhs = mat_from_scalars(slots, shape)
    return [
        "set_option maxHeartbeats 2000000 in",
        "set_option maxRecDepth 100000 in",
        f"@[simp] theorem {_recon_name(w)} (c : {dtype_to_lean_type(w)}) :",
        f"    {lhs} = c := by",
        "  funext i j",
        "  fin_cases i <;> fin_cases j <;> simp",
        "",
    ]


def _argmax_scalar_def_lines(sort, n: int) -> "list[str]":
    """Emit a scalar def and simp theorem for argmax over n elements.

    The fold of ``Relational.Mat.argmax_1d`` over the n elements written
    out as a list -- seeded with element 0 and updating on a strict ``<``
    so the first maximal index wins -- because ``{name}_eq`` below proves
    the two equal by unfolding both. A neutral seed or a non-strict
    comparison here would either break that proof or, worse, keep it
    provable against an equally wrong matrix definition.

    It is a `List.foldl`, as `argmax_1d` is, and not a chain of `let`s:
    each `let` read the one before it twice, so unfolding the chain
    doubled it with every element, and the `_eq` proof over 12 of them
    ran out of heartbeats.
    """
    name = argmax_scalar_name(sort, n)
    elem_ty = lean_elem_type(sort)
    noncomp = "noncomputable " if elem_ty == "Real" else ""
    params = " ".join(f"(s{i} : {elem_ty})" for i in range(n))
    rest = ", ".join(f"({i}, s{i})" for i in range(1, n))
    eq_params = " ".join(f"(v 0 {i})" for i in range(n))
    return [
        f"{noncomp}def {name} {params} : Int :=",
        f"  ((List.foldl (fun (best : Nat × {elem_ty}) p => "
        "if best.2 < p.2 then p else best)",
        f"    (0, s0) [{rest}]).1 : Int)",
        "",
        f"@[simp] theorem {name}_eq (v : Matrix (Fin 1) (Fin {n}) {elem_ty}) :",
        f"    {name} {eq_params} = (↑(argmax_1d v 0 0) : Int) := by",
        f"  simp only [{name}, argmax_1d, ↓Matrix.mk_apply, List.ofFn_succ, List.ofFn_zero]",
        "  rfl",
        "",
    ]


def atom_to_lean_scalar(ctx: LeanContext) -> str:
    """Generate the scalar encoding inside ``namespace Scalar``, and the
    lemmas its theorems rewrite with ahead of it."""
    atom = ctx.atom
    noncomp = ctx.noncomputable
    cod = product_type_scalar(ctx.ctrl_next)

    lines: list[str] = []
    for w in _recon_wires(ctx):
        lines.extend(_recon_lemma_lines(w))
    for sort, n in _argmax_variants(ctx):
        lines.extend(_argmax_scalar_def_lines(sort, n))

    lines += ["namespace Scalar", ""]

    # 1. Unpack functions. `reducible` for the reason the constants are
    #    (`literals.tensor_to_lean_def`): a `decide` over an unpacked input
    #    keeps `unpack_*` in its instance after `simp` has unfolded it in the
    #    proposition, and from Lean v4.34 that stops `simp` there.
    for param, wires in ctx.update_groups:
        if not wires:
            continue
        mat_ty = product_type(wires)
        scalar_ty = product_type_scalar(wires)
        body = _unpack_body(param, wires)
        lines.append(
            f"@[simp, reducible] {noncomp}def unpack_{param} ({param} : {mat_ty}) : {scalar_ty} :="
        )
        lines.append(f"  {body}")
        lines.append("")

    # 2. Pack function
    pack_body = _pack_body("r", ctx.ctrl_next)
    out_mat_ty = product_type(ctx.ctrl_next)
    lines.append(f"@[simp] {noncomp}def pack (r : {cod}) : {out_mat_ty} :=")
    lines.append(f"  {pack_body}")
    lines.append("")

    # 3. Scalar init / update
    init_recon, init_bindings, init_flat = scalar_bindings_with_recon(
        ctx.init_groups
    )
    init_body = translate_terms_scalar(
        atom.init,
        init_bindings,
        ctx.ctrl_next,
        ctx.constants,
        flat_slots=init_flat,
    )
    lines.append(
        f"@[simp] {noncomp}def init {_scalar_dom('extl_n', ctx.extl_next)} : {cod} :="
    )
    lines.append(prepend_recon(init_recon, init_body))
    lines.append("")

    update_recon, update_bindings_scalar, update_flat = scalar_bindings_with_recon(
        ctx.update_groups
    )
    update_body = translate_terms_scalar(
        atom.update,
        update_bindings_scalar,
        ctx.ctrl_next,
        ctx.constants,
        flat_slots=update_flat,
    )
    upd_dom = " ".join(_scalar_dom(p, w) for p, w in ctx.update_groups)
    lines.append(f"@[simp] {noncomp}def update {upd_dom} : {cod} :=")
    lines.append(prepend_recon(update_recon, update_body))
    lines.append("")

    lines.append("end Scalar")
    return "\n".join(lines)


def to_lean_scalar_equiv(ctx: LeanContext) -> str:
    """Generate theorems proving init/update = pack ∘ Scalar.init/update ∘ unpack."""
    lines: list[str] = []

    argmax_eq_lemmas = [
        f"{argmax_scalar_name(sort, n)}_eq"
        for sort, n in _argmax_variants(ctx)
    ]

    def _proof(
        groups: "list[tuple[str, list[Wire]]]",
        func_name: str,
        simp_extras: "list[str]",
    ) -> "list[str]":
        """`func_name` over `groups`, each a parameter and its wires."""
        vars_ = [p for p, _ in groups]
        binders = [f"({p} : {product_type(w)})" for p, w in groups]
        scalar_args = [scalar_unpack(p, w) for p, w in groups]
        scalar_call = f"Scalar.{func_name} {' '.join(scalar_args)}"
        rhs = f"Scalar.pack ({scalar_call})"
        simp_names = (
            ["Scalar.pack", f"Scalar.{func_name}", func_name]
            + simp_extras
            + [_recon_name(w) for w in _recon_wires(ctx)]
            + argmax_eq_lemmas
            + ["Fin.cons_zero", "Fin.cons_succ"]
        )
        return [
            # `fin_cases` enumerates every index of every wire, so the cost
            # grows with the state width: a 32-wide module exceeds the default
            # budget. Raised per theorem rather than file-wide.
            "set_option maxHeartbeats 2000000 in",
            "set_option maxRecDepth 100000 in",
            f"theorem {scalar_eq_name(func_name)} : ∀ {' '.join(binders)},",
            f"    {func_name} {' '.join(vars_)} = {rhs} := by",
            f"  intro {' '.join(vars_)}",
            f"  simp only [{', '.join(simp_names)}]",
            "  try rfl",
            # Enumerate the matrix indices first. `Fin.fin_one_eq_zero` alone
            # only collapses a `Fin 1`, so a wire with several elements is
            # left with a `match i, j with ...` on a symbolic `i`; and once an
            # earlier step has introduced `i`/`j`, a later `funext` can no
            # longer fire, so the general form has to come before the narrow
            # ones rather than after.
            "  try (apply Prod.ext <;> funext i j <;> fin_cases i <;> fin_cases j <;> simp)",
            "  try (funext i j <;> fin_cases i <;> fin_cases j <;> simp)",
            "  try (apply Prod.ext <;> funext i j <;> simp [Fin.fin_one_eq_zero])",
            "  try (funext i j; simp [Fin.fin_one_eq_zero])",
            "  try simp [Fin.fin_one_eq_zero]",
            # The scalar side repacks a 1×1 wire as `Matrix.mk fun _ _ => w 0 0`,
            # and the matrix side hands `w` itself to an operator that
            # consumes the whole matrix (`matVecAffine ... w`). Neither
            # `fin_cases` nor `Fin.fin_one_eq_zero` touches the argument, so
            # what is left is `op w 0 0 = op (Matrix.mk fun _ _ => w 0 0) 0 0` --
            # true, and closed only by rewriting the repacked wire back into
            # the wire. Over every goal: with two or more wires `Prod.ext`
            # leaves one per component, each with its own copy of the residue.
            "  all_goals try simp [← Matrix_1_1_eq]",
        ]

    for block, groups in (("init", ctx.init_groups),
                          ("update", ctx.update_groups)):
        if has_block(ctx, block):
            extras = [f"Scalar.unpack_{p}" for p, w in groups if w]
            lines.extend(_proof(groups, block, extras))
            lines.append("")

    return "\n".join(lines)
