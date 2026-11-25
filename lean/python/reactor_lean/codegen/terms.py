"""A block's terms as Lean text.

A block is a flat list of Terms (SSA-style dataflow). Each becomes a `let`
binding, its op printed by the op table's column for the encoding:
`translate_terms_lean` the functional encoding's body, of matrices, and
`translate_terms_scalar` the scalar encoding's, of their entries. With
them, the product types and tuples both print, and the names of the
scalar argmax helpers that the scalar body calls and `encodings.scalar`
defines.
"""

from __future__ import annotations
from reactor_lean.errors import Refused
from reactor_lean.lowering.sorts import (
    itype_name,
    dtype_to_lean_type,
    lean_elem_type,
    mat_mk,
    wire_shape,
)
from reactor_lean.lowering.elements import flat_indices, flat_layout
from reactor_lean.codegen.literals import (
    ConstantRegistry,
    constant_expr,
    linear_list_literals,
    tensor_to_lean_scalar,
)
from reactor_lean.lowering.ops import mat_emitter, scalar_emitter

from reactor import Wire


def product_type(wires: list[Wire]) -> str:
    """Build a right-nested product type: [w1, w2] -> 'Bool × Bool', [w1] -> 'Bool'."""
    if not wires:
        return "Unit"
    if len(wires) == 1:
        return dtype_to_lean_type(wires[0])
    parts = [dtype_to_lean_type(w) for w in wires]
    return " × ".join(parts)


def build_tuple(exprs: list[str]) -> str:
    """Build a tuple literal: [] -> '()', ['a'] -> 'a', ['a','b'] -> '(a, b)'."""
    if not exprs:
        return "()"
    if len(exprs) == 1:
        return exprs[0]
    return "(" + ", ".join(exprs) + ")"


def argmax_variant(term) -> str:
    """ReactLib's argmax for an `Argmax` term: `argmax_1d` over one row,
    `argmax` over a matrix. Both give one row-major flat index, as
    `torch.argmax` does, as a 1×1 matrix, which is what the theory's
    `Argmax` writes whatever it reads: it refuses a wider output wire."""
    return "argmax_1d" if wire_shape(term.read[0]).m == 1 else "argmax"


def _argmax_expr(term, wire_expr: dict[int, str]) -> str:
    """`Argmax` as a matrix (`argmax_variant`), its index an `Int`, as the
    theory's `Argmax` writes it."""
    arg = wire_expr[term.read[0].id]
    return mat_mk(f"(({argmax_variant(term)} {arg}) i j : Int)", "i", "j")


def _lift_scalar_reads(term, wire_expr: dict[int, str]) -> dict[int, str]:
    """`wire_expr` with this term's scalar-bound reads lifted to 1x1 matrices.

    In the scalar encoding a 1x1 wire is bound to a bare `Bool`/`Int`, but
    the matrix-form emitters apply their operands to `0 0`. Wrapping as
    `Matrix.mk fun _ _ => x` makes that reduce back to `x` and typecheck.
    """
    lifted = dict(wire_expr)
    for w in term.read:
        if wire_shape(w).is_scalar:
            # The ascription is load-bearing: `matVecAffine`'s batch
            # dimension is not determined by `Matrix.mk fun _ _ => x`, so an
            # un-ascribed lift leaves `n` unsolved ("don't know how to
            # synthesize implicit argument `n`").
            ty = f"Matrix (Fin 1) (Fin 1) {lean_elem_type(w.dtype)}"
            lifted[w.id] = f"({mat_mk(wire_expr[w.id])} : {ty})"
    return lifted


def _get_expr(term, wire_expr: dict[int, str]) -> str:
    """`Get`: element `[row][col]` of the read matrix, as a 1x1 one."""
    op = term.itype
    return mat_mk(f"{wire_expr[term.read[0].id]} {op.row} {op.col}")


def _linear_expr(term, wire_expr: dict[int, str]) -> str:
    """Emit a baked-constant `Linear` op, `Y = A·X + B`, or a `Scale` one,
    `diag(c)·X` (`literals.affine_tensors`), in the reflected form
    `matVecAffine m A b X`.

    `A` (shape `[out, in]`) and `B` (`[out, 1]`, or empty) are constants baked
    into the op; the read wire supplies `X` (`[in, batch]`). They are emitted as
    plain `List (List t)` / `List t` literals: cheap to elaborate (no dense
    `match`), and `matVecAffine` reduces to a linear expression under `simp`
    (no symbolic `∑`). `Relational.Mat.matVecAffine_eq` proves this equals
    `affineLinear (matrixOf A) X (colOf b)` for any batch width, so the
    contraction is machine-checked rather than trusted.
    """
    x_expr = wire_expr[term.read[0].id]
    out_m, a_lit, b_lit = linear_list_literals(term)
    return f"(matVecAffine {out_m} {a_lit} {b_lit} {x_expr})"


def reachable_terms(terms, output_wires: "list[Wire]") -> list:
    """Return the terms backward-reachable from output_wires, in original order.

    Traverses the term list in reverse: a term is kept if any wire it writes
    is needed; its read wires are then added to the needed set.
    """
    needed = {w.id for w in output_wires}
    selected = []
    for term in reversed(list(terms)):
        if {w.id for w in term.write} & needed:
            needed |= {w.id for w in term.read}
            selected.append(term)
    return list(reversed(selected))


def translate_terms_lean(
    terms,
    input_bindings: dict[int, str],
    block_outputs: list[Wire],
    constants: ConstantRegistry,
) -> str:
    """Compile a block of terms into a Lean function body with let-bindings.

    `input_bindings` maps block-input wire IDs to their Lean accessor expressions
    (e.g. ``{w.id: "ctrl.1"}``) — pre-built by the caller via `bind_wires`.

    Returns the body string (let x0 := ...; ... ; (x1, x2)). A block with no
    terms has no outputs -- an atom writes every controlled next wire in both
    blocks -- so its body is `()`, the only value of `Unit`.
    """
    wire_expr: dict[int, str] = dict(input_bindings)
    let_lines: list[str] = []

    for var_counter, term in enumerate(terms):
        name = itype_name(term.itype)

        if name == "Const":
            expr = constant_expr(term, term.write[0], constants)
        elif name == "Argmax":
            expr = _argmax_expr(term, wire_expr)
        elif name in ("Linear", "Scale"):
            expr = _linear_expr(term, wire_expr)
        elif name == "Get":
            expr = _get_expr(term, wire_expr)
        else:
            input_exprs = [wire_expr[w.id] for w in term.read]
            expr = mat_emitter(term.itype)(input_exprs)

        # Each term writes exactly one wire
        write_wire = term.write[0]
        var = f"x{var_counter}"
        wire_expr[write_wire.id] = var
        ty = dtype_to_lean_type(write_wire)
        let_lines.append(f"  let {var} : {ty} := {expr}")

    # Build output tuple
    out_exprs = [_output_expr(wire_expr, w) for w in block_outputs]
    result_line = f"  {build_tuple(out_exprs)}"

    return "\n".join(let_lines + [result_line])


def _output_expr(wire_expr: dict[int, str], w: Wire) -> str:
    """The expression a block output is bound to, by a term or as an input.

    Raises rather than emitting a placeholder: a build counts only the
    `sorry`s under `Certificate/` as unproved, so one in `System/` would pass
    for a proof.
    """
    if w.id not in wire_expr:
        raise Refused(
            f"block output wire #{w.id} is written by no term and is not an input"
        )
    return wire_expr[w.id]


def product_type_scalar(wires: list[Wire]) -> str:
    """Build a flat scalar product type, expanding multi-element wires.

    One component per slot, in slot order: the type view of the same
    flattening `flat_layout` owns, so the tuple's shape and the slot
    numbering cannot disagree.
    E.g. [Int(1), Float(2)] → ``"Int × Real × Real"``.
    """
    if not wires:
        return "Unit"
    parts = flat_layout(wires).element_types()
    if len(parts) == 1:
        return parts[0]
    return " × ".join(parts)


def _constant_expr_scalar(term, w: "Wire", constants: ConstantRegistry) -> str:
    """A constant in the scalar encoding: a 1×1 one as its bare element,
    any other by the name `ConstantRegistry` gave it, which it gives every
    constant that is not 1×1."""
    if wire_shape(w).is_scalar:
        return tensor_to_lean_scalar(term.itype.tensor, w)
    name = constants.lookup(w.id)
    assert name is not None, f"constant on wire #{w.id} was not registered"
    return name


def argmax_scalar_name(elem_ty: str, n: int) -> str:
    """Name of the scalar encoding's argmax over one row of n elements of
    `elem_ty` (`encodings.scalar._argmax_scalar_def_lines`).

    The element type belongs in the name: variants are collected per
    `(elem_ty, n)`, so naming them by `n` alone made an Argmax over
    a 1×4 `Int` matrix and one over a 1×4 `Real` one emit two definitions and two
    `_eq` theorems under one name.
    """
    return f"argmax1d_scalar_{elem_ty_slug(elem_ty)}_{n}"


def elem_ty_slug(elem_ty: str) -> str:
    """Lean-identifier-safe form of an element type, e.g. `(BitVec 8)` -> `bv8`."""
    if elem_ty.startswith("(BitVec "):
        return f"bv{elem_ty[len('(BitVec '):-1].strip()}"
    return elem_ty.lower()


def translate_terms_scalar(
    terms,
    input_bindings: dict[int, str],
    block_outputs: list[Wire],
    constants: ConstantRegistry,
    flat_slots: "dict[int, list[str]] | None" = None,
) -> str:
    """Compile terms into a Lean body using scalar types for 1×1 wires.

    ``flat_slots`` maps wire IDs to their flat scalar accessor strings.  An
    ``Argmax`` over one row whose slots are known is a call to
    ``argmax1d_scalar_*`` on them, instead of a matrix rebuilt from them.

    Falls back to the op table's matrix column for non-scalar output wires
    and for the ops whose scalar cell is `ViaMat`.
    """
    wire_expr: dict[int, str] = dict(input_bindings)
    flat_slots = dict(flat_slots or {})
    let_lines: list[str] = []

    for var_counter, term in enumerate(terms):
        name = itype_name(term.itype)
        write_wire = term.write[0]
        var = f"x{var_counter}"

        if name == "Const":
            expr = _constant_expr_scalar(term, write_wire, constants)
        elif name == "Argmax":
            # Unrolled over one row only: `argmax1d_scalar_*_eq` is stated
            # against `argmax_1d`, and nothing relates that to the `argmax`
            # the functional encoding takes of more than one row.
            in_wire = term.read[0]
            slots = flat_slots.get(in_wire.id)
            if slots is not None and argmax_variant(term) == "argmax_1d":
                helper = argmax_scalar_name(
                    lean_elem_type(in_wire.dtype), len(slots)
                )
                expr = f"({helper} {' '.join(slots)})"
            else:
                expr = f"({_argmax_expr(term, wire_expr)} 0 0)"
        elif name in ("Linear", "Scale"):
            # `matVecAffine` consumes an `n × batch` matrix and yields
            # an `out × batch` one, but a 1x1 wire is bound to a bare scalar
            # here and ascribed one. Lift the operand and take `0 0` off
            # the result, as the Argmax branch above does.
            expr = _linear_expr(term, _lift_scalar_reads(term, wire_expr))
            if wire_shape(write_wire).is_scalar:
                expr = f"({expr} 0 0)"
        elif name == "Get":
            # A 1x1 element of whatever is read: its flat slot where the
            # read wire has them, else the matrix applied to the indices.
            in_wire, op = term.read[0], term.itype
            slots = flat_slots.get(in_wire.id)
            if slots is not None:
                expr = slots[op.row * wire_shape(in_wire).n + op.col]
            else:
                expr = f"({wire_expr[in_wire.id]} {op.row} {op.col})"
        elif wire_shape(write_wire).is_scalar and scalar_emitter(term.itype) is not None:
            input_exprs = [wire_expr[w.id] for w in term.read]
            expr = scalar_emitter(term.itype)(input_exprs)
        else:
            # The `mat` column holds the matrix forms, which apply operands
            # to `0 0`. Scalar-bound reads must be lifted first: a matrix
            # `Ite` with a scalar Bool condition emitted
            # `if <Bool> 0 0 then ...`, which does not elaborate.
            lifted = _lift_scalar_reads(term, wire_expr)
            input_exprs = [lifted[w.id] for w in term.read]
            expr = mat_emitter(term.itype)(input_exprs)
            # A matrix form yields a `Matrix`; a scalar write wire is ascribed
            # the bare element type, so project it — the same step the
            # `Argmax` and `Linear` branches above already take. Without it
            # a 1x1 `Min`/`Max` emitted `let x : Int := matMax …`.
            if wire_shape(write_wire).is_scalar:
                expr = f"({expr} 0 0)"

        wire_expr[write_wire.id] = var
        # Track flat scalar slots for the written wire so downstream Argmax
        # terms can reference individual elements without matrix reconstruction.
        if wire_shape(write_wire).is_scalar:
            flat_slots[write_wire.id] = [var]
        else:
            # Parenthesised: these are spliced into an application
            # (`argmax1d_scalar_int_3 <slots>`), where a bare `x0 0 0`
            # would be read as three separate arguments.
            flat_slots[write_wire.id] = [
                f"({var} {r} {c})" for r, c in flat_indices(write_wire)
            ]
        ty = dtype_to_lean_type(write_wire, simple_types=True)
        let_lines.append(f"  let {var} : {ty} := {expr}")

    # The codomain every caller declares has one component per element
    # (`product_type_scalar`): an output with flat slots contributes them;
    # one without (no element was bound) contributes the matrix the terms
    # built.
    out_exprs: list[str] = []
    for w in block_outputs:
        if w.id in flat_slots:
            out_exprs.extend(flat_slots[w.id])
        else:
            out_exprs.append(_output_expr(wire_expr, w))
    result_line = f"  {build_tuple(out_exprs)}"

    return "\n".join(let_lines + [result_line])
