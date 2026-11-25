"""A block's terms as `Box` circuit layers (`translate_terms_circ`): the
circuit encoding, in which each op is the `Box` combinator the op table
names (`lowering.ops.box_for`) and the layers compose as
`Relational.Circ` circuits."""

from __future__ import annotations
from reactor_lean.errors import Refused
from reactor_lean.lowering.sorts import (
    dtype_to_lean_type,
    itype_name,
    is_constant_name,
)
from reactor_lean.lowering.elements import product_accessor
from reactor_lean.codegen.literals import (
    ConstantRegistry,
    constant_expr,
    linear_list_literals,
)

from reactor_lean.codegen.terms import argmax_variant
from reactor_lean.lowering.ops import box_for, entrywise_shape

from dataclasses import dataclass

from reactor import Wire, Term


# The Boxes written here rather than looked up in the op table: a layer's
# wiring, constants, and the rows the table marks `Inline` (`Get`,
# `Linear` and `Scale`, `Argmax`). `simp_circ` unfolds these and the
# table's own.
BUILT_HERE = (
    "Box.id",
    "Box.dup",
    "Box.swap",
    "Box.destr",
    "Box.const",
    "Box.get",
    "Box.linear",
    "Box.argmax_1d",
    "Box.argmax",
)


def ty_list(wires: list[Wire]) -> str:
    """The wires' types as a Lean list, a layer's interface:
    `[T₁ , T₂]`, and `[]` for no wires."""
    if not wires:
        return "[]"
    parts = [dtype_to_lean_type(w) for w in wires]
    return "[{}]".format(" , ".join(parts))


@dataclass
class CircLayer:
    """A single layer in a circuit composition."""

    in_tys: list[str]
    out_tys: list[str]
    body: str


def natives_to_vt(params: list[tuple[str, int]]) -> str:
    """Generate native-to-ValTuple conversion from one or more native args.

    Each (name, n) contributes n accessors; all are concatenated into a single
    right-nested ValTuple. Example: [("ctrl", 2), ("extl_n", 1)] ->
    "(ctrl.1, (ctrl.2, (extl_n, ())))".
    """
    parts = [
        f"{name}{product_accessor(i, n)}" for name, n in params if n > 0 for i in range(n)
    ]
    result = "()"
    for p in reversed(parts):
        result = f"({p}, {result})"
    return result


def native_to_vt(param: str, n_wires: int) -> str:
    """Generate native-to-ValTuple conversion from a single native arg."""
    return natives_to_vt([(param, n_wires)])


def _linear_box(term: Term) -> str:
    """Box for a baked-constant `Linear` op, or a `Scale` one as `diag(c)·x`
    (`literals.affine_tensors`): `Box.linear m A b` (x ↦ A·x + b).

    A and b are emitted as list literals; `Box.linear` applies the reflected
    `matVecAffine` (see `Relational.Mat`). Column output only (batch = 1).
    """
    out_m, a_lit, b_lit, _ = linear_list_literals(term)
    return f"(Box.linear {out_m} {a_lit} {b_lit})"


def _box(term: Term) -> str:
    """The op's Box: an `entrywise` op larger than 1×1 has none, its Box
    reading 1×1 operands."""
    shape = entrywise_shape(term.itype)
    if shape is not None:
        name = itype_name(term.itype)
        article = "an" if name[0] in "AEIOU" else "a"
        raise Refused(
            f"no Box for {article} `{name}` of shape {shape}: its Box reads "
            "1×1 matrices"
        )
    return box_for(term.itype)


def _get_box(term: Term) -> str:
    """Box for `Get`: element `[row][col]` of the read matrix, as a 1x1 one.
    The element is baked into the op, as `Linear`'s matrices are."""
    return f"(Box.get {term.itype.row} {term.itype.col})"


def _circ_translate_body(
    block_inputs: list[Wire],
    block_outputs: list[Wire],
    wire_to_term: dict[Wire, Term],
) -> list[list[Wire]]:

    # output layer are just the terms writing to output wires. However,
    # there are no terms e.g., for inputs (unless we want to create new
    # terms). So we use `(Wire, Term | None)` where if term is None we still
    # have the information about the element in the layer in the Wire
    output_layer: list[Wire] = block_outputs
    layers: list[list[Wire]] = [output_layer]

    # repeatedly take the last layer and cerate new layer
    # representing the inputs to the last layer.
    have_new = True
    while have_new:
        have_new = False

        layer: list[Wire] = layers[-1]
        new_layer: list[Wire] = []

        for w in layer:
            # if this is the input wire, just copy it
            if w in block_inputs:
                new_layer.append(w)
                continue

            term = wire_to_term[w]
            name = itype_name(term.itype)
            # constant terms have no inputs, the do not propagate into the upper layers
            if is_constant_name(name):
                continue

            new_layer.extend(term.read)
            have_new = bool(new_layer)

        if new_layer != layer:
            layers.append(new_layer)

        if not have_new:
            return layers


def _circ_compute_swapping(
    wires_ord: dict[Wire, int], layer: list[Wire]
) -> tuple[list[CircLayer], list[Wire]]:
    swapping: list[CircLayer] = []
    while True:
        new_layer: list[Wire] = []
        boxes: list[str] = []
        N = len(layer)
        changed = False
        in_ty, out_ty = [], []
        i = 0
        while i < N:
            if i >= N - 1 or wires_ord[layer[i]] <= wires_ord[layer[i + 1]]:
                # emit ID, this is the last wire or this and next wire are correctly ordered
                boxes.append(f"@Box.id {dtype_to_lean_type(layer[i])}")
                new_layer.append(layer[i])
                in_ty.append(dtype_to_lean_type(layer[i]))
                out_ty.append(dtype_to_lean_type(layer[i]))
                i += 1
            else:
                # Forward composition order: this Box receives the sorted
                # (new_layer) arrangement and produces the original `layer`
                # arrangement to feed into the next stage. So `Box.swap A B`
                # takes [A, B] → [B, A] where [A, B] are new_layer's types at
                # positions (i, i+1) and [B, A] are `layer`'s types there.
                changed = True
                boxes.append(
                    f"@Box.swap {dtype_to_lean_type(layer[i + 1])} {dtype_to_lean_type(layer[i])}"
                )
                new_layer.append(layer[i + 1])
                new_layer.append(layer[i])
                in_ty.append(dtype_to_lean_type(layer[i + 1]))
                in_ty.append(dtype_to_lean_type(layer[i]))
                out_ty.append(dtype_to_lean_type(layer[i]))
                out_ty.append(dtype_to_lean_type(layer[i + 1]))
                # `swap` consumes two wires
                i += 2

        if not changed:
            return swapping, layer
        assert boxes
        swapping.append(
            CircLayer(
                in_tys=list(in_ty),
                out_tys=list(out_ty),
                body=" ⊗ ".join(boxes),
            )
        )
        assert len(new_layer) == len(layer)
        layer = new_layer


def _circ_compute_dups(
    wires_ord: dict[Wire, int], layer: list[Wire]
) -> tuple[list[CircLayer], list[Wire]]:
    layers = []
    while True:
        new_layer: list[Wire] = []
        boxes: list[str] = []
        N = len(layer)
        changed = False
        in_ty, out_ty = [], []
        i = 0
        while i < N:
            if i >= N - 1 or wires_ord[layer[i]] != wires_ord[layer[i + 1]]:
                # emit ID, this is the last wire or this and the next wire differ
                boxes.append(f"@Box.id {dtype_to_lean_type(layer[i])}")
                new_layer.append(layer[i])
                in_ty.append(dtype_to_lean_type(layer[i]))
                out_ty.append(dtype_to_lean_type(layer[i]))
                i += 1
            else:
                changed = True
                boxes.append(f"@Box.dup {dtype_to_lean_type(layer[i])}")
                new_layer.append(layer[i])
                in_ty.append(dtype_to_lean_type(layer[i]))
                out_ty.append(dtype_to_lean_type(layer[i]))
                out_ty.append(dtype_to_lean_type(layer[i]))
                # `dup` consumes two wires
                i += 2

        if not changed:
            return layers, layer

        assert boxes

        layers.append(
            CircLayer(
                in_tys=list(in_ty),
                out_tys=list(out_ty),
                body=" ⊗ ".join(boxes),
            )
        )
        assert len(new_layer) < len(layer)
        layer = new_layer


def _circ_compute_dels(block_inputs: list[Wire], last_layer: list[Wire]):
    dels = []
    if len(block_inputs) != len(last_layer):
        boxes = []
        in_ty, out_ty = [], []
        for w in block_inputs:
            if w in last_layer:
                boxes.append(f"@Box.id {dtype_to_lean_type(w)}")
                in_ty.append(dtype_to_lean_type(w))
                out_ty.append(dtype_to_lean_type(w))
            else:
                boxes.append(f"@Box.destr {dtype_to_lean_type(w)}")
                in_ty.append(dtype_to_lean_type(w))
        assert boxes
        dels.append(
            CircLayer(
                in_tys=list(in_ty),
                out_tys=list(out_ty),
                body=" ⊗ ".join(boxes),
            )
        )
    return dels


def translate_terms_circ(
    terms,
    block_inputs: tuple[list[Wire], ...],
    block_outputs: list[Wire],
    constants: ConstantRegistry,
) -> list[CircLayer]:
    """Compile a block of terms into a list of CircLayer objects.

    Each CircLayer represents one layer of the circuit composition.
    Returns empty list if there are no terms.
    """
    term_list = list(terms)
    if not term_list:
        return []

    all_inputs = [w for wires in block_inputs for w in wires]

    # output wires to terms
    wire_to_term: dict[Wire, Term] = {t.write[0]: t for t in term_list}
    # order of the input wires
    wires_ord: dict[Wire, int] = {w: n for n, w in enumerate(all_inputs)}

    layers = _circ_translate_body(all_inputs, block_outputs, wire_to_term)

    # compute swapping wires
    swapping, last_layer = _circ_compute_swapping(wires_ord, layers[-1])

    # duplication of wires
    dups, last_layer = _circ_compute_dups(wires_ord, last_layer)

    # deletion of unused wires
    dels = _circ_compute_dels(all_inputs, last_layer)

    result: list[CircLayer] = list(reversed(swapping + dups + dels))

    for n, layer in enumerate(reversed(layers)):
        if not layer:
            # if there are unused variables, the first layer can be empty
            assert n == 0
            continue
        boxes: list[str] = []
        in_ty, out_ty = [], []
        for w in layer:
            if w in all_inputs:
                boxes.append(f"@Box.id {dtype_to_lean_type(w)}")
                in_ty.append(dtype_to_lean_type(w))
                out_ty.append(dtype_to_lean_type(w))
            else:
                term = wire_to_term[w]
                name = itype_name(term.itype)

                if is_constant_name(name):
                    expr = constant_expr(term, w, constants)

                    boxes.append(f"@Box.const {dtype_to_lean_type(w)} {expr}")
                    out_ty.append(dtype_to_lean_type(w))
                else:
                    if name == "Argmax":
                        boxes.append(f"Box.{argmax_variant(term)}")
                    elif name in ("Linear", "Scale"):
                        boxes.append(_linear_box(term))
                    elif name == "Get":
                        boxes.append(_get_box(term))
                    else:
                        boxes.append(_box(term))
                    in_ty.extend([dtype_to_lean_type(u) for u in term.read])
                    out_ty.extend([dtype_to_lean_type(u) for u in term.write])
        assert boxes
        result.append(
            CircLayer(
                in_tys=list(in_ty),
                out_tys=list(out_ty),
                body=" ⊗ ".join(boxes),
            )
        )

    return result
