"""What the scalar and relational encodings share: a block's wires as a
flat tuple of their entries and as the matrices rebuilt from it, and the
names by which one cites the other's theorems."""

import re

from ...lowering.sorts import dtype_to_lean_type, wire_shape
from ...lowering.elements import FlatLayout, product_accessor, mat_from_scalars, flat_layout
from ..terms import build_tuple, product_type_scalar


def has_block(ctx, block: str) -> bool:
    """Whether the module's atom has `block` ("init" or "update") terms. An
    encoding proves its theorems about a block only when the block has
    some, so whatever cites one of them asks this same question."""
    return bool(list(getattr(ctx.atom, block)))


def scalar_unpack(param: str, wires, var: str = "") -> str:
    """`var`, read back as a matrix by `Scalar.unpack_<param>`.

    Left alone when the group is empty: `unpack_extl_l` of no wires is `Unit`
    to `Unit`, and `scalar.py` does not emit it. `var` defaults to the
    parameter's own name; the relational encoding's init/trans theorems
    quantify the *next* ctrl state as `ctrl'` but still read it back with
    `unpack_ctrl`.
    """
    var = var or param
    return f"(Scalar.unpack_{param} {var})" if wires else var


def scalar_eq_name(block: str) -> str:
    """The theorem `scalar.py` proves for `block`: the functional block is
    `Scalar.pack` of `Scalar.<block>` of the unpacked inputs. It exists
    exactly when `has_block(ctx, block)`."""
    return f"{block}_scalar_eq"


def flat_slice(expr: str, layout: FlatLayout, wire) -> str:
    """The components of the flat tuple `expr` that `wire` owns.

    Takes the wire and the layout rather than an offset and a length,
    which are a position in one index space and a length in another, and
    easily mismatched.

    A wire covering the whole tuple slices to `expr` itself: projecting and
    re-tupling would be a no-op that repeats `expr` once per element, and
    `expr` is a whole `Scalar.update` call at the only call sites.
    """
    offset, size = layout.span(wire)
    if offset == 0 and size == layout.total:
        return expr
    return build_tuple(
        [f"{expr}{product_accessor(offset + k, layout.total)}" for k in range(size)]
    )


def effect_type(wire) -> str:
    """Codomain of a per-wire `effect_i`/`init_i`: its own flat element tuple.

    One component per element, so it can be compared to the matching slice of
    the scalar encoding's state tuple. Typing it as the wire's `Matrix` instead
    put a `Matrix (Fin 6) (Fin 1) Int` against an `Int x ... x Int`.
    """
    return product_type_scalar([wire])


def scalar_bindings_with_recon(
    params: "list[tuple[str, list]]",
) -> "tuple[list[tuple[str, str]], dict[int, str], dict[int, list[str]]]":
    """Build wire bindings and flat scalar slot tracking for scalar encoding.

    For each wire:
    - Scalar (one element): binding is the bare accessor (e.g. ``"ctrl.1"``).
    - Non-scalar (more than one): binding is a fresh let-var reconstructing the
      ``Matrix (Fin m) (Fin n) T`` (for operations that need the full matrix); flat_slots also
      records the individual scalar accessor strings for argmax and similar.

    Returns ``(recon, bindings, flat_slots)`` where:
    - ``recon`` is each matrix reconstruction as ``(var, let_line)``, for
      `prepend_recon`
    - ``bindings`` maps wire_id → matrix var or scalar accessor
    - ``flat_slots`` maps wire_id → list of flat scalar accessor strings
    """
    recon: list[tuple[str, str]] = []
    bindings: dict[int, str] = {}
    flat_slots: dict[int, list[str]] = {}

    for name, wires in params:
        layout = flat_layout(wires)
        flat_accrs = layout.flat_accessors(name)

        for w in wires:
            offset, size = layout.span(w)
            slots = flat_accrs[offset : offset + size]
            flat_slots[w.id] = slots
            if size == 1:
                bindings[w.id] = slots[0]
            else:
                var = f"_m{len(recon)}"
                mat_expr = mat_from_scalars(slots, wire_shape(w))
                recon.append(
                    (var, f"  let {var} : {dtype_to_lean_type(w)} := {mat_expr}"))
                bindings[w.id] = var

    return recon, bindings, flat_slots


def prepend_recon(recon: "list[tuple[str, str]]", body: str) -> str:
    """`body` after the matrix reconstructions it reads, in order.

    A body reads few of them when it is one `effect_i` of the relational
    encoding, which keeps the terms one ctrl wire needs; the others would be
    dead `let`s in every such definition. A reconstruction reads only flat
    accessors, never another reconstruction, so the body alone decides. A
    use is a whole word: `_m10` is no use of `_m1`.
    """
    lines = [line for var, line in recon if re.search(rf"\b{var}\b", body)]
    return "\n".join(lines + [body])
