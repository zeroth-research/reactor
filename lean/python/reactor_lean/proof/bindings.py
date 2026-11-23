"""What the two predicate printers, `from_smt` and `from_ir`, print alike:
the accessor each variable is read through on the one parameter, and the
`let` block a predicate opens with.

Nothing here loads cvc5, so `from_ir` shares it without loading a solver.
"""

from __future__ import annotations

from reactor import Wire

from ..lowering.elements import product_accessor
from ..lowering.sorts import wire_shape

# (prefix, base, wires): wire `i` is the variable `<prefix><i>`, read on
# the parameter `base`.
Bindings = "list[tuple[str, str, list[Wire]]]"


def build_var_map(bindings: Bindings) -> tuple[dict[str, str], dict[str, Wire]]:
    """Build Lean-accessor + wire maps for a list of (prefix, base, wires).

    For each binding, each wire at position `i` gets a variable name
    `<prefix><i>` mapped to a Lean accessor `<base><acc> [0 0]`. Scalar
    1×1 wires get the `0 0` scalar extraction; vector/matrix wires keep
    the raw function form.
    """
    var_accessor: dict[str, str] = {}
    var_wire: dict[str, Wire] = {}
    for prefix, base, wires in bindings:
        n = len(wires)
        for i, w in enumerate(wires):
            acc = product_accessor(i, n)
            name = f"{prefix}{i}"
            if wire_shape(w).is_scalar:
                var_accessor[name] = f"({base}{acc} 0 0)"
            else:
                var_accessor[name] = f"({base}{acc})"
            var_wire[name] = w
    return var_accessor, var_wire


def with_lets(body: str, lets: "list[tuple[str, str]]") -> str:
    """`body`, after a `let name := expr` for each of `lets`, in order."""
    if not lets:
        return body
    binds = "".join(f"\n  let {name} := {expr}" for name, expr in lets)
    return f"{binds}\n  {body}"
