"""Functional (init/update def) translation.

A module of one atom is its `init` and `update`, each one block of `let`s.
A module of several is each atom's own `init` and `update`, in a namespace
of its own (`Atom0`, `Atom1`, ...), and the module's `init` and `update`
composed of them: each atom's function is applied, in the module's order of
atoms, to the module's arguments and to the next values of the variables
it awaits, which the atoms before it computed. Either way `init` and
`update` have the same types, so what is stated over them -- the
certificate, `Main.lean` -- does not ask how many atoms there are.
"""

from reactor import Var, Wire, X

from reactor_lean.codegen.terms import build_tuple, translate_terms_lean, product_type
from reactor_lean.codegen.context import LeanContext, bind_wires
from reactor_lean.lowering.elements import product_accessor


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
    if len(ctx.atoms) == 1:
        return _one_atom(ctx)
    return _composed(ctx)


def _one_atom(ctx: LeanContext) -> str:
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
    noncomp = "noncomputable " if ctx.uses_real else ""

    init_dom = f"(extl_n: {product_type(ctx.extl_next)})"
    lines.append(f"@[simp] {noncomp}def init {init_dom} : {cod} :=")
    lines.append(init_body)
    lines.append("")

    upd_dom = (
        f"(ctrl: {product_type(ctx.ctrl_latched)}) "
        f"(extl_l: {product_type(ctx.extl_latched)}) "
        f"(extl_n: {product_type(ctx.extl_next)})"
    )
    lines.append(f"@[simp] {noncomp}def update {upd_dom} : {cod} :=")
    lines.append(update_body)
    lines.append("")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# Several atoms
# ══════════════════════════════════════════════════════════════════════════

# The parameter an atom's functions take the next values of the variables it
# awaits in, and the namespace of atom `i`'s functions.
WAIT = "wait"


def atom_namespace(i: int) -> str:
    return f"Atom{i}"


def _module_params(ctx: LeanContext, block: str) -> "list[tuple[str, list]]":
    """What the module's `block` function takes, by parameter: the inputs
    awaited in the round, and for `update` the state and the inputs latched
    before it."""
    if block == "init":
        return [("extl_n", ctx.extl_next)]
    return [
        ("ctrl", ctx.ctrl_latched),
        ("extl_l", ctx.extl_latched),
        ("extl_n", ctx.extl_next),
    ]


def awaited(ctx: LeanContext, atom) -> "list[Var]":
    """The variables of other atoms whose next values `atom` awaits, in its
    order. An awaited input is not one of them: it is in `extl_n`, which
    every atom's functions take."""
    ctrl = {v.id for v in ctx.ctrl_latched}
    return [v for v in atom.wait if v.id in ctrl]


def _signature(params: "list[tuple[str, list]]") -> str:
    return " ".join(f"({name}: {product_type(wires)})" for name, wires in params)


def _atom_defs(ctx: LeanContext, i: int, atom) -> "list[str]":
    """Atom `i`'s `init` and `update`, in its namespace: the module's
    parameters and, when it awaits variables of other atoms, their next
    values in `wait`, to the next values of the variables it controls."""
    waits = awaited(ctx, atom)
    extra = [(WAIT, [X(v) for v in waits])] if waits else []
    outputs: list[Wire] = [X(v) for v in atom.ctrl]
    noncomp = "noncomputable " if ctx.uses_real else ""
    lines = [f"namespace {atom_namespace(i)}", ""]
    for block in ("init", "update"):
        params = _module_params(ctx, block) + extra
        body = translate_terms_lean(
            getattr(atom, block), bind_wires(params), outputs, ctx.constants)
        lines.append(f"@[simp] {noncomp}def {block} {_signature(params)} : "
                     f"{product_type(outputs)} :=")
        lines.append(body)
        lines.append("")
    lines += [f"end {atom_namespace(i)}", ""]
    return lines


def _composition(ctx: LeanContext, block: str) -> "list[str]":
    """The module's `block`: each atom's applied in turn, `a<i>` its next
    values, and the module's next state read off them in `ctrl` order."""
    params = _module_params(ctx, block)
    args = [name for name, _ in params]
    noncomp = "noncomputable " if ctx.uses_real else ""
    lines = [f"@[simp] {noncomp}def {block} {_signature(params)} : "
             f"{product_type(ctx.ctrl_next)} :="]
    next_of: dict[int, str] = {}    # a variable's id -> its next value
    for i, atom in enumerate(ctx.atoms):
        waits = awaited(ctx, atom)
        missing = [v.id for v in waits if v.id not in next_of]
        if missing:
            raise ValueError(
                f"atom {i} awaits variables {missing} that no atom before it "
                "controls: the module's atoms are not in await order"
            )
        wait_arg = [build_tuple([next_of[v.id] for v in waits])] if waits else []
        lines.append(f"  let a{i} := {atom_namespace(i)}.{block} "
                     + " ".join(args + wait_arg))
        ctrl = list(atom.ctrl)
        for p, v in enumerate(ctrl):
            next_of[v.id] = f"a{i}{product_accessor(p, len(ctrl))}"
    lines.append(f"  {build_tuple([next_of[v.id] for v in ctx.ctrl_latched])}")
    lines.append("")
    return lines


def _composed(ctx: LeanContext) -> str:
    lines: list[str] = []
    for i, atom in enumerate(ctx.atoms):
        lines += _atom_defs(ctx, i, atom)
    for block in ("init", "update"):
        lines += _composition(ctx, block)
    return "\n".join(lines)
