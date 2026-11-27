"""`LeanContext`: what every encoding of one module shares, computed once."""

from __future__ import annotations

from reactor import Atom, Module, Real, Var, Wire, X
from ..core.errors import Refused
from ..lowering.elements import product_accessor
from .literals import ConstantRegistry


class LeanContext:
    """Pre-computed artifacts shared by module and certificate codegen.

    Runs a one-time discovery pass over the module's init/update terms to
    populate the constant registry and the wire-name bindings for each
    block. Read-only after construction; codegen consumers do not mutate it.

    `atoms` are the module's atoms in the order it keeps them, which is
    consistent with its await relation: an atom comes after every atom whose
    variables it awaits. The functional and the module encodings compose
    them; every other encoding is written over the module's one atom
    (`atom`).
    """

    def __init__(self, module: Module):
        self.module = module
        self.atoms: list[Atom] = list(module.atoms)

        self.extl_latched: list[Var] = list(module.extl)
        self.extl_next: list[Wire] = [X(v) for v in module.extl]
        self.ctrl_latched: list[Var] = list(module.ctrl)
        self.ctrl_next: list[Wire] = [X(v) for v in module.ctrl]

        self.constants = ConstantRegistry()
        for term in self.terms("init") + self.terms("update"):
            self.constants.intern(term)

        # Modules that use Float/Real require `noncomputable` on defs that
        # compare or branch on real-typed values (Real lacks decidable equality).
        self.uses_real = any_float_wire(
            self.extl_latched,
            self.extl_next,
            self.ctrl_latched,
            self.ctrl_next,
            [t.write[0] for t in self.terms("init")],
            [t.write[0] for t in self.terms("update")],
        )

        self.init_wire_names = bind_wires([("extl_n", self.extl_next)])
        self.update_wire_names = bind_wires(
            [
                ("ctrl", self.ctrl_latched),
                ("extl_l", self.extl_latched),
                ("extl_n", self.extl_next),
            ]
        )

    @property
    def atom(self) -> Atom:
        """The module's one atom, which every encoding but the functional
        and the module ones is written over. Refused for a module of any
        other number: those two are the only ones that compose atoms yet."""
        if len(self.atoms) != 1:
            raise Refused(
                f"the module has {len(self.atoms)} atoms, and only its "
                "functional and module encodings compose atoms: the other "
                "encodings take a module of one"
            )
        return self.atoms[0]

    def terms(self, block: str) -> list:
        """Every atom's `block` ("init" or "update") terms, atom after atom:
        one block in dataflow order, since an atom comes after the atoms it
        awaits. It is how `reactor.eval` runs a round."""
        return [t for atom in self.atoms for t in getattr(atom, block)]


def any_float_wire(*wire_lists: list[Wire]) -> bool:
    """True if any wire across the given lists has a Float dtype."""
    for wires in wire_lists:
        for w in wires:
            if isinstance(w.dtype, Real):
                return True
    return False


def bind_wires(params: list[tuple[str, list[Wire]]]) -> dict[int, str]:
    """Map each input wire id to its Lean accessor expression.

    E.g. [("ctrl", [w0, w1]), ("extl_n", [w2])] ->
        {w0.id: "ctrl.1", w1.id: "ctrl.2", w2.id: "extl_n"}
    """
    out: dict[int, str] = {}
    for name, wires in params:
        n = len(wires)
        for i, w in enumerate(wires):
            out[w.id] = f"{name}{product_accessor(i, n)}"
    return out
