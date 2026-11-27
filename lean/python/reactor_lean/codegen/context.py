"""`LeanContext`: what every encoding of one module shares, computed once."""

from __future__ import annotations

from reactor import Module, Real, Var, Wire

from ..errors import Refused
from ..lowering.elements import ModuleWires, product_accessor
from ..lowering.ops import lean_gap
from ..lowering.sorts import itype_name, theory_of
from .literals import ConstantRegistry, affine_gap


class LeanContext:
    """Pre-computed artifacts shared by module and certificate codegen.

    Runs a one-time discovery pass over the module's init/update terms to
    populate the constant registry and the wire-name bindings for each
    block. Read-only after construction; codegen consumers do not mutate it.
    """

    def __init__(self, module: Module):
        atoms = list(module.atoms)
        if len(atoms) != 1:
            raise Refused(
                f"the module is a composition of {len(atoms)} atoms; only a "
                "module of one atom is translated"
            )

        self.module = module
        self.atom = atoms[0]

        wires = ModuleWires.of(module)
        self.extl_latched: list[Var] = wires.extl_latched
        self.extl_next: list[Wire] = wires.extl_next
        self.ctrl_latched: list[Var] = wires.ctrl_latched
        self.ctrl_next: list[Wire] = wires.ctrl_next

        self.constants = ConstantRegistry()
        for term in self.atom.init:
            self.constants.intern(term)
        for term in self.atom.update:
            self.constants.intern(term)

        # Modules that use Float/Real require `noncomputable` on defs that
        # compare or branch on real-typed values (Real lacks decidable equality).
        self.uses_real = any_float_wire(
            self.extl_latched,
            self.extl_next,
            self.ctrl_latched,
            self.ctrl_next,
            [t.write[0] for t in self.atom.init],
            [t.write[0] for t in self.atom.update],
        )

        # What each block reads, in the order and under the names every
        # encoding binds it: `init` the inputs awaited in the first round,
        # `update` the state and the inputs latched before the round and
        # awaited in it (`extl_groups`).
        self.init_groups: list[tuple[str, list[Wire]]] = [
            ("extl_n", self.extl_next),
        ]
        self.extl_groups: list[tuple[str, list[Wire]]] = [
            ("extl_l", self.extl_latched),
            ("extl_n", self.extl_next),
        ]
        self.update_groups: list[tuple[str, list[Wire]]] = [
            ("ctrl", self.ctrl_latched),
            *self.extl_groups,
        ]
        self.init_wire_names = bind_wires(self.init_groups)
        self.update_wire_names = bind_wires(self.update_groups)

    @property
    def noncomputable(self) -> str:
        """What a definition of the module's blocks starts with: `noncomputable `
        when the module computes with Reals, else nothing."""
        return "noncomputable " if self.uses_real else ""


def lean_context(module) -> LeanContext:
    """The module's `LeanContext`, refusing a module with no Lean form.

    An operator with no Lean form is a fact about the module, said here
    once, before any of the project exists, rather than as a missing table
    cell in the middle of a file.

    Built once and threaded: its discovery pass interns every init/update
    term, and everything that prints the module needs the same one.
    """
    ctx = LeanContext(module)
    gaps = lean_gaps(ctx)
    if gaps:
        raise Refused("this module has no Lean form: " + "; also ".join(gaps))
    return ctx


def lean_gaps(ctx) -> list[str]:
    """Every operator in this module that no Lean column can write.

    One sentence per operator, in the order the module first reaches for it:
    which block, which wire, and the reason the op table records. Ops of
    two theories that share a name (`word.unsigned.Div`, `word.signed.Div`)
    are two operators. A `Linear` or `Scale` term is asked about on its
    own, since whether it has a Lean form is a fact of the constants baked
    into it (`literals.affine_gap`).

    Asked before anything is generated, which is the whole point. The same
    module met during codegen raises about a missing table cell, half way
    through the project. Every term is asked about, read or not, because
    the functional and scalar encodings translate a block's terms as they
    are. A module that generates has nothing to report here: the gap is an
    operator with no matrix form (:func:`lowering.ops.gap_of`), or an
    affine op that is no affine map, and the functional encoding every
    project has is made of matrix forms alone. So is a `delay` block that
    gives a variable a rate of change: every encoding here is of rounds.
    """
    out, seen = [], set()
    for block in ("init", "update"):
        for term in getattr(ctx.atom, block):
            theory, name = theory_of(term.itype), itype_name(term.itype)
            if name in ("Linear", "Scale"):
                why = affine_gap(term)
            elif (theory, name) in seen:
                why = None
            else:
                why = lean_gap(term.itype)
                if why is not None:
                    seen.add((theory, name))
            if why is None:
                continue
            op = f"{theory}.{name}" if theory else name
            out.append(
                f"the `{block}` block writes wire #{term.write[0].id} with "
                f"`{op}`, and {why}"
            )
    # The `delay` block is the variables' rate of change: all `Zero` in a
    # module of rounds alone, which is what every encoding here is.
    for term in ctx.atom.delay:
        if itype_name(term.itype) != "Zero":
            out.append(
                f"the `delay` block gives wire #{term.write[0].id} a rate of "
                "change, and no encoding here has continuous time"
            )
            break
    return out


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
