"""The module's Lean encodings: ``ModuleToLean4``, one submodule per encoding.

`functional`, `circuit`, `scalar` and `relational` (the matrix and the
scalar relational ones) are the encodings, and `common` is what the scalar
ones share.
"""

from functools import cached_property

from reactor_lean.codegen.context import LeanContext

from reactor_lean.codegen.encodings.functional import atom_to_lean_functional, constants_block
from reactor_lean.codegen.encodings.circuit import (
    atom_to_lean_circuit as _atom_to_lean_circuit,
    to_lean_equiv_theorems as _to_lean_equiv_theorems,
)
from reactor_lean.codegen.encodings.scalar import (
    atom_to_lean_scalar as _atom_to_lean_scalar,
    to_lean_scalar_equiv as _to_lean_scalar_equiv,
)
from reactor_lean.codegen.encodings.relational import (
    atom_to_lean_rel as _atom_to_lean_rel,
    atom_to_lean_mat_rel as _atom_to_lean_mat_rel,
)


class ModuleToLean4:
    """Convert a Python Module into Lean4 wiring diagram code.

    Read-only consumer of a pre-populated LeanContext — does not mutate it.

    Every method stands alone: what the circuit's equivalence theorems
    need of its encoding (its layer names) is computed once, by whichever
    method asks first.
    """

    def __init__(self, ctx: LeanContext):
        self.ctx = ctx

    @cached_property
    def _circuit(self) -> "tuple[str, list[str], list[str]]":
        """The circuit encoding, and its init and update layer names."""
        return _atom_to_lean_circuit(self.ctx)

    # ------------------------------------------------------------------
    # Functional encoding
    # ------------------------------------------------------------------

    def to_lean_functional(self) -> str:
        """Generate the functional init/update definitions (plus constants)."""
        return "{}\n\n{}".format(
            constants_block(self.ctx), atom_to_lean_functional(self.ctx)
        )

    # ------------------------------------------------------------------
    # Circuit encoding
    # ------------------------------------------------------------------

    def to_lean_equiv_theorems(self) -> str:
        """Generate theorems proving circuit ≡ functional."""
        _, init_names, update_names = self._circuit
        return _to_lean_equiv_theorems(self.ctx, init_names, update_names)

    def to_lean_circ(self) -> str:
        """The module as a combinational circuit, and circuit ≡ functional."""
        return "{}\n\n{}".format(
            self._circuit[0],
            self.to_lean_equiv_theorems(),
        )

    # ------------------------------------------------------------------
    # Scalar encoding
    # ------------------------------------------------------------------

    def to_lean_scalar_equiv(self) -> str:
        """Generate theorems proving init/update = pack ∘ Scalar.init/update ∘ unpack."""
        return _to_lean_scalar_equiv(self.ctx)

    def to_lean_scalar(self) -> str:
        """The scalar encoding inside ``namespace Scalar``, and its
        composition theorems."""
        return "{}\n\n{}".format(
            _atom_to_lean_scalar(self.ctx),
            self.to_lean_scalar_equiv(),
        )

    # ------------------------------------------------------------------
    # Relational encoding
    # ------------------------------------------------------------------

    def to_lean_rel(self) -> str:
        """Generate the relational encoding inside ``namespace ScalarRel``."""
        return _atom_to_lean_rel(self.ctx)

    def to_lean_mat_rel(self) -> str:
        """Generate the matrix-domain relational encoding inside ``namespace Rel``."""
        return _atom_to_lean_mat_rel(self.ctx)

