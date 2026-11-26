"""The module's Lean encodings: ``ModuleToLean4``, one submodule per encoding.

`functional`, `scalar` and `relational` (the matrix and the scalar
relational ones) are the encodings, and `common` is what the scalar ones
share.
"""

from ..context import LeanContext

from .functional import atom_to_lean_functional, constants_block
from .scalar import (
    atom_to_lean_scalar as _atom_to_lean_scalar,
    to_lean_scalar_equiv as _to_lean_scalar_equiv,
)
from .relational import (
    atom_to_lean_rel as _atom_to_lean_rel,
    atom_to_lean_mat_rel as _atom_to_lean_mat_rel,
)


class ModuleToLean4:
    """A module's Lean encodings, one method each, every one of them
    printed from the `LeanContext` alone, which none of them changes.
    """

    def __init__(self, ctx: LeanContext):
        self.ctx = ctx

    # ------------------------------------------------------------------
    # Functional encoding
    # ------------------------------------------------------------------

    def to_lean_functional(self) -> str:
        """Generate the functional init/update definitions (plus constants)."""
        return "{}\n\n{}".format(
            constants_block(self.ctx), atom_to_lean_functional(self.ctx)
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

