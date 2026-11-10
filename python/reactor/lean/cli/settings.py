"""A validated command line (`Settings`)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    """A validated command line.

    The decisions the grammar makes are made here, once: which proof rule
    (`kind`), which input format, and the SMT budget.  The pipeline reads;
    it does not re-derive.
    """

    module_file: str | None
    output_dir: Path
    project_name: str
    input_format: str | None            # an `inputs` row; None without input
    executable: bool

    prp: str | None                     # the property, whichever flag carried it
    kind: str                           # "safety" | "buchi"

    pre: str | None
    invariant: str | None
    ranking: str | None

    input_opts: dict = field(default_factory=dict)

    cert_file: str | None = None
    hammer_file: str | None = None
    build_cert: bool = False


    @property
    def is_safety(self) -> bool:
        return self.kind == "safety"

    @property
    def supplied_complete(self) -> bool:
        """Whether the flags alone carry a certificate worth building.

        A safety certificate is complete without a ranking function, so what
        counts follows the kind.
        """
        return bool(self.prp and self.invariant and (self.is_safety or self.ranking))
