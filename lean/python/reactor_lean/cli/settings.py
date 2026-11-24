"""A validated command line (`Settings`)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    """A validated command line.

    The decisions the grammar makes are made here, once: which proof rule
    (`kind`), say.  The pipeline reads; it does not re-derive.
    """

    module_file: str
    module_def: str                     # the callable in it that makes the module
    output_dir: Path
    project_name: str
    executable: bool

    prp: str | None                     # the property, whichever flag carried it
    kind: str                           # "safety" | "buchi"

    pre: str | None
    invariant: str | None
    ranking: str | None

    cert_file: str | None = None

    @property
    def is_safety(self) -> bool:
        return self.kind == "safety"
