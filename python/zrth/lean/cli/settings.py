"""A validated command line (`Settings`), and the predicate flags a route
can be seeded from (`SEED_FLAGS`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..infer import InferRoute


# One row per predicate flag: the seed it supplies, as a route's `seeds`
# names it; the `Settings` attribute that carries it; and the flag to name
# in a refusal. The first two differ for `inv`, which is why the attribute
# is in the table rather than patched at the one place that reads it.
SEED_FLAGS: tuple[tuple[str, str, str], ...] = (
    ("inv", "invariant", "--invariant"),
    ("ranking", "ranking", "--ranking"),
    ("pre", "pre", "--pre"),
)


@dataclass
class Settings:
    """A validated command line.

    The decisions the grammar makes are made here, once: which proof rule
    (`kind`), which route (`route`), what that route resolved at parse time
    (`route_config`), and the SMT budget.  The pipeline reads; it does not
    re-derive.
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

    route: InferRoute | None = None
    route_opts: dict = field(default_factory=dict)
    route_config: object | None = None

    cert_file: str | None = None
    hammer_file: str | None = None
    build_cert: bool = False


    model: str | None = None
    base_url: str | None = None

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
