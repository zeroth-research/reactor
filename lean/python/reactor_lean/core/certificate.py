"""What a certificate is about, and what it says.

Every predicate here is SMT-LIB source over the state `s0..sN-1` and the
inputs `e0..` (awaited) / `el0..` (latched) -- the text
`parse.native.read_predicates` reads. The Python predicate DSL is accepted at
the CLI and library edge only; Lean is produced from these in one place,
`proof.render.render_certificate`.

A leaf module: it imports no Jinja, so the CLI, the parsers and the code
generators can all depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Claim:
    """The property a certificate proves, under which rule and assumptions.

    `kind` is the proof rule, and so what `prp` means:

      "buchi"  -- `G (F prp)`, `rule_buchi`: an invariant and a ranking
                  function that drops on every step off `prp`. `--buchi`.
      "safety" -- `G prp`, `rule_globally`: an invariant that implies
                  `prp`, and no ranking. `--safety`.

    `init_pre` / `update_pre` constrain the inputs `init` and `update` see.
    """

    prp: str | None = None
    kind: Literal["safety", "buchi"] = "buchi"
    init_pre: str | None = None
    update_pre: str | None = None


@dataclass(frozen=True)
class Certificate:
    """The predicates that prove a `Claim`. A safety one has no ranking."""

    inv: str | None = None
    ranking: str | None = None
