"""A claim and a certificate read against the run's encoding of the module.

`ModuleQueries` holds their predicates as cvc5 terms over the module's
state, parsed once: the certificate's Lean is printed from them
(`proof.render`), and whatever asks cvc5 about the certificate states its
questions over them, on the solver and the round they come with.
`predicate_terms` is how a run gets one.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

from ..core.certificate import Certificate, Claim
from ..core.errors import Refused
from . import obligations as ob
from .limits import call_ms
from .encoding import encoding_of

if TYPE_CHECKING:
    from .limits import SmtBudget


# ══════════════════════════════════════════════════════════════════════
# Queries
# ══════════════════════════════════════════════════════════════════════


class ModuleQueries:
    """A cvc5 view of one module together with its certificate predicates.

    `source` is the module or its `smt.encoding.ModuleEncoding`; a run gets these
    from `ModuleEncoding.queries`, which builds one per claim and certificate,
    through `predicate_terms`, which refuses a predicate that does not parse
    by its role and anything else as the module's. A predicate the claim or
    the certificate leaves out is `None` here, and every reader states
    nothing for it.
    """

    def __init__(self, source, claim: Claim, cert: Certificate | None = None):
        enc = encoding_of(source)
        # Weak: `enc.queries` memoises this, and a cycle through the owner of
        # a `TermManager` is finalised in arbitrary order (`smt.module._LIVE`).
        self.enc = weakref.proxy(enc)
        self.tm = enc.tm
        self.msmt = enc.msmt
        self.env = enc.env
        self.state_vars = enc.state

        cert = cert or Certificate()
        self.kind = claim.kind
        self.prp = enc.parse(claim.prp, "property", kind=claim.kind)
        self.inv = enc.parse(cert.inv, "invariant")
        self.ranking = enc.parse(cert.ranking, "ranking", want="Int")
        self.init_pre = enc.parse(claim.init_pre, "pre")
        self.update_pre = enc.parse(claim.update_pre, "pre")

    def solver(self, budget: SmtBudget):
        """A solver on this module's terms, held to `budget`'s next call."""
        return self.enc.solver(call_ms(budget), logic="ALL")

    def frame(self, prefix: str) -> ob.Frame:
        """Fresh constants `{prefix}s0..` and the round from them, the
        preconditions read over their inputs."""
        return ob.Frame(self.enc, prefix, init_pre=self.init_pre,
                        update_pre=self.update_pre)


# ══════════════════════════════════════════════════════════════════════
# The claim, read against the module
# ══════════════════════════════════════════════════════════════════════


def predicate_terms(
    smt, claim: Claim, cert: Certificate = Certificate()
) -> "ModuleQueries":
    """`claim` and `cert` parsed against the module: what the Lean is printed
    from, and what a question to cvc5 about them is stated over. `smt` is
    the module, its encoding or a `smt.encoding.lazy` factory for one.

    A predicate that does not parse is refused by its role, and a module
    with no encoding (`ModuleSMT` reads one atom, `elem_sort` four sorts,
    and both raise `ValueError`) as the module's. Anything else is a bug,
    and keeps its traceback.
    """
    try:
        return encoding_of(smt).queries(claim, cert)
    except Refused:
        raise
    except ValueError as e:
        raise Refused(
            f"the certificate predicates cannot be read for this module: {e}"
        ) from e
