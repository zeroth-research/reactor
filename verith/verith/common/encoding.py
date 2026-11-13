"""The run's one encoding of the module, as a question to cvc5 is asked
on it.

`reactor` reads the claim and the certificate against the module
(`reactor.lean.smt.encoding.ModuleEncoding`). `Encoding` is that encoding
with what an obligation needs besides: the round, fresh constants to state
it over, the rewriter, and the solvers built on its terms. `lazy(module)`
is how a run holds one, so a run that asks cvc5 nothing never imports it;
`predicate_terms` is the claim and the certificate read on one.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING

from reactor.lean.core.certificate import Certificate, Claim
from reactor.lean.smt import queries
from reactor.lean.smt.encoding import ModuleEncoding

from .limits import bounded_solver, call_ms

if TYPE_CHECKING:
    from reactor.lean.smt.queries import ModuleQueries

    from .limits import SmtBudget


class Encoding(ModuleEncoding):
    """`module` in cvc5, encoded once, with its round.

    The transition is translated on first use over the parser's `s*`, `el*`
    and `e*`, and every other state or input is substituted into it; a
    module with no encoding raises the same error on every use. What is
    kept here is held to `ModuleEncoding`'s rule: nothing refers back to
    the encoding, so no exception that was raised is kept.
    """

    def __init__(self, module):
        super().__init__(module)
        self._blocks: dict = {}
        self._solvers: list = []
        self._rewriter = None

    # --- the transition ---------------------------------------------------

    def _block(self, name: str) -> list:
        # A translation that fails is not kept, and fails again on the next
        # use: kept, its exception's traceback would hold this frame, and
        # through `self` the encoding would keep its own error alive in a
        # cycle. Translating has no side effects, so the retry raises the
        # same error.
        hit = self._blocks.get(name)
        if hit is None:
            if name == "init":
                hit = self.msmt.init_state(self.extl_next)
            else:
                hit = self.msmt.update_state(
                    self.state, self.extl_latched, self.extl_next)
            self._blocks[name] = hit
        return hit

    def init_state(self, en: list) -> list:
        """The state after tick 0, given the inputs `en` awaited then."""
        old = list(self.extl_next)
        terms = self._block("init")
        return [t.substitute(old, list(en)) for t in terms] if old else list(terms)

    def next_state(self, s: list, el: list, en: list) -> list:
        """The state after one round from `s`, given the inputs latched
        (`el`) and awaited (`en`)."""
        old = list(self.state) + list(self.extl_latched) + list(self.extl_next)
        new = list(s) + list(el) + list(en)
        terms = self._block("next")
        return [t.substitute(old, new) for t in terms] if old else list(terms)

    # --- terms ------------------------------------------------------------

    def fresh(self, prefix: str = "") -> tuple[list, list, list]:
        """New constants for a state and the inputs latched and awaited:
        `{prefix}s0..`, `{prefix}el0..`, `{prefix}en0..`."""
        return (self.msmt.fresh_ctrl(f"{prefix}s"),
                self.msmt.fresh_extl_l(f"{prefix}el"),
                self.msmt.fresh_extl_n(f"{prefix}en"))

    def at(self, term, vs: list):
        """`term` with the state replaced by `vs`."""
        return None if term is None else term.substitute(self.state, vs)

    def with_inputs(self, term, el: list, en: list):
        """`term` with the parser's inputs replaced by `el` / `en`."""
        if term is None:
            return None
        old = list(self.extl_next) + list(self.extl_latched)
        new = list(en) + list(el)
        return term.substitute(old, new) if old else term

    def simplify(self, term):
        """`term` through cvc5's rewriter, on a solver this encoding keeps
        for that alone."""
        if self._rewriter is None:
            import cvc5

            self._rewriter = cvc5.Solver(self.tm)
            self._rewriter.setLogic("ALL")
        return self._rewriter.simplify(term)

    def solver(self, ms: int, *, logic: str | None = None, **options):
        """`limits.bounded_solver` on this encoding's terms."""
        s = bounded_solver(self.tm, ms, logic=logic, **options)
        self._solvers.append(s)
        return s


def lazy(module) -> Callable[[], Encoding]:
    """`module`'s encoding, built on the first call and the same after."""
    return functools.cache(lambda: Encoding(module))


def encoding_of(source) -> Encoding:
    """`source` as an encoding: itself, what a `lazy` factory builds, or a
    new one for a bare module."""
    if isinstance(source, ModuleEncoding):
        return source
    if callable(source):
        return source()
    return Encoding(source)


def predicate_terms(smt, claim: Claim,
                    cert: Certificate = Certificate()) -> "ModuleQueries":
    """`claim` and `cert` read against the module, as `reactor` reads them,
    on an `Encoding`: `smt` is the module, its encoding or a `lazy` factory
    for one. Refuses as `reactor.lean.smt.queries.predicate_terms` does."""
    return queries.predicate_terms(lambda: encoding_of(smt), claim, cert)


def solver_for(q: "ModuleQueries", budget: "SmtBudget"):
    """A solver on `q`'s terms, held to `budget`'s next call."""
    return q.enc.solver(call_ms(budget), logic="ALL")
