"""One module in cvc5 for a whole run.

`ModuleEncoding` owns the run's `TermManager`: the module's state and
inputs as cvc5 constants, and the predicates read against them. It is
built when the first predicate is read, so a run that reads none never
imports cvc5.
"""

from __future__ import annotations

from ..core.errors import PredicateRefused


class ModuleEncoding:
    """`module` in cvc5: its state and inputs as the parser declares them
    (`s*`, `el*`, `e*`), and each predicate read against them, once.

    Retained until exit (`smt.module.keep_alive`), its `TermManager` past
    it, and nothing it owns refers back to it: a cycle through a
    `TermManager`'s owner is finalised in arbitrary order, which the pinned
    manager makes safe and nothing makes cheap. So no exception that was
    raised is kept here: its traceback reaches the frame that raised it,
    and that frame's `self`.
    """

    def __init__(self, module):
        import cvc5

        from .module import ModuleSMT, keep_alive
        from .predicates import PredicateParser

        self.module = module
        self.tm = cvc5.TermManager()
        self.msmt = ModuleSMT(tm=self.tm, module=module)
        self.env = PredicateParser(self.msmt)
        self._parsed: dict = {}
        self._queries: dict = {}
        keep_alive(self, self.tm)

    @property
    def state(self) -> list:
        return self.env.state_vars

    @property
    def extl_latched(self) -> list:
        return self.env.extl_latched_vars

    @property
    def extl_next(self) -> list:
        return self.env.extl_next_vars

    # --- predicates -------------------------------------------------------

    def parse(self, src: str | None, role: str, want: str = "Bool",
              kind: str = "buchi"):
        """`src` as a term over the state and inputs, or `PredicateRefused`
        naming its `role` ("property" of `kind`, "pre", "invariant",
        "ranking").

        `want` is the sort the field is declared as and the Lean printer
        assumes: a property, an invariant and a precondition become the
        body of a `Prop`, a ranking function the body of a `Nat`. An Int
        where a Prop is expected elaborates to nothing, and a Bool ranking
        prints as Lean that does not parse, so the wrong sort is refused
        here, where it is known.
        """
        if src is None:
            return None
        from .predicates import Unreadable, parse_predicate

        hit = self._parsed.get(src)
        if hit is None:
            try:
                hit = parse_predicate(self.env, src)
            except Unreadable as e:
                # Its reason, in one that was never raised: a raised one's
                # traceback holds the frames it passed through, and this
                # one's `self`, which keeps it, would make a cycle.
                hit = Unreadable(str(e))
            self._parsed[src] = hit
        if isinstance(hit, Unreadable):
            raise PredicateRefused(
                role, f"cannot read `{src}`: {hit}", kind=kind) from hit
        got = hit.getSort()
        if not (got.isBoolean() if want == "Bool" else got.isInteger()):
            raise PredicateRefused(
                role, f"must have sort {want}, got {got}", kind=kind)
        return hit

    def queries(self, claim, cert=None):
        """`smt.queries.ModuleQueries` for `claim` and `cert`, built once per
        pair. Raises as its constructor does."""
        from ..core.certificate import Certificate
        from .queries import ModuleQueries

        key = (claim, cert or Certificate())
        q = self._queries.get(key)
        if q is None:
            q = self._queries[key] = ModuleQueries(self, *key)
        return q


def encoding_of(source) -> ModuleEncoding:
    """`source` as an encoding: itself, what a factory for one builds, or a
    new one for a bare module."""
    if isinstance(source, ModuleEncoding):
        return source
    if callable(source):
        return source()
    return ModuleEncoding(source)
