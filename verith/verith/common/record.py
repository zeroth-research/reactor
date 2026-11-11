"""What a run leaves in `artifacts/`: a predicate it found, or a note --
and the lines a found certificate is announced in.

The writers take the store the route was handed, and write nothing when it
is `None`: a run with no project to keep `artifacts/` in. Under
`--artifacts ignore` there is a store, and they write to it.
"""

from __future__ import annotations

from collections.abc import Callable

from reactor.lean.core.certificate import Certificate
from reactor.lean.core.errors import Refused


def announce(log: Callable, tag: str, cert: Certificate) -> Certificate:
    """`cert`, logged as the `[tag] inv:` and `[tag] ranking:` lines the
    bench harness reads back (`tests/bench_matrix/run_matrix.py`)."""
    log(f"[{tag}] inv: {cert.inv}")
    if cert.ranking is not None:
        log(f"[{tag}] ranking: {cert.ranking}")
    return cert


def found_certificate(store, cert: Certificate, *, inv: str,
                      ranking: str) -> None:
    """Both halves of `cert` as `found`, each with what it is."""
    found(store, "inv", cert.inv, what=inv)
    if cert.ranking is not None:
        found(store, "ranking", cert.ranking, what=ranking)


def found(store, role: str, src: str, *, what: str) -> None:
    """`src`, the `role` this run found and proved, as SMT-LIB.

    `proved` is the status that makes it resumable: a later run can take
    it as given.
    """
    if store is None:
        return
    store.put(role, src, status="proved", language="smt", what=what)


def note(store, text: str, *, status: str, what: str, why: str) -> None:
    """A note for the next run: what this one ruled out, or why it
    stopped."""
    if store is None:
        return
    store.note(text, status=status, what=what, why=why)


class GaveUp(Refused):
    """A route's refusal after it searched and found nothing, carrying the
    note it leaves in `artifacts/` for the next run.

    `status`, `what` and `why` are as `note` takes them; the note's text is
    the refusal itself, or `text` when the route has something else to say
    there -- a search's own account of the space, which goes on into a later
    prompt verbatim. The route only raises it: `verith.run.note_refusal`
    writes every give-up note, so there is one writer of them.
    """

    def __init__(self, message: str, *, flag: str, status: str, what: str,
                 why: str, text: "str | None" = None):
        super().__init__(message, searched=True, flag=flag)
        self.status, self.what, self.why = status, what, why
        self._text = text

    @property
    def text(self) -> str:
        return f"{self if self._text is None else self._text}\n"
