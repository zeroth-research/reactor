"""SMT-LIB out: the script printer.

Every script handed to a solver outside this process, or left in
`artifacts/` for a reader, is printed by :func:`script`.
"""

from __future__ import annotations

from collections.abc import Iterable

from .terms import free_constants


def declarations(terms: Iterable, *, skip=()) -> list[str]:
    """A `declare-fun` for each free constant `terms` mention, by symbol,
    less the names in `skip`."""
    seen = free_constants(terms)
    return [f"(declare-fun {sym} () {seen[sym].getSort()})"
            for sym in sorted(seen) if sym not in skip]


def script(asserts: Iterable = (), *, defines: Iterable = (),
           mentions: "Iterable | None" = None, logic: str = "ALL",
           command: str = "assert", spaced: bool = True) -> str:
    """`asserts` as a script that runs on its own under any SMT-LIB solver.

    `defines` are `(name, params, sort, body)`, each `params` a list of
    `(name, sort)`, emitted as `define-fun` ahead of the assertions.
    Declarations are collected from `mentions` -- by default the define
    bodies and the assertions, which are then terms -- less every name a
    define binds, so a caller states what it wants said and not what it
    depends on. An assertion may be text, for a form the caller writes
    itself (a `:named` one, say), and `mentions` then says what it reads.
    `command` is what each assertion is put under; `spaced` separates the
    sections with a blank line, for a reader.
    """
    asserts, defines = list(asserts), list(defines)
    if mentions is None:
        mentions = [body for _n, _p, _s, body in defines] + asserts
    bound = ({name for name, _p, _s, _b in defines}
             | {p for _n, params, _s, _b in defines for p, _ps in params})
    decls = declarations(mentions, skip=bound)
    gap = [""] if spaced else []
    lines = [f"(set-logic {logic})", *gap, *decls]
    if decls:
        lines += gap
    for name, params, sort, body in defines:
        args = " ".join(f"({p} {srt})" for p, srt in params)
        lines.append(f"(define-fun {name} ({args}) {sort} {body})")
    if defines:
        lines += gap
    lines += [f"({command} {term})" for term in asserts]
    lines += [*gap, "(check-sat)"]
    return "\n".join(lines) + "\n"
