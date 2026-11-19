"""SMT-LIB text as s-expressions: the reading half of the native parser,
with no notion of what a term means.

An expression is an `Atom` or a tuple of expressions. An atom keeps its
token's kind -- a numeral, a decimal, a bit-vector literal, a symbol or a
keyword -- because `5` and `|5|` are different things to SMT-LIB. What is
no SMT-LIB expression is `Unreadable`, saying where.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..core.errors import Unreadable


@dataclass(frozen=True)
class Atom:
    """One token: `kind` is "numeral", "decimal", "binary", "hex",
    "symbol" or "keyword"; `text` is as written (a quoted symbol without
    its bars); `at` is its offset in the source."""

    kind: str
    text: str
    at: int

    def __str__(self) -> str:
        return self.text


# SMT-LIB 2.6, section 3.1. A numeral has no leading zero, a simple symbol
# does not start with a digit, and `|...|` quotes any symbol but `|` and `\`.
_TOKEN = re.compile(r"""
    (?P<ws>\s+|;[^\n]*)
  | (?P<open>\()
  | (?P<close>\))
  | (?P<decimal>(?:0|[1-9][0-9]*)\.[0-9]+)
  | (?P<numeral>0|[1-9][0-9]*)
  | (?P<binary>\#b[01]+)
  | (?P<hex>\#x[0-9a-fA-F]+)
  | (?P<quoted>\|[^|\\]*\|)
  | (?P<keyword>:[A-Za-z0-9~!@$%^&*_\-+=<>.?/]+)
  | (?P<symbol>[A-Za-z~!@$%^&*_\-+=<>.?/][A-Za-z0-9~!@$%^&*_\-+=<>.?/]*)
""", re.VERBOSE)


_SYMBOL_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789~!@$%^&*_-+=<>.?/")


def tokens(src: str) -> list:
    """`src` as `Atom`s and the strings "(" and ")"; `Unreadable` at the
    first character no token starts with."""
    out, at = [], 0
    while at < len(src):
        m = _TOKEN.match(src, at)
        if m is None or m.end() == at:
            raise Unreadable(f"no SMT-LIB token starts at `{src[at:at + 12]}`")
        kind = m.lastgroup
        if kind in ("open", "close"):
            out.append(m.group())
        elif kind == "quoted":
            out.append(Atom("symbol", m.group()[1:-1], at))
        elif kind != "ws":
            out.append(Atom(kind, m.group(), at))
        at = m.end()
        # `007` and `1x` are not a numeral and something after it.
        if kind in ("numeral", "decimal") and at < len(src) and src[at] in _SYMBOL_CHARS:
            end = at
            while end < len(src) and src[end] in _SYMBOL_CHARS:
                end += 1
            raise Unreadable(f"`{src[m.start():end]}` is no numeral")
    return out


def read(src: str):
    """The one expression `src` holds: an `Atom` or a tuple of expressions."""
    toks = tokens(src)
    if not toks:
        raise Unreadable("the expression is empty")
    expr, k = _read(toks, 0)
    if k != len(toks):
        raise Unreadable(f"text after the expression: `{_show(toks[k:])}`")
    return expr


def _read(toks: list, k: int):
    tok = toks[k]
    if tok == ")":
        raise Unreadable("`)` with no `(` before it")
    if tok != "(":
        return tok, k + 1
    items, k = [], k + 1
    while True:
        if k == len(toks):
            raise Unreadable("`(` with no `)` to close it")
        if toks[k] == ")":
            return tuple(items), k + 1
        item, k = _read(toks, k)
        items.append(item)


def _show(toks: list) -> str:
    text = " ".join(t if isinstance(t, str) else t.text for t in toks)
    return text.replace("( ", "(").replace(" )", ")")


def show(expr) -> str:
    """`expr` as SMT-LIB text again, for an error message."""
    if isinstance(expr, Atom):
        return expr.text
    return "(" + " ".join(show(e) for e in expr) + ")"
