"""SMT-LIB text a route writes: the literals in it, and the sums and
conjunctions a predicate it found is written as.

What a script is, and how one is printed, is `reactor.lean.smt.script`.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterable
from fractions import Fraction
from re import Match, compile as _compile


# ══════════════════════════════════════════════════════════════════════════
# Literals
# ══════════════════════════════════════════════════════════════════════════
#
# Vampire's SMT-LIB front end sorts literals strictly, and cvc5's printer does
# not write for it: a rational prints as `(/ 1 2)`, whose arguments are Int
# ("invalid sort $int for interpretation /"), and an Int numeral anywhere a
# Real is expected is a parse error rather than a coercion. So every real
# literal written for a script is a decimal, and every term cvc5 prints is
# passed through `decimals` on its way into one. cvc5 reads a decimal as
# happily as it reads `(/ 1 2)`, so a candidate's source uses them too.


def smt_int(value: int) -> str:
    """An integer literal. SMT-LIB has no negative numerals, so one is a negation."""
    value = int(value)
    return str(value) if value >= 0 else f"(- {-value})"


def smt_real(value) -> str:
    """A Real literal: a decimal where one is exact, else a quotient of two.

    `1/2` is `0.5` and `1/3` is `(/ 1.0 3.0)` -- a decimal cannot say the
    second, and rounding it would state a different obligation.
    """
    q = Fraction(value)
    sign, q = ("(- ", -q) if q < 0 else ("", -(-q))
    close = ")" if sign else ""
    rest = q.denominator
    for factor in (2, 5):
        while rest % factor == 0:
            rest //= factor
    if rest != 1:
        return f"{sign}(/ {q.numerator}.0 {q.denominator}.0){close}"
    places = 0
    while 10 ** places % q.denominator:
        places += 1
    digits = str(q.numerator * 10 ** places // q.denominator).rjust(places + 1, "0")
    whole, frac = digits[:len(digits) - places], digits[len(digits) - places:]
    return f"{sign}{whole}.{frac or '0'}{close}"


def smt_lit(value, sort) -> str:
    """`value` as a literal of `sort`: an Int numeral, or a decimal."""
    return smt_real(value) if sort.isReal() else smt_int(int(value))


# `(/ 1 2)`, `(/ (- 1) 4)`: a rational as cvc5 prints one. Integer division
# prints as `div`, so a `/` whose arguments are both numerals is always this.
_RATIONAL = _compile(r"\(/ (?:\(- (\d+)\)|(\d+)) (\d+)\)")


def decimals(text: str) -> str:
    """`text` with cvc5's rational literals rewritten as Vampire reads them."""
    def fix(m: Match) -> str:
        neg, num, den = m.groups()
        return smt_real(Fraction(-int(neg) if neg else int(num), int(den)))

    return _RATIONAL.sub(fix, text)


def smtlib_source(src: str, read: Callable[[], object]) -> str:
    """The predicate `src` as SMT-LIB text.

    That is `src` as written if it is SMT-LIB, and cvc5's printing of the
    term `read()` returns for it if it is the DSL's Python syntax
    (`s0[0][0] == 0`); `read` is called for nothing else, so SMT-LIB text
    needs no cvc5. No SMT-LIB text a predicate is spliced into,
    `(ite P 0 r)` say, can read the DSL, and neither can z3. SMT-LIB that
    is also a Python expression (`s0`, `(not s0)`) prints as it was
    written.
    """
    src = src.strip()
    try:
        ast.parse(src, mode="eval")
    except SyntaxError:
        return src
    return decimals(str(read()))


# ══════════════════════════════════════════════════════════════════════════
# Predicates written out
# ══════════════════════════════════════════════════════════════════════════
#
# A predicate found as a term is written as text rather than printed from
# that term: the string is parsed back and rendered as the body of a Lean
# definition, and a printer that `let`-binds a repeated subterm writes
# something that cannot be one.


def sum_smt(parts: Iterable[str]) -> str:
    """SMT-LIB terms added up: `0` for none, one as itself."""
    parts = list(parts)
    if not parts:
        return "0"
    return parts[0] if len(parts) == 1 else "(+ " + " ".join(parts) + ")"


def affine_smt(const, coeffs, names, *, lit=smt_int) -> str:
    """``k + Σ cᵢ·nᵢ`` as SMT-LIB, with the terms that say nothing dropped.

    A coefficient of one is the term itself and minus one its negation;
    `lit` writes the rest and the constant (`smt_real` over a Real form).
    """
    parts = [lit(const)] if const else []
    for c, n in zip(coeffs, names):
        if c == 1:
            parts.append(n)
        elif c == -1:
            parts.append(f"(- {n})")
        elif c:
            parts.append(f"(* {lit(c)} {n})")
    return sum_smt(parts)


def conjunction_smt(parts: Iterable[str]) -> str:
    """SMT-LIB Bool terms as one: `true` for none, one as itself, each
    once and none of them `true`."""
    rendered = list(dict.fromkeys(p for p in parts if p and p != "true"))
    if not rendered:
        return "true"
    if len(rendered) == 1:
        return rendered[0]
    return "(and " + " ".join(rendered) + ")"
