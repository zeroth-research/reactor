"""Obligations as questions: named hypotheses, a goal, and the round they
read.

A `Query` is one obligation. What `rule_buchi` and `rule_globally` ask --
`init_inv`, `step_inv`, `inv_imp_P` and `hrank` -- is stated once, here,
over a *frame*: the constants an obligation is stated over and the round
from them.

A `Frame` is inline: fresh constants `{prefix}s0..`, `{prefix}el0..` and
`{prefix}en0..`, and the successor is the encoded round substituted in.

A precondition is taken as given: `None` is no hypothesis, and a literal
`true` is one.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property


@dataclass(frozen=True)
class Query:
    """One obligation: named hypotheses, a goal, and the round they read.

    `defines` states the round once -- the successor state over the latched
    state and the inputs -- so a query about twenty facts carries the
    transition once rather than twenty times.
    """

    hyps: tuple[tuple[str, object], ...]
    goal: object
    defines: tuple[tuple[object, object], ...] = ()

    @property
    def key(self):
        return (self.hyps, self.goal, self.defines)

    def negation(self, tm):
        """The hypotheses and the goal's negation, conjoined: unsat exactly
        when the obligation holds. Inline only -- a defined successor would
        be left free."""
        from cvc5 import Kind

        assert not self.defines, "a defined round has no negation form"
        return conjunction(tm, [t for _, t in self.hyps]
                           + [tm.mkTerm(Kind.NOT, self.goal)])

    def labelled(self, tm) -> dict:
        """`negation` as named parts, the goal's negation as `goal`. A
        literal `true` or `false` is left out: an unsat core never needs it."""
        from cvc5 import Kind

        parts = dict(self.hyps)
        parts["goal"] = tm.mkTerm(Kind.NOT, self.goal)
        return {k: v for k, v in parts.items()
                if v.getKind() != Kind.CONST_BOOLEAN}


# ══════════════════════════════════════════════════════════════════════════
# The obligations, over a frame
#
# A frame has `tm`; the constants `s`; `succ`, the state after one round from
# them, and `entry`, the state after tick 0; the preconditions `init_pre` and
# `update_pre` over its inputs; `at(pred, vs)`, a predicate read at the state
# `vs`; and `define_next()` / `define_entry()`, what a query must define for
# `succ` / `entry` to mean the round. A predicate is a term over the parser's
# state, or a function of a state (a template with its coefficients open).
# `given` are hypotheses about nothing a frame reads, such as the guard of
# one case of a round; they follow the precondition. Each statement builds
# its terms in the order written, which is the order cvc5 numbers them in.
# ══════════════════════════════════════════════════════════════════════════


def conjunction(tm, terms):
    """`terms` conjoined: `true` for none, the term itself for one."""
    from cvc5 import Kind

    if not terms:
        return tm.mkBoolean(True)
    return terms[0] if len(terms) == 1 else tm.mkTerm(Kind.AND, *terms)


def clamp(tm, t):
    """`t` as Lean's `Int.toNat` reads an Int: itself when non-negative,
    else 0."""
    from cvc5 import Kind

    zero = tm.mkInteger(0)
    return tm.mkTerm(Kind.ITE, tm.mkTerm(Kind.GEQ, t, zero), t, zero)


def _named(f, hyps) -> list:
    return [(name, f.at(term, f.s)) for name, term in hyps]


def _given(given) -> list:
    return [(f"case{i}", g) for i, g in enumerate(given)]


def _outside(f, hyps, prp) -> list:
    """What a ranking obligation assumes: `hyps` and `~P` at `s`, and the
    update precondition."""
    from cvc5 import Kind

    return _named(f, hyps) + [
        ("notP", f.tm.mkTerm(Kind.NOT, f.at(prp, f.s))),
        ("pre", f.update_pre),
    ]


def _query(hyps, goal, *, defines=()) -> Query:
    """Named hypotheses and a goal. An absent precondition is no
    hypothesis."""
    return Query(
        hyps=tuple((n, h) for n, h in hyps if h is not None),
        goal=goal,
        defines=tuple(defines),
    )


def init_inv(f, goals: list, *, given=()) -> Query:
    """`init_pre e -> goals (init e)`, for all of `goals` at once."""
    entry = f.entry
    pre = f.init_pre
    at = [f.at(g, entry) for g in goals]
    goal = conjunction(f.tm, at)
    return _query([("pre", pre)] + _given(given), goal,
                  defines=f.define_entry())


def step_inv(f, hyps: list, goals: list, *, given=()) -> Query:
    """`hyps s /\\ update_pre e -> goals (update s e)`; `hyps` are named."""
    succ = f.succ
    named = _named(f, hyps)
    pre = f.update_pre
    at = [f.at(g, succ) for g in goals]
    goal = conjunction(f.tm, at)
    return _query(named + [("pre", pre)] + _given(given), goal,
                  defines=f.define_next())


def inv_imp_P(f, hyps: list, goal) -> Query:
    """`hyps s -> goal s`: the invariant is a proof of the property."""
    named = _named(f, hyps)
    return _query(named, f.at(goal, f.s))


def hrank(f, hyps: list, prp, rank) -> Query:
    """`hyps s /\\ ~P s /\\ update_pre e -> V (update s e) < V s`.

    With `V = Int.toNat rank`, exactly as `rule_buchi` states it: a clamp
    on both sides, so the rank may be anything where the property holds.
    """
    from cvc5 import Kind

    succ = f.succ
    named = _outside(f, hyps, prp)
    goal = f.tm.mkTerm(Kind.LT, clamp(f.tm, f.at(rank, succ)),
                       clamp(f.tm, f.at(rank, f.s)))
    return _query(named, goal, defines=f.define_next())


class Frame:
    """An inline frame: fresh constants, and the encoded round from them.

    `enc` is the run's `smt.encoding.ModuleEncoding`; `init_pre` and
    `update_pre` are over the parser's inputs, as it reads them. `succ`,
    `entry` and the preconditions are built on first use, so a statement
    that never reads one never encodes it.
    """

    def __init__(self, enc, prefix: str, *, init_pre=None, update_pre=None,
                 consts=None):
        self.enc = enc
        self.tm = enc.tm
        self.s, self.el, self.en = consts or enc.fresh(prefix)
        self._pres = (init_pre, update_pre)

    @cached_property
    def succ(self) -> list:
        return self.enc.next_state(self.s, self.el, self.en)

    @cached_property
    def entry(self) -> list:
        return self.enc.init_state(self.en)

    @cached_property
    def init_pre(self):
        return self.enc.with_inputs(self._pres[0], self.el, self.en)

    @cached_property
    def update_pre(self):
        return self.enc.with_inputs(self._pres[1], self.el, self.en)

    def at(self, pred, vs):
        return pred(vs) if callable(pred) else self.enc.at(pred, vs)

    def define_next(self) -> tuple:
        return ()

    def define_entry(self) -> tuple:
        return ()
