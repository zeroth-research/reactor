"""The obligations of one module, as a search asks them: over a *defined*
frame, and about facts rather than terms.

`reactor.lean.smt.obligations` states what `rule_buchi` and `rule_globally`
ask once, over any frame. `Obligations` is the frame a search asks them
over: constants `v_*`, and a successor that is constants `v_sp*` each query
defines, so a query about twenty facts carries the round once. Its queries
take `Candidate`s -- a fact or a ranking function, its source and its term
-- and name each hypothesis by its position, which is how an unsat core is
read back.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from reactor.lean.smt.obligations import (
    Query,
    hrank,
    init_inv,
    inv_imp_P,
    step_inv,
)

if TYPE_CHECKING:
    from .claim import SynthContext


@dataclass(frozen=True)
class Candidate:
    """One invariant fact or ranking function: its source and its term."""

    src: str                # SMT-LIB over `s0..`, what the certificate carries
    term: object            # the same, parsed, over `ctx.state`


class Obligations:
    """One module's obligations, as questions rather than as files: a
    defined frame (see the module docstring).

    The round is stated once per query as `defines` -- the successor state
    `v_sp*` over the latched state `v_s*` and the inputs -- and every fact
    is read at it, so a query about twenty facts carries the transition once
    rather than twenty times.

    A query that asks about several facts at once names each of them as a
    *probe*, so a solver that returns a counter-model says which of them it
    breaks. That is the whole of what the engine needs to take Houdini's
    step from one answer instead of one answer per fact; a solver that
    returns no model ignores them and the engine asks fact by fact.
    """

    def __init__(self, ctx: "SynthContext"):
        tm = ctx.tm
        self.ctx = ctx

        def consts(vs, prefix):
            """One holder per component, over one *constant* per element.

            A scalar component is a constant, as it always was. A
            matrix-shaped one is the constructor over a constant per
            element -- `(tuple v_s0_0 v_s0_1 v_s0_2)` -- so that a
            component is still one thing to substitute while every name
            that reaches a solver is a scalar. `_readable` folds the
            selects that leaves behind, and a prover with no tuple theory
            then never sees a tuple: not in a `declare-fun`, not in a
            `define-fun`, and not in an obligation.
            """
            out = []
            for i, v in enumerate(vs):
                sort = v.getSort()
                if not sort.isTuple():
                    out.append(tm.mkConst(sort, f"{prefix}{i}"))
                    continue
                out.append(tm.mkTuple([tm.mkConst(e, f"{prefix}{i}_{k}")
                                       for k, e in
                                       enumerate(sort.getTupleSorts())]))
            return out

        self.s = consts(ctx.state, "v_s")
        self.sp = consts(ctx.state, "v_sp")
        self.si = consts(ctx.state, "v_si")
        self.el = consts(ctx.extl_latched, "v_el")
        self.en = consts(ctx.extl_next, "v_en")
        readable = self._readable

        self.next = [readable(t) for t in ctx.enc.next_state(self.s, self.el, self.en)]
        self.init = [readable(t) for t in ctx.enc.init_state(self.en)]
        self.update_pre = readable(ctx.enc.with_inputs(ctx.update_pre, self.el, self.en))
        self.init_pre = readable(ctx.enc.with_inputs(ctx.init_pre, self.el, self.en))

    def _readable(self, t):
        """`t`, with every tuple the encoding leaves folded away.

        A module with scalar state can still compute through a matrix --
        `m_max` takes the max of a 2-vector -- and cvc5 encodes that as
        `tuple`/`tuple.select`, which Vampire's SMT-LIB front end does not
        have. The rewriter folds a select of a constructor away, so the
        round a prover is shown is the same function without them. Since
        the state holders are constructors too (`consts`), that covers a
        matrix-shaped *component* and not only an intermediate. It is only
        called where a tuple is printed: a term that has none is handed
        over exactly as encoded.
        """
        return self.ctx.enc.simplify(t) if "tuple" in str(t) else t

    def flatten(self, holders, terms=None) -> list:
        """One term per element: `terms` shaped by `holders`, or the holders.

        `holders` says what has elements and how many -- a constructor is a
        matrix-shaped component, anything else is a scalar -- and `terms`
        are the terms standing in those positions, which for the round is
        the successor and for the holders themselves is the constants. The
        two are separate because a caller that wants the *names* and one
        that wants the round both need the same shape read off the same
        place.
        """
        from cvc5 import Kind

        out = []
        for h, t in zip(holders, holders if terms is None else terms):
            if h.getKind() != Kind.APPLY_CONSTRUCTOR:
                out.append(t)
            else:
                out += [self._select(t, k)
                        for k in range(h.getNumChildren() - 1)]
        return out

    def slots(self, holders) -> list:
        """`(name, sort)` for every constant in `holders` -- one per element.

        What a caller drawing input values needs: the names are the ones a
        term mentions, so a matrix-shaped input is drawn element by element
        rather than as a tuple under a name nothing uses.
        """
        return [(str(c), c.getSort()) for c in self.flatten(holders)]

    def env_of(self, state, holders=None) -> dict:
        """A simulated state as values for the constants `holders` names.

        One entry per element, because that is what the constants are: a
        matrix-shaped component contributes its elements under the names
        `consts` gave them, not a tuple under the component's name.
        """
        from cvc5 import Kind

        out: dict = {}
        for h, v in zip(holders if holders is not None else self.s, state):
            if h.getKind() == Kind.APPLY_CONSTRUCTOR:
                for c, x in zip(list(h)[1:], v):
                    out[c.getSymbol()] = x
            else:
                out[str(h)] = v
        return out

    def _select(self, t, k):
        """Element `k` of `t`, pushed down to where the tuple is built
        (`select_element`): a round that branches, as `m_mixed` does when
        it resets its vector instead of decrementing it, is an `ite`
        between tuples, which `simplify` does not fold."""
        from reactor.lean.smt.encode import select_element

        return select_element(self.tm, t, k, leaf=self._readable)

    def _defines(self, holders, values) -> list:
        """The round as `define-fun`s: one per constant, so one per element.

        There is no constant of tuple sort to define -- by construction, see
        `consts` -- so a matrix-shaped component is defined element by
        element, the `k`-th from the `k`-th element of the round.
        """
        from cvc5 import Kind

        out = []
        for h, v in zip(holders, values):
            if h.getKind() != Kind.APPLY_CONSTRUCTOR:
                out.append((h, v))
                continue
            for k, c in enumerate(list(h)[1:]):
                out.append((c, self._select(v, k)))
        return out

    # --- the frame ------------------------------------------------------

    @property
    def tm(self):
        return self.ctx.tm

    @property
    def succ(self) -> list:
        return self.sp

    @property
    def entry(self) -> list:
        return self.si

    def at(self, pred, vs):
        """`pred` read at `vs` -- a term folded, because substituting a
        matrix-shaped holder leaves a select of a constructor where the
        component was."""
        if callable(pred):
            return pred(vs)
        return self._readable(pred.substitute(self.ctx.state, vs))

    def case(self, state) -> "_Case":
        """This frame with `state` for both the successor and the initial
        state, which nothing defines: one case of a round, its `ite`s
        folded away."""
        return _Case(self, state)

    def define_next(self) -> list:
        return self._defines(self.sp, self.next)

    def define_entry(self) -> list:
        return self._defines(self.si, self.init)

    # --- the queries ------------------------------------------------------

    def holds_at_entry(self, facts: list[Candidate]) -> Query:
        """`init_pre e -> fact (init e)`, for all of `facts` at once."""
        return init_inv(self, [f.term for f in facts])

    def preserved(self, hyps: list[Candidate], goals: list[Candidate]) -> Query:
        """`hyps s /\\ update_pre e -> goals (update s e)`."""
        return step_inv(self, _facts(hyps), [g.term for g in goals])

    def implies(self, hyps: list[Candidate], goal) -> Query:
        """`hyps s -> goal s`: the invariant is a proof of the property."""
        return inv_imp_P(self, _facts(hyps), goal)

    def drops(self, hyps: list[Candidate], rank: Candidate) -> Query:
        """`hrank` over the facts `hyps`."""
        from cvc5 import Kind

        q = hrank(self, _facts(hyps), self.ctx.prp, rank.term)
        # A refutation is a round the rank does not drop on, and the round
        # is what the fit learns from -- read back only for a state of
        # scalars, which is the shape a simulated round has.
        if all(h.getKind() != Kind.APPLY_CONSTRUCTOR for h in self.s):
            q = replace(q, witness=tuple(self.s) + tuple(self.sp))
        return q


class _Case:
    """`Obligations.case`."""

    def __init__(self, ob: Obligations, state):
        self.tm, self.s, self.el, self.en = ob.tm, ob.s, ob.el, ob.en
        self.init_pre, self.update_pre = ob.init_pre, ob.update_pre
        self.succ = self.entry = list(state)
        self.at = ob.at

    def define_next(self) -> tuple:
        return ()

    def define_entry(self) -> tuple:
        return ()


def _facts(hyps: list[Candidate]) -> list:
    """`hyps` named `h<i>`, by position: how an unsat core is read back."""
    return [(f"h{i}", h.term) for i, h in enumerate(hyps)]
