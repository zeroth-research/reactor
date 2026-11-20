"""`--pre-check cvc5`: is each obligation of the certificate even true?

That is the question worth asking before a `lake build` that takes tens of
seconds, since a failing build cannot tell a wrong certificate from tactics
that are merely too weak. It is cheap: measured on the case matrix, all three
obligations resolve in 3-13 ms per module where `lake build` takes 9-79 s.
`pre_check` asks it of the certificate's predicates as `reactor` parsed
them (`smt.queries.ModuleQueries`), and leaves each question and its answer
in `artifacts/`.

Everything here is on a leash. Each query gets `per_call_ms` and the check
as a whole `phase_ms`, both enforced by cvc5's own `tlimit`, so a hard
problem returns `unknown` rather than hanging the run. Any failure at all --
a timeout, an `unknown`, an unsupported operator, a cvc5 exception -- is
reported as `Status.UNKNOWN`, and nothing the run writes changes because of
it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from reactor_lean.core.certificate import Certificate, Claim, LeanAnswer
from reactor_lean.core.errors import Refused

from ..common import obligations as ob
from ..common.encoding import encoding_of, solver_for
from ..common.smtlib import script

if TYPE_CHECKING:
    from reactor_lean.smt.queries import ModuleQueries

    from ..common.limits import SmtBudget


# ══════════════════════════════════════════════════════════════════════
# Results
# ══════════════════════════════════════════════════════════════════════


class Status(str, Enum):
    HOLDS = "holds"          # negation UNSAT -- the obligation is true
    REFUTED = "refuted"      # negation SAT   -- there is a counterexample
    UNKNOWN = "unknown"      # timeout, incompleteness, or anything went wrong


@dataclass
class Verdict:
    name: str
    status: Status
    elapsed_ms: float
    detail: str | None = None   # counterexample, or why it is unknown

    @property
    def line(self) -> str:
        mark = {
            Status.HOLDS: "holds   ",
            Status.REFUTED: "REFUTED ",
            Status.UNKNOWN: "unknown ",
        }[self.status]
        out = f"   {self.name:<9} {mark} {self.elapsed_ms:6.0f} ms"
        if self.detail:
            out += f"   {self.detail}"
        return out


# How a verdict is recorded on the obligation artifact it answers. The two
# vocabularies are separate on purpose: `Status` is what this module decides,
# `artifacts.STATUSES` is what every producer in the package shares, and a
# consumer filtering `artifacts/` should not have to know cvc5 wrote it.
_VERDICT_STATUS = {
    Status.HOLDS: "proved",
    Status.REFUTED: "refuted",
    Status.UNKNOWN: "unknown",
}


# ══════════════════════════════════════════════════════════════════════
# The obligations Certificate.lean states
# ══════════════════════════════════════════════════════════════════════


def _ask(q: "ModuleQueries", name, build, budget, artifacts=None) -> Verdict:
    """UNSAT on the negation means the obligation holds.

    `build` is a thunk returning `(negation, witness)` rather than the
    formula itself, because encoding the transition is where an
    unsupported operator surfaces -- `Uninterpreted` has no SMT form and
    raises from `update_state`, well before any solver is involved. The
    contract is that nothing in here escapes to the caller.
    """
    if budget.exhausted:
        return Verdict(name, Status.UNKNOWN, 0.0, "phase budget spent")
    t0 = time.perf_counter()
    written = None

    def record(verdict):
        return _record(artifacts, written, verdict)

    try:
        negation, witness = build()
        written = _dump(artifacts, name, negation)
        solver = solver_for(q, budget)
        if witness:
            solver.setOption("produce-models", "true")
        solver.assertFormula(negation)
        res = solver.checkSat()
    except Exception as e:  # cvc5 raises on unsupported terms and options
        dt = (time.perf_counter() - t0) * 1000
        return record(
            Verdict(name, Status.UNKNOWN, dt, f"cannot encode: {e}"[:120])
        )
    dt = (time.perf_counter() - t0) * 1000
    if res.isUnsat():
        return record(Verdict(name, Status.HOLDS, dt))
    if not res.isSat():
        return record(
            Verdict(name, Status.UNKNOWN, dt, res.getUnknownExplanation())
        )
    return record(Verdict(name, Status.REFUTED, dt, _witness(solver, witness)))


def _witness(solver, witness) -> str | None:
    if not witness:
        return None
    try:
        bits = [f"{label} = {solver.getValue(t)}" for label, t in witness]
    except Exception:
        return "counterexample: <unreadable>"
    return "counterexample: " + ", ".join(bits) if bits else None


def _labelled(s):
    return [(f"s{i}", v) for i, v in enumerate(s)]


def check_init_inv(q: "ModuleQueries", budget: SmtBudget,
                   artifacts=None) -> Verdict:
    """`∀ e, init_pre e → inv (init e)`."""

    def build():
        f = ob.frame(q, "i_")
        return ob.init_inv(f, [q.inv]).negation(q.tm), _labelled(f.entry)

    return _ask(q, "init_inv", build, budget, artifacts)


def check_step_inv(q: "ModuleQueries", budget: SmtBudget,
                   artifacts=None) -> Verdict:
    """`∀ s e, update_pre e ∧ inv s → inv (update s e)`."""

    def build():
        f = ob.frame(q, "p_")
        o = ob.step_inv(f, [("inv", q.inv)], [q.inv])
        return o.negation(q.tm), _labelled(f.s)

    return _ask(q, "step_inv", build, budget, artifacts)


def check_hrank(q: "ModuleQueries", budget: SmtBudget,
                artifacts=None) -> Verdict:
    """`inv s ∧ ¬P s ∧ update_pre e → ranking (update s e) < ranking s`."""

    def build():
        f = ob.frame(q, "r_")
        o = ob.hrank(f, [("inv", q.inv)], q.prp, q.ranking)
        return o.negation(q.tm), _labelled(f.s)

    return _ask(q, "hrank", build, budget, artifacts)


def check_inv_imp_P(q: "ModuleQueries", budget: SmtBudget,
                    artifacts=None) -> Verdict:
    """`inv s → P s` -- the safety route's third obligation.

    `rule_globally` needs the invariant to be a strengthening of the
    property; that is the whole difference from the Büchi route, where
    `P` is merely re-reached and a ranking says how soon.
    """

    def build():
        f = ob.frame(q, "p_")
        o = ob.inv_imp_P(f, [("inv", q.inv)], q.prp)
        return o.negation(q.tm), _labelled(f.s)

    return _ask(q, "inv_imp_P", build, budget, artifacts)


def check_obligations(q: "ModuleQueries", budget: SmtBudget,
                      artifacts=None) -> list[Verdict]:
    """Every obligation the certificate data actually states, each
    question and its answer left in `artifacts` when given.

    Which third obligation there is follows the certificate's kind:
    `hrank` for `G (F P)`, `inv_imp_P` for `G P`.
    """
    budget.start_phase()
    out: list[Verdict] = []
    if q.inv is None:
        return out
    out.append(check_init_inv(q, budget, artifacts))
    out.append(check_step_inv(q, budget, artifacts))
    if q.prp is None:
        return out
    if q.kind == "safety":
        out.append(check_inv_imp_P(q, budget, artifacts))
    elif q.ranking is not None:
        out.append(check_hrank(q, budget, artifacts))
    return out


# ══════════════════════════════════════════════════════════════════════
# What is left in `artifacts/`
# ══════════════════════════════════════════════════════════════════════


def dump_system(q: "ModuleQueries", artifacts) -> None:
    """Leave the module itself in `artifacts/`, as cvc5 encoded it.

    One `define-fun` per state component per block: `init<i>` over the
    inputs awaited at tick 0, `next<i>` over the latched state and both
    ends of the inputs. Together they are the transition every obligation
    above is stated against, which is the thing a reader most often wants
    to check by eye when an answer surprises them."""
    if artifacts is None:
        return
    enc = q.enc
    try:
        s, el, en = enc.fresh()
        ins = [(str(t), str(t.getSort())) for t in en]
        step = [(str(t), str(t.getSort())) for t in s + el + en]
        state = [(str(t), str(t.getSort())) for t in q.state_vars]
        defines = [
            (f"init{i}", ins, str(t.getSort()), t)
            for i, t in enumerate(enc.init_state(en))
        ] + [
            (f"next{i}", step, str(t.getSort()), t)
            for i, t in enumerate(enc.next_state(s, el, en))
        ] + [
            # The certificate's own predicates, encoded over the same
            # state. The `.smt` files beside this one are their source;
            # these are the form the obligations are actually stated in,
            # and the form a reader can apply to a state by hand.
            (name, state, str(term.getSort()), term)
            for name, term in (("P", q.prp), ("inv", q.inv),
                               ("ranking", q.ranking))
            if term is not None
        ]
        text = script(defines=defines)
    except Exception as e:                       # noqa: BLE001
        # The transition raises on an op with no SMT form -- an
        # uninterpreted symbol, say. The run carries on; the note says why
        # the file a reader expects is not here.
        text = f"; this module has no SMT encoding: {e}\n"
    artifacts.encoded(
        "system", "system", text,
        what="The whole problem as cvc5 encoded it. `init<i>` is state "
             "component `i` after tick 0 and `next<i>` the same component "
             "after one round, over the latched state `s*` and the inputs "
             "`el*` / `en*` latched and awaited; `P`, `inv` and `ranking` "
             "are the certificate's predicates over that state. No "
             "assertion -- this file is what the obligations beside it "
             "are stated against, and what to apply to a state by hand.",
    )


def _record(artifacts, written, verdict: Verdict) -> Verdict:
    """Write the answer back onto the obligation `_dump` already wrote.

    The question is left in `artifacts/` before the solver runs, so that
    one that cannot be solved still says what was asked; this is the other
    half, and without it `index.json` records only which obligations a run
    asked about and never which of them came back refuted.

    Returned rather than assigned, so the call reads as a pass-through on
    each of `_ask`'s return paths. There are four, and a verdict recorded
    on three of them would be worse than one recorded on none: a stale
    `encoded` beside three answered obligations reads as "not asked".
    """
    if artifacts is None or written is None:
        # No artifact for this obligation: `--artifacts ignore`, or the
        # encoding failed before `_dump` ran. Either way there is no
        # question on disk for this to be the answer to.
        return verdict
    why = verdict.detail or ""
    if verdict.status is Status.HOLDS:
        # Nothing was wrong with it. `_describe` prints no status line for
        # `proved` and no prose for an empty `why`, so an obligation that
        # holds says nothing beyond what it is -- which is right.
        why = ""
    elif verdict.status is Status.REFUTED and not why:
        why = "refuted, but the counterexample could not be read back"
    try:
        artifacts.resolve(
            written, status=_VERDICT_STATUS[verdict.status], why=why
        )
    except Exception:      # pragma: no cover -- never worth the run
        pass
    return verdict


def _dump(artifacts, name: str, negation):
    """Leave this obligation in `artifacts/`, as the script cvc5 is given.

    Inside `_ask`'s `try`, so its own failure would otherwise read as the
    obligation being unencodable: a record of the question is worth having
    and worth nothing at the price of changing the answer. What came back
    is written onto it afterwards, by `_record`."""
    if artifacts is None:
        return None
    try:
        text = script([negation])
    except Exception as e:                       # noqa: BLE001
        text = f"; this obligation could not be written out: {e}\n"
    return artifacts.encoded(
        "obligation", f"obligation-{name}", text,
        what=_OBLIGATIONS.get(name, f"The `{name}` obligation, negated."),
    )


_OBLIGATIONS = {
    "init_inv": "`init_pre e -> inv (init e)`: the invariant holds in the state "
                "the module starts in.",
    "step_inv": "`inv s /\\ update_pre e -> inv (update s e)`: every round "
                "preserves the invariant.",
    "hrank": "`inv s /\\ ~P s /\\ update_pre e -> ranking (update s e) < "
             "ranking s`: the ranking function drops on every round the "
             "property does not hold.",
    "inv_imp_P": "`inv s -> P s`: what makes the invariant a proof of the "
                 "safety property rather than merely a fact about the module.",
}


# ══════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════


_NOT_SMT = ("   the certificate predicates are not SMT-LIB, so cvc5 cannot "
            "state the obligations -- skipping the pre-check")


def pre_check(smt, claim: Claim, answer: "Certificate | LeanAnswer | None",
              budget: SmtBudget, log=print, artifacts=None) -> list[Verdict]:
    """Run the obligation pre-check and report it in a few lines.

    `smt` is the module, its encoding or a `common.encoding.lazy` factory for one.
    Never raises: a module cvc5 cannot encode, or an answer it cannot read,
    reports one line and the run carries on exactly as it would have.

    `artifacts` is where the module and each obligation are written out as
    runnable SMT-LIB. Nothing about the check changes when it is `None`.
    """
    log(
        f".. SMT pre-check (cvc5): <={budget.per_call_ms} ms per query, "
        f"<={budget.phase_ms} ms total"
    )
    if isinstance(answer, LeanAnswer):
        log(_NOT_SMT)
        return []
    try:
        q = encoding_of(smt).queries(claim, answer)
    except Refused:
        log(_NOT_SMT)
        return []
    except Exception:
        log("   cvc5 could not encode this module -- skipping the pre-check")
        return []
    dump_system(q, artifacts)
    verdicts = check_obligations(q, budget, artifacts)
    if not verdicts:
        log("   nothing to check (no invariant given)")
        return []
    for v in verdicts:
        log(v.line)
    refuted = [v.name for v in verdicts if v.status is Status.REFUTED]
    if refuted:
        log(
            f"   {len(refuted)} obligation(s) refuted ({', '.join(refuted)}): "
            "the certificate is wrong, not merely hard -- Lean cannot close it"
        )
    return verdicts
