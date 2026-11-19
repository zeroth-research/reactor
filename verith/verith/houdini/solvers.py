"""`--houdini-solver`: which solver judges the search's candidates, resolved
at parse time, and Vampire as one of them.

What every solver shares, and cvc5, are `common.solvers`; this is Vampire
answering a question (`VampireSolver`), the choice between the two, and the
flags that belong to the one chosen.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from reactor.lean.core.errors import Refused
from ..common.smtlib import script
from ..common.smtlib import decimals
from ..common.obligations import Question
from ..common.solvers import DEFAULT_TIMEOUT, Answer, Cvc5Solver, Solver, Verdict
from ..common.vampire import resolve_vampire, run_vampire

SOLVERS = ("cvc5", "vampire")
DEFAULT_SOLVER = "cvc5"
DEFAULT_CORES = 4


@dataclass(frozen=True)
class SolverSpec:
    """What `--houdini-solver` resolved to, at parse time.

    A route's `resolve` runs before any project is generated, which is where
    a missing Vampire binary has to be met. cvc5 resolves to nothing to
    look for: it is the package that encoded the module in the first place.
    """

    kind: str = DEFAULT_SOLVER
    exe: "str | None" = None
    cores: int = DEFAULT_CORES

    def build(self, tm, *, seconds: float, log=print) -> Solver:
        if self.kind == "vampire":
            return VampireSolver(self.exe, seconds=seconds,
                                 cores=self.cores, log=log)
        return Cvc5Solver(tm, seconds=seconds, log=log)


def resolve_solver(kind: str, vampire: "str | None" = None,
                   cores: "int | None" = None) -> SolverSpec:
    """`--houdini-solver` and the flags that belong to the one it names."""
    if kind not in SOLVERS:
        raise Refused(
            f"--houdini-solver takes {' or '.join(SOLVERS)}, not {kind!r}"
        )
    if kind != "vampire":
        return SolverSpec(kind)
    return SolverSpec(
        kind,
        resolve_vampire(vampire, who="--houdini-solver vampire",
                        instead="`--houdini-solver cvc5` needs no binary."),
        DEFAULT_CORES if cores is None else int(cores))


# ══════════════════════════════════════════════════════════════════════════
# Vampire: a printed script, and one-sided answers
# ══════════════════════════════════════════════════════════════════════════


class VampireSolver(Solver):
    """Vampire on a printed script, in its own process session.

    Everything here is downstream of the one-sidedness. A candidate that is
    false costs a whole time limit, because nothing comes back early to say
    so; `sat` is never reported and a model never exists, so a query's
    probes are ignored and the engine falls back to asking fact by fact.
    """

    name = "vampire"
    refutes = False
    inconclusive = (
        "Vampire cannot refute a candidate, so this is not a proof that "
        "none exists: a fact or a shape outside the candidates -- a "
        "disjunction, a relation of three components, a constant the "
        "program never mentions -- or a longer time limit may be what is "
        "missing. `--houdini-solver cvc5` decides these obligations, and "
        "would say which."
    )

    LADDER = (2, 10, 60)

    def __init__(self, exe: str, *, seconds: float = DEFAULT_TIMEOUT,
                 cores: int = DEFAULT_CORES, log=print):
        super().__init__(seconds=seconds, log=log)
        self.exe = exe
        self.cores = max(1, int(cores))
        # Portfolio calls side by side. Each spreads its strategies over
        # `cores` processes; more calls than the machine has cores for would
        # only make every limit mean less time.
        self.jobs = max(1, min(4, (os.cpu_count() or 4) // self.cores))

    def _run(self, q: Question, budget: float, core: bool, model: bool) -> Answer:
        run = run_vampire(self.exe, self._script(q), [
            "--mode", "portfolio", "--schedule", "smtcomp",
            "--cores", str(self.cores),
            "--output_mode", "ucore" if core else "smtcomp",
            # Deciseconds: a limit below a second is a real limit.
            "--time_limit", f"{max(1, int(budget * 10))}d",
        ], timeout=budget + 10)
        dt = run.secs
        self.calls += 1
        self.spent += dt
        lines = run.lines
        if "unsat" in lines:
            names = None
            if core:
                at = lines.index("unsat") + 1
                reported = frozenset(
                    ln for ln in lines[at:] if ln and ln not in "()"
                    and not ln.startswith("%")
                )
                names = frozenset(n for n, _t in q.hyps if n in reported)
            return Answer(Verdict.PROVED, dt, names)
        error = run.error
        if error is not None:
            detail = run.out[run.out.find(error):].strip().splitlines()
            raise Refused(
                f"Vampire cannot read this module's encoding: "
                f"{' '.join(detail[:2])[:300]}. `--houdini-solver cvc5` reads "
                f"what this module writes, because it is what wrote it."
            )
        # Not `REFUTED`: out of time and false are the same output here.
        return Answer(Verdict.UNKNOWN, dt)

    @staticmethod
    def _script(q: Question) -> str:
        """The query as a script whose `unsat` is a proof.

        The goal is asserted negated. Hypotheses are named so that an unsat
        core can say which of them the proof used; the goal is named too, so
        a core is never empty for a reason other than the goal being false.
        """
        defines = [(str(v), (), v.getSort(), body) for v, body in q.defines]
        asserts = [f"(! {t} :named {n})" for n, t in q.hyps]
        asserts.append(f"(! (not {q.goal}) :named goal)")
        mentions = ([body for _v, body in q.defines]
                    + [t for _n, t in q.hyps] + [q.goal])
        return decimals(script(asserts, defines=defines, mentions=mentions,
                               spaced=False))
