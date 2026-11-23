"""Shape-directed tactic generation for the certificate proofs.

The three obligations (`init_inv`, `step_inv`, and `hrank` -- or `inv_imp_P`
on the `--safety` route) used to be closed by one fixed tactic chain,
identical for every module. That is wasteful in both
directions. An integer module with a linear ranking wants `omega` and nothing
else, yet paid for `decide` and `bv_decide` on every goal — and `bv_decide`,
which can only ever help a BitVec module, took one BitVec certificate from
10s to 977s without closing anything. And a shape the
chain did not anticipate had no way to ask for the step it needed: a Real
state whose invariant pins exact values needs its equalities *substituted*
before any closer runs, which is a prep step, not one more alternative to try
at the end.

So the plan is read off the obligations instead:

* which theories the state actually uses — `omega` is emitted only where
  integers are in play, `bv_omega` only where a bitvector is, `decide` only
  for a finite state, and `bv_decide` not at all (see the note beside
  `decide` below);
* whether the predicates are nonlinear — `nlinarith` is expensive and useless
  on a linear goal;
* whether they branch, conjoin or disjoin — `split_ifs` and `casesm*` are
  skipped when there is nothing to split;
* whether a Real state carries equalities — then `simp_all` runs in *prep*, so
  every closer afterwards sees numerals rather than an opaque `⌊4 - x⌋`;
* how big the module is — `maxRecDepth` scales with the term count. The
  heartbeat budget does *not*: it is 2000000, what every other generated file
  carries, and climbs only for a net with more than 32 branch points or a
  transition with more than 12. A lower floor never bounded a failure,
  because a tactic that hits the cap throws, `first` catches that like any
  other failure and moves to a more expensive alternative — so it capped
  the cheap attempts and let the dear ones run anyway. Measured,
  `NN2RealWide4` fails in 106 s at 400k and succeeds in 71 s at 2M, and
  `svcomp_collatz_bounded` fails in 29 s and builds in 66 s;
* whether the state is finite *and* narrow — then each element is enumerated
  over its two values before anything tries to close, which is the only way
  to discharge a branch that is contradictory purely because the state has
  finitely many inhabitants.

`plan_for` returns a `TacticPlan`; `scripts.lean.j2` writes it out as the
proof of each obligation in `Certificate/Proofs.lean`. Each step is Lean text
over the state `s`, which every one of those proofs binds; a step the shape
does not call for is empty, and the proof leaves it out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from reactor import Bool, Int, Real, Word

from reactor_lean.lowering.sorts import itype_name, wire_shape
from reactor_lean.lowering.elements import flat_layout


# The lemmas `simp_mat` reduces matrix arithmetic with, after the
# certificate's own definitions. The bridge's cascade uses the same set:
# without it a `Linear` arrives as `matVecAffine 2 [[1, 0], [0, 1]] [0, 1]
# (Matrix.mk fun i j => match i, j with ...)` and no arithmetic prover can
# touch it.
MAT_SIMP: tuple[str, ...] = (
    "zero_Matrix_apply", "mul_Matrix_apply", "add_Matrix_apply",
    "↓Matrix.mk_apply", "Bool.or_eq_true",
    "decide_eq_true_eq",
    "Fin.sum_univ_succ", "Fin.sum_univ_zero", "Fin.isValue", "Fin.sum_univ_one",
    "Fin.sum_univ_two", "Fin.sum_univ_three", "matVecAffine", "dotL",
    "List.ofFn_succ", "List.ofFn_zero",
    # The three reductions unfold to a `List.ofFn` fold that the two lemmas
    # above then compute; without the defs here the fold never opens.
    "matMin", "matMax", "argmax_1d", "argmax",
)

# Mathlib's lemmas for reading an entry of a matrix operation, which
# `simp_defs`' `dsimp` leaves out. They are proved by `rfl`, so `dsimp`
# rewrites with them in place: inside a `decide` the proposition changes and
# its `Decidable` instance keeps stating the old terms, and the `if` that
# `decide` guards can no longer be split (`Relational.Mat` says why).
# `MAT_SIMP`'s own entry lemmas are not `rfl` ones.
DSIMP_ERASE: tuple[str, ...] = (
    "Matrix.add_apply", "Matrix.sub_apply", "Matrix.neg_apply",
    "Matrix.zero_apply", "Matrix.transpose_apply",
)


# ══════════════════════════════════════════════════════════════════════
# Feature detection
# ══════════════════════════════════════════════════════════════════════


@dataclass
class Features:
    """What the module and its certificate predicates actually contain."""

    theories: set[str] = field(default_factory=set)
    ops: set[str] = field(default_factory=set)
    n_terms: int = 0
    n_slots: int = 0

    # (accessor, lemma) per state element that ranges over exactly two values,
    # empty if any element does not. See `_two_valued_slots`.
    two_valued: list[tuple[str, str]] = field(default_factory=list)

    # The invariant lists the states it allows (`facts._pins`).
    pins: bool = False

    # `min`/`max`/`ite` occurrences in the predicates. Each is a case: omega
    # splits `min`/`max` internally, `split_ifs` fans an `ite` out, and
    # `hrank` mentions the ranking twice. This, not the module's size, is
    # what makes a neural certificate expensive.
    n_branch: int = 0

    # Distinct branch *conditions*. `split_ifs` splits once per condition
    # and reuses the hypothesis for repeats, so this, not `n_branch`, is
    # what its fan-out costs. Falls back to `n_branch` without cvc5.
    n_conditions: int = 0

    # Branch points in the module's own transition -- `Ite`, and the ReLU,
    # `Min` and `Max` that reduce to one in the goal. `n_branch` counts the
    # certificate's; this is what `split_ifs` fans `step_inv` out over
    # before a single closer runs, whatever the certificate looks like.
    n_trans_branch: int = 0

    has_ite: bool = False
    has_and: bool = False
    has_or: bool = False
    has_eq: bool = False
    has_floor: bool = False
    # A ceiling, `⌈x⌉` or `⌈x⌉₊`, or a floor into `ℕ`, `⌊x⌋₊`. Only a Lean
    # answer writes one: SMT-LIB's `to_int` is the floor, printed `⌊x⌋`.
    has_ceil: bool = False
    nonlinear: bool = False
    # The distinct `⌊·⌋` arguments the predicates take a floor of, as Lean
    # text over the state binder. A floored rank is the only thing that puts
    # one here, and `cert_floors` needs the argument: `linarith` reads
    # `⌊x⌋` as an opaque `Int` atom and cannot cross to a hypothesis about
    # the `Real` `x`, so the bridge has to be stated per argument.
    floor_args: tuple[str, ...] = ()

    @property
    def has_int(self) -> bool:
        # Bool predicates decide into `0`/`1` arithmetic, and every ranking is
        # a `Nat`, so omega is in play whenever nothing forces it out.
        return bool({"Int", "Bool"} & self.theories) or not self.theories

    @property
    def has_real(self) -> bool:
        return "Real" in self.theories

    @property
    def has_bitvec(self) -> bool:
        return "BitVec" in self.theories

    @property
    def finite_state(self) -> bool:
        """A state `decide` could enumerate: Bool or BitVec, no Int or Real."""
        return bool(self.theories) and not ({"Int", "Real"} & self.theories)


_SORT_NAMES = ((Bool, "Bool"), (Int, "Int"), (Real, "Real"), (Word, "BitVec"))


def _sort_name(dtype) -> str:
    """The theory a sort's elements are in, a tag (`BitVec` for a `Word`,
    without its width), not a Lean type: `lowering.sorts.lean_elem_type` is that."""
    for cls, name in _SORT_NAMES:
        if isinstance(dtype, cls):
            return name
    raise ValueError(f"Unsupported Sort: {dtype}")


# A `*` whose two operands both mention the state parameter. An operand is a
# parenthesised group or an application chain such as `s.1 0 0`, so scanning
# has to stop at an operator or an unmatched paren, not at the first space.
_VAR = re.compile(r"(?:^|[^A-Za-z0-9_])(?:s|e|el)\d*(?:[.\[ )]|$)")
_BOUNDARY = set("+-*/%<>=\u2264\u2265\u2260\u2227\u2228\u00ac,;")


def _skip_group_left(text: str, k: int) -> int:
    depth = 0
    while k >= 0:
        if text[k] == ")":
            depth += 1
        elif text[k] == "(":
            depth -= 1
            if depth == 0:
                return k
        k -= 1
    return 0


def _skip_group_right(text: str, k: int) -> int:
    depth = 0
    while k < len(text):
        if text[k] == "(":
            depth += 1
        elif text[k] == ")":
            depth -= 1
            if depth == 0:
                return k
        k += 1
    return len(text) - 1


def _operand_left(text: str, i: int) -> str:
    """The operand ending just before position `i`."""
    j = i - 1
    while j >= 0 and text[j] == " ":
        j -= 1
    if j < 0:
        return ""
    if text[j] == ")":
        return text[_skip_group_left(text, j) : j + 1]
    k = j
    while k >= 0 and text[k] not in _BOUNDARY and text[k] != "(":
        if text[k] == ")":
            k = _skip_group_left(text, k)
        k -= 1
    return text[k + 1 : j + 1]


def _operand_right(text: str, i: int) -> str:
    """The operand starting just after position `i`."""
    j = i + 1
    while j < len(text) and text[j] == " ":
        j += 1
    if j >= len(text):
        return ""
    if text[j] == "(":
        return text[j : _skip_group_right(text, j) + 1]
    k = j
    while k < len(text) and text[k] not in _BOUNDARY and text[k] != ")":
        if text[k] == "(":
            k = _skip_group_right(text, k)
        k += 1
    return text[j:k]


def _mentions_var(operand: str) -> bool:
    return bool(_VAR.search(operand))


def is_nonlinear(text: str) -> bool:
    """True if any `*` in `text` multiplies two state-dependent operands.

    Multiplication by a literal weight — every affine layer of a net — stays
    linear and must not drag `nlinarith` in. Over-reporting is the safe
    direction: an extra `nlinarith` is only ever tried after the cheaper
    closers have failed, whereas a missed one loses the proof.
    """
    for m in re.finditer(r"\*", text):
        i = m.start()
        if text[i : i + 2] == "**":
            continue
        if _mentions_var(_operand_left(text, i)) and _mentions_var(
            _operand_right(text, i)
        ):
            return True
    return False


# A finite state is only worth enumerating if the fan-out stays small: each
# slot doubles the number of branches every later tactic has to run on.
MAX_ENUMERABLE_SLOTS = 4


def _two_valued_slots(ctx) -> list[tuple[str, str]]:
    """One `(accessor, lemma)` per state element that has exactly two values.

    Returns `[]` unless *every* element does, since enumerating some of a
    state and not the rest buys nothing. `accessor` is relative to the state
    variable, e.g. `.2.1 0 0`, so the caller can prefix whichever binder the
    obligation uses.
    """
    layout = flat_layout(list(ctx.ctrl_next))
    lemmas: list[str] = []
    for w in layout.wires:
        dt = w.dtype
        if isinstance(dt, Bool):
            lemmas.append("Bool.eq_false_or_eq_true")
        elif isinstance(dt, Word) and dt.width == 1:
            lemmas.append("BitVec.eq_zero_or_eq_one")
        else:
            # Wider BitVec has 2^w values and no two-way lemma; Int and Real
            # are not finite at all.
            return []
    if layout.total > MAX_ENUMERABLE_SLOTS:
        return []
    return [(layout.wire_accessor("", s), lemmas[s.wire]) for s in layout.slots]


_BRANCHING_OPS = frozenset({"Ite", "ReLU", "Min", "Max"})


def features_for(ctx, pred_text: str, facts=None, verbatim: str = "",
                 floors: tuple = ()) -> Features:
    """Read the shape of `ctx`'s module and its certificate predicates.

    `facts` is `facts.PredicateFacts` when the predicates came from
    SMT-LIB and cvc5 could parse them: the same questions answered against
    the terms rather than against the printed Lean. It is `None` for
    predicates compiled from Python IR, and then everything below falls back
    to reading the text, exactly as it did before.
    """
    f = Features()

    for w in (*ctx.ctrl_next, *ctx.extl_next):
        f.theories.add(_sort_name(w.dtype))
        f.n_slots += wire_shape(w).total

    terms = list(ctx.atom.init) + list(ctx.atom.update)
    f.n_terms = len(terms)
    for t in terms:
        f.ops.add(itype_name(t.itype))
        if itype_name(t.itype) in _BRANCHING_OPS:
            f.n_trans_branch += 1
        for w in (*t.read, *t.write):
            f.theories.add(_sort_name(w.dtype))

    # The predicates' shape comes from the facts cvc5 read off their terms;
    # text is read only where no term stands behind it -- a Lean answer
    # (`verbatim`), or every predicate when there are no facts at all.
    scanned = verbatim if facts is not None else pred_text
    # Floors come printed from the terms (`floors`), over `s` as the text is.
    f.floor_args = tuple(dict.fromkeys(
        (*(floors if facts is not None else ()),
         *floor_arguments(scanned))
    ))
    f.has_floor = bool(f.floor_args)
    f.has_ceil = ("⌈" in scanned or "⌋₊" in scanned
                  or bool(re.search(r"\b(?:Nat|Int)\.(?:ceil|floor)\b", scanned)))
    # The transition branches if it has an `Ite`, and a ReLU/Min/Max reduces
    # to a branch in the goal even though the term itself is not an `Ite`.
    f.has_ite = bool(_BRANCHING_OPS & f.ops) or "if " in scanned
    f.has_and = " ∧ " in scanned
    f.has_or = " ∨ " in scanned
    f.has_eq = " = " in scanned
    f.nonlinear = is_nonlinear(scanned)
    text_branch = (
        scanned.count("(max ") + scanned.count("(min ") + scanned.count("if ")
    )
    f.n_branch = text_branch
    f.n_conditions = text_branch
    if facts is not None:
        f.nonlinear = f.nonlinear or facts.nonlinear
        f.has_and = f.has_and or facts.has_and
        f.has_or = f.has_or or facts.has_or
        f.has_eq = f.has_eq or facts.has_eq
        f.has_ite = f.has_ite or facts.has_ite
        f.n_branch = text_branch + facts.n_branch
        f.n_conditions = text_branch + facts.n_conditions
        f.pins = facts.pins
    if f.finite_state:
        f.two_valued = _two_valued_slots(ctx)
    return f


# ══════════════════════════════════════════════════════════════════════
# Plan
# ══════════════════════════════════════════════════════════════════════


@dataclass
class TacticPlan:
    prep: list[str]
    closers: list[str]
    max_rec_depth: int
    max_heartbeats: int
    features: Features
    # `facts.SolverHints` when `--smt-tactics=cvc5` ran and cvc5 had
    # something to add; `None` otherwise, and every property below degrades
    # to the tactic the plan would have emitted without it.
    hints: object | None = None

    @property
    def prep_tactic(self) -> str:
        """The prep steps as one `<;>`-chained tactic, each step optional."""
        if not self.prep:
            return "skip"
        return " <;> ".join(f"(try {p})" for p in self.prep)

    @property
    def close_failure(self) -> str:
        """What the last arm after the closers fails with: that none of them
        proved the goal. `first` reports the error of the arm it tried last,
        so without that arm a goal no closer proved read as that closer's
        complaint -- "`split_ifs` failed: no if-then-else conditions to
        split" -- about a tactic that was never the problem."""
        return f"none of the {len(self.closers)} closers proved this goal"

    @property
    def state_case_tactic(self) -> str:
        """Enumerate the state `s`.

        `decide` is in the closer list for a finite state but can never fire
        while the state is a free variable -- it evaluates closed
        propositions, and every goal here is quantified over `s`. Worse, a
        branch whose hypotheses are contradictory *only because the state is
        finite* (`x ≠ 0#1` together with `x ≠ 1#1`) reduces to a bare `False`
        that nothing in the chain can discharge: `contradiction` needs a
        literal `h`/`¬h` pair, and omega and simp know nothing about the
        cardinality of `BitVec 1`.

        Splitting each element into its two values first makes every later
        goal closed, which is what the rest of the plan was already built
        for.

        An invariant that lists the states it allows makes an `Int` state
        as finite: split on it and `simp_all` puts each listed value in for
        the state, and every `if` the transition and the rank branch on is
        decided. Split by `cert_prep` instead -- after `split_ifs` has fanned
        the conditions out over a state it does not know -- the closers ran
        over every branch, and `svcomp_collatz_bounded`'s `hrank` (a rank
        that compares the step's own `if` with four values) spent its
        2000000 heartbeats doing it. All or nothing: a case `simp_all` does
        not close leaves the goal as it was, for the plan to take.

        Empty otherwise.
        """
        if not self.features.two_valued:
            if self.features.pins:
                return "((casesm* _ ∨ _) <;> simp_all) <;> done"
            return ""
        steps = [
            f"rcases {lemma} (s{acc}) with h{k} | h{k}"
            for k, (acc, lemma) in enumerate(self.features.two_valued)
        ]
        names = ", ".join(f"h{k}" for k in range(len(self.features.two_valued)))
        steps.append(f"try simp only [{names}] at *")
        return " <;>\n     ".join(steps)

    @property
    def floor_tactic(self) -> str:
        """Bridge each `⌊x⌋` to a bound on `x`.

        `linarith` reads `⌊x⌋` as an opaque `Int` atom, and the hypotheses
        that would pin it are about the `Real` `x`, so it cannot cross:
        `0 < s 0 0 ⊢ 0 < 2 + ⌊s 0 0⌋` fails with the bracketing lemmas
        passed in, because half the facts are in the wrong sort. Stating the
        bridge in `Int` is what closes it -- and then `omega` finishes,
        since the floor is an atom it now has a bound for.

        The sign is *split* rather than assumed, so the step cannot fail:
        a `have` whose side condition needs proving would be discharged by a
        nested `by`, and a nested `by` that fails logs `unsolved goals` at
        its own position instead of raising, which `try` does not catch --
        the failure lands in the build rather than falling through to the
        next step. `rcases` has no side condition to get wrong.

        Empty when nothing is floored, which is every integer module.
        """
        if not self.features.floor_args:
            return ""
        steps = []
        for k, a in enumerate(self.features.floor_args):
            steps.append(
                f"by_cases hs{k} : (0:Real) ≤ ({a}) <;> "
                f"[(have hfl{k} : (0:Int) ≤ ⌊{a}⌋ := Int.floor_nonneg.mpr hs{k}); "
                f"skip]"
            )
        return " <;> ".join(steps)

    @property
    def facts_tactic(self) -> str:
        """Discharge the branch conditions the invariant already settles.

        `split_ifs` fans a goal out into 2^k over k conditions and each
        branch then pays for the whole prep chain. A condition cvc5 decided
        under `inv` need not be split at all: prove it once and rewrite the
        `if` away. Each condition is a `try` of its own, so a `have` the
        tactics cannot reproduce -- cvc5 is the stronger prover -- costs
        nothing and leaves the `split_ifs` route intact.

        That needs the condition proved as a goal of the proof's own:
        `have` with a `?_` leaves it second, behind the goal, hence `swap`.
        Proved under a nested `by` instead, a failure was logged rather
        than raised, so `try` never saw it and the build failed. And the
        rewrite is a line of its own: written after that `by` on one line,
        it was parsed into it, where it rewrote nothing.

        The condition is rendered by the same walker that printed the
        definition, so `ite_eq_left` matches on the nose -- but only until
        something rewrites the goal. Hence the call site: after `simp_defs`,
        which unfolds `inv` so the `have` is provable at all, and before
        `simp_mat`, which is where the arithmetic starts moving.
        """
        determined = getattr(self.hints, "determined", None)
        if not determined:
            return ""
        steps = []
        for k, (text, value) in enumerate(determined):
            prop = text if value else f"¬ ({text})"
            lemma = "ite_eq_left" if value else "ite_eq_right"
            steps.append(
                f"try\n"
                f"  have hf{k} : {prop} := ?_\n"
                f"  swap\n"
                f"  · first | omega | linarith | norm_num\n"
                f"  try simp only [{lemma} hf{k}]"
            )
        return "\n".join(steps)

    @property
    def clear_tactic(self) -> str:
        """Drop a precondition no refutation needed, before anything runs.

        `simp_all` is superlinear in the hypothesis count and `nlinarith`
        multiplies pairs of them, so an unused hypothesis is not free.

        `step_inv` and `hrank` only: both bind `update_pre`, which is what
        the cores were taken over. `init_inv`'s `hpre` is `init_pre`, a
        different predicate that nothing here has looked at.

        Not under `try`: both proofs bind `hpre` and nothing depends on it,
        so it cannot fail. It was a `try` inside a hygienic macro once,
        which could not see the proof's `hpre`, and the `try` hid that it
        cleared nothing. Empty when nothing is to be cleared.
        """
        if getattr(self.hints, "pre_unused", False):
            return "clear hpre"
        return ""

    def why(self) -> str:
        """One line per decision, emitted as a comment above the proofs."""
        f = self.features
        bits = [
            f"state: {'+'.join(sorted(f.theories)) or 'none'}"
            f", {f.n_slots} slot(s), {f.n_terms} term(s)",
            f"predicates: {'branching' if f.has_ite else 'flat'}"
            f"{', conjunctive' if f.has_and else ''}"
            f"{', disjunctive' if f.has_or else ''}"
            f"{', nonlinear' if f.nonlinear else ', linear'}"
            f"{', floor' if f.has_floor else ''}"
            f"{f', {f.n_branch} branch point(s)' if f.n_branch else ''}"
            f"{f' ({f.n_conditions} distinct)' if f.n_conditions != f.n_branch else ''}",
        ]
        if f.two_valued:
            bits.append(
                f"state is finite and narrow: enumerating "
                f"{len(f.two_valued)} two-valued element(s) before the closers"
            )
        elif f.pins:
            bits.append("the invariant lists its states: trying each before "
                        "the closers")
        out = "\n".join(f"-- {b}" for b in bits)
        if self.hints is not None and self.hints.any:
            out += "\n" + self.hints.why()
        return out


def floor_arguments(pred_text: str) -> tuple[str, ...]:
    """The distinct `x` of every `⌊x⌋` in `pred_text`, in order.

    A scan rather than a parse: what is between the brackets is Lean text
    this package printed itself, so the only thing that can nest inside is a
    balanced `⌊⌋` pair, and the depth counter handles that.
    """
    out: list[str] = []
    depth, start = 0, 0
    for i, ch in enumerate(pred_text):
        if ch == "⌊":
            if not depth:
                start = i + 1
            depth += 1
        elif ch == "⌋" and depth:
            depth -= 1
            if not depth:
                arg = pred_text[start:i].strip()
                if arg and arg not in out:
                    out.append(arg)
    return tuple(out)


def plan_for(ctx, pred_text: str, facts=None, hints=None,
             verbatim: str = "", floors: tuple = ()) -> TacticPlan:
    """Build the tactic plan for this module's proof obligations."""
    f = features_for(ctx, pred_text, facts, verbatim, floors)

    # ── prep: canonicalise the goal before anything tries to close it ──
    prep: list[str] = []
    if f.has_ite:
        prep.append("split_ifs")
    if f.has_and:
        prep.append("casesm* _ ∧ _")
    if f.has_and or f.has_or:
        prep.append("simp only [not_and_or] at *")
    if f.has_floor:
        # A scaled rank floors `k * (x - c)`, and `norm_num` knows
        # `⌊y - 1⌋ < ⌊y⌋` but not `⌊k*(x - c)⌋ < ⌊k*x⌋` -- the scale has to
        # be distributed first for the two arguments to differ by a literal.
        # `m_lra_half` ranks by `⌊2*s⌋` and steps by `1/2`; without this its
        # decrease obligation is the one goal left standing.
        prep.append("simp only [mul_sub]")
    if f.has_real:
        # Profiled, this is the single most expensive thing in the plan, and
        # over the reals it is also load-bearing: dropping it to the goal
        # alone leaves two obligations of NN2RealAllPos4 unproved, because
        # `linarith` needs the hypotheses normalised.
        #
        # For an integer state it is neither. `omega` normalises numerals
        # itself and reads `min`/`max`/`Int.toNat` natively, so `norm_num`
        # only re-traverses a goal omega is about to take apart anyway.
        # Measured on NN2Deep5, a five-layer net: 75.9 s with it, **7.0 s**
        # without, same proof, no errors -- `norm_num` was 47.3 s of the
        # 48.7 s of tactic execution while `omega`, which actually closed
        # the goal, took 0.79 s. Restricting it to the goal saves nothing
        # (77.9 s): the cost is the goal, not the context.
        #
        # `(norm_num; done)` stays in the closers either way, so a goal that
        # really does need it still gets it.
        prep.append("norm_num at *")
    if f.has_real and f.has_eq:
        # Over the reals an inductive invariant has to pin exact values, so
        # the hypotheses are equalities and the goal is a numeral once they
        # are substituted. `omega` would have done this for integers; nothing
        # does it here unless simp_all runs first.
        prep.append("simp_all")
    if f.has_eq:
        # `¬(x = c)` carries a strict bound only through integrality, which
        # linarith and nlinarith do not do. Split it so the `casesm*` below
        # hands each branch a usable inequality. Must follow `norm_num at *`
        # where that runs at all, since it renormalises `≠` back to `¬ =`.
        prep.append("simp only [← ne_eq, ne_iff_lt_or_gt] at *")
    if f.has_and or f.has_or or f.has_eq:
        prep.append("casesm* _ ∧ _, _ ∨ _")

    # ── closers: cheapest first, and only what this state can use ──
    # `linarith` stays in every plan regardless of theory. It is cheap, and
    # gating it on Real cost two integer certificates that were closing on it
    # — an integer goal that omega cannot phrase (a `Matrix`-valued comparison
    # simp left behind, say) is still ordinary linear arithmetic.
    closers: list[str] = []
    # `omega` first where it is in play, then `contradiction`. A branch whose
    # hypotheses are already contradictory is common once `split_ifs` has
    # fanned the transition out, and `omega` refutes an arithmetic one as
    # well; `contradiction` is for the rest (`b = true` beside `b = false`).
    # It is not free the other way round: it reduces every hypothesis with
    # `whnf` looking for an empty type or a decidably false proposition, and
    # an `Int` comparison over the unfolded transition reduces a long way.
    # Measured on a 6-slot Petri net's `step_inv`: `split_ifs` leaves 7680
    # goals, and with `contradiction` first they spent the 2000000
    # heartbeats; with `omega` first the certificate builds in 40 s.
    if f.has_int:
        closers.append("omega")
    closers.append("contradiction")
    if f.has_bitvec:
        # A bitvector read as the number it holds -- `BitVec.toNat`, which
        # a linear predicate weighs a BitVec component by -- is arithmetic no
        # other closer here can phrase: omega and linarith know nothing
        # about it, and `decide` cannot fire while the state is a free
        # variable. Worse than failing, `decide` *errors* there ("Expected
        # type must not contain free variables"), which is an elaboration
        # error rather than a tactic failure and so escapes the `first`
        # chain and fails the build. `bv_omega` is the cheap one for this
        # shape -- `simp` with the BitVec lemmas, then omega -- and it is
        # not `bv_decide`, which bit-blasts for minutes (see the note below).
        # Measured on an 8-bit counter's safety certificate: the build fails
        # without it and takes 3.4 s with it.
        closers.append("bv_omega")
    closers.append("linarith")
    if f.has_int:
        closers.append("(norm_cast; omega)")
    if f.has_real:
        closers.append("(norm_cast; linarith)")
    closers.append("(norm_num; done)")
    if f.nonlinear:
        closers += ["nlinarith", "positivity"]

    # Conjunctive goals: `ranking`'s `Int.toNat` comparison alone unfolds to
    # `a < b ∧ 0 < b`, so this fires far more often than the invariant shape
    # suggests. Give the branch every prover the plan already carries.
    per_conjunct = [c for c in ("omega", "linarith", "nlinarith") if c in closers]
    closers.append("(constructor <;> first | " + " | ".join(per_conjunct) + ")")

    if getattr(hints, "nlinarith_hints", None):
        # cvc5's refutation multiplied these; naming the squares up front
        # is the difference between `nlinarith` checking a certificate and
        # searching for one. It goes *before* the bare `nlinarith` and adds
        # to the list rather than replacing anything -- a `first` chain
        # costs nothing for the alternatives it never reaches, so there is
        # no case for taking a closer away on the solver's say-so.
        hint_list = ", ".join(hints.nlinarith_hints)
        idx = closers.index("nlinarith") if "nlinarith" in closers else len(closers)
        closers.insert(idx, f"nlinarith [{hint_list}]")

    closers.append("tauto")
    # Re-simplify with the hypotheses, then retry the provers this state can
    # use. `done` guards the ones that can make progress without closing,
    # which would otherwise end the chain on an open goal.
    retry = [c for c in ("omega", "linarith", "nlinarith") if c in closers]
    closers.append("(simp_all; done)")
    closers += [f"(simp_all; {c})" for c in retry]
    if f.has_real:
        closers.append("(simp_all; norm_num; done)")

    # Structure the closers above do not take apart, each behind every one of
    # them: a proof that closes today never reaches these, so they can only
    # turn a failure into a proof, at the price of failing a little later.
    # Measured on the cells of the bench matrix whose certificate cvc5
    # accepts and Lean did not, rebuilt with only the closers changed.
    if f.has_real:
        # `ZerothHammer`'s `lra_close`: implications and disjunctions taken
        # apart on both sides, and every floor bounded and the goal moved to
        # `ℝ`. A hybrid plant's invariant is mode-split -- `b = false → 5 ≤
        # x` -- and its rank floors a scaled level, which is exactly what
        # `linarith` cannot see through: `m_thermostat`, `m_tank_dist`,
        # `m_watertank` and `p_fluid` certificates all build with it and
        # none did without.
        closers.append("lra_close")
    if f.has_int and f.has_ite:
        # A Bool weighed as `if b then 1 else 0` *inside a hypothesis*:
        # `split_ifs` in prep splits the goal's branches only, and omega
        # cannot read the Bool the goal still mentions until `simp_all` has
        # substituted it. `m_boolint`'s linear invariant, for one.
        closers.append("(split_ifs at * <;> simp_all <;> omega)")
    if f.nonlinear:
        # A product with a `max` in it: omega refuses the product, and
        # `nlinarith` cannot see through the `max` until it is a case.
        # `cousot9`'s `s0 * (max s2 0 + 1) + max s1 0`.
        closers.append("(simp only [max_def, min_def] at *; split_ifs at * "
                       "<;> (repeat' apply And.intro) <;> nlinarith)")

    # ── last resorts ──
    # Each closer below was added for goals every closer above gave up on,
    # and comes after all of them: a goal one of those proves never reaches
    # these, so a proof that builds today builds the same way. A finite
    # state keeps `decide` last instead.
    if f.has_real and f.has_ceil:
        # `lra_close`, with `⌈x⌉₊` and `⌊x⌋₊` taken to `ℤ` and the ceilings
        # bounded as the floors are: `lra_close` reads `⌈x⌉₊` as an atom. A
        # Lean answer's `Nat.ceil ((12 - h) * 2 / 3)` for `m_watertank`'s
        # refills, which falls by one a tick.
        closers.append("lra_close_ceil")
    if f.has_real and f.has_floor:
        # `lra_close` with each floor split at `0`, so that `0 < ⌊x⌋` is
        # `1 ≤ ⌊x⌋`: the integrality `linarith` cannot see in `ℝ`.
        # `m_lra_half`'s certificate that `2 * x` is a whole number.
        closers.append("lra_close_frac")
    if f.nonlinear and f.has_int and not f.finite_state:
        # A product under a ranking's `Int.toNat`: `nlinarith` reads
        # `(a * b).toNat` as an atom, and omega refuses the product. Moved
        # to `ℤ` with each `toNat` a `max`, and the `max` split, the
        # products cancel and `nlinarith` finishes.
        # `LeikeHeizmann-TACAS2014-Fig1`'s `(x*2 + (1 - y)*(2 - y)).toNat`,
        # and `speedpldi3`'s `(n - i).toNat * (m + 1).toNat`.
        closers.append("(zify; simp only [Int.toNat_eq_max, max_def, min_def] at *; "
                       "split_ifs at * <;> (repeat' apply And.intro) <;> nlinarith)")
    if f.has_int and not f.finite_state:
        # `omega` does not implement the dark and grey shadows, so it gives
        # up on an integer problem whose rational relaxation is feasible:
        # `a + 2*b + c + 2*d ≤ 2` beside `a + b + c + 2*d ≥ 1` and the rest
        # of a Petri net's place invariant, where only integrality rules
        # out `d = -1/2`. `lia` (cutsat) decides linear integer arithmetic:
        # `petri/philo/forks`'s `step_inv` builds with it and did not
        # without.
        closers.append("lia")

    # `decide` goes last: on a finite state it settles goals nothing else
    # can, but it is the most expensive thing in the plan.
    #
    # `bv_decide` is deliberately *not* here. It only ever reaches goals
    # every cheaper prover has already failed on, and on those it bit-blasts
    # for minutes: including it took a BitVec certificate from 10s to 977s
    # while closing nothing `decide` had not already closed. The one
    # obligation it might have helped with — `hrank`, which mixes BitVec with
    # `Int.toNat` — it answers with "a potentially spurious counterexample".
    if f.finite_state:
        closers.append("decide")

    # ── budgets ──
    # `simp_mat` unfolds the whole transition at once, so recursion depth
    # tracks the term count rather than the state width.
    max_rec_depth = max(4096, 1500 * f.n_terms)
    # 2000000, which is what every other generated file carries
    # (`codegen/encodings/scalar.py`, `codegen/encodings/circuit.py`).
    # This file used to start at 400000 and
    # climb to 2M wherever the shape was known to be slow -- finite state, a
    # bitvector, more than 8 slots or 32 terms, a branchy Real predicate,
    # more than 16 branch points. The list kept growing because a low floor
    # is not what it looks like: a tactic that hits the cap *throws*, `first`
    # catches that like any other failure and moves to a more expensive
    # alternative, which burns up to the cap again. So the floor never cut a
    # failure short, it multiplied the wasted work, and every rung was a shape
    # someone had already been bitten by.
    #
    # Measured on the two cells that hit the old floor, one project apiece
    # with only this number changed: `svcomp_collatz_bounded` fails in 29 s
    # at 400k and *builds* in 66 s at 2M -- so the floor was the whole
    # difference, and it cost 37 s to find that out rather than saving any.
    # `ChenFlurMukhopadhyay-SAS2012-Ex1.01` still fails at 2M, but with
    # `linarith failed to find a contradiction` instead of a heartbeat
    # timeout, which is the truth about it: cvc5 refutes that certificate's
    # `hrank` outright at `s0 = 1`. A budget error was standing in front of
    # a wrong certificate. `NN2RealWide4` is the same shape of result at
    # 106 s failing against 71 s succeeding, and is why the Real rung
    # existed.
    max_heartbeats = 2000000
    # A wide or deep net puts far more branch points in one predicate than the
    # module's own size suggests, and the whole cost of a neural certificate
    # lives there. Measured on nets built from explicit weight matrices: 14
    # branch points (a 12-unit layer, or three dense hidden layers) close
    # inside this budget, 30 needs it, 62 needs 4M. Raising the cap costs
    # nothing on a proof that closes; it only makes a failing one give up
    # later, and a predicate this size was never going to fail fast anyway.
    #
    # The transition can be the branchy half instead. A Petri net with a
    # transition per input value is a nested `ite` per transition, and
    # `step_inv` pays for every case of it under one declaration's budget:
    # `mutex` (16 branch points) and `philo` (18) ran out at 2M on invariants
    # cvc5 accepts and build at 8M in 78-160 s, four cells of five. The
    # svcomp, hybrid and smaller Petri modules have 3-12 and are untouched.
    if f.n_branch > 32 or f.n_trans_branch > 12:
        max_heartbeats = 8000000

    return TacticPlan(
        prep=prep,
        closers=closers,
        max_rec_depth=max_rec_depth,
        max_heartbeats=max_heartbeats,
        features=f,
        hints=hints,
    )
