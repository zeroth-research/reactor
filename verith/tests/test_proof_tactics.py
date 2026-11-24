"""Tests for shape-directed certificate tactic generation (`verith.proofs.tactics`).

The plan decides what the three proof obligations are allowed to try. Getting
it wrong is expensive in both directions — a missing prover loses a proof that
used to go through, a superfluous one costs seconds per goal — so the
decisions are pinned here rather than only observed end to end in the slow
Lean tests.
"""

import re

import torch
import pytest

from reactor import Module, Term, Wire, Var, X, Int, Real, Bool, Word, tensor
from reactor_lean.codegen.context import LeanContext
from verith.proofs.tactics import features_for, is_nonlinear, plan_for


# ── the nonlinearity detector ────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        # A weight times a state element is affine — every layer of a net
        # looks like this, and none of them needs nlinarith.
        ("(2 * (s 0 0))", False),
        ("(2 * (s 0 0)) + (3 * (s 1 0))", False),
        ("(-1 * s0)", False),
        ("(* 2.0 (ite (>= x 0) x 0))", False),
        ("fun s => ((s 0 0) * 4)", False),
        # Two state-dependent operands are not.
        ("(s0 * s0)", True),
        ("((s 0 0) * (s 0 0))", True),
        # An application chain is one operand, not a token ending at a space.
        ("(s.1 0 0 * s.2 0 0)", True),
        ("fun s => (s 0 0 * s 0 0)", True),
    ],
)
def test_is_nonlinear(text, expected):
    assert is_nonlinear(text) is expected


# ── plans ────────────────────────────────────────────────────────────────


def _int_module() -> Module:
    """1x1 integer countdown."""
    x = Var(Int([1, 1]))
    one, hundred = Wire(Int([1, 1])), Wire(Int([1, 1]))
    zero = Wire(Int([1, 1]))
    cond = Wire(Bool([1, 1]))
    dec = Wire(Int([1, 1]))
    return Module.sequential(
        [x],
        [Term(tensor.int.Const(torch.tensor([[100]])), [X(x)])],
        [
            Term(tensor.int.Const(torch.tensor([[0]])), [zero]),
            Term(tensor.int.Const(torch.tensor([[1]])), [one]),
            Term(tensor.int.Const(torch.tensor([[100]])), [hundred]),
            Term(tensor.int.Eq([1, 1]), [cond], [x, zero]),
            Term(tensor.int.Sub([1, 1]), [dec], [x, one]),
            Term(tensor.int.Ite([1, 1]), [X(x)], [cond, hundred, dec]),
        ],
    )


def _real_module() -> Module:
    x = Var(Real([1, 1]))
    one = Wire(Real([1, 1]))
    return Module.sequential(
        [x],
        [Term(tensor.real.Const(torch.tensor([[5.0]])), [X(x)])],
        [
            Term(tensor.real.Const(torch.tensor([[1.0]])), [one]),
            Term(tensor.real.Sub([1, 1]), [X(x)], [x, one]),
        ],
    )


def _bv_module() -> Module:
    b = Var(Word(1, [1, 1]))
    nb = Wire(Word(1, [1, 1]))
    return Module.sequential(
        [b],
        [Term(tensor.word.Const(torch.tensor([[0]]), 1), [X(b)])],
        [Term(tensor.word.Not(1, [1, 1]), [nb], [b]),
         Term(tensor.Skip(Word(1, [1, 1])), [X(b)], [nb])],
    )


def _plan(module, pred_text):
    return plan_for(LeanContext(module), pred_text)


def test_integer_plan_omits_real_and_finite_provers():
    """An integer module pays for neither `decide` nor `nlinarith`.

    `decide` cannot enumerate an unbounded state and `nlinarith` has nothing
    to do on a linear goal; both used to run on every goal of every module.
    """
    plan = _plan(_int_module(), "fun s => ((s 0 0) ≥ 0 ∧ (s 0 0) ≤ 100)")
    assert "omega" in plan.closers
    assert "decide" not in plan.closers
    assert "nlinarith" not in plan.closers
    assert not any("bv_decide" in c for c in plan.closers)


def test_linarith_is_always_available():
    """Gating linarith on Real cost two integer certificates that used it."""
    for module, text in (
        (_int_module(), "fun s => ((s 0 0) = 0)"),
        (_real_module(), "fun s => ((s 0 0) = 0)"),
    ):
        assert "linarith" in _plan(module, text).closers


def test_nonlinear_predicate_pulls_in_nlinarith():
    plan = _plan(_int_module(), "fun s => (((s 0 0) * (s 0 0) : Int)).toNat")
    assert "nlinarith" in plan.closers
    # and the conjunction branch gets it too: an `Int.toNat` comparison
    # unfolds to `a < b ∧ 0 < b`.
    assert any("constructor" in c and "nlinarith" in c for c in plan.closers)


def test_real_state_substitutes_equalities_in_prep():
    """Over the reals the equalities have to go into the goal *before* the
    closers run; omega would have done it for integers."""
    real = _plan(_real_module(), "fun s => ((s 0 0) = 2)")
    assert "simp_all" in real.prep
    integer = _plan(_int_module(), "fun s => ((s 0 0) = 2)")
    assert "simp_all" not in integer.prep


def test_disequality_split_follows_norm_num():
    """`norm_num at *` renormalises `≠` back to `¬ =`, so the split that
    turns it into a usable bound has to come after it.

    Only a Real state runs `norm_num` in prep at all now, so that is where
    the ordering can be observed.
    """
    steps = _plan(_real_module(), "fun s => ((s 0 0) = 0)").prep
    assert "norm_num at *" in steps
    split = "simp only [← ne_eq, ne_iff_lt_or_gt] at *"
    assert split in steps
    assert steps.index(split) > steps.index("norm_num at *")


def test_finite_state_gets_decide_last():
    """`decide` earns its place on a BitVec state, but it is the most
    expensive thing in the plan, so nothing may follow it."""
    plan = _plan(_bv_module(), "fun s => ((s 0 0) = 1#1)")
    assert plan.closers[-1] == "decide"


def test_finite_narrow_state_is_enumerated():
    """A two-valued state element gets split before the closers run.

    `decide` is in the plan for a finite state but can never fire while the
    state is a free variable, and a branch that is contradictory only because
    `BitVec 1` has two inhabitants (`x ≠ 0#1` and `x ≠ 1#1`) reduces to a bare
    `False` that nothing else discharges.
    """
    plan = _plan(_bv_module(), "fun s => ((s 0 0) = 1#1)")
    tac = plan.state_case_tactic
    assert "BitVec.eq_zero_or_eq_one" in tac
    assert "(s 0 0)" in tac, "the enumeration must take the proof's state binder"
    assert "simp only [h0" in tac, "the two values have to reach the hypotheses"


def test_unbounded_state_is_not_enumerated():
    """Nothing to enumerate over Int or Real, so the step is a no-op."""
    assert _plan(_int_module(), "fun s => True").state_case_tactic == ""
    assert _plan(_real_module(), "fun s => True").state_case_tactic == ""


def test_an_invariant_that_lists_its_states_is_enumerated():
    """An `Int` state the invariant pins to `1, 2, 3, 4, 7` is as finite as
    a Bool one: split on it before `split_ifs` fans the step's conditions
    out over a state it does not know. `svcomp_collatz_bounded`'s `hrank`
    spent 2000000 heartbeats in the closers otherwise. All or nothing, so
    a case `simp_all` cannot close leaves the plan as it was."""
    from verith.proofs.facts import PredicateFacts

    plan = plan_for(LeanContext(_int_module()), "fun s => True",
                    PredicateFacts(pins=True))
    assert plan.state_case_tactic == "((casesm* _ ∨ _) <;> simp_all) <;> done"
    assert _plan(_int_module(), "fun s => True").state_case_tactic == ""


@pytest.mark.parametrize("inv, pins", [
    ("(or (= s0 1) (= s0 2) (= s0 7))", True),
    ("(or (and (= s0 1) (= s1 2)) (= 3 s0))", True),
    ("(or (= s0 1) (<= s0 2))", False),
    ("(or (= s0 s1) (= s0 2))", False),
    ("(and (= s0 1) (= s1 2))", False),
    ("(= s0 1)", False),
])
def test_an_invariant_lists_its_states_when_every_disjunct_is_equalities(inv, pins):
    import cvc5

    from verith.proofs.facts import _pins

    tm = cvc5.TermManager()
    solver = cvc5.Solver(tm)
    solver.setLogic("ALL")
    sm = cvc5.SymbolManager(tm)
    parser = cvc5.InputParser(solver, sm)
    parser.setStringInput(cvc5.InputLanguage.SMT_LIB_2_6,
                          "(declare-const s0 Int)(declare-const s1 Int)", "decls")
    while not (cmd := parser.nextCommand()).isNull():
        cmd.invoke(solver, sm)
    parser.setStringInput(cvc5.InputLanguage.SMT_LIB_2_6, inv, "inv")
    assert _pins(parser.nextTerm()) is pins


def test_omega_comes_before_contradiction_where_it_is_in_play():
    """`contradiction` reduces every hypothesis with `whnf`; ahead of
    `omega` on a Petri net's 7680 `step_inv` branches it spent the whole
    heartbeat budget, where `omega` closes them all."""
    closers = _plan(_int_module(), "fun s => ((s 0 0) ≥ 0)").closers
    assert closers[:2] == ["omega", "contradiction"]
    assert _plan(_real_module(), "fun s => ((s 0 0) ≥ 0)").closers[0] == "contradiction"


def test_wide_finite_state_is_not_enumerated():
    """Each element doubles the fan-out, so a wide state opts out."""
    from verith.proofs.tactics import MAX_ENUMERABLE_SLOTS

    b = Var(Word(1, [MAX_ENUMERABLE_SLOTS + 1, 1]))
    wide = Module.sequential(
        [b],
        [Term(tensor.word.Const(
            torch.zeros((MAX_ENUMERABLE_SLOTS + 1, 1), dtype=torch.int64), 1), [X(b)])],
        [Term(tensor.Skip(b.dtype), [X(b)], [b])],
    )
    assert _plan(wide, "fun s => True").state_case_tactic == ""


def test_wide_bitvec_is_not_enumerated():
    """`BitVec.eq_zero_or_eq_one` is width-1 only; wider has 2^w values."""
    b = Var(Word(8, [1, 1]))
    m = Module.sequential(
        [b],
        [Term(tensor.word.Const(torch.tensor([[0]]), 8), [X(b)])],
        [Term(tensor.Skip(b.dtype), [X(b)], [b])],
    )
    assert _plan(m, "fun s => True").state_case_tactic == ""


def test_flat_predicates_skip_the_splitting_steps():
    """Nothing to split, nothing to case on."""
    x = Var(Int([1, 1]))
    flat = Module.sequential(
        [x],
        [Term(tensor.int.Const(torch.tensor([[0]])), [X(x)])],
        [Term(tensor.Skip(Int([1, 1])), [X(x)], [x])],
    )
    plan = _plan(flat, "fun s => True")
    assert "split_ifs" not in plan.prep
    assert not any("casesm" in p for p in plan.prep)


def test_budget_scales_with_predicate_branch_points():
    """A net's cost lives in its branch points, not in the module's size.

    A 12-unit ranking net over a one-wire countdown leaves `n_slots` and
    `n_terms` tiny, so nothing about the module says what the predicate
    costs. Only a net past 32 branch points goes above the standard budget
    -- everything else is already at it.
    """
    m = _int_module()
    flat = _plan(m, "fun s => ((s 0 0) ≥ 0)")
    assert flat.max_heartbeats == 2000000

    mid = _plan(m, "fun s => " + " + ".join(f"(max ((s 0 0) - {k}) 0)" for k in range(20)))
    assert mid.max_heartbeats >= 2000000

    wide = _plan(m, "fun s => " + " + ".join(f"(max ((s 0 0) - {k}) 0)" for k in range(70)))
    assert wide.max_heartbeats > mid.max_heartbeats


def test_branch_points_count_ite_as_well_as_min_max():
    """Over the reals a ReLU stays an `ite`, and costs a `split_ifs` branch."""
    plan = _plan(
        _real_module(),
        "fun s => " + " + ".join(
            f"(if (s 0 0) ≥ {k} then (s 0 0) else 0)" for k in range(20)
        ),
    )
    assert plan.features.n_branch >= 20
    assert plan.max_heartbeats >= 2000000


def test_branchy_real_gets_a_higher_budget():
    """A ReLU over the reals cannot fold to `max`, so it keeps its branches.

    `linarith` has no min/max support, so a Real net still pays a `split_ifs`
    branch per unit. Measured, the old 400k floor made such a case *fail* in
    106 s where 2M *succeeds* in 71 s -- which is why there is no longer a
    floor below 2M for it to fall to.
    """
    branchy = _plan(_real_module(), "fun s => (if (s 0 0) ≥ 0 then (s 0 0) else 0) = 0")
    assert branchy.max_heartbeats >= 2000000
    flat = _plan(_real_module(), "fun s => ((s 0 0) ≥ 0)")
    assert flat.max_heartbeats == 2000000


def test_only_recursion_depth_scales_with_the_module():
    """`maxRecDepth` tracks the term count. The heartbeat budget does not:
    it is the same 2000000 every other generated file carries, whatever the
    module's size, because a lower floor never made a failure faster -- a
    tactic that hits the cap throws and `first` moves to a dearer one."""
    small = _plan(_int_module(), "fun s => ((s 0 0) = 0)")
    assert small.max_heartbeats == 2000000

    x = Var(Int([64, 1]))
    wide = Module.sequential(
        [x],
        [Term(tensor.int.Const(torch.zeros((64, 1), dtype=torch.int64)), [X(x)])],
        [Term(tensor.Skip(Int([64, 1])), [X(x)], [x])],
    )
    plan = _plan(wide, "fun s => True")
    # 64 slots used to be one of the shapes that bought a bigger budget.
    assert plan.max_heartbeats == small.max_heartbeats
    assert plan.max_rec_depth >= 4096


# ══════════════════════════════════════════════════════════════════════
# Features taken from the cvc5 terms rather than from the printed Lean
# ══════════════════════════════════════════════════════════════════════


def test_facts_override_the_text_scan_for_nonlinearity():
    """The regex over-reports by design; the term does not."""
    from verith.proofs.facts import PredicateFacts

    ctx = LeanContext(_int_module())
    linear = features_for(ctx, "fun s => ((s 0 0) * (s 0 0))")
    assert linear.nonlinear is True                       # text says yes
    facts = PredicateFacts(nonlinear=False)
    assert features_for(ctx, "fun s => ((s 0 0) * (s 0 0))", facts).nonlinear is False


def test_a_lean_answers_branches_add_to_the_terms():
    """Terms cover only the SMT-derived fields; a Lean answer's text is
    counted on top of them, never in place of them."""
    from verith.proofs.facts import PredicateFacts

    ctx = LeanContext(_int_module())
    text = "fun s => (max (s 0 0) 0) + (max (s 0 0) 1) + (max (s 0 0) 2)"
    f = features_for(ctx, text, PredicateFacts(n_branch=1), verbatim=text)
    assert f.n_branch == 4


def test_with_facts_the_printed_text_is_not_reread():
    """The facts speak for what was printed from terms; reading the text
    as well would count a shared subterm by how the printer laid it out."""
    from verith.proofs.facts import PredicateFacts

    ctx = LeanContext(_int_module())
    text = "fun s => ((s 0 0) * (s 0 0)) + (max (s 0 0) 0)"
    f = features_for(ctx, text, PredicateFacts(n_branch=1))
    assert (f.n_branch, f.nonlinear) == (1, False)


def test_facts_raise_the_branch_count_above_what_the_printer_shows():
    """Sharing shrinks the output; the goal the closers see is unchanged."""
    from verith.proofs.facts import PredicateFacts

    ctx = LeanContext(_int_module())
    f = features_for(ctx, "fun s => (max (s 0 0) 0)", PredicateFacts(n_branch=62))
    assert f.n_branch == 62


def test_without_facts_everything_falls_back_to_the_text():
    ctx = LeanContext(_int_module())
    f = features_for(ctx, "fun s => ((s 0 0) * (s 0 0)) + (max (s 0 0) 0)")
    assert f.nonlinear is True
    assert f.n_branch == 1
    assert f.n_conditions == 1


# ══════════════════════════════════════════════════════════════════════
# Solver-informed tactics
#
# The contract is that hints only ever *add*: with none, the plan is byte
# for byte the one that was emitted before cvc5 was asked anything.
# ══════════════════════════════════════════════════════════════════════


def _hinted(**fields):
    from verith.proofs.scripts import LeanHints

    return plan_for(
        LeanContext(_int_module()),
        "fun s => (s 0 0)",
        hints=LeanHints(**fields),
    )


def test_without_hints_the_hint_steps_are_empty():
    plan = plan_for(LeanContext(_int_module()), "fun s => (s 0 0)")
    assert plan.facts_tactic == ""
    assert plan.clear_tactic == ""


def test_a_settled_condition_is_proved_as_a_goal_and_then_rewritten():
    """Proved under a nested `by`, a condition the closers cannot prove
    fails the build instead of the `try`; and the rewrite, written after
    that `by`, was parsed into it."""
    plan = _hinted(determined=[("(s 0 0) ≥ 0", True)])
    assert plan.facts_tactic.splitlines() == [
        "try",
        "  have hf0 : (s 0 0) ≥ 0 := ?_",
        "  swap",
        "  · first | omega | linarith | norm_num",
        "  try simp only [ite_eq_left hf0]",
    ]


def test_a_refuted_condition_is_negated_and_rewrites_to_the_else_branch():
    plan = _hinted(determined=[("(s 0 0) ≥ 0", False)])
    tac = plan.facts_tactic
    assert "have hf0 : ¬ ((s 0 0) ≥ 0) := ?_" in tac
    assert "ite_eq_right hf0" in tac


def test_every_settled_condition_gets_its_own_guarded_step():
    """A `have` the tactics cannot reproduce must cost nothing."""
    plan = _hinted(determined=[("a", True), ("b", False), ("c", True)])
    lines = plan.facts_tactic.splitlines()
    assert lines.count("try") == 3
    assert [ln for ln in lines if ln.startswith("  have")] == [
        "  have hf0 : a := ?_", "  have hf1 : ¬ (b) := ?_", "  have hf2 : c := ?_"]


def test_an_unused_precondition_is_cleared():
    assert _hinted(pre_unused=True).clear_tactic == "clear hpre"


def test_the_proofs_that_bind_the_precondition_clear_it():
    """`step_inv` and `hrank` bind `update_pre`, which the cores were taken
    over; `init_inv`'s `hpre` is `init_pre`, which nothing asked about."""
    proofs = _proofs_text(_solver_hints(pre_unused=True))
    cleared = re.findall(r"^theorem (\w+)_proof(?:(?!^theorem).)*?^  clear hpre$",
                         proofs, re.M | re.S)
    assert cleared == ["step_inv", "hrank"], cleared


def test_the_settled_conditions_are_steps_of_each_proof():
    """Written into a proof at its own indentation, after `simp_defs`."""
    proofs = _proofs_text(_solver_hints(determined=[("(s 0 0) ≥ 0", True)]))
    step = ("  try simp_defs\n"
            "  try\n"
            "    have hf0 : (s 0 0) ≥ 0 := ?_\n"
            "    swap\n"
            "    · first | omega | linarith | norm_num\n"
            "    try simp only [ite_eq_left hf0]\n")
    assert proofs.count(step) == 2       # step_inv and hrank


def test_product_hints_are_added_before_the_bare_nlinarith():
    """Naming the squares turns nlinarith's search into a check."""
    plan = plan_for(
        LeanContext(_int_module()),
        "fun s => ((s 0 0) * (s 0 0))",       # nonlinear, so nlinarith is in
        hints=_solver_hints(nlinarith_hints=["mul_self_nonneg ((s 0 0) - 1)"]),
    )
    closers = plan.closers
    assert "nlinarith [mul_self_nonneg ((s 0 0) - 1)]" in closers
    assert closers.index("nlinarith [mul_self_nonneg ((s 0 0) - 1)]") < closers.index(
        "nlinarith"
    )


def test_a_product_hint_reads_the_obligations_own_state():
    """The hints are printed over `s`, so every proof binds the state by
    that name, and a hint reaches the closers of each as it was printed."""
    hint = "mul_self_nonneg ((s 0 0) - 1)"
    proofs = _proofs_text(_solver_hints(nlinarith_hints=[hint]))
    binders = re.findall(r"^theorem (\w+)_proof .*?:= by\n  intro (\w+)",
                         proofs, re.M | re.S)
    assert [b for _, b in binders] == ["s"] * 3, binders
    assert proofs.count(f"      | nlinarith [{hint}]\n") == 3


def test_hints_never_remove_a_closer():
    """`first | a | b` costs nothing for `b`, so there is no case for it."""
    ctx = LeanContext(_int_module())
    text = "fun s => ((s 0 0) * (s 0 0))"
    plain = plan_for(ctx, text)
    hinted = plan_for(ctx, text, hints=_solver_hints(linear_proof=True))
    assert set(plain.closers) <= set(hinted.closers)


def test_the_reasons_are_recorded_in_the_generated_comment():
    plan = _hinted(determined=[("(s 0 0) ≥ 0", True)], pre_unused=True)
    why = plan.why()
    assert "cvc5: inv forces" in why
    assert "no refutation needed the precondition: clearing it" in why


def _solver_hints(**fields):
    from verith.proofs.scripts import LeanHints

    return LeanHints(**fields)


def _proofs_text(hints) -> str:
    """`Proofs.lean` for `_int_module`'s Buchi certificate, proved by the
    plan `hints` inform."""
    from reactor_lean.proof.render import generate_proofs_lean
    from verith.proofs.scripts import proof_scripts

    return generate_proofs_lean(
        None, proof_scripts(LeanContext(_int_module()), hints=hints))


def test_norm_num_is_only_for_real_states():
    """It is 47 s of NN2Deep5's 49 s of tactic execution, and omega does
    not need it: 75.9 s with, 7.0 s without, same proof."""
    int_plan = plan_for(LeanContext(_int_module()), "fun s => (s 0 0)")
    assert not any("norm_num" in p for p in int_plan.prep)
    # still available as a closer for a goal that really does need it
    assert any("norm_num" in c for c in int_plan.closers)


def test_a_real_state_keeps_norm_num_in_prep():
    """Dropping it there leaves two of NN2RealAllPos4's obligations open."""
    plan = plan_for(LeanContext(_real_module()), "fun s => ((s 0 0) ≥ 0)")
    assert "norm_num at *" in plan.prep


# ══════════════════════════════════════════════════════════════════════
# The plan is about the predicates the certificate carries
#
# A route answers after the project is first generated, when there was no
# invariant to plan from. `verith.run.regenerate` plans again from the
# answer; without it an inferred certificate is planned for predicates it
# does not have.
# ══════════════════════════════════════════════════════════════════════


def _countdown():
    from reactor.analyzer import convert_method

    s = Var(Int([1, 1]))

    def init():
        return 100

    def update(old_x):
        return 100 if old_x == 0 else old_x - 1

    return Module.sequential(
        [s],
        convert_method(init, {}, [X(s)], number=Int),
        convert_method(update, {"old_x": s}, [X(s)], number=Int),
    )


def _before_and_after(inv, ranking, smt_tactics="cvc5", verbatim=None):
    """The certificate as the pipeline renders it before a route runs, and
    after the route has answered: its definitions (`defs`), and what the
    plan reads of its terms (`facts`) and of cvc5 (`hints`)."""
    pytest.importorskip("cvc5")
    from types import SimpleNamespace

    from reactor_lean.core.certificate import Certificate, Claim
    from reactor_lean.pipeline.certificate import render
    from verith.common.encoding import predicate_terms
    from verith.common.limits import SmtBudget
    from verith.proofs.facts import predicate_facts, solver_hints
    from verith.proofs.scripts import print_hints

    module = _countdown()
    claim = Claim(prp="(= s0 0)")

    def rendered(cert, hinted, verbatim=None):
        q = predicate_terms(module, claim, cert)
        return SimpleNamespace(
            defs=render(module, claim, cert, verbatim=verbatim),
            facts=predicate_facts(q),
            hints=(print_hints(solver_hints(q, SmtBudget()), q.msmt.ctrl_next)
                   if hinted else None),
        )

    # Before: the property is all there is to read the shape of.
    before = rendered(Certificate(), False)
    after = rendered(Certificate(inv, ranking),
                     smt_tactics == "cvc5" and verbatim is None, verbatim)
    return module, before, after


def test_the_inferred_certificate_brings_its_nonlinearity_to_the_plan():
    """`features_for` takes `nonlinear` from the facts *over* the text scan,
    so stale facts do not merely omit a closer -- they remove one the text
    had already found. An inferred `s0 * s0` ranking with the pre-inference
    facts loses `nlinarith` and `positivity` outright."""
    module, before, after = _before_and_after(
        "(and (>= s0 0) (<= s0 100))", "(* s0 s0)"
    )
    assert before.facts.nonlinear is False, "the property alone is linear"
    assert after.facts.nonlinear is True, "the inferred ranking is not"

    ctx = LeanContext(module)
    d = after.defs
    text = f"{d.prp}\n{d.inv}\n{d.ranking}"
    stale = plan_for(ctx, text, facts=before.facts)
    fresh = plan_for(ctx, text, facts=after.facts)
    assert not any("nlinarith" in c for c in stale.closers)
    assert any("nlinarith" in c for c in fresh.closers)
    assert "positivity" in fresh.closers


def test_cvc5_is_asked_about_the_invariant_that_now_exists():
    """`solver_hints` returns immediately on a `None` invariant, so before
    inference it answers nothing at all however branchy the certificate ends
    up. The condition below is one the invariant settles."""
    _module, _before, after = _before_and_after(
        "(and (>= s0 0) (<= s0 100))", "(ite (<= s0 200) s0 999)"
    )
    hints = after.hints
    assert hints is not None
    assert [value for _text, value in hints.determined] == [True]
    assert "≤ (200 : Int)" in hints.determined[0][0]


def test_a_settled_condition_is_printed_unshared():
    """A `have` outside the definition cannot name a `let` inside it, so the
    predicates are printed expanded once a condition is settled -- and all
    of them, since a condition can come from the property."""
    _module, _before, after = _before_and_after(
        "(and (>= s0 0) (<= s0 100))", "(ite (<= s0 200) s0 999)"
    )
    assert after.hints.determined
    assert "let " not in (after.defs.ranking or "")


def test_the_facts_do_not_wait_for_smt_tactics():
    """Reading the terms' shape is free -- they are parsed either way -- so
    it does not wait for `--smt-tactics cvc5`. Only the queries do."""
    _module, _before, after = _before_and_after(
        "(and (>= s0 0) (<= s0 100))", "(* s0 s0)", smt_tactics="none"
    )
    assert after.facts.nonlinear is True
    assert after.hints is None, "the queries are what the flag gates"


def test_an_answer_in_lean_is_planned_from_the_property_alone():
    """Nothing parses printed Lean back into a term, so a route that
    answers in Lean leaves the facts those of the property and asks cvc5
    nothing."""
    from reactor_lean.core.certificate import LeanAnswer

    answer = LeanAnswer(inv="fun s => (s 0 0) ≥ 0", ranking="fun s => 1")
    _module, before, after = _before_and_after(None, None, verbatim=answer)
    assert after.facts == before.facts
    assert after.hints is None
    assert (after.defs.inv, after.defs.ranking) == (answer.inv, answer.ranking)


# ── floored ranks ────────────────────────────────────────────────────────


def test_a_floored_rank_gets_the_bridge_to_the_real_under_it():
    """`linarith` reads `⌊x⌋` as an opaque `Int` atom.

    The hypotheses that would pin it are about the `Real` `x`, so nothing in
    the closer chain can cross: `0 < s 0 0 ⊢ 0 < 2 + ⌊s 0 0⌋` fails with the
    bracketing lemmas passed in, because half the facts are in the wrong
    sort. The floor step states the bridge in `Int`, where `omega` can use it.
    """
    plan = _plan(_real_module(), "fun s => ((2 + ⌊(s 0 0)⌋ : Int)).toNat")
    assert plan.features.has_floor
    assert plan.features.floor_args == ("(s 0 0)",)
    body = plan.floor_tactic
    assert "Int.floor_nonneg" in body
    assert "(s 0 0)" in body, "over the proofs' state binder"


def test_the_floor_sign_is_split_rather_than_assumed():
    """A `have` whose side condition needs a nested `by` cannot be recovered.

    A nested `by` that fails logs `unsolved goals` at its own position
    instead of raising, so the enclosing `try` does not catch it and the
    failure lands in the build. `by_cases` has no side condition to get
    wrong, and one branch of it is `skip`.
    """
    plan = _plan(_real_module(), "fun s => ((2 + ⌊(s 0 0)⌋ : Int)).toNat")
    assert "by_cases" in plan.floor_tactic
    assert ":= by" not in plan.floor_tactic


def test_a_scaled_floor_distributes_before_the_closers():
    """`norm_num` knows `⌊y - 1⌋ < ⌊y⌋` and not `⌊k*(x - c)⌋ < ⌊k*x⌋`.

    `m_lra_half` ranks by `⌊2*s⌋` and steps by `1/2`, so the scale has to be
    distributed for the two arguments to differ by a literal.
    """
    plan = _plan(_real_module(), "fun s => ((2 + ⌊((2 : Real) * (s 0 0))⌋ : Int)).toNat")
    assert "simp only [mul_sub]" in plan.prep
    assert plan.features.floor_args == ("((2 : Real) * (s 0 0))",)


def test_an_integer_module_pays_nothing_for_floors():
    plan = _plan(_int_module(), "fun s => ((s 0 0) = 2)")
    assert not plan.features.has_floor
    assert plan.floor_tactic == ""
    assert "simp only [mul_sub]" not in plan.prep


# ══════════════════════════════════════════════════════════════════════
# Closers for structure the cheap provers do not take apart
# ══════════════════════════════════════════════════════════════════════


def test_a_real_state_gets_lra_close_after_every_cheaper_closer():
    """Behind everything else, so a proof that closes today never reaches it."""
    plan = _plan(_real_module(), "fun s => ((s 0 0) ≥ 0)")
    assert "lra_close" in plan.closers
    assert plan.closers.index("lra_close") > plan.closers.index("(simp_all; norm_num; done)")


def test_an_integer_state_does_not_pay_for_lra_close():
    plan = _plan(_int_module(), "fun s => ((s 0 0) ≥ 0)")
    assert "lra_close" not in plan.closers


def test_a_branching_integer_state_splits_the_hypotheses_too():
    """`split_ifs` in prep splits the goal; a Bool weighed inside a hypothesis
    needs `split_ifs at *` and the substitution before omega can read it."""
    plan = _plan(_int_module(), "fun s => ((if (s 0 0) = 0 then 1 else 0) ≥ 0)")
    assert "(split_ifs at * <;> simp_all <;> omega)" in plan.closers


def test_a_nonlinear_predicate_gets_its_max_split_into_cases():
    plan = _plan(_int_module(), "fun s => (((s 0 0) * (max (s 0 0) 0) : Int)).toNat")
    assert any(c.startswith("(simp only [max_def, min_def] at *") and c.endswith("nlinarith)")
               for c in plan.closers)
    linear = _plan(_int_module(), "fun s => ((max (s 0 0) 0 : Int)).toNat")
    assert not any("max_def" in c for c in linear.closers)


def test_the_last_resorts_come_after_every_other_closer():
    """A closer added for goals nothing else proves goes after all the
    closers that were there, so a proof that builds today never reaches it."""
    old = {"omega", "contradiction", "linarith", "(norm_cast; omega)", "(norm_num; done)",
           "nlinarith", "positivity", "tauto", "(simp_all; done)", "(simp_all; omega)",
           "(simp_all; linarith)", "(simp_all; nlinarith)", "lra_close",
           "(split_ifs at * <;> simp_all <;> omega)"}
    plan = _plan(_int_module(), "fun s => (if (s 0 0) = 0 then (s 0 0) * (s 0 0) else 0 : Int).toNat")
    new = [c for c in plan.closers if c.startswith("(zify;") or c == "lia"]
    assert len(new) == 2
    first_new = min(plan.closers.index(c) for c in new)
    assert all(plan.closers.index(c) < first_new for c in plan.closers
               if c in old or c.startswith("(simp only [max_def") or c.startswith("(constructor"))


def test_a_product_under_a_toNat_is_moved_to_int_first():
    """`nlinarith` reads `(a * b).toNat` as an atom; the closer that takes
    each `toNat` to a `max` in `ℤ` comes after the one that does not."""
    plan = _plan(_int_module(), "fun s => (((s 0 0) * (s 0 0) + 1 : Int)).toNat")
    zified = [c for c in plan.closers if c.startswith("(zify;")]
    assert len(zified) == 1 and "Int.toNat_eq_max" in zified[0]
    assert plan.closers.index(zified[0]) > plan.closers.index("nlinarith")
    linear = _plan(_int_module(), "fun s => (((s 0 0) + 1 : Int)).toNat")
    assert not any(c.startswith("(zify;") for c in linear.closers)


def test_lia_is_the_last_integer_closer():
    """omega gives up on integer problems only integrality refutes; `lia`
    does not, and runs only where every other closer gave up."""
    closers = _plan(_int_module(), "fun s => ((s 0 0) ≥ 0)").closers
    assert closers[-1] == "lia"
    assert "lia" not in _plan(_real_module(), "fun s => ((s 0 0) ≥ 0)").closers
    assert "lia" not in _plan(_bv_module(), "fun s => ((s 0 0) = 1#1)").closers


def test_a_ceiling_gets_its_own_lra_close_last():
    """`lra_close` reads `⌈x⌉₊` as an atom; the variant that bounds it comes
    after every other closer, and only where a ceiling, or a floor into
    `ℕ`, is written."""
    for rank in ("⌈(s 0 0) * 2⌉₊", "Nat.ceil ((s 0 0) * 2)", "(⌈s 0 0⌉).toNat", "⌊s 0 0⌋₊"):
        closers = _plan(_real_module(), f"fun s => {rank}").closers
        assert closers.index("lra_close_ceil") > closers.index("lra_close"), rank
    assert "lra_close_ceil" not in _plan(_real_module(), "fun s => ⌊s 0 0⌋.toNat").closers


def test_a_floored_real_rank_gets_the_floors_split_last():
    closers = _plan(_real_module(), "fun s => ⌊2 * s 0 0⌋.toNat").closers
    assert closers.index("lra_close_frac") > closers.index("lra_close")
    assert "lra_close_frac" not in _plan(_real_module(), "fun s => ((s 0 0) ≥ 0)").closers


def test_the_hammer_defines_every_closer_the_plan_can_name():
    """A closer the plan names and the project does not define is a parse
    error in every certificate that carries it, not a failed alternative."""
    from verith.proofs import generate_zeroth_hammer_lean
    hammer = generate_zeroth_hammer_lean()
    for name in ("lra_close", "lra_leaf", "lra_atoms", "floor_bounds",
                 "lra_close_ceil", "ceil_bounds", "lra_close_frac", "floor_split"):
        assert f'"{name}"' in hammer


def _branchy_transition(n: int) -> Module:
    """A countdown whose next value is chosen by an `n`-deep chain of `ite`s."""
    x = Var(Int([1, 1]))
    zero, one = Wire(Int([1, 1])), Wire(Int([1, 1]))
    terms = [Term(tensor.int.Const(torch.tensor([[0]])), [zero]),
             Term(tensor.int.Const(torch.tensor([[1]])), [one])]
    prev = x
    for k in range(n):
        cond, out = Wire(Bool([1, 1])), (X(x) if k == n - 1 else Wire(Int([1, 1])))
        dec = Wire(Int([1, 1]))
        terms += [Term(tensor.int.Eq([1, 1]), [cond], [prev, zero]),
                  Term(tensor.int.Sub([1, 1]), [dec], [prev, one]),
                  Term(tensor.int.Ite([1, 1]), [out], [cond, zero, dec])]
        prev = out
    return Module.sequential([x], [Term(tensor.int.Const(torch.tensor([[9]])), [X(x)])], terms)


def test_a_branchy_transition_gets_the_higher_budget():
    """`step_inv` splits the transition's branches before any closer runs, so
    a Petri net's 16 of them cost what a 32-branch certificate does."""
    narrow = _plan(_branchy_transition(3), "fun s => ((s 0 0) ≥ 0)")
    assert narrow.features.n_trans_branch == 3
    assert narrow.max_heartbeats == 2000000
    wide = _plan(_branchy_transition(16), "fun s => ((s 0 0) ≥ 0)")
    assert wide.features.n_trans_branch == 16
    assert wide.max_heartbeats == 8000000


def test_the_simp_set_unfolds_the_reductions():
    """`matMin`, `matMax` and the argmaxes unfold to a `List.ofFn` fold that
    `simp_mat` computes (`lean/tests/test_lean_templates.py` holds the
    library to that); computing the fold is no use if the definition never
    opens."""
    from reactor_lean.proof.render import generate_proofs_lean
    from verith.proofs.scripts import proof_scripts

    proofs = generate_proofs_lean(
        None, proof_scripts(LeanContext(_int_module())))
    simp_mat = [ln for ln in proofs.splitlines()
                if "simp [" in ln and "add_Matrix_apply" in ln]
    assert simp_mat, "the simp_mat macro moved"
    for name in ("matMin", "matMax", "argmax_1d", "argmax"):
        assert name in simp_mat[0], f"{name} is not in simp_mat"


# ══════════════════════════════════════════════════════════════════════
# The simp sets, per module
# ══════════════════════════════════════════════════════════════════════


def _twobitcounter():
    """Bool-only module: two-bit counter with enable."""
    b0, b1, enable = Var(Bool([1, 1])), Var(Bool([1, 1])), Var(Bool([1, 1]))
    not_b0, not_b1 = Wire(Bool([1, 1])), Wire(Bool([1, 1]))
    b0_and_enable = Wire(Bool([1, 1]))
    init = [
        Term(tensor.bool.Const(torch.tensor([[False]])), [X(b0)]),
        Term(tensor.bool.Const(torch.tensor([[False]])), [X(b1)]),
    ]
    update = [
        Term(tensor.bool.Not([1, 1]), [not_b0], [b0]),
        Term(tensor.bool.Ite([1, 1]), [X(b0)], [X(enable), not_b0, b0]),
        Term(tensor.bool.And([1, 1]), [b0_and_enable], [b0, X(enable)]),
        Term(tensor.bool.Not([1, 1]), [not_b1], [b1]),
        Term(tensor.bool.Ite([1, 1]), [X(b1)], [b0_and_enable, not_b1, b1]),
    ]
    return Module.sequential([b0, b1, enable], init, update)


def _affine_module():
    """`x' = A·u` at init and `x' = x + e1` after: both integer `Linear`, with
    the matrices baked into the op rather than interned as constants."""
    x, u = Var(Int([3, 1])), Var(Int([2, 1]))
    a = torch.tensor([[0, 0], [1, 0], [0, 1]], dtype=torch.int64)
    init = [Term(tensor.int.Linear(a, torch.zeros((3, 1), dtype=torch.int64)),
                 [X(x)], [X(u)])]
    e1 = torch.tensor([[1], [0], [0]], dtype=torch.int64)
    update = [Term(tensor.int.Linear(torch.eye(3, dtype=torch.int64), e1), [X(x)], [x])]
    return Module.sequential([x, u], init, update)


def _simp_macros(module) -> str:
    from verith.proofs.scripts import proof_scripts

    return proof_scripts(LeanContext(module)).preamble


def test_a_bool_module_unfolds_its_definitions_and_no_constants():
    macros = _simp_macros(_twobitcounter())
    assert 'macro "simp_mat"' in macros
    assert "init, update, inv" in macros
    # A Bool module has no matrix constants, so no c0/c1 in simp.
    assert "c0, c1" not in macros


def test_an_affine_module_reduces_its_matrix_products():
    """`affineLinear` unfolds to `A * x + b`, so the simp set carries
    `mul_Matrix_apply` to reduce the matrix obligations."""
    assert "mul_Matrix_apply" in _simp_macros(_affine_module())
