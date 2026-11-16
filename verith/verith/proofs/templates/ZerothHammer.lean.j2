import Lean
import Mathlib.Tactic
import Smt

open Lean Elab Tactic

-- Default stub macros; certificate files redefine these for their specific module.
macro "simp_mat"     : tactic => `(tactic| simp)
macro "simp_defs"    : tactic => `(tactic| simp only [])
macro "mat_collapse" : tactic => `(tactic| simp only [])

-- ── Linear arithmetic under propositional structure ─────────────────────
--
-- What `linarith` is handed after `split_ifs` is often not a conjunction of
-- inequalities: an implication left in the goal, a disjunctive goal, a
-- hypothesis `b = false → 5 ≤ x`, a floor on either side. Each of those is
-- still linear arithmetic once the structure is taken apart, and `lra_close`
-- is the taking apart. The certificates `verith/proofs/tactics.py` gives
-- it to are ones cvc5 has already checked, so what fails here is the proof
-- search, never the claim.

/-- Every `⌊a⌋` in `e`, by its argument `a`. Only the outermost application
    is a floor of something: descending into its function part would meet
    partial applications of `Int.floor` and take an instance for `a`. -/
partial def collectFloorArgs (e : Expr) (acc : Array Expr) : Array Expr :=
  if e.getAppFn.isConstOf ``Int.floor && e.getAppNumArgs > 0 then
    let a := e.appArg!
    collectFloorArgs a (if acc.contains a then acc else acc.push a)
  else match e with
  | .app f a => collectFloorArgs a (collectFloorArgs f acc)
  | .lam _ t b _ | .forallE _ t b _ => collectFloorArgs b (collectFloorArgs t acc)
  | .letE _ t v b _ => collectFloorArgs b (collectFloorArgs v (collectFloorArgs t acc))
  | .mdata _ b | .proj _ _ b => collectFloorArgs b acc
  | _ => acc

/-- For every `⌊a⌋` in the goal or the context, add `↑⌊a⌋ ≤ a` and
    `a < ↑⌊a⌋ + 1`. `linarith` reads `⌊a⌋` as an opaque atom; with these two
    facts beside it, and the goal moved to `ℝ` by `rify`, a decrease such as
    `⌊-7x - 30⌋ < ⌊-8x⌋` under `x < 21` is ordinary linear arithmetic. -/
elab "floor_bounds" : tactic => withMainContext do
  let mut es := #[← instantiateMVars (← getMainTarget)]
  for d in ← getLCtx do
    if !d.isImplementationDetail then es := es.push (← instantiateMVars d.type)
  for a in es.foldl (fun acc e => collectFloorArgs e acc) #[] do
    let stx ← Term.exprToSyntax a
    evalTactic (← `(tactic| have := Int.floor_le $stx))
    evalTactic (← `(tactic| have := Int.lt_floor_add_one $stx))

-- Every implication and negation becomes a disjunction of atoms.
macro "lra_atoms" : tactic =>
  `(tactic| (try simp only [imp_iff_not_or, not_and_or, not_or, not_lt, not_le, not_not,
                             Bool.not_eq_true, Bool.not_eq_false] at *))

-- One branch: a Bool contradiction, plain linear arithmetic, or the same
-- with every floor bounded and the context and goal moved to `ℝ` -- the
-- context too, since a refuted goal is `False` and what is left to move is
-- the negation `by_contra` put among the hypotheses. Last, the same once
-- every strict `Int` inequality is stated as the non-strict one it is
-- (`⌊a⌋ < 0` as `⌊a⌋ + 1 ≤ 0`): in `ℝ` the floor is no longer an integer,
-- and `0 ≤ ⌊x - 1⌋` from `0 ≤ x - 1` needs it to be.
macro "lra_leaf" : tactic =>
  `(tactic| first
     | (simp_all; done)
     | linarith
     | (floor_bounds; (try rify at *); linarith)
     | (floor_bounds; simp only [Int.lt_iff_add_one_le] at *; (try rify at *); linarith))

-- Take `max`/`min` and every `if` -- in the context as well as the goal --
-- apart into cases, case-split the context, split the goal's conjunction,
-- and refute a disjunctive goal rather than choosing a side of it. A ReLU
-- reaches the goal as `max 0 x`, which `linarith` reads as an opaque atom,
-- and a mode-split invariant as `if b then .. else ..` in a hypothesis that
-- `split_ifs` in the plan's prep, which splits the goal, never reached.
macro "lra_close" : tactic =>
  `(tactic| (intros; (try simp only [max_def, min_def] at *); (try split_ifs at *) <;>
             (lra_atoms; (try casesm* _ ∧ _, _ ∨ _); (repeat' apply And.intro);
              all_goals first
                | lra_leaf
                | (by_contra hneg; lra_atoms; (try casesm* _ ∧ _, _ ∨ _); all_goals lra_leaf))))

/-- Every `⌈a⌉` in `e`, by its argument `a`, as `collectFloorArgs` collects
    the floors. -/
partial def collectCeilArgs (e : Expr) (acc : Array Expr) : Array Expr :=
  if e.getAppFn.isConstOf ``Int.ceil && e.getAppNumArgs > 0 then
    let a := e.appArg!
    collectCeilArgs a (if acc.contains a then acc else acc.push a)
  else match e with
  | .app f a => collectCeilArgs a (collectCeilArgs f acc)
  | .lam _ t b _ | .forallE _ t b _ => collectCeilArgs b (collectCeilArgs t acc)
  | .letE _ t v b _ => collectCeilArgs b (collectCeilArgs v (collectCeilArgs t acc))
  | .mdata _ b | .proj _ _ b => collectCeilArgs b acc
  | _ => acc

/-- For every `⌈a⌉` in the goal or the context, add `a ≤ ↑⌈a⌉` and
    `↑⌈a⌉ < a + 1`: `floor_bounds` for a ceiling. -/
elab "ceil_bounds" : tactic => withMainContext do
  let mut es := #[← instantiateMVars (← getMainTarget)]
  for d in ← getLCtx do
    if !d.isImplementationDetail then es := es.push (← instantiateMVars d.type)
  for a in es.foldl (fun acc e => collectCeilArgs e acc) #[] do
    let stx ← Term.exprToSyntax a
    evalTactic (← `(tactic| have := Int.le_ceil $stx))
    evalTactic (← `(tactic| have := Int.ceil_lt_add_one $stx))

-- `lra_close` over a rank that takes a ceiling, or a floor into `ℕ`:
-- `⌈x⌉₊` is `⌈x⌉.toNat` and `⌊x⌋₊` is `⌊x⌋.toNat`, a truncated `m - n` is
-- `(↑m - ↑n).toNat`, each `toNat` is a `max` with `0` in `ℤ`, and the
-- ceilings are bounded as the floors are. A ranking written
-- `Nat.ceil ((12 - h) * 2 / 3)` decreases by one per step, and
-- `⌈x - 1⌉₊ < ⌈x⌉₊` was out of reach while `⌈·⌉₊` was an atom; the
-- subtraction is `norm_num`'s, which writes `⌊8 * x - 95⌋₊` as
-- `⌊8 * x⌋₊ - 95`.
macro "lra_close_ceil" : tactic =>
  `(tactic| ((try simp only [← Int.ceil_toNat, ← Int.floor_toNat] at *);
             (try simp only [← Int.toNat_sub] at *); (try zify at *);
             (try simp only [Int.toNat_eq_max] at *); ceil_bounds; lra_close))

/-- Every floor `⌊a⌋` in the goal or the context, split at `0`: `⌊a⌋ ≤ 0`
    or `0 < ⌊a⌋`, on every goal. In `ℝ` a floor is no longer an integer,
    and `lra_leaf` turns the second into `1 ≤ ⌊a⌋`, which is where the
    integrality is. -/
elab "floor_split" : tactic => withMainContext do
  let mut es := #[← instantiateMVars (← getMainTarget)]
  for d in ← getLCtx do
    if !d.isImplementationDetail then es := es.push (← instantiateMVars d.type)
  for a in es.foldl (fun acc e => collectFloorArgs e acc) #[] do
    let stx ← Term.exprToSyntax a
    evalTactic (← `(tactic| all_goals rcases le_or_gt ⌊$stx⌋ 0 with _ | _))

-- `lra_close` once each floor is split at `0`, and `Int.fract x` is
-- `x - ⌊x⌋` again (`norm_num` folds the difference into it). An invariant
-- that says `2 * x` is a whole number, `2 * x - ⌊2 * x⌋ = 0`, rules out
-- `0 < x < 1/2` only through `⌊2 * x⌋` being an integer: `m_lra_half`'s,
-- beside a rank of `⌊2 * x⌋`.
macro "lra_close_frac" : tactic =>
  `(tactic| ((try simp only [← Int.self_sub_floor] at *); floor_split; all_goals lra_close))

syntax "zeroth_hammer" : tactic

/-- Zeroth hammer: cascading automated prover for reactive module goals.
    Phase 0: simp alone (closes trivially-True invariants)
    Phase 1: fast arithmetic — omega, norm_cast+omega, simp+omega, simp+linarith
    Phase 2: push Not + simp + omega (negated arithmetic)
    Phase 3: simp + deep case-split + omega/linarith/norm_cast (branching goals)
    Phase 4: simp_defs + case-split (unfold defs before splitting)
    Phase 5: full reduction + mat_collapse + split_ifs + omega/linarith (hrank)
    Phase 6: aesop (general-purpose proof search)
    Phase 7: smt fallback (cvc5)
    Phase 8: sorry (explicit give-up) -/
elab_rules : tactic
  | `(tactic| zeroth_hammer) => do
      -- Phase 0: simp_mat alone (closes trivial True goals without needing omega)
      -- Note: simp never throws when it makes partial progress, so we must
      -- check goals explicitly rather than relying on try/return/catch.
      try evalTactic (← `(tactic| simp_mat)) catch _ => pure ()
      if (← Lean.Elab.Tactic.getUnsolvedGoals).isEmpty then return
      -- Phase 1: fast arithmetic passes
      -- 1a: omega alone (goal already in linear arithmetic fragment after intros)
      try evalTactic (← `(tactic| omega)); return catch _ => pure ()
      -- 1b: norm_cast + omega (normalises Nat/Int coercions, e.g. Int.toNat in rankings)
      try evalTactic (← `(tactic| norm_cast <;> omega)); return catch _ => pure ()
      -- 1c: simp_mat + omega (main fast path)
      try
        evalTactic (← `(tactic| simp_mat <;> omega))
        return
      catch _ => pure ()
      -- 1d: simp_mat + linarith (ordered-ring arithmetic; fallback when omega is too weak)
      try
        evalTactic (← `(tactic| simp_mat <;> linarith))
        return
      catch _ => pure ()
      -- Phase 2: push Not normalises negated arithmetic before omega
      -- (e.g. ¬(x > 0) → x ≤ 0; useful when inv or P contains negated comparisons)
      try
        evalTactic (← `(tactic| push Not; simp_mat <;> omega))
        return
      catch _ => pure ()
      -- Phase 3: simp_mat + deep case-split cascade (branching state machines)
      -- Four levels of if-branch splitting; tries omega/linarith/norm_cast at each leaf
      try
        evalTactic (← `(tactic|
          simp_mat
          <;> first
            | omega
            | linarith
            | (norm_cast; omega)
            | (push Not; omega)
            | (simp_all; omega)
            | (split <;> simp_all <;> omega)
            | (split <;> split <;> simp_all <;> omega)
            | (split <;> split <;> split <;> simp_all <;> omega)
            | (split <;> split <;> split <;> split <;> simp_all <;> omega)))
        return
      catch _ => pure ()
      -- Phase 4: unfold definitions everywhere first, then case-split
      -- Useful when inv/P/ranking are not yet visible to the split heuristic
      try
        evalTactic (← `(tactic|
          simp_defs
          <;> first
            | omega
            | linarith
            | (norm_cast; omega)
            | (simp_all; omega)
            | (split <;> simp_all <;> omega)
            | (split <;> split <;> simp_all <;> omega)
            | (split <;> split <;> split <;> simp_all <;> omega)))
        return
      catch _ => pure ()
      -- Phase 5: full pipeline for ranking proofs
      -- Unfold defs in hypotheses → reduce matrices → collapse Mat 1 1 to scalar
      -- → split all ite → omega / linarith / norm_cast
      try evalTactic (← `(tactic| simp_defs)) catch _ => pure ()
      if (← Lean.Elab.Tactic.getUnsolvedGoals).isEmpty then return
      -- Check for contradictory hypotheses (e.g. ¬True from vacuous hrank)
      -- `contradiction`, `decide` and `native_decide` act on the main goal
      -- only. Returning unconditionally after one of them therefore left any
      -- sibling goal an earlier split had produced unproved *and* skipped the
      -- `sorry` fallback below, surfacing as "unsolved goals" rather than a
      -- give-up. Apply them to every goal and return only if none remain.
      try evalTactic (← `(tactic| all_goals contradiction)) catch _ => pure ()
      if (← Lean.Elab.Tactic.getUnsolvedGoals).isEmpty then return
      -- Try decide/native_decide after full reduction (works for finite Bool state)
      try evalTactic (← `(tactic| all_goals decide)) catch _ => pure ()
      if (← Lean.Elab.Tactic.getUnsolvedGoals).isEmpty then return
      try evalTactic (← `(tactic| all_goals native_decide)) catch _ => pure ()
      if (← Lean.Elab.Tactic.getUnsolvedGoals).isEmpty then return
      -- Reduce matrices and collapse Mat 1 1 to bare scalar arithmetic
      try evalTactic (← `(tactic| simp_mat)) catch _ => pure ()
      try evalTactic (← `(tactic| mat_collapse)) catch _ => pure ()
      if (← Lean.Elab.Tactic.getUnsolvedGoals).isEmpty then return
      try
        evalTactic (← `(tactic|
          split_ifs at *
          <;> first
            | omega
            | linarith
            | (norm_cast; omega)
            | (norm_cast; linarith)
            | simp_all
            | (simp_all; omega)
            | (simp_all; linarith)
            | positivity))
        return
      catch _ => pure ()
      -- Phase 6: aesop (general-purpose proof search before SMT)
      try
        evalTactic (← `(tactic| aesop))
        return
      catch _ => pure ()
      -- Phase 7: smt after full reduction
      try
        evalTactic (← `(tactic| smt))
        return
      catch _ => pure ()
      -- Phase 8: sorry (explicit give-up)
      evalTactic (← `(tactic| sorry))
