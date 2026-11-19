import ZerothHammer
import Relational.Mat

/-!
# ManualTests.Basic

Tests that ``zeroth_hammer`` (imported from ZerothHammer) closes goals
representative of each hammer phase.

Unlike the certificate files in Certs/, there is no reactive module here.
We define generic stubs for the macros that zeroth_hammer calls internally
so all phases remain exercisable.

Run via pytest:
    just pytest tests/lean/           # fast: only checks generated files exist
    just pytest tests/lean/ -m slow   # slow: runs lake build
-/

open Lean Elab Tactic

-- Generic macros for zeroth_hammer.
-- In certificate files these are generated from the module; here we use
-- a module-independent version that reduces standard matrix/arithmetic lemmas.
macro "simp_mat" : tactic =>
  `(tactic| simp [zero_Matrix_apply, Pi.add_apply,
                  mul_Matrix_apply, add_Matrix_apply, Bool.or_eq_true, decide_eq_true_eq,
                  Fin.sum_univ_succ, Fin.sum_univ_zero, Fin.isValue,
                  Fin.sum_univ_one, Fin.sum_univ_two, Fin.sum_univ_three])

macro "simp_defs" : tactic =>
  `(tactic| (simp only [] at *; try dsimp at *))

macro "mat_collapse" : tactic =>
  `(tactic| simp only [Matrix_1_1_lt_iff, Matrix_1_1_le_iff, Matrix_1_1_eq_iff, Matrix_1_1_ne_iff,
                        ite_fun_apply, decide_eq_true_eq, Bool.or_eq_true,
                        Bool.and_eq_true, Bool.not_eq_true'] at *)

-- ──────────────────────────────────────────────────────────────
-- Phase 0: simp_mat alone closes trivially-True goals
-- ──────────────────────────────────────────────────────────────

example : True := by zeroth_hammer
example (s : Int) : True := by zeroth_hammer
example (s : Matrix (Fin 1) (Fin 1) Int) : True := by zeroth_hammer

-- ──────────────────────────────────────────────────────────────
-- Phase 1a: omega alone
-- ──────────────────────────────────────────────────────────────

example (x : Int) (h : x ≥ 0) : x + 1 ≥ 1 := by zeroth_hammer
example (x x' : Int) (hinv : x ≥ 0) (hstep : x' = x - 1) (hP : x > 0) :
    x' ≥ 0 := by zeroth_hammer

-- ──────────────────────────────────────────────────────────────
-- Phase 1b: norm_cast + omega (Int.toNat in rankings)
-- ──────────────────────────────────────────────────────────────

example (s : Int) (hinv : 0 ≤ s) (hP : ¬(s ≤ 0)) :
    (s - 1).toNat < s.toNat := by zeroth_hammer

example (s target : Int) (hinv : target ≤ s) (hP : s ≠ target) :
    (s - 1 - target).toNat < (s - target).toNat := by zeroth_hammer

-- ──────────────────────────────────────────────────────────────
-- Phase 1d: linarith
-- ──────────────────────────────────────────────────────────────

example (a b c : Int) (h1 : 2 * a ≤ b + 3) (h2 : b ≤ a + c) (h3 : c ≤ 1) :
    a ≤ 4 := by zeroth_hammer

-- ──────────────────────────────────────────────────────────────
-- Phase 2: push Not normalises negated arithmetic
-- ──────────────────────────────────────────────────────────────

example (x : Int) (h : ¬(x > 10)) : x ≤ 10 := by zeroth_hammer
example (s : Int) (hP : ¬(s = 0)) : s ≠ 0 := by zeroth_hammer

-- ──────────────────────────────────────────────────────────────
-- Phase 3: deep case-split cascade
-- ──────────────────────────────────────────────────────────────

example (x : Int) (b : Bool) (h : x ≥ 0) :
    (if b then x + 1 else x) ≥ 0 := by zeroth_hammer

example (x : Int) (b₁ b₂ : Bool) (h : x ≥ 2) :
    (if b₁ then if b₂ then x - 2 else x - 1 else x) ≥ 0 := by zeroth_hammer

-- ──────────────────────────────────────────────────────────────
-- Phase 6: aesop (general-purpose Prop reasoning)
-- ──────────────────────────────────────────────────────────────

example (p q : Prop) (hp : p) (hpq : p → q) : q := by zeroth_hammer
example (p q r : Prop) (h : p ∧ q) (hqr : q → r) : p ∧ r := by zeroth_hammer

-- ──────────────────────────────────────────────────────────────
-- lra_close: what a Real plan's last closer has to take apart
-- ──────────────────────────────────────────────────────────────

-- A ReLU reaches the goal as `max 0 x`, which `linarith` reads as an atom.
example (s : ℝ) (h : 1 < s) : ⌊max 0 (s - 1)⌋ < ⌊s⌋ := by lra_close
example (s : ℝ) : 0 ≤ ⌊max 0 (s - 1)⌋ := by lra_close
-- A floor is an integer, which it stops being once the goal is in `ℝ`.
example (s : ℝ) (h : 0 ≤ s - 1) : 0 ≤ ⌊s - 1⌋ := by lra_close
-- A mode-split invariant: an `if` in a hypothesis, not in the goal.
example (x : ℝ) (b : Bool) (hb : b = true)
    (h : if b = true then x ≤ 27 / 2 else 11 / 2 ≤ x) : x + 3 / 2 ≤ 15 := by lra_close
