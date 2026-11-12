# Verifying a reactive module with `reactor`, `verith` and `certify`

`reactor` is the command-line front end of the Lean pipeline. It takes a
**module file** — a Python file exposing a callable that returns a
`reactor.Module` — and writes a Lean 4 project holding the module's transition
system *and* a machine-checkable **certificate** for one temporal property,
whose obligations it states and leaves for `verith` to prove:

```
module.py ──reactor─▶ <OUT>/Rea/                 a Lean 4 package
                        ├── System/              the module: init / update, in five encodings
                        └── Certificate/
                            ├── Data.lean        the certificate data: init_pre, update_pre, inv, P, ranking
                            ├── Module.lean      the module as a transition system
                            ├── Proofs.lean      the proof obligations, proved by `sorry`
                            └── Certificate.lean the property, from the obligations
                                    │
                                    ▼   verith --project <OUT>/Rea
                        Proofs.lean proved by generated tactic scripts,
                        ZerothHammer.lean the tactics they call
                                    │
                                    ▼   lake build Certificate
                              proved, or not
```

A certificate proves exactly one property under exactly one proof rule, and
the flag you pass the formula under **is** the choice of rule:

| flag | property | rule | the certificate consists of |
|---|---|---|---|
| `--safety P` | `G P` — `P` holds in every reachable state | `rule_globally` | an inductive **invariant** that implies `P` |
| `--buchi P` | `G (F P)` — `P` holds infinitely often | `rule_buchi` | an inductive **invariant** *and* a **ranking function** that strictly decreases wherever `¬P` |

Properties, invariants, ranking functions and preconditions are all written as
**SMT-LIB 2** expressions over the module's wires:

* `(X @intf.0)`, `(X @intf.1)`, … — the controlled (state) wires after the round, in the order the module declares them
* `(X @extl.0)`, … — the external inputs read in the round; `@extl.0`, … — the ones read in the round before
* components of a vector wire take a tuple selector: `((_ tuple.select 1) (X @extl.0))`
* a module file that names its variables (`reactor.lean.inputs.Named`) can be written in those names: `(X x)` is `x` after the round, a bare `x` is an input of the round before

Inside certificates, SMT queries, Lean and the artifacts the same wires are
called `s0, s1, …`, `e0, …` and `el0, …`; the outputs below show that
spelling. A bare `s0`, `e0` or `el0` is refused on the command line.

**What this tutorial does.** Every command below is one you can paste into a
shell, in order, and every output is what it printed here.

1. **Setup** — the output directory, the warm Lean build, the optional keys
2. **Safety** on a module written in the Python API
3. **Büchi** (liveness) on a module written in the Python API
4. A module read out of a **gymnasium** environment
5. **When verification fails** — an invalid invariant, and a missing precondition
6. Letting an **LLM** find the certificate (`--infer`)
7. **Learning** the certificate instead (`--infer nuterm`)
8. **Searching a shape** instead of proposing one (`--infer smt-linear`, `--infer sygus`)
9. Letting **ic3ia** find the invariant (`--fbk-proveit`)
10. A flag reference

The proofs, and the search for a certificate that is not given, are the
[`verith/`](../verith/README.md) project's. `verith` proves the certificate
of a project `reactor` generated (section 2), or searches for one with
`--infer` and prints what it found, in the form `--invariant` and
`--ranking` take. `certify` is both in one run: `reactor`'s flags,
`verith`'s, one project. From section 3 on, every command is `certify`.

Every module used below is a fixture that already lives in `python/tests/`, so
nothing has to be written first.

---

## 1. Setup

`uv run reactor`, `uv run verith` and `uv run certify` are not installed
globally; they run out of the checkout. All commands below are run from `python/`:

```bash
cd <checkout>/python
uv sync                    # builds reactor via maturin and installs dependencies
uv run reactor --help
uv run verith --help       # --infer with every route's flags, and the proofs'
uv run certify --help      # both
```

**One output directory for every command.** `reactor` regenerates
`<OUT>/<PROJ>` in place — it never wipes the directory — so the project's
`.lake` survives from one example to the next, and with it every `.olean` of
Mathlib, cslib, lean-smt and `ZerothHammer`; reactor-lean's `Relational` is
built once, in the checkout itself. Only `System/*` and `Certificate/*`
recompile per example: the first `lake build` below costs about 15 s and each
later one about 10 s, against an hour from cold.

```bash
export REPO=$(cd .. && pwd)
export OUT=/tmp/verith-tutorial.noindex       # keep the .noindex suffix on macOS
export PROJ=Rea
export REACTOR_LEAN=$(cd "$REPO/../reactor-lean" && pwd)   # a reactor-lean checkout

mkdir -p "$OUT/shared_lake"
ln -sfn "$REPO/python/tests/lean/.lake/packages" "$OUT/shared_lake/packages"
```

Three things make that work:

* `<OUT>/<PROJ>/.lake` is a symlink to one shared build directory whose
  `packages` points at the already-built packages of `python/tests/lean`.
  Build those once with `cd python/tests/lean && lake build`.
* The Lean library the project is stated in, `Relational`, is not one of
  those packages. It is a package of [reactor-lean], which the lakefile
  requires from git; with `REACTOR_LEAN` set to a checkout of it, `reactor`
  writes the lakefile to require the checkout by path instead.
* Keep the `.noindex` suffix on `OUT` if you are on macOS. A build writes tens
  of thousands of `.olean` files; Spotlight indexing them has turned a 9 s
  build into a 928 s one, and a directory whose name ends in `.noindex` is
  skipped.

[reactor-lean]: https://github.com/zeroth-research/reactor-lean

One shared build directory takes **one writer** — don't run two of these
sessions, or `tests/limits/run_limits.py`, against the same `OUT` at the same
time. Module names are identical in every generated project, so concurrent
builds serve each other's oleans, and the symptom is
`unknown constant 'hrank'`, which reads exactly like a codegen bug.

Building the certificate is `lake build Certificate` in the generated project.
`certify --build-cert` (or `verith --build-cert`) does it in one flag, which
is what you would type at a terminal — but it runs `lake update` first, and
that re-queries every dependency's git remote. Writing the warm manifest instead, with reactor-lean's
packages entered as path dependencies on the checkout, keeps every example
below off the network:

```bash
lake_cert() {                                  # build the certificate of $OUT/$PROJ
  ln -sfn "$OUT/shared_lake" "$OUT/$PROJ/.lake"
  uv run python -c '
import json, sys, torch
from pathlib import Path
from reactor.lean.project.lakefile import reactor_lean_checkout, with_reactor_lean
warm = json.loads(Path(sys.argv[1]).read_text())
Path(sys.argv[2]).write_text(json.dumps(with_reactor_lean(warm, reactor_lean_checkout())))
' "$REPO/python/tests/lean/lake-manifest.json" "$OUT/$PROJ/lake-manifest.json"
  (cd "$OUT/$PROJ" && lake build Certificate)
}
```

`reactor` prints the whole module and every file it writes. That is worth
reading once (section 2 below) and in the way afterwards, so the rest of this
tutorial pipes it through one filter — the lines about the *certificate*:

```bash
says() {                                       # what the run says about the certificate
  sed $'s/\x1b\\[[0-9;]*m//g' |                # drop the colour codes
  grep -E 'pre-check|holds|REFUTED|refuted|\[CEGIS\]|\[nuterm\]|\[sygus\]|\[smt-linear\]|inv:|ranking:|resuming|artifacts:|error:|Project ready|property is|invariant witness|certificate:|Installed|Wrote the'
}
```

Two routes are optional and need more than the checkout:

```bash
# `--infer` / `--infer ai-cegis`: an LLM proposes the certificate
export ANTHROPIC_API_KEY=sk-ant-...            # or --base-url for an OpenAI-compatible server
uv sync --extra ai                             # --extra ai-local for --base-url

# `--fbk-proveit`: ic3ia finds the invariant (section 9)
export PROVEIT=~/zeroth/proof-prototyping/lean-ltl-certifying
export IC3IA=~/zeroth/fbk/ic3ia/build/ic3ia
export PYTHONPATH=~/zeroth/fbk/mathsat/python  # vmt2lean.py does `from mathsat import *`
```

---

## 2. Safety: a module written in the Python API

`tests/limits/mods/m_countdown.py` is the smallest interesting module in the
repository: one integer state wire that counts down from 100 and wraps back to
100 at zero. It uses the **analyzer** front end — `convert_method` reads the
bytecode of two plain Python functions and turns them into the module's `init`
and `update` term lists.

```bash
cat tests/limits/mods/m_countdown.py
```

```python
"""LIA 1x1: x = 100, then x-1 each step, reset to 100 at 0. Baseline control."""
from reactor import Module, Int, LIA, Var, X
from reactor.analyzer import convert_method


def init():
    return 100


def update(old_x):
    if old_x == 0:
        return 100
    return old_x - 1


def module() -> Module:
    s = Var(Int([1, 1]))
    return Module.sequential(
        [s],
        convert_method(init, {}, [X(s)], theory=LIA),
        convert_method(update, {"old_x": s}, [X(s)], theory=LIA),
    )
```

Generated with no property at all, `reactor` prints the module it read and the
files it wrote. This is the one run shown in full:

```bash
uv run reactor tests/limits/mods/m_countdown.py -o "$OUT" -p "$PROJ"
```

```
module
  interface
    #0 : Int([1,1])
  atom controls #0 reads #0
  init
    #3 := [[100]] 
    X(#0) := Id #3
  delay
    d(#0) := ZERO 
  update
    #4 := [[0]] 
    #5 := Eq (#0, #4)
    #6 := [[100]] 
    #7 := [[1]] 
    #8 := Sub (#0, #7)
    #9 := Ite (#5, #6, #8)
    X(#0) := Id #9

.. Generating lean code
Created project directory: `/tmp/verith-tutorial.noindex/Rea`
Wrote system.txt
Wrote system.py
Wrote /tmp/verith-tutorial.noindex/Rea/lakefile.toml
Wrote /tmp/verith-tutorial.noindex/Rea/lean-toolchain
Wrote root module /tmp/verith-tutorial.noindex/Rea/System.lean
++ Generated /tmp/verith-tutorial.noindex/Rea/System/System.lean ++
++ Generated /tmp/verith-tutorial.noindex/Rea/System/Circ.lean ++
++ Generated /tmp/verith-tutorial.noindex/Rea/System/Rel.lean ++
++ Generated /tmp/verith-tutorial.noindex/Rea/System/Scalar.lean ++
++ Generated /tmp/verith-tutorial.noindex/Rea/System/ScalarRel.lean ++
Wrote /tmp/verith-tutorial.noindex/Rea/Certificate/Data.lean
Wrote /tmp/verith-tutorial.noindex/Rea/Certificate/Module.lean
Wrote /tmp/verith-tutorial.noindex/Rea/Certificate/Proofs.lean
Wrote /tmp/verith-tutorial.noindex/Rea/Certificate/Certificate.lean

DONE: Project created at: /tmp/verith-tutorial.noindex/Rea

Project ready at: /tmp/verith-tutorial.noindex/Rea
```

The module dump is the SSA term list: `#0` is the one state wire, `init`
writes `100` into its next value, and `update` computes
`ite (#0 = 0, 100, #0 - 1)`. Everything under `Wrote` is the Lean package:

```bash
find "$OUT/$PROJ" -maxdepth 2 -not -path '*/.lake*' -not -path '*/artifacts*' | sort
```

```
.
./Certificate
./Certificate.lean
./Certificate/Certificate.lean
./Certificate/Data.lean
./Certificate/Module.lean
./Certificate/Proofs.lean
./dbg
./dbg/system.py
./dbg/system.txt
./lakefile.toml
./lean-toolchain
./System
./System.lean
./System/Circ.lean
./System/Rel.lean
./System/Scalar.lean
./System/ScalarRel.lean
./System/System.lean
```

With no property the certificate is a skeleton. The safety property is that
the counter never leaves `[0, 100]`:

```
--safety '(and (>= (X @intf.0) 0) (<= (X @intf.0) 100))'
```

`s0` is the module's one controlled wire. Under `rule_globally` the whole
certificate is an **invariant** that holds initially, is preserved by `update`,
and is strong enough to imply the property — here the property is itself
inductive, so the same formula serves as both.

Given the property and the invariant, `reactor` writes the certificate's
definitions and states its obligations — and proves none of them:

```bash
uv run reactor tests/limits/mods/m_countdown.py \
    --safety    '(and (>= (X @intf.0) 0) (<= (X @intf.0) 100))' \
    --invariant '(and (>= (X @intf.0) 0) (<= (X @intf.0) 100))' \
    -o "$OUT" -p "$PROJ" | says
grep -A1 '^theorem' "$OUT/$PROJ/Certificate/Proofs.lean"
```

```
Project ready at: /tmp/verith-tutorial.noindex/Rea
theorem init_inv_proof : ∀ s, RM.init_pre s → inv (RM.init s) := by
  sorry
--
theorem step_inv_proof : ∀ s e, (RM.update_pre e ∧ inv s) → inv (RM.update s e) := by
  sorry
--
theorem inv_imp_P_proof : ∀ s, inv s → P s := by
  sorry
```

The proofs are `verith`'s. Given the same module, claim and certificate, and
the project, it writes them into `Proofs.lean`. With `--pre-check cvc5` it
first asks cvc5 whether the obligations are actually true. cvc5 answers in
milliseconds and a refutation comes with a counterexample; a failing `lake
build` takes tens of seconds and still cannot tell a wrong certificate from
tactics that are merely too weak.

```bash
uv run verith tests/limits/mods/m_countdown.py \
    --safety    '(and (>= (X @intf.0) 0) (<= (X @intf.0) 100))' \
    --invariant '(and (>= (X @intf.0) 0) (<= (X @intf.0) 100))' \
    --project "$OUT/$PROJ" --pre-check cvc5 | says
```

```
.. SMT pre-check (cvc5): <=5000 ms per query, <=20000 ms total
   init_inv  holds         2 ms
   step_inv  holds         2 ms
   inv_imp_P holds         2 ms
```

Three obligations, all discharged by cvc5:

* `init_inv` — every initial state satisfies the invariant
* `step_inv` — `update` preserves it, i.e. it is inductive
* `inv_imp_P` — the invariant implies the property

The SMT-LIB predicates are compiled into Lean in the certificate's
`Data.lean`. `s0` becomes `s 0 0`, the single entry of a `Mat Int 1 1`; a
module with several wires gets a product, and `s0, s1, s2, …` become `s.1`,
`s.2.1`, `s.2.2.1`, ….

```bash
cat "$OUT/$PROJ"/*/Data.lean
```

```lean
import Relational.Basic

set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unusedTactic false
set_option linter.unreachableTactic false
set_option linter.unnecessarySeqFocus false

def init_pre (e : (Unit) × (Unit)) : Prop := True

def update_pre (e : (Unit) × (Unit)) : Prop := True

def inv : (Mat Int 1 1) → Prop := fun s => (((s 0 0) ≥ (0 : Int)) ∧ ((s 0 0) ≤ (100 : Int)))

def P : (Mat Int 1 1) → Prop := fun s => (((s 0 0) ≥ (0 : Int)) ∧ ((s 0 0) ≤ (100 : Int)))

instance : DecidablePred P := fun s => by unfold P; first | infer_instance | dsimp; infer_instance
```

`Certificate/Certificate.lean` is what Lean checks: the three obligations
as theorems, and the `rule_globally` application that turns them into the
property. It is the same text for every safety certificate, and
`Module.lean`, the module as a `ReactiveModule` and its transition system
`lts`, for every certificate. The proofs are in `Proofs.lean`, which states
each obligation again with `_proof` after its name. `reactor` proved each by
`sorry`; `verith` wrote it again, proving each with tactics generated from
the shape of the module and the certificate, and its header comment records
that shape. `Certificate.lean` takes each proof by `exact`, which checks
that the two statements agree.

```bash
grep -E '^-- |^theorem ' "$OUT/$PROJ/Certificate/Proofs.lean"
grep -E '^theorem |exact .*_proof|^def ' "$OUT/$PROJ/Certificate/Certificate.lean"
```

```lean
-- state: Bool+Int, 1 slot(s), 9 term(s)
-- predicates: branching, conjunctive, linear
-- Unfold the definitions and reduce the matrix arithmetic, on the goal
-- only: on the hypotheses too it blows up exponentially.
-- Unfold the definitions everywhere (cheap: no matrix arithmetic), so the
-- closers see the invariant's conditions.
theorem init_inv_proof : ∀ s, RM.init_pre s → inv (RM.init s) := by
theorem step_inv_proof : ∀ s e, (RM.update_pre e ∧ inv s) → inv (RM.update s e) := by
theorem inv_imp_P_proof : ∀ s, inv s → P s := by
theorem init_inv : ∀ s, RM.init_pre s → inv (RM.init s) := by
  exact init_inv_proof
theorem step_inv : ∀ s e, (RM.update_pre e ∧ inv s) → inv (RM.update s e) := by
  exact step_inv_proof
theorem hinv' : lts.StateSet_isInductiveInitial inv := by
theorem hinv : lts.StateSet_isInvariant inv := by
theorem inv_imp_P : ∀ s, inv s → P s := by
  exact inv_imp_P_proof
theorem hP : lts.StateSet_isInvariant P := by
def safety := rule_globally
```

```bash
lake_cert
```

```
Note: This linter can be disabled with `set_option linter.defProp false`
✔ [3485/3486] Built Certificate (6.0s)
Build completed successfully (3486 jobs).
```

---

## 3. Büchi: the same idea, plus a ranking function

`tests/fixtures/counter.py` counts *up*: 0, 1, …, 9, then back to 0. "The
counter is at zero infinitely often" is a liveness property, `G (F (s0 = 0))`,
and it goes under `--buchi`.

An invariant alone cannot prove it — an invariant says where the system *may*
be, never that it gets anywhere. `rule_buchi` therefore also takes a **ranking
function**: a natural number, non-negative under the invariant, that strictly
decreases on every step taken from a state where the property is false. Here
`10 - s0` counts the steps left until the wrap, and the `ite` pins it to `0`
where the property already holds.

From here on one command does both steps: `certify` generates the project as
`reactor` does and proves it as `verith` does, in one run.

```bash
uv run certify tests/fixtures/counter.py \
    --buchi     '(= (X @intf.0) 0)' \
    --invariant '(and (>= (X @intf.0) 0) (<= (X @intf.0) 9))' \
    --ranking   '(ite (= (X @intf.0) 0) 0 (- 10 (X @intf.0)))' \
    --pre-check cvc5 -o "$OUT" -p "$PROJ" | says
```

```
.. SMT pre-check (cvc5): <=5000 ms per query, <=20000 ms total
   init_inv  holds         2 ms
   step_inv  holds         2 ms
   hrank     holds         3 ms
Project ready at: /tmp/verith-tutorial.noindex/Rea
```

The obligations are now `init_inv`, `step_inv` and **`hrank`**, the decrease.
`inv_imp_P` is gone: a Büchi property is not implied by the invariant, it is
*reached*.

`Data.lean` gains a `ranking` definition, `Proofs.lean` proves `hrank`
where it proved `inv_imp_P`, and the certificate ends in `rule_buchi` rather
than `rule_globally`:

```bash
grep -E 'def (inv|P|ranking)' "$OUT/$PROJ"/*/Data.lean
grep -E '^theorem |exact .*_proof|^def ' "$OUT/$PROJ/Certificate/Certificate.lean"
```

```lean
def inv : (Mat Int 1 1) → Prop := fun s => (((s 0 0) ≥ (0 : Int)) ∧ ((s 0 0) ≤ (9 : Int)))
def P : (Mat Int 1 1) → Prop := fun s => ((s 0 0) = (0 : Int))
def ranking : (Mat Int 1 1) → Nat := fun s => (((if ((s 0 0) = (0 : Int)) then (0 : Int) else ((10 : Int) - (s 0 0))) : Int)).toNat
theorem init_inv : ∀ s, RM.init_pre s → inv (RM.init s) := by
  exact init_inv_proof
theorem step_inv : ∀ s e, (RM.update_pre e ∧ inv s) → inv (RM.update s e) := by
  exact step_inv_proof
theorem hinv' : lts.StateSet_isInductiveInitial inv := by
theorem hinv : lts.StateSet_isInvariant inv := by
theorem hrank : ∀ s s', (inv s ∧ ¬(P s) ∧ (∃ l, lts.Tr s l s')) →
  exact hrank_proof
def buchi := rule_buchi
```

```bash
lake_cert
```

```
Note: This linter can be disabled with `set_option linter.defProp false`
✔ [3485/3486] Built Certificate (6.0s)
Build completed successfully (3486 jobs).
```

---

## 4. A module read out of a gymnasium environment

Nothing above is specific to hand-written modules. `reactor.gym.Env` extracts a
module from a plain `gymnasium.Env` by symbolically analyzing its `reset` and
`step` methods, and the result is a module file like any other.
`tests/fixtures/simple_env.py` wraps `SimpleEnv` — a three-cell chain where
the agent walks left or right and the goal is cell 2:

```bash
cat tests/fixtures/simple_env.py
sed -n '/class SimpleEnv/,/class TwoBitCounterEnv/p' tests/gym/environments.py
```

```python
"""SimpleEnv gym wrapper fixture.

Chain environment: state ∈ {0,1,2}, action moves left/right.
Property: state reaches 2 infinitely often.
"""
from reactor.gym import Env
from reactor import Module, Real
from tests.gym.environments import SimpleEnv


def module() -> Module:
    # `state` is SimpleEnv's only private attribute; its sort must be explicit.
    return Env(SimpleEnv(), attrs=Real([1, 1]))
```

`reset` becomes the module's `init` and `step` its `update`. Instance
attributes written by either (`self.state`) become **private** wires; the
observation, reward, `terminated` and `truncated` returns become **interface**
wires; the action becomes an **external** input.

### Which wire is `(X @intf.0)`?

`(X @intf.0) … (X @intf.N-1)` (`s0 … sN-1` internally) are the module's controlled wires in order, and `reactor` prints
that order — the `controls` line of the module dump. Generate the project
bare, with no property, and read it (the same dump is written to
`dbg/system.txt`, so a later run need not reprint it):

```bash
uv run reactor tests/fixtures/simple_env.py -o "$OUT" -p "$PROJ" | head -24
```

```
module
  external
    #3 : Real([1,2])
  interface
    #6 : Real([1,1])
    #9 : Real([1,1])
    #12 : Bool([1,1])
    #15 : Bool([1,1])
  private
    #0 : Real([1,1])
  atom controls #0, #6, #9, #12, #15 reads #0, #3
  init
    #18 := [[0]] 
    X(#0) := Id #18
    #19 := [[2]] 
    #20 := Eq (#18, #19)
    #21 := [[1]] 
    #22 := [[0]] 
    #23 := Ite (#20, #21, #22)
    X(#6) := Id #23
    X(#9) := [[0]] 
    X(#12) := [[false]] 
    X(#15) := [[false]] 
  delay
```


`atom controls #0, #6, #9, #12, #15` is the state order, so for this module:

| SMT var | wire | Lean type | in `Data.lean` |
|---|---|---|---|
| `s0` | `self.state` (private) | `Mat Real 1 1` | `s.1` |
| `s1` | observation | `Mat Real 1 1` | `s.2.1` |
| `s2` | reward | `Mat Real 1 1` | `s.2.2.1` |
| `s3` | terminated | `Mat Bool 1 1` | `s.2.2.2.1` |
| `s4` | truncated | `Mat Bool 1 1` | `s.2.2.2.2` |
| `e0` / `el0` | action, one-hot of width 2 | `Mat Real 1 2` | — |

Note that the arithmetic is **real**, not integer: under the default theory a
`Discrete` action space becomes a one-hot real vector (read back with `argmax`)
and rewards are real-valued. Hence `0.0` and `2.0` in the property below rather
than `0` and `2`.

### A safety certificate for the chain

"The chain position never leaves `[0, 2]`" — true whatever the agent does,
since `step` clamps with `min` and `max`:

```bash
uv run certify tests/fixtures/simple_env.py \
    --safety    '(and (>= (X @intf.0) 0.0) (<= (X @intf.0) 2.0))' \
    --invariant '(and (>= (X @intf.0) 0.0) (<= (X @intf.0) 2.0))' \
    --pre-check cvc5 -o "$OUT" -p "$PROJ" | says

grep -E 'def (inv|P)' "$OUT/$PROJ"/*/Data.lean
```

```
.. SMT pre-check (cvc5): <=5000 ms per query, <=20000 ms total
   init_inv  holds         2 ms
   step_inv  holds         3 ms
   inv_imp_P holds         2 ms
Project ready at: /tmp/verith-tutorial.noindex/Rea
```

```lean
def inv : (Mat Real 1 1) × (Mat Real 1 1) × (Mat Real 1 1) × (Mat Bool 1 1) × (Mat Bool 1 1) → Prop := fun s => (((s.1 0 0) ≥ (0 : Real)) ∧ ((s.1 0 0) ≤ (2 : Real)))
def P : (Mat Real 1 1) × (Mat Real 1 1) × (Mat Real 1 1) × (Mat Bool 1 1) × (Mat Bool 1 1) → Prop := fun s => (((s.1 0 0) ≥ (0 : Real)) ∧ ((s.1 0 0) ≤ (2 : Real)))
```

Two things worth knowing about the real-valued case before reaching for a
Büchi property here:

* Every Lean definition becomes `noncomputable`, and the proofs lean on
  `linarith` rather than the more robust `omega`.
* An *interval* invariant is not inductive enough for a ranking argument over
  the reals. `(and (>= s0 0.0) (<= s0 2.0))` admits `s0 = 5/4`, and cvc5 duly
  refutes `hrank` there. Pinning the reachable values instead —
  `(or (= s0 0.0) (= s0 1.0) (= s0 2.0))` — makes cvc5 accept all three
  obligations, but the generated tactic cascade still does not close `hrank` on
  this module: `argmax` over real wires is a known gap.

If the dynamics are genuinely discrete, design the environment so the
extraction stays in integer arithmetic: pass `theory=LIA` to `Env`, use an
integer `Box` action space rather than `Discrete`, and return integer rewards.
[`verith_gym.md`](verith_gym.md) works that variant through in full.

---

## 5. When verification fails

A certificate can be wrong in two ways, and `--pre-check cvc5` tells them
apart from "the tactics were too weak" before any Lean runs.

### 5a. An invalid invariant

Take the counter of section 3 and claim it stays below 5. The property is
still true and the ranking function is still fine; the invariant is not
inductive, because the counter runs to 9:

```bash
uv run certify tests/fixtures/counter.py \
    --buchi     '(= (X @intf.0) 0)' \
    --invariant '(and (>= (X @intf.0) 0) (<= (X @intf.0) 5))' \
    --ranking   '(ite (= (X @intf.0) 0) 0 (- 10 (X @intf.0)))' \
    --pre-check cvc5 -o "$OUT" -p "$PROJ" | says
```

```
.. SMT pre-check (cvc5): <=5000 ms per query, <=20000 ms total
   init_inv  holds         2 ms
   step_inv  REFUTED       2 ms   counterexample: s0 = 5
   hrank     holds         3 ms
   1 obligation(s) refuted (step_inv): the certificate is wrong, not merely hard -- Lean cannot close it
Project ready at: /tmp/verith-tutorial.noindex/Rea
```

`step_inv REFUTED … counterexample: s0 = 5` is the whole diagnosis: from
`s0 = 5`, which the invariant allows, `update` steps to 6, which it does not.
The project is still generated — `certify` does not refuse to write a
certificate it believes is false — but `lake build` would spend tens of
seconds failing to prove it.

### 5b. A missing precondition

`tests/limits/mods/m_relu_input.py` decrements by an *input*: the environment
chooses how much to subtract, and the module clamps at zero. Nothing about the
module alone makes the counter go down — an environment that offers `0` every
round leaves it where it is:

```bash
uv run certify tests/limits/mods/m_relu_input.py \
    --buchi     '(= (X @intf.0) 0)' \
    --invariant '(and (>= (X @intf.0) 0) (<= (X @intf.0) 5))' \
    --ranking   '(X @intf.0)' \
    --pre-check cvc5 -o "$OUT" -p "$PROJ" | says
```

```
.. SMT pre-check (cvc5): <=5000 ms per query, <=20000 ms total
   init_inv  holds         2 ms
   step_inv  REFUTED       3 ms   counterexample: s0 = 0
   hrank     REFUTED       3 ms   counterexample: s0 = 1
   2 obligation(s) refuted (step_inv, hrank): the certificate is wrong, not merely hard -- Lean cannot close it
Project ready at: /tmp/verith-tutorial.noindex/Rea
```

`--pre` supplies the missing hypothesis as a precondition over the input
wires. It is added to `init_pre` and `update_pre` alike, and every obligation
may then assume it:

```bash
uv run certify tests/limits/mods/m_relu_input.py \
    --buchi     '(= (X @intf.0) 0)' \
    --invariant '(and (>= (X @intf.0) 0) (<= (X @intf.0) 5))' \
    --ranking   '(X @intf.0)' \
    --pre       '(>= (X @extl.0) 1)' \
    --pre-check cvc5 -o "$OUT" -p "$PROJ" | says

grep -E 'def (init_pre|update_pre)' "$OUT/$PROJ"/*/Data.lean
```

```
.. SMT pre-check (cvc5): <=5000 ms per query, <=20000 ms total
   init_inv  holds         3 ms
   step_inv  holds         3 ms
   hrank     holds         3 ms
Project ready at: /tmp/verith-tutorial.noindex/Rea
```

```lean
def init_pre : ((Mat Int 1 1)) × ((Mat Int 1 1)) → Prop := fun e => ((e.2 0 0) ≥ (1 : Int))
def update_pre : ((Mat Int 1 1)) × ((Mat Int 1 1)) → Prop := fun e => ((e.2 0 0) ≥ (1 : Int))
```

A precondition is a promise about the environment, and the certificate is only
as true as that promise. It belongs in the specification, not in the tactics.

---

## 6. Letting an LLM find the certificate (`--infer`)

Everything so far supplied the certificate by hand. `--infer` searches for it.
The default route is `ai-cegis`: an LLM proposes an invariant and a ranking
function as SMT-LIB, cvc5 puts them to the obligations, and a refuted
candidate goes back to the model *with its counterexample* — a CEGIS loop
whose checker is a decision procedure, so a wrong proposal costs one round
trip and never reaches Lean.

It needs `ANTHROPIC_API_KEY` and `uv sync --extra ai` for the default backend.

```bash
uv run certify tests/limits/mods/m_countdown.py \
    --buchi '(= (X @intf.0) 0)' \
    --infer -o "$OUT" -p "$PROJ" | says            # no --invariant, no --ranking

grep -E 'def (inv|ranking)' "$OUT/$PROJ"/*/Data.lean
```

```
[CEGIS] attempt 0
  inv: (and (<= 0 s0) (<= s0 100))
  ranking: s0
[CEGIS] all obligations UNSAT — accepted
Project ready at: /tmp/verith-tutorial.noindex/Rea
```

```lean
def inv : (Mat Int 1 1) → Prop := fun s => ((0 ≤ (s 0 0)) ∧ ((s 0 0) ≤ 100))
def ranking : (Mat Int 1 1) → Nat := fun s => (((s 0 0) : Int)).toNat
```

```bash
lake_cert
```

```
Note: This linter can be disabled with `set_option linter.unusedSimpArgs false`
✔ [3465/3466] Built Certificate (2.7s)
Build completed successfully (3466 jobs).
```

The inferred predicates are printed as SMT-LIB, so a good result can be frozen
by passing it back with `--invariant` / `--ranking` on later runs. The two
modes mix: fixing the invariant by hand makes `--infer` search only for a
ranking function, and fixing **both** makes no LLM call at all — the CEGIS
loop just verifies your certificate with cvc5, which is `--pre-check` by
another route.

For a local model or another provider, add `--base-url` (and `--model`); the
only extra dependency is the `openai` client, `uv sync --extra ai-local`:

```bash
uv run certify tests/limits/mods/m_countdown.py --buchi '(= (X @intf.0) 0)' --infer \
    --model qwen3-coder --base-url http://localhost:11434/v1 \
    -o "$OUT" -p "$PROJ"
```

---

## 7. Learning the certificate instead (`--infer nuterm`)

`--infer nuterm` answers the same question with no LLM in it, and the two
halves of the certificate come from two different places.

The **invariant** is Houdini's: seed a lattice of candidate facts about the
state — signs, pairwise relations, and the constants the program itself
mentions — drop the ones that do not hold at entry or are not preserved by a
step, and certify what survives. The **ranking function** is *learned*: a small
ReLU network is trained on rollouts of the rounds where the property fails, its
weights are rounded to integers, and the candidate is composed back into the
module as two ordinary atoms, `V(s)` and `V(s')`. That composed module goes to
a Farkas/CEGAR decision procedure over its own wires, which certifies the drop
or rejects the candidate and sends the trainer round again.

Only a candidate the procedure certifies comes back. So unlike the LLM routes,
what `certify` is handed here has *already* been proved, and `--pre-check` is an
independent confirmation rather than the first check of it. It needs no key, no
network and no ic3ia build, and it is deterministic: the same module and the
same seed learn the same rank. What it does need is a state of scalar integers
and a transition that reads no nondeterministic input — anything else it
refuses by name rather than approximating.

```bash
uv run certify tests/limits/mods/m_countdown.py \
    --buchi '(= (X @intf.0) 0)' \
    --infer nuterm \
    --pre-check cvc5 -o "$OUT" -p "$PROJ" | says
```

```
[nuterm] columns: s0
[nuterm] training a ranking function that drops on every round the property fails
[nuterm] 991 sampled rounds, loss 0 -- certified
[nuterm] invariant: 2 of 3 conjuncts after pruning
[nuterm] inv: (and (>= s0 0) (<= s0 100))
[nuterm] ranking: (+ (* 2 (ite (> (+ 2 s0) 0) (+ 2 s0) 0)) (* 2 (ite (> (* 2 s0) 0) (* 2 s0) 0)) (* 2 (ite (> (* (- 2) s0) 0) (* (- 2) s0) 0)) (* 2 (ite (> (+ (- 1) (- s0)) 0) (+ (- 1) (- s0)) 0)) (* 2 (ite (> (- s0) 0) (- s0) 0)) (* 2 (ite (> (+ (- 1) s0) 0) (+ (- 1) s0) 0)) (* 2 (ite (> (- 1) 0) (- 1) 0)))
.. SMT pre-check (cvc5): <=5000 ms per query, <=20000 ms total
   init_inv  holds         2 ms
   step_inv  holds         2 ms
   hrank     holds         3 ms
Project ready at: /tmp/verith-tutorial.noindex/Rea
```

`[nuterm] inv:` and `[nuterm] ranking:` are the SMT-LIB the certificate is
written from — the invariant is the familiar `0 ≤ s0 ≤ 100`, and the ranking
function is the learned network written out, one `ite` per ReLU unit. It is
longer than the `s0` a human would write, and the three obligations underneath
it hold all the same.

```bash
lake_cert
```

```
Note: This linter can be disabled with `set_option linter.unusedSimpArgs false`
✔ [3465/3466] Built Certificate (2.7s)
Build completed successfully (3466 jobs).
```

### Neither route dominates

`m_step2` steps by two and resets:

```c
x = 0;
while (true) { x = (x == 10) ? 0 : x + 2; }
```

`x == 0` recurs, so `G (F (= s0 0))` holds. But the invariant that makes it
provable is that `x` is **even**, and Houdini's lattice of signs, pairwise
relations and program constants cannot say so. The route says that rather than
guessing:

```bash
uv run certify tests/limits/mods/m_step2.py \
    --buchi '(= (X @intf.0) 0)' --infer nuterm -o "$OUT" -p "$PROJ" | says
```

```
[nuterm] columns: s0
[nuterm] training a ranking function that drops on every round the property fails
[nuterm] 833 sampled rounds, loss 0.9688 -- trained but not verified
[nuterm] training a ranking function that drops until the property is reached
[nuterm] 667 sampled rounds, loss 0 -- trained but not verified
error: --infer: no ranking function was certified. The rank is a non-negative sum of ReLUs, so it is convex in the state: a program whose run wraps around needs one that falls along the ramp and again across the reset, which no convex rank does. It is learned from rollouts too, so a property that never fails on one leaves nothing to train on.
```

The LLM has no lattice, so a congruence costs it nothing to propose:

```bash
uv run certify tests/limits/mods/m_step2.py \
    --buchi '(= (X @intf.0) 0)' --infer ai-cegis --pre-check cvc5 -o "$OUT" -p "$PROJ" | says
```

```
[CEGIS] attempt 0
  inv: (and (<= 0 s0) (<= s0 10) (= (mod s0 2) 0))
  ranking: (ite (= s0 0) 0 (- 12 s0))
[CEGIS] all obligations UNSAT — accepted
.. SMT pre-check (cvc5): <=5000 ms per query, <=20000 ms total
   init_inv  holds         2 ms
   step_inv  holds         3 ms
   hrank     holds         3 ms
Project ready at: /tmp/verith-tutorial.noindex/Rea
```

`(= (mod s0 2) 0)` — the model proposes the congruence directly, and cvc5
confirms it. The route is non-deterministic, so re-running this will not
reproduce that attempt exactly; what is stable is that it gets there, and that
every attempt costs a round trip.

The learner also answers offline and deterministically, which is what lets it
run over a whole corpus: it proves 45 of the 57 SV-COMP termination benchmarks,
and 29 of the 32 safety properties the ic3ia route is meant to certify. What it
cannot do is invent a fact outside its candidate lattice.

So the order that makes sense is the order of cost: `--infer nuterm` first, and
the LLM for what it leaves behind. Section 8 changes that order again, because
the routes there can *prove* that a shape is empty — and hand the proof to the
LLM.

---

## 8. Searching a shape instead of proposing one (`--infer smt-linear`, `--infer sygus`)

Sections 6 and 7 both *propose*: the model proposes a certificate and cvc5
checks it, or the learner proposes a rank and a decision procedure certifies
it. Two more routes do neither. They fix the **shape** the certificate may
have and hand the whole space to cvc5 at once:

* **`--infer smt-linear`** fixes the arithmetic and leaves the coefficients
  open, so the search is one query — `exists c. forall s. obligations(c · s)`.
  `--buchi` asks it for a ranking function `c0 + c1*s0 + …` over a fixed
  invariant; `--safety` asks it for the invariant itself, as a conjunction of
  `a0 + a1*s0 + … >= 0` rows.
* **`--infer sygus`** (`--safety` only) fixes a **grammar** instead, and hands
  cvc5's SyGuS invariant track the three formulas `G P` is made of:
  `pre → inv`, `inv ∧ trans → inv'`, `inv → post`.

Neither needs a key, a network or a learner, and both are deterministic. But
the reason to reach for them is not that they are cheap — it is what they can
say when they **fail**, which is 8a, and what the next route does with that,
which is 8b.

```bash
uv run certify tests/limits/mods/m_countdown.py \
    --safety '(<= (X @intf.0) 100)' \
    --infer smt-linear \
    --artifacts reset \
    --pre-check cvc5 -o "$OUT" -p "$PROJ" | says
```

```
.. artifacts: cleared 10 file(s) from /tmp/verith-tutorial.noindex/Rea/artifacts
[smt-linear] columns: s0
[smt-linear] inv: (>= (+ 100 (- s0)) 0)
.. SMT pre-check (cvc5): <=5000 ms per query, <=20000 ms total
   init_inv  holds         1 ms
   step_inv  holds         1 ms
   inv_imp_P holds         1 ms
Project ready at: /tmp/verith-tutorial.noindex/Rea
```

One query, and the answer is the coefficients of a single row. `100 - s0 ≥ 0`
is the invariant a person would write, and it is written that way rather than
as the `200 - 2*s0 ≥ 0` the solver first handed over: the row is divided
through by its gcd, which is the same predicate and a smaller one to restate
in Lean.

The widths are tried one at a time — one row, then two, up to `--linear-rows`
— because the width is where the cost is. One row here is a few milliseconds
of solving; asking for two rows at once does not finish in thirty seconds,
because every extra row multiplies a nonlinear search. So the first width that
answers wins, and each `unsat` on the way is kept, which is 8a.

```bash
lake_cert
```

```
Note: This linter can be disabled with `set_option linter.unnecessarySeqFocus false`
✔ [3465/3466] Built Certificate (2.7s)
Build completed successfully (3466 jobs).
```

### 8a. The answer that is worth more than a certificate

Ask for something that is not there. `m_toward5` walks toward 5 from both
sides, so a rank has to *branch* — and no linear function branches:

```bash
uv run certify tests/limits/mods/m_toward5.py \
    --buchi     '(= (X @intf.0) 5)' \
    --invariant '(and (>= (X @intf.0) 0) (<= (X @intf.0) 10))' \
    --infer smt-linear -o "$OUT" -p "$PROJ" | says
```

```
[smt-linear] columns: s0
error: --infer: --infer smt-linear found no ranking function. No ranking function linear in the state -- `c0 + c1*s0`, integer coefficients, no branching -- drops by at least one and stays positive wherever the property fails, under the invariant `(and (>= s0 0) (<= s0 10))`. This is a proof that the space is empty, not a search that ran out of time: the query is `exists c. forall s. obligations(c)` over the coefficients, and cvc5 came back unsat. A ranking function for this module needs something outside that shape: a branch (`ite`), which is what a program whose run wraps around needs, or a stronger invariant to rank over. `--infer ai-cegis` and `--infer nuterm` both search shapes that have one.
```

That is not "the search gave up". `exists c. forall s. …` came back **unsat**,
which is a proof that *nothing* of that shape satisfies the obligations — a
fact about the module, established by a decision procedure, in a couple of
milliseconds. An LLM asked the same question would spend a round trip per
attempt discovering it, and could not prove it at the end.

So it is written down. `artifacts/` is the project's workspace across runs
(`--artifacts use|ignore|reset`), and a refuted shape goes in it as a note
whose status is `no_solution` — kept apart from `unknown`, which is what a
search that merely ran out of budget leaves:

```bash
cat "$(ls -t "$OUT/$PROJ"/artifacts/note-*.md | head -1)"
```

```
No ranking function linear in the state -- `c0 + c1*s0`, integer coefficients, no branching -- drops by at least one and stays positive wherever the property fails, under the invariant `(and (>= s0 0) (<= s0 10))`. This is a proof that the space is empty, not a search that ran out of time: the query is `exists c. forall s. obligations(c)` over the coefficients, and cvc5 came back unsat. A ranking function for this module needs something outside that shape: a branch (`ite`), which is what a program whose run wraps around needs, or a stronger invariant to rank over. `--infer ai-cegis` and `--infer nuterm` both search shapes that have one.
```

### 8b. Cascading: what one route rules out, the next one takes up

Here is the pattern the workspace is for. `m_step2` from section 7 needs `x`
even, and no conjunction of linear inequalities can say that either.
`--infer smt-linear` does not merely fail to find one; it **proves** there is
none, at every width up to `--linear-rows`:

```bash
uv run certify tests/limits/mods/m_step2.py \
    --safety '(not (= (X @intf.0) 1))' \
    --infer smt-linear --artifacts reset -o "$OUT" -p "$PROJ" | says
```

```
.. artifacts: cleared 10 file(s) from /tmp/verith-tutorial.noindex/Rea/artifacts
[smt-linear] columns: s0
[smt-linear] no invariant with 1 row(s); widening
[smt-linear] no invariant with 2 row(s); widening
error: --infer: --infer smt-linear found no invariant. No inductive invariant implying the property is a conjunction of 2 linear inequalities `a0 + a1*s0 >= 0`. This is a proof that the space is empty, not a search that ran out of time: the query is `exists c. forall s. obligations(c)` over the coefficients, and cvc5 came back unsat. Wider conjunctions were not tried: `--linear-rows` stopped at 2. An invariant may exist with more rows, or outside linear arithmetic altogether -- `--infer sygus` adds congruences (`x` even), which no conjunction of inequalities can state.
```

Now the second route, on the same module and the same property, into the same
project. Its grammar carries congruences — `(= (mod a0 + a1*s0 + … k) 0)`, for
the `k` the program itself mentions — which is exactly the atom the first
route proved it did not have:

```bash
uv run certify tests/limits/mods/m_step2.py \
    --safety '(not (= (X @intf.0) 1))' \
    --infer sygus --pre-check cvc5 -o "$OUT" -p "$PROJ" | says

lake_cert
```

```
[sygus] columns: s0
[sygus] grammar: congruence, constants [-11, -10, -9, -3, -2, -1, 0, 1, 2, 3, 9, 10, 11]
[sygus] inv: (= (mod (+ 2 (* 11 s0)) 2) 0)
.. SMT pre-check (cvc5): <=5000 ms per query, <=20000 ms total
   init_inv  holds         1 ms
   step_inv  holds         2 ms
   inv_imp_P holds         1 ms
Project ready at: /tmp/verith-tutorial.noindex/Rea
```

```
Note: This linter can be disabled with `set_option linter.unnecessarySeqFocus false`
✔ [3465/3466] Built Certificate (2.7s)
Build completed successfully (3466 jobs).
```

That is the cascade: the cheap route could not close the obligations, said so
in a form the next run can read, and the route with the missing atom closed
them. Both halves are in the workspace — what was ruled out, and what was
found:

```bash
uv run python -c "
import json, sys
for e in sorted(json.load(open(sys.argv[1])), key=lambda e: e['seq']):
    if e['role'] in ('inv', 'ranking', 'note'):
        print(f\"{e['role']:8s} {e['status']:12s} {e['producer']:11s} {e['name']}\")
" "$OUT/$PROJ/artifacts/index.json"
```

```
note     no_solution  smt-linear  note-0002-smt-linear.md
inv      proved       sygus       inv-0003-sygus.smt2
inv      encoded      sygus       inv.smt
```

The `inv` row is not only a record. It is written in SMT-LIB with the status a
later run filters on, so the next `certify` in this project **takes it as
given** — and for a safety property that is the whole certificate, so the run
below closes it without an LLM call even though `ai-cegis` is an LLM route:

```bash
uv run certify tests/limits/mods/m_step2.py \
    --safety '(not (= (X @intf.0) 1))' \
    --infer ai-cegis --pre-check cvc5 -o "$OUT" -p "$PROJ" | says
```

```
.. resuming from inv-0003-sygus.smt2 (proved): taking it as the certificate
.. resuming from note-0002-smt-linear.md: a space an earlier run ruled out
[CEGIS] attempt 0
  inv: (= (mod (+ 2 (* 11 s0)) 2) 0)
[CEGIS] all obligations UNSAT — accepted
.. SMT pre-check (cvc5): <=5000 ms per query, <=20000 ms total
   init_inv  holds         1 ms
   step_inv  holds         2 ms
   inv_imp_P holds         1 ms
Project ready at: /tmp/verith-tutorial.noindex/Rea
```

`.. resuming from inv-…-sygus.smt2 (proved)` is the handoff, and
`[CEGIS] all obligations UNSAT — accepted` is the loop finding it has nothing
left to ask. The store drops anything found for a different module, a
different property or a different proof rule before offering it, which is what
keeps a workspace from turning two identical command lines into two different
runs.

### 8c. The other direction: a refutation in an LLM's prompt

The same note that made 8a readable by a human is written to be read by a
model. `--infer ai-cegis` resumes `no_solution` notes and states them in its
prompt as established facts — so the attempt that would have gone on a linear
ranking function goes somewhere else instead:

```bash
uv run certify tests/limits/mods/m_toward5.py \
    --buchi     '(= (X @intf.0) 5)' \
    --invariant '(and (>= (X @intf.0) 0) (<= (X @intf.0) 10))' \
    --infer smt-linear --artifacts reset -o "$OUT" -p "$PROJ" | says

uv run certify tests/limits/mods/m_toward5.py \
    --buchi     '(= (X @intf.0) 5)' \
    --invariant '(and (>= (X @intf.0) 0) (<= (X @intf.0) 10))' \
    --infer ai-cegis --pre-check cvc5 -o "$OUT" -p "$PROJ" | says
```

```
.. artifacts: cleared 10 file(s) from /tmp/verith-tutorial.noindex/Rea/artifacts
[smt-linear] columns: s0
error: --infer: --infer smt-linear found no ranking function. No ranking function linear in the state -- `c0 + c1*s0`, integer coefficients, no branching -- drops by at least one and stays positive wherever the property fails, under the invariant `(and (>= s0 0) (<= s0 10))`. This is a proof that the space is empty, not a search that ran out of time: the query is `exists c. forall s. obligations(c)` over the coefficients, and cvc5 came back unsat. A ranking function for this module needs something outside that shape: a branch (`ite`), which is what a program whose run wraps around needs, or a stronger invariant to rank over. `--infer ai-cegis` and `--infer nuterm` both search shapes that have one.
```

```
.. resuming from note-0003-smt-linear.md: a space an earlier run ruled out
[CEGIS] attempt 0
  inv: (and (>= s0 0) (<= s0 10))
  ranking: (ite (>= s0 5) (- s0 5) (- 5 s0))
[CEGIS] all obligations UNSAT — accepted
.. SMT pre-check (cvc5): <=5000 ms per query, <=20000 ms total
   init_inv  holds         1 ms
   step_inv  holds         2 ms
   hrank     holds         3 ms
Project ready at: /tmp/verith-tutorial.noindex/Rea
```

Attempt **0**, and the model proposes a rank that branches, on a module where
the note in front of it had just proved that nothing without a branch can
work. The LLM route is non-deterministic and this will not reproduce that term
exactly, but what it is being asked has changed. It is no longer "find a
ranking function"; it is "find one that is not in this space", with the space
named and the proof cited.

### 8d. A state that is not integers

`--infer nuterm` reads scalar integers and refuses everything else by name.
The template route weighs **every** scalar component, whatever its sort: an
`Int` is itself, a `Bool` is `0`/`1` (`(ite s0 1 0)`, which is how a row says
`b` or `¬b`), and a bitvector is its unsigned value (`(ubv_to_int s0)`). One
integer template then covers a mixed state instead of one template per sort.

`m_boolint` is a Bool beside an Int — the flag chooses whether the counter
walks up or down:

```bash
uv run certify tests/limits/mods/m_boolint.py \
    --safety '(and (>= (X @intf.1) 0) (<= (X @intf.1) 5))' \
    --infer smt-linear --artifacts reset \
    --pre-check cvc5 -o "$OUT" -p "$PROJ" | says
```

```
.. artifacts: cleared 10 file(s) from /tmp/verith-tutorial.noindex/Rea/artifacts
[smt-linear] columns: (ite s0 1 0), s1
[smt-linear] no invariant with 1 row(s); widening
[smt-linear] inv: (and (>= (+ (ite s0 1 0) (* 3 s1)) 0) (>= (+ 5 (- s1)) 0))
.. SMT pre-check (cvc5): <=5000 ms per query, <=20000 ms total
   init_inv  holds         1 ms
   step_inv  holds         2 ms
   inv_imp_P holds         1 ms
Project ready at: /tmp/verith-tutorial.noindex/Rea
```

A bitvector costs one thing more. Put one in the formula and cvc5 answers the
quantified query `unknown (INCOMPLETE)` in about a millisecond, at every
width — not a timeout, a refusal to state it. So the route asks the same
question a second way, as a **counterexample-guided loop** whose two halves
are both quantifier-free: propose coefficients that fit the states seen so
far, verify the proposal against the whole transition, and make the
counterexample the next sample. An 8-bit counter to try it on:

```bash
cat > "$OUT/m_bvcount.py" <<'PY'
# BV 8-bit counter: 0, 1, ... 10, then back to 0.
import torch
from reactor import Module, Term, Wire, BitVec, BV, Var, X


def module() -> Module:
    s = Var(BitVec(8, [1, 1]))
    one, ten, zero = (Wire(BitVec(8, [1, 1])) for _ in range(3))
    at_ten = Wire(BitVec(1, [1, 1]))      # BV has no Bool wires: one bit
    bumped = Wire(BitVec(8, [1, 1]))
    return Module.sequential(
        [s],
        [Term(BV.Const(torch.tensor([[0]])), [X(s)])],
        [Term(BV.Const(torch.tensor([[1]])), [one]),
         Term(BV.Const(torch.tensor([[10]])), [ten]),
         Term(BV.Const(torch.tensor([[0]])), [zero]),
         Term(BV.Eq(), [at_ten], [s, ten]),
         Term(BV.Add(), [bumped], [s, one]),
         Term(BV.Ite(), [X(s)], [at_ten, zero, bumped])],
    )
PY

uv run certify "$OUT/m_bvcount.py" \
    --safety '(<= (ubv_to_int (X @intf.0)) 10)' \
    --infer smt-linear --artifacts reset \
    --pre-check cvc5 -o "$OUT" -p "$PROJ" | says

lake_cert
```

```
.. artifacts: cleared 9 file(s) from /tmp/verith-tutorial.noindex/Rea/artifacts
[smt-linear] columns: (ubv_to_int s0)
[smt-linear] cvc5 will not state the quantified query for this module; searching coefficients in [-24, 24] by counterexample instead
[smt-linear] inv: (>= (+ 10 (- (ubv_to_int s0))) 0)
.. SMT pre-check (cvc5): <=5000 ms per query, <=20000 ms total
   init_inv  holds         1 ms
   step_inv  holds         1 ms
   inv_imp_P holds         1 ms
Project ready at: /tmp/verith-tutorial.noindex/Rea
```

```
Note: This linter can be disabled with `set_option linter.unnecessarySeqFocus false`
✔ [3465/3466] Built Certificate (2.7s)
Build completed successfully (3466 jobs).
```

The loop needs a *finite* candidate space to terminate, so the coefficients
are bounded — read off the constants the program mentions — and a refutation
from it is a proof about that box rather than about every integer. The note
says which, because the difference is exactly what a later prompt must not be
told wrongly.

Where the four no-LLM routes sit, then:

| | finds | answers *no* | state it reads |
|---|---|---|---|
| `--infer nuterm` | Houdini's lattice + a learned rank | no — it refuses, without deciding the space | scalar `Int` |
| `--infer smt-linear` | coefficients of a fixed linear shape | **yes, as a proof** (bounded, when the loop answered) | scalar `Int`, `Bool`, `BitVec` |
| `--infer sygus` | anything its grammar generates, congruences included | **yes, as a proof** — the grammar is finite | scalar `Int` |
| `--infer fbk-proveit` | ic3ia's interpolants | no | whatever the NA encoding expresses |

The order that pays is still the order of cost — the cheap decisive routes
first, the LLM for what they leave behind — with one addition: run the routes
that can say *no* first, because what they rule out is what the expensive
route no longer has to try.

---

## 9. Letting ic3ia find the invariant (`--fbk-proveit`)

The last route hands the whole safety question to an external model checker.
`proveit.py`, from a [`lean-ltl-certifying`](../verith/tests/fbk/README.md)
checkout, runs `lean2vmt → ic3ia → vmt2lean`: the module is re-encoded as a VMT
transition system, ic3ia proves `G P` and emits an *invariant witness*, and the
witness comes back as a Lean certificate that `certify` installs into the
project. Nothing about the invariant is supplied or guessed — `--invariant` is
refused on this route.

It needs `ic3ia`, MathSAT's Python bindings and the checkout (section 1):

```bash
uv run certify tests/limits/mods/m_countdown.py \
    --safety '(and (>= (X @intf.0) 0) (<= (X @intf.0) 100))' \
    --fbk-proveit "$PROVEIT" --ic3ia "$IC3IA" -o "$OUT" -p "$PROJ" | says

ls "$OUT/$PROJ/Certificate" "$OUT/$PROJ/ProveIt"
```

```
    ✔ property is SAFE
    ✔ invariant witness: /var/folders/p2/6pl6scg57lb35sp3r_ncs69h0000gn/T/certify-ReaNA-_15wx6sa/ReaNA_witness.smt2
    ✔ certificate: /tmp/verith-tutorial.noindex/Rea/ProveIt/ReaCert.lean (3936 bytes)
.. proveit.py's certificate: 1 normalised out of Bool, 2 offered omega before smt
Project ready at: /tmp/verith-tutorial.noindex/Rea
```

```
/tmp/verith-tutorial.noindex/Rea/Certificate:
Certificate.lean
Data.lean
Equivalence.lean
Module.lean
Proofs.lean

/tmp/verith-tutorial.noindex/Rea/ProveIt:
ReaCert.lean
ReaNA.lean
```

Two proofs rather than one. `Certificate/Certificate.lean` is ic3ia's proof,
about the *NA-encoded model*; `Certificate/Equivalence.lean` is the proof that
the model **is** the module, which is what carries the result back to the
system you wrote. Without it a mistranslation anywhere in the SMT path would
be a machine-checked certificate about something else. `--fbk-equiv none`
skips it, and on a wide state that is the expensive half. ic3ia's
certificate proves its obligations itself, so `Proofs.lean` is left with
nothing to prove.

This project's lakefile requires the `lean-ltl-certifying` checkout, which the
warm manifest has never heard of, so the build has to resolve it with
`lake update` — the only command here that goes to the network:

```bash
cd "$OUT/$PROJ" && lake update && lake build Certificate
```

```
Using cache (Azure) from origin: leanprover-community/mathlib4
No files to download
Already decompressed 8010 file(s)
Note: This linter can be disabled with `set_option linter.unreachableTactic false`
✔ [3472/3473] Built Certificate (2.6s)
Build completed successfully (3473 jobs).
```

---

## 10. Flag reference

`reactor` takes the module's and the claim's flags (`-o` to `--pre`),
`--artifacts` and `--cert-file`. `verith` takes the module's and the claim's
but `-o`, `-p` and `-x`, and everything from `--pre-check` on but
`--cert-file` and `--hammer-file`, with `--project` and `--json` of its own.
`certify` takes them all but those two.

| flag | default | meaning |
|---|---|---|
| `-o` / `--output-dir` | `.` | where the project is created |
| `-p` / `--project-name` | `Rea` | Lean package name; the project is `<-o>/<-p>` |
| `-d` / `--module-def` | `module` | name of the callable in the module file |
| `-x` / `--executable` | off | also emit `Main.lean` and a `lean_exe` target |
| `--safety FORMULA` | — | `G FORMULA`, proved by an invariant alone |
| `--buchi FORMULA` | — | `G (F FORMULA)`, proved by invariant + ranking |
| `--invariant` | — | SMT-LIB 2 Bool over `(X @intf.N)` … |
| `--ranking` | — | SMT-LIB 2 Int over `(X @intf.N)` …; `--buchi` only |
| `--pre` | — | SMT-LIB 2 Bool over `(X @extl.N)` / `@extl.N`, added to `init_pre` and `update_pre` |
| `--project DIR` | — | `verith`: the project `reactor` generated, which the certificate is written into, proved |
| `--pre-check cvc5` | `none` | ask cvc5 whether the obligations hold, before the proofs are written |
| `--smt-tactics cvc5` | `none` | let cvc5 settle branch conditions, hint `nlinarith` and drop an unused precondition |
| `--smt-timeout`, `--smt-budget` | 5000, 20000 ms | per-query and per-phase limits for both of the above |
| `--infer [ai\|ai-cegis\|nuterm\|sygus\|smt-linear\|fbk-proveit]` | `ai-cegis` | which route finds the certificate: an LLM, (`nuterm`) a learned and certified ranking function, (`sygus`) an invariant synthesised over a grammar, (`smt-linear`) the coefficients of a fixed linear shape, or (`fbk-proveit`) ic3ia |
| `--model`, `--base-url` | `claude-sonnet-4-6`, — | which LLM, and which endpoint; rejected by a route that calls none |
| `--sygus-grammar`, `--sygus-conjuncts` | `congruence`, 3 | `--infer sygus`: what an atom may be, and how many of them — the bound is what makes the space finite, and so decidably empty |
| `--linear-rows N` | 2 | `--infer smt-linear --safety`: how many linear inequalities the invariant may conjoin; tried one width at a time |
| `--artifacts [use\|ignore\|reset]` | `use` | the project's `artifacts/` workspace: resume from what earlier runs left, ignore it, or empty it first |
| `--build-cert` | off | `lake update` + `lake build Certificate` once the proofs are written, failing on a surviving `sorry` |
| `--proveit-dir DIR` | — | `--infer fbk-proveit`: the `lean-ltl-certifying` checkout; `--safety` only. `--fbk-proveit DIR` is the older spelling of both flags at once |
| `--ic3ia PATH` | `$IC3IA`, then `PATH` | the ic3ia executable |
| `--fbk-simplify`, `--fbk-equiv` | `cvc5`, `lean` | run cvc5's rewriter on the NA model; emit the model-is-the-module proof |
| `--cert-file PATH` | — | write one standalone `.lean` file instead of a project |
| `--hammer-file PATH` | — | write `ZerothHammer.lean`, the tactics the proofs call, alone |

Related reading: [`verith_gym.md`](verith_gym.md) for a gymnasium environment
worked through in integer arithmetic, `python/reactor/lean/README.md` for
the pipeline's own documentation, and `verith/README.md` for the routes,
the proofs and the `verith` command.
