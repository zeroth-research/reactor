# `reactor/lean` — Lean 4 Certificate Generation

`uv run reactor` takes a reactive module and a property of it, and ends with a
Lean project in which `lake build` checks a proof of the property. The proof
rests on a certificate: an inductive invariant, and for a Büchi property
also a ranking function. `reactor` takes the certificate as given
(`--invariant`, `--ranking`); finding one is a decision procedure's job,
and those live in the [`verith/`](../../../verith/README.md) project beside this one: `verith`
infers a certificate and prints it, and `certify` is `reactor` and
`--infer` in one run. The pipeline does not trust a certificate, whoever
wrote it. It is written into the project, and Lean checks it.

`reactor.lean` imports nothing of `verith`: the routes are built on it.

## The pipeline, stage by stage

`pipeline.main` runs the stages in this order, all but stage 5; `certify`
runs all seven. Each one names the code that does it and the section of
this file (or of [`verith`'s](../../../verith/README.md)) that covers it in full.

### 1. Input (`inputs/`)

The input is a file, read by the adapter that `--format` names or that the
file's suffix picks:

| format | suffix | what it gives |
|---|---|---|
| `python` | `.py` | the `reactor.Module` that `module()` (or `-d NAME`) returns. Returning `inputs.Named(module, {"x": var})` instead adds names to write claims in. A Gymnasium environment comes in this way, through `reactor.gym` ([tutorial](../../../tutorials/verith_gym.md)) |
| `smv` | `.smv` | a NuSMV model with one `MODULE main`, read as NuSMV reads it; a model that NuSMV would read differently is refused. Its variable names are the claim's names, and its `INVARSPEC` / `LTLSPEC G F` is the claim when neither `--safety` nor `--buchi` is given (`--spec N` picks one of several). The picture does not show this input |

The adapter returns an `inputs.Problem`: the module, the source text, the
names, and the claims the input states. A new input format is a new
`InputAdapter` in `inputs/`.

### 2. The claim (`cli/`, `inputs/names.py`)

The property goes under the flag of its proof rule:

- `--safety P` claims `G P`, that `P` holds in every reachable state. The
  certificate is an invariant that implies `P`.
- `--buchi P` claims `G F P`, that `P` holds infinitely often. The
  certificate is an invariant and a ranking function.

`P` is SMT-LIB written in the input's names, and so are `--pre`,
`--invariant` and `--ranking`. For example, `(X x)` is `x` after the round
and `(X @intf.N)` is interface variable N by position. See [Writing a claim
on the command line](#writing-a-claim-on-the-command-line) for the full
spelling. This stage translates the names to the internal `s{i}`, `e{i}`
and `el{i}`, and everything after it uses those. They are translated back
when a certificate is shown.

### 3. Reactive module (`reactor.Module`)

The module is the IR: `init` and `update` in SSA form over typed matrix
wires ([Reactive Module IR](#reactive-module-ir)). Two checks run before any
Lean is written:

- under `certify`, the route's `precheck` refuses a module that its
  procedure cannot read;
- `lean_gaps` refuses an operator with no Lean form.

### 4. Lean project (`project/`, `codegen/encodings/`, `templates/`)

`create_project` writes the project to `-o DIR -p NAME`:

```
NAME/
  System.lean, System/          the module in five encodings: functional,
                                circuit, relational, scalar and
                                scalar-relational, each tied to the
                                functional one by equivalence theorems
  Certificate/Data.lean         the certificate: init_pre, update_pre, P,
                                inv, ranking
  Certificate/Certificate.lean  the proof rule's obligations, and the
                                tactics that close them
  ZerothHammer.lean             the zeroth_hammer tactic
  artifacts/                    what earlier runs found, for the next run
  dbg/                          the module as printed, and a copy of the input
  lakefile.toml, lean-toolchain
```

The Lean library the project is stated in is not part of it: the lakefile
requires two packages of [reactor-lean][reactor-lean], each in its own
directory of that repository.

| Package | Modules | Contents |
|---------|---------|----------|
| `Relational` | `Relational.Mat`, `.Box`, `.LTL`, `.Basic` | matrices, ReactiveModule, LTL, the proof rules (`rule_globally`, `rule_buchi`) |
| `LeanAI` | `LeanAI.*` | the `ai_solve` tactic |

By default they are git requires at `project/lakefile.py`'s
`REACTOR_LEAN_REV`. With `REACTOR_LEAN` set to a reactor-lean checkout they
are path requires on it instead, which is how to change the library and the
generator together; `reactor` (and `certify`) refuse a `REACTOR_LEAN` that
is not a checkout before they write anything. The tests set it to the checkout beside
this repository when there is one, and `tests/lean`'s lakefile requires
`Relational` from there.

[reactor-lean]: https://github.com/zeroth-research/reactor-lean

The certificate in `Data.lean` depends on the run:

- with no certificate given and no route, the predicates are `sorry`;
- with no route, it is the certificate given by `--invariant` and
  `--ranking`;
- with a route (`certify --infer`), it is a placeholder that stage 6
  replaces.

`--cert-file PATH.lean` writes a self-contained certificate instead, with
the relational and scalar encodings beside it, and the run ends there.
[What gets generated](#what-gets-generated) has the details.

### 5. Decision procedure (`verith`), untrusted

Not `reactor`'s: the routes are the `verith` package, in the
[`verith/`](../../../verith/README.md) project. `certify --infer NAME` runs one against the
project stage 4 wrote, and `verith --infer NAME` runs one without a project
and prints what it found, in the form `--invariant` and `--ranking` take.
Either way the answer is only a certificate to check: stage 6 renders it
and stage 7 has Lean check it. What `reactor` provides a route with is
data -- the encodings it reads, the lakefile entries it adds
(`pipeline.outputs.generate(lakefile=...)`), `artifacts/` to resume from --
so `reactor.lean` never imports one.


### 6. Lean certificate (`proof/render.py`, `smt/precheck.py`, `proof/tactics.py`, `proof/from_smt.py`)

Under `certify`, the route's answer becomes the project's certificate.
Under `reactor` there is no route, and the same steps turn the given
predicates into the certificate, before stage 4:

- `predicate_terms` parses the predicates once, into cvc5 terms over the
  module's encoding.
- That parse is printed as Lean by `render_certificate`, and read for its
  shape by `predicate_facts`.
- Under `--smt-tactics cvc5`, it is also put to cvc5 for hints
  (`solver_hints`).
- `proof.render.prepare` joins these into a `LeanCertificate`: the definitions in
  `Data.lean`, and the tactic plan in `Certificate.lean`.

`--pre-check cvc5` first asks cvc5 whether the obligations hold, so that a
wrong answer fails in seconds rather than in the build. `certify` then
prints the certificate in the input's names.

`fbk-proveit` is different: it returns files, and nothing is regenerated.
It installs ic3ia's certificate and, unless `--fbk-equiv none` is given,
`Certificate/Equivalence.lean`, which ties ic3ia's model back to the
module.

### 7. The check (`--build-cert`)

`lake update` and `lake build Certificate` run in the project. This is the
step the result rests on: Lean checks the certificate's obligations against
the module's encoding. Lake reports a `sorry` as a warning and exits 0, so
a `sorry` left in `Certificate/` also fails the run. Without `--build-cert`
the run stops at the project, which can then be built by hand. See [Building
the certificate](#building-the-certificate---build-cert).

## Checking a change to the pipeline

Record each of these at the commit before the change and after it, and
compare; every difference must be one the change means.

- **Route transcripts**: `uv run python scripts/route_transcripts.py OUTDIR`
  from `verith/`, then `diff -r`. 42 `certify` runs over all eight routes, with the
  LLM replies scripted; the docstring names the variables that find the
  external tools. Record on a quiet machine, and re-run a `vampire_*` case
  alone before reading a difference in it as a change: under load its
  templates run out of their shared deadline.
- **Encoding snapshots**: `scripts/encoding_snapshots.py OUTDIR` writes what
  the transcripts do not cover: the `System/*.lean` encodings, the bare
  `--cert-file` certificate, and (with `verith` installed) the NA model and
  bridge, per fixture.
- **Goldens**: `tests/test_pipeline_golden.py`, and `certify`'s in
  `verith/tests/test_certify_golden.py`; `REACTOR_UPDATE_GOLDEN=1` rewrites
  them.
- **Suites**: `just py-test` and `just vr-test` (`uv run pytest -m 'not
  slow' -q` from `python/` and from `verith/`; set `VAMPIRE` to the binary
  for the Vampire tests), and `just test-lean` for the lake builds after
  any change to the generated Lean.
- **Encodings against the module**: after a change to an encoder or to the
  op table, `tests/test_encodings_agree.py` (the SMT encoding, two seconds)
  and `tests/lean/test_lean_agree.py` (the Lean one, `-m slow`, a minute)
  run every module of the suite's corpus on random inputs and hold both
  encodings to `reactor.eval`, the interpreter. The equivalence theorems
  cannot catch an encoder that is wrong on every side at once; these can.

## Further reading

- [`TODO.md`](../../../TODO.md): what is still to do, known defects
  included.
- [`verith/README.md`](../../../verith/README.md): the routes, and the `verith` and `certify`
  commands.
- [`tests/limits/`](../../tests/limits/README.md): the limit matrix the
  routes are measured on, with its runners and baseline.
- [`tutorials/verith.md`](../../../tutorials/verith.md): a walk through
  every route on worked examples.

The rest of this file is the reference, starting from the IR.

---

## Reactive Module IR

A **`Module`** is the top-level object.  It has:

- **`ctrl`** — the module's mutable state, as a sequence of `Var`s.  A `Var`
  stands for its *latched* wire (the value from the previous step), and
  `X(v)` is its *next* wire, which `update` writes.
- **`extl`** — external/environment inputs, also a sequence of `Var`s.
- **atoms** — currently always a single atom (the combinational logic block).
  Each atom has an **`init`** term list and an **`update`** term list.

A **`Wire`** carries a `Sort` (`Bool`, `Int`, `Real`, `BitVec`) and a shape,
always 2-D here: `[1, 1]` = scalar, `[1, n]` = row vector, `[m, n]` = matrix.

A **`Term`** is one SSA node: a single operation drawn from a theory
namespace (`LIA`, `LRA`, `BV`), reading from some wires (`term.read`) and
writing one output wire (`term.write[0]`).  The term lists form a
topologically-sorted dataflow graph.

---

## Module Files

The rows run from the bottom layer up: a package imports only the packages
above it in the table. `tests/test_layering.py` holds every module to its
layer.

| Package | What it holds |
|------|------|
| `core/` | The vocabulary every layer is written in, importing nothing of `reactor.lean`: `certificate` (`Claim`, `Certificate` as SMT-LIB, `LeanAnswer` for an answer that is Lean only), `errors` (`Refused`, the error a run is refused with), `options` (`Opt`, an option a route or an input format declares) |
| `lowering/` | What every backend reads of a module besides its terms: `sorts` (sorts and shapes of wires, and the Lean types they print as), `elements` (tuple accessors and `flat_layout`, the row-major element order every scalar encoding shares), `values` (a constant as the exact number every backend states), `ops` (the op table: one row per theory variant, one column per backend) |
| `artifacts/` | The project's `artifacts/`, what a run leaves for the next: `store` (`ArtifactStore`, `Artifact`) and `record` (`found`, `note`, the writes a route makes) |
| `codegen/` | The module as Lean text, never importing cvc5: `context` (`LeanContext`, what every encoding of one module shares), `literals` (constants as Lean text, `ConstantRegistry`), `terms` (term lists → functional `let`-binding bodies, and `lean_gaps`), `boxes` (term lists → `Box` circuit layers), `encodings/` (`ModuleToLean4` and one module per encoding: `functional`, `circuit`, `scalar`, `relational`; `skeleton`, the shape every relational encoding has, and `common`, what the scalar ones share) |
| `smt/` | The module in cvc5, and what the cvc5 routes and the pre-check share: `encode` (the encoder), `encoding` (`ModuleEncoding`, the run's one encoding, over `module.ModuleSMT`), `predicates` (`PredicateParser`), `obligations` (stated once over a frame), `precheck` (`--pre-check`: `ModuleQueries`, `pre_check`, `predicate_terms`), `limits` (time limits), `terms` (walks over terms), `script` (the SMT-LIB printer) |
| `project/` | The generated project on disk: `names` (the files and modules it is made of), `templates` (the Jinja environment), `lakefile` (its dependencies), `build` (`lake build`), `main_lean` (the `--executable` program), and `scaffold`, which writes it (`create_project`, the `ENCODINGS` table, ZerothHammer), the certificate's files as the text it is handed |
| `proof/` | The certificate as Lean: `render` (`prepare`, `LeanCertificate`, the project's `certificate_files` and the `--cert-file` one), `from_smt` (a cvc5 term printed as Lean), `facts` (what the terms and cvc5 say about the proof the tactics attempt: `predicate_facts`, and `solver_hints` under `--smt-tactics`), `tactics` (the plan read off them) |
| `inputs/` | The input formats, one `InputAdapter` each (`python`, `smv`), and `names`, which translates a claim's names to the internal ones and back |
| `cli/` | The command line: `parser` (the flags, one `add_*` per group, which `certify` and `verith` assemble too), `checks` (the rules about which go together, read off the rows), `settings` (`Settings`, what they are read into); `parse_args` runs the three |
| `pipeline/` | `uv run reactor`: `main` runs the stages in order -- `claim` (the module and what is to be proved of it), `certificate` (the module's Lean form, `artifacts/`, the certificate rendered), `outputs` (the project, `--cert-file`, `--hammer-file`, `--build-cert`) |

The inference routes are the `verith` package, in [`verith/`](../../../verith/README.md), with the commands that run them; it is built on these packages, and none of them imports it.

---

## Type Mapping (`lowering/sorts.py`)

`dtype_to_lean_type(wire, simple_types=False)` maps wire types to Lean:

| Python DType | Lean type (default) | Lean type (`simple_types=True`) |
|---|---|---|
| `Bool` (scalar) | `(Mat Bool 1 1)` | `Bool` |
| `Int` (scalar) | `(Mat Int 1 1)` | `Int` |
| `Float` (scalar) | `(Mat Real 1 1)` | `Real` |
| `Int` shape `[n]` | `(Mat Int 1 n)` | — |
| `Int` shape `[m,n]` | `(Mat Int m n)` | — |

All intermediate values are `Mat T m n` (Lean: `Fin m → Fin n → T`) even when
scalar.  This keeps the algebra uniform.  Float-typed modules require
`noncomputable` on all definitions because `Real` lacks decidable equality.

**`product_accessor(pos, total)`** builds right-nested product accessors:
- `total=1` → `""` (value itself)
- `total=2` → `.1`, `.2`
- `total=3` → `.1`, `.2.1`, `.2.2`

---

## `LeanContext` (`codegen/context.py`)

Created once per module; shared across all code generators.  It extracts:

- `extl_latched`, `extl_next`, `ctrl_latched`, `ctrl_next` — the four wire
  groups.
- `constants` — a `ConstantRegistry` that assigns top-level `@[simp] def c0`
  names to non-scalar `Tensor` constants (scalar tensors are inlined at use).
- `init_wire_names`, `update_wire_names` — `dict[wire_id, lean_accessor]`
  mapping each input wire to its Lean expression (e.g. `"ctrl.1"`).
- `uses_real` — True if any wire is Float-typed (triggers `noncomputable`).

`bind_wires(params)` builds the wire-id → accessor dict from a list of
`(param_name, [wire, ...])` pairs.

---

## Encoding 1 — Functional (`codegen/terms.py`, `codegen/encodings/functional.py`)

`translate_terms_lean(terms, input_bindings, block_outputs, constants)` produces a
Lean function body as a string of `let` bindings followed by the output tuple:

```lean
  let x0 : (Mat Bool 1 1) := (fun _ _ => decide (ctrl.1 0 0 < ctrl.2 0 0))
  let x1 : (Mat Int 1 1) := (fun _ _ => (1 : Int))
  ...
  (x6, x5)
```

Each term is looked up in the op table's `mat` column
(`lowering.ops.mat_emitter`, keyed by theory and variant name).  Operations
produce `Mat T 1 1` values even for scalar results — conditions extract
`0 0`, boolean ops wrap in `fun _ _ => ...`.  Non-scalar ops (`MatMul`,
`Linear`, `ReLU`) use their native matrix forms.

`ModuleToLean4.atom_to_lean_functional()` wraps the body into:

```lean
@[simp] def init (extl_n: Unit) : (Mat Int 1 1) × (Mat Int 1 1) := ...
@[simp] def update (ctrl: ...) (extl_l: ...) (extl_n: ...) : ... := ...
```

---

## Encoding 2 — Circuit (`codegen/boxes.py`, `codegen/encodings/circuit.py`)

The same computation is re-expressed as a `Box` (categorical wiring diagram)
inside `namespace Circ`.  The translation:

1. Builds a reversed dependency graph layer by layer.
2. Emits swap/dup/delete layers to route wires into the right positions.
3. Each operation becomes `Box.add`, `Box.lt`, etc. composed with `⊗` (par) and
   `≫` (seq).

`ModuleToLean4.atom_to_lean_circuit()` emits one `@[simp] def init_l0 ...`
per layer and a composed `def init := init_l0 ≫ init_l1 ≫ ...`.

**Equivalence theorems** (`to_lean_equiv_theorems`) prove that the circuit and
functional encodings compute the same result:

```lean
theorem init_circ_eq : ∀ (extl_n : Unit),
    Circ.init.fn ⟨extl_n, ()⟩ =
    let r := init extl_n
    (r.1, (r.2, ())) := by
  intro extl_n
  simp_circ [Circ.init, Box.seq]
  simp_circ [Circ.init_l0]
  ...
  simp [init]
```

The helper macro `simp_circ` unfolds one layer name plus all Box/ValTuple
plumbing lemmas.

---

## Encoding 3 — Scalar (`codegen/terms.py`, `codegen/encodings/scalar.py`)

`translate_terms_scalar` is like `translate_terms_lean` but uses bare scalar types
(`Bool`, `Int`, `Real`) instead of `Mat T 1 1` for all 1×1 wires:

```lean
namespace Scalar
@[simp] def update (ctrl: (Mat Int 1 1) × (Mat Int 1 1)) ... : Int × Int :=
  let x0 : Bool := (decide (ctrl.1 0 0 < ctrl.2 0 0))
  let x1 : Int  := (1 : Int)
  let x2 : Int  := (ctrl.1 0 0 + x1)
  ...
  (x7, x8)
end Scalar
```

Key differences from functional:
- Input parameter types are **unchanged** — they still accept `Mat T m n`.
- Input scalars are pre-extracted at the call site via `ctrl.1 0 0`.
- All intermediate `let` bindings use `Bool`/`Int`/`Real` directly.
- Output type is a product of bare scalars (e.g. `Int × Int`).
- Non-scalar wires (actual matrices, `MatMul` results with shape `[m,n]`) fall
  back to matrix types transparently.

`codegen.encodings.common.scalar_bindings_with_recon` builds the input bindings: a
one-element wire is bound to its bare accessor, so downstream ops receive bare
values, and a wider one to a `let` that rebuilds its `Mat` from the elements.

**Equivalence theorems** (`to_lean_scalar_equiv`) connect the scalar and
functional encodings:

```lean
theorem update_scalar_eq : ∀ ctrl extl_l extl_n,
    update ctrl extl_l extl_n =
    let r := Scalar.update ctrl extl_l extl_n
    (fun _ _ => r.1, fun _ _ => r.2) := by
  intro ctrl extl_l extl_n
  simp only [Scalar.update, update]
  try rfl
  try (apply Prod.ext <;> funext i j <;> simp [Fin.fin_one_eq_zero])
  try (funext i j; simp [Fin.fin_one_eq_zero])
```

The reconstruction `fun _ _ => r.i` wraps each scalar output back into a
`Mat T 1 1`.  The proof reduces via `Fin.fin_one_eq_zero` (every `i : Fin 1`
equals `0`), making the two sides propositionally equal.

**Scalar encoding lives in its own file** (`<Name>Scalar.lean`), separate from
the certificate.  It contains: functional `init`/`update`, `namespace Scalar`,
and the equivalence theorems.

---

## `ModuleToLean4` (`codegen/encodings/__init__.py`)

The main entry point.  Accepts a `Module` or a pre-built `LeanContext`.

| Method | Returns |
|--------|---------|
| `to_lean_functional()` | Constants block + `init`/`update` defs (`System.lean`) |
| `to_lean_circ()` | Circuit namespace + equivalence theorems (`SystemCirc.lean`) |
| `to_lean_mat_rel()` | `namespace Rel`, the matrix-domain relational encoding (`SystemRel.lean`) |
| `to_lean_scalar()` | `namespace Scalar` block + equivalence theorems (`SystemScalar.lean`) |
| `to_lean_rel()` | `namespace ScalarRel`, the scalar-relational encoding (`SystemScalarRel.lean`) |
| `to_lean(circuit=True, scalar=True, rel=True)` | Functional, then each enabled one of circuit, scalar and scalar-relational, joined (never `Rel`); used by tests |
| `atom_to_lean_functional()` | Just `init`/`update` bodies |
| `atom_to_lean_circuit()` | Just `namespace Circ` block |
| `atom_to_lean_scalar()` | Just `namespace Scalar` block |
| `atom_to_lean_mat_rel()` / `atom_to_lean_rel()` | Just the `Rel` / `ScalarRel` namespace |
| `to_lean_equiv_theorems()` | `init_circ_eq` / `update_circ_eq` theorems |
| `to_lean_scalar_equiv()` | `init_scalar_eq` / `update_scalar_eq` theorems |

The project writes one file per row of `project.scaffold.ENCODINGS`, each
printed by the row's `producer`, a `ModuleToLean4.to_lean_*` method.

---

## Certificate Generation (`proof/render.py`)

A **certificate** adds five predicates and a proof skeleton on top of the
module encoding.  The predicates are SMT-LIB over `s0..`, `e0..`, `el0..`
(internally; on the command line the same variables are spelled
`(X @intf.0)`, `(X @extl.0)`, `@extl.0`, see "Writing a claim on the command
line" below)
(`core.certificate.Claim` and `core.certificate.Certificate`). `predicate_terms`
parses them against the module, and `render_certificate` prints them as
Lean `fun` expressions into a `LeanPredicates`; it is the only place they
are printed. Codegen takes a `LeanCertificate`: those definitions and
`PlanInputs`, the facts and solver hints the tactic plan reads. A predicate
that is `None` becomes a `True` / `sorry` placeholder.

`generate_certificate_lean(ctx, lean_cert, *, module_inline=None)`:

1. Emits imports (`Mathlib`, `Relational.Basic`, module import or inline, hammer).
2. Emits `init_pre`, `update_pre`, `inv`, `P`, `DecidablePred P`, `ranking`.
3. Emits `def RM : ReactiveModule Extl State` wiring `init`/`update` in.
4. Emits module-specific macros `simp_mat`, `simp_defs`, `mat_collapse` that
   include all local definition names and matrix reduction lemmas.
5. Emits the ZerothHammer tactic (inlined or via import).
6. Emits proof skeletons: `init_inv`, `step_inv`, `hinv`, `hrank`, `buchi`.

**Wire bindings in certificates:**  State is accessed as `s`, externals as
`e.1` (latched) and `e.2` (next).

**Where Lean text comes from.** Jinja templates (`templates/project/`) render
the files and their fixed structure: the project skeleton, `Data.lean`, the
certificate's proof skeleton. Python renders expressions and the parts whose
shape follows the module: the encodings (`codegen/encodings/`), the predicates
(`render_certificate`), the tactic plan (`proof/tactics.py`). A piece both a project
and a `--cert-file` certificate need is one template partial, rendered by both
(`Certificate/_defs.lean.j2`, the five definitions), and a name or a text more
than one file uses is one constant in `project/names.py`.

### `predicate_terms` / `render_certificate`

`predicate_terms` parses with cvc5 and refuses a predicate by the flag it
came from; `render_certificate(q, share=..., verbatim=...)` translates the
terms. `share=False` prints repeated subterms in full; `proof.render.prepare`
always shares, because a hint's `if_pos` matches once `simp` has
zeta-reduced the `let`s. `verbatim` takes the invariant and ranking of a
`LeanAnswer` as they are. Every certificate is printed through `prepare`,
the pipeline's (`pipeline.certify.render`) and the tests'
(`lean_certificate(module, claim, cert)`, which is `prepare` with no solver
hints) alike; `certificate_files` renders the project's `Data.lean` and
`Certificate.lean` from it, and `create_project` and
`ProjectHandle.write_predicates` write that text as it is.

---

## Tactic plans (`proof/tactics.py`)

The three obligations used to be closed by one hardcoded chain, textually
identical in every certificate — wasteful in both directions: an integer
module with a linear ranking paid for `decide` and `bv_decide` on every goal,
and a shape nobody had anticipated had no way to ask for the step it needed.

`plan_for` reads the module and its predicates and emits two macros,
`cert_prep` (canonicalise) and `cert_close` (decide), which the three proofs
share; each certificate carries a comment saying what was detected.

| detected | consequence |
|---|---|
| Int / Bool in the state | `omega` is available |
| Real | `omega` is dropped; `linarith`, `norm_num` and `norm_cast; linarith` take over |
| Real *and* equalities in the predicates | `simp_all` moves into **prep**, so the closers see numerals, not an opaque `⌊4 - x⌋` |
| a product of two state-dependent terms | `nlinarith`, `positivity` |
| finite state (Bool/BitVec only) | `decide`, last, because it is the most expensive step |
| an `Ite`/`ReLU`/`Min`/`Max` anywhere | `split_ifs` in prep |
| `∧` / `∨` in the predicates | `casesm*` and `simp only [not_and_or]` in prep |
| an `=` in the predicates | the `ne_iff_lt_or_gt` split, *after* `norm_num at *` |
| term count | `maxRecDepth` scales with it |
| branch points in the predicates | scales the heartbeat budget — this, not module size, is what a net costs |
| nothing known to be slow | heartbeat budget stays at 400 000 for economy (it does *not* bound failures — see "Nets over ℝ") |

Two decisions are worth recording because measurement contradicted the
obvious guess. **`linarith` belongs in every plan** — gating it on `Real`
looked right and cost two integer certificates that were closing on it; an
integer goal `omega` cannot phrase is still ordinary linear arithmetic.
**`bv_decide` belongs in none** — it only ever reaches goals every cheaper
prover has already failed on, and there it bit-blasts: including it took a
BitVec certificate from 10 s to 977 s while closing nothing `decide` had not
already closed.

Because the plan depends on the predicates, `Certificate.lean` is not stable
across a change of invariant: `--infer` rewrites it alongside `Data.lean`.
`tests/test_lean_tactics.py` pins these decisions in the fast suite, so a
plan regression does not wait for a Lean build.

## ZerothHammer (`project/scaffold.py`)

A Lean 4 elaborator tactic that cascades proof strategies:

| Phase | Tactic |
|-------|--------|
| 0 | `simp_mat` alone (trivial / True goals) |
| 1 | `omega`, `norm_cast; omega`, `simp_mat; omega`, `simp_mat; linarith` |
| 2 | `push Not; simp_mat; omega` |
| 3 | `simp_mat` + nested `split` (up to 4 levels) + `omega`/`linarith` |
| 4 | `simp_defs` + nested `split` + `omega` |
| 5 | `simp_defs` → `simp_mat` → `mat_collapse` → `split_ifs` → `omega` |
| 6 | `aesop` |
| 7 | `smt` (cvc5) |
| 8 | `sorry` (give-up) |

The three macro names (`simp_mat`, `simp_defs`, `mat_collapse`) are expected to
be defined in the importing file; `ZerothHammer.lean` ships stub definitions
that certificate files override.

---

## Key Lemmas (in `Relational.Basic`)

| Lemma | Effect |
|-------|--------|
| `MatAdd_apply`, `MatMul_apply` | Reduce matrix `+`, `*` to pointwise scalar ops |
| `Mat_1_1_lt_iff`, `_eq_iff`, `_le_iff`, `_ne_iff` | Collapse `(m : Mat T 1 1) 0 0 op n` to bare scalar comparison |
| `ite_fun_apply` | Push `if` through function application |
| `Fin.sum_univ_one/two/three` | Reduce finite sums |

---

## `uv run reactor` CLI (`cli/`, `pipeline/`, `project/`)

`reactor` is the command-line tool that drives the pipeline: Python module
→ Lean project or standalone certificate files, with the certificate as
given. `certify` takes every flag below and `--infer`; the flags of the
inference routes are documented with them, in [`verith/README.md`](../../../verith/README.md).

### Module file format

```python
# mymodule.py
from reactor import Module, Int, Var, X
from reactor.analyzer import convert_method

def init():
    return 0

def update(old_x):
    x = old_x + 1
    if x == 10:
        return 0
    return x

def module() -> Module:
    state = Var(Int([1, 1]))
    init_terms   = convert_method(init,   {},               [X(state)])
    update_terms = convert_method(update, {"old_x": state}, [X(state)])
    return Module.sequential([state], init_terms, update_terms)
```

To write claims in the module's own names (`--safety "(<= (X x) 9)"`), return
`reactor.lean.inputs.Named(module, {"x": state})` instead of the bare module.

### SMV input (`--format smv`, `.smv`)

A NuSMV model with one `MODULE main` is read by `reactor.smv.parse_smv(strict=True)`:
it means what NuSMV means by it, or it is refused with every reason listed. A
model without words lowers to LIA (`integer`, ranges and enum codes are `Int`,
`boolean` is `Bool`), one with words to BV. Refused: a variable with no `init`
or no `next` (NuSMV leaves it free), an `INIT`/`TRANS`/`INVAR` that is not an
assignment to a variable, fairness constraints, an integer type in a model
with words, `*` of two non-constants and `/`/`mod` in LIA, and a `case` whose
last condition is not `TRUE`. An input (`IVAR`) is read in the round it labels,
and `INVAR x = e` / `x := e` hold in every state. Claims use the model's
variable names: `uv run reactor tests/smv_fixtures/mod10.smv --safety "(<= (X x) 9)"`.

A model's own specification is the claim when no `--safety`/`--buchi` is
given: `INVARSPEC p` and `LTLSPEC G p` are `--safety p`, `LTLSPEC G F p` is
`--buchi p` (any other spec is refused). The command line is parsed again with
it added, so every flag rule applies as if it had been typed. A model that
states several needs `--spec N` (counted from 1); one that states any, given
`--safety`/`--buchi` as well, is refused rather than proving one and ignoring
the other.

### Common invocations

```bash
# Bare Lean project (all certificate fields left as sorry)
uv run reactor mymodule.py -o out/ -p MyProject

# Specify the property to prove (SMT-LIB 2 Bool over the state after a round: `(X @intf.N)`). Which flag it
# goes under is the choice of proof rule -- see "Safety or Büchi" below.
uv run reactor mymodule.py --buchi "(= (X @intf.0) 0)" -o out/ -p MyProject
uv run reactor mymodule.py --safety "(<= (X @intf.0) 10)" -o out/ -p MyProject

# Also supply invariant and ranking manually
uv run reactor mymodule.py --buchi "(= (X @intf.0) 0)" \
    --invariant "(and (>= (X @intf.0) 0) (<= (X @intf.0) 10))" \
    --ranking   "(ite (= (X @intf.0) 0) 0 (X @intf.0))" \
    -o out/ -p MyProject

# A safety certificate is an invariant alone -- no ranking function exists
uv run reactor mymodule.py --safety "(<= (X @intf.0) 10)" \
    --invariant "(and (>= (X @intf.0) 0) (<= (X @intf.0) 10))" \
    -o out/ -p MyProject

# Standalone self-contained certificate (no project scaffold)
uv run reactor mymodule.py --buchi "(= (X @intf.0) 0)" --cert-file out/MyCert.lean
# → writes out/MyCert.lean  (certificate)
# → and out/MyCertRel.lean, out/MyCertScalar.lean, out/MyCertScalarRel.lean
#   (the other encodings + equivalence theorems)
```

### What gets generated

**Full project** (`-o`, `-p`):

```
<output_dir>/<ProjectName>/
  lakefile.toml
  lean-toolchain
  System.lean                 # root: imports the encodings
  System/
    System.lean               # functional encoding of init/update
    Circ.lean                 # circuit encoding + equivalence theorems
    Rel.lean                  # relational encoding + equivalence theorems
    Scalar.lean               # scalar encoding + equivalence theorems
    ScalarRel.lean            # scalar-relational encoding + equivalence theorems
  Certificate.lean            # re-export shim
  Certificate/
    Data.lean                 # certificate predicates
    Certificate.lean          # obligations + proofs
  ZerothHammer.lean           # standalone zeroth_hammer tactic
  artifacts/                  # what runs found, for the next run (see below)
  dbg/                        # system.txt (the module) and a copy of the input
  Main.lean                   # only with --executable
```

`certify --infer fbk-proveit` adds `ProveIt/` and `Certificate/Equivalence.lean`.
A file in `Certificate/` the run does not write -- an earlier
`Equivalence.lean`, say -- is deleted when the project is generated
again: the `Certificate` library globs the directory, and the file would be
built against this run's module.

**Standalone certificate** (`--cert-file path/Name.lean`):

```
path/Name.lean               # self-contained: init/update + certificate
path/NameRel.lean            # relational encoding + equivalence theorems
path/NameScalar.lean         # scalar encoding + equivalence theorems
path/NameScalarRel.lean      # scalar-relational encoding + equivalence theorems
```

### Safety or Büchi

The property comes in under one of two flags, and the choice is not a label:
it decides which proof rule the certificate is built on, and so what the
certificate consists of.

| | `--safety P` | `--buchi P` |
|---|---|---|
| proves | `G P` — `P` in every reachable state | `G (F P)` — `P` infinitely often |
| proof rule | `rule_globally` | `rule_buchi` |
| certificate | an invariant that **implies** `P` | an invariant **and** a ranking function that decreases wherever `P` is false |
| obligations | `init_inv`, `step_inv`, `inv_imp_P` | `init_inv`, `step_inv`, `hrank` |
| `--ranking` | rejected — there is nowhere to put one | the other half of the certificate |
| routes (`certify`, `verith`) | `--fbk-proveit` (ic3ia finds the invariant), `--infer ai-cegis`, `--infer nuterm`, `--infer sygus`, `--infer smt-linear`, `--infer houdini`, or `--infer vampire` | `--infer ai`, `--infer ai-cegis`, `--infer nuterm`, `--infer smt-linear`, `--infer houdini`, or `--infer vampire` |

Neither flag *requires* a route: `reactor` generates the project from
whatever predicates were supplied, and the two flags simply say which
certificate to emit. `--safety` and `--buchi` are mutually exclusive, and
`--fbk-proveit` takes only `--safety` — ic3ia decides reachability of `¬P`,
which is not a question about recurrence.

The distinction is not academic. Countdown starts at 100 and counts down, so
`(= (X @intf.0) 0)` is reached over and over — true under `--buchi`, and false under
`--safety` at step 0.

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `-o` / `--output-dir` | `.` | Where to create the project |
| `-p` / `--project-name` | `Rea` | Lean package name |
| `--format` | by suffix | The input's format (`python`: `.py`, `smv`: `.smv`) |
| `-d` / `--module-def` | `module` | Name of the factory function in the Python file (python input only) |
| `--safety` | — | SMT-LIB 2 Bool over `(X @intf.0)..(X @intf.N-1)` (or names, see below), to hold in every reachable state (`G P`) |
| `--buchi` | — | SMT-LIB 2 Bool over `(X @intf.0)..(X @intf.N-1)` (or names, see below), to hold infinitely often (`G (F P)`) |
| `--invariant` | — | SMT-LIB 2 Bool invariant (under `certify --infer`, skips invariant inference) |
| `--ranking` | — | SMT-LIB 2 Int ranking (under `certify --infer`, skips ranking inference) |
| `--cert-file` | — | Write standalone `.lean` file instead of full project |
| `--hammer-file` | — | Regenerate `ZerothHammer.lean` only |
| `--artifacts` | `use` | What to do with the project's `artifacts/`: `use` lets a route resume from what an earlier run left, `ignore` searches afresh, `reset` empties it first |
| `--build-cert` | off | `lake update` + `lake build Certificate` in the generated project |

### Writing a claim on the command line

`--safety`, `--buchi`, `--pre`, `--invariant` and `--ranking` are written in
the reactive module's own terms. The bare names `s0`, `e0`, `el0` (and the
Python-DSL aliases `s`, `e`, `el`) are refused there.

| surface (SMT-LIB) | surface (Python DSL) | means |
|---|---|---|
| `(X v)` | `X(v)` | the value of `v` after the round, for a name `v` the input gives |
| `(X @intf.N)` | `X(@intf.N)` | the same, for interface variable N by position |
| `(X @extl.N)` | `X(@extl.N)` | the input N read in the round |
| `v` | `v` | a bare input name: the input of the round before |
| `@extl.N` | `@extl.N` | the same, by position |

An interface variable's latched value cannot be named yet. A name like `x`
comes from an input that provides one (a module file returning
`reactor.lean.inputs.Named(module, {"x": var})`); `@intf.N` works for any
module. `reactor.lean.inputs.to_surface(text)` turns internal text into this
spelling. Each command prints one `.. --FLAG, in internal names: ...` line
per claim flag it was given. Everything past the command line (`Claim`,
`Certificate`, SMT, Lean, `artifacts/`, LLM replies) still uses the internal
names below.

### State variable naming in SMT-LIB predicates (internal)

`ctrl` wires are named `s0`, `s1`, … in left-to-right order (matching the
`vars` list passed to `Module.sequential`).  For tuple/matrix wires, use
SMT-LIB tuple selectors: `((_ tuple.select 0) s0)`.  External inputs are
`e0..eM-1` (next) and `el0..elM-1` (latched).

## Building the certificate (`--build-cert`)

`--build-cert` is route-independent: it builds whatever certificate the run
produced, with `lake update` then `lake build Certificate` in the generated
project. All three ways of filling a certificate qualify — `reactor`'s, and
`certify`'s two —

```bash
# predicates supplied
uv run reactor mymodule.py --buchi "(= (X @intf.0) 0)" --invariant "(<= (X @intf.0) 100)" \
    --ranking "(X @intf.0)" --build-cert -o out/ -p Counter

# predicates inferred
uv run certify mymodule.py --buchi "(= (X @intf.0) 0)" --infer --build-cert -o out/ -p Counter

# predicates from ic3ia
uv run certify mymodule.py --safety "(not (= (X @intf.0) 15))" --build-cert -o out/ -p Counter \
    --fbk-proveit ~/zeroth/proof-prototyping/lean-ltl-certifying
```

— and the two that do not are refused rather than silently doing nothing:

| refused | because |
|---|---|
| no `--invariant`/`--ranking`, no `--infer`, no `--fbk-proveit` | every obligation in a bare project is `sorry`. Lake compiles that and exits 0, so "built" would mean nothing |
| `--cert-file`, `--hammer-file` | they write a file and return; there is no project to build |

`--invariant` without `--ranking` (or either without a property) is the
first row: a missing predicate is a `sorry`, not a weaker proof.

Two checks sit past the parse-time gate, because a flag cannot promise what
inference returns and lake cannot be trusted to fail loudly:

* **After the route, before the build.** `--infer` can come back without an
  invariant or a ranking — an LLM that answers nothing usable — and the
  error names which one is missing instead of building a proof of `sorry`.
  (`--fbk-proveit` is exempt: its certificate is ic3ia's, installed by the
  route itself, and no `Certificate` describes it.)
* **After the build.** `zeroth_hammer` leaves a `sorry` where it cannot
  close an obligation, and lake reports that as a *warning* and exits 0. So
  `build_certificate` also scans the log for `sorry` under `Certificate/` —
  a dependency's own `sorry` (lean-smt ships one) is not ours — and raises
  if it finds any.

A build that fails is a `LakeBuildError`, and the run exits non-zero with
the Lean diagnostics restated: lake's log ends with thousands of replayed
Mathlib jobs, so a bare "exit 1" as the last line would bury them.

Off by default because the build resolves and compiles `cslib`, Mathlib,
`lean-smt` and cvc5 — minutes against a warm `.lake/packages`, far longer
against a cold one. Because the require set is pinned to the checkout's, one
already-built `.lake/packages` serves both: copying (or cloning, on APFS)
the checkout's into the generated project makes the first `lake update` a
no-op fetch.

## Adding a New Op (`lowering/ops.py`)

Every backend dispatches through one table, so an op is added in one place:
a row in `lowering.ops.OPS` giving its name, the theories that expose it,
and a cell for each of `mat`, `scalar`, `box` and `smt`.  A cell is an
emitter, or `ViaMat`/`Inline`/`Unsupported` saying why it is not one —
`Unsupported`'s reason is what the caller is told when it hits the gap.

`python -m reactor.lean.lowering.ops` prints the whole matrix and every gap in it.
`tests/test_lean_ops.py` checks the table against `LIA`/`LRA`/`BV` in both
directions: a variant a theory gained but the table has not fails there, and
so does a row no theory backs.

---

## `artifacts/` — the project as a workspace

The project is generated *before* inference runs, and what a run leaves in
`<project>/artifacts/` outlives it: an invariant that was inductive but too
weak, a ranking function that used an op the certificate cannot express, a
note for the next prompt.  A second `uv run certify` into the same `-o`
continues rather than restarts — take that invariant and strengthen it, or
take it as given and infer only the ranking function.

Artifacts are arbitrary files (`.smt2` and `.md`, mostly, plus whatever a
subprocess drops there); `artifacts/index.json` types the ones it knows, and
a file with no entry is still listed, as `role="other"`.  Each entry carries
`status` — what a consumer filters on — and `why`, the prose it puts in a
prompt.  Three more fields exist only so a stale artifact is not trusted:
`module_digest`, `prp` and `kind`.  An invariant found for another property,
another proof rule or a since-edited module is not about this run, and
`usable()` drops it and says so, because two identical command lines that
resume differently is the one thing a workspace must not do quietly.

`--artifacts ignore` searches afresh; `--artifacts reset` empties the
directory first, which is what to pass when a bad artifact is being
inherited by every run.  `artifacts/` belongs to no route, so a regenerating
run must never treat it as its own output to clean up.

### The questions, not only the answers

The same directory holds what the run *asked*, which is the other half of
being able to check it.  A run encodes a good deal on its way to a
certificate — the module's transition, the property, the invariant, one query
per obligation — and until it was written down, all of it lived and died
inside the process.  What reached the user was a project and a line saying
`holds` or `REFUTED`: the conclusion without the question.

| file | what it is |
|---|---|
| `system.smt2` | the whole problem as cvc5 encoded it: `init<i>` / `next<i>` per state component, and `P`, `inv`, `ranking` over the same state |
| `obligation-<name>.smt2` | one obligation, **negated**, exactly as it was asked |
| `property.smt`, `inv.smt`, `ranking.smt` | the predicate source, as a flag was given it or a route found it |

`.smt2` is a script — declarations, definitions, `(check-sat)` — so
`cvc5 artifacts/obligation-step_inv.smt2` answers without `reactor` in the
loop, and `unsat` there is the `holds` the pre-check printed.  That is what
makes it worth having when the answer surprises: a refuted obligation becomes
a file to bisect rather than a message to trust.  `.smt` is one predicate and
not a script; the encoded form of the same thing is in `system.smt2`.

These are written by `smt.precheck` as it builds each query, so they appear when
something was encoded — `--pre-check cvc5` is what asks for the obligations.
`--artifacts ignore` still records them: it is about where a run *starts*, not
about what it leaves.

### `README.md` — the index components write

`artifacts/README.md` is the human-readable half of `index.json`, and nothing
assembles it centrally: no one place knows what a run will produce, which
routes will run, or what a component added later will want to leave behind.
So each component passes `what=` to `store.put` / `store.encoded` — one
sentence saying what its file *is* — and the README is rendered from the index
after every write.  It is therefore always exactly what was written, in the
order it was written, and a file a later run rewrites replaces its own section
rather than gaining a second one.

`what` is the counterpart of `why`: `why` says what was wrong with a candidate,
`what` says what the file is.  A question the run asked has a `what` and no
`why`, which is why its status (`encoded`) is not flagged in the README.

---

## Adding a New Encoding

1. Add a translation function in `codegen/terms.py` (or a new file) analogous to
   `translate_terms_lean`.  It receives `input_bindings: dict[wire_id, str]` and
   returns a Lean body string.
2. Add a module under `codegen/encodings/` that wraps the result in the
   appropriate namespace / function signature, and a method on
   `ModuleToLean4` (`codegen/encodings/__init__.py`) that calls it.
3. Optionally add an equivalence theorem connecting the new encoding to the
   existing functional one.
4. Wire it into `to_lean()`, and give it a row in `ENCODINGS`
   (`project/scaffold.py`) to be written into the project.

If the new encoding is *relational* — a per-slot body def, a per-slot
relation, and a conjunction over them — do not spell that shape a fourth
time: build a `RelSyntax` and a `RelBlock` and call `emit_rel_block` from
`codegen/encodings/skeleton.py`, as `Rel`, `ScalarRel` (both in
`codegen/encodings/relational.py`) and the NA model
(`verith/fbk_proveit/na.py`, in the `verith/` project) do.  It is parameterised on the type builder,
the projection, and the spelling
(`def`/`abbrev`, `Prop`/`Bool`, `=`/`==`, `∧`/`&&`); the body producer stays
yours, and the per-slot `*_eq` theorems are emitted only if you name a
functional counterpart to prove them against.
