# `reactor_lean` — Lean 4 Certificate Generation

`uv run reactor` takes a reactive module and a property of it, and ends with a
Lean project in which `lake build` checks a proof of the property. The proof
rests on a certificate: an inductive invariant, and for a Büchi property
also a ranking function. `reactor` takes the certificate as given
(`--invariant`, `--ranking`). The pipeline does not trust a certificate,
whoever wrote it. It is written into the project, and Lean checks it.

## The pipeline, stage by stage

`pipeline.main` runs the stages in this order. Each one names the code that
does it and the section of this file that covers it in full.

### 1. Input (`inputs/`)

The input is a file, read by the adapter that `--format` names or that the
file's suffix picks:

| format | suffix | what it gives |
|---|---|---|
| `python` | `.py` | the `reactor.Module` that `module()` (or `-d NAME`) returns. Returning `inputs.Named(module, {"x": var})` instead adds names to write claims in. A Gymnasium environment comes in this way, through `reactor.gym` |

The adapter returns an `inputs.Problem`: the module, the source text and the
names. A new input format is a new `InputAdapter` in `inputs/`.

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
wires ([Reactive Module IR](#reactive-module-ir)). Before any Lean is
written, `lean_gaps` refuses an operator with no Lean form.

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
  Certificate/Module.lean       RM and lts, the module as a ReactiveModule and
                                its transition system: fixed text
  Certificate/Proofs.lean       the obligations, each stated again and proved
                                by `sorry`, for a prover to replace
  Certificate/Certificate.lean  the proof rule's obligations, each taken from
                                Proofs.lean: fixed text, one per kind of
                                property
  dbg/                          the module as printed, and a copy of the input
  lakefile.toml, lean-toolchain
```

The Lean library the project is stated in is not part of it: the lakefile
requires two packages of [ReactLib][ReactLib], each in its own
directory of that repository.

| Package | Library | Modules | Contents |
|---------|---------|---------|----------|
| `ReactLib` | `Spec` | `Spec.LTL` | LTL formulas and their semantics |
| | `Relational` | `Relational.Mat`, `.Box`, `.Basic` | matrices, ReactiveModule, the proof rules (`rule_globally`, `rule_buchi`) |
| `LeanAI` | `LeanAI` | `LeanAI.*` | the `ai_solve` tactic |

By default they are git requires at `project/lakefile.py`'s
`REACTLIB_REV`. With `REACTLIB` set to a ReactLib checkout they
are path requires on it instead, which is how to change the library and the
generator together; `reactor` refuses a `REACTLIB` that is not a
checkout before it writes anything. `just` sets it to the checkout beside
this repository when there is one, and the tests' Lake project
(`tests/lean`) requires `ReactLib` from it.

[ReactLib]: https://github.com/zeroth-research/ReactLib

The certificate in `Data.lean` is the one given by `--invariant` and
`--ranking`; with none given, its predicates are `sorry`.

`--cert-file PATH.lean` writes a self-contained certificate instead, with
the relational and scalar encodings beside it, and the run ends there.
[What gets generated](#what-gets-generated) has the details.

### 5. Lean certificate (`proof/render.py`, `parse/`, `proof/from_ir.py`)

These steps turn the given predicates into the certificate, before stage 4:

- The predicates are parsed once: `parse.native.read_predicates` reads them
  into reactor's own terms, the IR the module is written in, over the
  module's wires -- the Python predicate syntax first and SMT-LIB for the
  rest.
- That parse is printed as Lean by `render_certificate` (`from_ir`), into a
  `LeanCertificate`: the definitions in `Data.lean`. An Int numeral compared
  with a Real is `(2 : Real)`, and a nested `and`, `or`, `+` or `*` is
  printed flat.
  `Certificate.lean` is the same text for every certificate of a safety
  property, and for every certificate of a Büchi one.

### 6. The proofs

`reactor` states the obligations and proves none of them: each is `sorry`
in `Proofs.lean`, and lake reports a `sorry` as a warning and builds. The
run stops at the project. Proving the obligations -- and `lake build
Certificate`, which is what the result rests on, Lean checking them against
the module's encoding -- is a prover's: it writes `Proofs.lean`, either as a
whole file or as `ProofScripts` handed to `certificate_files`, which puts
each script under its statement. See [What a certificate
rewrites](#certificate-generation-proofrenderpy).

## Checking a change to the pipeline

Record each of these at the commit before the change and after it, and
compare; every difference must be one the change means.

- **Goldens**: `tests/test_pipeline_golden.py`; `REACTOR_UPDATE_GOLDEN=1`
  rewrites them.
- **Suites**: `just lean-test` (`uv run pytest -m 'not slow' -q` from
  `lean/`), and `just test-lean` for the lake builds after any change to
  the generated Lean.
- **Encodings against the module**: after a change to an encoder or to the
  op table, `tests/lean/test_lean_agree.py` (`-m slow`, a minute) runs every
  module of the suite's corpus on random inputs and holds the Lean encoding
  to `reactor.eval`, the interpreter. The equivalence theorems cannot catch
  an encoder that is wrong on every side at once; this can.

## Further reading

- [`TODO.md`](../../../TODO.md): what is still to do, known defects
  included.

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
| `core/` | The vocabulary every layer is written in, importing nothing of `reactor_lean`: `certificate` (`Claim`, `Certificate` as SMT-LIB, `LeanAnswer` for an answer that is Lean only), `errors` (`Refused`, the error a run is refused with), `options` (`Opt`, an option an input format declares) |
| `lowering/` | What every backend reads of a module besides its terms: `sorts` (sorts and shapes of wires, and the Lean types they print as), `elements` (tuple accessors and `flat_layout`, the row-major element order every scalar encoding shares), `values` (a constant as the exact number every backend states), `ops` (the op table: one row per theory variant, one column per backend) |
| `codegen/` | The module as Lean text: `context` (`LeanContext`, what every encoding of one module shares), `literals` (constants as Lean text, `ConstantRegistry`), `terms` (term lists → functional `let`-binding bodies, and `lean_gaps`), `boxes` (term lists → `Box` circuit layers), `encodings/` (`ModuleToLean4` and one module per encoding: `functional`, `circuit`, `scalar`, `relational`; `skeleton`, the shape every relational encoding has, and `common`, what the scalar ones share) |
| `parse/` | Reading a predicate's text: `dsl` (the Python predicate syntax), `sexpr` and `smtlib` (SMT-LIB into reactor's terms), `formula` (`Formula`, a predicate in reactor's terms), `native` (the parser: `NativeParser`, `read_predicates`) |
| `project/` | The generated project on disk: `names` (the files and modules it is made of), `templates` (the Jinja environment), `lakefile` (its dependencies), `main_lean` (the `--executable` program), and `scaffold`, which writes it (`create_project`, the `ENCODINGS` table), the certificate's files as the text it is handed |
| `proof/` | The certificate as Lean: `render` (`render_certificate`, `LeanCertificate`, `ProofScripts`, the project's `certificate_files` and the `--cert-file` one), `from_ir` (a `parse.Formula` printed as Lean) |
| `inputs/` | The input formats, one `InputAdapter` each (`python`), and `names`, which translates a claim's names to the internal ones and back |
| `cli/` | The command line: `parser` (the flags, one `add_*` per group), `checks` (the rules about which go together, read off the rows), `settings` (`Settings`, what they are read into); `parse_args` runs the three |
| `pipeline/` | `uv run reactor`: `main` runs the stages in order -- `claim` (the module and what is to be proved of it), `certificate` (the module's Lean form, the certificate rendered), `outputs` (the project, `--cert-file`) |

---

## Type Mapping (`lowering/sorts.py`)

`dtype_to_lean_type(wire, simple_types=False)` maps wire types to Lean:

| Python DType | Lean type (default) | Lean type (`simple_types=True`) |
|---|---|---|
| `Bool` (scalar) | `(Matrix (Fin 1) (Fin 1) Bool)` | `Bool` |
| `Int` (scalar) | `(Matrix (Fin 1) (Fin 1) Int)` | `Int` |
| `Float` (scalar) | `(Matrix (Fin 1) (Fin 1) Real)` | `Real` |
| `Int` shape `[n]` | `(Matrix (Fin 1) (Fin n) Int)` | — |
| `Int` shape `[m,n]` | `(Matrix (Fin m) (Fin n) Int)` | — |

All intermediate values are Mathlib's `Matrix (Fin m) (Fin n) T`, even when
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
  let x0 : (Matrix (Fin 1) (Fin 1) Bool) := (Matrix.mk fun _ _ => decide (ctrl.1 0 0 < ctrl.2 0 0))
  let x1 : (Matrix (Fin 1) (Fin 1) Int) := (Matrix.mk fun _ _ => (1 : Int))
  ...
  (x6, x5)
```

Each term is looked up in the op table's `mat` column
(`lowering.ops.mat_emitter`, keyed by theory and variant name).  Operations
produce 1×1 `Matrix` values even for scalar results — conditions extract
`0 0`, boolean ops wrap in `Matrix.mk fun _ _ => ...`.  Non-scalar ops
(`MatMul`, as Mathlib's `*`, `Linear`, `ReLU`) use their native matrix
forms.  Every matrix built from its entries is a `Matrix.mk`
(`lowering.sorts.mat_mk`), never a bare lambda: to `simp` and instance
search a bare `fun i j => …` is a function, not a matrix.  `Matrix.mk` is
ReactLib's, not Mathlib's (`Relational.Mat` says why).

`functional.atom_to_lean_functional(ctx)` wraps the body into:

```lean
@[simp] def init (extl_n: Unit) : (Matrix (Fin 1) (Fin 1) Int) × (Matrix (Fin 1) (Fin 1) Int) := ...
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

`circuit.atom_to_lean_circuit` emits one `@[simp] def init_l0 ...`
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
(`Bool`, `Int`, `Real`) instead of 1×1 matrices for all 1×1 wires:

```lean
namespace Scalar
@[simp] def update (ctrl: (Matrix (Fin 1) (Fin 1) Int) × (Matrix (Fin 1) (Fin 1) Int)) ... : Int × Int :=
  let x0 : Bool := (decide (ctrl.1 0 0 < ctrl.2 0 0))
  let x1 : Int  := (1 : Int)
  let x2 : Int  := (ctrl.1 0 0 + x1)
  ...
  (x7, x8)
end Scalar
```

Key differences from functional:
- Input parameter types are **unchanged** — they still accept `Matrix (Fin m) (Fin n) T`.
- Input scalars are pre-extracted at the call site via `ctrl.1 0 0`.
- All intermediate `let` bindings use `Bool`/`Int`/`Real` directly.
- Output type is a product of bare scalars (e.g. `Int × Int`).
- Non-scalar wires (actual matrices, `MatMul` results with shape `[m,n]`) fall
  back to matrix types transparently.

`codegen.encodings.common.scalar_bindings_with_recon` builds the input bindings: a
one-element wire is bound to its bare accessor, so downstream ops receive bare
values, and a wider one to a `let` that rebuilds its matrix from the elements.

**Equivalence theorems** (`to_lean_scalar_equiv`) connect the scalar and
functional encodings:

```lean
@[simp] def pack (r : Int × Int) : (Matrix (Fin 1) (Fin 1) Int) × (Matrix (Fin 1) (Fin 1) Int) :=
  ((Matrix.mk fun _ _ => r.1), (Matrix.mk fun _ _ => r.2))

theorem update_scalar_eq : ∀ (ctrl : (Matrix (Fin 1) (Fin 1) Int) × (Matrix (Fin 1) (Fin 1) Int)) (extl_l : Unit) (extl_n : Unit),
    update ctrl extl_l extl_n = Scalar.pack (Scalar.update (Scalar.unpack_ctrl ctrl) extl_l extl_n) := by
  intro ctrl extl_l extl_n
  simp only [Scalar.pack, Scalar.update, update, Scalar.unpack_ctrl, Fin.cons_zero, Fin.cons_succ]
  try rfl
  try (apply Prod.ext <;> funext i j <;> fin_cases i <;> fin_cases j <;> simp)
  ...
  all_goals try simp [← Matrix_1_1_eq]
```

`Scalar.pack` wraps each scalar output back into a 1×1 matrix, as
`Matrix.mk fun _ _ => r.i`, and `Scalar.unpack_*` reads a wire's elements out.
The proof enumerates every index of every wire (`fin_cases`), or, for a 1×1
wire, rewrites by `Fin.fin_one_eq_zero` (every `i : Fin 1` equals `0`).

**Scalar encoding lives in its own file** (`<Name>Scalar.lean`), separate from
the certificate.  It contains: functional `init`/`update`, `namespace Scalar`,
and the equivalence theorems.

---

## `ModuleToLean4` (`codegen/encodings/__init__.py`)

The main entry point.  Accepts a `Module` or a pre-built `LeanContext`.

| Method | Returns |
|--------|---------|
| `to_lean_functional()` | Constants block + `init`/`update` defs (`System/System.lean`) |
| `to_lean_circ()` | Circuit namespace + equivalence theorems (`System/Circ.lean`) |
| `to_lean_mat_rel()` | `namespace Rel`, the matrix-domain relational encoding (`System/Rel.lean`) |
| `to_lean_scalar()` | `namespace Scalar` block + equivalence theorems (`System/Scalar.lean`) |
| `to_lean_rel()` | `namespace ScalarRel`, the scalar-relational encoding (`System/ScalarRel.lean`) |
| `to_lean(circuit=True, scalar=True, rel=True)` | Functional, then each enabled one of circuit, scalar and scalar-relational, joined (never `Rel`); used by tests |
| `to_lean_equiv_theorems()` | `init_circ_eq` / `update_circ_eq` theorems |
| `to_lean_scalar_equiv()` | `init_scalar_eq` / `update_scalar_eq` theorems |

Each wraps the function of its encoding's module that prints the bare
block -- `atom_to_lean_functional`, `atom_to_lean_circuit`,
`atom_to_lean_scalar`, `atom_to_lean_mat_rel`, `atom_to_lean_rel` -- each
taking the `LeanContext`.

The project writes one file per row of `project.scaffold.ENCODINGS`, each
printed by the row's `producer`, a `ModuleToLean4.to_lean_*` method.

---

## Certificate Generation (`proof/render.py`)

A **certificate** adds five predicates and a proof skeleton on top of the
module encoding.  The predicates are SMT-LIB over `s0..`, `e0..`, `el0..`
(internally; on the command line the same variables are spelled
`(X @intf.0)`, `(X @extl.0)`, `@extl.0`, see "Writing a claim on the command
line" below)
(`core.certificate.Claim` and `core.certificate.Certificate`). `read_predicates`
parses them against the module, and `render_certificate` prints them as
Lean `fun` expressions into a `LeanCertificate`; it is the only place they
are printed. A predicate that is `None` becomes a `True` / `sorry`
placeholder.

`certificate_files(ctx, lean_cert, proofs)` renders the project's four
files:

1. `Data.lean`: `init_pre`, `update_pre`, `inv`, `P`, `DecidablePred P`,
   `ranking`.
2. `Module.lean`: `def RM`, wiring `init`/`update` and the preconditions into
   a `ReactiveModule`, and `def lts`, its transition system.
3. `Proofs.lean`: `init_inv_proof`, `step_inv_proof` and `hrank_proof` (or
   `inv_imp_P_proof`), each obligation stated again and proved by
   `proofs.script(name)` -- `sorry`, unless a prover handed its scripts
   over as `ProofScripts`, with the modules they import, the options they
   run under and what they share.
4. `Certificate.lean`: `init_inv`, `step_inv` and `hrank` (or `inv_imp_P`),
   each `exact` its `_proof`; `hinv`; and `buchi` (or `safety`).

`generate_certificate_lean(ctx, lean_cert, module_inline=..., proofs=...)`
is the `--cert-file` certificate: all four in one file, the module inlined,
under the proofs' options.

**Wire bindings in certificates:**  State is accessed as `s`, externals as
`e.1` (latched) and `e.2` (next).

**Where Lean text comes from.** Jinja templates (`templates/project/`) render
the files and their fixed structure: the project skeleton, `Data.lean`,
`Module.lean`, `Proofs.lean`, the certificate. Python renders expressions and
the parts whose shape follows the module: the encodings (`codegen/encodings/`)
and the predicates (`render_certificate`). A piece both a project and a
`--cert-file` certificate need is one template partial, rendered by both
(`Certificate/_defs.lean.j2`, the five definitions;
`Certificate/_module.lean.j2`, `RM` and `lts`; `Certificate/_proofs.lean.j2`,
the obligations stated again and proved), and a name or a text more than one
file uses is one constant in `project/names.py`.

**What a certificate rewrites.** Everything that changes with the module or
its certificate is in `Certificate/Data.lean` and `Certificate/Proofs.lean`;
`Certificate/Module.lean` is fixed text, and `Certificate/Certificate.lean` is
fixed text, one per kind of property, so the two can be rewritten -- by
`reactor`, by a prover or by hand -- without them. `RM` leaves its types to Lean
(`({ ... } : ReactiveModule _ _)`), which reads them off `init_pre` and
`update`, so `Module.lean` names none of the module's. `Proofs.lean` states
each obligation again, so a proof of it is complete in one file, and
`Certificate.lean` takes it with `exact`, which checks the two statements
agree: a `Proofs.lean` that proves something else does not build, and the
certificate trusts nothing in it.

### `read_predicates` / `render_certificate`

`read_predicates` parses and refuses a predicate by the flag it came from;
`render_certificate(q, share=..., verbatim=...)` translates the terms. `share=False` prints repeated subterms in full, for reading; a
certificate is printed shared, which Lean reads the same once `simp` has
zeta-reduced the `let`s. `verbatim` takes the invariant and ranking of a
`LeanAnswer` as they are. Every certificate is printed through it, by
`pipeline.certificate.render(module, claim, cert)`; `certificate_files` renders
the project's `Data.lean`, `Module.lean`, `Proofs.lean` and
`Certificate.lean` from it, and `create_project` writes that text as it is.

---

## Key Lemmas (in `Relational.Mat`)

| Lemma | Effect |
|-------|--------|
| `Matrix.mk_apply` | Read an entry of a matrix built from its entries |
| `add_Matrix_apply`, `sub_Matrix_apply`, `neg_Matrix_apply`, `mul_Matrix_apply`, `zero_Matrix_apply`, `transpose_Matrix_apply` | Reduce Mathlib's `+`, `-`, `*`, `0`, `Matrix.transpose` to pointwise scalar ops |
| `Matrix_1_1_lt_iff`, `_eq_iff`, `_le_iff`, `_ne_iff` | Collapse `(m : Matrix (Fin 1) (Fin 1) T) 0 0 op n` to bare scalar comparison |
| `Matrix_ite_apply` | Read an `if` between matrices at an entry |
| `Fin.sum_univ_one/two/three` | Reduce finite sums |

The lemmas that read an entry are pre-lemmas (`simp↓`), proved by `id rfl`;
`Relational.Mat` says why. Mathlib's own `Matrix.add_apply` and the like are
`rfl` lemmas, which a `dsimp` over generated matrices has to leave out.

---

## `uv run reactor` CLI (`cli/`, `pipeline/`, `project/`)

`reactor` is the command-line tool that drives the pipeline: Python module
→ Lean project or standalone certificate files, with the certificate as
given.

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
`reactor_lean.inputs.Named(module, {"x": state})` instead of the bare module.

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
    Module.lean               # RM and lts (fixed text)
    Proofs.lean               # the obligations, stated again, proved by `sorry`
    Certificate.lean          # obligations, each from Proofs.lean (fixed text)
  dbg/                        # system.txt (the module) and a copy of the input
  Main.lean                   # only with --executable
```

A file in `Certificate/` the run does not write -- one an earlier run left
there, say -- is deleted when the project is generated again: the
`Certificate` library globs the directory, and the file would be built
against this run's module.

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

`reactor` generates the project from whatever predicates were supplied, and
the two flags say which certificate to emit. `--safety` and `--buchi` are
mutually exclusive.

The distinction is not academic. Countdown starts at 100 and counts down, so
`(= (X @intf.0) 0)` is reached over and over — true under `--buchi`, and false under
`--safety` at step 0.

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `-o` / `--output-dir` | `.` | Where to create the project |
| `-p` / `--project-name` | `Rea` | Lean package name |
| `--format` | by suffix | The input's format (`python`: `.py`) |
| `-d` / `--module-def` | `module` | Name of the factory function in the Python file (python input only) |
| `--safety` | — | SMT-LIB 2 Bool over `(X @intf.0)..(X @intf.N-1)` (or names, see below), to hold in every reachable state (`G P`) |
| `--buchi` | — | SMT-LIB 2 Bool over `(X @intf.0)..(X @intf.N-1)` (or names, see below), to hold infinitely often (`G (F P)`) |
| `--invariant` | — | SMT-LIB 2 Bool invariant |
| `--ranking` | — | SMT-LIB 2 Int ranking (`--buchi` only) |
| `--cert-file` | — | Write standalone `.lean` file instead of full project |

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
`reactor_lean.inputs.Named(module, {"x": var})`); `@intf.N` works for any
module. `reactor_lean.inputs.to_surface(text)` turns internal text into this
spelling. A run prints one `.. --FLAG, in internal names: ...` line
per claim flag it was given. Everything past the command line (`Claim`,
`Certificate`, Lean) still uses the internal names below.

### State variable naming in SMT-LIB predicates (internal)

`ctrl` wires are named `s0`, `s1`, … in left-to-right order (matching the
`vars` list passed to `Module.sequential`).  For tuple/matrix wires, use
SMT-LIB tuple selectors: `((_ tuple.select 0) s0)`.  External inputs are
`e0..eM-1` (next) and `el0..elM-1` (latched).

## Adding a New Op (`lowering/ops.py`)

Every backend dispatches through one table, so an op is added in one place:
a row in `lowering.ops.OPS` giving its name, the theories that expose it,
and a cell for each of `mat`, `scalar` and `box`.  A cell is an
emitter, or `ViaMat`/`Inline`/`Unsupported` saying why it is not one —
`Unsupported`'s reason is what the caller is told when it hits the gap.

`python -m reactor_lean.lowering.ops` prints the whole matrix and every gap in it.
`tests/test_lean_ops.py` checks the table against `LIA`/`LRA`/`BV` in both
directions: a variant a theory gained but the table has not fails there, and
so does a row no theory backs.

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
`codegen/encodings/skeleton.py`, as `Rel` and `ScalarRel` (both in
`codegen/encodings/relational.py`) do.  It is parameterised on the type
builder, the projection, and the spelling
(`def`/`abbrev`, `Prop`/`Bool`, `=`/`==`, `∧`/`&&`); the body producer stays
yours, and the per-slot `*_eq` theorems are emitted only if you name a
functional counterpart to prove them against.
