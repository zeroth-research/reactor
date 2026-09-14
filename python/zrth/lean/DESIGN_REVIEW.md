# Design review: the encoding paths in `zrth/lean`

Defects already catalogued in [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) are not
repeated here; this is about shape, not breakage. Where the two overlap, the
issue number is cited.

`MEASURED` was reproduced by running the code. `READ` comes from reading it.

---

## The question that prompted this: can `Rel`/`ScalarRel` reuse the SMT path?

**Not as a body producer, and the reason is structural rather than
incidental.**

`Rel` and `ScalarRel` exist *for* their equivalence theorems — rows 3-6 of
`GENERATED.md`'s "Summary of what is claimed". Those proofs are:

```python
# translate/rel.py:75-79
f"  simp only [{scalar_func}, {func_name}, Fin.cons_zero, Fin.cons_succ]",
f"  try rfl",
```

Swapping the body producer would trade a `rfl`-provable equivalence for one
nothing can discharge.

**What the fbk work does make reusable** is the skeleton (A), the slot layout
(B), and the SMT path *as a checker* rather than a generator (G).

---

## Findings

### A. Three relational emitters spell one skeleton three times · MEASURED

| emitter | namespace | domain | body from |
|---|---|---|---|
| `translate/mat_rel.py:16` `atom_to_lean_mat_rel` | `Rel` | `Mat`, Prop | `_translate_terms` |
| `translate/rel.py:19` `atom_to_lean_rel` | `ScalarRel` | flat scalars, Prop | `_translate_terms_scalar` |

All three emit the same shape: a per-slot body def (`effect_i` / `init_i`), a
per-slot relation (`R_i` / `Init_i`), and a conjunction over them
(`TransRel` / `INIT`).

94 of `mat_rel.py`'s 162 body lines are byte-identical to lines in `rel.py`
(`diff -u`, counting context lines).

They vary on exactly three axes:

1. **type builder** — `_product_type` vs `_product_type_scalar` vs the
   per-slot type list;
2. **body producer** — the three in the table above;
3. **projection** — `_accessor(i, n)` vs `_flat_slice(offset, size, total)`
   vs `var_k state`.

One emitter parameterised on those three, with the `*_eq` theorem block
optional, collapses `rel.py` and `mat_rel.py` into one and lets `fbk.py`
supply its own body producer without owning a copy of the skeleton. It
touches no proof: the theorem block stays exactly where it is, emitted only
by the two instances that can prove it.

### B. The element flattening has three owners · READ

Three views of one row-major flattening, derived independently:

| function | returns |
|---|---|
| `translate/_shared.py:14` `_flat_layout` | `[(offset, size)]` per wire, plus total |
| `translate/fbk.py:192` `_slot_layout` | `(wire index, row, col)` per slot |
| `translate/fbk.py:199` `_slot_accessors` | `s{i}` to its `(var_k state)` reads |

`KNOWN_ISSUES` #20 and #22 are both exactly these drifting apart — a per-wire
layout meeting a per-element one. One owner returning all three views makes
that class of bug unrepresentable rather than merely fixed.

### F. `ModuleToLean4` is silently order-dependent · MEASURED

`to_lean_scalar_equiv()` reads `self._argmax_scalar_variants`, which only
`atom_to_lean_scalar()` populates. Both are public (README documents them as
a table of methods). Called alone on a module using `Argmax`:

```
with atom_to_lean_scalar first, argmax lemma present: True
called alone,                   argmax lemma present: False
```

The missing `argmax1d_scalar_int_3_eq` drops out of the `simp only` list and
the emitted proof fails at `lake build` rather than at generation. Same
coupling for `atom_to_lean_circuit()` to `to_lean_equiv_theorems()` through
`_init_layer_names` / `_update_layer_names`.

Current callers happen to be in order. Returning the variants from the
producer, or caching on first use, removes the trap.

### G. The SMT encoding is never used to *check* the Lean encoding · READ

This is the reuse with the best payoff, and the one `KNOWN_ISSUES`' own
lesson points at:

> the equivalence theorems can stay provable while both sides are wrong

### H. Minor

* Emission glue is hand-rolled five times in `project.py:551-610` and three
  times in `main.py:644-676` — header comment, imports, `m2l.to_lean_*()`. A
  `(filename, header, imports, producer)` table would also make the
  `--cert-file` and project paths provably consistent instead of
  consistent-by-inspection.
* `common.py:_constant_expr` and `native.py:_constant_expr_scalar` take a
  `const_name` argument neither uses.

---

## Deliberately not changed

* **`LeanContext`** — built once, read-only, shared by every consumer.
* **The `translate/` split by encoding.** The right seam. What is missing is
  the shared skeleton behind it (A), not a different split.
* **The sharing/`let` logic, the `min`/`max` folding, the rule that
  `_simplify` never touches a Bool slot.** Each is a measured fix carrying its
  measurement in the comment.

---

## Order of work, if any

1. **C** — one op registry with an explicit coverage matrix and a
   completeness test. Highest value, independent of everything else.
2. **A + B** — one relational skeleton, one slot-layout owner. Removes the
   duplication that #20 and #22 came out of.
3. **G** — the SMT path as a cross-check on the Lean path. Builds on C.
4. **D, E, F** — small, independent, each a few lines.

Overall the package is in good shape; the discipline of recording *why* in
the comment next to the code is unusually strong and should survive any of
the above.
