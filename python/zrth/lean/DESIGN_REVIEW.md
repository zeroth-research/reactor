Defects already catalogued in [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) are not
repeated here; this is about shape, not breakage. Where the two overlap, the
issue number is cited.

---


## Findings


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
