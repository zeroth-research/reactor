"""The module encoding's proofs go through for every module of the corpus.

`System/Module.lean` states the module as a `Relational.Module` and proves,
module by module, that it is well-formed and that its rounds are those of
the functional `init` and `update` (`codegen/encodings/module.py`). Those
proofs are tactic scripts printed for the module at hand, so each shape of
module -- one atom or several, inputs or none, Real, matrices, deep terms --
is a case of its own: here every module of the corpus (`agree.CORPUS`) is
written as its functional encoding followed by its module encoding, one file
a module, and each file has to elaborate without an error or a `sorry`.

A module with no Lean form is skipped, as `reactor` refuses it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from agree import CORPUS
from lake import lake_build, lean_outputs
from reactor_lean.codegen.encodings import ModuleToLean4
from reactor_lean.core.errors import Refused

IMPORTS = ("Relational.Mat", "Relational.Semantics", "Relational.ReactiveTS")


@pytest.mark.slow
def test_the_module_encoding_of_every_module_elaborates(tmp_path: Path):
    files: list[tuple[str, Path]] = []
    for k, (name, build) in enumerate(CORPUS.items()):
        try:
            m2l = ModuleToLean4(build())
            text = f"{m2l.to_lean_functional()}\n\n{m2l.to_lean_module()}"
        except Refused:
            continue
        path = tmp_path / f"Module{k}.lean"
        path.write_text("".join(f"import {i}\n" for i in IMPORTS) + "\n" + text + "\n")
        files.append((name, path))
    assert any(name == "fixtures/composed" for name, _ in files)

    r = lake_build(*IMPORTS)
    assert r.returncode == 0, f"lake build failed:\n{r.stdout[-2000:]}\n{r.stderr[-800:]}"
    outputs = lean_outputs([path for _, path in files])

    bad = []
    for (name, _), out in zip(files, outputs):
        errors = [ln for ln in out.splitlines() if ": error" in ln or "sorry" in ln]
        if errors:
            bad.append(f"{name}:\n  " + "\n  ".join(errors[:5]))
    assert not bad, f"{len(bad)} of {len(files)} modules:\n" + "\n".join(bad)
