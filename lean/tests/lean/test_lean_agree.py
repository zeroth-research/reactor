"""The Lean encoding takes the rounds the module takes.

The corpus and runs of `tests/agree.py`, held to `reactor.eval`, the
interpreter. Each module's functional encoding
-- the `init`/`update` every certificate is a theorem about -- is `#eval`ed
at each round's state and inputs, and has to print what the interpreter
computed.

A module with a Real is skipped (Lean's `Real` is noncomputable), as is one
with no Lean form at all. Two runs a module:
a file costs about six seconds to start and a run about one more, and the
modules are compiled `_PER_FILE` to a file, in parallel, against the
`Relational` this lake project requires. A module whose encoding does not
compile is still reported as that module, since its rounds print nothing.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from agree import CORPUS, Skip, lean_script, runs
from lake import lean_outputs

_RUNS = 2
_PER_FILE = 8


def _disagreements(names: list[str], tmp_path: Path) -> "tuple[list[str], int]":
    """What Lean printed that the interpreter did not compute, over the
    modules `names`; and how many modules were checked."""
    scripts = []
    for name in names:
        module = CORPUS[name]()
        try:
            text, want = lean_script(module, runs(module, count=_RUNS),
                                     f"M{len(scripts)}")
        except Skip:
            continue
        scripts.append((name, text, want))

    files = []
    for k in range(0, len(scripts), _PER_FILE):
        chunk = scripts[k:k + _PER_FILE]
        body = "\n".join(t for _, t, _ in chunk)
        path = tmp_path / f"agree{k // _PER_FILE}.lean"
        path.write_text("import Relational.Mat\n" + body)
        files.append((path, chunk))

    outputs = lean_outputs([path for path, _chunk in files])

    bad = []
    for (_path, chunk), out in zip(files, outputs):
        got = {ln.split("|", 1)[0]: ln for ln in out.splitlines() if "|" in ln}
        errors = [ln for ln in out.splitlines() if ": error" in ln]
        for name, _text, want in chunk:
            for tag, line in want.items():
                if tag not in got:
                    bad.append(f"{name}: {tag} printed nothing"
                               + (f" (first error: {errors[0]})" if errors else ""))
                    break
                if got[tag] != line:
                    bad.append(f"{name}: Lean printed {got[tag]!r}, the "
                               f"interpreter computed {line!r}")
    return bad, len(scripts)


@pytest.mark.slow
def test_the_lean_encoding_takes_the_rounds_the_module_takes(tmp_path, relational_mat):
    bad, checked = _disagreements(sorted(CORPUS), tmp_path)
    assert checked, "nothing to check"
    assert not bad, f"{len(bad)} disagreement(s):\n" + "\n".join(bad)


@pytest.mark.slow
def test_a_wrong_lean_emitter_is_caught(tmp_path, monkeypatch, relational_mat):
    """The check can fail: with `Sub`'s matrix emitter swapping its
    operands, the countdown's every `update` round prints something else."""
    from reactor_lean.lowering import ops

    for key, row in list(ops._BY_KEY.items()):
        if key[1] == "Sub":
            monkeypatch.setitem(ops._BY_KEY, key, dataclasses.replace(
                row, mat=lambda a: f"({a[1]} - {a[0]})"))
    bad, checked = _disagreements(["limits/m_countdown"], tmp_path)
    assert checked == 1
    assert bad and "the interpreter computed" in bad[0]
