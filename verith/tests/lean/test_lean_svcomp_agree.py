"""The SV-COMP benchmarks' Lean encoding takes the rounds the module takes.

The Lean side of `nuterm/tests/test_svcomp_agree.py`: the same benchmarks and
runs (`nuterm.tests._agree`), held to the same reference, `reactor.eval`.
Each benchmark's functional encoding is `#eval`ed at each round's state and
inputs, and has to print what the interpreter computed. Two runs a module
rather than the SMT side's six: a file costs about six seconds to start and
a run about one more, and the modules are compiled `_PER_FILE` to a file, in
parallel, against the `Relational` this lake project requires. A module
whose encoding does not compile is still reported as that module, since its
rounds print nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nuterm.tests._agree import CORPUS, Skip, lean_script, runs

_RUNS = 2
_PER_FILE = 8


@pytest.mark.slow
def test_the_lean_encoding_takes_the_rounds_the_benchmark_takes(
        tmp_path: Path, relational_mat, lean_outputs):
    scripts = []
    for name in sorted(CORPUS):
        module = CORPUS[name]()
        try:
            text, want = lean_script(module, runs(module, count=_RUNS),
                                     f"M{len(scripts)}")
        except Skip:
            continue
        scripts.append((name, text, want))
    assert scripts, "nothing to check"

    files = []
    for k in range(0, len(scripts), _PER_FILE):
        chunk = scripts[k:k + _PER_FILE]
        path = tmp_path / f"agree{k // _PER_FILE}.lean"
        path.write_text("import Relational.Mat\n"
                        + "\n".join(t for _, t, _ in chunk))
        files.append((path, chunk))

    bad = []
    for (_path, chunk), out in zip(files, lean_outputs([p for p, _ in files])):
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
    assert not bad, f"{len(bad)} disagreement(s):\n" + "\n".join(bad)
