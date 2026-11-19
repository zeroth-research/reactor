"""The circuit encoding is the functional one, proved.

`System/Circ.lean` states a module's `init` and `update` again as a diagram
of `Box` combinators (`Relational.Circ`), one per op (the op table's `box` column),
and proves each equal to the functional encoding's (`init_circ_eq`,
`update_circ_eq`). Here that file is compiled, below the functional encoding
it quotes, for every module of the corpus `test_lean_agree.py` runs: a `Box`
that computes something other than its op's matrix form leaves the proof a
goal it cannot close. The matrix forms are held to the interpreter there,
so each `Box` is held to it too, at every argument rather than at random
ones.

A module with no circuit -- an op with no `Box`, or with no Lean form at
all -- is left out: `test_lean_ops.py` lists those gaps. One module to a
file, since each file defines its own `simp_circ`.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from reactor.lean.codegen.encodings import ModuleToLean4
from reactor.lean.project.names import LINTERS_OFF
from tests.agree import CORPUS

_LEAN_DIR = Path(__file__).parent

# The modules whose circuit is generated and does not compile.
_KNOWN = {
    # The circuit types `argmax_1d` Int-valued where the functional
    # encoding has it Real (tutorials/verith_gym.md).
    "fixtures/simple_env",
}


def _build_relational() -> None:
    built = subprocess.run(["lake", "build", "Relational.Circ"], cwd=_LEAN_DIR,
                           capture_output=True, text=True, timeout=1800)
    assert built.returncode == 0, built.stdout[-2000:] + built.stderr[-2000:]


def _unproved(names: list[str], tmp_path: Path) -> "tuple[dict[str, str], int]":
    """The modules among `names` whose circuit does not compile, each with
    Lean's first error; and how many circuits were compiled."""
    files = []
    for k, name in enumerate(names):
        m2l = ModuleToLean4(CORPUS[name]())
        try:
            body = f"{m2l.to_lean_functional()}\n\n{m2l.to_lean_circ()}"
        except ValueError as e:
            if not str(e).startswith("No Lean expression mapping"):
                raise
            continue
        path = tmp_path / f"circ{k}.lean"
        path.write_text(f"-- {name}\nimport Relational.Circ\n\n{LINTERS_OFF}\n\n{body}\n")
        files.append((name, path))

    def lean(path: Path) -> str:
        return subprocess.run(["lake", "env", "lean", str(path)], cwd=_LEAN_DIR,
                              capture_output=True, text=True, timeout=1800).stdout

    with ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool:
        outputs = list(pool.map(lambda f: lean(f[1]), files))

    bad = {}
    for (name, _path), out in zip(files, outputs):
        errors = [ln for ln in out.splitlines() if ": error" in ln]
        if errors:
            bad[name] = errors[0]
    return bad, len(files)


@pytest.mark.slow
def test_each_circuit_is_proved_to_be_its_functional_encoding(tmp_path):
    _build_relational()
    bad, compiled = _unproved(sorted(CORPUS), tmp_path)
    assert compiled, "nothing to compile"
    new = {name: bad[name] for name in sorted(set(bad) - _KNOWN)}
    assert not new, f"{len(new)} circuit(s) not proved:\n" + "\n".join(
        f"{name}: {error}" for name, error in new.items())
    assert _KNOWN <= set(bad), (
        f"{sorted(_KNOWN - set(bad))} compile now: drop them from `_KNOWN`")


@pytest.mark.slow
def test_a_wrong_box_is_caught(tmp_path, monkeypatch):
    """The check can fail: with `Sub`'s `Box` adding, the countdown's
    circuit counts up, and its `update` is no longer the functional one."""
    from reactor.lean.lowering import ops

    for key, row in list(ops._BY_KEY.items()):
        if key[1] == "Sub":
            monkeypatch.setitem(ops._BY_KEY, key,
                                dataclasses.replace(row, box="Box.add"))
    _build_relational()
    bad, compiled = _unproved(["limits/m_countdown"], tmp_path)
    assert compiled == 1
    assert "limits/m_countdown" in bad
