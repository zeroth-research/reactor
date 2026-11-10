"""`certify` transcripts, pinned byte for byte.

Whole in-process `certify` runs that infer the certificate -- stdout, the
two certificate files, `artifacts/` -- compared with the snapshots under
`tests/golden/pipeline/`; set `REACTOR_UPDATE_GOLDEN=1` to rewrite them, and
only for a change to the output that is intended. The runs that do not
infer are `reactor`'s, pinned the same way in `python/tests/
test_pipeline_golden.py`, as is the codegen they share.

No run reaches an LLM: `make_client` is replaced in both modules that build
one, and the `ai-cegis` run has every predicate fixed, so it never asks.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest
import torch  # noqa: F401  (before the reactor extension)

pytest.importorskip("cvc5")

GOLDEN = Path(__file__).parent / "golden" / "pipeline"
# The CLI fixtures are `reactor`'s, in the `python/` project beside this one.
FIXTURES = Path(__file__).resolve().parents[2] / "python" / "tests" / "fixtures"
UPDATE = os.environ.get("REACTOR_UPDATE_GOLDEN") == "1"


def _check(name: str, text: str) -> None:
    path = GOLDEN / name
    if UPDATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return
    assert path.exists(), f"no snapshot {path}; run with REACTOR_UPDATE_GOLDEN=1"
    assert text == path.read_text(), f"{name} differs from its snapshot"


# ══════════════════════════════════════════════════════════════════════════
# `certify`, in process
# ══════════════════════════════════════════════════════════════════════════

COUNTER = "counter"
COUNTDOWN = "svcomp_countdown"
BOUNDED = "svcomp_bounded_input"

_AI_REPLIES = {
    "generate": "INVARIANT: fun s => (s 0 0) ≥ 0 ∧ (s 0 0) ≤ 9\n"
                "RANKING: fun s => (if (s 0 0) = 0 then 0 else 10 - (s 0 0)).toNat",
    "verify": "CORRECT",
}

# (name, fixture, arguments). `-o` and `-p` are added.
RUNS = [
    ("smt_linear", COUNTDOWN, [
        "--buchi", "(= (X @intf.0) 0)", "--invariant", "(and (>= (X @intf.0) 0) (<= (X @intf.0) 100))",
        "--infer", "smt-linear", "--pre-check", "cvc5", "--smt-tactics", "cvc5",
    ]),
    ("houdini", COUNTDOWN, [
        "--safety", "(<= (X @intf.0) 100)", "--infer", "houdini",
        "--pre-check", "cvc5", "--smt-tactics", "cvc5",
    ]),
    ("nuterm", COUNTDOWN, [
        "--buchi", "(= (X @intf.0) 0)", "--infer", "nuterm",
        "--pre-check", "cvc5", "--smt-tactics", "cvc5",
    ]),
    ("nuterm_pre", BOUNDED, [
        "--safety", "(>= (X @intf.0) 0)", "--pre", "(>= (X @extl.0) 0)", "--infer", "nuterm",
        "--pre-check", "cvc5", "--smt-tactics", "cvc5",
    ]),
    ("ai_cegis_fixed", COUNTER, [
        "--buchi", "(= (X @intf.0) 0)",
        "--invariant", "(and (>= (X @intf.0) 0) (<= (X @intf.0) 9))",
        "--ranking", "(ite (<= (X @intf.0) 200) (ite (= (X @intf.0) 0) 0 (- 10 (X @intf.0))) 999)",
        "--infer", "ai-cegis", "--pre-check", "cvc5", "--smt-tactics", "cvc5",
    ]),
    ("ai", COUNTER, [
        "--buchi", "(= (X @intf.0) 0)", "--infer", "ai",
        "--pre-check", "cvc5", "--smt-tactics", "cvc5",
    ]),
    ("sygus", COUNTDOWN, [
        "--safety", "(<= (X @intf.0) 100)", "--infer", "sygus",
        "--pre-check", "cvc5", "--smt-tactics", "cvc5",
    ]),
    # A short declared limit: Vampire answers this one in about a second
    # under 10 and in about six under the default 120.
    ("vampire", COUNTDOWN, [
        "--safety", "(<= (X @intf.0) 100)", "--infer", "vampire",
        "--vampire-timeout", "10",
        "--pre-check", "cvc5", "--smt-tactics", "cvc5",
    ]),
]

# `ModuleSMT`s built per run. The pre-check, both renderings and a route on
# `InferInput.smt` share the run's `smt.encoding.lazy` encoding.
ENCODINGS = {
    "smt_linear": 1, "houdini": 1, "nuterm": 1, "nuterm_pre": 1,
    "ai_cegis_fixed": 1, "ai": 1, "sygus": 1, "vampire": 1,
}


def _has_vampire() -> bool:
    from reactor.lean.core.errors import Refused
    from verith.common.vampire import resolve_vampire

    try:
        resolve_vampire(None)
    except Refused:
        return False
    return True


def _runs() -> list:
    out = []
    for name, fixture, args in RUNS:
        marks = ()
        if "vampire" in args:
            marks = pytest.mark.skipif(
                not _has_vampire(),
                reason="no Vampire binary: set $VAMPIRE or put `vampire` on PATH")
        out.append(pytest.param(name, fixture, args, id=name, marks=marks))
    return out


def _offline(monkeypatch) -> None:
    """Both LLM routes answer from `_AI_REPLIES` and never open a client."""
    from verith.ai import engine as ai
    from verith.ai_cegis import engine as cegis

    def client(base_url, model):
        def chat(system: str, user: str) -> str:
            return _AI_REPLIES[
                "verify" if system is ai.VERIFY_SYSTEM else "generate"]
        return chat

    def no_client(base_url, model):
        def chat(system: str, user: str) -> str:
            raise AssertionError("ai-cegis with every predicate fixed asked")
        return chat

    monkeypatch.setattr(ai, "make_client", client)
    monkeypatch.setattr(cegis, "make_client", no_client)


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _normalise(text: str, out: Path) -> str:
    text = _ANSI.sub("", text)
    text = text.replace(str(out), "<OUT>").replace(str(FIXTURES), "<FIX>")
    text = re.sub(r"#\d+", "#N", text)                     # wire ids
    text = re.sub(r"\s+\d+ ms\b", " N ms", text)            # verdict timings
    text = re.sub(r"\b\d+(\.\d+)? s\b", "N s", text)        # route timings
    return text


def _index(path: Path) -> str:
    if not path.exists():
        return "(none)\n"
    entries = json.loads(path.read_text())
    for e in entries:
        e.pop("created", None)
        e["module_digest"] = "<digest>"      # hashes wire ids
    return json.dumps(entries, indent=2, sort_keys=True) + "\n"


def _transcript(monkeypatch, capsys, tmp_path, fixture, args) -> str:
    from certify.pipeline import main

    out = tmp_path / "out"
    argv = [a.replace("{out}", str(out)) for a in args]
    monkeypatch.setattr(sys, "argv", [
        "certify", str(FIXTURES / f"{fixture}.py"), *argv,
        "-o", str(out), "-p", "G",
    ])
    _offline(monkeypatch)
    try:
        main()
        status = "exit 0"
    except SystemExit as e:
        status = f"exit {e.code}"
    captured = capsys.readouterr()
    project = out / "G"

    def read(rel: str) -> str:
        p = project / rel
        return p.read_text() if p.exists() else "(none)\n"

    parts = [
        f"=== {status}",
        "=== stdout", captured.out,
        "=== stderr", captured.err,
        "=== Certificate/Data.lean", read("Certificate/Data.lean"),
        "=== Certificate/Certificate.lean", read("Certificate/Certificate.lean"),
        "=== artifacts/index.json", _index(project / "artifacts" / "index.json"),
    ]
    if (out / "Cert.lean").exists():
        parts += ["=== Cert.lean", (out / "Cert.lean").read_text()]
    # Every file the run left in `artifacts/` but the two the index already
    # pins: the SMT-LIB it asked (`system.smt2`, `obligation-*.smt2`), what
    # it found and the notes it left.
    store = project / "artifacts"
    if store.is_dir():
        for path in sorted(store.iterdir()):
            if path.name not in ("index.json", "README.md"):
                parts += [f"=== artifacts/{path.name}", path.read_text()]
    return _normalise("\n".join(parts), out)


def _count_encodings(monkeypatch) -> list:
    """Counts each `ModuleSMT` built from here on, in a one-element list."""
    from reactor.lean.smt.module import ModuleSMT

    built = [0]
    post_init = ModuleSMT.__post_init__

    def counted(self):
        built[0] += 1
        post_init(self)

    monkeypatch.setattr(ModuleSMT, "__post_init__", counted)
    return built


@pytest.mark.parametrize("name,fixture,args", _runs())
def test_certify_run_is_unchanged(name, fixture, args, monkeypatch, capsys,
                                 tmp_path):
    built = _count_encodings(monkeypatch)
    _check(f"certify-{name}.golden",
           _transcript(monkeypatch, capsys, tmp_path, fixture, args))
    assert built[0] == ENCODINGS.get(name), (
        f"{name} encoded the module {built[0]} times")
