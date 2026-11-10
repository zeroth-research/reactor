"""`verith`: a module and a claim in, the certificate printed.

Run in process, as `main(argv)`; `certify`'s own tests are
`test_certify_cli.py`.
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import pytest

pytest.importorskip("cvc5")

# The CLI fixtures are `reactor`'s, in the `python/` project beside this one.
FIXTURES = Path(__file__).resolve().parents[2] / "python" / "tests" / "fixtures"
COUNTDOWN = FIXTURES / "svcomp_countdown.py"
COUNTER = FIXTURES / "counter.py"
BUCHI = ["--buchi", "(= (X @intf.0) 0)"]


def verith(capsys, *argv) -> tuple[int, str, str]:
    """`verith *argv`, in process: its exit status, stdout and stderr."""
    from verith.cli import main

    try:
        main([str(a) for a in argv])
        code = 0
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        if not isinstance(e.code, int):
            print(e.code, file=sys.stderr)
    out = capsys.readouterr()
    return code, out.out, out.err


def reactor(monkeypatch, capsys, *argv) -> str:
    """`reactor *argv`, in process; its stdout."""
    from reactor.lean.pipeline import main

    monkeypatch.setattr(sys, "argv", ["reactor", *map(str, argv)])
    main()
    return capsys.readouterr().out


def shown_flags(out: str) -> list[list[str]]:
    """The `--invariant` / `--ranking` lines of a printed answer."""
    shown = out.split(".. The certificate, in the input's names:\n")[1]
    return [shlex.split(line) for line in shown.splitlines()[:2]]


def test_the_answer_is_printed_as_reactor_takes_it(capsys, monkeypatch,
                                                   tmp_path):
    code, out, err = verith(capsys, COUNTDOWN, *BUCHI, "--infer", "houdini")
    assert code == 0, err
    flags = shown_flags(out)
    assert [f for f, _ in flags] == ["--invariant", "--ranking"]

    # What verith printed is what reactor takes, and cvc5 agrees with it.
    out = reactor(monkeypatch, capsys, COUNTDOWN, *BUCHI,
                  *(w for pair in flags for w in pair), "--pre-check", "cvc5",
                  "--cert-file", tmp_path / "C.lean")
    verdicts = [ln.split()[:2] for ln in out.splitlines()
                if ln.strip().startswith(("init_inv", "step_inv", "hrank"))]
    assert verdicts == [["init_inv", "holds"], ["step_inv", "holds"],
                        ["hrank", "holds"]], out


def test_json_is_the_answer_alone_on_stdout(capsys):
    code, out, err = verith(capsys, COUNTDOWN, *BUCHI, "--infer", "houdini",
                            "--json")
    assert code == 0, err
    answer = json.loads(out)
    assert (answer["route"], answer["kind"], answer["returns"]) == \
        ("houdini", "buchi", "smt")
    assert answer["invariant"] and answer["ranking"]
    assert "@intf.0" in answer["invariant"]           # the input's names
    assert "s0" in answer["smt"]["inv"]                # the internal ones
    assert answer["lean"] is None and answer["files"] == []
    assert "[houdini]" in err                          # the log is stderr's


def test_a_route_that_reads_the_project_needs_one(capsys):
    code, _, err = verith(capsys, COUNTER, *BUCHI, "--infer", "ai-cegis")
    assert code == 2
    assert "--infer ai-cegis reads the generated Lean project" in err
    assert "--project" in err


def test_artifacts_mean_a_project(capsys):
    code, _, err = verith(capsys, COUNTDOWN, *BUCHI, "--infer", "houdini",
                          "--artifacts", "reset")
    assert code == 2
    assert "no --project was given" in err


def test_a_project_is_a_generated_one(capsys, tmp_path):
    code, _, err = verith(capsys, COUNTDOWN, *BUCHI, "--infer", "houdini",
                          "--project", tmp_path)
    assert code == 2
    assert "is not a generated Lean project" in err


def test_a_project_of_another_module_is_refused(capsys, monkeypatch, tmp_path):
    reactor(monkeypatch, capsys, COUNTER, *BUCHI, "-o", tmp_path, "-p", "P")
    code, _, err = verith(capsys, COUNTDOWN, *BUCHI, "--infer", "houdini",
                          "--project", tmp_path / "P")
    assert code == 1
    assert "was not generated from this module" in err


def test_with_a_project_the_answer_is_recorded_and_nothing_rewritten(
        capsys, monkeypatch, tmp_path):
    """The route reads and writes the project's `artifacts/`, as under
    `certify`; the certificate stays `reactor`'s to write."""
    reactor(monkeypatch, capsys, COUNTDOWN, *BUCHI, "-o", tmp_path, "-p", "P")
    project = tmp_path / "P"
    data = (project / "Certificate" / "Data.lean").read_text()

    code, out, err = verith(capsys, COUNTDOWN, *BUCHI, "--infer", "houdini",
                            "--project", project)
    assert code == 0, err
    assert (project / "Certificate" / "Data.lean").read_text() == data
    index = json.loads((project / "artifacts" / "index.json").read_text())
    inferred = {e["role"] for e in index
                if "Inferred by this run's route." in e.get("what", "")}
    assert inferred == {"inv", "ranking"}


def test_an_llm_route_reads_the_project_it_is_given(capsys, monkeypatch,
                                                     tmp_path):
    """`--infer ai` prompts on the project's functional encoding and answers
    in Lean, which is printed as it came."""
    from verith.ai import engine as ai

    def client(base_url, model):
        def chat(system: str, user: str) -> str:
            assert "def update" in user or system is ai.VERIFY_SYSTEM
            if system is ai.VERIFY_SYSTEM:
                return "CORRECT"
            return ("INVARIANT: fun s => (s 0 0) ≥ 0 ∧ (s 0 0) ≤ 9\n"
                    "RANKING: fun s => (if (s 0 0) = 0 then 0 else 10 - (s 0 0)).toNat")
        return chat

    monkeypatch.setattr(ai, "make_client", client)
    reactor(monkeypatch, capsys, COUNTER, *BUCHI, "-o", tmp_path, "-p", "P")
    code, out, err = verith(capsys, COUNTER, *BUCHI, "--infer", "ai",
                            "--project", tmp_path / "P")
    assert code == 0, err
    assert ".. The certificate, in Lean:" in out
    assert "inv := fun s => (s 0 0) ≥ 0" in out
