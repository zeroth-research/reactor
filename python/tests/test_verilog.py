"""Translated SystemVerilog designs (`reactor.verilog`) run in `reactor.eval`
compared with the `verilog` crate's own simulator, on random input traces.

The designs: the examples of the `verilog` crate, and the small designs its
tests accept, if they were written out by running
`V2RM_SNIPPETS=$PWD/target/v2rm-snippets cargo test -p verilog` in the
repository root. Designs that `reactor.eval` cannot run are skipped, with
the reason: words of 64 bits or more, `HAVOC` terms (`eval_itype` has no
case for them), and written-out test designs that load a memory file their
test made in a temporary directory.
"""

import random
import warnings
import zlib
from pathlib import Path

import pytest
import torch

import reactor.verilog as verilog
from reactor import X
from reactor.eval import execute_init, execute_update

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "verilog" / "examples"
SNIPPETS = ROOT / "target" / "v2rm-snippets"
ROUNDS = 8
EXTERNAL = {"input", "clock", "init", "blackbox"}


def designs():
    paths = sorted(EXAMPLES.glob("*.sv"))
    if SNIPPETS.is_dir():
        paths += sorted(SNIPPETS.glob("*.sv"))
    return paths


def _havoc(module):
    return any(
        "havoc" in type(t.itype).__name__.lower()
        for a in module.atoms
        for t in list(a.init) + list(a.update)
    )


def _translate(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", verilog.VerilogWarning)
        try:
            return verilog.translate(path)
        except verilog.UnsupportedError as e:
            pytest.skip(f"not supported: {e}")
        except verilog.UserError as e:
            # a written-out test design loading a file its test made
            if path.parent == SNIPPETS and "cannot read" in str(e):
                pytest.skip(f"its memory file is gone: {e}")
            raise


def _run_eval(module, names, rounds):
    """Each round's next value of every variable, by name, from `reactor.eval`."""
    state, out = {}, []
    for k, inputs in enumerate(rounds):
        for name, value in inputs.items():
            state[X(names[name])] = torch.tensor([[value]], dtype=torch.int64)
        (execute_init if k == 0 else execute_update)(state, module.atoms)
        out.append({n: int(state[X(v)].item()) for n, v in names.items()})
        for v in names.values():
            state[v] = state[X(v)]
    return out


@pytest.mark.parametrize("path", designs(), ids=lambda p: p.stem)
def test_eval_matches_rm_sim(path):
    module, names, roles = _translate(path)
    if _havoc(module):
        pytest.skip("HAVOC term: reactor.eval cannot evaluate it")
    rand = random.Random(zlib.crc32(path.read_bytes()))
    rounds = [
        {
            n: rand.getrandbits(_width(names[n]))
            for n, r in sorted(roles.items())
            if r in EXTERNAL
        }
        for _ in range(ROUNDS)
    ]
    ours = verilog._simulate(path, rounds)
    theirs = _run_eval(module, names, rounds)
    for k, (o, t) in enumerate(zip(ours, theirs)):
        assert t == o, f"round {k}, inputs {rounds[k]}"


def _width(var):
    return var.dtype[0]


def _write(tmp_path, text):
    path = tmp_path / "t.sv"
    path.write_text(text)
    return path


def _wide(w):
    return (
        f"module t(input logic [{w - 1}:0] a, b, output logic [{w - 1}:0] y);\n"
        "  assign y = a + b;\n"
        "endmodule\n"
    )


def test_63_bits(tmp_path):
    path = _write(tmp_path, _wide(63))
    module, names, roles = verilog.translate(path)
    a, b = (1 << 63) - 1, 5
    rounds = [{"a": a, "b": b}]
    assert _run_eval(module, names, rounds) == verilog._simulate(path, rounds)
    assert roles == {"a": "input", "b": "input", "y": "output"}


def test_64_bits_refused(tmp_path):
    with pytest.raises(verilog.UnsupportedError, match=r"`a` \(64 bits\)"):
        verilog.translate(_write(tmp_path, _wide(64)))


def test_errors_and_warnings(tmp_path):
    with pytest.raises(verilog.UserError, match=r"t\.sv:2:\d+: "):
        verilog.translate(_write(tmp_path, "module t(output logic y);\n  assign y = z;\nendmodule\n"))
    with pytest.warns(verilog.VerilogWarning, match="never read"):
        verilog.translate(_write(tmp_path, "module t(input logic a, output logic y);\n  assign y = 1;\nendmodule\n"))


def test_params_and_top(tmp_path):
    path = _write(
        tmp_path,
        "module c #(parameter W = 2)(input logic [W-1:0] a, output logic [W-1:0] y);\n"
        "  assign y = a;\nendmodule\n"
        "module d(output logic y);\n  assign y = 0;\nendmodule\n",
    )
    _, names, _ = verilog.translate([path], top="c", params={"W": 5})
    assert _width(names["a"]) == 5
