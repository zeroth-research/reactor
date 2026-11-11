"""`ModuleEncoding` keeps nothing that refers back to it.

An encoding is retained until exit, and at exit the collector finalises a
cycle through it in arbitrary order with its `TermManager`; cvc5 crashes
freeing a term after its manager. The run has printed everything by then
and only its exit status says -11, so the two ways an encoding kept a
raised exception -- whose traceback reaches the frame that raised it, and
that frame's `self` -- are checked here: one by a child process's status,
the other by asking the collector who holds the encoding.
"""

import gc
import subprocess
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("cvc5")

from reactor.lean.core.errors import PredicateRefused
from reactor.lean.inputs.python import load_module
from reactor.lean.smt.encoding import ModuleEncoding

MODS = Path(__file__).parent / "limits" / "mods"

_TRANSLATE = r"""
import sys
from reactor.lean.inputs.python import load_module
from reactor.lean.smt.encoding import ModuleEncoding

for path in sys.argv[1:]:
    enc = ModuleEncoding(load_module(path))
    for _ in range(2):
        try:
            enc.next_state(enc.state, enc.extl_latched, enc.extl_next)
        except ValueError:
            print("refused", flush=True)
"""


def test_modules_cvc5_has_no_term_for_exit_cleanly():
    """Two encodings whose translation failed exited -11 after their
    refusals: each failure was kept, traceback and all."""
    mods = [str(MODS / f"{m}.py") for m in ("m_transpose", "m_uninterp")]
    r = subprocess.run([sys.executable, "-c", _TRANSLATE, *mods],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, (r.returncode, r.stdout[-2000:], r.stderr[-2000:])
    assert r.stdout.splitlines() == ["refused"] * 4


def test_a_predicate_it_could_not_read_holds_no_frame_of_it():
    """The second read is answered from what the first kept."""
    enc = ModuleEncoding(load_module(MODS / "m_relu_net.py"))
    for _ in range(2):
        with pytest.raises(PredicateRefused, match="cannot read"):
            enc.parse("(ite s0[0][0] == 0 0 1)", "ranking", want="Int")
    here = sys._getframe()
    gc.collect()
    frames = [r for r in gc.get_referrers(enc)
              if isinstance(r, types.FrameType) and r is not here]
    assert frames == []
