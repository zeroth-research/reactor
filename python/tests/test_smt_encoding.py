"""`ModuleEncoding` keeps nothing that refers back to it, and a cycle that
does still exits cleanly.

An encoding is retained until exit, and at exit the collector finalises a
cycle through it in arbitrary order with its `TermManager`; cvc5 crashes
freeing a term after its manager. The run has printed everything by then
and only its exit status says -11. The two ways an encoding kept a raised
exception -- whose traceback reaches the frame that raised it, and that
frame's `self` -- are checked by asking the collector who holds the
encoding; the pinned manager, by putting one of them back in a child
process and reading its status.
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


def _frames_holding(enc) -> list:
    ours = (sys._getframe(0), sys._getframe(1))
    gc.collect()
    return [r for r in gc.get_referrers(enc)
            if isinstance(r, types.FrameType) and r not in ours]


def test_a_predicate_it_could_not_read_holds_no_frame_of_it():
    """The second read is answered from what the first kept."""
    enc = ModuleEncoding(load_module(MODS / "m_relu_net.py"))
    for _ in range(2):
        with pytest.raises(PredicateRefused, match="cannot read"):
            enc.parse("(ite s0[0][0] == 0 0 1)", "ranking", want="Int")
    assert _frames_holding(enc) == []


def test_a_translation_that_failed_holds_no_frame_of_it():
    """`m_transpose` has no cvc5 term for its round; asked twice, it is
    refused twice, the same way."""
    enc = ModuleEncoding(load_module(MODS / "m_transpose.py"))
    for _ in range(2):
        with pytest.raises(ValueError, match="unsupported IType Transpose"):
            enc.next_state(enc.state, enc.extl_latched, enc.extl_next)
    assert _frames_holding(enc) == []


_KEPT_FAILURE = r"""
import sys
from reactor.lean.inputs.python import load_module
from reactor.lean.smt.encoding import ModuleEncoding

def kept_block(self, name):
    # `_block` as it was: the failure kept, traceback and all, and raised
    # again -- a cycle through the encoding that holds cvc5 terms.
    hit = self._blocks.get(name)
    if hit is None:
        try:
            hit = (self.msmt.init_state(self.extl_next) if name == "init" else
                   self.msmt.update_state(self.state, self.extl_latched,
                                          self.extl_next))
        except Exception as e:
            hit = e
        self._blocks[name] = hit
    if isinstance(hit, Exception):
        raise hit
    return hit

ModuleEncoding._block = kept_block
for path in sys.argv[1:]:
    enc = ModuleEncoding(load_module(path))
    for _ in range(2):
        try:
            enc.next_state(enc.state, enc.extl_latched, enc.extl_next)
        except ValueError:
            print("refused", flush=True)
"""


def test_a_cycle_through_an_encoding_still_exits_cleanly():
    """With `_block` keeping its failures again, two encodings exited -11
    after their refusals; their managers are pinned now, and the cycle is
    only memory."""
    mods = [str(MODS / f"{m}.py") for m in ("m_transpose", "m_uninterp")]
    r = subprocess.run([sys.executable, "-c", _KEPT_FAILURE, *mods],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, (r.returncode, r.stdout[-2000:], r.stderr[-2000:])
    assert r.stdout.splitlines() == ["refused"] * 4
