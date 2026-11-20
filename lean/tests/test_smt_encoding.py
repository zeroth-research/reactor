"""`ModuleEncoding` keeps nothing that refers back to it.

An encoding is retained until exit, and at exit the collector finalises a
cycle through it in arbitrary order with its `TermManager`; cvc5 crashes
freeing a term after its manager. The run has printed everything by then
and only its exit status says -11. A predicate it could not read keeps no
raised exception -- whose traceback reaches the frame that raised it, and
that frame's `self` -- which is checked by asking the collector who holds
the encoding.
"""

import gc
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("cvc5")

from reactor_lean.core.errors import PredicateRefused
from reactor_lean.inputs.python import load_module
from reactor_lean.smt.encoding import ModuleEncoding

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
