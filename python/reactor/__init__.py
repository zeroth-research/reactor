from .reactor import *
import sys as _sys

# The operations of each theory, and the logics, are native submodules:
# registered here, so that `from reactor.op.real import Add` imports them.
_sys.modules[__name__ + ".op"] = op
for _theory in ("real", "int", "bool", "word"):
    _sys.modules[f"{__name__}.op.{_theory}"] = getattr(op, _theory)
from .op import NRA, LRA, NIA, LIA, Prop, BV

from .builder import (
    builder_for,
    LRATermBuilder,
    LIATermBuilder,
    BVTermBuilder,
    TheoryError,
    NonLinearError,
)
from .gym import Env
from .smv import parse_smv

# Submodule access: from reactor.sugar import Module ; from reactor.gym import Env
from . import sugar as sugar
from . import gym as gym
from . import torch as torch
from .smt import z3 as z3
from .sort import *
from .expr import expr

_sys.modules[__name__ + ".z3"] = z3
