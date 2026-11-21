from .reactor import *
import sys as _sys

# The namespaces of the operations and sorts, `reactor.tensor` and those within it, are
# importable by their names in this package too (`import reactor.tensor.real`), not only
# by those in the extension module (`reactor.reactor.tensor.real`).
for _name, _module in list(_sys.modules.items()):
    if _name.startswith(__name__ + ".reactor.tensor"):
        _sys.modules[__name__ + _name[len(__name__ + ".reactor"):]] = _module

from .builder import TermBuilder, TheoryError, builder_for
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
