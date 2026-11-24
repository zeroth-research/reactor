"""The sorts, as defined by the bindings in ``reactor.tensor.sort``: ``Real``, ``Int``,
``Bool`` and ``Word`` matrices, and the singleton ``Zero``. Each is built from its shape,
``Real([rows, cols])``, with a width for words, ``Word(width, [rows, cols])``, and matches
positionally the same way (``case Word(width, shape):``).
"""

import torch

from .reactor import tensor as _tensor
from .reactor.tensor.sort import Real, Int, Bool, Word, Zero


# Exhaustive on purpose: a new sort must be added here, never given a default.
_KINDS = (
    (Bool, torch.bool, "boolean"),
    (Int, torch.int64, "integer"),
    (Real, torch.float32, "real"),
    (Word, torch.int64, "word"),
)


def shape(sort) -> list:
    """The ``[rows, cols]`` shape of a sort."""
    match sort:
        case Real(s) | Int(s) | Bool(s) | Word(_, s):
            return list(s)
    raise TypeError(f"sort has no shape: {sort}")


def with_shape(sort, new_shape) -> Real | Int | Bool | Word:
    """The sort of the same family as ``sort`` (and width, for words, and degree, for
    reals), of another shape."""
    match sort:
        case Real(_, degree=d):
            return Real(new_shape, degree=d)
        case Int():
            return Int(new_shape)
        case Bool():
            return Bool(new_shape)
        case Word(w):
            return Word(w, new_shape)
    raise TypeError(f"sort has no shape: {sort}")


def width(sort) -> int:
    """The width of a word sort."""
    match sort:
        case Word(w):
            return w
    raise TypeError(f"not a word sort: {sort}")


def family(sort) -> str:
    """A key identifying a sort's family (with its width, for words), ignoring its shape:
    two values may combine only if their families match."""
    match sort:
        case Real():
            return "Real"
        case Int():
            return "Int"
        case Bool():
            return "Bool"
        case Word(w):
            return f"Word{w}"
    raise TypeError(f"unknown sort: {sort}")


def namespace(sort):
    """The namespace of the operations on the family of ``sort``: ``reactor.tensor.real``
    for ``Real``, ``.int`` for ``Int``, ``.bool`` for ``Bool`` and ``.word`` for ``Word``."""
    match sort:
        case Real():
            return _tensor.real
        case Int():
            return _tensor.int
        case Bool():
            return _tensor.bool
        case Word():
            return _tensor.word
    raise TypeError(f"no operations on {sort}")


def tensor_for(value, sort):
    """``value`` as a tensor of the kind ``sort`` requires (``sort`` may be a family or an
    instance). A value the conversion would alter is rejected, never silently cast."""
    for fam, dtype, name in _KINDS:
        if sort is fam or isinstance(sort, fam):
            break
    else:
        raise TypeError(f"no tensor representation for sort {sort!r}")

    given = value if isinstance(value, torch.Tensor) else torch.tensor(value)
    if dtype is torch.bool and given.dtype is not torch.bool:
        raise TypeError(
            f"only a boolean value can be a {name} constant: "
            f"{given.flatten().tolist()} would collapse to True/False"
        )
    if dtype is torch.int64 and given.is_floating_point():
        raise TypeError(
            f"{name} constants cannot hold a floating-point value: "
            f"{given.flatten().tolist()} would be truncated"
        )
    return given.to(dtype)



def T(sort):
    """The tangent sort `T(sort)`: the sort of the rate of change of the values of `sort`,
    one degree higher for a real, and the singleton `Zero` for a discrete sort."""
    return sort.tangent()


# Required, not boilerplate: without it `from .sort import *` in the package __init__
# re-exports `torch`, replacing the `reactor.torch` submodule with the torch library.
__all__ = ["Real", "Int", "Bool", "Word", "Zero", "T", "tensor_for"]
