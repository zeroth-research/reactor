from .reactor import Real, Int, Bool, Word, Zero
from abc import ABC as _ABC

import torch


class Sort(_ABC):
    """The sorts: `Real`, `Int`, `Bool`, `Word` and `Zero`, the classes of the
    extension, which are virtual subclasses of `Sort`."""


for _sort in (Real, Int, Bool, Word, Zero):
    Sort.register(_sort)

# Exhaustive on purpose: a new sort must be added here, never given a default.
# `Zero` has a single value, and no constants.
_KINDS = (
    (Bool, torch.bool, "boolean"),
    (Int, torch.int64, "integer"),
    (Real, torch.float32, "real"),
    (Word, torch.int64, "word"),
)


def tensor_for(value, sort):
    """``value`` as a tensor of the kind ``sort`` requires (``sort`` may be a family or an
    instance). A value the conversion would alter is rejected, never silently cast."""
    for family, dtype, name in _KINDS:
        if sort is family or isinstance(sort, family):
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


def shape(sort) -> list:
    """The ``[rows, cols]`` shape of a sort; `Zero` has none."""
    match sort:
        case Real(s) | Int(s) | Bool(s) | Word(_, s):
            return list(s)
    raise TypeError(f"sort has no shape: {sort!r}")


def with_shape(sort, shape: list):
    """``sort`` with the given shape, keeping its degree or width."""
    match sort:
        case Real(_, degree):
            return Real(shape, degree)
        case Int(_):
            return Int(shape)
        case Bool(_):
            return Bool(shape)
        case Word(width, _):
            return Word(width, shape)
    raise TypeError(f"sort has no shape: {sort!r}")


# Required, not boilerplate: without it `from .sort import *` in the package __init__
# re-exports `torch`, replacing the `reactor.torch` submodule with the torch library.
__all__ = ["Sort", "tensor_for", "Real", "Int", "Bool", "Word", "Zero"]
