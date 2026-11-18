"""Type stubs for `reactor.op`: the operations of each theory, and the logics.

The operations of each theory are in its module, as in Rust's `theory::op`: `real`,
`int`, `bool` and `word`. Each is a class that takes the fields of its operation, as
`real.Add(m, n)`, and lists them in `__match_args__`, so that `match` takes it apart:
`case real.Linear(a, b):`.

The logics select the theory of a frontend, as in `builder_for(LIA)`, and tell which
operations they hold.
"""

from . import bool as bool, int as int, real as real, word as word

# An operation of any theory.
type Op = real.Op | int.Op | bool.Op | word.Op


class NRA:
    """Non-linear real arithmetic, `QF_NRA`: the operations of `real` and `bool`."""

    @staticmethod
    def contains(op: Op) -> bool: ...


class LRA:
    """Linear real arithmetic, `QF_LRA`: the linear operations of `real`, and `bool`."""

    @staticmethod
    def contains(op: Op) -> bool: ...


class NIA:
    """Non-linear integer arithmetic, `QF_NIA`: the operations of `int` and `bool`."""

    @staticmethod
    def contains(op: Op) -> bool: ...


class LIA:
    """Linear integer arithmetic, `QF_LIA`: the linear operations of `int`, and `bool`."""

    @staticmethod
    def contains(op: Op) -> bool: ...


class Prop:
    """Propositional logic: the operations of `bool`."""

    @staticmethod
    def contains(op: Op) -> bool: ...


class BV:
    """Fixed-size bit-vectors, `QF_BV`: the operations of `word` and `bool`."""

    @staticmethod
    def contains(op: Op) -> bool: ...
