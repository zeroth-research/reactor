"""Type stubs for `reactor.op.bool`: the operations of propositional logic, on boolean matrices.

Generated from the operations of `theory::op`; keep them in step with it.
"""

from typing import Any

# A matrix that an operation holds: a `torch.Tensor` or a `jax.Array` (with JAX), whose
# storage it shares, or a list of rows, which it copies.
Matrix = Any


class Const:
    """A constant boolean matrix, held in a `Matrix` of boolean elements."""

    __match_args__ = ("tensor",)
    def __init__(self, tensor: Matrix) -> None: ...
    @property
    def tensor(self) -> Matrix: ...

class And:
    """Pointwise conjunction `x ∧ y` of `[m, n]` matrices."""

    __match_args__ = ("m", "n")
    def __init__(self, m: int, n: int) -> None: ...
    @property
    def m(self) -> int: ...
    @property
    def n(self) -> int: ...

class Or:
    """Pointwise disjunction `x ∨ y` of `[m, n]` matrices."""

    __match_args__ = ("m", "n")
    def __init__(self, m: int, n: int) -> None: ...
    @property
    def m(self) -> int: ...
    @property
    def n(self) -> int: ...

class Xor:
    """Pointwise exclusive disjunction `x ⊕ y` of `[m, n]` matrices."""

    __match_args__ = ("m", "n")
    def __init__(self, m: int, n: int) -> None: ...
    @property
    def m(self) -> int: ...
    @property
    def n(self) -> int: ...

class Not:
    """Pointwise negation `¬x` of `[m, n]` matrices."""

    __match_args__ = ("m", "n")
    def __init__(self, m: int, n: int) -> None: ...
    @property
    def m(self) -> int: ...
    @property
    def n(self) -> int: ...

class Id:
    """The identity `x ↦ x` on `[m, n]` matrices."""

    __match_args__ = ("m", "n")
    def __init__(self, m: int, n: int) -> None: ...
    @property
    def m(self) -> int: ...
    @property
    def n(self) -> int: ...

class Ite:
    """If-then-else `if c then x else y` of `[m, n]` matrices, on a single boolean `c`, which picks either matrix whole."""

    __match_args__ = ("m", "n")
    def __init__(self, m: int, n: int) -> None: ...
    @property
    def m(self) -> int: ...
    @property
    def n(self) -> int: ...

class Havoc:
    """An arbitrary boolean matrix of shape `[m, n]`, chosen nondeterministically."""

    __match_args__ = ("m", "n")
    def __init__(self, m: int, n: int) -> None: ...
    @property
    def m(self) -> int: ...
    @property
    def n(self) -> int: ...

class Uninterpreted:
    """A boolean matrix of shape `[m, n]` of unknown meaning, named by the given symbol, as `Uninterpreted(symbol, m, n)`: e.g., the result of a call that the frontend cannot translate."""

    __match_args__ = ("symbol", "m", "n")
    def __init__(self, symbol: str, m: int, n: int) -> None: ...
    @property
    def symbol(self) -> str: ...
    @property
    def m(self) -> int: ...
    @property
    def n(self) -> int: ...

class Zero:
    """The only value of the sort `Zero`: the rate of change of discrete values."""

    __match_args__ = ()
    def __init__(self) -> None: ...


# The operations of the theory, as `Term.itype` may hold them.
type Op = (
      Const
    | And
    | Or
    | Xor
    | Not
    | Id
    | Ite
    | Havoc
    | Uninterpreted
    | Zero
)
