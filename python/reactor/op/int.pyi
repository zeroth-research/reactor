"""Type stubs for `reactor.op.int`: the operations of integer arithmetic, on integer matrices.

Generated from the operations of `theory::op`; keep them in step with it.
"""

from typing import Any

# A matrix that an operation holds: a `torch.Tensor` or a `jax.Array` (with JAX), whose
# storage it shares, or a list of rows, which it copies.
Matrix = Any


class Const:
    """A constant integer matrix, held in a `Matrix` of integer elements."""

    __match_args__ = ("tensor",)
    def __init__(self, tensor: Matrix) -> None: ...
    @property
    def tensor(self) -> Matrix: ...

class Add:
    """Pointwise addition `x + y` of `[m, n]` matrices."""

    __match_args__ = ("m", "n")
    def __init__(self, m: int, n: int) -> None: ...
    @property
    def m(self) -> int: ...
    @property
    def n(self) -> int: ...

class Sub:
    """Pointwise subtraction `x - y` of `[m, n]` matrices."""

    __match_args__ = ("m", "n")
    def __init__(self, m: int, n: int) -> None: ...
    @property
    def m(self) -> int: ...
    @property
    def n(self) -> int: ...

class Mul:
    """Pointwise multiplication `x ⊙ y` of `[m, n]` matrices: the Hadamard product, not the matrix product `MatMul`."""

    __match_args__ = ("m", "n")
    def __init__(self, m: int, n: int) -> None: ...
    @property
    def m(self) -> int: ...
    @property
    def n(self) -> int: ...

class Div:
    """Pointwise Euclidean division `x div y` of `[m, n]` matrices: the quotient `q` such that `x = q·y + r` with `0 ≤ r < |y|`."""

    __match_args__ = ("m", "n")
    def __init__(self, m: int, n: int) -> None: ...
    @property
    def m(self) -> int: ...
    @property
    def n(self) -> int: ...

class Mod:
    """Pointwise Euclidean remainder `x mod y` of `[m, n]` matrices: the `r` such that `x = (x div y)·y + r` with `0 ≤ r < |y|`, so never negative, unlike Rust's `%`."""

    __match_args__ = ("m", "n")
    def __init__(self, m: int, n: int) -> None: ...
    @property
    def m(self) -> int: ...
    @property
    def n(self) -> int: ...

class Linear:
    """The affine map `x ↦ A·x + B` of integer matrices, with `A` and `B` held in `Matrix`es of integer elements, of shapes `[m, n]` and `[m, k]`."""

    __match_args__ = ("a", "b")
    def __init__(self, a: Matrix, b: Matrix) -> None: ...
    @property
    def a(self) -> Matrix: ...
    @property
    def b(self) -> Matrix: ...

class ReLU:
    """The rectified-linear map `x ↦ max(x, 0)`, applied pointwise to `[m, n]` matrices."""

    __match_args__ = ("m", "n")
    def __init__(self, m: int, n: int) -> None: ...
    @property
    def m(self) -> int: ...
    @property
    def n(self) -> int: ...

class Min:
    """Pointwise minimum `min(x, y)` of `[m, n]` matrices."""

    __match_args__ = ("m", "n")
    def __init__(self, m: int, n: int) -> None: ...
    @property
    def m(self) -> int: ...
    @property
    def n(self) -> int: ...

class Max:
    """Pointwise maximum `max(x, y)` of `[m, n]` matrices."""

    __match_args__ = ("m", "n")
    def __init__(self, m: int, n: int) -> None: ...
    @property
    def m(self) -> int: ...
    @property
    def n(self) -> int: ...

class Transpose:
    """Transposition `xᵀ` of `[m, n]` matrices."""

    __match_args__ = ("m", "n")
    def __init__(self, m: int, n: int) -> None: ...
    @property
    def m(self) -> int: ...
    @property
    def n(self) -> int: ...

class MatMul:
    """The matrix product `x·y` of an `[m, n]` matrix `x` and an `[n, k]` matrix `y`, with the sizes given as `MatMul(m, n, k)`."""

    __match_args__ = ("m", "n", "k")
    def __init__(self, m: int, n: int, k: int) -> None: ...
    @property
    def m(self) -> int: ...
    @property
    def n(self) -> int: ...
    @property
    def k(self) -> int: ...

class Lt:
    """Pointwise comparison `x < y` of `[m, n]` matrices, writing an `[m, n]` boolean matrix."""

    __match_args__ = ("m", "n")
    def __init__(self, m: int, n: int) -> None: ...
    @property
    def m(self) -> int: ...
    @property
    def n(self) -> int: ...

class Le:
    """Pointwise comparison `x ≤ y` of `[m, n]` matrices, writing an `[m, n]` boolean matrix."""

    __match_args__ = ("m", "n")
    def __init__(self, m: int, n: int) -> None: ...
    @property
    def m(self) -> int: ...
    @property
    def n(self) -> int: ...

class Gt:
    """Pointwise comparison `x > y` of `[m, n]` matrices, writing an `[m, n]` boolean matrix."""

    __match_args__ = ("m", "n")
    def __init__(self, m: int, n: int) -> None: ...
    @property
    def m(self) -> int: ...
    @property
    def n(self) -> int: ...

class Ge:
    """Pointwise comparison `x ≥ y` of `[m, n]` matrices, writing an `[m, n]` boolean matrix."""

    __match_args__ = ("m", "n")
    def __init__(self, m: int, n: int) -> None: ...
    @property
    def m(self) -> int: ...
    @property
    def n(self) -> int: ...

class Eq:
    """Pointwise comparison `x = y` of `[m, n]` matrices, writing an `[m, n]` boolean matrix."""

    __match_args__ = ("m", "n")
    def __init__(self, m: int, n: int) -> None: ...
    @property
    def m(self) -> int: ...
    @property
    def n(self) -> int: ...

class Ne:
    """Pointwise comparison `x ≠ y` of `[m, n]` matrices, writing an `[m, n]` boolean matrix."""

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

class Argmax:
    """The index of the first largest entry of `[m, n]` matrices, counted from `0` in row-major order, as a `[1, 1]` integer matrix."""

    __match_args__ = ("m", "n")
    def __init__(self, m: int, n: int) -> None: ...
    @property
    def m(self) -> int: ...
    @property
    def n(self) -> int: ...

class Havoc:
    """An arbitrary integer matrix of shape `[m, n]`, chosen nondeterministically."""

    __match_args__ = ("m", "n")
    def __init__(self, m: int, n: int) -> None: ...
    @property
    def m(self) -> int: ...
    @property
    def n(self) -> int: ...

class Uninterpreted:
    """An integer matrix of shape `[m, n]` of unknown meaning, named by the given symbol, as `Uninterpreted(symbol, m, n)`: e.g., the result of a call that the frontend cannot translate."""

    __match_args__ = ("symbol", "m", "n")
    def __init__(self, symbol: str, m: int, n: int) -> None: ...
    @property
    def symbol(self) -> str: ...
    @property
    def m(self) -> int: ...
    @property
    def n(self) -> int: ...


# The operations of the theory, as `Term.itype` may hold them.
type Op = (
      Const
    | Add
    | Sub
    | Mul
    | Div
    | Mod
    | Linear
    | ReLU
    | Min
    | Max
    | Transpose
    | MatMul
    | Lt
    | Le
    | Gt
    | Ge
    | Eq
    | Ne
    | Id
    | Ite
    | Argmax
    | Havoc
    | Uninterpreted
)
