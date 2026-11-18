"""Type stubs for `reactor.op.real`: the operations of real arithmetic, on real matrices and their rates of change.

Generated from the operations of `theory::op`; keep them in step with it.
"""

from typing import Any

# A matrix that an operation holds: a `torch.Tensor` or a `jax.Array` (with JAX), whose
# storage it shares, or a list of rows, which it copies.
Matrix = Any


class Const:
    """A constant real matrix, held in a `Matrix` of floating-point elements."""

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
    """Pointwise division `x / y` of `[m, n]` matrices: the Hadamard division `x ⊘ y`."""

    __match_args__ = ("m", "n")
    def __init__(self, m: int, n: int) -> None: ...
    @property
    def m(self) -> int: ...
    @property
    def n(self) -> int: ...

class Linear:
    """The affine map `x ↦ A·x + B` of real matrices, with `A` and `B` held in `Matrix`es of floating-point elements, of shapes `[m, n]` and `[m, k]`."""

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
    """The index of the first largest entry of `[m, n]` matrices, counted from `0` in row-major order, as a `[1, 1]` real matrix."""

    __match_args__ = ("m", "n")
    def __init__(self, m: int, n: int) -> None: ...
    @property
    def m(self) -> int: ...
    @property
    def n(self) -> int: ...

class Havoc:
    """An arbitrary real matrix of shape `[m, n]`, chosen nondeterministically."""

    __match_args__ = ("m", "n")
    def __init__(self, m: int, n: int) -> None: ...
    @property
    def m(self) -> int: ...
    @property
    def n(self) -> int: ...

class Uninterpreted:
    """A real matrix of shape `[m, n]` of unknown meaning, named by the given symbol, as `Uninterpreted(symbol, m, n)`: e.g., the result of a call that the frontend cannot translate."""

    __match_args__ = ("symbol", "m", "n")
    def __init__(self, symbol: str, m: int, n: int) -> None: ...
    @property
    def symbol(self) -> str: ...
    @property
    def m(self) -> int: ...
    @property
    def n(self) -> int: ...

class Zero:
    """The zero `[m, n]` matrix of the rates of change of degree `d`, as `Zero(m, n, d)`: with `d = 1`, the derivative of a constant."""

    __match_args__ = ("m", "n", "d")
    def __init__(self, m: int, n: int, d: int) -> None: ...
    @property
    def m(self) -> int: ...
    @property
    def n(self) -> int: ...
    @property
    def d(self) -> int: ...

class TAdd:
    """Pointwise addition `x + y` of `[m, n]` matrices of the rates of change of degree `d`, as `TAdd(m, n, d)`."""

    __match_args__ = ("m", "n", "d")
    def __init__(self, m: int, n: int, d: int) -> None: ...
    @property
    def m(self) -> int: ...
    @property
    def n(self) -> int: ...
    @property
    def d(self) -> int: ...

class TSub:
    """Pointwise subtraction `x - y` of `[m, n]` matrices of the rates of change of degree `d`, as `TSub(m, n, d)`."""

    __match_args__ = ("m", "n", "d")
    def __init__(self, m: int, n: int, d: int) -> None: ...
    @property
    def m(self) -> int: ...
    @property
    def n(self) -> int: ...
    @property
    def d(self) -> int: ...

class TLinear:
    """The affine map `x ↦ A·x + B` on the rates of change of degree `d`, as `TLinear(A, B, d)`, with `A` and `B` as for `Linear`; `TLinear(A, B, 0)` is `Linear(A, B)`."""

    __match_args__ = ("a", "b", "d")
    def __init__(self, a: Matrix, b: Matrix, d: int) -> None: ...
    @property
    def a(self) -> Matrix: ...
    @property
    def b(self) -> Matrix: ...
    @property
    def d(self) -> int: ...

class TTranspose:
    """Transposition `xᵀ` of `[m, n]` matrices of the rates of change of degree `d`, as `TTranspose(m, n, d)`."""

    __match_args__ = ("m", "n", "d")
    def __init__(self, m: int, n: int, d: int) -> None: ...
    @property
    def m(self) -> int: ...
    @property
    def n(self) -> int: ...
    @property
    def d(self) -> int: ...

class TIte:
    """If-then-else `if c then x else y` of `[m, n]` matrices of the rates of change of degree `d`, on a single (plain) boolean `c`, as `TIte(m, n, d)`; `TIte(m, n, 0)` is `Ite`."""

    __match_args__ = ("m", "n", "d")
    def __init__(self, m: int, n: int, d: int) -> None: ...
    @property
    def m(self) -> int: ...
    @property
    def n(self) -> int: ...
    @property
    def d(self) -> int: ...

class TId:
    """The identity `x ↦ x` on `[m, n]` matrices of the rates of change of degree `d`, as `TId(m, n, d)`; `TId(m, n, 0)` is `Id`."""

    __match_args__ = ("m", "n", "d")
    def __init__(self, m: int, n: int, d: int) -> None: ...
    @property
    def m(self) -> int: ...
    @property
    def n(self) -> int: ...
    @property
    def d(self) -> int: ...


# The operations of the theory, as `Term.itype` may hold them.
type Op = (
      Const
    | Add
    | Sub
    | Mul
    | Div
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
    | Zero
    | TAdd
    | TSub
    | TLinear
    | TTranspose
    | TIte
    | TId
)
