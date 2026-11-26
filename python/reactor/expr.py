"""Expression: symbolic expressions for the DSL.

An ``Expr`` is a thin wrapper over a wire that lets you build ``Term``s by writing expressions
like ``x + 1`` or ``ite(x < y, x + 1, y)``: composing one immediately builds the underlying
``Term`` and **records it in the active collector** (see ``collecting``). Exprs
are split by *operation family* (the sort picks the class):

    IExpr  — Integer      Int            +  -  *  /  @   and comparisons (-> BExpr)
    RExpr  — Real         Real           +  -  *  /  @   and comparisons (-> BExpr)
    BExpr  — Boolean       Bool           &  |  ~  ^
    SExpr  — Signed word   Word           arithmetic + bitwise + width, signed  // % and comparisons
    UExpr  — Unsigned word Word           arithmetic + bitwise + width, unsigned // % and comparisons

The operations are those of the namespace of the operands' sort, ``reactor.real``,
``.int``, ``.bool`` or ``.word``, built from the operands' sorts, their domain: ``x + y`` on
``Real([2, 3])`` builds ``real.Add([Real([2, 3]), Real([2, 3])])``, and writes the sort of
its codomain.

An ``Expr`` itself is a pure handle — it stores its wire, the ``next`` wire (for a
variable), and, for a constant, its tensor ``value``. It does **not** hold the term: terms
are gathered as they are built. To collect the terms of a block, build the exprs inside a
``with collecting() as terms:`` context; ``terms`` ends up holding every term in dependency
order (and a shared sub-expression appears once, because a term is emitted exactly when it
is created).

Construction and coercion go through ``expr()``:

    expr(var)                                  # a variable, or a bare wire
    expr(True)                                 # a Bool constant
    expr(3, sort=Real)                         # a numeric constant (sort is REQUIRED)
    expr(5, sort=Word(32, [1, 1]), signed=True)
    expr(var, tag="x")                         # `tag` is a display label only, no meaning

``expr()`` is **not idempotent** — passing an existing ``Expr`` is an error. Operators coerce
a *raw* operand through it (using the sibling's sort) but never convert between sorts: mixing
sorts raises, and an explicit conversion is ``cast()``.

``==`` and ``!=`` build the equality predicates (as in z3 and gurobi); put the ``Expr`` on the
left, since a numpy array on the left broadcasts instead of deferring. Truth-testing
(``if e``, ``and``, ``or``, ``not``) **raises**: Python cannot give it a symbolic meaning, and
silently yielding a bool would make a wrong branch look correct — use ``ite()`` and
``&`` / ``|`` / ``~`` instead. An ``Expr`` generates terms rather than holding a value, so it
is also not hashable (no use as a dict key or in a set).
"""

from typing import override

import torch

import reactor as _rr
from .real import Real
from .int import Int
from .bool import Bool
from .word import Word
from .zero import Real0
from . import real as _rl
from .real import broadcast as _rb
from . import int as _ii
from .int import broadcast as _ib
from . import bool as _bl
from .bool import broadcast as _bb
from . import word as _ww
from .word import broadcast as _wb
from .word import signed as _ws
from .word import unsigned as _wu
from . import zero as _zz
from .reactor import Term, Wire, Var

__all__ = [
    "collecting",
    "Expr",
    "AExpr",
    "IExpr",
    "RExpr",
    "BExpr",
    "WExpr",
    "SExpr",
    "UExpr",
    "expr",
    "cast",
    "X",
    "d",
    "ite",
    "relu",
    "argmax",
]


# ---------------------------------------------------------------------------
# Term collector: every Term is recorded, in creation (dependency) order, into
# the innermost active `collecting()` context. sugar.Module opens one per block.
# ---------------------------------------------------------------------------

_open_collectors: list[list[Term]] = []

def _family(sort) -> str:
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


class collecting:
    """Context manager: records every Term built inside it, in dependency order.

        with collecting() as terms:
            e = x + 1          # terms == [<Num 1>, <Add>]
    """

    def __enter__(self) -> list[Term]:
        _open_collectors.append([])
        return _open_collectors[-1]

    def __exit__(self, *exc):
        _open_collectors.pop()


def _emit(term: Term) -> Term:
    if not _open_collectors:
        raise RuntimeError("build Exprs inside a `with collecting():` block")
    _open_collectors[-1].append(term)
    return term


# --- result-sort -> subclass (the ONE place mapping sort to class) ----------


def _wrap(wire, *, next=None, value=None, signed=False, tag=None) -> "Expr":
    match wire.dtype:
        case Bool():
            return BExpr(wire, next=next, value=value, tag=tag)
        case Int():
            return IExpr(wire, next=next, value=value, tag=tag)
        case Real():
            return RExpr(wire, next=next, value=value, tag=tag)
        case Word() if signed:
            return SExpr(wire, next=next, value=value, tag=tag)
        case Word() if not signed:
            return UExpr(wire, next=next, value=value, tag=tag)
        case Real0():
            # the singleton: a value with no operations, only ever written as is
            return Expr(wire, next=next, value=value, tag=tag)
    raise TypeError(f"no Expr class for sort {wire.dtype}")


# ---------------------------------------------------------------------------
# Base Expr
# ---------------------------------------------------------------------------


class Expr:
    """Shared plumbing; instances are always one of IExpr / RExpr / BExpr / SExpr / UExpr (via ``expr()``)."""

    def __init__(self, wire, *, next=None, value=None, tag=None):
        if next is not None and next.dtype != wire.dtype:
            raise TypeError(
                f"a variable's (latched, next) wires must have the same sort, "
                f"got {wire.dtype} and {next.dtype}"
            )
        self._wire = wire
        self._next = next  # next wire, for a variable leaf (nxt(v))
        self._value = value  # tensor, for a constant leaf
        self._tag = tag  # optional human label; carries NO logical meaning

    # --- accessors ---
    @property
    def wire(self) -> Wire:
        return self._wire

    @property
    def dtype(self) -> Real | Int | Bool | Word | Real0:
        return self._wire.dtype

    @property
    def shape(self) -> list:
        """The ``[rows, cols]`` matrix shape of this expression's wire sort (always 2-D)."""
        return self._wire.dtype.shape

    @property
    def tag(self):
        """Optional label for display/debugging only — it has no logical meaning."""
        return self._tag

    # --- coercion (raw literal -> Expr of my sort; never converts an Expr) ---
    def _coerce(self, o) -> "Expr":
        # a coerced literal inherits this expr's signedness (so `5 + xs` and `xs + 5`
        # agree for a signed word `xs`); it is ignored for other sorts. A scalar literal
        # is broadcast to my shape, as the operations are pointwise on equal shapes.
        if isinstance(o, Expr):
            return o
        if self.shape != [1, 1]:
            if isinstance(o, torch.Tensor) and o.numel() == 1:
                o = o.reshape(())
            if isinstance(o, (bool, int, float)) or (isinstance(o, torch.Tensor) and o.dim() == 0):
                o = torch.as_tensor(o).expand(self.shape).clone()
        # a literal takes my sort, shape included: of another shape, as the other factor
        # of `@` may be, it is not coerced but built with `expr()`
        return expr(o, sort=self.dtype, signed=isinstance(self, SExpr))

    def _coerce_same(self, o) -> "Expr":
        o = self._coerce(o)
        if _family(self.dtype) != _family(o.dtype):
            raise TypeError(f"cannot combine {self.dtype} and {o.dtype} implicitly; use cast()")
        if isinstance(self, SExpr) != isinstance(o, SExpr):
            raise TypeError("cannot combine a signed and an unsigned word; make both the same")
        return o

    # --- truth-testing ---
    def __bool__(self):
        raise TypeError(
            "an Expr has no truth value: use ite(cond, a, b) instead of `if`/`and`/`or`, "
            "and & | ~ to combine predicates"
        )

    # --- build helpers (emit the term, wrap the result by its sort) ---
    def _result(self, term: Term) -> "Expr":
        _emit(term)
        return _wrap(term.write[0], signed=isinstance(self, SExpr))

    def _apply(self, op, *args: "Expr") -> "Expr":
        """The Term of `op` reading `args`, writing a wire of its codomain."""
        return self._result(Term(op, [Wire(op.cod)], [a._wire for a in args]))

    def _unary(self, op) -> "Expr":
        """The Term of `op` reading me."""
        return self._apply(op, self)

    def _binary(self, op, o: "Expr") -> "Expr":
        """The Term of `op` reading me and `o`."""
        return self._apply(op, self, o)

    @override
    def __repr__(self) -> str:
        label = f"{self._tag!r} " if self._tag is not None else ""
        return f"{type(self).__name__}({label}#{self._wire.id}, {self._wire.dtype})"


# ---------------------------------------------------------------------------
# IExpr / RExpr — arithmetic (Int / Real): the operations of `int` / `real`
# ---------------------------------------------------------------------------


class AExpr(Expr):
    """The arithmetic shared by integers and reals: the product, by a variable or by a
    constant, and the operations with the operands reversed."""

    def __mul__(self, o):
        self_is_constant = self._value is not None
        o_is_constant = not isinstance(o, Expr) or o._value is not None

        # variable * variable: the pointwise product
        if not self_is_constant and not o_is_constant:
            return self._mul(o)
        # constant * variable: scale the variable `o` by my value
        elif self_is_constant and not o_is_constant:
            x, c = o, self._value
        # variable * constant: scale me by its value
        elif not self_is_constant and o_is_constant:
            x, c = self, (o._value if isinstance(o, Expr) else o)
        # constant * constant: scale my term by its value
        elif self_is_constant and o_is_constant:
            x, c = self, (o._value if isinstance(o, Expr) else o)

        # by a scalar, broadcast to its shape
        if isinstance(c, bool | int | float) or (isinstance(c, torch.Tensor) and c.dim() == 0):
            return x._scale(torch.as_tensor(c).expand(x.shape).clone())
        # by a tensor of its shape, sharing it
        else:
            return x._scale(c)

    def __radd__(self, o):  return self._coerce(o).__add__(self)

    def __rsub__(self, o):  return self._coerce(o).__sub__(self)

    def __rmul__(self, o):  return self.__mul__(o)  # pointwise, so it commutes

    def __rtruediv__(self, o):  return self._coerce(o).__truediv__(self)


class IExpr(AExpr):
    """Integer arithmetic (Int): the operations of `_rr.int`; `/` is Euclidean division."""

    def __add__(self, o):
        o = self._coerce_same(o)
        return self._binary(_ii.Add([self.dtype, o.dtype]), o)

    def __sub__(self, o):
        o = self._coerce_same(o)
        return self._binary(_ii.Sub([self.dtype, o.dtype]), o)

    def _mul(self, o) -> "Expr":
        """The pointwise product with the variable `o`."""
        assert isinstance(o, Expr) and o._value is None, "the other factor must be a variable"
        assert self._value is None, "a constant factor goes second, as a Scale"
        assert o.shape == self.shape, f"cannot multiply {self.shape} and {o.shape} pointwise"
        o = self._coerce_same(o)
        return self._binary(_ii.Mul([self.dtype, o.dtype]), o)

    def _scale(self, c) -> "Expr":
        """The linear map `x ↦ C ⊙ x` by the constant `c` of my shape, sharing its tensor
        unless its element kind is not of my sort."""
        assert isinstance(c, torch.Tensor), f"the constant must be a tensor, got {type(c)}"
        assert list(c.shape) == self.shape, f"cannot scale {self.shape} by a constant of {list(c.shape)}"
        integer = not c.is_floating_point() and c.dtype != torch.bool
        return self._unary(_ii.Scale(c if integer else c.long(), self.dtype))

    def __neg__(self):
        return self._unary(_ii.Neg(self.dtype))

    def __abs__(self):
        return self._unary(_ii.Abs(self.dtype))

    def __truediv__(self, o):
        o = self._coerce_same(o)
        return self._binary(_ii.Div([self.dtype, o.dtype]), o)

    def __matmul__(self, o):
        o = self._coerce_same(o)
        return self._binary(_ii.MatMul([self.dtype, o.dtype]), o)

    # comparisons -> Bool -> BExpr
    def __eq__(self, o):
        o = self._coerce_same(o)
        return self._binary(_ii.Eq([self.dtype, o.dtype]), o)

    def __ne__(self, o):
        o = self._coerce_same(o)
        return self._binary(_ii.Ne([self.dtype, o.dtype]), o)

    def __lt__(self, o):
        o = self._coerce_same(o)
        return self._binary(_ii.Lt([self.dtype, o.dtype]), o)

    def __le__(self, o):
        o = self._coerce_same(o)
        return self._binary(_ii.Le([self.dtype, o.dtype]), o)

    def __gt__(self, o):
        o = self._coerce_same(o)
        return self._binary(_ii.Gt([self.dtype, o.dtype]), o)

    def __ge__(self, o):
        o = self._coerce_same(o)
        return self._binary(_ii.Ge([self.dtype, o.dtype]), o)


class RExpr(AExpr):
    """Real arithmetic (Real): the operations of `real`, which keep, add up or take
    off the degrees of their operands."""

    def __add__(self, o):
        o = self._coerce_same(o)
        return self._binary(_rl.Add([self.dtype, o.dtype]), o)

    def __sub__(self, o):
        o = self._coerce_same(o)
        return self._binary(_rl.Sub([self.dtype, o.dtype]), o)

    def _mul(self, o) -> "Expr":
        """The pointwise product with the variable `o`."""
        assert isinstance(o, Expr) and o._value is None, "the other factor must be a variable"
        assert self._value is None, "a constant factor goes second, as a Scale"
        assert o.shape == self.shape, f"cannot multiply {self.shape} and {o.shape} pointwise"
        o = self._coerce_same(o)
        # the degrees of the factors add up
        return self._binary(_rl.Mul([self.dtype, o.dtype]), o)

    def _scale(self, c) -> "Expr":
        """The linear map `x ↦ C ⊙ x` by the constant `c` of my shape, sharing its tensor
        unless its element kind is not of my sort."""
        assert isinstance(c, torch.Tensor), f"the constant must be a tensor, got {type(c)}"
        assert list(c.shape) == self.shape, f"cannot scale {self.shape} by a constant of {list(c.shape)}"
        return self._unary(_rl.Scale(c if c.is_floating_point() else c.float(), self.dtype))

    def __neg__(self):
        return self._unary(_rl.Neg(self.dtype))

    def __abs__(self):
        return self._unary(_rl.Abs(self.dtype))

    def __truediv__(self, o):
        o = self._coerce_same(o)
        # the degree of the divisor is taken off that of the dividend
        return self._binary(_rl.Div([self.dtype, o.dtype]), o)

    def __matmul__(self, o):
        o = self._coerce_same(o)
        # the degrees of the factors add up
        return self._binary(_rl.MatMul([self.dtype, o.dtype]), o)

    # comparisons -> Bool -> BExpr
    def __eq__(self, o):
        o = self._coerce_same(o)
        return self._binary(_rl.Eq([self.dtype, o.dtype]), o)

    def __ne__(self, o):
        o = self._coerce_same(o)
        return self._binary(_rl.Ne([self.dtype, o.dtype]), o)

    def __lt__(self, o):
        o = self._coerce_same(o)
        return self._binary(_rl.Lt([self.dtype, o.dtype]), o)

    def __le__(self, o):
        o = self._coerce_same(o)
        return self._binary(_rl.Le([self.dtype, o.dtype]), o)

    def __gt__(self, o):
        o = self._coerce_same(o)
        return self._binary(_rl.Gt([self.dtype, o.dtype]), o)

    def __ge__(self, o):
        o = self._coerce_same(o)
        return self._binary(_rl.Ge([self.dtype, o.dtype]), o)


# ---------------------------------------------------------------------------
# BExpr — boolean (Bool): the operations of `bool`
# ---------------------------------------------------------------------------


class BExpr(Expr):
    def __eq__(self, o):
        o = self._coerce_same(o)
        return self._apply(_bl.Eq([self.dtype, o.dtype]), self, o)

    def __ne__(self, o):
        o = self._coerce_same(o)
        return self._apply(_bl.Ne([self.dtype, o.dtype]), self, o)

    def __and__(self, o):
        o = self._coerce_same(o)
        return self._apply(_bl.And([self.dtype, o.dtype]), self, o)

    def __or__(self, o):
        o = self._coerce_same(o)
        return self._apply(_bl.Or([self.dtype, o.dtype]), self, o)

    def __xor__(self, o):
        o = self._coerce_same(o)
        return self._apply(_bl.Xor([self.dtype, o.dtype]), self, o)

    def __invert__(self):
        return self._apply(_bl.Not(self.dtype), self)


# ---------------------------------------------------------------------------
# SExpr / UExpr — word (Word): the operations of `word`, and of `word.signed` or
# `word.unsigned`
# ---------------------------------------------------------------------------


class WExpr(Expr):
    """The operations shared by signed and unsigned words: arithmetic modulo 2^width,
    equality and the bitwise operations, which do not depend on the sign."""

    @property
    def width(self) -> int:
        """The number of bits of the words of my sort."""
        return self.dtype.width

    def __eq__(self, o):
        o = self._coerce_same(o)
        return self._binary(_ww.Eq([self.dtype, o.dtype]), o)

    def __ne__(self, o):
        o = self._coerce_same(o)
        return self._binary(_ww.Ne([self.dtype, o.dtype]), o)

    # arithmetic, modulo 2^width
    def __add__(self, o):
        o = self._coerce_same(o)
        return self._binary(_ww.Add([self.dtype, o.dtype]), o)

    def __sub__(self, o):
        o = self._coerce_same(o)
        return self._binary(_ww.Sub([self.dtype, o.dtype]), o)

    def __mul__(self, o):
        o = self._coerce_same(o)
        return self._binary(_ww.Mul([self.dtype, o.dtype]), o)

    def __radd__(self, o):  return self._coerce(o).__add__(self)

    def __rsub__(self, o):  return self._coerce(o).__sub__(self)

    def __rmul__(self, o):  return self._coerce(o).__mul__(self)

    def __neg__(self):
        return self._unary(_ww.Neg(self.dtype))

    # bitwise, width-preserving
    def __and__(self, o):
        o = self._coerce_same(o)
        return self._binary(_ww.And([self.dtype, o.dtype]), o)

    def __or__(self, o):
        o = self._coerce_same(o)
        return self._binary(_ww.Or([self.dtype, o.dtype]), o)

    def __xor__(self, o):
        o = self._coerce_same(o)
        return self._binary(_ww.Xor([self.dtype, o.dtype]), o)

    def __invert__(self):
        return self._unary(_ww.Not(self.dtype))


class SExpr(WExpr):
    """Signed words (Word): two's complement division, remainder and comparisons, of
    `word.signed`."""

    def __floordiv__(self, o):
        o = self._coerce_same(o)
        return self._binary(_ws.Div([self.dtype, o.dtype]), o)

    def __mod__(self, o):
        o = self._coerce_same(o)
        return self._binary(_ws.Mod([self.dtype, o.dtype]), o)

    # comparisons write booleans
    def __lt__(self, o):
        o = self._coerce_same(o)
        return self._binary(_ws.Lt([self.dtype, o.dtype]), o)

    def __le__(self, o):
        o = self._coerce_same(o)
        return self._binary(_ws.Le([self.dtype, o.dtype]), o)

    def __gt__(self, o):
        o = self._coerce_same(o)
        return self._binary(_ws.Gt([self.dtype, o.dtype]), o)

    def __ge__(self, o):
        o = self._coerce_same(o)
        return self._binary(_ws.Ge([self.dtype, o.dtype]), o)


class UExpr(WExpr):
    """Unsigned words (Word): the division, remainder and comparisons of
    `word.unsigned`."""

    def __floordiv__(self, o):
        o = self._coerce_same(o)
        return self._binary(_wu.Div([self.dtype, o.dtype]), o)

    def __mod__(self, o):
        o = self._coerce_same(o)
        return self._binary(_wu.Rem([self.dtype, o.dtype]), o)

    # comparisons write booleans
    def __lt__(self, o):
        o = self._coerce_same(o)
        return self._binary(_wu.Lt([self.dtype, o.dtype]), o)

    def __le__(self, o):
        o = self._coerce_same(o)
        return self._binary(_wu.Le([self.dtype, o.dtype]), o)

    def __gt__(self, o):
        o = self._coerce_same(o)
        return self._binary(_wu.Gt([self.dtype, o.dtype]), o)

    def __ge__(self, o):
        o = self._coerce_same(o)
        return self._binary(_wu.Ge([self.dtype, o.dtype]), o)


# ---------------------------------------------------------------------------
# expr() — single construction / coercion entry point
# ---------------------------------------------------------------------------


def expr(value, *, sort=None, signed=False, tag=None) -> Expr:
    if isinstance(value, Expr):
        raise TypeError("expr() builds from a raw value; got an Expr already")

    if isinstance(value, (Wire, Var)):  # a variable, or a bare wire
        return _wrap(value, signed=signed, tag=tag)
    elif isinstance(value, bool) and sort is None:
        return _const(value, Bool, signed=signed, tag=tag)
    else:
        return _const(value, sort, signed=signed, tag=tag)


def _term_from_torch(value, sort):
    """The constant of the tensor `value`, of the sort `sort`: a concrete sort, its codomain,
    which the constant checks against the tensor, or a family (Real/Int/Bool), whose sort the
    constant takes from the tensor."""
    assert isinstance(value, torch.Tensor)

    # make it 2D to make the theory crate happy
    match len(value.shape):
        case 0:
            shape = [1, 1]
            value = value.reshape(shape)
        case 1:
            shape = [1, value.shape[-1]]
            value = value.reshape(shape)
        case 2:
            shape = list(value.shape)
        case _:
            raise TypeError(f"expr() requires shape at most 2d; got {value.shape}")

    # a family leaves the codomain to the constant; a concrete sort is its codomain
    if sort in (Bool, Real, Int):
        family, cod = sort, None
    elif sort is Word:
        raise TypeError("expr(): a word literal needs a width: pass sort=Word(width, [...])")
    elif isinstance(sort, Real | Int | Bool | Word):
        family, cod = type(sort), sort
    else:
        raise TypeError(f"expr(): unsupported sort {sort}")

    # the tensor is kept as given: no conversion of its elements
    integer = value.dtype in (torch.int, torch.long)
    if family is Bool and value.dtype == torch.bool:
        op = _bl.Truth(value, cod)
    elif family is Real and (value.dtype in (torch.half, torch.float, torch.double) or integer):
        op = _rl.Num(value, cod)
    elif family is Int and integer:
        op = _ii.Num(value, cod)
    elif family is Word and integer:
        op = _ww.Num(value, cod)
    else:
        raise TypeError(f"expr() unsupported literal type {value.dtype} / {sort}")
    return Term.constant(op, [Wire(op.cod)]), value


def _term_from_other(value, sort):
    """The constant of the Python value `value` (a number, a boolean or a nested list), of
    the sort `sort`, as `_term_from_torch`."""
    family = sort if isinstance(sort, type) else type(sort)

    # a boolean is not a number: no conversion of `True` to 1
    if family is not Bool and torch.tensor(value).dtype == torch.bool:
        raise TypeError(f"expr(): a boolean value {value!r} cannot be given the sort {family.__name__}")

    if family is Bool:
        dtype = torch.bool
    elif family is Real:
        dtype = torch.float
    else:
        dtype = torch.int64

    return _term_from_torch(torch.tensor(value, dtype=dtype), sort)


def _const(value, sort, *, signed, tag=None) -> Expr:
    # a sort is either None, or a type/family, or a concrete sort instance
    if sort is None:
        raise TypeError("expr(): a numeric literal needs an explicit sort= ")

    # the singleton sort, the rate of change of discrete values: its only value is
    # written by the operation `Zero()`, given as is, unambiguously, and no literal
    if sort is Real0 or isinstance(sort, Real0):
        if not isinstance(value, _zz.Zero):
            raise TypeError(
                f"expr(): the rate of change of discrete values, of the sort Real0, is "
                f"written `Zero()`, got {value!r}"
            )
        term = Term.constant(_zz.Zero(), [Wire(Real0())])
        _emit(term)
        return _wrap(term.write[0], value=value, tag=tag)

    # build term
    if isinstance(value, torch.Tensor):
        term, value = _term_from_torch(value, sort)
    elif isinstance(value, int | float | bool | list):
        term, value = _term_from_other(value, sort)
    else:
        raise TypeError(f"expr() unsupported value type {type(value)}")

    # emit into collector
    _emit(term)

    return _wrap(term.write[0], value=value, signed=signed, tag=tag)


# ---------------------------------------------------------------------------
# Cast ops
# ---------------------------------------------------------------------------

def _resolve_sort(sort, shape) -> Real | Int | Bool | Word:
    """The sort `sort` of `shape`: a family (Real/Int/Bool) made of it, or a sort of it."""
    if sort is Word:
        raise TypeError("a word needs a width: pass sort=Word(width, [...])")
    if sort in (Bool, Int, Real):
        return sort(shape)
    if sort.shape != shape:
        raise TypeError(f"the sort {sort} is not of the shape {shape}")
    return sort


def cast(e: Expr, sort) -> Expr:
    """Explicit sort conversion. Only trivial (same-sort) casts are supported for now;
    a genuine conversion op is not yet in the IR."""
    target = _resolve_sort(sort, e.shape)
    if e.dtype == target:
        return e
    raise NotImplementedError(f"cast {e.dtype} -> {target} is not supported yet")


# ---------------------------------------------------------------------------
# Non-operator ops
# ---------------------------------------------------------------------------


def X(v: Expr) -> Expr:
    """The `next` wire of a variable expr."""
    if isinstance(v, Expr):
        return _wrap(_rr.X(v._wire), signed=isinstance(v, SExpr))
    else:
        raise ValueError(f"expr.X expects an expression, got {v}")


def d(v: Expr) -> Expr:
    """The `derived` wire of a variable expr."""
    return _wrap(_rr.d(v._wire), signed=isinstance(v, SExpr))


def ite(cond: "BExpr", iftrue, iffalse) -> Expr:
    if not isinstance(cond, BExpr):
        raise TypeError(f"ite(): the condition must be a boolean expression, got {cond!r}")
    a, b = iftrue, iffalse
    if not isinstance(a, Expr) and not isinstance(b, Expr):
        raise TypeError("ite(): at least one branch must be an Expr")
    if not isinstance(a, Expr):
        a = expr(a, sort=b.dtype, signed=isinstance(b, SExpr))
    if not isinstance(b, Expr):
        b = expr(b, sort=a.dtype, signed=isinstance(a, SExpr))
    if _family(a.dtype) != _family(b.dtype):
        raise TypeError(f"ite branches have different sorts {a.dtype}, {b.dtype}; use cast()")
    # a condition of the shape of the branches chooses pointwise; a scalar one, the whole
    # matrix, by the choice of the theory that broadcasts it
    match a:
        case RExpr() if cond.shape == a.shape:
            choice = _rl.Ite
        case RExpr() if cond.shape == [1, 1]:
            choice = _rb.Ite
        case IExpr() if cond.shape == a.shape:
            choice = _ii.Ite
        case IExpr() if cond.shape == [1, 1]:
            choice = _ib.Ite
        case BExpr() if cond.shape == a.shape:
            choice = _bl.Ite
        case BExpr() if cond.shape == [1, 1]:
            choice = _bb.Ite
        case WExpr() if cond.shape == a.shape:
            choice = _ww.Ite
        case WExpr() if cond.shape == [1, 1]:
            choice = _wb.Ite
        case _:
            raise TypeError(
                f"ite(): the condition must be of the shape {a.shape} of the branches, "
                f"choosing pointwise, or a scalar, [1, 1], choosing the whole matrix; "
                f"got {cond.shape}"
            )
    return a._apply(choice([cond.dtype, a.dtype, b.dtype]), cond, a, b)


def relu(e: Expr) -> Expr:
    match e:
        case RExpr():
            return e._unary(_rl.ReLU(e.dtype))
        case IExpr():
            return e._unary(_ii.ReLU(e.dtype))
        case _:
            raise TypeError(f"relu() needs a Real or Int operand, got {e.dtype}")


def argmax(e: Expr) -> Expr:
    """The index of a maximal entry, as a [1, 1] matrix of the operand's sort."""
    match e:
        case RExpr():
            # the index is a plain real
            return e._unary(_rl.Argmax(e.dtype))
        case IExpr():
            return e._unary(_ii.Argmax(e.dtype))
        case _:
            raise TypeError(f"argmax() needs a Real or Int operand, got {e.dtype}")
