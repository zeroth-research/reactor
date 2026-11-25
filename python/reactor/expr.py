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

The operations are those of the namespace of the operands' sort, ``reactor.matrix.real``,
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

from . import reactor
from .reactor import Term, Wire, Var
from .reactor import matrix, Zero
from .sort import ZeroSort
from .matrix.sort import Real, Int, Bool, Word


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
            e = x + 1          # terms == [<Const 1>, <Add>]
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
        case ZeroSort():
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
    def dtype(self) -> Real | Int | Bool | Word | ZeroSort:
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
    """Integer arithmetic (Int): the operations of `matrix.int`; `/` is Euclidean division."""

    def __add__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.int.Add([self.dtype, o.dtype]), o)

    def __sub__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.int.Sub([self.dtype, o.dtype]), o)

    def _mul(self, o) -> "Expr":
        """The pointwise product with the variable `o`."""
        assert isinstance(o, Expr) and o._value is None, "the other factor must be a variable"
        assert self._value is None, "a constant factor goes second, as a Scale"
        assert o.shape == self.shape, f"cannot multiply {self.shape} and {o.shape} pointwise"
        o = self._coerce_same(o)
        return self._binary(matrix.int.Mul([self.dtype, o.dtype]), o)

    def _scale(self, c) -> "Expr":
        """The linear map `x ↦ C ⊙ x` by the constant `c` of my shape, sharing its tensor
        unless its element kind is not of my sort."""
        assert isinstance(c, torch.Tensor), f"the constant must be a tensor, got {type(c)}"
        assert list(c.shape) == self.shape, f"cannot scale {self.shape} by a constant of {list(c.shape)}"
        integer = not c.is_floating_point() and c.dtype != torch.bool
        return self._unary(matrix.int.Scale(c if integer else c.long(), self.dtype))

    def __neg__(self):
        return self._unary(matrix.int.Neg(self.dtype))

    def __abs__(self):
        return self._unary(matrix.int.Abs(self.dtype))

    def __truediv__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.int.Div([self.dtype, o.dtype]), o)

    def __matmul__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.int.MatMul([self.dtype, o.dtype]), o)

    # comparisons -> Bool -> BExpr
    def __eq__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.int.Eq([self.dtype, o.dtype]), o)

    def __ne__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.int.Ne([self.dtype, o.dtype]), o)

    def __lt__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.int.Lt([self.dtype, o.dtype]), o)

    def __le__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.int.Le([self.dtype, o.dtype]), o)

    def __gt__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.int.Gt([self.dtype, o.dtype]), o)

    def __ge__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.int.Ge([self.dtype, o.dtype]), o)


class RExpr(AExpr):
    """Real arithmetic (Real): the operations of `matrix.real`, which keep, add up or take
    off the degrees of their operands."""

    def __add__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.real.Add([self.dtype, o.dtype]), o)

    def __sub__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.real.Sub([self.dtype, o.dtype]), o)

    def _mul(self, o) -> "Expr":
        """The pointwise product with the variable `o`."""
        assert isinstance(o, Expr) and o._value is None, "the other factor must be a variable"
        assert self._value is None, "a constant factor goes second, as a Scale"
        assert o.shape == self.shape, f"cannot multiply {self.shape} and {o.shape} pointwise"
        o = self._coerce_same(o)
        # the degrees of the factors add up
        return self._binary(matrix.real.Mul([self.dtype, o.dtype]), o)

    def _scale(self, c) -> "Expr":
        """The linear map `x ↦ C ⊙ x` by the constant `c` of my shape, sharing its tensor
        unless its element kind is not of my sort."""
        assert isinstance(c, torch.Tensor), f"the constant must be a tensor, got {type(c)}"
        assert list(c.shape) == self.shape, f"cannot scale {self.shape} by a constant of {list(c.shape)}"
        return self._unary(matrix.real.Scale(c if c.is_floating_point() else c.float(), self.dtype))

    def __neg__(self):
        return self._unary(matrix.real.Neg(self.dtype))

    def __abs__(self):
        return self._unary(matrix.real.Abs(self.dtype))

    def __truediv__(self, o):
        o = self._coerce_same(o)
        # the degree of the divisor is taken off that of the dividend
        return self._binary(matrix.real.Div([self.dtype, o.dtype]), o)

    def __matmul__(self, o):
        o = self._coerce_same(o)
        # the degrees of the factors add up
        return self._binary(matrix.real.MatMul([self.dtype, o.dtype]), o)

    # comparisons -> Bool -> BExpr
    def __eq__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.real.Eq([self.dtype, o.dtype]), o)

    def __ne__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.real.Ne([self.dtype, o.dtype]), o)

    def __lt__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.real.Lt([self.dtype, o.dtype]), o)

    def __le__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.real.Le([self.dtype, o.dtype]), o)

    def __gt__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.real.Gt([self.dtype, o.dtype]), o)

    def __ge__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.real.Ge([self.dtype, o.dtype]), o)


# ---------------------------------------------------------------------------
# BExpr — boolean (Bool): the operations of `bool`
# ---------------------------------------------------------------------------


class BExpr(Expr):
    def __eq__(self, o):
        o = self._coerce_same(o)
        return self._apply(matrix.bool.Eq([self.dtype, o.dtype]), self, o)

    def __ne__(self, o):
        o = self._coerce_same(o)
        return self._apply(matrix.bool.Ne([self.dtype, o.dtype]), self, o)

    def __and__(self, o):
        o = self._coerce_same(o)
        return self._apply(matrix.bool.And([self.dtype, o.dtype]), self, o)

    def __or__(self, o):
        o = self._coerce_same(o)
        return self._apply(matrix.bool.Or([self.dtype, o.dtype]), self, o)

    def __xor__(self, o):
        o = self._coerce_same(o)
        return self._apply(matrix.bool.Xor([self.dtype, o.dtype]), self, o)

    def __invert__(self):
        return self._apply(matrix.bool.Not(self.dtype), self)


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
        return self._binary(matrix.word.Eq([self.dtype, o.dtype]), o)

    def __ne__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.word.Ne([self.dtype, o.dtype]), o)

    # arithmetic, modulo 2^width
    def __add__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.word.Add([self.dtype, o.dtype]), o)

    def __sub__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.word.Sub([self.dtype, o.dtype]), o)

    def __mul__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.word.Mul([self.dtype, o.dtype]), o)

    def __radd__(self, o):  return self._coerce(o).__add__(self)

    def __rsub__(self, o):  return self._coerce(o).__sub__(self)

    def __rmul__(self, o):  return self._coerce(o).__mul__(self)

    def __neg__(self):
        return self._unary(matrix.word.Neg(self.dtype))

    # bitwise, width-preserving
    def __and__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.word.And([self.dtype, o.dtype]), o)

    def __or__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.word.Or([self.dtype, o.dtype]), o)

    def __xor__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.word.Xor([self.dtype, o.dtype]), o)

    def __invert__(self):
        return self._unary(matrix.word.Not(self.dtype))


class SExpr(WExpr):
    """Signed words (Word): two's complement division, remainder and comparisons, of
    `matrix.word.signed`."""

    def __floordiv__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.word.signed.Div([self.dtype, o.dtype]), o)

    def __mod__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.word.signed.Mod([self.dtype, o.dtype]), o)

    # comparisons write booleans
    def __lt__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.word.signed.Lt([self.dtype, o.dtype]), o)

    def __le__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.word.signed.Le([self.dtype, o.dtype]), o)

    def __gt__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.word.signed.Gt([self.dtype, o.dtype]), o)

    def __ge__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.word.signed.Ge([self.dtype, o.dtype]), o)


class UExpr(WExpr):
    """Unsigned words (Word): the division, remainder and comparisons of
    `matrix.word.unsigned`."""

    def __floordiv__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.word.unsigned.Div([self.dtype, o.dtype]), o)

    def __mod__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.word.unsigned.Rem([self.dtype, o.dtype]), o)

    # comparisons write booleans
    def __lt__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.word.unsigned.Lt([self.dtype, o.dtype]), o)

    def __le__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.word.unsigned.Le([self.dtype, o.dtype]), o)

    def __gt__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.word.unsigned.Gt([self.dtype, o.dtype]), o)

    def __ge__(self, o):
        o = self._coerce_same(o)
        return self._binary(matrix.word.unsigned.Ge([self.dtype, o.dtype]), o)


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
        op = matrix.bool.Const(value, cod)
    elif family is Real and (value.dtype in (torch.half, torch.float, torch.double) or integer):
        op = matrix.real.Const(value, cod)
    elif family is Int and integer:
        op = matrix.int.Const(value, cod)
    elif family is Word and integer:
        op = matrix.word.Const(value, cod)
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
    if sort is ZeroSort or isinstance(sort, ZeroSort):
        if not isinstance(value, Zero):
            raise TypeError(
                f"expr(): the rate of change of discrete values, of the sort Zero, is "
                f"written `Zero()`, got {value!r}"
            )
        term = Term.constant(Zero(), [Wire(ZeroSort())])
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
        return _wrap(reactor.X(v._wire), signed=isinstance(v, SExpr))
    else:
        raise ValueError(f"expr.X expects an expression, got {v}")


def d(v: Expr) -> Expr:
    """The `derived` wire of a variable expr."""
    return _wrap(reactor.d(v._wire), signed=isinstance(v, SExpr))


def ite(cond: Expr, iftrue, iffalse) -> Expr:
    a, b = iftrue, iffalse
    if not isinstance(a, Expr) and not isinstance(b, Expr):
        raise TypeError("ite(): at least one branch must be an Expr")
    if not isinstance(a, Expr):
        a = expr(a, sort=b.dtype, signed=isinstance(b, SExpr))
    if not isinstance(b, Expr):
        b = expr(b, sort=a.dtype, signed=isinstance(a, SExpr))
    if _family(a.dtype) != _family(b.dtype):
        raise TypeError(f"ite branches have different sorts {a.dtype}, {b.dtype}; use cast()")
    match a:
        case RExpr():
            op = matrix.real.Ite([cond.dtype, a.dtype, b.dtype])
        case IExpr():
            op = matrix.int.Ite([cond.dtype, a.dtype, b.dtype])
        case BExpr():
            op = matrix.bool.Ite([cond.dtype, a.dtype, b.dtype])
        case WExpr():
            op = matrix.word.Ite([cond.dtype, a.dtype, b.dtype])
    return a._apply(op, cond, a, b)


def relu(e: Expr) -> Expr:
    match e:
        case RExpr():
            return e._unary(matrix.real.ReLU(e.dtype))
        case IExpr():
            return e._unary(matrix.int.ReLU(e.dtype))
        case _:
            raise TypeError(f"relu() needs a Real or Int operand, got {e.dtype}")


def argmax(e: Expr) -> Expr:
    """The index of a maximal entry, as a [1, 1] matrix of the operand's sort."""
    match e:
        case RExpr():
            # the index is a plain real
            return e._unary(matrix.real.Argmax(e.dtype))
        case IExpr():
            return e._unary(matrix.int.Argmax(e.dtype))
        case _:
            raise TypeError(f"argmax() needs a Real or Int operand, got {e.dtype}")
