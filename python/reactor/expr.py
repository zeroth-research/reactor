"""Expression: symbolic expressions for the DSL.

An ``Expr`` is a thin wrapper over a wire that lets you build ``Term``s by writing expressions
like ``x + 1`` or ``ite(x < y, x + 1, y)``: composing one immediately builds the underlying
``Term`` and **records it in the active collector** (see ``collecting``). Exprs
are split by *operation family* (the sort picks the class):

    AExpr  — Arithmetic   Int / Real     +  -  *  /  @   and comparisons (-> BExpr)
    BExpr  — Boolean       Bool           &  |  ~  ^
    WExpr  — Word          Word           arithmetic + bitwise + signed/unsigned + width

The operations are those of the namespace of the operands' sort, ``reactor.tensor.real``,
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

from .reactor import Term, Wire, Var, X as _X, d as _d
from .reactor import tensor
from .sort import Real, Int, Bool, Word, Zero
from .sort import shape as _shape, with_shape as _with_shape, family as _family


def _normalize_shape(shape: list) -> list:
    if len(shape) == 0:
        return [1, 1]
    if len(shape) == 1:
        return [1, shape[0]]
    return shape


# ---------------------------------------------------------------------------
# Term collector: every Term is recorded, in creation (dependency) order, into
# the innermost active `collecting()` context. sugar.Module opens one per block.
# ---------------------------------------------------------------------------

_open_collectors: list[list[Term]] = []


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
        case Int() | Real():
            return AExpr(wire, next=next, value=value, tag=tag)
        case Word():
            return WExpr(wire, next=next, value=value, signed=signed, tag=tag)
        case Zero():
            # the singleton: a value with no operations, only ever written as is
            return Expr(wire, next=next, value=value, tag=tag)
    raise TypeError(f"no Expr class for sort {wire.dtype}")


# ---------------------------------------------------------------------------
# Base Expr
# ---------------------------------------------------------------------------


class Expr:
    """Shared plumbing; instances are always one of AExpr / BExpr / WExpr (via ``expr()``)."""

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
    def dtype(self) -> Real | Int | Bool | Word | Zero:
        return self._wire.dtype

    @property
    def shape(self) -> list:
        """The ``[rows, cols]`` matrix shape of this expression's wire sort (always 2-D)."""
        return _shape(self._wire.dtype)

    @property
    def tag(self):
        """Optional label for display/debugging only — it has no logical meaning."""
        return self._tag

    def _bw(self):
        match self.dtype:
            case Word(bw):
                return bw
            case _:
                return None

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
        # a literal takes the degree of a real, as it takes the width of a word
        degree = self.dtype.degree if isinstance(self.dtype, Real) else 0
        return expr(o, sort=type(self.dtype), bw=self._bw(), degree=degree, signed=getattr(self, "_signed", False))

    def _coerce_same(self, o) -> "Expr":
        o = self._coerce(o)
        if _family(self.dtype) != _family(o.dtype):
            raise TypeError(f"cannot combine {self.dtype} and {o.dtype} implicitly; use cast()")
        if getattr(self, "_signed", False) != getattr(o, "_signed", False):
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
        return _wrap(term.write[0], signed=getattr(self, "_signed", False))

    def _apply(self, op, *args: "Expr") -> "Expr":
        """The Term of `op` reading `args`, writing a wire of its codomain."""
        return self._result(Term(op, [Wire(op.cod)], [a._wire for a in args]))

    @override
    def __repr__(self) -> str:
        label = f"{self._tag!r} " if self._tag is not None else ""
        return f"{type(self).__name__}({label}#{self._wire.id}, {self._wire.dtype})"


# ---------------------------------------------------------------------------
# AExpr — arithmetic (Real / Int): the operations of `real` / `int`
# ---------------------------------------------------------------------------


class AExpr(Expr):
    def __add__(self, o):
        o = self._coerce_same(o)
        match self.dtype:
            case Real():
                op = tensor.real.Add([self.dtype, o.dtype])
            case Int():
                op = tensor.int.Add([self.dtype, o.dtype])
        return self._apply(op, self, o)

    def __sub__(self, o):
        o = self._coerce_same(o)
        match self.dtype:
            case Real():
                op = tensor.real.Sub([self.dtype, o.dtype])
            case Int():
                op = tensor.int.Sub([self.dtype, o.dtype])
        return self._apply(op, self, o)

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

    def _mul(self, o) -> "Expr":
        """The pointwise product with the variable `o`."""
        assert isinstance(o, Expr) and o._value is None, "the other factor must be a variable"
        assert self._value is None, "a constant factor goes second, as a Scale"
        assert o.shape == self.shape, f"cannot multiply {self.shape} and {o.shape} pointwise"
        o = self._coerce_same(o)
        match self.dtype:
            case Real():
                # the degrees of the factors add up
                op = tensor.real.Mul([self.dtype, o.dtype])
            case Int():
                op = tensor.int.Mul([self.dtype, o.dtype])
        return self._apply(op, self, o)

    def _scale(self, c) -> "Expr":
        """The linear map `x ↦ C ⊙ x` by the constant `c` of my shape, sharing its tensor
        unless its element kind is not of my sort."""
        assert isinstance(c, torch.Tensor), f"the constant must be a tensor, got {type(c)}"
        assert list(c.shape) == self.shape, f"cannot scale {self.shape} by a constant of {list(c.shape)}"
        match self.dtype:
            case Real():
                op = tensor.real.Scale(c if c.is_floating_point() else c.float(), self.dtype)
            case Int():
                integer = not c.is_floating_point() and c.dtype != torch.bool
                op = tensor.int.Scale(c if integer else c.long(), self.dtype)
        return self._apply(op, self)

    def __neg__(self):
        match self.dtype:
            case Real():
                op = tensor.real.Neg(self.dtype)
            case Int():
                op = tensor.int.Neg(self.dtype)
        return self._apply(op, self)

    def __abs__(self):
        match self.dtype:
            case Real():
                op = tensor.real.Abs(self.dtype)
            case Int():
                op = tensor.int.Abs(self.dtype)
        return self._apply(op, self)

    def __truediv__(self, o):
        o = self._coerce_same(o)
        # of reals, the degree of the divisor is taken off that of the dividend
        match self.dtype:
            case Real():
                op = tensor.real.Div([self.dtype, o.dtype])
            case Int():
                op = tensor.int.Div([self.dtype, o.dtype])
        return self._apply(op, self, o)

    def __radd__(self, o):  return self._coerce(o).__add__(self)

    def __rsub__(self, o):  return self._coerce(o).__sub__(self)

    def __rmul__(self, o):  return self.__mul__(o)  # pointwise, so it commutes

    def __rtruediv__(self, o):  return self._coerce(o).__truediv__(self)

    def __matmul__(self, o):
        o = self._coerce_same(o)
        # of reals, the degrees of the factors add up
        match self.dtype:
            case Real():
                op = tensor.real.MatMul([self.dtype, o.dtype])
            case Int():
                op = tensor.int.MatMul([self.dtype, o.dtype])
        return self._apply(op, self, o)

    # comparisons -> Bool -> BExpr
    def __eq__(self, o):
        o = self._coerce_same(o)
        match self.dtype:
            case Real():
                op = tensor.real.Eq([self.dtype, o.dtype])
            case Int():
                op = tensor.int.Eq([self.dtype, o.dtype])
        return self._apply(op, self, o)

    def __ne__(self, o):
        o = self._coerce_same(o)
        match self.dtype:
            case Real():
                op = tensor.real.Ne([self.dtype, o.dtype])
            case Int():
                op = tensor.int.Ne([self.dtype, o.dtype])
        return self._apply(op, self, o)

    def __lt__(self, o):
        o = self._coerce_same(o)
        match self.dtype:
            case Real():
                op = tensor.real.Lt([self.dtype, o.dtype])
            case Int():
                op = tensor.int.Lt([self.dtype, o.dtype])
        return self._apply(op, self, o)

    def __le__(self, o):
        o = self._coerce_same(o)
        match self.dtype:
            case Real():
                op = tensor.real.Le([self.dtype, o.dtype])
            case Int():
                op = tensor.int.Le([self.dtype, o.dtype])
        return self._apply(op, self, o)

    def __gt__(self, o):
        o = self._coerce_same(o)
        match self.dtype:
            case Real():
                op = tensor.real.Gt([self.dtype, o.dtype])
            case Int():
                op = tensor.int.Gt([self.dtype, o.dtype])
        return self._apply(op, self, o)

    def __ge__(self, o):
        o = self._coerce_same(o)
        match self.dtype:
            case Real():
                op = tensor.real.Ge([self.dtype, o.dtype])
            case Int():
                op = tensor.int.Ge([self.dtype, o.dtype])
        return self._apply(op, self, o)


# ---------------------------------------------------------------------------
# BExpr — boolean (Bool): the operations of `bool`
# ---------------------------------------------------------------------------


class BExpr(Expr):
    def __eq__(self, o):
        o = self._coerce_same(o)
        return self._apply(tensor.bool.Eq([self.dtype, o.dtype]), self, o)

    def __ne__(self, o):
        o = self._coerce_same(o)
        return self._apply(tensor.bool.Ne([self.dtype, o.dtype]), self, o)

    def __and__(self, o):
        o = self._coerce_same(o)
        return self._apply(tensor.bool.And([self.dtype, o.dtype]), self, o)

    def __or__(self, o):
        o = self._coerce_same(o)
        return self._apply(tensor.bool.Or([self.dtype, o.dtype]), self, o)

    def __xor__(self, o):
        o = self._coerce_same(o)
        return self._apply(tensor.bool.Xor([self.dtype, o.dtype]), self, o)

    def __invert__(self):
        return self._apply(tensor.bool.Not(self.dtype), self)


# ---------------------------------------------------------------------------
# WExpr — word (Word): the operations of `word`, `word.signed` and `word.unsigned`
# ---------------------------------------------------------------------------


class WExpr(Expr):
    def __init__(self, wire, *, next=None, value=None, signed=False, tag=None):
        super().__init__(wire, next=next, value=value, tag=tag)
        self._signed = signed

    @property
    def signed(self) -> bool:
        return self._signed

    def __eq__(self, o):
        o = self._coerce_same(o)
        return self._apply(tensor.word.Eq([self.dtype, o.dtype]), self, o)

    def __ne__(self, o):
        o = self._coerce_same(o)
        return self._apply(tensor.word.Ne([self.dtype, o.dtype]), self, o)

    # arithmetic, modulo 2^width
    def __add__(self, o):
        o = self._coerce_same(o)
        return self._apply(tensor.word.Add([self.dtype, o.dtype]), self, o)

    def __sub__(self, o):
        o = self._coerce_same(o)
        return self._apply(tensor.word.Sub([self.dtype, o.dtype]), self, o)

    def __mul__(self, o):
        o = self._coerce_same(o)
        return self._apply(tensor.word.Mul([self.dtype, o.dtype]), self, o)

    def __radd__(self, o):  return self._coerce(o).__add__(self)

    def __rsub__(self, o):  return self._coerce(o).__sub__(self)

    def __rmul__(self, o):  return self._coerce(o).__mul__(self)

    def __neg__(self):
        return self._apply(tensor.word.Neg(self.dtype), self)

    def __floordiv__(self, o):
        o = self._coerce_same(o)
        if self._signed:
            op = tensor.word.signed.Div([self.dtype, o.dtype])
        else:
            op = tensor.word.unsigned.Div([self.dtype, o.dtype])
        return self._apply(op, self, o)

    def __mod__(self, o):
        o = self._coerce_same(o)
        if self._signed:
            op = tensor.word.signed.Mod([self.dtype, o.dtype])
        else:
            op = tensor.word.unsigned.Rem([self.dtype, o.dtype])
        return self._apply(op, self, o)

    # comparisons pick signed/unsigned; they write booleans
    def __lt__(self, o):
        o = self._coerce_same(o)
        if self._signed:
            op = tensor.word.signed.Lt([self.dtype, o.dtype])
        else:
            op = tensor.word.unsigned.Lt([self.dtype, o.dtype])
        return self._apply(op, self, o)

    def __le__(self, o):
        o = self._coerce_same(o)
        if self._signed:
            op = tensor.word.signed.Le([self.dtype, o.dtype])
        else:
            op = tensor.word.unsigned.Le([self.dtype, o.dtype])
        return self._apply(op, self, o)

    def __gt__(self, o):
        o = self._coerce_same(o)
        if self._signed:
            op = tensor.word.signed.Gt([self.dtype, o.dtype])
        else:
            op = tensor.word.unsigned.Gt([self.dtype, o.dtype])
        return self._apply(op, self, o)

    def __ge__(self, o):
        o = self._coerce_same(o)
        if self._signed:
            op = tensor.word.signed.Ge([self.dtype, o.dtype])
        else:
            op = tensor.word.unsigned.Ge([self.dtype, o.dtype])
        return self._apply(op, self, o)

    # bitwise, width-preserving
    def __and__(self, o):
        o = self._coerce_same(o)
        return self._apply(tensor.word.And([self.dtype, o.dtype]), self, o)

    def __or__(self, o):
        o = self._coerce_same(o)
        return self._apply(tensor.word.Or([self.dtype, o.dtype]), self, o)

    def __xor__(self, o):
        o = self._coerce_same(o)
        return self._apply(tensor.word.Xor([self.dtype, o.dtype]), self, o)

    def __invert__(self):
        return self._apply(tensor.word.Not(self.dtype), self)


# ---------------------------------------------------------------------------
# expr() — single construction / coercion entry point
# ---------------------------------------------------------------------------


def expr(value, *, sort=None, signed=False, bw=None, degree=0, tag=None) -> Expr:
    if isinstance(value, Expr):
        raise TypeError("expr() builds from a raw value; got an Expr already")

    if isinstance(value, (Wire, Var)):  # a variable, or a bare wire
        return _wrap(value, signed=signed, tag=tag)
    elif isinstance(value, bool) and sort is None:
        return _const(value, Bool, signed=signed, tag=tag)
    else:
        return _const(value, sort, signed=signed, tag=tag, bw=bw, degree=degree)


def _term_from_torch(value, sort, bw=None, degree=0):
    assert isinstance(value, torch.Tensor)
    assert sort in (Bool, Real, Int, Word)

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

    match value.dtype:
        case torch.bool if sort == Bool:
            out = Bool(shape)
        # the tensor is kept as given: no conversion of its elements
        case torch.half | torch.float | torch.double if sort == Real:
            out = Real(shape, degree=degree)
        case torch.int | torch.long if sort == Int:
            out = Int(shape)
        case torch.int | torch.long if sort == Real:
            out = Real(shape, degree=degree)
        case torch.int | torch.long if sort == Word and bw is not None:
            out = Word(bw, shape)
        case _:
            raise TypeError(f"expr() unsupported literal type {type(value)} / {sort}")
    match out:
        case Bool():
            op = tensor.bool.Const(value, out)
        case Real():
            op = tensor.real.Const(value, out)
        case Int():
            op = tensor.int.Const(value, out)
        case Word():
            op = tensor.word.Const(value, out)
    return Term.constant(op, [Wire(out)]), value


def _term_from_other(value, sort, bw=None, degree=0):
    # torch will figure out the errors, and the base
    assert sort in (Bool, Real, Int, Word)

    # a boolean is not a number: no conversion of `True` to 1
    if sort != Bool and torch.tensor(value).dtype == torch.bool:
        raise TypeError(f"expr(): a boolean value {value!r} cannot be given the sort {sort.__name__}")

    if sort == Bool:
        dtype = torch.bool
    elif sort == Real:
        dtype = torch.float
    else:
        dtype = torch.int64

    return _term_from_torch(torch.tensor(value, dtype=dtype), sort, bw=bw, degree=degree)


def _unpack_sort_user_argument(sort, *, bw=None):
    assert sort is not None

    if isinstance(sort, Real | Int | Bool | Word | Zero):
        match sort:
            case Bool(shape):
                return Bool, None, list(shape)
            case Real(shape):
                return Real, None, list(shape)
            case Int(shape):
                return Int, None, list(shape)
            case Word(w, shape):
                return Word, w, list(shape)
        raise TypeError(f"expr(): unsupported sort {sort}")

    if sort in (Bool, Real, Int):
        return sort, None, None
    elif sort == Word and bw is not None:
        return sort, bw, None

    raise TypeError(f"expr(): unsupported sort {sort}")


def _const(value, sort, *, signed, bw=None, degree=0, tag=None) -> Expr:
    # a sort is either None, or a type/family, or a concrete sort instance
    if sort is None:
        raise TypeError("expr(): a numeric literal needs an explicit sort= ")

    # the singleton sort, the rate of change of discrete values: its only value is
    # written by the operation `tensor.zero.Zero()`, given as is, unambiguously, and no literal
    if sort is Zero or isinstance(sort, Zero):
        if not isinstance(value, tensor.zero.Zero):
            raise TypeError(
                f"expr(): the rate of change of discrete values, of the sort Zero, is "
                f"written `Zero`, got {value!r}"
            )
        term = Term.constant(tensor.zero.Zero(), [Wire(Zero())])
        _emit(term)
        return _wrap(term.write[0], value=value, tag=tag)

    # unpack sort family and required shape if any
    given = sort
    sort, bw, required_shape = _unpack_sort_user_argument(sort, bw=bw)
    if sort is Real and isinstance(given, Real):
        degree = given.degree  # a real sort carries its degree

    # build term
    if isinstance(value, torch.Tensor):
        term, value = _term_from_torch(value, sort, bw=bw, degree=degree)
    elif isinstance(value, int | float | bool | list):
        term, value = _term_from_other(value, sort, bw=bw, degree=degree)
    else:
        raise TypeError(f"expr() unsupported value type {type(value)}")

    # check built shape corresponds with required one, if any
    assert len(term.write) == 1
    shape = _shape(term.write[0].dtype)
    if required_shape is not None and shape != required_shape:
        raise TypeError(f"expr(): value {value} with shape {shape} cannot be given shape {required_shape}")

    # emit into collector
    _emit(term)

    return _wrap(term.write[0], value=value, signed=signed, tag=tag)


# ---------------------------------------------------------------------------
# Cast ops
# ---------------------------------------------------------------------------

def _resolve_sort(sort, shape) -> Real | Int | Bool | Word | Zero:
    """`sort` may be a concrete Sort (e.g. from coercion) or a family (Real/Int/Bool)."""

    if isinstance(sort, Real | Int | Bool | Word | Zero):
        return _with_shape(sort, shape)
    if sort is Bool:
        return Bool(shape)
    if sort is Int:
        return Int(shape)
    if sort is Real:
        return Real(shape)
    raise TypeError("a word literal needs a width: pass sort=Word(width, [...])")


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
        return _wrap(_X(v._wire), signed=getattr(v, "_signed", False))
    else:
        raise ValueError(f"expr.X expects an expression, got {v}")


def d(v: Expr) -> Expr:
    """The `derived` wire of a variable expr."""
    return _wrap(_d(v._wire), signed=getattr(v, "_signed", False))


def ite(cond: Expr, iftrue, iffalse) -> Expr:
    a, b = iftrue, iffalse
    if not isinstance(a, Expr) and not isinstance(b, Expr):
        raise TypeError("ite(): at least one branch must be an Expr")
    if not isinstance(a, Expr):
        a = expr(a, sort=b.dtype, signed=getattr(b, "_signed", False))
    if not isinstance(b, Expr):
        b = expr(b, sort=a.dtype, signed=getattr(a, "_signed", False))
    if _family(a.dtype) != _family(b.dtype):
        raise TypeError(f"ite branches have different sorts {a.dtype}, {b.dtype}; use cast()")
    match a.dtype:
        case Real():
            op = tensor.real.Ite([cond.dtype, a.dtype, b.dtype])
        case Int():
            op = tensor.int.Ite([cond.dtype, a.dtype, b.dtype])
        case Bool():
            op = tensor.bool.Ite([cond.dtype, a.dtype, b.dtype])
        case Word():
            op = tensor.word.Ite([cond.dtype, a.dtype, b.dtype])
    return a._apply(op, cond, a, b)


def relu(e: Expr) -> Expr:
    match e.dtype:
        case Real():
            op = tensor.real.ReLU(e.dtype)
        case Int():
            op = tensor.int.ReLU(e.dtype)
        case _:
            raise TypeError(f"relu() needs a Real or Int operand, got {e.dtype}")
    return e._apply(op, e)


def argmax(e: Expr) -> Expr:
    """The index of a maximal entry, as a [1, 1] matrix of the operand's sort."""
    match e.dtype:
        case Real():
            # the index is a plain real
            op = tensor.real.Argmax(e.dtype)
        case Int():
            op = tensor.int.Argmax(e.dtype)
        case _:
            raise TypeError(f"argmax() needs a Real or Int operand, got {e.dtype}")
    return e._apply(op, e)
