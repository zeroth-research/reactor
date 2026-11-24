"""Expression: symbolic expressions for the DSL.

An ``Expr`` is a thin wrapper over a wire that lets you build ``Term``s by writing expressions
like ``x + 1`` or ``ite(x < y, x + 1, y)``: composing one immediately builds the underlying
``Term`` and **records it in the active collector** (see ``collecting``). Exprs
are split by *operation family* (the sort picks the class):

    AExpr  — Arithmetic   Int / Real     +  -  *  /  @   and comparisons (-> BExpr)
    BExpr  — Boolean       Bool           &  |  ~  ^
    WExpr  — Word          Word           arithmetic + bitwise + signed/unsigned + width

The operations are those of the namespace of the operands' sort, ``reactor.tensor.real``,
``.int``, ``.bool`` or ``.word``, sized by their shape: ``x + y`` on ``Real([2, 3])``
builds ``real.Add([2, 3])``, and on ``Word(8, [1, 1])`` builds ``word.Add(8, [1, 1])``.

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
        case Bool(_):
            return BExpr(wire, next=next, value=value, tag=tag)
        case Int(_) | Real(_):
            return AExpr(wire, next=next, value=value, tag=tag)
        case Word(_, _):
            return WExpr(wire, next=next, value=value, signed=signed, tag=tag)
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
            case Word(bw, _):
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

    def _binop(self, op, out: Real | Int | Bool | Word | Zero, o) -> "Expr":
        """Coerce `o` to my sort, then build a binary Term with `op` and output sort `out`."""
        o = self._coerce_same(o)
        return self._result(Term(op, [Wire(out)], [self._wire, o._wire]))

    def _unop(self, op, out: Real | Int | Bool | Word | Zero | None = None) -> "Expr":
        out = out if out is not None else self.dtype
        return self._result(Term(op, [Wire(out)], [self._wire]))

    @override
    def __repr__(self) -> str:
        label = f"{self._tag!r} " if self._tag is not None else ""
        return f"{type(self).__name__}({label}#{self._wire.id}, {self._wire.dtype})"


# ---------------------------------------------------------------------------
# AExpr — arithmetic (Real / Int): the operations of `real` / `int`
# ---------------------------------------------------------------------------


class AExpr(Expr):
    def __add__(self, o):
        match self.dtype:
            case Real(s):
                op = tensor.real.Add(s, degree=self.dtype.degree)
            case Int(s):
                op = tensor.int.Add(s)
        return self._binop(op, self.dtype, o)

    def __sub__(self, o):
        match self.dtype:
            case Real(s):
                op = tensor.real.Sub(s, degree=self.dtype.degree)
            case Int(s):
                op = tensor.int.Sub(s)
        return self._binop(op, self.dtype, o)

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
        match self.dtype:
            case Real(s):
                # the degrees of the factors add up
                p, q = self.dtype.degree, o.dtype.degree
                return self._binop(tensor.real.Mul(s, degrees=[p, q]), Real(s, degree=p + q), o)
            case Int(s):
                op = tensor.int.Mul(s)
        return self._binop(op, self.dtype, o)

    def _scale(self, c) -> "Expr":
        """The linear map `x ↦ C ⊙ x` by the constant `c` of my shape, sharing its tensor
        unless its element kind is not of my sort."""
        assert isinstance(c, torch.Tensor), f"the constant must be a tensor, got {type(c)}"
        assert list(c.shape) == self.shape, f"cannot scale {self.shape} by a constant of {list(c.shape)}"
        match self.dtype:
            case Real(_):
                op = tensor.real.Scale(c if c.is_floating_point() else c.float(), degree=self.dtype.degree)
            case Int(_):
                integer = not c.is_floating_point() and c.dtype != torch.bool
                op = tensor.int.Scale(c if integer else c.long())
        return self._unop(op)

    def __neg__(self):
        match self.dtype:
            case Real(s):
                op = tensor.real.Neg(s, degree=self.dtype.degree)
            case Int(s):
                op = tensor.int.Neg(s)
        return self._unop(op)

    def __abs__(self):
        match self.dtype:
            case Real(s):
                op = tensor.real.Abs(s, degree=self.dtype.degree)
            case Int(s):
                op = tensor.int.Abs(s)
        return self._unop(op)

    def __truediv__(self, o):
        match self.dtype:
            case Real(s):
                # the degree of the divisor is taken off that of the dividend
                o = self._coerce_same(o)
                p, q = self.dtype.degree, o.dtype.degree
                return self._binop(tensor.real.Div(s, degrees=[p, q]), Real(s, degree=p - q), o)
            case Int(s):
                op = tensor.int.Div(s)
        return self._binop(op, self.dtype, o)

    def __radd__(self, o):  return self._coerce(o).__add__(self)

    def __rsub__(self, o):  return self._coerce(o).__sub__(self)

    def __rmul__(self, o):  return self.__mul__(o)  # pointwise, so it commutes

    def __rtruediv__(self, o):  return self._coerce(o).__truediv__(self)

    def __matmul__(self, o):
        o = self._coerce_same(o)
        out = _with_shape(self.dtype, [self.shape[0], o.shape[1]])
        match self.dtype:
            case Real(s):
                # the degrees of the factors add up
                p, q = self.dtype.degree, o.dtype.degree
                op = tensor.real.MatMul([s, o.shape], degrees=[p, q])
                out = Real(out.shape, degree=p + q)
            case Int(s):
                op = tensor.int.MatMul([s, o.shape])
        return self._result(Term(op, [Wire(out)], [self._wire, o._wire]))

    # comparisons -> Bool -> BExpr
    def __eq__(self, o):
        match self.dtype:
            case Real(s):
                op = tensor.real.Eq(s, degree=self.dtype.degree)
            case Int(s):
                op = tensor.int.Eq(s)
        return self._binop(op, Bool(self.shape), o)

    def __ne__(self, o):
        match self.dtype:
            case Real(s):
                op = tensor.real.Ne(s, degree=self.dtype.degree)
            case Int(s):
                op = tensor.int.Ne(s)
        return self._binop(op, Bool(self.shape), o)

    def __lt__(self, o):
        match self.dtype:
            case Real(s):
                op = tensor.real.Lt(s, degree=self.dtype.degree)
            case Int(s):
                op = tensor.int.Lt(s)
        return self._binop(op, Bool(self.shape), o)

    def __le__(self, o):
        match self.dtype:
            case Real(s):
                op = tensor.real.Le(s, degree=self.dtype.degree)
            case Int(s):
                op = tensor.int.Le(s)
        return self._binop(op, Bool(self.shape), o)

    def __gt__(self, o):
        match self.dtype:
            case Real(s):
                op = tensor.real.Gt(s, degree=self.dtype.degree)
            case Int(s):
                op = tensor.int.Gt(s)
        return self._binop(op, Bool(self.shape), o)

    def __ge__(self, o):
        match self.dtype:
            case Real(s):
                op = tensor.real.Ge(s, degree=self.dtype.degree)
            case Int(s):
                op = tensor.int.Ge(s)
        return self._binop(op, Bool(self.shape), o)


# ---------------------------------------------------------------------------
# BExpr — boolean (Bool): the operations of `bool`
# ---------------------------------------------------------------------------


class BExpr(Expr):
    def __eq__(self, o):   return self._binop(tensor.bool.Eq(self.shape), self.dtype, o)

    def __ne__(self, o):   return self._binop(tensor.bool.Ne(self.shape), self.dtype, o)

    def __and__(self, o):  return self._binop(tensor.bool.And(self.shape), self.dtype, o)

    def __or__(self, o):   return self._binop(tensor.bool.Or(self.shape), self.dtype, o)

    def __xor__(self, o):  return self._binop(tensor.bool.Xor(self.shape), self.dtype, o)

    def __invert__(self):  return self._unop(tensor.bool.Not(self.shape))


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

    def _w(self):
        """The sizes of my word operations: the width, and the shape."""
        return self._bw(), self.shape

    def __eq__(self, o):   return self._binop(tensor.word.Eq(*self._w()), Bool(self.shape), o)

    def __ne__(self, o):   return self._binop(tensor.word.Ne(*self._w()), Bool(self.shape), o)

    # arithmetic, modulo 2^width
    def __add__(self, o):  return self._binop(tensor.word.Add(*self._w()), self.dtype, o)

    def __sub__(self, o):  return self._binop(tensor.word.Sub(*self._w()), self.dtype, o)

    def __mul__(self, o):  return self._binop(tensor.word.Mul(*self._w()), self.dtype, o)

    def __radd__(self, o):  return self._coerce(o).__add__(self)

    def __rsub__(self, o):  return self._coerce(o).__sub__(self)

    def __rmul__(self, o):  return self._coerce(o).__mul__(self)

    def __neg__(self):     return self._unop(tensor.word.Neg(*self._w()))

    def __floordiv__(self, o):
        op = tensor.word.signed.Div(*self._w()) if self._signed else tensor.word.unsigned.Div(*self._w())
        return self._binop(op, self.dtype, o)

    def __mod__(self, o):
        op = tensor.word.signed.Mod(*self._w()) if self._signed else tensor.word.unsigned.Rem(*self._w())
        return self._binop(op, self.dtype, o)

    # comparisons pick signed/unsigned; they write booleans
    def __lt__(self, o):
        op = tensor.word.signed.Lt(*self._w()) if self._signed else tensor.word.unsigned.Lt(*self._w())
        return self._binop(op, Bool(self.shape), o)

    def __le__(self, o):
        op = tensor.word.signed.Le(*self._w()) if self._signed else tensor.word.unsigned.Le(*self._w())
        return self._binop(op, Bool(self.shape), o)

    def __gt__(self, o):
        op = tensor.word.signed.Gt(*self._w()) if self._signed else tensor.word.unsigned.Gt(*self._w())
        return self._binop(op, Bool(self.shape), o)

    def __ge__(self, o):
        op = tensor.word.signed.Ge(*self._w()) if self._signed else tensor.word.unsigned.Ge(*self._w())
        return self._binop(op, Bool(self.shape), o)

    # bitwise, width-preserving
    def __and__(self, o):  return self._binop(tensor.word.And(*self._w()), self.dtype, o)

    def __or__(self, o):   return self._binop(tensor.word.Or(*self._w()), self.dtype, o)

    def __xor__(self, o):  return self._binop(tensor.word.Xor(*self._w()), self.dtype, o)

    def __invert__(self):  return self._unop(tensor.word.Not(*self._w()))


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
        case Bool(_):
            op = tensor.bool.Const(value)
        case Real(_):
            op = tensor.real.Const(value, degree=degree)
        case Int(_):
            op = tensor.int.Const(value)
        case Word(_, _):
            op = tensor.word.Const(value, bw)
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
    """`iftrue` if `cond` holds, else `iffalse`: one of the two whole branches, by the one
    boolean of `cond`, a [1, 1] matrix (`all_` reduces a comparison of matrices to one)."""
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
        case Real(s):
            op = tensor.real.Ite(s, degree=a.dtype.degree)
        case Int(s):
            op = tensor.int.Ite(s)
        case Bool(s):
            op = tensor.bool.Ite(s)
        case Word(w, s):
            op = tensor.word.Ite(w, s)
    term = Term(op, [Wire(a.dtype)], [cond._wire, a._wire, b._wire])
    _emit(term)
    return _wrap(term.write[0], signed=getattr(a, "_signed", False))


def all_(e: Expr) -> Expr:
    """Whether every entry of the boolean matrix `e` holds, as a [1, 1] matrix."""
    match e.dtype:
        case Bool(s):
            return e._unop(tensor.bool.All(s), out=Bool([1, 1]))
    raise TypeError(f"all_() needs a Bool operand, got {e.dtype}")


def relu(e: Expr) -> Expr:
    match e.dtype:
        case Real(s):
            return e._unop(tensor.real.ReLU(s, degree=e.dtype.degree))
        case Int(s):
            return e._unop(tensor.int.ReLU(s))
    raise TypeError(f"relu() needs a Real or Int operand, got {e.dtype}")


def _transcendental(name: str, *es: Expr) -> Expr:
    """The term of the transcendental function `name` (an operation of
    `reactor.tensor.real`, of plain reals) of the expressions `es`."""
    for e in es:
        if not isinstance(e, Expr):
            raise TypeError(f"{name.lower()}() needs expressions; fold a constant before")
        if not isinstance(e.dtype, Real) or e.dtype.degree != 0:
            raise TypeError(f"{name.lower()}() needs plain reals (degree 0), got {e.dtype}")
        if e.shape != es[0].shape:
            raise TypeError(f"{name.lower()}() needs operands of one shape, got {[x.shape for x in es]}")
    shape = es[0].shape
    term = Term(getattr(tensor.real, name)(shape), [Wire(Real(shape))], [e._wire for e in es])
    _emit(term)
    return _wrap(term.write[0])


def sqrt(e: Expr) -> Expr:
    """The pointwise square root, unspecified below zero."""
    return _transcendental("Sqrt", e)


def exp(e: Expr) -> Expr:
    """The pointwise exponential."""
    return _transcendental("Exp", e)


def log(e: Expr) -> Expr:
    """The pointwise natural logarithm, unspecified at and below zero."""
    return _transcendental("Log", e)


def sin(e: Expr) -> Expr:
    """The pointwise sine, of radians."""
    return _transcendental("Sin", e)


def cos(e: Expr) -> Expr:
    """The pointwise cosine, of radians."""
    return _transcendental("Cos", e)


def tan(e: Expr) -> Expr:
    """The pointwise tangent, unspecified at its poles."""
    return _transcendental("Tan", e)


def asin(e: Expr) -> Expr:
    """The pointwise arcsine, unspecified outside [-1, 1]."""
    return _transcendental("Asin", e)


def acos(e: Expr) -> Expr:
    """The pointwise arccosine, unspecified outside [-1, 1]."""
    return _transcendental("Acos", e)


def atan(e: Expr) -> Expr:
    """The pointwise arctangent."""
    return _transcendental("Atan", e)


def sinh(e: Expr) -> Expr:
    """The pointwise hyperbolic sine."""
    return _transcendental("Sinh", e)


def cosh(e: Expr) -> Expr:
    """The pointwise hyperbolic cosine."""
    return _transcendental("Cosh", e)


def tanh(e: Expr) -> Expr:
    """The pointwise hyperbolic tangent."""
    return _transcendental("Tanh", e)


def atan2(y: Expr, x: Expr) -> Expr:
    """The pointwise angle of the point `(x, y)`, in (-π, π]: `y` first, as in C;
    unspecified at (0, 0)."""
    return _transcendental("Atan2", y, x)


def argmax(e: Expr) -> Expr:
    """The index of a maximal entry, as a [1, 1] matrix of the operand's sort."""
    match e.dtype:
        case Real(s):
            # the index is a plain real
            return e._unop(tensor.real.Argmax(s, degree=e.dtype.degree), out=Real([1, 1]))
        case Int(s):
            op = tensor.int.Argmax(s)
        case _:
            raise TypeError(f"argmax() needs a Real or Int operand, got {e.dtype}")
    return e._unop(op, out=_with_shape(e.dtype, [1, 1]))
