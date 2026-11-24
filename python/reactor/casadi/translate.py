"""Translate CasADi functions into Reactor terms over real arithmetic (the logic NRAT).

The translation walks the instruction list of the function's scalar (SX) form, one
instruction at a time, and emits one Reactor ``Term`` per instruction that is not
constant. Every value is a scalar ``Real([1, 1])`` wire, so no tensor support is needed
on the Reactor side.

Semantics
---------
* **Constants are float64**, exactly as CasADi evaluates them. Constant subexpressions
  (``1 / mass``, ``2 * L``, ``sin(0.3)``) are folded in float64 by the translator, so
  they never reach Reactor as operations.
* **Products and quotients.** A product with a constant becomes ``real.Scale``, and so
  does a division by a constant, as the product with its float64 reciprocal: this keeps
  the linear fragment linear, and is the one place where the translation deviates from
  real arithmetic on the CasADi graph (at most one ulp relative per division). A product
  of two variables becomes ``real.Mul``, a division by a variable ``real.Div`` (as in
  SMT-LIB2, ``x / 0`` is unspecified there, where CasADi gives ``inf`` or ``nan``),
  ``sq`` a ``Mul`` of the operand with itself, and an integer power repeated ``Mul``\\ s.
* **Comparisons** in CasADi yield the reals 0/1. Here they become ``Bool`` wires and are
  converted back (``ite(b, 1, 0)``) only where a real is needed, so ``if_else`` becomes a
  plain ``Ite``.
* ``neg`` and ``fabs`` become ``real.Neg`` and ``real.Abs``; ``fmin``/``fmax``/``sign``
  become ``Ite`` terms (``real.Min``/``real.Max`` reduce a single matrix, they are not
  binary).
* **Transcendental functions** become the operations of the same name in
  ``reactor.tensor.real`` (the logic NRAT): ``sqrt``, ``exp``, ``log``, ``sin``,
  ``cos``, ``tan``, ``asin``, ``acos``, ``atan``, ``sinh``, ``cosh``, ``tanh`` and
  ``atan2`` (``y`` first, as in CasADi and C); ``hypot(a, b)`` becomes
  ``sqrt(a*a + b*b)``. Where CasADi gives ``nan`` (``sqrt`` of a negative), the theory
  leaves the value unspecified. ``asinh``, ``acosh``, ``atanh``, ``expm1``, ``log1p``,
  ``erf``, ``floor``, ``ceil``, ``fmod`` and non-integer powers of variables raise
  ``UnsupportedOp``. All are folded when their arguments are constant.

Entry points
------------
``translate``     translate a function given Reactor expressions for its inputs (call
                  inside ``reactor.expr.collecting()``)
``to_terms``      translate over fresh input wires; returns terms and wires
``sequential``    a discrete-time module ``x' = step(x, u...)``
``differential``  a continuous-time module ``d(x) = f(x, u...)``
``euler``, ``rk4``  turn a continuous-time CasADi function into a step function
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import casadi as cs
import torch

from ..builder import TheoryError
import sys

from ..expr import AExpr, BExpr, Expr, _emit, _wrap, collecting, expr, ite
from ..reactor import Module, Term, Var, Wire, X, d, tensor
from ..reactor.tensor import Havoc, Skip
from ..sort import Real

__all__ = [
    "UnsupportedOp",
    "Value",
    "translate",
    "as_real",
    "fold",
    "to_terms",
    "sequential",
    "differential",
    "CasadiModule",
    "Terms",
    "euler",
    "rk4",
    "scalar_vars",
    "op_name",
]

DTYPE = torch.float64
SCALAR = [1, 1]
REAL = Real(SCALAR)
RATE = Real(SCALAR, degree=1)  # the sort of d(x) for a real scalar x

#: A translated value: a float64 constant, a real expression or a boolean expression.
Value = float | AExpr | BExpr

_OP_NAMES = {getattr(cs, n): n[3:].lower() for n in dir(cs) if n.startswith("OP_")}


def op_name(op: int) -> str:
    """Human-readable name of a CasADi opcode (``cs.OP_MUL`` -> ``"mul"``)."""
    return _OP_NAMES.get(op, f"op#{op}")


# The module `reactor.expr`, whose functions build the transcendental operations
# (`reactor.expr` itself names the function `expr`, re-exported by the package).
_dsl = sys.modules[AExpr.__module__]

#: The CasADi opcodes of the transcendental functions, and the functions of
#: ``reactor.expr`` that build the operations of ``reactor.tensor.real`` for them.
TRANSCENDENTAL = {
    cs.OP_SQRT: "sqrt",
    cs.OP_EXP: "exp",
    cs.OP_LOG: "log",
    cs.OP_SIN: "sin",
    cs.OP_COS: "cos",
    cs.OP_TAN: "tan",
    cs.OP_ASIN: "asin",
    cs.OP_ACOS: "acos",
    cs.OP_ATAN: "atan",
    cs.OP_SINH: "sinh",
    cs.OP_COSH: "cosh",
    cs.OP_TANH: "tanh",
}


class UnsupportedOp(TheoryError):
    """A CasADi operation with no counterpart in Reactor's theories (e.g. ``sin`` of a
    variable)."""

    def __init__(self, op: int, detail: str = ""):
        self.op = op
        self.name = op_name(op)
        msg = f"no Reactor operation for CasADi '{self.name}' of a non-constant operand"
        super().__init__(msg + (f" ({detail})" if detail else ""))


# ---------------------------------------------------------------------------
# Constant folding (shared with the Z3 reference in `validate`, so both sides
# of a miter agree on what a constant subexpression evaluates to).
# ---------------------------------------------------------------------------


def _div(a: float, b: float) -> float:
    # IEEE semantics, as CasADi computes; Python's `/` raises on zero.
    return float(torch.tensor(a, dtype=DTYPE) / torch.tensor(b, dtype=DTYPE))


def _sign(a: float) -> float:
    return 1.0 if a > 0 else -1.0 if a < 0 else 0.0


_FOLD1 = {
    cs.OP_NEG: lambda a: -a,
    cs.OP_TWICE: lambda a: 2.0 * a,
    cs.OP_SQ: lambda a: a * a,
    cs.OP_INV: lambda a: _div(1.0, a),
    cs.OP_FABS: abs,
    cs.OP_SIGN: _sign,
    cs.OP_NOT: lambda a: float(a == 0.0),
    cs.OP_SQRT: math.sqrt,
    cs.OP_SIN: math.sin,
    cs.OP_COS: math.cos,
    cs.OP_TAN: math.tan,
    cs.OP_ASIN: math.asin,
    cs.OP_ACOS: math.acos,
    cs.OP_ATAN: math.atan,
    cs.OP_SINH: math.sinh,
    cs.OP_COSH: math.cosh,
    cs.OP_TANH: math.tanh,
    cs.OP_ASINH: math.asinh,
    cs.OP_ACOSH: math.acosh,
    cs.OP_ATANH: math.atanh,
    cs.OP_EXP: math.exp,
    cs.OP_EXPM1: math.expm1,
    cs.OP_LOG: math.log,
    cs.OP_LOG1P: math.log1p,
    cs.OP_FLOOR: lambda a: float(math.floor(a)),
    cs.OP_CEIL: lambda a: float(math.ceil(a)),
    cs.OP_ERF: math.erf,
}

_FOLD2 = {
    cs.OP_ADD: lambda a, b: a + b,
    cs.OP_SUB: lambda a, b: a - b,
    cs.OP_MUL: lambda a, b: a * b,
    cs.OP_DIV: _div,
    cs.OP_LT: lambda a, b: float(a < b),
    cs.OP_LE: lambda a, b: float(a <= b),
    cs.OP_EQ: lambda a, b: float(a == b),
    cs.OP_NE: lambda a, b: float(a != b),
    cs.OP_AND: lambda a, b: float(a != 0.0 and b != 0.0),
    cs.OP_OR: lambda a, b: float(a != 0.0 or b != 0.0),
    cs.OP_IF_ELSE_ZERO: lambda c, a: a if c != 0.0 else 0.0,
    cs.OP_FMIN: min,
    cs.OP_FMAX: max,
    cs.OP_POW: math.pow,
    cs.OP_CONSTPOW: math.pow,
    cs.OP_ATAN2: math.atan2,
    cs.OP_HYPOT: math.hypot,
    cs.OP_FMOD: math.fmod,
    cs.OP_REMAINDER: math.remainder,
    cs.OP_COPYSIGN: math.copysign,
}


def fold(op: int, args: Sequence[float]) -> float:
    """Evaluate opcode `op` on float64 constants."""
    table = _FOLD1 if len(args) == 1 else _FOLD2
    if op not in table:
        raise TheoryError(f"cannot fold CasADi op '{op_name(op)}' on constants")
    return float(table[op](*args))


def integer_exponent(b: float) -> int | None:
    """`b` as an integer exponent the translation expands into products, else None."""
    return int(b) if float(b).is_integer() and abs(b) <= 64 else None


# ---------------------------------------------------------------------------
# Term construction (float64 throughout: the `expr` literals default to float32)
# ---------------------------------------------------------------------------


def _const(c: float, sort=REAL) -> AExpr:
    t = torch.tensor([[c]], dtype=DTYPE)
    term = Term.constant(tensor.real.Const(t, degree=sort.degree), [Wire(sort)])
    _emit(term)
    return _wrap(term.write[0], value=t)


def _scale(a: AExpr, c: float) -> Value:
    if c == 1.0:
        return a
    if c == 0.0:
        return 0.0
    if c == -1.0:
        return -a
    t = torch.tensor([[c]], dtype=DTYPE)
    term = Term(tensor.real.Scale(t, degree=a.dtype.degree), [Wire(a.dtype)], [a.wire])
    _emit(term)
    return _wrap(term.write[0])


def _mul(a: AExpr, b: AExpr) -> AExpr:
    p, q = a.dtype.degree, b.dtype.degree
    op = tensor.real.Mul(SCALAR, degrees=[p, q])
    term = Term(op, [Wire(Real(SCALAR, degree=p + q))], [a.wire, b.wire])
    _emit(term)
    return _wrap(term.write[0])


def _quot(a: AExpr, b: AExpr) -> AExpr:
    p, q = a.dtype.degree, b.dtype.degree
    op = tensor.real.Div(SCALAR, degrees=[p, q])
    term = Term(op, [Wire(Real(SCALAR, degree=p - q))], [a.wire, b.wire])
    _emit(term)
    return _wrap(term.write[0])


def _power(a: AExpr, n: int) -> Value:
    """`a ** n` for an integer `n`, by repeated squaring."""
    if n == 0:
        return 1.0
    if n < 0:
        return _quot(_const(1.0), _real(_power(a, -n)))
    result, base = None, a
    while n:
        if n & 1:
            result = base if result is None else _mul(result, base)
        n >>= 1
        if n:
            base = _mul(base, base)
    return result


def _real(v: Value) -> AExpr:
    """`v` as a real expression (materializing constants, 0/1 for booleans)."""
    if isinstance(v, float):
        return _const(v)
    if isinstance(v, BExpr):
        return ite(v, _const(1.0), _const(0.0))
    return v


def _bool(v: AExpr | BExpr) -> BExpr:
    """`v` as a condition, with CasADi's reading: nonzero is true."""
    if isinstance(v, BExpr):
        return v
    return v != _const(0.0)


def _is_const(v: Value) -> bool:
    return isinstance(v, float)


def _is_zero(v: Value) -> bool:
    # never compare an Expr with `==`: that builds an Eq term
    return isinstance(v, float) and v == 0.0


# ---------------------------------------------------------------------------
# Instruction translation
# ---------------------------------------------------------------------------


def _unary(op: int, a: Value) -> Value:
    if _is_const(a):
        return fold(op, [a])
    match op:
        case cs.OP_NEG:
            return -_real(a)
        case cs.OP_TWICE:
            return _scale(_real(a), 2.0)
        case cs.OP_SQ:
            r = _real(a)
            return _mul(r, r)
        case cs.OP_INV:
            return _quot(_const(1.0), _real(a))
        case cs.OP_NOT:
            return ~_bool(a)
        case cs.OP_FABS:
            return abs(_real(a))
        case cs.OP_SIGN:
            r = _real(a)
            zero = _const(0.0)
            return ite(r > zero, _const(1.0), ite(r < zero, _const(-1.0), zero))
    if op in TRANSCENDENTAL:
        return getattr(_dsl, TRANSCENDENTAL[op])(_real(a))
    raise UnsupportedOp(op)


def _binary(op: int, a: Value, b: Value) -> Value:
    if _is_const(a) and _is_const(b):
        return fold(op, [a, b])
    match op:
        case cs.OP_ADD:
            if _is_zero(a):
                return b
            if _is_zero(b):
                return a
            return _real(a) + _real(b)
        case cs.OP_SUB:
            if _is_zero(b):
                return a
            if _is_zero(a):
                return -_real(b)
            return _real(a) - _real(b)
        case cs.OP_MUL:
            if _is_const(a):
                return _scale(_real(b), a)
            if _is_const(b):
                return _scale(_real(a), b)
            return _mul(_real(a), _real(b))
        case cs.OP_DIV:
            if _is_const(b):
                if b == 0.0:
                    raise TheoryError("division of a non-constant by the constant zero")
                return _scale(_real(a), _div(1.0, b))
            return _quot(_real(a), _real(b))
        case cs.OP_LT:
            return _real(a) < _real(b)
        case cs.OP_LE:
            return _real(a) <= _real(b)
        case cs.OP_EQ:
            return _real(a) == _real(b)
        case cs.OP_NE:
            return _real(a) != _real(b)
        case cs.OP_AND:
            if _is_const(a):
                return _bool(b) if a != 0.0 else 0.0
            if _is_const(b):
                return _bool(a) if b != 0.0 else 0.0
            return _bool(a) & _bool(b)
        case cs.OP_OR:
            if _is_const(a):
                return _bool(b) if a == 0.0 else 1.0
            if _is_const(b):
                return _bool(a) if b == 0.0 else 1.0
            return _bool(a) | _bool(b)
        case cs.OP_IF_ELSE_ZERO:
            if _is_const(a):  # constant condition
                return b if a != 0.0 else 0.0
            if _is_zero(b):
                return 0.0
            return ite(_bool(a), _real(b), _const(0.0))
        case cs.OP_FMIN:
            ra, rb = _real(a), _real(b)
            return ite(ra <= rb, ra, rb)
        case cs.OP_FMAX:
            ra, rb = _real(a), _real(b)
            return ite(ra >= rb, ra, rb)
        case cs.OP_POW | cs.OP_CONSTPOW if _is_const(b) and integer_exponent(b) is not None:
            return _power(_real(a), integer_exponent(b))
        case cs.OP_POW | cs.OP_CONSTPOW:
            raise UnsupportedOp(op, "only constant integer exponents expand into products")
        case cs.OP_ATAN2:
            return _dsl.atan2(_real(a), _real(b))
        case cs.OP_HYPOT:
            ra, rb = _real(a), _real(b)
            return _dsl.sqrt(_mul(ra, ra) + _mul(rb, rb))
    raise UnsupportedOp(op)


def _as_sx(f: cs.Function) -> cs.Function:
    if not isinstance(f, cs.Function):
        raise TypeError(f"expected a casadi.Function, got {type(f).__name__}")
    if f.has_free():
        raise ValueError(f"{f.name()}: function has free variables {f.get_free()}")
    return f if f.is_a("SXFunction") else f.expand()


def _input_value(v) -> Value:
    if isinstance(v, (AExpr, BExpr)):
        return v
    if isinstance(v, Expr):
        raise TypeError(f"unsupported input expression {v!r}")
    if isinstance(v, (Var, Wire)):
        return expr(v)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    raise TypeError(f"input must be an Expr, Var, Wire or number, got {type(v).__name__}")


def _dense_input(f: cs.Function, i: int, arg) -> list:
    n = f.numel_in(i)
    vals = list(arg) if isinstance(arg, (list, tuple)) else [arg]
    if len(vals) != n:
        raise ValueError(
            f"{f.name()}: input {i} ('{f.name_in(i)}') has {n} elements "
            f"(column-major), got {len(vals)}"
        )
    return vals


def translate(f: cs.Function, inputs: Sequence) -> list[list[Value]]:
    """Translate `f` applied to `inputs`; returns its outputs as Reactor values.

    Must be called inside ``reactor.expr.collecting()``: every emitted ``Term`` is
    recorded there, in dependency order.

    Args:
        f: A CasADi function (SX, or MX that can be expanded to SX).
        inputs: One entry per input of `f`: a sequence of ``numel`` values in
            column-major order (a bare value for a scalar input). A value is an
            ``Expr``, a ``Var``/``Wire`` (wrapped as an expression) or a number (a
            constant, which partially evaluates `f`).

    Returns:
        One list per output of `f`, of ``numel`` values in column-major order. A value
        is a float (when the entry is constant) or an ``AExpr``/``BExpr``; ``as_real``
        turns any of them into a real expression with a wire.

    Raises:
        UnsupportedOp: `f` applies a function with no Reactor operation (``erf``,
            ``floor``, a non-integer power, ...) to a non-constant.
    """
    f = _as_sx(f)
    if len(inputs) != f.n_in():
        raise ValueError(f"{f.name()}: expected {f.n_in()} inputs, got {len(inputs)}")
    nz_in = []
    for i, arg in enumerate(inputs):
        dense = _dense_input(f, i, arg)
        nz_in.append([_input_value(dense[k]) for k in f.sparsity_in(i).find()])

    outs: list[list[Value]] = [[0.0] * f.numel_out(j) for j in range(f.n_out())]
    nz_out = [f.sparsity_out(j).find() for j in range(f.n_out())]
    reg: dict[int, Value] = {}
    for k in range(f.n_instructions()):
        op = f.instruction_id(k)
        i, o = f.instruction_input(k), f.instruction_output(k)
        if op == cs.OP_INPUT:
            reg[o[0]] = nz_in[i[0]][i[1]]
        elif op == cs.OP_OUTPUT:
            outs[o[0]][nz_out[o[0]][o[1]]] = reg[i[0]]
        elif op == cs.OP_CONST:
            reg[o[0]] = float(f.instruction_constant(k))
        elif op == cs.OP_ASSIGN:
            reg[o[0]] = reg[i[0]]
        elif len(i) == 1:
            reg[o[0]] = _unary(op, reg[i[0]])
        elif len(i) == 2:
            reg[o[0]] = _binary(op, reg[i[0]], reg[i[1]])
        else:
            raise UnsupportedOp(op, f"{len(i)} operands")
    return outs


def as_real(v: Value) -> AExpr:
    """A translated value as a real expression (emits a term for a constant)."""
    return _real(v)


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------


class Terms(NamedTuple):
    """`f` translated over fresh, undriven input wires."""

    terms: list[Term]
    inputs: list[list[Wire]]  # per input of f, numel wires (column-major)
    outputs: list[list[Wire]]  # per output of f, numel wires (column-major)


def to_terms(f: cs.Function) -> Terms:
    """Translate `f` over fresh input wires, materializing every output entry."""
    f = _as_sx(f)
    in_wires = [[Wire(REAL) for _ in range(f.numel_in(i))] for i in range(f.n_in())]
    with collecting() as terms:
        args = [[expr(w) for w in ws] for ws in in_wires]
        outs = translate(f, args)
        out_wires = [[_real(v).wire for v in out] for out in outs]
    return Terms(terms, in_wires, out_wires)


def scalar_vars(n: int) -> tuple[Var, ...]:
    """`n` fresh scalar real variables."""
    return tuple(Var(REAL) for _ in range(n))


class CasadiModule(NamedTuple):
    """A module built from a CasADi function."""

    module: Module
    state: tuple[Var, ...]  # controlled: one per entry of the function's first output
    inputs: tuple[tuple[Var, ...], ...]  # external: one tuple per further input


def _prepare(f, init, state, inputs, what, wait=None):
    f = _as_sx(f)
    n = f.numel_in(0)
    if f.n_out() < 1 or f.numel_out(0) != n:
        raise ValueError(
            f"{f.name()}: first output ({what}) must have the state's {n} elements, "
            f"got {f.numel_out(0) if f.n_out() else 'none'}"
        )
    state = tuple(state) if state is not None else scalar_vars(n)
    if len(state) != n:
        raise ValueError(f"expected {n} state variables, got {len(state)}")
    if inputs is None:
        inputs = tuple(scalar_vars(f.numel_in(i)) for i in range(1, f.n_in()))
    inputs = tuple(tuple(u) for u in inputs)
    if len(inputs) != f.n_in() - 1:
        raise ValueError(f"expected {f.n_in() - 1} input groups, got {len(inputs)}")
    for i, u in enumerate(inputs):
        if len(u) != f.numel_in(i + 1):
            raise ValueError(f"input group {i}: expected {f.numel_in(i + 1)} variables, got {len(u)}")
    wait = _wait(wait, len(inputs))
    if isinstance(init, cs.Function):
        init = _as_sx(init)
        if init.n_in() != len(inputs) or any(init.numel_in(i) != len(u) for i, u in enumerate(inputs)):
            raise ValueError(f"{init.name()}: an initial-state function takes the {len(inputs)} input "
                             "groups (not the state)")
        if init.numel_out(0) != n:
            raise ValueError(f"{init.name()}: first output must have the state's {n} elements")
        _check_init_reads(init, wait, "an initial-state function")
    else:
        init = list(init) if init is not None else [None] * n
        if len(init) != n:
            raise ValueError(f"expected {n} initial values, got {len(init)}")
    return f, state, inputs, init, wait


def _wait(wait, k: int) -> tuple[bool, ...]:
    """Per input group, whether it is read awaited (``X(v)``, this round's value)."""
    if wait is None or isinstance(wait, bool):
        return (bool(wait),) * k
    wait = tuple(bool(w) for w in wait)
    if len(wait) != k:
        raise ValueError(f"expected {k} wait flags (one per input group), got {len(wait)}")
    return wait


def _check_init_reads(f: cs.Function, wait, what: str) -> None:
    """The init block may read only this round's values (``X(v)``): refuse a latched input
    group that `f` depends on, which the core would reject later with a bare wire id."""
    x = [cs.SX.sym(f"i{i}", f.sparsity_in(i)) for i in range(f.n_in())]
    y = cs.vertcat(*[cs.vec(o) for o in f.call(x)])
    latched = [i for i, w in enumerate(wait) if not w and cs.depends_on(y, x[i])]
    if latched:
        raise ValueError(f"{f.name()}: {what} runs in the init round, which reads only this round's "
                         f"values; input group(s) {latched} are read latched (pass wait=True for them)")


def _read(inputs, wait) -> list[list[AExpr]]:
    """The expressions an atom reads its input groups as: awaited (``X(v)``) or latched."""
    return [[expr(X(v)) if w else expr(v) for v in u] for u, w in zip(inputs, wait)]


def _init_block(state, init, inputs=(), wait=()) -> list[Term]:
    with collecting() as terms:
        if isinstance(init, cs.Function):
            x0 = translate(init, _read(inputs, wait))[0]
            for v, val in zip(state, x0):
                terms.append(Term(Skip(REAL), [X(v)], [_real(val).wire]))
            return terms
        for v, c in zip(state, init):
            if c is None:
                terms.append(Term(Havoc(REAL), [X(v)], []))
            else:
                terms.append(Term(Skip(REAL), [X(v)], [_const(float(c)).wire]))
    return terms


def sequential(
    step: cs.Function,
    init: Sequence[float | None] | cs.Function | None = None,
    *,
    state: Sequence[Var] | None = None,
    inputs: Sequence[Sequence[Var]] | None = None,
    wait: bool | Sequence[bool] | None = None,
    hide=None,
) -> CasadiModule:
    """Build a sequential module ``x' = step(x, u1, ..., uk)[0]``.

    Args:
        step: Function whose first input is the state, the remaining inputs are
            external, and whose first output is the next state (same size).
        init: Initial value per state entry; ``None`` (or an entry ``None``) leaves it
            unconstrained (a havoc). Or a CasADi function of the input groups (not the
            state) whose first output is the initial state; the groups it depends on must be
            awaited (`wait`), since the init round has no latched values.
        state: Variables for the state, created if not given.
        inputs: Variables for each further input, created if not given.
        wait: Per input group (or one flag for all), whether the module reads this
            round's value ``X(v)`` (it then awaits the module that controls `v`, which
            must not await it back) instead of the latched one; default: latched.
        hide: Variables to hide in the built module.
    """
    step, state, inputs, init, wait = _prepare(step, init, state, inputs, "the next state", wait)
    init_terms = _init_block(state, init, inputs, wait)
    with collecting() as update_terms:
        nxt = translate(step, [[expr(v) for v in state]] + _read(inputs, wait))[0]
        for v, val in zip(state, nxt):
            update_terms.append(Term(Skip(REAL), [X(v)], [_real(val).wire]))
    variables = state + tuple(u for group in inputs for u in group)
    module = Module(init=init_terms, update=update_terms, vars=variables, hide=hide)
    return CasadiModule(module, state, inputs)


def combinatorial(
    f: cs.Function,
    *,
    inputs: Sequence[Sequence[Var]] | None = None,
    outputs: Sequence[Var] | None = None,
    wait: bool | Sequence[bool] | None = True,
    hide=None,
) -> CasadiModule:
    """Build a combinatorial module ``y = f(u1, ..., uk)[0]``: no state of its own; every
    round (the init round too) it assigns its outputs from its inputs.

    Args:
        f: Function of the input groups; its first output is assigned to `outputs`.
        inputs: Variables for each input, created if not given.
        outputs: Variables for the output, created if not given (the module controls them).
        wait: Per input group (or one flag for all), whether it reads this round's value
            ``X(v)`` (default) or the latched one. A group `f` depends on must be awaited:
            the module also runs in the init round, which has no latched values yet.
        hide: Variables to hide in the built module.

    The returned `CasadiModule` has the outputs as its `state` (what it controls).
    """
    f = _as_sx(f)
    n = f.numel_out(0)
    outputs = tuple(outputs) if outputs is not None else scalar_vars(n)
    if len(outputs) != n:
        raise ValueError(f"expected {n} output variables, got {len(outputs)}")
    if inputs is None:
        inputs = tuple(scalar_vars(f.numel_in(i)) for i in range(f.n_in()))
    inputs = tuple(tuple(u) for u in inputs)
    if len(inputs) != f.n_in():
        raise ValueError(f"expected {f.n_in()} input groups, got {len(inputs)}")
    for i, u in enumerate(inputs):
        if len(u) != f.numel_in(i):
            raise ValueError(f"input group {i}: expected {f.numel_in(i)} variables, got {len(u)}")
    wait = _wait(wait, len(inputs))
    _check_init_reads(f, wait, "a combinatorial module")
    with collecting() as terms:
        y = translate(f, _read(inputs, wait))[0]
        for v, val in zip(outputs, y):
            terms.append(Term(Skip(REAL), [X(v)], [_real(val).wire]))
    variables = outputs + tuple(u for group in inputs for u in group)
    module = Module.combinatorial(variables, terms, hide=hide)
    return CasadiModule(module, outputs, inputs)


def differential(
    f: cs.Function,
    init: Sequence[float | None] | None = None,
    *,
    state: Sequence[Var] | None = None,
    inputs: Sequence[Sequence[Var]] | None = None,
    hide=None,
) -> CasadiModule:
    """Build a differential module ``d(x) = f(x, u1, ..., uk)[0]``.

    The derivative wires carry rates of change (``Real([1, 1], degree=1)``): each entry
    of `f` is made one as the product with the rate ``1`` (a constant of degree 1), so
    the dynamics are kept exactly as `f` states them. The inputs are read as they are
    while the state flows.

    Args: as for ``sequential``, with `f` the time derivative of the state.
    """
    f, state, inputs, init, wait = _prepare(f, init, state, inputs, "the derivative")
    if isinstance(init, cs.Function):
        raise ValueError("a differential module takes initial values, not an initial-state function")
    init_terms = _init_block(state, init)
    with collecting() as delay_terms:
        xdot = translate(f, [list(state)] + [list(u) for u in inputs])[0]
        unit = None
        for v, val in zip(state, xdot):
            if isinstance(val, float):
                rate = _const(val, RATE)
            else:
                unit = unit if unit is not None else _const(1.0, RATE)
                rate = _mul(unit, _real(val))
            delay_terms.append(Term(Skip(RATE), [d(v)], [rate.wire]))
    variables = state + tuple(u for group in inputs for u in group)
    module = Module(init=init_terms, delay=delay_terms, vars=variables, hide=hide)
    return CasadiModule(module, state, inputs)


# ---------------------------------------------------------------------------
# Discretization, in CasADi (so a miter compares against exactly this step)
# ---------------------------------------------------------------------------


def _step_function(f: cs.Function, dt: float, method: str) -> cs.Function:
    f = _as_sx(f)
    if f.numel_out(0) != f.numel_in(0):
        raise ValueError(f"{f.name()}: first output (derivative) must match the state size")
    args = [cs.SX.sym(f.name_in(i), f.sparsity_in(i)) for i in range(f.n_in())]
    x, rest = args[0], args[1:]

    def rhs(x_):
        out = f(x_, *rest)
        return out[0] if isinstance(out, (list, tuple)) else out

    if method == "euler":
        x_next = x + dt * rhs(x)
    else:
        k1 = rhs(x)
        k2 = rhs(x + dt / 2 * k1)
        k3 = rhs(x + dt / 2 * k2)
        k4 = rhs(x + dt * k3)
        x_next = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    names_in = [f.name_in(i) for i in range(f.n_in())]
    return cs.Function(f"{f.name()}_{method}", args, [x_next], names_in, ["x_next"])


def euler(f: cs.Function, dt: float) -> cs.Function:
    """Explicit Euler step ``x + dt * f(x, ...)`` of a continuous-time function."""
    return _step_function(f, dt, "euler")


def rk4(f: cs.Function, dt: float) -> cs.Function:
    """Classical Runge-Kutta 4 step of a continuous-time function."""
    return _step_function(f, dt, "rk4")
