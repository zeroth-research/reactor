"""``reactor.sugar``: author a reactive Module by subclassing (à la ``torch.nn.Module``).

Subclass ``sugar.Module``, pass a ``theory`` and the ``ctrl`` (and optional ``extl``)
variables — ``reactor.Var`` objects — and override any of ``init`` / ``update`` / ``delay``.
**Instantiating the subclass *is* a base Module** — the methods run symbolically and the
module is built in the constructor:

    from reactor import LIA, Int, Var
    from reactor.sugar import Module

    class Counter(Module):
        def init(self):         return 0
        def update(self, x):    return x + 1

    x = Var(Int([1, 1]))
    m = Counter(theory=LIA, ctrl=(x,))      # a closed sequential base Module

The module is a single atom, and its interfaces come from the configuration: it
controls ``ctrl``, awaits ``extl``, and reads both ``ctrl`` and ``extl``. The methods
receive the variables **unpacked, as positional parameters**: ``init`` takes the ``extl``
variables, ``update`` and ``delay`` take ``ctrl`` followed by ``extl`` (all of them —
arities are checked). A parameter reads as the variable's latched value; for an ``extl``
variable ``v``, ``X(v)`` is its next value and ``d(v)`` its derivative. ``init`` has no
latched values: it uses ``X(v)`` only. Each method returns a tuple aligned with ``ctrl`` (a
single value for a single variable): entry *i* drives the next value (``init`` /
``update``) or the derivative (``delay``) of ``ctrl[i]``.

A method not overridden takes its default: ``init`` leaves the initial values arbitrary
(havoc), ``update`` keeps the values (skip), and ``delay`` keeps them constant (zero).
For example, an open hybrid clock:

    class Clock(Module):
        def init(self, t):        return 0
        def update(self, x, t):   return ite(X(t) == 1, 0, x)   # discrete reset
        def delay(self, x, t):    return 1 * d(t)               # continuous drift

    x, t = Var(Real([1, 1])), Var(Real([1, 1]))
    m = Clock(theory=LRA, ctrl=(x,), extl=(t,))

``ctrl`` and ``extl`` must be **ordered tuples**: their order is meaningful — it
determines how the variables bind to the positional parameters of ``init`` / ``update``
/ ``delay`` and how the returned values align with ``ctrl``. ``hide`` is only tested
for membership, so it can be any collection — preferably a ``set``. ``extl`` variables
are external inputs: declared but not driven here, so a module with ``extl`` is open (it
awaits them all, even those its methods do not use); ``hide`` hides the given variables in
the built module (partially observable).

Config comes from **constructor kwargs** (``theory=``, ``ctrl=``, ``extl=``, ``hide=``).
Two base-class constraints force this (both flagged for design review):
  * it cannot flow through ``super().__init__`` — the base Module is a frozen pyo3 class
    whose constructor runs at ``__new__`` (before any ``__init__``), so this builds there;
    and
  * config *class attributes* named ``ctrl``/``extl`` would shadow the base Module's own
    ``ctrl``/``extl`` getters, so those names can't be reused declaratively.
"""

import inspect

from .reactor import Atom as _Atom, Module as _Module, Term as _Term, Wire, X as base_X, d as base_d, Var
from .ops import op as _op
from .blocks import boundary as _boundary, block as _block, havoc as _havoc, skip as _skip, zero as _zero

from .expr import expr, cast, ite, relu, argmax, collecting, binding, Expr, X as expr_X, d as expr_d  # re-exported for authoring

# Public authoring surface: `from reactor.sugar import Module, expr, X, d, ite, cast, ...`
__all__ = ["Module", "expr", "cast", "ite", "relu", "argmax", "Expr", "X", "d", "Var"]


def _as_tuple(r) -> tuple:
    if r is None:
        return ()
    if isinstance(r, tuple):
        return r
    if isinstance(r, list):
        return tuple(r)
    return (r,)


def _id(theory, write, read) -> _Term:
    """The term copying the wire ``read`` to the wire ``write`` (of the same sort)."""
    return _Term(_op(theory, "Id", write.dtype), [write], [read])


def _build_block(cls, kind, ctrl, extl, theory):
    """The block of the ``kind`` of the atom of the module, from the method of the same
    name, or None if the class does not override it.

    The method takes the variables ``extl`` (``init``), or ``ctrl`` followed by ``extl``
    (``update``, ``delay``), and returns the values of ``ctrl`` the block gives: their next
    values, or their derivatives (``delay``)."""
    fn = getattr(cls, kind, None)
    if fn is None:
        return None
    nparams = len(inspect.signature(fn).parameters)

    inputs, outputs = _boundary(kind, ctrl=ctrl, wait=extl, read=ctrl + extl)
    params = extl if kind == "init" else ctrl + extl
    out = base_d if kind == "delay" else base_X

    # the init takes no latched values: its parameters stand for the variables, so that
    # X(v) gives their next values, on wires off the boundary of the block
    unbound = {v: Wire(v.dtype) for v in extl} if kind == "init" else {}

    with collecting() as terms, binding({**unbound, **inputs, **outputs}):
        # we assume all arguments are taken, otherwise it's too ambiguous
        args = tuple(expr(v, theory=theory) for v in params)
        if nparams != len(args) + 1:
            names = "len(extl)" if kind == "init" else "len(ctrl + extl)"
            raise ValueError(f"{kind} expects {names} == {len(args)} params, got {nparams - 1}")

        vals = _as_tuple(fn(None, *args))
        if len(vals) != len(ctrl):
            raise ValueError(f"{kind} expects {len(ctrl)} return values, got {len(vals)}")
        for var, val in zip(ctrl, vals):
            e = val if isinstance(val, Expr) else expr(val, theory=theory, sort=var.dtype)
            terms.append(_id(theory, outputs[out(var)], e.wire))

    latched = {w: v for v, w in unbound.items()}
    for term in terms:
        for w in term.read:
            if w in latched:
                raise ValueError(
                    f"{cls.__name__}.init reads the latched value of {latched[w]}, which an "
                    f"init does not have; use X() for its next value"
                )
    return _block(inputs, outputs, terms)


def X(var):
    if isinstance(var, Expr):
        return expr_X(var)
    elif isinstance(var, Var):
        return base_X(var)
    else:
        raise ValueError(f"sugar.X expects Expr or Var, got {var}")


def d(var):
    if isinstance(var, Expr):
        return expr_d(var)
    elif isinstance(var, Var):
        return base_d(var)
    else:
        raise ValueError(f"sugar.d expects Expr or Var, got {var}")


class Module(_Module):
    def __new__(cls, ctrl=(), extl=(), hide=frozenset(), theory=None):
        if theory is None:
            raise TypeError(f"{cls.__name__}: `theory` is a required constructor arg")

        ctrl = tuple(v for v in ctrl)
        extl = tuple(v for v in extl)

        read = ctrl + extl

        def build(kind, default):
            block = _build_block(cls, kind, ctrl, extl, theory)
            return default() if block is None else block

        init = build("init", lambda: _havoc(ctrl, extl))
        update = build("update", lambda: _skip(ctrl, extl, read))
        delay = build("delay", lambda: _zero(ctrl, extl, read))

        atom = _Atom(ctrl, extl, read, init, update, delay)
        self = super().__new__(cls, atom, hide=hide)
        self._theory = theory
        return self
