"""The native parser: a run's predicates read into reactor's terms, with no
solver.

A predicate is read in two dialects -- the Python predicate syntax first
(`dsl`), SMT-LIB (`smtlib`) for text that is no Python -- against the
module's own wires (`ModuleWires`): `s{i}` is `X`
of the `i`th controlled variable, `e{i}` the `i`th input awaited in the
round and `el{i}` the one latched before it. What it makes of one is a
`Formula`. `NativeParser` reads them one at a time, and `read_predicates`
a claim's and a certificate's together (`Claim`, `Certificate`), as
`proof.render` prints them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from reactor import Word, X

from ..errors import PredicateRefused, Unreadable
from ..lowering.sorts import decimal_fraction, wire_shape
from ..lowering.elements import ModuleWires
from . import dsl
from .formula import Builder, Formula, family
from .smtlib import Elaborator, Lit, Matrix, Scope, read_smtlib


# ══════════════════════════════════════════════════════════════════════
# What is claimed of a module, and what certifies it
# ══════════════════════════════════════════════════════════════════════
# Predicates as text over the state `s0..` and the inputs `e0..` (awaited)
# and `el0..` (latched): what `read_predicates` reads.


@dataclass(frozen=True)
class Claim:
    """The property a certificate proves, under which rule and assumptions.

    `kind` is the proof rule, and so what `prp` means:

      "buchi"  -- `G (F prp)`, `rule_buchi`: an invariant and a ranking
                  function that drops on every step off `prp`. `--buchi`.
      "safety" -- `G prp`, `rule_globally`: an invariant that implies
                  `prp`, and no ranking. `--safety`.

    `init_pre` / `update_pre` constrain the inputs `init` and `update` see.
    """

    prp: str | None = None
    kind: Literal["safety", "buchi"] = "buchi"
    init_pre: str | None = None
    update_pre: str | None = None


@dataclass(frozen=True)
class Certificate:
    """The predicates that prove a `Claim`. A safety one has no ranking."""

    inv: str | None = None
    ranking: str | None = None


# ══════════════════════════════════════════════════════════════════════
# The names a predicate reads the module's wires by
# ══════════════════════════════════════════════════════════════════════


def _prefixed(wires: ModuleWires) -> list[tuple[str, list]]:
    """Each name's prefix, with the wires it numbers."""
    return [("s", wires.ctrl_next), ("e", wires.extl_next),
            ("el", wires.extl_latched)]


def scope_of(wires: ModuleWires) -> Scope:
    """The names a predicate reads the module's wires by."""
    return Scope({f"{prefix}{i}": w for prefix, ws in _prefixed(wires)
                  for i, w in enumerate(ws)})


# ══════════════════════════════════════════════════════════════════════
# The Python predicate syntax, over reactor's terms
# ══════════════════════════════════════════════════════════════════════


class _Term:
    """A term while the syntax is read -- a wire, a `Lit` or a `Matrix` --
    with the operators the syntax applies to one."""

    __slots__ = ("_x", "_v")

    def __init__(self, x: Elaborator, v):
        self._x, self._v = x, v

    def _apply(self, name: str, *operands) -> "_Term":
        return _Term(self._x, self._x.apply(name, [o._v for o in operands], name))

    def _lift(self, o) -> "_Term":
        return lift(self._x, o, like=self._v)

    def _bits(self) -> bool:
        v = self._v
        return not isinstance(v, (Lit, Matrix)) and isinstance(v.dtype, Word)

    def __gt__(self, o): return self._apply(">", self, self._lift(o))
    def __lt__(self, o): return self._apply("<", self, self._lift(o))
    def __ge__(self, o): return self._apply(">=", self, self._lift(o))
    def __le__(self, o): return self._apply("<=", self, self._lift(o))
    def __eq__(self, o): return self._apply("=", self, self._lift(o))
    def __ne__(self, o): return self._apply("distinct", self, self._lift(o))
    def __add__(self, o): return self._apply("+", self, self._lift(o))
    def __radd__(self, o): return self._apply("+", self._lift(o), self)
    def __sub__(self, o): return self._apply("-", self, self._lift(o))
    def __rsub__(self, o): return self._apply("-", self._lift(o), self)
    def __mul__(self, o): return self._apply("*", self, self._lift(o))
    def __rmul__(self, o): return self._apply("*", self._lift(o), self)

    def __and__(self, o):
        return self._apply("bvand" if self._bits() else "and", self, self._lift(o))

    def __rand__(self, o):
        return self._apply("bvand" if self._bits() else "and", self._lift(o), self)

    def __or__(self, o):
        return self._apply("bvor" if self._bits() else "or", self, self._lift(o))

    def __ror__(self, o):
        return self._apply("bvor" if self._bits() else "or", self._lift(o), self)

    def __neg__(self):
        return self._apply("bvneg" if self._bits() else "-", self)

    def __invert__(self):
        return self._apply("bvnot" if self._bits() else "not", self)

    __hash__ = None

    def __getitem__(self, k):
        """`e0[k]`: element `k` of a matrix variable, row-major."""
        if not isinstance(k, int):
            raise TypeError("index must be int")
        if not isinstance(self._v, Matrix):
            raise TypeError(f"{self._v} is not a matrix; cannot index")
        return _Term(self._x, self._x.select(self._v, k))


class _Var(_Term):
    """A state or input variable, which knows its shape: `v[i][j]` (or
    `v[i, j]`) is its element, and a scalar's only one is `v[0][0]`."""

    __slots__ = ("_rows", "_cols")

    def __init__(self, x: Elaborator, v, rows: int, cols: int):
        super().__init__(x, v)
        self._rows, self._cols = rows, cols

    def _element(self, i: int, j: int) -> _Term:
        if not (0 <= i < self._rows and 0 <= j < self._cols):
            raise IndexError(f"index ({i}, {j}) out of bounds for "
                             f"{self._rows}×{self._cols} matrix")
        if not isinstance(self._v, Matrix):
            return _Term(self._x, self._v)
        return _Term(self._x, self._x.select(self._v, i * self._cols + j))

    def __getitem__(self, key):
        if isinstance(key, tuple):
            i, j = key
            return self._element(i, j)
        return dsl.Row(self._element, key)


def lift(x: Elaborator, o, like=None) -> _Term:
    """A Python value as a term: an int is a numeral -- at the width of a
    bit-vector it meets -- a float a decimal,
    and a bool a Bool."""
    if isinstance(o, _Term):
        return o
    if isinstance(o, bool):
        return _Term(x, x.b.boolean(o))
    if isinstance(o, int):
        if (like is not None and not isinstance(like, (Lit, Matrix))
                and isinstance(like.dtype, Word)):
            return _Term(x, x.b.bitvec(like.dtype.width, o))
        return _Term(x, Lit(o, decimal=False))
    if isinstance(o, float):
        return _Term(x, Lit(decimal_fraction(o), decimal=True))
    raise TypeError(f"Cannot read {type(o).__name__} as a term")


def namespace(x: Elaborator, wires: ModuleWires) -> dict:
    """What a predicate in the Python syntax can name: the variables, and
    the connectives `And`, `Or`, `Not`, `Ite`, `Implies`, `Eq`, `Ne` and
    `ToInt`."""

    def apply(name: str, *args) -> _Term:
        # A numeral among the arguments meets the first that is a term.
        first = next((t._v for t in map(lambda a: lift(x, a), args)
                      if not isinstance(t._v, Lit)), None)
        return _Term(x, x.apply(name, [lift(x, a, like=first)._v for a in args],
                                name))

    ns = {
        "And": lambda *a: apply("and", *a),
        "Or": lambda *a: apply("or", *a),
        "Not": lambda a: apply("not", a),
        "Ite": lambda c, t, e: apply("ite", c, t, e),
        "Implies": lambda a, b: apply("=>", a, b),
        "Eq": lambda a, b: apply("=", a, b),
        "Ne": lambda a, b: apply("distinct", a, b),
        "ToInt": lambda a: apply("to_int", a),
        "true": _Term(x, x.b.boolean(True)),
        "false": _Term(x, x.b.boolean(False)),
        "True": _Term(x, x.b.boolean(True)),
        "False": _Term(x, x.b.boolean(False)),
    }
    for prefix, ws in _prefixed(wires):
        for i, w in enumerate(ws):
            shape = wire_shape(w)
            ns[f"{prefix}{i}"] = _Var(x, x.scope.lookup(f"{prefix}{i}"),
                                      shape.m, shape.n)
        # A lone component can be written without its index: `s[i][j]`.
        if len(ws) == 1:
            ns[prefix] = ns[f"{prefix}0"]
    return ns


# ══════════════════════════════════════════════════════════════════════
# The parser
# ══════════════════════════════════════════════════════════════════════


def parse_predicate(wires: ModuleWires, src: str) -> Formula:
    """`src` as a `Formula`: the Python syntax first, SMT-LIB for text that
    is no Python. Text that is the syntax and wrong in it is refused in the
    syntax's terms; anything else raised is a bug, and is left to say so."""
    b = Builder()
    x = Elaborator(scope_of(wires), b)
    try:
        result = dsl.evaluate(src, namespace(x, wires))
    except (SyntaxError, dsl.NotPredicate):
        result = None
    except (TypeError, ValueError, IndexError) as e:
        raise Unreadable(
            f"`{src.strip()}` is predicate syntax, and not a predicate of "
            f"this module: {e}"
        ) from e
    if isinstance(result, _Term) and not isinstance(result._v, (Lit, Matrix)):
        return b.formula(result._v, x.scope.names)
    # Not the syntax, or a value no predicate is -- `5`, a whole matrix
    # variable: SMT-LIB reads the rest.
    return read_smtlib(x.scope, src)


class NativeParser:
    """A run's predicates read against `module` into reactor's terms, each
    once."""

    def __init__(self, module):
        self.module = module
        self.wires = ModuleWires.of(module)
        self._parsed: dict = {}

    def parse(self, src: "str | None", role: str, want: str = "Bool",
              kind: str = "buchi") -> "Formula | None":
        """`src` as a term over the state and inputs, or `PredicateRefused`
        naming its `role`: `want` is the sort the predicate's field is
        declared as."""
        if src is None:
            return None
        hit = self._parsed.get(src)
        if hit is None:
            try:
                hit = parse_predicate(self.wires, src)
            except Unreadable as e:
                hit = Unreadable(str(e))
            self._parsed[src] = hit
        if isinstance(hit, Unreadable):
            raise PredicateRefused(
                role, f"cannot read `{src}`: {hit}", kind=kind) from hit
        got = family(hit.sort)
        if got != want:
            raise PredicateRefused(role, f"must have sort {want}, got {got}",
                                   kind=kind)
        return hit


@dataclass(frozen=True)
class NativePredicates:
    """A claim and a certificate read by the native parser: what
    `proof.render.render_certificate` prints. A predicate left out is
    `None`."""

    kind: str
    wires: ModuleWires
    prp: "Formula | None" = None
    inv: "Formula | None" = None
    ranking: "Formula | None" = None
    init_pre: "Formula | None" = None
    update_pre: "Formula | None" = None


def read_predicates(module, claim: Claim,
                    cert: "Certificate | None" = None) -> NativePredicates:
    """`claim` and `cert` read against `module`, each predicate refused by
    its role."""
    p = NativeParser(module)
    cert = cert or Certificate()
    return NativePredicates(
        kind=claim.kind,
        wires=p.wires,
        prp=p.parse(claim.prp, "property", kind=claim.kind),
        inv=p.parse(cert.inv, "invariant"),
        ranking=p.parse(cert.ranking, "ranking", want="Int"),
        init_pre=p.parse(claim.init_pre, "pre"),
        update_pre=p.parse(claim.update_pre, "pre"),
    )
