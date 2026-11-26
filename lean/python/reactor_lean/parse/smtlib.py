"""SMT-LIB predicates read into reactor's terms.

`read_smtlib(scope, src)` reads one SMT-LIB term over the module's state
and inputs -- `s0..`, `e0..` and `el0..`, as a `Scope` binds them -- into a
`Formula`. It reads the fragment a certificate is written in: the core
theory, integer and real arithmetic, `to_real`,
`to_int` and `ubv_to_int`, bit-vector literals, and
`((_ tuple.select k) v)` on a matrix variable. Of these:

* a chain `(< a b c)` or `(= a b c)` is the conjunction of its links;
* `-` and `=>` nest, to the left and to the right; `and`, `or`, `+` and `*`
  take any number of arguments, and are built as a chain the printer
  prints flat again;
* an Int written where a Real is wanted is a Real: a numeral is read as
  one, and anything else is cast (`cast.ToReal`);
* `(= v 5)` and `(distinct 5 v)`, with `v` a bit-vector variable, read the
  numeral at `v`'s width, which plain SMT-LIB does not.

`(distinct a b c)` is every pair distinct. What this parser does not read
-- `/`, `abs`, `xor`, the bit-vector operations, which a predicate has no
Lean form for (`proof.from_ir`) -- is refused, by name; so is `let`, which
a claim is refused before it gets here (`parse.names`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

from reactor import Bool, Int, Real, Word, tensor

from ..errors import Unreadable
from ..lowering.sorts import wire_shape
from .formula import SCALAR, Builder, Formula, element, family
from .sexpr import Atom, read, show


def _theory(sort) -> "tuple[object, tuple[int, ...]]":
    """The namespace of `reactor.tensor` whose ops are on `sort`'s
    elements, and what those ops take ahead of their shape: a word's
    width."""
    match sort:
        case Word(w, _):
            return tensor.word, (w,)
        case Real(_):
            return tensor.real, ()
        case Bool(_):
            return tensor.bool, ()
    return tensor.int, ()


@dataclass(frozen=True)
class Lit:
    """A numeral or a decimal not yet given a sort: what it becomes depends
    on what it meets. `neg` is `(- n)`, kept as a negation of
    the literal; `bare` is a lone token, which is what a bit-vector
    variable's width reaches."""

    value: "int | Fraction"
    decimal: bool
    neg: bool = False
    bare: bool = False


@dataclass(frozen=True)
class Matrix:
    """A matrix variable: SMT-LIB's tuple, which only `tuple.select` reads."""

    wire: object
    rows: int
    cols: int


@dataclass(frozen=True)
class Scope:
    """What a predicate may name: `s{i}`, `e{i}` and `el{i}`, each a module
    wire."""

    names: dict          # name -> Wire

    def lookup(self, name: str):
        """The wire `name` names, or its `Matrix`; `None` for no name."""
        wire = self.names.get(name)
        if wire is None:
            return None
        shape = wire_shape(wire)
        return wire if shape.is_scalar else Matrix(wire, shape.m, shape.n)

    def holds(self, wire) -> bool:
        """Whether `wire` is one of the variables, not a term over them."""
        return any(wire == w for w in self.names.values())


# `-1` and `-0.5` are read as numbers, though SMT-LIB has them as symbols.
_NEG_NUMBER = re.compile(r"-(0|[1-9][0-9]*)(\.[0-9]+)?$")
_BV_LITERAL = re.compile(r"bv([0-9]+)$")
_CHAINS = {"<": "Lt", "<=": "Le", ">": "Gt", ">=": "Ge"}
_ARITH = {"Int": tensor.int, "Real": tensor.real}


def _scalar(cls):
    """A maker of `cls` on 1×1 matrices, for the folds."""
    return lambda: cls(SCALAR)


def read_smtlib(scope: Scope, src: str) -> Formula:
    """`src`, one SMT-LIB term over `scope`, as a `Formula`; `Unreadable`
    naming what is wrong with it."""
    b = Builder()
    x = Elaborator(scope, b)
    return b.formula(x.settle(x.term(read(src)), None), scope.names)


class Elaborator:
    """SMT-LIB's terms, and the Python syntax's operators (`parse.predicates`),
    built as reactor's terms: each method takes values -- wires, `Lit`s,
    `Matrix`es -- and gives the wire of what it builds."""

    def __init__(self, scope: Scope, b: Builder):
        self.scope, self.b = scope, b

    # --- what an expression is -----------------------------------------------

    def term(self, e):
        """`e` as a wire, a `Lit` or a `Matrix`."""
        if isinstance(e, Atom):
            return self.atom(e)
        if not e:
            raise Unreadable("`()` is no term")
        head, args = e[0], e[1:]
        if isinstance(head, tuple):
            return self.indexed_application(head, args)
        if head.kind != "symbol":
            raise Unreadable(f"`{show(e)}` applies `{head}`, which is no function")
        name = head.text
        if name == "let":
            raise Unreadable(f"`{show(e)}`: `let` is not read")
        if name == "_":
            return self.indexed_constant(e, args)
        if name == "!":
            raise Unreadable(f"`{show(e)}`: an annotation is not read")
        vals = [self.term(a) for a in args]
        if name == "-" and len(vals) == 1 and isinstance(vals[0], Lit) and not vals[0].neg:
            # `(- 5)` takes the sort it meets, as `5` does.
            return Lit(vals[0].value, vals[0].decimal, neg=True)
        return self.apply(name, vals, show(e))

    def atom(self, a: Atom):
        if a.kind == "numeral":
            return Lit(int(a.text), decimal=False, bare=True)
        if a.kind == "decimal":
            return Lit(Fraction(a.text), decimal=True, bare=True)
        if a.kind == "binary":
            return self.b.bitvec(len(a.text) - 2, int(a.text[2:], 2))
        if a.kind == "hex":
            return self.b.bitvec(4 * (len(a.text) - 2), int(a.text[2:], 16))
        if a.kind == "keyword":
            raise Unreadable(f"`{a}` is a keyword, not a term")
        if a.text in ("true", "false"):
            return self.b.boolean(a.text == "true")
        hit = self.scope.lookup(a.text)
        if hit is None and _NEG_NUMBER.match(a.text):
            decimal = "." in a.text
            return Lit(Fraction(a.text) if decimal else int(a.text), decimal, bare=True)
        if hit is None:
            raise Unreadable(f"`{a}` is not declared: a predicate names "
                             f"{', '.join(sorted(self.scope.names)) or 'nothing'}")
        return hit

    def indexed_constant(self, e, args):
        """`(_ bvN W)`."""
        if (len(args) == 2 and isinstance(args[0], Atom)
                and _BV_LITERAL.match(args[0].text)
                and isinstance(args[1], Atom) and args[1].kind == "numeral"):
            return self.b.bitvec(int(args[1].text),
                                 int(_BV_LITERAL.match(args[0].text)[1]))
        raise Unreadable(f"`{show(e)}` is no indexed constant this parser reads")

    def indexed_application(self, head, args):
        """`((_ tuple.select k) v)`, on a matrix variable `v`."""
        if (len(head) == 3 and isinstance(head[0], Atom) and head[0].text == "_"
                and isinstance(head[1], Atom) and head[1].text == "tuple.select"
                and isinstance(head[2], Atom) and head[2].kind == "numeral"
                and len(args) == 1):
            target = self.term(args[0])
            if not isinstance(target, Matrix):
                raise Unreadable(f"`{show((head, *args))}`: `{show(args[0])}` "
                                 f"is no matrix variable to select from")
            return self.select(target, int(head[2].text))
        raise Unreadable(f"`{show(head)}` is no indexed function this parser reads")

    def select(self, m: Matrix, k: int):
        """Element `k` of `m`, row-major: a `Get`."""
        if k >= m.rows * m.cols:
            raise Unreadable(f"a {m.rows}×{m.cols} matrix has no element {k}")
        row, col = divmod(k, m.cols)
        elem = element(m.wire.dtype)
        theory, width = _theory(elem)
        op = theory.Get(row, col, *width, [m.rows, m.cols])
        return self.b.apply(op, [m.wire], elem)

    # --- sorts -----------------------------------------------------------------

    def settle(self, v, want: "str | None"):
        """`v` as a wire: a literal as a `want` -- an Int, or a Real for a
        decimal, when nothing wants anything -- and a `Matrix` refused."""
        if isinstance(v, Matrix):
            raise Unreadable("a matrix variable is read one element at a time, "
                             "with `((_ tuple.select k) v)`")
        if not isinstance(v, Lit):
            return v
        if want == "Real" or (want is None and v.decimal):
            w = self.b.real(Fraction(v.value))
            fam = "Real"
        elif v.decimal:
            raise Unreadable(f"`{v.value}` is a decimal where {want} is wanted")
        elif want in (None, "Int"):
            w = self.b.integer(int(v.value))
            fam = "Int"
        else:
            raise Unreadable(f"the numeral `{v.value}` where {want} is wanted")
        return self.b.apply(_ARITH[fam].Neg(SCALAR), [w], w.dtype) if v.neg else w

    def numeric(self, vals: list, what: str) -> tuple:
        """`vals` as wires of one arithmetic sort, Real if any is and each
        Int cast up: the sort's name, and the wires."""
        fams = {family(v.dtype) for v in vals if not isinstance(v, (Lit, Matrix))}
        if any(isinstance(v, Lit) and v.decimal for v in vals):
            fams.add("Real")
        if not fams <= {"Int", "Real"}:
            raise Unreadable(f"`{what}` is arithmetic, and was given "
                             f"{', '.join(sorted(fams - {'Int', 'Real'}))}")
        fam = "Real" if "Real" in fams else "Int"
        out = []
        for v in vals:
            w = self.settle(v, fam)
            if fam == "Real" and family(w.dtype) == "Int":
                w = self.b.apply(tensor.cast.ToReal(SCALAR), [w], Real(SCALAR))
            out.append(w)
        return fam, out

    def boolean(self, v, what: str):
        w = self.settle(v, "Bool")
        if family(w.dtype) != "Bool":
            raise Unreadable(f"`{what}` wants a Bool, and was given "
                             f"{family(w.dtype)}")
        return w

    def bitvecs(self, vals: list, what: str) -> list:
        ws = [self.settle(v, None) for v in vals]
        fams = {family(w.dtype) for w in ws}
        if len(fams) != 1 or not isinstance(ws[0].dtype, Word):
            raise Unreadable(f"`{what}` wants bit-vectors of one width, and was "
                             f"given {', '.join(sorted(fams))}")
        return ws

    # --- applications ----------------------------------------------------------

    def apply(self, name: str, vals: list, what: str):
        """`(name vals...)`; `what` is the application as written."""
        arity = len(vals)

        def need(n: int, at_least: bool = False) -> None:
            if (arity < n) if at_least else (arity != n):
                raise Unreadable(f"`{what}`: `{name}` takes "
                                 f"{'at least ' if at_least else ''}{n} "
                                 f"argument{'s' if n > 1 else ''}")

        if name == "not":
            need(1)
            return self.b.apply(tensor.bool.Not(SCALAR), [self.boolean(vals[0], what)],
                                Bool(SCALAR))
        if name in ("and", "or"):
            need(1, at_least=True)
            op = _scalar(tensor.bool.And if name == "and" else tensor.bool.Or)
            return self.fold_right(op, [self.boolean(v, what) for v in vals], Bool(SCALAR))
        if name == "=>":
            need(2, at_least=True)
            return self.fold_right(_scalar(tensor.bool.Implies),
                                   [self.boolean(v, what) for v in vals], Bool(SCALAR))
        if name == "ite":
            need(3)
            return self.ite(vals, what)
        if name == "=":
            need(2, at_least=True)
            return self.conjoin([self.equal(a, b, what, ne=False)
                                 for a, b in zip(vals, vals[1:])])
        if name == "distinct":
            need(2, at_least=True)
            return self.conjoin([self.equal(a, b, what, ne=True)
                                 for i, a in enumerate(vals) for b in vals[i + 1:]])
        if name in _CHAINS:
            need(2, at_least=True)
            links = []
            for a, b in zip(vals, vals[1:]):
                fam, ws = self.numeric([a, b], what)
                op = getattr(_ARITH[fam], _CHAINS[name])(SCALAR)
                links.append(self.b.apply(op, ws, Bool(SCALAR)))
            return self.conjoin(links)
        if name in ("+", "*"):
            need(2, at_least=True)
            fam, ws = self.numeric(vals, what)
            op = _scalar(_ARITH[fam].Add if name == "+" else _ARITH[fam].Mul)
            return self.fold_left(op, ws, ws[0].dtype)
        if name == "-":
            need(1, at_least=True)
            fam, ws = self.numeric(vals, what)
            if arity == 1:
                return self.b.apply(_ARITH[fam].Neg(SCALAR), ws, ws[0].dtype)
            return self.fold_left(_scalar(_ARITH[fam].Sub), ws, ws[0].dtype)
        if name in ("div", "mod"):
            need(2, at_least=name == "div")
            fam, ws = self.numeric(vals, what)
            if fam != "Int":
                raise Unreadable(f"`{what}`: `{name}` is integer division")
            if name == "mod":
                return self.b.apply(tensor.int.Mod(SCALAR), ws, Int(SCALAR))
            return self.fold_left(_scalar(tensor.int.Div), ws, Int(SCALAR))
        if name == "to_real":
            need(1)
            w = self.settle(vals[0], "Int")
            if family(w.dtype) != "Int":
                raise Unreadable(f"`{what}`: `to_real` reads an Int")
            return self.b.apply(tensor.cast.ToReal(SCALAR), [w], Real(SCALAR))
        if name == "to_int":
            need(1)
            w = self.settle(vals[0], "Real")
            if family(w.dtype) != "Real":
                raise Unreadable(f"`{what}`: `to_int` reads a Real")
            return self.b.apply(tensor.cast.ToInt(SCALAR), [w], Int(SCALAR))
        if name in ("ubv_to_int", "bv2nat"):
            need(1)
            (w,) = self.bitvecs(vals, what)
            return self.b.apply(tensor.cast.UbvToInt(w.dtype.width, SCALAR), [w], Int(SCALAR))
        raise Unreadable(f"`{what}`: `{name}` is no function this parser reads")

    def equal(self, a, b, what: str, *, ne: bool):
        """`a = b`, or `a ≠ b`: a Bool wire."""
        a, b = self.at_bitvec_width(a, b)
        sorts = {family(v.dtype) for v in (a, b) if not isinstance(v, (Lit, Matrix))}
        if sorts <= {"Int", "Real"}:
            fam, ws = self.numeric([a, b], what)
            op = getattr(_ARITH[fam], "Ne" if ne else "Eq")(SCALAR)
            return self.b.apply(op, ws, Bool(SCALAR))
        wa, wb = (self.settle(v, None) for v in (a, b))
        if family(wa.dtype) != family(wb.dtype):
            raise Unreadable(f"`{what}` compares {family(wa.dtype)} with "
                             f"{family(wb.dtype)}")
        if isinstance(wa.dtype, Word):
            op = (tensor.word.Ne if ne else tensor.word.Eq)(wa.dtype.width, SCALAR)
            return self.b.apply(op, [wa, wb], Bool(SCALAR))
        op = (tensor.bool.Ne if ne else tensor.bool.Eq)(SCALAR)
        return self.b.apply(op, [wa, wb], Bool(SCALAR))

    def at_bitvec_width(self, a, b):
        """A bare numeral against a bit-vector variable, at its width."""
        for x, y in ((a, b), (b, a)):
            if (isinstance(y, Lit) and y.bare and not y.decimal
                    and not isinstance(x, (Lit, Matrix))
                    and isinstance(x.dtype, Word) and self.scope.holds(x)):
                lit = self.b.bitvec(x.dtype.width, int(y.value))
                return (x, lit) if x is a else (lit, x)
        return a, b

    def ite(self, vals: list, what: str):
        """`(ite c t e)`, with `t` and `e` of one sort: unlike arithmetic,
        neither branch is cast, a numeral included."""
        cond = self.boolean(vals[0], what)
        wt, we = (self.settle(v, None) for v in vals[1:])
        if family(wt.dtype) != family(we.dtype):
            raise Unreadable(f"`{what}`: the branches are {family(wt.dtype)} "
                             f"and {family(we.dtype)}")
        theory, width = _theory(wt.dtype)
        op = theory.Ite(*width, SCALAR)
        return self.b.apply(op, [cond, wt, we], wt.dtype)

    # --- chains ------------------------------------------------------------------

    def fold_left(self, op, ws: list, sort):
        acc = ws[0]
        for w in ws[1:]:
            acc = self.b.apply(op(), [acc, w], sort)
        return acc

    def fold_right(self, op, ws: list, sort):
        acc = ws[-1]
        for w in reversed(ws[:-1]):
            acc = self.b.apply(op(), [w, acc], sort)
        return acc

    def conjoin(self, links: list):
        return self.fold_right(_scalar(tensor.bool.And), links, Bool(SCALAR))
