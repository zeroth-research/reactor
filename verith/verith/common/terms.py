"""Reading and taking apart cvc5 terms, as more than one route does.

`free_constants` is what a script declares; `ite_conditions`, the branch
conditions under a term; `literals`, every numeral under one;
`select_element` takes one element of a tuple-sorted term without leaving
a select of an `ite` behind. The walk they are built on is
`reactor_lean.smt.terms.nodes`.

`Kind` is imported where it is used, so importing this loads no cvc5.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from reactor_lean.smt.terms import nodes


def free_constants(roots: Iterable, *, order: str = "pre") -> dict:
    """Every free constant under `roots`, by symbol, first occurrence kept.

    cvc5 has no "the declarations this term needs", so the term is walked. A
    constant is `Kind.CONSTANT`; an operator, a bound variable and a literal
    are none of them declared.
    """
    from cvc5 import Kind

    out: dict = {}
    for node in nodes(roots, order=order):
        if node.getKind() == Kind.CONSTANT:
            out.setdefault(node.getSymbol(), node)
    return out


def ite_conditions(roots: Iterable, *, order: str = "pre",
                   key: Callable = str,
                   where: "Callable | None" = None) -> list:
    """The condition of every `ite` under `roots`, one per `key`, in `order`.

    `key` is what makes two conditions one: the printed form by default, so
    a transition testing `s0 = 0` in three components is one condition.
    `where`, given an `ite`, says whether its condition is wanted.
    """
    from cvc5 import Kind

    out: dict = {}
    for node in nodes(roots, order=order):
        if node.getKind() == Kind.ITE and (where is None or where(node)):
            cond = node[0]
            out.setdefault(key(cond), cond)
    return list(out.values())


def literals(roots: Iterable) -> list:
    """Every Int and Real literal node under `roots`."""
    from cvc5 import Kind

    kinds = (Kind.CONST_INTEGER, Kind.CONST_RATIONAL)
    return [node for node in nodes(roots) if node.getKind() in kinds]


def select_element(tm, t, k: int, leaf: Callable = lambda s: s):
    """Element `k` of a tuple-sorted `t`, pushed down to where the tuple is
    built.

    A select of a constructor is its `k`-th argument, and a select of an
    `ite` between tuples is the `ite` of the two selects: the tuple axiom
    in each case. cvc5's simplifier folds the first and not the second, and
    the second is the shape of any round that branches. Anything else gets
    the selector, through `leaf`.
    """
    from cvc5 import Kind

    from reactor_lean.smt.encode import tuple_select

    kind = t.getKind()
    if kind == Kind.APPLY_CONSTRUCTOR:
        return t[k + 1]                 # child 0 is the constructor
    if kind == Kind.ITE:
        return tm.mkTerm(Kind.ITE, t[0], select_element(tm, t[1], k, leaf),
                         select_element(tm, t[2], k, leaf))
    return leaf(tuple_select(tm, t, k))
