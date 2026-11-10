"""Reading and taking apart cvc5 terms, as more than one route does.

`literals` is every numeral under a term; `select_element` takes one
element of a tuple-sorted term without leaving a select of an `ite`
behind. The walks they are built on are `reactor.lean.smt.terms`.

`Kind` is imported where it is used, so importing this loads no cvc5.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from reactor.lean.smt.terms import nodes


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

    from reactor.lean.smt.encode import tuple_select

    kind = t.getKind()
    if kind == Kind.APPLY_CONSTRUCTOR:
        return t[k + 1]                 # child 0 is the constructor
    if kind == Kind.ITE:
        return tm.mkTerm(Kind.ITE, t[0], select_element(tm, t[1], k, leaf),
                         select_element(tm, t[2], k, leaf))
    return leaf(tuple_select(tm, t, k))
