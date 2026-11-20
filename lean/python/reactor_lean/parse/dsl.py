"""The Python predicate syntax, read without `eval` and without a solver.

`s0 > 0 and s1 == s4`, `Ite(e0 < 3, s0, -s0)`, `s0[1][2] >= 0`: a
predicate written as a Python expression over the module's state and
inputs. `walk` reads the expression's tree over a whitelist and builds the
predicate with the term algebra its namespace holds: reactor's terms
(`namespace`).

The source can be text nobody vetted, and no namespace makes `eval` safe
on it: `().__class__.__mro__` reaches every class in the process. So the
tree is walked over a whitelist -- numeric and Bool constants, names bound
in the namespace, the operators a term overloads, calls to the namespace's
own connectives, and `v[i][j]` / `v[i, j]` with int indices -- and anything
else raises `NotPredicate`. Attribute access and `_`-names never resolve.

`and`, `or`, `not`, `x if c else y` and a chain `a < b < c` mean `And`,
`Or`, `Not`, `Ite` and a conjunction: Python's own reading would ask
whether a symbolic term is truthy, which it always is.
"""

from __future__ import annotations

import ast
import operator


class NotPredicate(ValueError):
    """Python that the predicate syntax does not read."""


# The operators a term overloads. No `**`: on two constants it is evaluated
# here, and its size is not bounded by the length of the source.
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
}
_UNARYOPS = {ast.USub: operator.neg, ast.Invert: operator.invert}
_CMPOPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


def evaluate(src: str, ns: dict):
    """`src` as a predicate expression over `ns`, read without `eval`;
    `SyntaxError` for text that is no Python, `NotPredicate` for Python
    the syntax does not read."""
    # `ast.parse` reads a leading blank as an indent.
    return walk(ast.parse(src.lstrip(" \t"), mode="eval").body, ns)


def walk(node: ast.AST, ns: dict):
    def recur(n):
        return walk(n, ns)

    if isinstance(node, ast.Constant) and type(node.value) in (int, float, bool):
        return node.value
    if isinstance(node, ast.Name) and not node.id.startswith("_") and node.id in ns:
        return ns[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](recur(node.left), recur(node.right))
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return ns["Not"](recur(node.operand))
        if type(node.op) in _UNARYOPS:
            return _UNARYOPS[type(node.op)](recur(node.operand))
    if isinstance(node, ast.BoolOp):
        conn = ns["And"] if isinstance(node.op, ast.And) else ns["Or"]
        return conn(*[recur(v) for v in node.values])
    if isinstance(node, ast.IfExp):
        return ns["Ite"](recur(node.test), recur(node.body), recur(node.orelse))
    if isinstance(node, ast.Compare) and all(type(op) in _CMPOPS for op in node.ops):
        operands = [recur(node.left)] + [recur(c) for c in node.comparators]
        links = [
            _CMPOPS[type(op)](a, b)
            for op, a, b in zip(node.ops, operands, operands[1:])
        ]
        return links[0] if len(links) == 1 else ns["And"](*links)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and not node.func.id.startswith("_")
        and callable(ns.get(node.func.id))
        and not node.keywords
        and not any(isinstance(a, ast.Starred) for a in node.args)
    ):
        return ns[node.func.id](*[recur(a) for a in node.args])
    if isinstance(node, ast.Subscript):
        keys = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        index = [recur(k) for k in keys]
        if all(type(i) is int for i in index):
            target = recur(node.value)
            return target[tuple(index)] if len(index) > 1 else target[index[0]]
    raise NotPredicate(f"`{ast.unparse(node)}` is not predicate syntax")
