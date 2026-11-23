"""Reactor's terms (a `parse.formula.Formula`) -> a Lean 4 expression.

The printer of what the native parser reads (`parse.native`). It states what
`from_smt` states of the same predicate read by cvc5, in the same words: one
parameter `s`, a left-nested tuple whose components are reached by `.1`,
`.2.1`, ... with a scalar taken out of its 1×1 `Matrix` by `0 0`; an `ite`
that is a `min` or a `max` printed as one; an arithmetic subterm read more
than once `let`-bound, when that makes the text shorter; and a refusal for
what has no Lean form, as `Refused`.

Two things it spells differently, each the same Lean term:

* `and`, `or`, `+` and `*` arrive as chains of two-argument terms, and are
  printed flat however they were written: `(and a (and b c))` is
  `(a ∧ b ∧ c)` here and `(a ∧ (b ∧ c))` from cvc5;
* an Int numeral where a Real is wanted was read as a Real, and is
  `(2 : Real)` here and `(2 : Int)` from cvc5, which Lean casts.
"""

from __future__ import annotations

from reactor import Wire

from ..codegen.literals import lean_bitvec, lean_int, lean_real
from ..core.errors import Refused
from ..lowering.ops import theory_of
from ..lowering.sorts import itype_name, wire_shape
from ..parse.formula import family
from .bindings import Bindings, build_var_map, with_lets

_INFIX = {
    "Lt": " < ", "Le": " ≤ ", "Gt": " > ", "Ge": " ≥ ", "Eq": " = ", "Ne": " ≠ ",
    "Sub": " - ", "Div": " / ", "Mod": " % ", "Implies": " → ",
}
# The chains the parser builds of one n-ary application, and the side they
# grow on: `a ∧ (b ∧ c)`, `(a + b) + c`.
_RIGHT = {"And": " ∧ ", "Or": " ∨ "}
_LEFT = {"Add": " + ", "Mul": " * "}
# `ite (a ≥ b) a b` is `max a b`, and so on; the flag says whether the
# branches are in the comparison's order (`from_smt._fold_min_max`).
_MIN_MAX = {("Ge", True): "max", ("Gt", True): "max",
            ("Le", True): "min", ("Lt", True): "min",
            ("Ge", False): "min", ("Gt", False): "min",
            ("Le", False): "max", ("Lt", False): "max"}


def ir_to_lean(formula, state_wires: list[Wire], *, param_name: str = "s",
               extra: "Bindings | None" = None, share: bool = True) -> str:
    """`formula` as `fun {param_name} => ...`, as `from_smt.smt_to_lean`
    prints a cvc5 term. `extra` binds more variables -- `e0..`, `el0..` --
    to accessors on the parameter."""
    p = _Printer(formula, [("s", param_name, state_wires)] + (extra or []))
    body, lets = p.parts(share)
    return f"fun {param_name} => {with_lets(body, lets)}"


def ir_to_lean_nat(formula, state_wires: list[Wire], *, param_name: str = "s",
                   extra: "Bindings | None" = None, share: bool = True) -> str:
    """An Int-sorted `formula` as a Lean `Nat`, clamped by `Int.toNat`."""
    p = _Printer(formula, [("s", param_name, state_wires)] + (extra or []))
    body, lets = p.parts(share)
    return f"fun {param_name} => {with_lets(f'(({body} : Int)).toNat', lets)}"


class _Printer:
    def __init__(self, formula, bindings):
        self.formula = formula
        self.by_wire = {t.write[0].id: t for t in formula.terms}
        # wire id -> (accessor, wire, name)
        accessor, wire = build_var_map(bindings)
        self.vars: dict[int, tuple[str, Wire, str]] = {
            wire[name].id: (accessor[name], wire[name], name) for name in accessor}
        self.bound: dict[int, str] = {}
        self.shared: set[int] = set()

    # --- sharing -------------------------------------------------------------

    def parts(self, share: bool) -> tuple[str, list]:
        """`(body, lets)`, the `lets` in dependency order."""
        plain = self.walk(self.formula.out)
        if not share:
            return plain, []
        self.shared = self.shared_wires()
        if not self.shared:
            return plain, []
        lets: list = []

        def emit(w: Wire) -> str:
            if w.id in self.bound:
                return self.bound[w.id]
            text = self.walk(w, emit)
            if w.id in self.shared:
                name = f"u{len(lets)}"
                lets.append((name, text))
                self.bound[w.id] = name
                return name
            return text

        body = self.walk(self.formula.out, emit)
        # A binding is only worth it when the output is shorter for it.
        if len(with_lets(body, lets)) >= len(plain):
            return plain, []
        return body, lets

    def shared_wires(self) -> set[int]:
        """The wires two or more reads reach that are compound,
        state-dependent and not Bool: `from_smt._shared_nodes`'s test."""
        edges: dict[int, int] = {}
        stack = [self.formula.out]
        while stack:
            w = stack.pop()
            edges[w.id] = edges.get(w.id, 0) + 1
            if edges[w.id] > 1:
                continue
            t = self.by_wire.get(w.id)
            if t is not None:
                stack.extend(t.read)
        return {k for k, n in edges.items() if n > 1 and self.worth_binding(k)}

    def worth_binding(self, wid: int) -> bool:
        t = self.by_wire.get(wid)
        if t is None or not list(t.read) or family(t.write[0].dtype) == "Bool":
            return False
        seen, stack = set(), [wid]
        while stack:
            k = stack.pop()
            if k in seen:
                continue
            seen.add(k)
            if k in self.vars or k in self.formula.names:
                return True
            if (inner := self.by_wire.get(k)) is not None:
                stack.extend(r.id for r in inner.read)
        return False

    # --- one term --------------------------------------------------------------

    def walk(self, w: Wire, emit=None) -> str:
        """`w` as Lean; `emit`, when given, prints each operand instead of
        recursing directly, which is how `parts` puts its `let` names in."""
        var = self.vars.get(w.id)
        if var is not None:
            if not wire_shape(var[1]).is_scalar:
                raise Refused(f"`{var[2]}` is a matrix, read one element at a time")
            return var[0]
        t = self.by_wire.get(w.id)
        if t is None:
            name = self.formula.names.get(w.id, f"wire #{w.id}")
            known = [v[2] for v in self.vars.values()]
            raise Refused(f"Unknown free variable `{name}` (known: {known})")
        recur = emit or self.walk
        name, theory = itype_name(t.itype), theory_of(t.itype)
        args = list(t.read)

        if not args:
            return self.constant(t, name)
        if name == "Get":
            acc = self.vars.get(args[0].id)
            if acc is None:
                raise Refused(f"`Get` against a term that is no variable: {t}")
            return f"({acc[0]} {t.itype.row} {t.itype.col})"
        if theory == "Cast":
            return self.cast(t, name, args, recur)
        if theory == "BV" and name != "Ite":
            raise Refused(f"no Lean form for the bit-vector `{name}` in a predicate")
        if name == "Not":
            return f"¬ ({recur(args[0])})"
        if name in _RIGHT:
            return "(" + _RIGHT[name].join(map(recur, self.chain(t, right=True))) + ")"
        if name in _LEFT:
            return "(" + _LEFT[name].join(map(recur, self.chain(t, right=False))) + ")"
        if name in _INFIX:
            return f"({recur(args[0])}{_INFIX[name]}{recur(args[1])})"
        if name == "Neg":
            return f"(- {recur(args[0])})"
        if name == "Ite":
            return self.ite(t, theory, args, recur)
        raise Refused(f"no Lean form for {theory}.{name} in a predicate")

    def constant(self, t, name: str) -> str:
        value = t.itype._0.item()
        if name == "Bool":
            return "true" if value else "false"
        if name == "Int":
            return lean_int(value)
        if name == "Real":
            return lean_real(value)
        if name == "Const":
            return lean_bitvec(t.write[0].dtype._0, value)
        raise Refused(f"no Lean literal for {name}")

    def cast(self, t, name: str, args: list, recur) -> str:
        if name == "RealToInt":
            return f"⌊{recur(args[0])}⌋"
        if name == "BVToInt":
            return f"(Int.ofNat (BitVec.toNat {recur(args[0])}))"
        if name == "IntToReal":
            # An Int literal becomes a Real one; anything else is cast.
            inner = self.by_wire.get(args[0].id)
            if inner is not None and itype_name(inner.itype) == "Int":
                return lean_real(inner.itype._0.item())
            return f"((({recur(args[0])}) : Int) : Real)"
        if name == "BitToBool":
            # What a predicate compares bit-vectors with: `=` and `≠`.
            inner = self.by_wire.get(args[0].id)
            op = itype_name(inner.itype) if inner is not None else None
            if op in ("Eq", "Ne"):
                a, b = inner.read
                return f"({recur(a)}{_INFIX[op]}{recur(b)})"
            raise Refused(f"no Lean form for the bit-vector `{op}` in a predicate")
        raise Refused(f"no Lean form for Cast.{name} in a predicate")

    def ite(self, t, theory: str, args: list, recur) -> str:
        cond = args[0]
        if theory == "BV":
            # Its condition is a bit, cast from the Bool the predicate wrote.
            inner = self.by_wire.get(cond.id)
            if inner is None or itype_name(inner.itype) != "BoolToBit":
                raise Refused(f"a bit-vector `ite` on a bit no Bool was cast to: {t}")
            cond = inner.read[0]
        folded = self.min_max(t, theory, recur)
        if folded is not None:
            return folded
        return f"(if {recur(cond)} then {recur(args[1])} else {recur(args[2])})"

    def chain(self, t, *, right: bool) -> list[Wire]:
        """The operands of the run of `t`'s op that `t` heads: the parser's
        chain of one n-ary application. A link that is `let`-bound ends it."""
        name = itype_name(t.itype)
        out: list[Wire] = []
        while True:
            a, b = t.read
            nxt = b if right else a
            inner = self.by_wire.get(nxt.id)
            linked = (inner is not None and itype_name(inner.itype) == name
                      and nxt.id not in self.shared)
            if right:
                out.append(a)
            else:
                out.insert(0, b)
            if not linked:
                if right:
                    out.append(b)
                else:
                    out.insert(0, a)
                return out
            t = inner

    def min_max(self, t, theory: str, recur) -> "str | None":
        """An Int `ite` that is a `min` or a `max`, as one (`from_smt`'s
        `_fold_min_max`: `omega` reads them, with no case split)."""
        if theory != "LIA" or family(t.write[0].dtype) != "Int":
            return None
        cond = self.by_wire.get(t.read[0].id)
        if cond is None or itype_name(cond.itype) not in ("Ge", "Gt", "Le", "Lt"):
            return None
        lhs, rhs = cond.read
        then_, else_ = t.read[1], t.read[2]
        if lhs == then_ and rhs == else_:
            op = _MIN_MAX[(itype_name(cond.itype), True)]
        elif lhs == else_ and rhs == then_:
            op = _MIN_MAX[(itype_name(cond.itype), False)]
        else:
            return None
        return f"({op} {recur(lhs)} {recur(rhs)})"
