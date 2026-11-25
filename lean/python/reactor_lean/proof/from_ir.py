"""Reactor's terms (a `parse.formula.Formula`) -> a Lean 4 expression.

The printer of the predicates `parse.predicates` reads: one
parameter, the state `s` or the inputs `e`, a tuple whose components are
reached by `.1`, `.2.1`, ... with a scalar taken out of its 1×1 `Matrix`
by `0 0`; a cast as the op table's scalar column writes it; an `ite`
that is a `min` or a `max` printed as one; an arithmetic subterm read more
than once `let`-bound, when that makes the text shorter; and a refusal for
what has no Lean form, as `Refused`.

`and`, `or`, `+` and `*` arrive as chains of two-argument terms, and are
printed flat however they were written: `(and a (and b c))` is
`(a ∧ b ∧ c)`. An Int numeral where a Real is wanted was read as a Real,
and is `(2 : Real)`.
"""

from __future__ import annotations

from reactor import Wire

from ..codegen.literals import lean_literal, lean_real
from ..errors import Refused
from ..lowering.elements import product_accessor
from ..lowering.ops import scalar_emitter
from ..lowering.sorts import itype_name, theory_of, wire_shape
from ..parse.formula import family

# (prefix, base, wires): wire `i` is the variable `<prefix><i>`, read on
# the parameter `base`.
Bindings = list[tuple[str, str, list[Wire]]]


def build_var_map(bindings: Bindings) -> tuple[dict[str, str], dict[str, Wire]]:
    """Build Lean-accessor + wire maps for a list of (prefix, base, wires).

    For each binding, each wire at position `i` gets a variable name
    `<prefix><i>` mapped to a Lean accessor `<base><acc> [0 0]`. Scalar
    1×1 wires get the `0 0` scalar extraction; vector/matrix wires keep
    the raw function form.
    """
    var_accessor: dict[str, str] = {}
    var_wire: dict[str, Wire] = {}
    for prefix, base, wires in bindings:
        n = len(wires)
        for i, w in enumerate(wires):
            acc = product_accessor(i, n)
            name = f"{prefix}{i}"
            if wire_shape(w).is_scalar:
                var_accessor[name] = f"({base}{acc} 0 0)"
            else:
                var_accessor[name] = f"({base}{acc})"
            var_wire[name] = w
    return var_accessor, var_wire


def with_lets(body: str, lets: "list[tuple[str, str]]") -> str:
    """`body`, after a `let name := expr` for each of `lets`, in order."""
    if not lets:
        return body
    binds = "".join(f"\n  let {name} := {expr}" for name, expr in lets)
    return f"{binds}\n  {body}"


_INFIX = {
    "Lt": " < ", "Le": " ≤ ", "Gt": " > ", "Ge": " ≥ ", "Eq": " = ", "Ne": " ≠ ",
    "Sub": " - ", "Div": " / ", "Mod": " % ", "Implies": " → ",
}
# The chains the parser builds of one n-ary application, and the side they
# grow on: `a ∧ (b ∧ c)`, `(a + b) + c`.
_RIGHT = {"And": " ∧ ", "Or": " ∨ "}
_LEFT = {"Add": " + ", "Mul": " * "}
# `ite (a ≥ b) a b` is `max a b`, and so on; the flag says whether the
# branches are in the comparison's order.
_MIN_MAX = {("Ge", True): "max", ("Gt", True): "max",
            ("Le", True): "min", ("Lt", True): "min",
            ("Ge", False): "min", ("Gt", False): "min",
            ("Le", False): "max", ("Lt", False): "max"}


def ir_to_lean(formula, param: str, bindings: Bindings, *,
               nat: bool = False) -> str:
    """`formula` as `fun {param} => ...`, its variables read on `param` as
    `bindings` say. With `nat`, an Int-sorted `formula` as a Lean `Nat`,
    clamped by `Int.toNat`."""
    body, lets = _Printer(formula, bindings).parts()
    if nat:
        body = f"(({body} : Int)).toNat"
    return f"fun {param} => {with_lets(body, lets)}"


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

    def parts(self) -> tuple[str, list]:
        """`(body, lets)`, the `lets` in dependency order."""
        plain = self.walk(self.formula.out)
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
        state-dependent and not Bool."""
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
        if theory == "cast":
            return self.cast(t, name, args, recur)
        if theory.startswith("word") and name not in ("Ite", "Eq", "Ne"):
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
        if name != "Const":
            raise Refused(f"no Lean literal for {name}")
        return lean_literal(t.write[0].dtype, t.itype.tensor.item())

    def cast(self, t, name: str, args: list, recur) -> str:
        """The casts the parser builds, as the module's terms write them;
        an Int literal cast to Real is a Real literal."""
        if name not in ("ToInt", "UbvToInt", "ToReal"):
            raise Refused(f"no Lean form for cast.{name} in a predicate")
        inner = self.by_wire.get(args[0].id)
        if (name == "ToReal" and inner is not None
                and itype_name(inner.itype) == "Const"):
            return lean_real(inner.itype.tensor.item())
        return scalar_emitter(t.itype)([recur(args[0])])

    def ite(self, t, theory: str, args: list, recur) -> str:
        cond = args[0]
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
        """An Int `ite` that is a `min` or a `max`, as one (`omega` reads
        them, with no case split)."""
        if theory != "int":
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
