"""SMV (NuSMV 2.1 subset) parser — Lark frontend + lowerer to reactor Module."""

from __future__ import annotations

import warnings
import torch
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from lark import Lark, Transformer, Tree, Token

from .. import op, overload as ov
from ..reactor import Wire, Term, Module, Var, X
from ..sort import Sort, Word, Bool


def _bw(sort) -> int:
    """Width of a Word Sort (Sorts expose no methods; match instead)."""
    match sort:
        case Word(bw, _):
            return bw
    raise TypeError(f"not a Word sort: {sort}")


def _const(value, sort):
    """The scalar constant op of `value` on `sort`: a word, or a boolean (2-D initializer)."""
    match sort:
        case Bool(_):
            return op.bool.Const(torch.tensor([[bool(value)]]))
        case Word(w, _):
            return op.word.Const(torch.tensor([[int(value)]], dtype=torch.int64), w)
    raise TypeError(f"no SMV constant of sort {sort}")


_BUILTINS = {"bool", "word1", "unsigned", "signed", "extend"}


# ---------------------------------------------------------------------------
# 1. Lark parser init
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_parser():
    grammar = (Path(__file__).parent / "grammar.lark").read_text()
    return Lark(grammar, parser="earley", propagate_positions=True)


# ---------------------------------------------------------------------------
# 2. Entry point
# ---------------------------------------------------------------------------


def parse_smv(
        text: str,
        overrides: dict[str, tuple[Wire, Wire]] | None = None,
) -> tuple[Module, dict[str, tuple[Wire, Wire]]]:
    """Parse an SMV source string and return ``(Module, name_map)``.

    *name_map* maps every variable name to its ``(latched, next)`` wire pair.
    *overrides* lets the caller supply pre-existing wires for specific
    variables (used for module composition via shared wires).
    """
    tree = _get_parser().parse(text)
    decls = _collect_declarations(tree)
    _promote_constraints(decls)
    return _build_module(decls, overrides or {})


# ---------------------------------------------------------------------------
# 3. Pass 1 — collect declarations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SMVType:
    """SMV variable type: a Word or Bool Sort paired with its signedness."""
    dtype: Sort
    signed: bool


@dataclass(slots=True)
class _Decls:
    var_decls: list[tuple[str, _SMVType]] = field(default_factory=list)
    ivar_decls: list[tuple[str, _SMVType]] = field(default_factory=list)
    frozen_decls: list[tuple[str, _SMVType]] = field(default_factory=list)
    define_map: dict[str, Tree] = field(default_factory=dict)
    init_assigns: dict[str, Tree] = field(default_factory=dict)
    next_assigns: dict[str, Tree] = field(default_factory=dict)
    init_exprs: list[Tree] = field(default_factory=list)
    trans_exprs: list[Tree] = field(default_factory=list)
    invar_exprs: list[Tree] = field(default_factory=list)
    enum_values: dict[str, int] = field(default_factory=dict)


def _typed_children(section: Tree, data: str):
    return (c for c in section.children if isinstance(c, Tree) and c.data == data)


_ASSIGN_TARGETS = {
    "init_ref": "init_assigns",
    "next_ref": "next_assigns",
    "bare_assign": "init_assigns",
}


def _var_collector(attr):
    def method(self, children):
        target = getattr(self.decls, attr)
        for child in children:
            if isinstance(child, tuple):
                target.append(child)

    return method


class _DeclCollector(Transformer):
    """Lark Transformer that collects SMV declarations into a _Decls struct."""

    def __init__(self):
        super().__init__()
        self.decls = _Decls()

    # --- variable declarations ----------------------------------------------

    def var_decl(self, children):
        return (str(children[0]), _resolve_type(children[1], self.decls.enum_values))

    var_section = _var_collector("var_decls")
    ivar_section = _var_collector("ivar_decls")
    frozenvar_section = _var_collector("frozen_decls")

    # --- defines ------------------------------------------------------------

    def define_decl(self, children):
        return (str(children[0]), children[1])

    def define_section(self, children):
        for child in children:
            if isinstance(child, tuple):
                self.decls.define_map[child[0]] = child[1]

    # --- assignments --------------------------------------------------------

    def assign_stmt(self, children):
        target_node, expr = children[0], children[1]
        if isinstance(target_node, Tree) and target_node.data in _ASSIGN_TARGETS:
            return (
                _ASSIGN_TARGETS[target_node.data],
                str(target_node.children[0]),
                expr,
            )
        return ("init_assigns", str(target_node), expr)

    def assign_section(self, children):
        for child in children:
            if isinstance(child, tuple) and len(child) == 3:
                attr, name, expr = child
                getattr(self.decls, attr)[name] = expr

    # --- constraint sections ------------------------------------------------

    def init_section(self, children):
        self.decls.init_exprs.append(children[0])

    def trans_section(self, children):
        self.decls.trans_exprs.append(children[0])

    def invar_section(self, children):
        self.decls.invar_exprs.append(children[0])


def _collect_declarations(tree: Tree) -> _Decls:
    collector = _DeclCollector()
    collector.transform(tree)
    return collector.decls


# ---------------------------------------------------------------------------
# 4. Constraint-to-assignment promotion
# ---------------------------------------------------------------------------


def _promote_loop(
        exprs, var_names, target: dict[str, Tree], *, expect_next=False
) -> list[Tree]:
    remaining = []
    for expr in exprs:
        result = _try_extract_assign(expr, var_names, expect_next=expect_next)
        if result and result[0] not in target:
            target[result[0]] = result[1]
        else:
            remaining.append(expr)
    return remaining


def _promote_constraints(d: _Decls) -> None:
    var_names = {n for n, _ in d.var_decls} | {n for n, _ in d.frozen_decls}

    # TRANS: next(var) = expr -> next_assigns
    d.trans_exprs = _promote_loop(
        d.trans_exprs, var_names, d.next_assigns, expect_next=True
    )

    # INIT: var = expr -> init_assigns
    d.init_exprs = _promote_loop(d.init_exprs, var_names, d.init_assigns)

    # INVAR: var = expr -> define_map + next_assigns
    parsed = [_try_extract_assign(expr, var_names) for expr in d.invar_exprs]
    counts = Counter(p[0] for p in parsed if p)

    remaining = []
    for expr, p in zip(d.invar_exprs, parsed):
        if p and counts[p[0]] == 1 and p[0] not in d.next_assigns:
            d.define_map[p[0]] = p[1]
            d.next_assigns[p[0]] = p[1]
        else:
            remaining.append(expr)
    d.invar_exprs = remaining


def _peel(tree):
    """Peel single-child wrappers down to a leaf."""
    while isinstance(tree, Tree) and len(tree.children) == 1:
        tree = tree.children[0]
    return tree


def _extract_name(node) -> str | None:
    """Get variable name from a peeled Lark node (var_ref Tree or IDENT Token)."""
    if isinstance(node, Tree) and node.data == "var_ref":
        return str(node.children[0])
    if isinstance(node, Token) and node.type == "IDENT":
        return str(node)
    return None


def _try_extract_assign(
        expr, var_names: set[str], *, expect_next: bool = False
) -> tuple[str, Tree] | None:
    """Detect ``var = rhs`` (or ``next(var) = rhs`` when *expect_next*) in an expression tree."""
    node = _peel(expr)
    if not isinstance(node, Tree):
        return None

    children = node.children
    if len(children) != 3:
        return None

    op = children[1]
    if not isinstance(op, Token) or str(op) != "=":
        return None

    lhs = _peel(children[0])
    if expect_next:
        if not isinstance(lhs, Tree) or lhs.data != "next_expr":
            return None
        lhs = _peel(lhs.children[0])

    name = _extract_name(lhs)
    return (name, children[2]) if name in var_names else None


# ---------------------------------------------------------------------------
# 5. Type resolution
# ---------------------------------------------------------------------------

# The SMV parser is BV-only: every SMV value lowers to a word or a boolean wire,
# every emitted op is in the logic BV. The default width below is used for
# SMV's unbounded `integer` (and enum / range types). Adjust freely; the
# tests don't depend on the specific value.
_DEFAULT_INT_BW = 32


def _resolve_type(tree: Tree, enum_values: dict[str, int]) -> _SMVType:
    if isinstance(tree, Tree):
        match tree.data:
            case "type_bool":
                return _SMVType(Bool([1, 1]), signed=False)
            case "type_int" | "range_type":
                return _SMVType(Word(_DEFAULT_INT_BW, [1, 1]), signed=True)
            case "type_spec":
                return _resolve_type(tree.children[0], enum_values)
            case "word_type":
                sign_tok = str(tree.children[0])
                width = int(str(tree.children[1]))
                return _SMVType(Word(width, [1, 1]), signed=(sign_tok == "signed"))
            case "enum_type":
                for code, ev in enumerate(_typed_children(tree, "enum_value")):
                    name = _extract_name(ev.children[0])
                    if name is not None:
                        enum_values.setdefault(name, code)
                return _SMVType(Word(_DEFAULT_INT_BW, [1, 1]), signed=False)
    # Token fallback
    text = str(tree).strip()
    if text == "boolean":
        return _SMVType(Bool([1, 1]), signed=False)
    if text == "integer":
        return _SMVType(Word(_DEFAULT_INT_BW, [1, 1]), signed=True)
    raise ValueError(f"unsupported type: {tree}")


# ---------------------------------------------------------------------------
# 6. Lowerer
# ---------------------------------------------------------------------------

# Table-driven binary operator dispatch. Values name the overloaded operations of
# `reactor.overload`, which pick the op of the operands' sort: logic on booleans and
# bitwise on words. Division and remainder are unsigned, whatever the operands.
_BINOPS: dict[str, dict[str, str]] = {
    "iff": {"<->": "iff", "xnor": "iff"},
    "or_expr": {"|": "or_", "or": "or_", "xor": "xor_"},
    "and_expr": {"&": "and_", "and": "and_"},
    "arith_expr": {"+": "add", "-": "sub"},
    "mod_expr": {"mod": "mod"},
    "term_expr": {"*": "mul", "/": "div"},
}

# Ordered comparisons pick signed/unsigned by their left operand; see _lower_cmp.
_CMP_ORDERED = {"<": ov.lt, "<=": ov.le, ">": ov.gt, ">=": ov.ge}

_OP_TOKENS = {"NOT_OP"}

_NARY_OPS = {
    "ternary": "Ite",
    "implies_expr": "Implies",
    "not_expr": "Not",
    "neg": "Neg",
    "abs_call": "Abs",
}

_METHODS = {
    "builtin_call": "_lower_builtin",
    "case_expr": "_lower_case",
    "cmp_expr": "_lower_cmp",
    "next_expr": "_lower_next",
    "word_lit": "_lower_word_lit",
    "paren_expr": "_lower_paren",
}


class _Lowerer:
    """Recursive expression-to-Term lowerer."""

    def __init__(
            self,
            name_map: dict[str, tuple[Wire, Wire]],
            defines: dict[str, Tree],
            enum_values: dict[str, int],
            var_types: dict[str, _SMVType],
            is_init: bool = False,
    ):
        self.name_map = name_map
        self.defines = defines
        self.enum_values = enum_values
        self.var_types = var_types
        self.is_init = is_init
        self.terms: list[Term] = []
        self._expanding: set[str] = set()
        self._wire_types: dict[int, _SMVType] = {}

    # --- public entry -------------------------------------------------------

    def lower(self, tree, target: Wire | None = None) -> Wire:
        """Lower *tree* into Term(s), return the wire carrying the result."""
        # Token leaf (from Lark inlining)
        if isinstance(tree, Token):
            if tree.type == "IDENT":
                return self._lower_name(str(tree), target)
            if tree.type == "NUMBER":
                return self._emit_const(int(str(tree)), target)
            raise ValueError(f"unexpected token type: {tree.type}")

        name = tree.data

        # Table-driven binary ops
        if name in _BINOPS:
            return self._lower_binop(tree, _BINOPS[name], target)

        # Table-driven n-ary ops (unary / implies / ternary)
        if name in _NARY_OPS:
            return self._lower_nary(tree, _NARY_OPS[name], target)

        # Table-driven method dispatch
        if name in _METHODS:
            return getattr(self, _METHODS[name])(tree, target)

        if name == "var_ref":
            return self._lower_name(str(tree.children[0]), target)
        if name == "int_lit":
            return self._emit_const(int(str(tree.children[0])), target)
        if name in ("true_lit", "false_lit"):
            w = self._fresh(target, Bool([1, 1]))
            self.terms.append(Term(_const(name == "true_lit", w.dtype), [w]))
            return self._enforce(w, target)

        # Fallback: descend into single child
        if tree.children:
            return self.lower(tree.children[0], target)
        raise ValueError(f"unexpected empty tree: {tree.data}")

    # --- binary ops ---------------------------------------------------------

    def _emit(self, itype, out_dtype, inputs, out: Wire | None = None) -> Wire:
        """Emit `itype` reading `inputs` into `out`, or a fresh wire of `out_dtype`."""
        w = self._fresh(out, out_dtype)
        self.terms.append(Term(itype, [w], list(inputs)))
        return w

    def _emit_binop(self, name: str, left: Wire, rhs: Wire, out: Wire | None) -> Wire:
        sort = left.dtype
        if name == "iff":  # x <-> y: Xnor on words, and Not(Xor) on booleans
            if isinstance(sort, Word):
                return self._emit(op.word.Xnor(_bw(sort), 1, 1), sort, [left, rhs], out)
            xor = self._emit(ov.xor_(sort), sort, [left, rhs])
            return self._emit(ov.not_(sort), sort, [xor], out)
        return self._emit(getattr(ov, name)(sort), sort, [left, rhs], out)

    def _lower_binop(self, tree: Tree, op_map: dict, target: Wire | None) -> Wire:
        children = tree.children
        left = self.lower(children[0])
        i = 1
        while i < len(children):
            op_tok = str(children[i])
            rhs = self.lower(children[i + 1])
            out = target if i + 2 >= len(children) else None
            w = self._emit_binop(op_map[op_tok], left, rhs, out)
            self._propagate_type(w, left, rhs)
            left = w
            i += 2
        return self._enforce(left, target)

    # --- comparison ops (sign-aware) ----------------------------------------

    def _emit_cmp(self, op_tok: str, left: Wire, rhs: Wire, out: Wire | None) -> Wire:
        sort = left.dtype
        if isinstance(sort, Bool) and op_tok in ("=", "!="):
            # booleans have no Eq: x = y is x <-> y, and x != y is x xor y
            if op_tok == "=":
                return self._emit_binop("iff", left, rhs, out)
            return self._emit(ov.xor_(sort), sort, [left, rhs], out)
        if op_tok == "=":
            itype = ov.eq(sort)
        elif op_tok == "!=":
            itype = ov.ne(sort)
        else:
            itype = _CMP_ORDERED[op_tok](sort, signed=self._is_signed(left))
        return self._emit(itype, Bool([1, 1]), [left, rhs], out)

    def _lower_cmp(self, tree: Tree, target: Wire | None) -> Wire:
        children = tree.children
        left = self.lower(children[0])
        i = 1
        while i < len(children):
            op_tok = str(children[i])
            rhs = self.lower(children[i + 1])
            out = target if i + 2 >= len(children) else None
            left = self._emit_cmp(op_tok, left, rhs, out)
            i += 2
        return self._enforce(left, target)

    # --- n-ary (unary / implies / ternary) ----------------------------------

    def _lower_nary(self, tree: Tree, op_name: str, target: Wire | None) -> Wire:
        inputs = [
            self.lower(c)
            for c in tree.children
            if not (isinstance(c, Token) and c.type in _OP_TOKENS)
        ]
        match op_name:
            case "Ite":  # output matches the then-branch
                sort = inputs[1].dtype
                w = self._emit(ov.ite(sort), sort, inputs, target)
            case "Not":
                w = self._emit(ov.not_(inputs[0].dtype), inputs[0].dtype, inputs, target)
            case "Neg":
                w = self._emit(ov.neg(inputs[0].dtype), inputs[0].dtype, inputs, target)
            case "Implies":  # x -> y is !x | y
                x, y = inputs
                not_x = self._emit(ov.not_(x.dtype), x.dtype, [x])
                w = self._emit(ov.or_(x.dtype), x.dtype, [not_x, y], target)
            case "Abs":  # abs(x) is ite(x < 0, -x, x), with x signed
                (x,) = inputs
                zero = self._emit_const(0, Wire(x.dtype))
                negative = self._emit(ov.lt(x.dtype, signed=True), Bool([1, 1]), [x, zero])
                minus_x = self._emit(ov.neg(x.dtype), x.dtype, [x])
                w = self._emit(ov.ite(x.dtype), x.dtype, [negative, minus_x, x], target)
            case _:
                raise ValueError(f"unknown operator: {op_name}")
        return self._enforce(w, target)

    # --- builtin_call -------------------------------------------------------

    def _lower_builtin(self, tree: Tree, target: Wire | None) -> Wire:
        fn_name = str(tree.children[0])
        exprs = [
            c
            for c in tree.children[1:]
            if not (isinstance(c, Token) and c.type == "BUILTIN")
        ]
        arg_w = self.lower(exprs[0])
        sort = arg_w.dtype

        if fn_name not in _BUILTINS:
            raise ValueError(f"unknown builtin: {fn_name}")
        if fn_name == "bool" and isinstance(sort, Word):  # bool(x) is x != 0
            zero = self._emit_const(0, Wire(sort))
            w = self._emit(ov.ne(sort), Bool([1, 1]), [arg_w, zero], target)
        elif fn_name == "word1" and isinstance(sort, Bool):  # word1(b) is ite(b, 1, 0)
            word1 = Word(1, [1, 1])
            one, zero = self._emit_const(1, Wire(word1)), self._emit_const(0, Wire(word1))
            w = self._emit(ov.ite(word1), word1, [arg_w, one, zero], target)
        elif fn_name == "extend":
            extra = int(str(_peel(exprs[1])))
            itype = ov.extend(sort, extra, signed=self._is_signed(arg_w))
            w = self._emit(itype, Word(_bw(sort) + extra, [1, 1]), [arg_w], target)
        else:  # signed / unsigned, and bool / word1 of their own sort, keep the value
            w = self._emit(ov.id_(sort), sort, [arg_w], target)

        if fn_name in ("signed", "unsigned"):
            self._wire_types[id(w)] = _SMVType(w.dtype, signed=(fn_name == "signed"))
        return self._enforce(w, target)

    # --- case..esac ---------------------------------------------------------

    def _lower_case(self, tree: Tree, target: Wire | None) -> Wire:
        branches = [
            c for c in tree.children if isinstance(c, Tree) and c.data == "case_branch"
        ]
        return self._lower_case_rec(branches, 0, target)

    def _lower_case_rec(self, branches, idx, target):
        if idx >= len(branches):
            return self._emit_const(0, target)
        cond_tree, val_tree = branches[idx].children
        if idx == len(branches) - 1:
            return self.lower(val_tree, target)
        cond = self.lower(cond_tree)
        then_w = self.lower(val_tree)
        else_w = self._lower_case_rec(branches, idx + 1, None)
        w = self._emit(ov.ite(then_w.dtype), then_w.dtype, [cond, then_w, else_w], target)
        return self._enforce(w, target)

    # --- next(expr) ---------------------------------------------------------

    def _lower_next(self, tree: Tree, target: Wire | None) -> Wire:
        inner = tree.children[0]
        name = _extract_name(_peel(inner))
        if name and name in self.name_map:
            _, next_w = self.name_map[name]
            return self._enforce(next_w, target)
        return self.lower(inner, target)

    # --- word literal -------------------------------------------------------

    def _lower_word_lit(self, tree: Tree, target: Wire | None) -> Wire:
        lit_tok = str(tree.children[0])
        # Parse e.g. "0ud16_0" or "0sd32_300"
        signed = lit_tok[1] == "s"
        rest = lit_tok[3:]  # skip "0ud" or "0sd"
        upos = rest.index("_")
        width = int(rest[:upos])
        value = int(rest[upos + 1:])

        if signed and value < 0:
            value = value + (1 << width)
        dtype = Word(width, [1, 1])
        w = Wire(dtype)
        self._wire_types[id(w)] = _SMVType(dtype, signed=signed)
        self.terms.append(Term(_const(value, dtype), [w]))
        return self._maybe_bit_select(tree, w, 1, target)

    # --- paren_expr ---------------------------------------------------------

    def _lower_paren(self, tree: Tree, target: Wire | None) -> Wire:
        inner = self.lower(
            tree.children[0], target if len(tree.children) == 1 else None
        )
        return self._maybe_bit_select(tree, inner, 1, target)

    # --- name (ident / define / enum) ---------------------------------------

    def _lower_name(self, name: str, target: Wire | None) -> Wire:
        # Check DEFINE
        if name in self.defines:
            if name in self._expanding:
                raise ValueError(f"circular DEFINE: {name}")
            self._expanding.add(name)
            result = self.lower(self.defines[name], target)
            self._expanding.discard(name)
            return result
        # Check enum
        if name in self.enum_values:
            return self._emit_const(self.enum_values[name], target)
        # Wire lookup
        if name in self.name_map:
            var = self.name_map[name]
            w = X(var) if self.is_init else var
            if name in self.var_types:
                self._wire_types[id(w)] = self.var_types[name]
            return self._enforce(w, target)
        # Unknown name — emit zero with warning
        warnings.warn(f"undeclared variable '{name}', defaulting to 0")
        return self._emit_const(0, target)

    # --- helpers ------------------------------------------------------------

    def _smv_type(self, wire: Wire) -> _SMVType | None:
        return self._wire_types.get(id(wire))

    def _is_signed(self, wire: Wire) -> bool:
        t = self._smv_type(wire)
        return t.signed if t is not None else False

    def _propagate_type(self, result: Wire, *inputs: Wire) -> None:
        """Set result's _SMVType to signed if any input is signed."""
        signed = any(self._is_signed(w) for w in inputs)
        if signed or any(self._smv_type(w) is not None for w in inputs):
            self._wire_types[id(result)] = _SMVType(result.dtype, signed=signed)

    def _fresh(self, target: Wire | None, dtype=None) -> Wire:
        if target is not None:
            return target
        return Wire(dtype if dtype is not None else Word(_DEFAULT_INT_BW, [1, 1]))

    def _emit_const(self, value: int, target: Wire | None) -> Wire:
        """The constant `value` on `target`, or on a fresh word of the default width."""
        w = self._fresh(target)
        self.terms.append(Term(_const(value, w.dtype), [w]))
        return self._enforce(w, target)

    def _enforce(self, wire: Wire, target: Wire | None) -> Wire:
        if target is not None and target != wire:
            self.terms.append(Term(ov.id_(wire.dtype), [target], [wire]))
            t = self._smv_type(wire)
            if t is not None:
                self._wire_types[id(target)] = t
            return target
        return wire

    def _maybe_bit_select(
            self, tree: Tree, base_w: Wire, idx: int, target: Wire | None
    ) -> Wire:
        if (
                len(tree.children) > idx
                and isinstance(tree.children[idx], Tree)
                and tree.children[idx].data == "bit_select"
        ):
            bs = tree.children[idx]
            high, low = int(str(bs.children[0])), int(str(bs.children[1]))
            out_dtype = Word(high - low + 1, [1, 1])
            itype = ov.extract(base_w.dtype, high, low)
            w = self._emit(itype, out_dtype, [base_w], target)
            return self._enforce(w, target)
        return self._enforce(base_w, target)


# ---------------------------------------------------------------------------
# 7. Pass 2 — build Module
# ---------------------------------------------------------------------------


def _build_module(
        d: _Decls,
        overrides: dict[str, Var],
) -> tuple[Module, dict[str, Var]]:
    # Prepend frozen vars before regular state vars
    all_var_decls = list(d.frozen_decls) + list(d.var_decls)

    # Combine: state vars first, then ivars
    all_decls = all_var_decls + list(d.ivar_decls)
    # Create wire pairs and type map
    name_map: dict[str, Var] = {}
    var_types: dict[str, _SMVType] = {}
    for name, smv_type in all_decls:
        var_types[name] = smv_type
        name_map[name] = (
            overrides[name]
            if name in overrides
            else Var(smv_type.dtype)
        )

    init_low = _Lowerer(name_map, d.define_map, d.enum_values, var_types, is_init=True)
    update_low = _Lowerer(name_map, d.define_map, d.enum_values, var_types, is_init=False)

    for name, smv_type in all_var_decls:
        var = name_map[name]

        # --- INIT ---
        init_expr = d.init_assigns.get(name)
        if init_expr is not None:
            init_low.lower(init_expr, X(var))
        else:
            init_low.terms.append(Term(_const(0, X(var).dtype), [X(var)]))

        # --- UPDATE ---
        next_expr = d.next_assigns.get(name)
        if next_expr is not None:
            update_low.lower(next_expr, X(var))
        else:
            update_low.terms.append(Term(ov.id_(var.dtype), [X(var)], [var]))

    # Build obs_pairs: all variables (state + ivar)
    obs_pairs = [name_map[n] for n, _ in all_decls]

    module = Module.sequential(obs_pairs, init_low.terms, update_low.terms)
    return module, name_map
