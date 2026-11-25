"""SMV (NuSMV 2.1 subset) parser — Lark frontend + lowerer to reactor Module."""

from __future__ import annotations

import warnings
import torch
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from lark import Lark, Transformer, Tree, Token

from ..reactor import Wire, Term, Module, Var, X
from ..reactor import matrix
from ..matrix import Skip
from ..sort import ZeroSort
from ..matrix.sort import Real, Int, Bool, Word



def _bw(sort) -> int:
    """Width of a word sort."""
    match sort:
        case Word(w):
            return w
    raise TypeError(f"not a word sort: {sort}")


def _const_op(value, sort):
    """The constant `value` as a scalar of `sort`: a boolean (nonzero is TRUE), an
    integer, or a word reduced modulo 2^width."""
    match sort:
        case Bool():
            return matrix.bool.Const(torch.tensor([[bool(value)]]), sort)
        case Int():
            return matrix.int.Const(torch.tensor([[int(value)]], dtype=torch.int64), sort)
        case Word(w):
            return matrix.word.Const(torch.tensor([[int(value) % (1 << w)]], dtype=torch.int64), sort)
    raise TypeError(f"no constant of sort {sort}")


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
    """SMV variable type: its sort (a boolean or a word) paired with its signedness."""
    dtype: Real | Int | Bool | Word | ZeroSort
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

# Booleans lower to `Bool`, and every other SMV value to a word: SMV's unbounded
# `integer` (and enum / range types) to words of the default width below. Adjust
# freely; the tests don't depend on the specific value.
_DEFAULT_INT_BW = 32
_SCALAR = [1, 1]


def _resolve_type(tree: Tree, enum_values: dict[str, int]) -> _SMVType:
    if isinstance(tree, Tree):
        match tree.data:
            case "type_bool":
                return _SMVType(Bool(_SCALAR), signed=False)
            case "type_int" | "range_type":
                return _SMVType(Word(_DEFAULT_INT_BW, _SCALAR), signed=True)
            case "type_spec":
                return _resolve_type(tree.children[0], enum_values)
            case "word_type":
                sign_tok = str(tree.children[0])
                width = int(str(tree.children[1]))
                return _SMVType(Word(width, _SCALAR), signed=(sign_tok == "signed"))
            case "enum_type":
                for code, ev in enumerate(_typed_children(tree, "enum_value")):
                    name = _extract_name(ev.children[0])
                    if name is not None:
                        enum_values.setdefault(name, code)
                return _SMVType(Word(_DEFAULT_INT_BW, _SCALAR), signed=False)
    # Token fallback
    text = str(tree).strip()
    if text == "boolean":
        return _SMVType(Bool(_SCALAR), signed=False)
    if text == "integer":
        return _SMVType(Word(_DEFAULT_INT_BW, _SCALAR), signed=True)
    raise ValueError(f"unsupported type: {tree}")


# ---------------------------------------------------------------------------
# 6. Lowerer
# ---------------------------------------------------------------------------

# Table-driven binary operator dispatch, by the operator's name; `_binary_op`
# resolves it, for booleans and for words, at term-construction time.
_BINOPS: dict[str, dict[str, str]] = {
    "iff": {"<->": "Xnor", "xnor": "Xnor"},
    "or_expr": {"|": "Or", "or": "Or", "xor": "Xor"},
    "and_expr": {"&": "And", "and": "And"},
    "arith_expr": {"+": "Add", "-": "Sub"},
    "mod_expr": {"mod": "Mod"},
    "term_expr": {"*": "Mul", "/": "Div"},
}

# Comparison ops split by sign; handled in _lower_cmp.
_CMP_FIXED: dict[str, str] = {"=": "Eq", "!=": "Ne"}
_CMP_ORDERED: dict[str, str] = {"<": "Lt", "<=": "Le", ">": "Gt", ">=": "Ge"}

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


def _is_literal(tree) -> bool:
    """Whether *tree* is a bare number, which takes its sort from its context."""
    node = _peel(tree)
    if isinstance(node, Token):
        return node.type == "NUMBER"
    return isinstance(node, Tree) and node.data == "int_lit"


class _Lowerer:
    """Recursive expression-to-Term lowerer.

    A number literal has no sort of its own: it takes that of the wire it is lowered
    to, or else the sort *hint* of the operand it combines with, or else a word of the
    default width.
    """

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

    def lower(self, tree, target: Wire | None = None, hint: Real | Int | Bool | Word | ZeroSort | None = None) -> Wire:
        """Lower *tree* into Term(s), return the wire carrying the result."""
        # Token leaf (from Lark inlining)
        if isinstance(tree, Token):
            if tree.type == "IDENT":
                return self._lower_name(str(tree), target, hint)
            if tree.type == "NUMBER":
                return self._emit_const(int(str(tree)), target, hint)
            raise ValueError(f"unexpected token type: {tree.type}")

        name = tree.data

        # Table-driven binary ops
        if name in _BINOPS:
            return self._lower_binop(tree, _BINOPS[name], target, hint)

        # Table-driven n-ary ops (unary / implies / ternary)
        if name in _NARY_OPS:
            return self._lower_nary(tree, _NARY_OPS[name], target, hint)

        # Table-driven method dispatch
        if name in _METHODS:
            return getattr(self, _METHODS[name])(tree, target, hint)

        if name == "var_ref":
            return self._lower_name(str(tree.children[0]), target, hint)
        if name == "int_lit":
            return self._emit_const(int(str(tree.children[0])), target, hint)
        if name in ("true_lit", "false_lit"):
            w = self._fresh(target, Bool(_SCALAR))
            self.terms.append(Term(_const_op(name == "true_lit", w.dtype), [w]))
            return self._enforce(w, target)

        # Fallback: descend into single child
        if tree.children:
            return self.lower(tree.children[0], target, hint)
        raise ValueError(f"unexpected empty tree: {tree.data}")

    def _lower_pair(self, left_tree, right_tree, hint=None):
        """Lower two operands, a number literal taking its sort from the other."""
        if _is_literal(left_tree) and not _is_literal(right_tree):
            right = self.lower(right_tree, hint=hint)
            return self.lower(left_tree, hint=right.dtype), right
        left = self.lower(left_tree, hint=hint)
        return left, self.lower(right_tree, hint=left.dtype)

    # --- operations, for booleans and for words -----------------------------

    def _emit(self, op, sort, inputs, target=None) -> Wire:
        w = self._fresh(target, sort)
        self.terms.append(Term(op, [w], inputs))
        return w

    def _binary_op(self, name: str, left: Wire, right: Wire, target=None) -> Wire:
        sort = left.dtype
        dom = [left.dtype, right.dtype]
        match sort:
            case Bool():
                if name == "Xnor":  # x <-> y is x = y
                    return self._emit(matrix.bool.Eq(dom), sort, [left, right], target)
                return self._emit(getattr(matrix.bool, name)(dom), sort, [left, right], target)
            case Int() if name in ("Add", "Sub", "Mul", "Div", "Mod"):
                return self._emit(getattr(matrix.int, name)(dom), sort, [left, right], target)
            case Word():
                signed = self._is_signed(left) or self._is_signed(right)
                if name == "Div":
                    op = (matrix.word.signed.Div if signed else matrix.word.unsigned.Div)(dom)
                elif name == "Mod":
                    op = (matrix.word.signed.Rem if signed else matrix.word.unsigned.Rem)(dom)
                else:
                    op = getattr(matrix.word, name)(dom)
                return self._emit(op, sort, [left, right], target)
        raise TypeError(f"no {name} on {sort}")

    # --- binary ops ---------------------------------------------------------

    def _lower_binop(self, tree: Tree, op_map: dict, target: Wire | None, hint=None) -> Wire:
        children = tree.children
        if len(children) == 1:  # an operand alone, with no operator
            return self.lower(children[0], target, hint)
        left, rhs = self._lower_pair(children[0], children[2], hint)
        i = 1
        while True:
            op_tok = str(children[i])
            out = target if i + 2 >= len(children) else None
            w = self._binary_op(op_map[op_tok], left, rhs, out)
            self._propagate_type(w, left, rhs)
            left = w
            i += 2
            if i >= len(children):
                break
            rhs = self.lower(children[i + 1], hint=left.dtype)
        return self._enforce(left, target)

    # --- comparison ops (sign-aware): write booleans ------------------------

    def _compare(self, op_tok: str, left: Wire, right: Wire, target=None) -> Wire:
        out = Bool(_SCALAR)
        dom = [left.dtype, right.dtype]
        match left.dtype:
            case Bool():
                if op_tok not in _CMP_FIXED:
                    raise TypeError(f"no ordering {op_tok} on booleans")
                op = getattr(matrix.bool, _CMP_FIXED[op_tok])(dom)
                return self._emit(op, out, [left, right], target)
            case Word():
                if op_tok in _CMP_FIXED:
                    op = getattr(matrix.word, _CMP_FIXED[op_tok])(dom)
                elif self._is_signed(left) or self._is_signed(right):
                    op = getattr(matrix.word.signed, _CMP_ORDERED[op_tok])(dom)
                else:
                    op = getattr(matrix.word.unsigned, _CMP_ORDERED[op_tok])(dom)
                return self._emit(op, out, [left, right], target)
        raise TypeError(f"no comparison on {left.dtype}")

    def _lower_cmp(self, tree: Tree, target: Wire | None, hint=None) -> Wire:
        children = tree.children
        if len(children) == 1:  # an operand alone, with no comparison
            return self.lower(children[0], target, hint)
        left, rhs = self._lower_pair(children[0], children[2])
        i = 1
        while True:
            out = target if i + 2 >= len(children) else None
            left = self._compare(str(children[i]), left, rhs, out)
            i += 2
            if i >= len(children):
                break
            rhs = self.lower(children[i + 1], hint=left.dtype)
        return self._enforce(left, target)

    # --- n-ary (unary / implies / ternary) ----------------------------------

    def _lower_nary(self, tree: Tree, op_name: str, target: Wire | None, hint=None) -> Wire:
        trees = [c for c in tree.children if not (isinstance(c, Token) and c.type in _OP_TOKENS)]
        if op_name == "Ite":
            cond = self._as_bool(self.lower(trees[0]))
            hint = target.dtype if target is not None else hint
            then_w, else_w = self._lower_pair(trees[1], trees[2], hint)
            return self._ite(cond, then_w, else_w, target)
        if op_name == "Implies":  # x -> y is (not x) or y
            a, b = self._lower_pair(trees[0], trees[1], hint)
            return self._binary_op("Or", self._not(a), b, target)
        x = self.lower(trees[0], hint=target.dtype if target is not None else hint)
        match op_name, x.dtype:
            case "Not", _:
                return self._enforce(self._not(x), target)
            case "Neg", Word():
                return self._enforce(self._emit(matrix.word.Neg(x.dtype), x.dtype, [x]), target)
            case "Abs", Word():
                return self._enforce(self._emit(matrix.word.signed.Abs(x.dtype), x.dtype, [x]), target)
        raise TypeError(f"no {op_name} on {x.dtype}")

    def _not(self, x: Wire) -> Wire:
        match x.dtype:
            case Bool():
                return self._emit(matrix.bool.Not(x.dtype), x.dtype, [x])
            case Word():
                return self._emit(matrix.word.Not(x.dtype), x.dtype, [x])
        raise TypeError(f"no Not on {x.dtype}")

    def _ite(self, cond: Wire, then_w: Wire, else_w: Wire, target=None) -> Wire:
        dom = [cond.dtype, then_w.dtype, else_w.dtype]
        match then_w.dtype:
            case Bool():
                op = matrix.bool.Ite(dom)
            case Word():
                op = matrix.word.Ite(dom)
            case other:
                raise TypeError(f"no Ite on {other}")
        w = self._emit(op, then_w.dtype, [cond, then_w, else_w], target)
        self._propagate_type(w, then_w, else_w)
        return self._enforce(w, target)

    def _as_bool(self, w: Wire) -> Wire:
        """A condition as a boolean: a 1-bit word `x` as `bool(x)`."""
        match w.dtype:
            case Bool():
                return w
            case Word(1):
                return self._bool_of_word1(w)
        raise TypeError(f"a condition must be a boolean, got {w.dtype}")

    def _bool_of_word1(self, w: Wire) -> Wire:
        """`bool(x)` of a word `x`: its lowest bit, `Extract(0, 0, x) = 1`."""
        s = list(w.dtype.shape)
        bit = self._emit(matrix.word.Extract(0, 0, w.dtype), Word(1, s), [w])
        one = self._emit(_const_op(1, Word(1, s)), Word(1, s), [])
        return self._emit(matrix.word.Eq([Word(1, s), Word(1, s)]), Bool(s), [bit, one])

    # --- builtin_call -------------------------------------------------------

    def _lower_builtin(self, tree: Tree, target: Wire | None, hint=None) -> Wire:
        fn_name = str(tree.children[0])
        exprs = [
            c
            for c in tree.children[1:]
            if not (isinstance(c, Token) and c.type == "BUILTIN")
        ]
        arg_w = self.lower(exprs[0])

        if fn_name == "bool":
            w = self._as_bool(arg_w) if isinstance(arg_w.dtype, Bool) else self._bool_of_word1(arg_w)
        elif fn_name == "word1":
            # a boolean as a 1-bit word: if b then 1 else 0
            one = self._emit(_const_op(1, Word(1, _SCALAR)), Word(1, _SCALAR), [])
            zero = self._emit(_const_op(0, Word(1, _SCALAR)), Word(1, _SCALAR), [])
            cond = self._as_bool(arg_w)
            op = matrix.word.Ite([cond.dtype, one.dtype, zero.dtype])
            w = self._emit(op, Word(1, _SCALAR), [cond, one, zero])
        elif fn_name in ("signed", "unsigned"):
            w = self._emit(Skip(arg_w.dtype), arg_w.dtype, [arg_w])
            self._wire_types[id(w)] = _SMVType(w.dtype, signed=(fn_name == "signed"))
        elif fn_name == "extend":
            extra = int(str(_peel(exprs[1])))
            width = _bw(arg_w.dtype)
            op = (matrix.word.signed.Extend if self._is_signed(arg_w) else matrix.word.unsigned.Extend)(extra, arg_w.dtype)
            w = self._emit(op, Word(width + extra, _SCALAR), [arg_w])
            self._propagate_type(w, arg_w)
        else:
            raise ValueError(f"unknown builtin: {fn_name}")
        return self._enforce(w, target)

    # --- case..esac ---------------------------------------------------------

    def _lower_case(self, tree: Tree, target: Wire | None, hint=None) -> Wire:
        branches = [
            c for c in tree.children if isinstance(c, Tree) and c.data == "case_branch"
        ]
        hint = target.dtype if target is not None else hint
        return self._lower_case_rec(branches, 0, target, hint)

    def _lower_case_rec(self, branches, idx, target, hint):
        if idx >= len(branches):
            return self._emit_const(0, target, hint)
        cond_tree, val_tree = branches[idx].children
        if idx == len(branches) - 1:
            return self.lower(val_tree, target, hint)
        cond = self._as_bool(self.lower(cond_tree))
        then_w = self.lower(val_tree, hint=hint)
        else_w = self._lower_case_rec(branches, idx + 1, None, then_w.dtype)
        return self._ite(cond, then_w, else_w, target)

    # --- next(expr) ---------------------------------------------------------

    def _lower_next(self, tree: Tree, target: Wire | None, hint=None) -> Wire:
        inner = tree.children[0]
        name = _extract_name(_peel(inner))
        if name and name in self.name_map:
            next_w = X(self.name_map[name])
            if name in self.var_types:
                self._wire_types[id(next_w)] = self.var_types[name]
            return self._enforce(next_w, target)
        return self.lower(inner, target, hint)

    # --- word literal -------------------------------------------------------

    def _lower_word_lit(self, tree: Tree, target: Wire | None, hint=None) -> Wire:
        lit_tok = str(tree.children[0])
        # Parse e.g. "0ud16_0" or "0sd32_300"
        signed = lit_tok[1] == "s"
        rest = lit_tok[3:]  # skip "0ud" or "0sd"
        upos = rest.index("_")
        width = int(rest[:upos])
        value = int(rest[upos + 1:])

        dtype = Word(width, _SCALAR)
        w = Wire(dtype)
        self._wire_types[id(w)] = _SMVType(dtype, signed=signed)
        self.terms.append(Term(_const_op(value, dtype), [w]))
        return self._maybe_bit_select(tree, w, 1, target)

    # --- paren_expr ---------------------------------------------------------

    def _lower_paren(self, tree: Tree, target: Wire | None, hint=None) -> Wire:
        inner = self.lower(
            tree.children[0], target if len(tree.children) == 1 else None, hint
        )
        return self._maybe_bit_select(tree, inner, 1, target)

    # --- name (ident / define / enum) ---------------------------------------

    def _lower_name(self, name: str, target: Wire | None, hint=None) -> Wire:
        # Check DEFINE
        if name in self.defines:
            if name in self._expanding:
                raise ValueError(f"circular DEFINE: {name}")
            self._expanding.add(name)
            result = self.lower(self.defines[name], target, hint)
            self._expanding.discard(name)
            return result
        # Check enum
        if name in self.enum_values:
            return self._emit_const(self.enum_values[name], target, hint)
        # Wire lookup
        if name in self.name_map:
            var = self.name_map[name]
            w = X(var) if self.is_init else var
            if name in self.var_types:
                self._wire_types[id(w)] = self.var_types[name]
            return self._enforce(w, target)
        # Unknown name — emit zero with warning
        warnings.warn(f"undeclared variable '{name}', defaulting to 0")
        return self._emit_const(0, target, hint)

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
        return Wire(dtype if dtype is not None else Word(_DEFAULT_INT_BW, _SCALAR))

    def _emit_const(self, value: int, target: Wire | None, hint=None) -> Wire:
        """A number literal, of the sort of the target, or else of the hint, or else an
        integer, as SMV's integer constants are."""
        w = self._fresh(target, hint if hint is not None else Int(_SCALAR))
        self.terms.append(Term(_const_op(value, w.dtype), [w]))
        return self._enforce(w, target)

    def _enforce(self, wire: Wire, target: Wire | None) -> Wire:
        if target is not None and target != wire:
            self.terms.append(Term(Skip(wire.dtype), [target], [wire]))
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
            op = matrix.word.Extract(high, low, base_w.dtype)
            w = self._fresh(target, Word(high - low + 1, _SCALAR))
            self.terms.append(Term(op, [w], [base_w]))
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
            init_low.terms.append(Term(_const_op(0, var.dtype), [X(var)]))

        # --- UPDATE ---
        next_expr = d.next_assigns.get(name)
        if next_expr is not None:
            update_low.lower(next_expr, X(var))
        else:
            update_low.terms.append(Term(Skip(var.dtype), [X(var)], [var]))

    # Build obs_pairs: all variables (state + ivar)
    obs_pairs = [name_map[n] for n, _ in all_decls]

    module = Module.sequential(obs_pairs, init_low.terms, update_low.terms)
    return module, name_map
