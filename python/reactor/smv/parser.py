"""SMV (NuSMV 2.1 subset) parser — Lark frontend + lowerer to reactor Module."""

from __future__ import annotations

import warnings
import torch
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import lru_cache, partial
from pathlib import Path
from lark import Lark, Transformer, Tree, Token
from lark.exceptions import LarkError

from ..reactor import Wire, Term, Module, Var, X
from ..reactor import tensor
from ..reactor.tensor import Skip
from ..sort import Real, Int, Bool, Word, Zero



def _bw(sort) -> int:
    """Width of a word sort."""
    match sort:
        case Word(w, _):
            return w
    raise TypeError(f"not a word sort: {sort}")


def _const_op(value, sort):
    """The constant `value` as a scalar of `sort`: a boolean (nonzero is TRUE), an
    integer, or a word reduced modulo 2^width."""
    match sort:
        case Bool(_):
            return tensor.bool.Const(torch.tensor([[bool(value)]]))
        case Int(_):
            return tensor.int.Const(torch.tensor([[int(value)]], dtype=torch.int64))
        case Word(w, _):
            return tensor.word.Const(torch.tensor([[int(value) % (1 << w)]], dtype=torch.int64), w)
    raise TypeError(f"no constant of sort {sort}")


def _sort_name(sort) -> str:
    """`sort` as NuSMV names it, with its article: "a boolean", "an integer",
    "a word[8]"."""
    match sort:
        case Bool(_):
            return "a boolean"
        case Int(_):
            return "an integer"
        case Word(w, _):
            return f"a word[{w}]"
    return f"a {sort}"


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


class SMVRefused(ValueError):
    """A model `parse_smv(strict=True)` would read differently from NuSMV."""


class SMVSyntaxError(ValueError):
    """Text the SMV grammar does not read: the parser's own account of where it
    stopped, and what it expected there."""


@dataclass(frozen=True)
class Spec:
    """A property the model states, as a claim: `kind` "safety" (`INVARSPEC p`,
    `LTLSPEC G p`) or "buchi" (`LTLSPEC G F p`). `wire` carries `p`, read at the state
    after a round -- every variable's next value -- and `terms` compute it from those
    values."""
    kind: str
    where: str  # e.g. "line 12: INVARSPEC"
    wire: Wire
    terms: tuple


def read_smv(text: str) -> tuple[Module, dict[str, Var], list[Spec]]:
    """`parse_smv(text, strict=True)`, and the properties the model states.

    A spec that is not `INVARSPEC p`, `LTLSPEC G p` or `LTLSPEC G F p` -- any other
    temporal operator, or one inside `p` -- is refused, as is a `p` that reads an
    input.
    """
    return _read(text, {}, strict=True, specs=True)


def parse_smv(
        text: str,
        overrides: dict[str, Var] | None = None,
        *,
        strict: bool = False,
) -> tuple[Module, dict[str, Var]]:
    """Parse an SMV source string and return ``(Module, name_map)``.

    *name_map* maps every variable name to its variable.
    *overrides* lets the caller supply pre-existing variables for specific
    names (used for module composition via shared wires).

    With *strict*, the module means what NuSMV means by the text, or `SMVRefused`
    says why it cannot: a constraint that is not an assignment, a fairness
    constraint, a variable left free (no ``init`` or no ``next``), an integer type in
    a model with words (it would lower to a wrapping word), an operand of the wrong
    sort. A model with no words lowers to integers: `integer`, ranges and enum codes
    are `Int`, booleans `Bool`, and only linear integer arithmetic is read. Inputs
    are read in the round they label, ``next(e)`` reads ``e`` at the next state, and
    ``INVAR x = e`` / ``x := e`` hold in every state. Without it, such text is read
    leniently: constraints dropped, free variables started at 0 and held. Either
    way, text the grammar does not read is `SMVSyntaxError`.
    """
    return _read(text, overrides or {}, strict=strict)[:2]


def _read(text: str, overrides: dict[str, Var], *, strict: bool, specs: bool = False):
    """The passes both entry points run, in order; `_build_module`'s result."""
    try:
        tree = _get_parser().parse(text)
    except LarkError as e:
        raise SMVSyntaxError(str(e)) from e
    decls = _collect_declarations(tree)
    _split_assigns(decls, strict=strict)
    _promote_constraints(decls, strict=strict)
    theory = _theory(tree) if strict else "word"
    if strict:
        _check_strict(decls, theory)
    return _build_module(decls, overrides, strict=strict, theory=theory, specs=specs)


def _theory(tree: Tree) -> str:
    """"word" for a model with words (a word type, a word constant, a word builtin or
    a bit selection), else "int"."""
    words = {"word_type", "word_lit", "builtin_call", "bit_select"}
    return "word" if any(t.data in words for t in tree.iter_subtrees()) else "int"


# ---------------------------------------------------------------------------
# 3. Pass 1 — collect declarations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SMVType:
    """SMV variable type: its sort paired with its signedness.

    `kind` is the SMV type it came from: "boolean", "word", "enum", "integer"
    (`integer`, a range, or an enum of numbers only) or "mixed-enum" (an enum of
    names and numbers)."""
    dtype: Real | Int | Bool | Word | Zero
    signed: bool
    kind: str = "word"


@dataclass(slots=True)
class _Decls:
    var_decls: list[tuple[str, _SMVType]] = field(default_factory=list)
    ivar_decls: list[tuple[str, _SMVType]] = field(default_factory=list)
    frozen_decls: list[tuple[str, _SMVType]] = field(default_factory=list)
    define_map: dict[str, Tree] = field(default_factory=dict)
    # ASSIGN statements in source order: (kind, name, rhs), kind one of "init"
    # (`init(x) :=`), "next" (`next(x) :=`), "invar" (`x :=`).
    assigns: list[tuple[str, str, Tree]] = field(default_factory=list)
    init_assigns: dict[str, Tree] = field(default_factory=dict)
    next_assigns: dict[str, Tree] = field(default_factory=dict)
    # Strict only: `x` equals the rhs in every state, the initial one included.
    invar_assigns: dict[str, Tree] = field(default_factory=dict)
    init_exprs: list[Tree] = field(default_factory=list)
    trans_exprs: list[Tree] = field(default_factory=list)
    invar_exprs: list[Tree] = field(default_factory=list)
    fairness: list[tuple[str, int | None]] = field(default_factory=list)
    # (keyword, formula) for each INVARSPEC / LTLSPEC, in source order.
    specs: list[tuple[str, Tree]] = field(default_factory=list)
    enum_values: dict[str, int] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)


def _typed_children(section: Tree, data: str):
    return (c for c in section.children if isinstance(c, Tree) and c.data == data)


def _line(node) -> int | None:
    if isinstance(node, Token):
        return node.line
    if isinstance(node, Tree) and not node.meta.empty:
        return node.meta.line
    return None


def _at(node) -> str:
    line = _line(node)
    return f"line {line}: " if line is not None else ""


_ASSIGN_TARGETS = {
    "init_ref": "init",
    "next_ref": "next",
    "bare_assign": "invar",
}


def _var_collector(attr):
    def method(self, children):
        target = getattr(self.decls, attr)
        for child in children:
            if isinstance(child, tuple):
                target.append(child)

    return method


def _fairness_collector(keyword):
    def method(self, children):
        self.decls.fairness.append((keyword, _line(children[0])))

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
        return ("invar", str(target_node), expr)

    def assign_section(self, children):
        for child in children:
            if isinstance(child, tuple) and len(child) == 3:
                self.decls.assigns.append(child)

    # --- constraint sections ------------------------------------------------

    def init_section(self, children):
        self.decls.init_exprs.append(children[0])

    def trans_section(self, children):
        self.decls.trans_exprs.append(children[0])

    def invar_section(self, children):
        self.decls.invar_exprs.append(children[0])

    fairness_section = _fairness_collector("FAIRNESS")
    justice_section = _fairness_collector("JUSTICE")
    compassion_section = _fairness_collector("COMPASSION")

    # --- specifications -----------------------------------------------------

    def invarspec_section(self, children):
        self.decls.specs.append(("INVARSPEC", children[-1]))

    def ltlspec_section(self, children):
        self.decls.specs.append(("LTLSPEC", children[-1]))


def _collect_declarations(tree: Tree) -> _Decls:
    collector = _DeclCollector()
    collector.transform(tree)
    return collector.decls


def _split_assigns(d: _Decls, *, strict: bool) -> None:
    """Sort the ASSIGN statements into `init_assigns` / `next_assigns` (and, when
    strict, `invar_assigns`).

    Lenient: `x := e` is an initial value, and a later assignment to the same target
    replaces an earlier one. Strict: `x := e` holds in every state, and a second
    assignment to the same target is a problem.
    """
    for kind, name, expr in d.assigns:
        if not strict:
            target = d.next_assigns if kind == "next" else d.init_assigns
            target[name] = expr
            continue
        target = {"init": d.init_assigns, "next": d.next_assigns,
                  "invar": d.invar_assigns}[kind]
        if name in target:
            d.problems.append(f"{_at(expr)}`{name}` is assigned twice")
            continue
        target[name] = expr


# ---------------------------------------------------------------------------
# 4. Constraint-to-assignment promotion
# ---------------------------------------------------------------------------


def _promote_loop(
        exprs, var_names, target: dict[str, Tree], *, expect_next=False, taken=(),
) -> list[Tree]:
    remaining = []
    for expr in exprs:
        result = _try_extract_assign(expr, var_names, expect_next=expect_next)
        if result and result[0] not in target and result[0] not in taken:
            target[result[0]] = result[1]
        else:
            remaining.append(expr)
    return remaining


def _promote_constraints(d: _Decls, *, strict: bool = False) -> None:
    var_names = {n for n, _ in d.var_decls} | {n for n, _ in d.frozen_decls}
    taken = d.invar_assigns

    # TRANS: next(var) = expr -> next_assigns
    d.trans_exprs = _promote_loop(
        d.trans_exprs, var_names, d.next_assigns, expect_next=True, taken=taken
    )

    # INIT: var = expr -> init_assigns
    d.init_exprs = _promote_loop(d.init_exprs, var_names, d.init_assigns, taken=taken)

    # INVAR: var = expr -> define_map + next_assigns (strict: invar_assigns)
    parsed = [_try_extract_assign(expr, var_names) for expr in d.invar_exprs]
    counts = Counter(p[0] for p in parsed if p)

    remaining = []
    for expr, p in zip(d.invar_exprs, parsed):
        if strict:
            if (p and counts[p[0]] == 1 and p[0] not in d.next_assigns
                    and p[0] not in d.init_assigns and p[0] not in d.invar_assigns):
                d.invar_assigns[p[0]] = p[1]
            else:
                remaining.append(expr)
        elif p and counts[p[0]] == 1 and p[0] not in d.next_assigns:
            d.define_map[p[0]] = p[1]
            d.next_assigns[p[0]] = p[1]
        else:
            remaining.append(expr)
    d.invar_exprs = remaining


def _check_strict(d: _Decls, theory: str) -> None:
    """Raise `SMVRefused` listing everything strict parsing would read differently
    from NuSMV, as far as the declarations show it. What only the expressions show
    (an undeclared name, an input read where NuSMV does not allow one, an operand of
    the wrong sort) is refused while lowering them."""
    problems = list(d.problems)
    for keyword, line in d.fairness:
        at = f"line {line}: " if line is not None else ""
        problems.append(f"{at}{keyword} constraints are not read")
    for keyword, form, exprs in (
            ("INIT", "`x = e`", d.init_exprs),
            ("TRANS", "`next(x) = e`", d.trans_exprs),
            ("INVAR", "`x = e`", d.invar_exprs)):
        for expr in exprs:
            problems.append(
                f"{_at(expr)}{keyword} is not {form} for a variable with no other "
                "such value"
            )

    var_names = [n for n, _ in d.var_decls]
    frozen = [n for n, _ in d.frozen_decls]
    ivars = {n for n, _ in d.ivar_decls}
    declared = set(var_names) | set(frozen)
    for assigns in (d.init_assigns, d.next_assigns, d.invar_assigns):
        for name, expr in assigns.items():
            if name in ivars:
                problems.append(f"{_at(expr)}input variable `{name}` is assigned")
            elif name not in declared:
                problems.append(f"{_at(expr)}`{name}` is assigned but not declared")
    for name in var_names + frozen:
        if name in d.invar_assigns:
            if name in d.init_assigns or name in d.next_assigns:
                problems.append(
                    f"`{name}` is assigned in every state and also its initial or "
                    "next value"
                )
            continue
        if name not in d.init_assigns:
            problems.append(f"`{name}` has no initial value (NuSMV leaves it free)")
        if name in frozen:
            if name in d.next_assigns:
                problems.append(f"FROZENVAR `{name}` is assigned a next value")
        elif name not in d.next_assigns:
            problems.append(
                f"`{name}` has no next value (NuSMV leaves it free at every step)"
            )

    for name, t in d.var_decls + d.frozen_decls + d.ivar_decls:
        if t.kind == "integer" and theory == "word":
            problems.append(
                f"`{name}` has an integer type in a model with words, so it would "
                f"lower to a {_DEFAULT_INT_BW}-bit word and wrap"
            )
        elif t.kind == "mixed-enum":
            problems.append(f"`{name}`'s type mixes names and numbers")
    clashes = sorted(set(d.enum_values) & (declared | ivars | set(d.define_map)))
    for name in clashes:
        problems.append(f"`{name}` is both an enum value and a variable or DEFINE")

    if problems:
        raise SMVRefused(
            "the model cannot be read as NuSMV reads it:\n  " + "\n  ".join(problems)
        )


# Single-child nodes that are operators, not wrappers: `_peel` stops at them.
_UNARY_OPS = frozenset({"next_expr", "neg", "abs_call"})


def _peel(tree):
    """Peel single-child wrappers down to a leaf or a unary operator."""
    while (isinstance(tree, Tree) and len(tree.children) == 1
           and tree.data not in _UNARY_OPS):
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
# freely; the tests don't depend on the specific value. Strict parsing lowers a
# model without words to `Int` instead (`_build_module`).
_DEFAULT_INT_BW = 32
_SCALAR = [1, 1]


def _resolve_type(tree: Tree, enum_values: dict[str, int]) -> _SMVType:
    if isinstance(tree, Tree):
        match tree.data:
            case "type_bool":
                return _SMVType(Bool(_SCALAR), signed=False, kind="boolean")
            case "type_int" | "range_type":
                return _SMVType(Word(_DEFAULT_INT_BW, _SCALAR), signed=True, kind="integer")
            case "type_spec":
                return _resolve_type(tree.children[0], enum_values)
            case "word_type":
                sign_tok = str(tree.children[0])
                width = int(str(tree.children[1]))
                return _SMVType(Word(width, _SCALAR), signed=(sign_tok == "signed"))
            case "enum_type":
                names = numbers = 0
                for code, ev in enumerate(_typed_children(tree, "enum_value")):
                    name = _extract_name(ev.children[0])
                    if name is not None:
                        names += 1
                        enum_values.setdefault(name, code)
                    else:
                        numbers += 1
                kind = "enum" if not numbers else "integer" if not names else "mixed-enum"
                return _SMVType(Word(_DEFAULT_INT_BW, _SCALAR), signed=False, kind=kind)
    # Token fallback
    text = str(tree).strip()
    if text == "boolean":
        return _SMVType(Bool(_SCALAR), signed=False, kind="boolean")
    if text == "integer":
        return _SMVType(Word(_DEFAULT_INT_BW, _SCALAR), signed=True, kind="integer")
    raise ValueError(f"unsupported type: {tree}")


# ---------------------------------------------------------------------------
# 6. Lowerer
# ---------------------------------------------------------------------------

# Table-driven binary operator dispatch, by the operator's name; `_binary_op`
# resolves it, for booleans, integers and words, at term-construction time.
_BINOPS: dict[str, dict[str, str]] = {
    "iff": {"<->": "Xnor", "xnor": "Xnor"},
    "or_expr": {"|": "Or", "or": "Or", "xor": "Xor"},
    "and_expr": {"&": "And", "and": "And"},
    "arith_expr": {"+": "Add", "-": "Sub"},
    "mod_expr": {"mod": "Mod"},
    "term_expr": {"*": "Mul", "/": "Div"},
}

# The binary operators on booleans (and bitwise on words); the rest are arithmetic.
_LOGICAL = frozenset({"And", "Or", "Xor", "Xnor"})

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


def _number(tree) -> int | None:
    """The value of *tree* if it is a bare number or minus one, else None."""
    node = _peel(tree)
    sign = 1
    while isinstance(node, Tree) and node.data == "neg":
        node, sign = _peel(node.children[0]), -sign
    if isinstance(node, Token) and node.type == "NUMBER":
        return sign * int(str(node))
    return None


def _is_literal(tree) -> bool:
    """Whether *tree* is a bare number, under unary operators or none, which takes its
    sort from its context."""
    node = _peel(tree)
    while isinstance(node, Tree) and node.data in _UNARY_OPS:
        node = _peel(node.children[0])
    return isinstance(node, Token) and node.type == "NUMBER"


def _is_true(tree) -> bool:
    node = _peel(tree)
    return isinstance(node, Tree) and node.data == "true_lit"


class _Lowerer:
    """Recursive expression-to-Term lowerer.

    A number literal has no sort of its own: it takes that of the wire it is lowered
    to, or else the sort *hint* of the operand it combines with, or else an integer.

    With *strict*, *theory* is the model's reading: "word", or "int" for a model
    without words, where numbers and enum codes are integers and only linear
    integer arithmetic is read. An operand of the wrong sort is refused, and a name
    reads what NuSMV reads: the init block reads every variable's initial value and
    no input; the update block reads a variable's latched value and the input of its
    own round, and under `next()` (`at_next`) a variable's next value and no input.
    """

    def __init__(
            self,
            name_map: dict[str, Var],
            defines: dict[str, Tree],
            enum_values: dict[str, int],
            var_types: dict[str, _SMVType],
            is_init: bool = False,
            *,
            strict: bool = False,
            ivars: frozenset[str] = frozenset(),
            theory: str = "word",
    ):
        self.name_map = name_map
        self.defines = defines
        self.enum_values = enum_values
        self.var_types = var_types
        self.is_init = is_init
        self.strict = strict
        self.ivars = ivars
        self.theory = theory
        self.at_next = False
        self.context = ""
        self.terms: list[Term] = []
        self._expanding: set[str] = set()
        # Keyed by the wire, not its `id()`: a temporary's Python object is freed
        # once its `Term` holds a copy, and the next wire allocated at that address
        # took its entry. Two objects for one wire (`X(v)` twice) compare and hash
        # equal.
        self._wire_types: dict[Wire, _SMVType] = {}
        # Integer constants, by the wire that carries them, and the term that writes
        # it: a product with one is a scaling (the "int" theory).
        self._int_consts: dict[Wire, tuple[int, Term]] = {}

    def refuse(self, why: str, node=None) -> SMVRefused:
        where = f" (in {self.context})" if self.context else ""
        return SMVRefused(f"{_at(node) if node is not None else ''}{why}{where}")

    # --- public entry -------------------------------------------------------

    def lower(self, tree, target: Wire | None = None, hint: Real | Int | Bool | Word | Zero | None = None) -> Wire:
        """Lower *tree* into Term(s), return the wire carrying the result."""
        # Token leaf (from Lark inlining)
        if isinstance(tree, Token):
            if tree.type == "IDENT":
                return self._lower_name(str(tree), target, hint)
            if tree.type == "NUMBER":
                return self._lower_number(tree, target, hint)
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
            return self._lower_number(tree.children[0], target, hint)
        if name in ("true_lit", "false_lit"):
            return self._lower_bool_lit(name == "true_lit", target)

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

    # --- terms and sorts ----------------------------------------------------

    def _term(self, op, write: list[Wire], read: list[Wire], node=None) -> None:
        """Append the term `op` writing *write* from *read*: when strict, wires of the
        wrong sorts for `op` are refused."""
        try:
            term = Term(op, write, read)
        except Exception as e:
            if not self.strict:
                raise
            raise self.refuse(f"an operand of the wrong sort: {e}", node) from e
        self.terms.append(term)

    def _emit(self, op, sort, inputs, target=None, node=None) -> Wire:
        if target is not None:
            self._need_sort(sort, target.dtype, node)
        w = self._fresh(target, sort)
        self._term(op, [w], inputs, node)
        return w

    def _need_sort(self, sort, needed, node=None) -> None:
        """When strict, refuse a value of `sort` where one of `needed` is needed."""
        if self.strict and sort != needed:
            raise self.refuse(f"{_sort_name(sort)} where {_sort_name(needed)} is needed", node)

    def _need(self, w: Wire, sort, node=None) -> None:
        self._need_sort(w.dtype, sort, node)

    def _no(self, what: str, sort, node=None) -> Exception:
        """The error for the operation `what` on a value of `sort`: when strict, a
        refusal."""
        if self.strict:
            return self.refuse(f"no `{what}` on {_sort_name(sort)}", node)
        return TypeError(f"no {what} on {sort}")

    # --- operations, for booleans, integers and words -----------------------

    def _binary_op(self, name: str, left: Wire, right: Wire, target=None, node=None) -> Wire:
        if self.strict:
            # Both operands of one sort: a word's if either is one, else a boolean
            # for a logical operator and an integer for an arithmetic one.
            sort = next((w.dtype for w in (left, right) if isinstance(w.dtype, Word)),
                        Bool(_SCALAR) if name in _LOGICAL else Int(_SCALAR))
            self._need(left, sort, node)
            self._need(right, sort, node)
        sort = left.dtype
        match sort:
            case Bool(s) if name in _LOGICAL:
                if name == "Xnor":  # x <-> y is x = y
                    return self._emit(tensor.bool.Eq(s), sort, [left, right], target, node)
                return self._emit(getattr(tensor.bool, name)(s), sort, [left, right], target, node)
            case Int(s) if name not in _LOGICAL:
                if self.theory == "int" and name not in ("Add", "Sub"):
                    return self._linear_product(name, left, right, target, node)
                return self._emit(getattr(tensor.int, name)(s), sort, [left, right], target, node)
            case Word(w, s):
                signed = self._is_signed(left) or self._is_signed(right)
                if name == "Div":
                    op = tensor.word.signed.Div(w, s) if signed else tensor.word.unsigned.Div(w, s)
                elif name == "Mod":
                    op = tensor.word.signed.Rem(w, s) if signed else tensor.word.unsigned.Rem(w, s)
                else:
                    op = getattr(tensor.word, name)(w, s)
                return self._emit(op, sort, [left, right], target, node)
        raise self._no(str(node) if isinstance(node, Token) else name, sort, node)

    def _linear_product(self, name: str, left: Wire, right: Wire, target, node) -> Wire:
        """`left * right` with one of them a constant, as a scaling; any other product,
        `/` and `mod` are refused (the "int" theory)."""
        if name == "Mul":
            for c, x in ((left, right), (right, left)):
                if c in self._int_consts:
                    value, term = self._int_consts.pop(c)
                    self.terms = [t for t in self.terms if t is not term]
                    op = tensor.int.Scale(torch.tensor([[value]], dtype=torch.int64))
                    return self._emit(op, Int(_SCALAR), [x], target, node)
            raise self.refuse("`*` of two non-constants is not linear integer arithmetic", node)
        raise self.refuse(f"`{node}` is not linear integer arithmetic", node)

    # --- binary ops ---------------------------------------------------------

    def _lower_binop(self, tree: Tree, op_map: dict, target: Wire | None, hint=None) -> Wire:
        children = tree.children
        if len(children) == 1:  # an operand alone, with no operator
            return self.lower(children[0], target, hint)
        left, rhs = self._lower_pair(children[0], children[2], hint)
        i = 1
        while True:
            op_tok = children[i]
            out = target if i + 2 >= len(children) else None
            w = self._binary_op(op_map[str(op_tok)], left, rhs, out, op_tok)
            self._propagate_type(w, left, rhs)
            left = w
            i += 2
            if i >= len(children):
                break
            rhs = self.lower(children[i + 1], hint=left.dtype)
        return self._enforce(left, target)

    # --- comparison ops (sign-aware): write booleans ------------------------

    def _compare(self, op_tok, left: Wire, right: Wire, target=None) -> Wire:
        out = Bool(_SCALAR)
        op_name = str(op_tok)
        self._need(right, left.dtype, op_tok)
        match left.dtype:
            case Bool(s) if op_name in _CMP_FIXED:
                op = getattr(tensor.bool, _CMP_FIXED[op_name])(s)
                return self._emit(op, out, [left, right], target, op_tok)
            case Int(s):
                op = getattr(tensor.int, _CMP_FIXED.get(op_name) or _CMP_ORDERED[op_name])(s)
                return self._emit(op, out, [left, right], target, op_tok)
            case Word(w, s):
                if op_name in _CMP_FIXED:
                    op = getattr(tensor.word, _CMP_FIXED[op_name])(w, s)
                elif self._is_signed(left) or self._is_signed(right):
                    op = getattr(tensor.word.signed, _CMP_ORDERED[op_name])(w, s)
                else:
                    op = getattr(tensor.word.unsigned, _CMP_ORDERED[op_name])(w, s)
                return self._emit(op, out, [left, right], target, op_tok)
        raise self._no(op_name, left.dtype, op_tok)

    def _lower_cmp(self, tree: Tree, target: Wire | None, hint=None) -> Wire:
        children = tree.children
        if len(children) == 1:  # an operand alone, with no comparison
            return self.lower(children[0], target, hint)
        left, rhs = self._lower_pair(children[0], children[2])
        i = 1
        while True:
            out = target if i + 2 >= len(children) else None
            left = self._compare(children[i], left, rhs, out)
            i += 2
            if i >= len(children):
                break
            rhs = self.lower(children[i + 1], hint=left.dtype)
        return self._enforce(left, target)

    # --- n-ary (unary / implies / ternary) ----------------------------------

    def _lower_nary(self, tree: Tree, op_name: str, target: Wire | None, hint=None) -> Wire:
        trees = [c for c in tree.children if not (isinstance(c, Token) and c.type in _OP_TOKENS)]
        if op_name == "Ite":
            cond = self._as_bool(self.lower(trees[0]), trees[0])
            hint = target.dtype if target is not None else hint
            then_w, else_w = self._lower_pair(trees[1], trees[2], hint)
            return self._ite(cond, then_w, else_w, target, tree)
        if op_name == "Implies":  # x -> y is (not x) or y
            a, b = self._lower_pair(trees[0], trees[1], hint)
            w = self._binary_op("Or", self._not(a, tree), b, target, tree)
            self._propagate_type(w, a, b)
            return w
        if op_name == "Neg" and self.theory == "int" and _number(tree) is not None:
            return self._int_const(_number(tree), target, tree)
        x = self.lower(trees[0], hint=target.dtype if target is not None else hint)
        match op_name, x.dtype:
            case "Not", _:
                w = self._not(x, tree)
            case "Neg", Word(width, s):
                w = self._emit(tensor.word.Neg(width, s), x.dtype, [x], node=tree)
            case "Neg", Int(s):
                w = self._emit(tensor.int.Neg(s), x.dtype, [x], node=tree)
            case "Abs", Word(width, s):
                w = self._emit(tensor.word.signed.Abs(width, s), x.dtype, [x], node=tree)
            case "Abs", Int(s):
                w = self._emit(tensor.int.Abs(s), x.dtype, [x], node=tree)
            case _:
                raise self._no({"Neg": "-", "Abs": "abs"}[op_name], x.dtype, tree)
        self._propagate_type(w, x)
        return self._enforce(w, target)

    def _not(self, x: Wire, node=None) -> Wire:
        match x.dtype:
            case Bool(s):
                w = self._emit(tensor.bool.Not(s), x.dtype, [x], node=node)
            case Word(width, s):
                w = self._emit(tensor.word.Not(width, s), x.dtype, [x], node=node)
            case _:
                raise self._no("!", x.dtype, node)
        self._propagate_type(w, x)
        return w

    def _ite(self, cond: Wire, then_w: Wire, else_w: Wire, target=None, node=None) -> Wire:
        self._need(else_w, then_w.dtype, node)
        match then_w.dtype:
            case Bool(s):
                op = tensor.bool.Ite(s)
            case Int(s):
                op = tensor.int.Ite(s)
            case Word(w, s):
                op = tensor.word.Ite(w, s)
            case other:
                raise TypeError(f"no Ite on {other}")
        w = self._emit(op, then_w.dtype, [cond, then_w, else_w], target, node)
        # What the condition is says nothing about the value's sign.
        self._propagate_type(w, then_w, else_w)
        return self._enforce(w, target)

    def _as_bool(self, w: Wire, node=None) -> Wire:
        """A condition as a boolean: a 1-bit word `x` as `bool(x)` (when strict, only a
        boolean)."""
        self._need(w, Bool(_SCALAR), node)
        match w.dtype:
            case Bool(_):
                return w
            case Word(1, _):
                return self._bool_of_word1(w)
        raise TypeError(f"a condition must be a boolean, got {w.dtype}")

    def _bool_of_word1(self, w: Wire) -> Wire:
        """`bool(x)` of a word `x`: its lowest bit, `Extract(0, 0, x) = 1`."""
        width, s = _bw(w.dtype), list(w.dtype.shape)
        bit = self._emit(tensor.word.Extract(0, 0, width, s), Word(1, s), [w])
        one = self._emit(_const_op(1, Word(1, s)), Word(1, s), [])
        return self._emit(tensor.word.Eq(1, s), Bool(s), [bit, one])

    # --- builtin_call -------------------------------------------------------

    def _lower_builtin(self, tree: Tree, target: Wire | None, hint=None) -> Wire:
        fn_name = str(tree.children[0])
        exprs = [
            c
            for c in tree.children[1:]
            if not (isinstance(c, Token) and c.type == "BUILTIN")
        ]
        arg_w = self.lower(exprs[0])
        if self.strict:
            word = isinstance(arg_w.dtype, Word)
            if fn_name == "bool" and word and _bw(arg_w.dtype) != 1:
                raise self._no("bool()", arg_w.dtype, tree)
            if fn_name == "word1":
                self._need(arg_w, Bool(_SCALAR), tree)
            if fn_name in ("signed", "unsigned", "extend") and not word:
                raise self._no(f"{fn_name}()", arg_w.dtype, tree)

        if fn_name == "bool":
            w = self._as_bool(arg_w) if isinstance(arg_w.dtype, Bool) else self._bool_of_word1(arg_w)
        elif fn_name == "word1":
            # a boolean as a 1-bit word: if b then 1 else 0
            one = self._emit(_const_op(1, Word(1, _SCALAR)), Word(1, _SCALAR), [])
            zero = self._emit(_const_op(0, Word(1, _SCALAR)), Word(1, _SCALAR), [])
            w = self._emit(tensor.word.Ite(1, _SCALAR), Word(1, _SCALAR), [self._as_bool(arg_w), one, zero])
        elif fn_name in ("signed", "unsigned"):
            w = self._emit(Skip(arg_w.dtype), arg_w.dtype, [arg_w])
            self._wire_types[w] = _SMVType(w.dtype, signed=(fn_name == "signed"))
        elif fn_name == "extend":
            extra = int(str(_peel(exprs[1])))
            width = _bw(arg_w.dtype)
            op = (tensor.word.signed.Extend if self._is_signed(arg_w) else tensor.word.unsigned.Extend)(extra, width, _SCALAR)
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
            # The last branch is taken whatever its condition: only right when that
            # condition is TRUE (NuSMV has no value otherwise).
            if self.strict and not _is_true(cond_tree):
                raise self.refuse(
                    "the last branch of a `case` must have condition TRUE", cond_tree
                )
            return self.lower(val_tree, target, hint)
        cond = self._as_bool(self.lower(cond_tree), cond_tree)
        then_w = self.lower(val_tree, hint=hint)
        else_w = self._lower_case_rec(branches, idx + 1, None, then_w.dtype)
        return self._ite(cond, then_w, else_w, target, cond_tree)

    # --- next(expr) ---------------------------------------------------------

    def _lower_next(self, tree: Tree, target: Wire | None, hint=None) -> Wire:
        inner = tree.children[0]
        if self.strict:
            if self.is_init or self.at_next:
                raise self.refuse("`next()` is not allowed here", tree)
            with self.next_values():
                return self.lower(inner, target, hint)
        name = _extract_name(_peel(inner))
        if name and name in self.name_map:
            next_w = X(self.name_map[name])
            if name in self.var_types:
                self._wire_types[next_w] = self.var_types[name]
            return self._enforce(next_w, target)
        return self.lower(inner, target, hint)

    @contextmanager
    def next_values(self):
        """Names read their next values while inside (strict only)."""
        self.at_next = True
        try:
            yield
        finally:
            self.at_next = False

    # --- numbers and enum codes ---------------------------------------------

    def _lower_number(self, token, target: Wire | None, hint=None) -> Wire:
        if self.theory == "int":
            return self._int_const(int(str(token)), target, token)
        if self.strict:
            # Every integer type is refused in a model with words, so a bare number
            # would be a word of whatever width it lands in.
            raise self.refuse(
                f"the number {token} is not a word constant (write e.g. 0ud8_{token})",
                token,
            )
        return self._emit_const(int(str(token)), target, hint)

    def _lower_code(self, code: int, target: Wire | None, hint=None) -> Wire:
        """An enum value's code: an integer in the "int" theory, else a word of the
        default width, the sort of an enum variable (leniently, of the hint)."""
        if self.theory == "int":
            return self._int_const(code, target)
        sort = Word(_DEFAULT_INT_BW, _SCALAR)
        if self.strict:
            return self._emit(_const_op(code, sort), sort, [], target)
        return self._emit_const(code, target, hint if hint is not None else sort)

    def _int_const(self, value: int, target: Wire | None, node=None) -> Wire:
        w = self._emit(_const_op(value, Int(_SCALAR)), Int(_SCALAR), [], target, node)
        self._int_consts[w] = (value, self.terms[-1])
        return w

    def _lower_bool_lit(self, value: bool, target: Wire | None) -> Wire:
        if self.strict:
            return self._emit(_const_op(value, Bool(_SCALAR)), Bool(_SCALAR), [], target)
        return self._emit_const(int(value), target, Bool(_SCALAR))

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
        self._wire_types[w] = _SMVType(dtype, signed=signed)
        self._term(_const_op(value, dtype), [w], [], tree)
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
            return self._lower_code(self.enum_values[name], target, hint)
        # Wire lookup
        if name in self.name_map:
            var = self.name_map[name]
            if self.strict:
                w = self._strict_read(name, var)
            else:
                w = X(var) if self.is_init else var
            if name in self.var_types:
                self._wire_types[w] = self.var_types[name]
            return self._enforce(w, target)
        if self.strict:
            raise self.refuse(f"`{name}` is not declared")
        # Unknown name — emit zero with warning
        warnings.warn(f"undeclared variable '{name}', defaulting to 0")
        return self._emit_const(0, target, hint)

    def _strict_read(self, name: str, var: Var) -> Wire:
        if name in self.ivars:
            if self.is_init or self.at_next:
                raise self.refuse(
                    f"input variable `{name}` is read where NuSMV allows no input (an "
                    "initial value, an INVAR or `x :=`, or under `next()`)"
                )
            return X(var)
        return X(var) if (self.is_init or self.at_next) else var

    # --- helpers ------------------------------------------------------------

    def _smv_type(self, wire: Wire) -> _SMVType | None:
        return self._wire_types.get(wire)

    def _is_signed(self, wire: Wire) -> bool:
        t = self._smv_type(wire)
        return t.signed if t is not None else False

    def _propagate_type(self, result: Wire, *inputs: Wire) -> None:
        """Set result's _SMVType to signed if any input is signed."""
        signed = any(self._is_signed(w) for w in inputs)
        if signed or any(self._smv_type(w) is not None for w in inputs):
            self._wire_types[result] = _SMVType(result.dtype, signed=signed)

    def _fresh(self, target: Wire | None, dtype=None) -> Wire:
        if target is not None:
            return target
        return Wire(dtype if dtype is not None else Word(_DEFAULT_INT_BW, _SCALAR))

    def _emit_const(self, value: int, target: Wire | None, hint=None) -> Wire:
        """A number literal, of the sort of the target, or else of the hint, or else an
        integer, as SMV's integer constants are."""
        w = self._fresh(target, hint if hint is not None else Int(_SCALAR))
        self._term(_const_op(value, w.dtype), [w], [])
        return self._enforce(w, target)

    def _enforce(self, wire: Wire, target: Wire | None) -> Wire:
        if target is not None and target != wire:
            self._need(wire, target.dtype)
            self._term(Skip(wire.dtype), [target], [wire])
            t = self._smv_type(wire)
            if t is not None:
                self._wire_types[target] = t
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
            if not isinstance(base_w.dtype, Word):
                raise self._no("[:]", base_w.dtype, bs)
            high, low = int(str(bs.children[0])), int(str(bs.children[1]))
            op = tensor.word.Extract(high, low, _bw(base_w.dtype), _SCALAR)
            w = self._emit(op, Word(high - low + 1, _SCALAR), [base_w], target, bs)
            return self._enforce(w, target)
        return self._enforce(base_w, target)


# ---------------------------------------------------------------------------
# 7. Pass 2 — build Module
# ---------------------------------------------------------------------------


def _build_module(
        d: _Decls,
        overrides: dict[str, Var],
        *,
        strict: bool = False,
        theory: str = "word",
        specs: bool = False,
) -> tuple[Module, dict[str, Var], list[Spec]]:
    # Prepend frozen vars before regular state vars
    all_var_decls = list(d.frozen_decls) + list(d.var_decls)

    # Combine: state vars first, then ivars
    all_decls = all_var_decls + list(d.ivar_decls)
    # Create wire pairs and type map
    name_map: dict[str, Var] = {}
    var_types: dict[str, _SMVType] = {}
    for name, smv_type in all_decls:
        if theory == "int":
            sort = Bool(_SCALAR) if smv_type.kind == "boolean" else Int(_SCALAR)
            smv_type = replace(smv_type, dtype=sort, signed=False)
        var_types[name] = smv_type
        name_map[name] = (
            overrides[name]
            if name in overrides
            else Var(smv_type.dtype)
        )

    lowerer = partial(
        _Lowerer, name_map, d.define_map, d.enum_values, var_types, strict=strict,
        ivars=frozenset(n for n, _ in d.ivar_decls), theory=theory)
    init_low = lowerer(is_init=True)
    update_low = lowerer(is_init=False)

    for name, smv_type in all_var_decls:
        var = name_map[name]

        if strict:
            _lower_strict(d, name, var, init_low, update_low)
            continue

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

    init_terms, update_terms = init_low.terms, update_low.terms
    if strict:
        init_terms = _in_dataflow_order(init_terms, "the initial state")
        update_terms = _in_dataflow_order(update_terms, "the next state")
    module = Module.sequential(obs_pairs, init_terms, update_terms)
    stated = []
    if specs:
        stated = [_lower_spec(keyword, formula, lowerer(is_init=True))
                  for keyword, formula in d.specs]
    return module, name_map, stated


def _lower_strict(d: _Decls, name: str, var: Var,
                  init_low: _Lowerer, update_low: _Lowerer) -> None:
    """`name`'s initial and next value, which `_check_strict` has made sure the text
    gives (a FROZENVAR's next value is its own)."""
    invar = d.invar_assigns.get(name)
    if invar is not None:
        # In every state, so the next value is the rhs at the next state.
        init_low.context = update_low.context = f"`{name}` in every state"
        init_low.lower(invar, X(var))
        with update_low.next_values():
            update_low.lower(invar, X(var))
        return
    init_low.context = f"init({name})"
    init_low.lower(d.init_assigns[name], X(var))
    update_low.context = f"next({name})"
    next_expr = d.next_assigns.get(name)
    if next_expr is not None:
        update_low.lower(next_expr, X(var))
    else:
        update_low.terms.append(Term(Skip(var.dtype), [X(var)], [var]))


def _lower_spec(keyword: str, formula, low: _Lowerer) -> Spec:
    """`formula`'s claim, lowered by `low`: an init-block reader, so a variable reads
    its value after the round and an input is refused."""
    where = f"{_at(formula)}{keyword}"
    ops: list[str] = []
    body = formula
    if keyword == "LTLSPEC":
        while isinstance(body, Tree) and body.data == "ltl_op":
            ops.append(str(body.children[0]))
            body = body.children[1]
    kind = ("safety" if keyword == "INVARSPEC"
            else {("G",): "safety", ("G", "F"): "buchi"}.get(tuple(ops)))
    nested = isinstance(body, Tree) and any(
        t.data == "ltl_op" for t in body.iter_subtrees())
    if kind is None or nested:
        raise SMVRefused(
            f"{where}: only `INVARSPEC p`, `LTLSPEC G p` and `LTLSPEC G F p` are read "
            "as a claim"
        )
    low.context = where
    wire = low.lower(body)
    if not isinstance(wire.dtype, Bool):
        raise SMVRefused(f"{where}: the property is not a boolean")
    return Spec(kind, where, wire, tuple(low.terms))


def _in_dataflow_order(terms: list[Term], block: str) -> list[Term]:
    """`terms` with each wire written before it is read.

    A strict block reads next values (an INVAR, `next()`), so one variable's term may
    read a wire another variable's term writes later in declaration order."""
    writer = {w: i for i, t in enumerate(terms) for w in t.write}
    deps = [{writer[w] for w in t.read if w in writer} for t in terms]
    order: list[Term] = []
    state = [0] * len(terms)  # 0 unvisited, 1 on the stack, 2 placed
    for root in range(len(terms)):
        if state[root]:
            continue
        state[root] = 1
        stack = [(root, iter(sorted(deps[root])))]
        while stack:
            i, it = stack[-1]
            j = next(it, None)
            if j is None:
                stack.pop()
                state[i] = 2
                order.append(terms[i])
            elif state[j] == 1:
                raise SMVRefused(
                    f"{block} depends on itself (a cycle through `next()` or INVAR)"
                )
            elif state[j] == 0:
                state[j] = 1
                stack.append((j, iter(sorted(deps[j]))))
    return order
