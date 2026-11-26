"""One row per op, one column per backend: the op table.

An op is one class of one theory: the namespace of `reactor.tensor` it is in
(`theory_of`: `"int"` for `tensor.int.Ne([1, 1])`) and the name of its class
(`itype_name`: `"Ne"`). Three backends turn that pair into something they can
emit:

| column   | backend                                | a cell holds                    |
|----------|----------------------------------------|---------------------------------|
| `mat`    | `codegen.terms.translate_terms_lean`   | args -> Lean expr (on `Matrix`) |
| `scalar` | `codegen.terms.translate_terms_scalar` | args -> Lean expr (bare)        |
| `smt`    | `smt.encode.translate_terms`           | `SmtOp` -> `cvc5.Term`          |

These were six dicts and an if/elif chain — one per backend, doubled for the
bit-vector forms — with nothing relating them. A test can see a *dead* key (one no
theory defines) but not a *missing* one, so a variant the SMT encoder
handled and no Lean emitter did went unnoticed: every unsigned and signed
bit-vector comparison passed cvc5's check of the obligations and then died at
project generation in `No Lean expression mapping for: ULt`.

A row is one op of one or more theories, and it must say something about
every column. A cell is either an emitter or one of

* `ViaMat(...)` — scalar column only: the matrix emitter, with scalar reads
  lifted to 1x1 and `0 0` projected back off the result;
* `Inline(...)` — the backend emits this op from a branch of its own,
  because the argument strings are not enough (constants need the registry,
  `Argmax`, `Linear` and `Scale` need shapes and the baked tensors);
* `Unsupported(...)` — no emitter, and the reason it is missing rather than
  forgotten. That reason is what the caller is told.

A row's two Lean columns are both emitters or both refusals, so that a
module no encoding can write is refused before any is generated
(`codegen.terms.lean_gaps`), not half way through a project.

`python -m reactor_lean.lowering.ops` prints the matrix, gaps and all.
`tests/test_lean_ops.py` checks it against `reactor.tensor` in both
directions, so an op added to a theory fails the suite until it has a row,
and a row no theory backs fails it too.

This module never imports cvc5, on purpose: `codegen/terms.py` imports
*it*, and the Lean path has to keep working where
cvc5 is absent. The SMT emitters reach cvc5 only through the `SmtOp` context
`smt.encode` hands them — `op.K` is `cvc5.Kind`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Union

from reactor import Real

from ..core.errors import Refused
from .sorts import THEORIES, itype_name, mat_mk, theory_of

if TYPE_CHECKING:  # pragma: no cover - typing only, cvc5 is not imported here
    import cvc5

    from ..smt.encode import SmtOp


# ══════════════════════════════════════════════════════════════════════
#  Cells
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Unsupported:
    """No emitter for this (op, backend), and why not.

    The reason reaches the user: it is appended to the error the backend
    raises, so "no mapping" stops being the whole message.
    """

    reason: str


@dataclass(frozen=True)
class Inline:
    """The backend emits this op from a dedicated branch, not from a table.

    `site` names that branch. These ops need more than the argument
    strings — the constant registry, the read wire's shape, the tensors
    baked into the op — so a table cell could not hold them.
    """

    site: str


@dataclass(frozen=True)
class ViaMat:
    """Scalar column only: there is no scalar form, the matrix one serves.

    `translate_terms_scalar` lifts scalar-bound reads to `Matrix.mk fun _ _ => x`
    and projects `0 0` off the result, which is right whenever the matrix
    form is elementwise or folds to a 1x1.
    """

    note: str


MatCell = Union[Callable[[list[str]], str], Inline, Unsupported]
ScalarCell = Union[Callable[[list[str]], str], ViaMat, Inline, Unsupported]
SmtCell = Union[Callable[["SmtOp"], "cvc5.Term"], Unsupported]




# ══════════════════════════════════════════════════════════════════════
#  Rows
# ══════════════════════════════════════════════════════════════════════

NUM = ("int", "real")
WORD = ("word",)
SIGNED = ("word.signed",)
UNSIGNED = ("word.unsigned",)
STRUCTURAL = ("",)


@dataclass(frozen=True)
class Op:
    """One op, as every backend sees it.

    `entrywise`: the Lean cells are forms of 1×1 operands, read at `0 0`. The
    op of a larger shape is its scalar form applied entry by entry
    (`codegen.terms`, `entrywise_shape`).
    """

    name: str
    theories: tuple[str, ...]
    mat: MatCell
    scalar: ScalarCell
    smt: SmtCell
    entrywise: bool = False


# ══════════════════════════════════════════════════════════════════════
#  SMT emitters
#
#  Everything cvc5 arrives through `op` (see `smt.encode.SmtOp`), including
#  the `Kind` enum as `op.K`.
# ══════════════════════════════════════════════════════════════════════


def _smt_id(op: SmtOp) -> cvc5.Term:
    return op.args[0]


def _smt_const(op: SmtOp) -> cvc5.Term:
    """`Const` of every theory, scalar or per element; the element type is
    the written wire's."""
    payload = op.term.itype.tensor
    if payload.numel() > 1:
        return op.tensor(op.write, payload)
    v = op.const(op.write.dtype, payload.item())
    return op.pack([v] * op.out_shape.total)


def _smt_not(op: SmtOp) -> cvc5.Term:
    kind = op.K.BITVECTOR_NOT if op.is_bv else op.K.NOT
    return op.elementwise(op.unop(kind), op.args[0])


def _smt_and(op: SmtOp) -> cvc5.Term:
    kind = op.K.BITVECTOR_AND if op.is_bv else op.K.AND
    return op.elementwise(op.binop(kind), *op.args)


def _smt_or(op: SmtOp) -> cvc5.Term:
    kind = op.K.BITVECTOR_OR if op.is_bv else op.K.OR
    return op.elementwise(op.binop(kind), *op.args)


def _smt_xor(op: SmtOp) -> cvc5.Term:
    kind = op.K.BITVECTOR_XOR if op.is_bv else op.K.XOR
    return op.elementwise(op.binop(kind), *op.args)


def _smt_negated(kind_name: str):
    """`Nand`, `Nor` and `Xnor`: the complement of `bvand`, `bvor`, `bvxor`."""

    def emit(op: SmtOp) -> cvc5.Term:
        kind = getattr(op.K, kind_name)
        return op.elementwise(
            lambda a, b: op.mk(op.K.BITVECTOR_NOT, op.mk(kind, a, b)), *op.args
        )

    return emit


def _smt_implies(op: SmtOp) -> cvc5.Term:
    return op.elementwise(op.binop(op.K.IMPLIES), *op.args)


def _smt_ite(op: SmtOp) -> cvc5.Term:
    # one of the whole branches, by the scalar of the 1×1 Bool condition
    return op.mk(op.K.ITE, op.scalar(0), op.args[1], op.args[2])


def _smt_all(op: SmtOp) -> cvc5.Term:
    """The conjunction of the entries, as a 1×1 matrix."""
    shape = op.in_shapes[0]
    entries = [op.scalar(0, i, j) for i in range(shape.m) for j in range(shape.n)]
    return op.pack([entries[0] if len(entries) == 1 else op.mk(op.K.AND, *entries)])


def _smt_add(op: SmtOp) -> cvc5.Term:
    kind = op.K.BITVECTOR_ADD if op.is_bv else op.K.ADD
    return op.elementwise(op.binop(kind), *op.args)


def _smt_sub(op: SmtOp) -> cvc5.Term:
    kind = op.K.BITVECTOR_SUB if op.is_bv else op.K.SUB
    return op.elementwise(op.binop(kind), *op.args)


def _smt_mul(op: SmtOp) -> cvc5.Term:
    kind = op.K.BITVECTOR_MULT if op.is_bv else op.K.MULT
    return op.elementwise(op.binop(kind), *op.args)


def _smt_int_divmod(op: SmtOp) -> cvc5.Term:
    """Euclidean, as SMT-LIB's `div`/`mod` are; SMT-LIB leaves the zero
    divisor open and the theory has Lean's answer there (0, and the
    dividend), so that case is spelled out. Not the `_TOTAL` kinds: they
    print as cvc5's own syntax, and these terms are written out as SMT-LIB
    for other solvers too."""
    div = op.name == "Div"
    kind = op.K.INTS_DIVISION if div else op.K.INTS_MODULUS
    zero = op.integer(0)

    def one(a, b):
        return op.mk(op.K.ITE, op.mk(op.K.EQUAL, b, zero),
                     zero if div else a, op.mk(kind, a, b))

    return op.elementwise(one, *op.args)


def _smt_real_div(op: SmtOp) -> cvc5.Term:
    """SMT-LIB leaves `x / 0` open, and Lean's is 0, so that case is spelled
    out as `_smt_int_divmod` does."""
    zero = op.const(op.write.dtype, 0)

    def one(a, b):
        return op.mk(op.K.ITE, op.mk(op.K.EQUAL, b, zero), zero,
                     op.mk(op.K.DIVISION, a, b))

    return op.elementwise(one, *op.args)


def _smt_neg(op: SmtOp) -> cvc5.Term:
    kind = op.K.BITVECTOR_NEG if op.is_bv else op.K.NEG
    return op.elementwise(op.unop(kind), op.args[0])


def _smt_abs(op: SmtOp) -> cvc5.Term:
    if op.is_bv:
        zero = op.bitvector(op.write.dtype.width, 0)
        return op.elementwise(
            lambda a: op.mk(op.K.ITE, op.mk(op.K.BITVECTOR_SLT, a, zero),
                            op.mk(op.K.BITVECTOR_NEG, a), a),
            op.args[0],
        )
    zero = op.const(op.write.dtype, 0)
    return op.elementwise(
        lambda a: op.mk(op.K.ITE, op.mk(op.K.GEQ, a, zero), a, op.mk(op.K.NEG, a)),
        op.args[0],
    )


def _smt_real_unary(kind_name: str):
    """A transcendental function that is one cvc5 `Kind`, pointwise; cvc5
    leaves it open where it is undefined (`sqrt` of a negative, `asin`
    outside `[-1, 1]`), as the theory does."""

    def emit(op: SmtOp) -> cvc5.Term:
        return op.elementwise(op.unop(getattr(op.K, kind_name)), op.args[0])

    return emit


def _smt_hyperbolic(op: SmtOp) -> cvc5.Term:
    """`Sinh`, `Cosh` and `Tanh`, which cvc5 has no kinds for, through `exp`:
    `(eˣ ∓ e⁻ˣ) / 2` and their quotient."""
    two = op.const(op.write.dtype, 2)

    def one(a):
        ep = op.mk(op.K.EXPONENTIAL, a)
        em = op.mk(op.K.EXPONENTIAL, op.mk(op.K.NEG, a))
        if op.name == "Sinh":
            return op.mk(op.K.DIVISION, op.mk(op.K.SUB, ep, em), two)
        if op.name == "Cosh":
            return op.mk(op.K.DIVISION, op.mk(op.K.ADD, ep, em), two)
        return op.mk(op.K.DIVISION, op.mk(op.K.SUB, ep, em), op.mk(op.K.ADD, ep, em))

    return op.elementwise(one, op.args[0])


def _smt_atan2(op: SmtOp) -> cvc5.Term:
    """`atan2(y, x)`, which cvc5 has no kind for, by quadrant from `arctan`
    and `π`; `(0, 0)`, where the theory leaves it open, gives 0, as in C."""
    zero = op.const(op.write.dtype, 0)
    two = op.const(op.write.dtype, 2)
    pi = op.tm.mkPi()

    def one(y, x):
        r = op.mk(op.K.ARCTANGENT, op.mk(op.K.DIVISION, y, x))
        on_axis = op.mk(
            op.K.ITE, op.mk(op.K.GT, y, zero), op.mk(op.K.DIVISION, pi, two),
            op.mk(op.K.ITE, op.mk(op.K.LT, y, zero),
                  op.mk(op.K.NEG, op.mk(op.K.DIVISION, pi, two)), zero))
        left = op.mk(op.K.ITE, op.mk(op.K.GEQ, y, zero),
                     op.mk(op.K.ADD, r, pi), op.mk(op.K.SUB, r, pi))
        return op.mk(op.K.ITE, op.mk(op.K.GT, x, zero), r,
                     op.mk(op.K.ITE, op.mk(op.K.LT, x, zero), left, on_axis))

    return op.elementwise(one, *op.args)


def _smt_bv(kind_name: str):
    """A word operation that is one cvc5 `Kind`, pointwise."""

    def emit(op: SmtOp) -> cvc5.Term:
        return op.elementwise(op.binop(getattr(op.K, kind_name)), *op.args)

    return emit


def _smt_compare(op: SmtOp) -> cvc5.Term:
    kind = {
        "Lt": op.K.LT,
        "Le": op.K.LEQ,
        "Gt": op.K.GT,
        "Ge": op.K.GEQ,
        "Eq": op.K.EQUAL,
        "Ne": op.K.DISTINCT,
    }[op.name]
    # pointwise, as the comparisons are in every theory
    return op.elementwise(op.binop(kind), *op.args)


def _smt_word_compare(signed: bool):
    """The orderings of words, read unsigned or signed; they write Bool."""

    def emit(op: SmtOp) -> cvc5.Term:
        suffix = {"Lt": "LT", "Le": "LE", "Gt": "GT", "Ge": "GE"}[op.name]
        kind = getattr(op.K, f"BITVECTOR_{'S' if signed else 'U'}{suffix}")
        return op.elementwise(op.binop(kind), *op.args)

    return emit


def _smt_min_max(op: SmtOp) -> cvc5.Term:
    """The least (greatest) entry, folded row by row, as a 1×1 matrix."""
    return op.pack([op.reduce_flat(0, take_min=op.name == "Min")])


def _smt_matmul(op: SmtOp) -> cvc5.Term:
    return op.matmul(op.args[0], op.args[1], op.in_shapes[0], op.in_shapes[1])


def _smt_affine(op: SmtOp, A_tensor, B_tensor) -> cvc5.Term:
    """`Y = A·X + B`, with `A` ([out,in]) and `B` ([out,batch]) baked into
    the op; `args[0]` is the single read wire `X` ([in,batch])."""
    a_rows, a_cols = int(A_tensor.shape[0]), int(A_tensor.shape[1])
    A_shape = op.shape(a_rows, a_cols)
    A_data = A_tensor.reshape(a_rows, a_cols)
    A_term = op.pack_at(
        A_shape,
        [
            op.const(op.write.dtype, A_data[i, j].item())
            for i in range(a_rows)
            for j in range(a_cols)
        ],
    )
    ax = op.matmul(A_term, op.args[0], A_shape, op.in_shapes[0])
    if B_tensor.numel() == 0:
        return ax
    # `B` is of the output's shape
    out_shape = op.out_shape
    B_data = B_tensor.reshape(out_shape.m, out_shape.n)
    B_term = op.pack(
        [
            op.const(op.write.dtype, B_data[i, j].item())
            for i in range(out_shape.m)
            for j in range(out_shape.n)
        ]
    )
    return op.elementwise(op.binop(op.K.ADD), ax, B_term)


def _smt_linear(op: SmtOp) -> cvc5.Term:
    return _smt_affine(op, op.term.itype.a, op.term.itype.b)


def _smt_scale(op: SmtOp) -> cvc5.Term:
    """`Scale(c)`: the pointwise product of `c` and the operand."""
    c = op.tensor(op.write, op.term.itype.tensor)
    return op.elementwise(op.binop(op.K.MULT), c, op.args[0])


def _smt_relu(op: SmtOp) -> cvc5.Term:
    zero = op.const(op.write.dtype, 0)
    return op.elementwise(
        lambda a: op.mk(op.K.ITE, op.mk(op.K.GEQ, a, zero), a, zero), op.args[0]
    )


def _smt_argmax(op: SmtOp) -> cvc5.Term:
    idx = op.argmax_flat(0)
    # `argmax_flat` builds an Int index; a module of reals carries it on a
    # Real wire
    if isinstance(op.write.dtype, Real):
        idx = op.mk(op.K.TO_REAL, idx)
    return op.pack([idx])


def _smt_get(op: SmtOp) -> cvc5.Term:
    return op.pack([op.scalar(0, op.term.itype.row, op.term.itype.col)])


def _smt_cast(op: SmtOp) -> cvc5.Term:
    kind = {
        "ToReal": op.K.TO_REAL,
        "ToInt": op.K.TO_INTEGER,
        "UbvToInt": op.K.BITVECTOR_UBV_TO_INT,
    }[op.name]
    return op.elementwise(op.unop(kind), op.args[0])


def _smt_word_indexed(kind_name: str, indices):
    """A word operation that is a cvc5 `Kind` indexed by sizes the op holds
    (`indices(itype)`), pointwise."""

    def emit(op: SmtOp) -> cvc5.Term:
        kind = getattr(op.K, kind_name)
        idx = indices(op.term.itype)
        return op.elementwise(lambda a: op.mk_indexed(kind, idx, a), op.args[0])

    return emit


def _smt_concat(op: SmtOp) -> cvc5.Term:
    return op.elementwise(op.binop(op.K.BITVECTOR_CONCAT), *op.args)


# ══════════════════════════════════════════════════════════════════════
#  Reasons shared by several rows
# ══════════════════════════════════════════════════════════════════════

# The two Lean columns share one reason whenever an op is missing from both;
# only the matrix cell spells the reason out.
_SEE_MAT = Unsupported("see the matrix column")

_NONDET = Unsupported(
    "a nondeterministic value (`Havoc`: nullary, one written wire). Every "
    "encoding here is a function of the read wires, so there is nowhere to "
    "put the choice."
)

_ZERO = Unsupported(
    "the zero rate of change, of a `delay` block. No encoding here has "
    "continuous time."
)


def _not_asked_for(lean_form: str) -> Unsupported:
    """An op whose Lean forms would be easy, and that nothing has asked
    for yet."""
    return Unsupported(
        f"no Lean form: nothing has asked for one. {lean_form}"
    )


def _word_only_in_smt(lean_form: str) -> Unsupported:
    """A word op cvc5 encodes and no Lean backend emits.

    This is the divergence that made the table necessary: cvc5's check of
    the obligations runs on the SMT encoding and passes, and generation then
    fails -- which `codegen.terms.lean_gaps` now reports up front.
    """
    return Unsupported(
        f"encoded for SMT, never for Lean. {lean_form} A module using it "
        "passes cvc5's check of the obligations and is refused at project "
        "generation."
    )


# ══════════════════════════════════════════════════════════════════════
#  The table
# ══════════════════════════════════════════════════════════════════════

OPS: tuple[Op, ...] = (
    # ── structural ────────────────────────────────────────────────────
    Op("Skip", STRUCTURAL, mat=lambda a: a[0], scalar=lambda a: a[0],
       smt=_smt_id),
    Op("Havoc", STRUCTURAL, mat=_NONDET, scalar=_SEE_MAT, smt=_NONDET),
    Op("Zero", STRUCTURAL, mat=_ZERO, scalar=_SEE_MAT, smt=_ZERO),
    # ── constants ─────────────────────────────────────────────────────
    # The element type is the written wire's; scalar-vs-matrix is decided by
    # the wire's shape.
    Op(
        "Const",
        ("bool", "int", "real", "word"),
        mat=Inline("codegen.literals.constant_expr"),
        scalar=Inline("codegen.terms._constant_expr_scalar"),
        smt=_smt_const,
    ),
    # ── propositional logic ───────────────────────────────────────────
    # The matrix forms are of 1×1 operands, read at `0 0`; a larger
    # op is the scalar form entry by entry (`entrywise`).
    Op(
        "Not",
        ("bool",),
        mat=lambda a: mat_mk(f"!({a[0]} 0 0)"),
        scalar=lambda a: f"(!{a[0]})",
        smt=_smt_not,
        entrywise=True,
    ),
    Op(
        "And",
        ("bool",),
        mat=lambda a: mat_mk(f"({a[0]} 0 0 && {a[1]} 0 0)"),
        scalar=lambda a: f"({a[0]} && {a[1]})",
        smt=_smt_and,
        entrywise=True,
    ),
    Op(
        "Or",
        ("bool",),
        mat=lambda a: mat_mk(f"({a[0]} 0 0 || {a[1]} 0 0)"),
        scalar=lambda a: f"({a[0]} || {a[1]})",
        smt=_smt_or,
        entrywise=True,
    ),
    Op(
        "Xor",
        ("bool",),
        mat=Unsupported("no Lean form; nothing has asked for one."),
        scalar=_SEE_MAT,
        smt=_smt_xor,
    ),
    Op(
        "Implies",
        ("bool",),
        mat=lambda a: mat_mk(f"(!({a[0]} 0 0) || {a[1]} 0 0)"),
        scalar=lambda a: f"(!{a[0]} || {a[1]})",
        smt=_smt_implies,
        entrywise=True,
    ),
    # `All` reduces a matrix to the 1×1 condition of an `Ite`, which picks
    # one of two whole matrices by it, in every theory. It quantifies over
    # the pairs of indices: `∀ i j, ...` has no `Decidable` instance, which
    # instance search would have to find through `Matrix`. Ascribed, as
    # `matMin`'s type is, so that the scalar encoding can take its `0 0`.
    Op(
        "All",
        ("bool",),
        mat=lambda a: (f"({mat_mk(f'decide (∀ p : _ × _, {a[0]} p.1 p.2 = true)')}"
                       " : Matrix (Fin 1) (Fin 1) Bool)"),
        scalar=ViaMat("a reduction of a matrix; there is no bare-scalar form"),
        smt=_smt_all,
    ),
    Op(
        "Ite",
        ("bool", "int", "real", "word"),
        mat=lambda a: f"(if {a[0]} 0 0 then {a[1]} else {a[2]})",
        scalar=lambda a: f"(if {a[0]} then {a[1]} else {a[2]})",
        smt=_smt_ite,
    ),
    # ── bitwise ───────────────────────────────────────────────────────
    # BitVec's own operators, entry by entry.
    Op(
        "Not",
        WORD,
        mat=lambda a: mat_mk(f"~~~({a[0]} i j)", "i", "j"),
        scalar=lambda a: f"(~~~{a[0]})",
        smt=_smt_not,
    ),
    Op(
        "And",
        WORD,
        mat=lambda a: mat_mk(f"({a[0]} i j) &&& ({a[1]} i j)", "i", "j"),
        scalar=lambda a: f"({a[0]} &&& {a[1]})",
        smt=_smt_and,
    ),
    Op(
        "Or",
        WORD,
        mat=lambda a: mat_mk(f"({a[0]} i j) ||| ({a[1]} i j)", "i", "j"),
        scalar=lambda a: f"({a[0]} ||| {a[1]})",
        smt=_smt_or,
    ),
    Op(
        "Xor",
        WORD,
        mat=lambda a: mat_mk(f"({a[0]} i j) ^^^ ({a[1]} i j)", "i", "j"),
        scalar=lambda a: f"({a[0]} ^^^ {a[1]})",
        smt=_smt_xor,
    ),
    *(
        Op(
            name,
            WORD,
            mat=_word_only_in_smt(
                f"The matrix form would be `~~~(a {sym} b)`."
            ),
            scalar=_SEE_MAT,
            smt=_smt_negated(kind),
        )
        for name, sym, kind in (
            ("Nand", "&&&", "BITVECTOR_AND"),
            ("Nor", "|||", "BITVECTOR_OR"),
            ("Xnor", "^^^", "BITVECTOR_XOR"),
        )
    ),
    # ── arithmetic ────────────────────────────────────────────────────
    Op(
        "Add",
        ("int", "real", "word"),
        mat=lambda a: f"({a[0]} + {a[1]})",
        scalar=lambda a: f"({a[0]} + {a[1]})",
        smt=_smt_add,
    ),
    Op(
        "Sub",
        ("int", "real", "word"),
        mat=lambda a: f"({a[0]} - {a[1]})",
        scalar=lambda a: f"({a[0]} - {a[1]})",
        smt=_smt_sub,
    ),
    # `*` on two `Matrix`es is their matrix product (Mathlib's), so the
    # pointwise product is spelled out entry by entry.
    Op(
        "Mul",
        ("int", "real", "word"),
        mat=lambda a: mat_mk(f"{a[0]} i j * {a[1]} i j", "i", "j"),
        scalar=lambda a: f"({a[0]} * {a[1]})",
        smt=_smt_mul,
    ),
    # Lean's `/` and `%` on `Int` are Euclidean, as SMT-LIB's `div` and
    # `mod` are, and by zero give what the theory gives (0, the dividend).
    *(
        Op(
            name,
            ("int",),
            mat=lambda a, sym=sym: mat_mk(f"{a[0]} i j {sym} {a[1]} i j", "i", "j"),
            scalar=lambda a, sym=sym: f"({a[0]} {sym} {a[1]})",
            smt=_smt_int_divmod,
        )
        for name, sym in (("Div", "/"), ("Mod", "%"))
    ),
    # Lean's `x / 0` on `Real` is 0, which the SMT form spells out.
    Op(
        "Div",
        ("real",),
        mat=lambda a: mat_mk(f"{a[0]} i j / {a[1]} i j", "i", "j"),
        scalar=lambda a: f"({a[0]} / {a[1]})",
        smt=_smt_real_div,
    ),
    Op(
        "Neg",
        ("int", "real", "word"),
        mat=lambda a: f"(-{a[0]})",
        scalar=lambda a: f"(-{a[0]})",
        smt=_smt_neg,
    ),
    Op(
        "Abs",
        NUM,
        mat=_not_asked_for("The matrix form would be `|a i j|`."),
        scalar=_SEE_MAT,
        smt=_smt_abs,
    ),
    Op(
        "Abs",
        SIGNED,
        mat=_word_only_in_smt(
            "The matrix form would negate the entries below zero."
        ),
        scalar=_SEE_MAT,
        smt=_smt_abs,
    ),
    Op(
        "Div",
        UNSIGNED,
        mat=_word_only_in_smt(
            "The matrix form would be `BitVec.udiv`."
        ),
        scalar=_SEE_MAT,
        smt=_smt_bv("BITVECTOR_UDIV"),
    ),
    Op(
        "Div",
        SIGNED,
        mat=_word_only_in_smt(
            "The matrix form would be `BitVec.sdiv`."
        ),
        scalar=_SEE_MAT,
        smt=_smt_bv("BITVECTOR_SDIV"),
    ),
    Op(
        "Rem",
        UNSIGNED,
        mat=lambda a: mat_mk(f"BitVec.umod ({a[0]} i j) ({a[1]} i j)", "i", "j"),
        scalar=lambda a: f"(BitVec.umod {a[0]} {a[1]})",
        smt=_smt_bv("BITVECTOR_UREM"),
    ),
    Op(
        "Rem",
        SIGNED,
        mat=_word_only_in_smt(
            "The matrix form would be `BitVec.srem`."
        ),
        scalar=_SEE_MAT,
        smt=_smt_bv("BITVECTOR_SREM"),
    ),
    Op(
        "Mod",
        SIGNED,
        mat=lambda a: mat_mk(f"BitVec.smod ({a[0]} i j) ({a[1]} i j)", "i", "j"),
        scalar=lambda a: f"(BitVec.smod {a[0]} {a[1]})",
        smt=_smt_bv("BITVECTOR_SMOD"),
    ),
    # ── comparisons: write Bool ───────────────────────────────────────
    *(
        Op(
            name,
            NUM,
            mat=lambda a, f=f: mat_mk(f"decide ({f(f'{a[0]} 0 0', f'{a[1]} 0 0')})"),
            scalar=lambda a, f=f: f"(decide ({f(a[0], a[1])}))",
            smt=_smt_compare,
            entrywise=True,
        )
        for name, f in (
            ("Lt", lambda x, y: f"{x} < {y}"),
            ("Le", lambda x, y: f"{x} ≤ {y}"),
            ("Gt", lambda x, y: f"{y} < {x}"),
            ("Ge", lambda x, y: f"{y} ≤ {x}"),
        )
    ),
    *(
        Op(
            name,
            ("bool", "int", "real", "word"),
            mat=lambda a, f=f: mat_mk(f"decide ({f(f'{a[0]} 0 0', f'{a[1]} 0 0')})"),
            scalar=lambda a, f=f: f"(decide ({f(a[0], a[1])}))",
            smt=_smt_compare,
            entrywise=True,
        )
        for name, f in (
            ("Eq", lambda x, y: f"{x} = {y}"),
            ("Ne", lambda x, y: f"{x} ≠ {y}"),
        )
    ),
    # Words compare unsigned and signed apart.
    *(
        Op(
            name,
            theories,
            mat=_word_only_in_smt(
                "The matrix form would compare with `BitVec`'s own "
                f"{order} ordering."
            ),
            scalar=_SEE_MAT,
            smt=_smt_word_compare(signed),
        )
        for theories, order, signed in ((UNSIGNED, "unsigned", False),
                                        (SIGNED, "signed", True))
        for name in ("Lt", "Le", "Gt", "Ge")
    ),
    # ── reductions and matrix ops ─────────────────────────────────────
    # `Min`/`Max` are reductions -- one read wire of any shape, a 1×1
    # matrix written -- folded row by row from entry (0, 0).
    *(
        Op(
            name,
            NUM,
            mat=lambda a, name=name: f"(mat{name} {a[0]})",
            scalar=ViaMat(f"`mat{name}` folds a matrix; there is no bare-scalar fold"),
            smt=_smt_min_max,
        )
        for name in ("Min", "Max")
    ),
    Op(
        "MatMul",
        NUM,
        mat=lambda a: f"({a[0]} * {a[1]})",
        scalar=lambda a: f"({a[0]} * {a[1]})",
        smt=_smt_matmul,
    ),
    # `Get`'s entry is baked into the op, as `Linear`'s matrices are.
    Op(
        "Get",
        ("bool", "int", "real", "word"),
        mat=Inline("codegen.terms._get_expr"),
        scalar=Inline("codegen.terms._get_expr, or the entry's flat slot"),
        smt=_smt_get,
    ),
    Op(
        "Transpose",
        NUM,
        mat=lambda a: f"(Matrix.transpose {a[0]})",
        scalar=ViaMat("`Matrix.transpose` is a matrix rearrangement"),
        smt=Unsupported(
            "no cvc5 term. A transpose is a permutation of the flat tuple "
            "and could be encoded; nothing has needed it, and "
            "`check_na_supported` reports the refusal by name."
        ),
    ),
    Op(
        "ReLU",
        NUM,
        mat=lambda a: f"ReLu {a[0]}",
        # `ReLu` is a matrix op; on a scalar-bound wire it has to be the
        # bare `max 0 x`. Lifting the operand and projecting the result
        # instead leaves `ReLu (Matrix.mk fun _ _ => x) 0 0` with unconstrained
        # dimensions, so this is not a `ViaMat` cell.
        scalar=lambda a: f"(Max.max 0 {a[0]})",
        smt=_smt_relu,
    ),
    # `Linear`'s A and B are baked into the op rather than read from wires,
    # `Scale`'s factor too (it is written as the affine map `diag(c)·X`),
    # and `Argmax` picks its form from the read wire's shape — none fits a
    # cell taking argument strings.
    Op(
        "Linear",
        NUM,
        mat=Inline("codegen.terms._linear_expr"),
        scalar=Inline("codegen.terms._linear_expr, lifted and projected"),
        smt=_smt_linear,
    ),
    Op(
        "Scale",
        NUM,
        mat=Inline("codegen.terms._linear_expr, as `diag(c)·X`"),
        scalar=Inline("codegen.terms._linear_expr, lifted and projected"),
        smt=_smt_scale,
    ),
    Op(
        "Argmax",
        NUM,
        mat=Inline("codegen.terms._argmax_expr"),
        scalar=Inline("native: the `argmax1d_scalar_*` axiom, else _argmax_expr"),
        smt=_smt_argmax,
    ),
    # ── shifts, rotations and resizing of words ───────────────────────
    *(
        Op(
            name,
            theories,
            mat=_word_only_in_smt(f"The matrix form would be `{lean}`."),
            scalar=_SEE_MAT,
            smt=smt,
        )
        for name, theories, lean, smt in (
            ("Shl", WORD, "a <<< b", _smt_bv("BITVECTOR_SHL")),
            ("Shr", UNSIGNED, "a >>> b", _smt_bv("BITVECTOR_LSHR")),
            ("Shr", SIGNED, "BitVec.sshiftRight' a b", _smt_bv("BITVECTOR_ASHR")),
            ("RotateLeft", WORD, "BitVec.rotateLeft a i",
             _smt_word_indexed("BITVECTOR_ROTATE_LEFT", lambda t: (t.i,))),
            ("RotateRight", WORD, "BitVec.rotateRight a i",
             _smt_word_indexed("BITVECTOR_ROTATE_RIGHT", lambda t: (t.i,))),
            ("Repeat", WORD, "BitVec.replicate i a",
             _smt_word_indexed("BITVECTOR_REPEAT", lambda t: (t.i,))),
            ("Extract", WORD, "BitVec.extractLsb i j a",
             _smt_word_indexed("BITVECTOR_EXTRACT", lambda t: (t.i, t.j))),
            ("Extend", UNSIGNED, "BitVec.zeroExtend (w + i) a",
             _smt_word_indexed("BITVECTOR_ZERO_EXTEND", lambda t: (t.i,))),
            ("Extend", SIGNED, "BitVec.signExtend (w + i) a",
             _smt_word_indexed("BITVECTOR_SIGN_EXTEND", lambda t: (t.i,))),
            ("Concat", WORD, "a ++ b", _smt_concat),
        )
    ),
    # ── transcendental functions of reals (the logic NRAT) ────────────
    # Mathlib has each as a total function (`Real.sqrt` is 0 below zero,
    # `Real.log` is `log |x|`, ...), but nothing has asked for their Lean
    # forms yet, so both Lean columns refuse them, as they do `Abs`.
    *(
        Op(
            name,
            ("real",),
            mat=_not_asked_for(f"The matrix form would be `{lean} (a i j)`."),
            scalar=_SEE_MAT,
            smt=smt,
        )
        for name, lean, smt in (
            ("Sqrt", "Real.sqrt", _smt_real_unary("SQRT")),
            ("Exp", "Real.exp", _smt_real_unary("EXPONENTIAL")),
            ("Log", "Real.log", Unsupported("cvc5 has no logarithm.")),
            ("Sin", "Real.sin", _smt_real_unary("SINE")),
            ("Cos", "Real.cos", _smt_real_unary("COSINE")),
            ("Tan", "Real.tan", _smt_real_unary("TANGENT")),
            ("Asin", "Real.arcsin", _smt_real_unary("ARCSINE")),
            ("Acos", "Real.arccos", _smt_real_unary("ARCCOSINE")),
            ("Atan", "Real.arctan", _smt_real_unary("ARCTANGENT")),
            ("Sinh", "Real.sinh", _smt_hyperbolic),
            ("Cosh", "Real.cosh", _smt_hyperbolic),
            ("Tanh", "Real.tanh", _smt_hyperbolic),
        )
    ),
    Op(
        "Atan2",
        ("real",),
        mat=_not_asked_for("The matrix form would be `Complex.arg ⟨b i j, a i j⟩`."),
        scalar=_SEE_MAT,
        smt=_smt_atan2,
    ),
    # ── casts between the theories' sorts ─────────────────────────────
    *(
        Op(
            name,
            ("cast",),
            mat=lambda a, one=one: mat_mk(one(f"({a[0]} i j)"), "i", "j"),
            scalar=lambda a, one=one: one(a[0]),
            smt=_smt_cast,
        )
        for name, one in (
            ("ToReal", lambda x: f"(({x} : Int) : Real)"),
            ("ToInt", lambda x: f"⌊{x}⌋"),
            ("UbvToInt", lambda x: f"((BitVec.toNat {x} : Nat) : Int)"),
        )
    ),
)


# Rows are keyed by (theory, name); the structural ops are of no theory, and
# keyed by ("", name), which is what `theory_of` returns for them.
_BY_KEY: dict[tuple[str, str], Op] = {}
for _op in OPS:
    for _theory in _op.theories or ("",):
        _key = (_theory, _op.name)
        if _key in _BY_KEY:
            raise RuntimeError(f"two op-table rows for {_theory}.{_op.name}")
        _BY_KEY[_key] = _op
del _op, _theory, _key


# ══════════════════════════════════════════════════════════════════════
#  Lookup
# ══════════════════════════════════════════════════════════════════════


def op_for(itype) -> Op:
    """The table row for this op, or a `ValueError` naming what is missing."""
    theory = theory_of(itype)
    name = itype_name(itype)
    row = _BY_KEY.get((theory, name))
    if row is None:
        raise ValueError(
            f"no op-table row for {theory or '<structural>'}.{name}; "
            "add one in reactor_lean/lowering/ops.py"
        )
    return row


def entrywise_shape(itype) -> "list[int] | None":
    """The shape of an `entrywise` op when it is larger than 1×1, where its
    scalar form applies entry by entry; None for every other op."""
    if not op_for(itype).entrywise:
        return None
    shape = list(itype.shape)
    return None if shape == [1, 1] else shape


def _reason(row: Op, cell) -> str:
    """What a non-emitter cell has to say for itself.

    `_SEE_MAT` carries no reason of its own: the matrix cell holds the one
    that covers both Lean columns.
    """
    if cell is _SEE_MAT:
        cell = row.mat
    if isinstance(cell, Unsupported):
        return cell.reason
    if isinstance(cell, Inline):
        return f"handled outside the table, by {cell.site}"
    return f"the matrix form serves it ({cell.note})"


def gap_of(row: Op) -> "str | None":
    """Why no Lean column can write this row's operator, or None if one can.

    Asked of both Lean columns, not of `mat` alone. No row in the table
    has an unsupported matrix form and an emitter in another Lean column --
    but that is a fact about today's rows, and a check that refuses a
    *module* on the strength of it would start refusing modules that build
    the day someone adds the first such row. It is one `isinstance` to not
    depend on it.

    Per column: `Unsupported` cannot write, and neither can `ViaMat`, which
    says the matrix form serves it -- and the matrix form is the gap.
    `Inline` can: it is the table saying the emitter lives beside it, in a
    branch that is reached before the table is asked.
    """
    if not isinstance(row.mat, Unsupported):
        return None
    if not isinstance(row.scalar, (Unsupported, ViaMat)):
        return None
    return _reason(row, row.mat)


def lean_gap(itype) -> "str | None":
    """:func:`gap_of` for an operator, looked up by theory and name.

    What :func:`mat_emitter` would raise with, as a value: a caller checking
    a whole module before generating any of it should not have to provoke an
    exception per term.
    """
    return gap_of(op_for(itype))


def _lean_emitter(itype, row: Op, cell, column: str):
    """Unwrap a Lean cell, or raise carrying the reason the table records."""
    if isinstance(cell, (Unsupported, Inline, ViaMat)):
        raise Refused(
            f"No Lean expression mapping for: {itype_name(itype)} "
            f"({column}) — {_reason(row, cell)}"
        )
    return cell


def mat_emitter(itype) -> Callable[[list[str]], str]:
    """The `Matrix`-form Lean emitter, or a `ValueError` carrying the reason."""
    row = op_for(itype)
    return _lean_emitter(itype, row, row.mat, "matrix form")


def scalar_emitter(itype) -> "Callable[[list[str]], str] | None":
    """The bare-scalar Lean emitter, or None when the matrix form serves.

    None covers `ViaMat` (deliberate), `Inline` (the caller has its own
    branch) and `Unsupported` alike: in every case the scalar printer falls
    through to the matrix column, which raises there if it has to.
    """
    cell = op_for(itype).scalar
    if isinstance(cell, (Unsupported, Inline, ViaMat)):
        return None
    return cell


def smt_emitter(itype) -> Callable[[SmtOp], cvc5.Term]:
    """The cvc5 emitter, or a `ValueError` carrying the reason.

    The SMT column holds no `Inline` or `ViaMat` cells: `translate_terms`
    dispatches every variant through the table, constants included.
    """
    cell = op_for(itype).smt
    if isinstance(cell, Unsupported):
        raise ValueError(
            f"SMT translator: unsupported op {itype_name(itype)} — {cell.reason}"
        )
    return cell


# ══════════════════════════════════════════════════════════════════════
#  The matrix, printed
# ══════════════════════════════════════════════════════════════════════

COLUMNS = ("mat", "scalar", "smt")


def cell_of(op: Op, column: str):
    return getattr(op, column)


def cell_mark(cell) -> str:
    """One-word rendering of a cell, for the printed matrix."""
    if isinstance(cell, Unsupported):
        return "-"
    if isinstance(cell, Inline):
        return "inline"
    if isinstance(cell, ViaMat):
        return "via mat"
    return "yes"


def rows() -> list[tuple[str, Op]]:
    """(theory, row) for every op, theory-major then alphabetical."""
    order = {t: i for i, t in enumerate(THEORIES)}
    order[""] = len(order)
    return sorted(
        ((theory, op) for (theory, _), op in _BY_KEY.items()),
        key=lambda pair: (order[pair[0]], pair[1].name),
    )


def format_matrix() -> str:
    """The whole table as text: one line per op, then every gap.

    Gaps are grouped by reason — the same one covers several ops, and
    reading it once per group is the point of having it written down.
    """
    head = f"{'theory':13} {'op':14} " + " ".join(f"{c:8}" for c in COLUMNS)
    lines = [head, "-" * len(head)]
    gaps: dict[str, list[str]] = {}
    for theory, op in rows():
        marks = " ".join(f"{cell_mark(cell_of(op, c)):8}" for c in COLUMNS)
        lines.append(f"{theory or '(struct)':13} {op.name:14} {marks}")
        for column in COLUMNS:
            cell = cell_of(op, column)
            if isinstance(cell, Unsupported) and cell is not _SEE_MAT:
                gaps.setdefault(cell.reason, []).append(
                    f"{theory or '(struct)'}.{op.name} [{column}]"
                )
    lines.append("\ngaps:")
    for reason, sites in gaps.items():
        lines.append(f"  * {', '.join(sites)}")
        lines.append(f"      {reason}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(format_matrix())
