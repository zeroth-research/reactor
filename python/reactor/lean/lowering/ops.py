"""One row per op variant, one column per backend: the op table.

`itype_name` turns a theory op into a variant name (`LIA.Ne()` -> `"Ne"`),
and four backends turn that name into something they can emit:

| column   | backend                          | a cell holds                 |
|----------|----------------------------------|------------------------------|
| `mat`    | `codegen.terms.translate_terms_lean`        | args -> Lean expr (on `Mat`) |
| `scalar` | `codegen.terms.translate_terms_scalar` | args -> Lean expr (bare)     |
| `box`    | `codegen.boxes.translate_terms_circ`     | a `Box` combinator name      |

A row is one variant of one or more theories, and it must say something
about every column. A cell is either an emitter or one of

* `ViaMat(...)` — scalar column only: the matrix emitter, with scalar reads
  lifted to 1x1 and `0 0` projected back off the result;
* `Inline(...)` — the backend emits this op from a branch of its own,
  because the argument strings are not enough (constants need the registry,
  `Argmax` and `Linear` need shapes and the baked tensors);
* `Unsupported(...)` — no emitter, and the reason it is missing rather than
  forgotten. That reason is what the caller is told.

`python -m reactor.lean.lowering.ops` prints the matrix, gaps and all.
`tests/test_lean_ops.py` checks it against `LIA`/`LRA`/`BV` in both
directions, so a variant added to a theory fails the suite until it has a
row, and a row no theory backs fails it too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

from reactor import Bool, Int, Real, BitVec

from ..core.errors import Refused
from .sorts import itype_name


# ══════════════════════════════════════════════════════════════════════
#  Cells
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Unsupported:
    """No emitter for this (variant, backend), and why not.

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

    `translate_terms_scalar` lifts scalar-bound reads to `fun _ _ => x`
    and projects `0 0` off the result, which is right whenever the matrix
    form is elementwise or folds to a 1x1.
    """

    note: str


MatCell = Union[Callable[[list[str]], str], Inline, Unsupported]
ScalarCell = Union[Callable[[list[str]], str], ViaMat, Inline, Unsupported]
BoxCell = Union[str, Inline, Unsupported]


# ══════════════════════════════════════════════════════════════════════
#  Rows
# ══════════════════════════════════════════════════════════════════════

# `Cast` is no theory of its own sorts: it is what crosses between the
# other three's (`theory/src/cast.rs`), and has rows of its own only.
THEORIES = ("LIA", "LRA", "BV", "Cast")

ALL = ("LIA", "LRA", "BV")
LIA_LRA = ("LIA", "LRA")
BV_ONLY = ("BV",)
# No theory defines these any more; the emitters are kept until it is
# decided whether to drop them (TODO.md). Dispatch goes through `itype_name`,
# so nothing can reach them.
NONE: tuple[str, ...] = ()


@dataclass(frozen=True)
class Op:
    """One variant, as every backend sees it."""

    name: str
    theories: tuple[str, ...]
    mat: MatCell
    scalar: ScalarCell
    box: BoxCell


# ══════════════════════════════════════════════════════════════════════
#  Reasons shared by several rows
# ══════════════════════════════════════════════════════════════════════

# The three Lean columns share one reason whenever a variant is missing from
# all of them; only the matrix cell spells the reason out.
_SEE_MAT = Unsupported("see the matrix column")

_NONDET = Unsupported(
    "a nondeterministic value (nullary, one written wire — `check_havoc` in "
    "theory/src/lib.rs). Every encoding here is a function of the read "
    "wires, so there is nowhere to put the choice."
)

_UNINTERPRETED = Unsupported(
    "an uninterpreted symbol has no obvious translation — emit an `opaque` "
    "nothing can prove anything about, or reject it at the front end. "
    "Which of the two is an open decision."
)

_ZEROGRAD = Unsupported(
    "the zero of the `Differential` trait (`theory/src/lra.rs`, marked "
    "unstable), used for gradients rather than for a transition. No "
    "encoding has needed it."
)


# ══════════════════════════════════════════════════════════════════════
#  The table
# ══════════════════════════════════════════════════════════════════════

OPS: tuple[Op, ...] = (
    # ── structural ────────────────────────────────────────────────────
    Op("Id", ALL, mat=lambda a: a[0], scalar=lambda a: a[0], box="Box.id"),
    # ── constants ─────────────────────────────────────────────────────
    # The element type is in the variant name for LIA/LRA and on the write
    # wire for BV; scalar-vs-matrix is decided by the wire's shape.
    *(
        Op(
            name,
            theories,
            mat=Inline("codegen.literals.constant_expr"),
            scalar=Inline("codegen.terms._constant_expr_scalar"),
            box=Inline("circ: @Box.const"),
        )
        for name, theories in (
            ("Bool", LIA_LRA),
            ("Int", ("LIA",)),
            ("Real", ("LRA",)),
            ("Const", BV_ONLY),
        )
    ),
    # ── Boolean forms (LIA/LRA) ───────────────────────────────────────
    # Operands and results are all `Mat _ 1 1`, so the matrix forms extract
    # position `0 0`. FIXME: element-wise ops should be defined over the
    # whole matrix, to match Rust.
    Op(
        "Not",
        LIA_LRA,
        mat=lambda a: f"(fun _ _ => !({a[0]} 0 0))",
        scalar=lambda a: f"(!{a[0]})",
        box="Box.not",
    ),
    Op(
        "And",
        LIA_LRA,
        mat=lambda a: f"(fun _ _ => ({a[0]} 0 0 && {a[1]} 0 0))",
        scalar=lambda a: f"({a[0]} && {a[1]})",
        box="Box.and",
    ),
    Op(
        "Or",
        LIA_LRA,
        mat=lambda a: f"(fun _ _ => ({a[0]} 0 0 || {a[1]} 0 0))",
        scalar=lambda a: f"({a[0]} || {a[1]})",
        box="Box.or",
    ),
    Op(
        "Xor",
        LIA_LRA,
        mat=Unsupported(
            "no Lean form; nothing has asked for one."
        ),
        scalar=_SEE_MAT,
        box=_SEE_MAT,
    ),
    Op(
        "Implies",
        LIA_LRA,
        mat=lambda a: f"(fun _ _ => (!({a[0]} 0 0) || {a[1]} 0 0))",
        scalar=lambda a: f"(!{a[0]} || {a[1]})",
        box=Unsupported(
            "no `Box.implies` in `Relational.Box`; `Box.or` after `Box.not` "
            "on the first operand is the circuit a new one would name."
        ),
    ),
    Op(
        "Ite",
        LIA_LRA,
        mat=lambda a: f"(if {a[0]} 0 0 then {a[1]} else {a[2]})",
        scalar=lambda a: f"(if {a[0]} then {a[1]} else {a[2]})",
        box="Box.ite",
    ),
    # ── BitVec forms ──────────────────────────────────────────────────
    # BV has no Bool wires at all: an `Ite` condition and an `Eq` result are
    # both `BitVec 1`. The Boolean forms above do not apply (`!`, `&&`,
    # `||` and a bare `if c then` were emitted against `BitVec 1` values,
    # which does not elaborate). These use BitVec's own operators and
    # compare a 1-bit value against 1 directly, rather than converting
    # through `BV.BVToBool`. `Box.and`/`not`/`or` are likewise fixed to
    # `Mat Bool 1 1` — see the note beside the `bv*` boxes in
    # `Relational.Box`.
    Op(
        "Not",
        BV_ONLY,
        mat=lambda a: f"(fun i j => ~~~({a[0]} i j))",
        scalar=lambda a: f"(~~~{a[0]})",
        box="Box.bvNot",
    ),
    Op(
        "And",
        BV_ONLY,
        mat=lambda a: f"(fun i j => ({a[0]} i j) &&& ({a[1]} i j))",
        scalar=lambda a: f"({a[0]} &&& {a[1]})",
        box="Box.bvAnd",
    ),
    Op(
        "Or",
        BV_ONLY,
        mat=lambda a: f"(fun i j => ({a[0]} i j) ||| ({a[1]} i j))",
        scalar=lambda a: f"({a[0]} ||| {a[1]})",
        box="Box.bvOr",
    ),
    Op(
        "Xor",
        BV_ONLY,
        mat=lambda a: f"(fun i j => ({a[0]} i j) ^^^ ({a[1]} i j))",
        scalar=lambda a: f"({a[0]} ^^^ {a[1]})",
        box="Box.bvXor",
    ),
    Op(
        "Ite",
        BV_ONLY,
        mat=lambda a: f"(if {a[0]} 0 0 = 1 then {a[1]} else {a[2]})",
        scalar=lambda a: f"(if {a[0]} = 1 then {a[1]} else {a[2]})",
        box="Box.bvIte",
    ),
    # ── arithmetic ────────────────────────────────────────────────────
    Op(
        "Add",
        ALL,
        mat=lambda a: f"({a[0]} + {a[1]})",
        scalar=lambda a: f"({a[0]} + {a[1]})",
        box="Box.add",
    ),
    Op(
        "Sub",
        ALL,
        mat=lambda a: f"({a[0]} - {a[1]})",
        scalar=lambda a: f"({a[0]} - {a[1]})",
        box="Box.sub",
    ),
    Op(
        "Mul",
        BV_ONLY,
        mat=lambda a: f"({a[0]} * {a[1]})",
        scalar=lambda a: f"({a[0]} * {a[1]})",
        box="Box.mul",
    ),
    # `*` on two `Mat`s is their matrix product (`Relational.Mat`), so the
    # elementwise product is spelled out element by element.
    Op(
        "Mul",
        LIA_LRA,
        mat=lambda a: f"(fun i j => {a[0]} i j * {a[1]} i j)",
        scalar=lambda a: f"({a[0]} * {a[1]})",
        box=Unsupported(
            "`Box.mul` is the matrix product; the elementwise one would be a "
            "new Box in `Relational.Box`."
        ),
    ),
    # Lean's `/` and `%` on `Int` are Euclidean, as SMT-LIB's `div` and
    # `mod` are, and by zero give what the theory gives (0, the dividend).
    *(
        Op(
            name,
            ("LIA",),
            mat=lambda a, sym=sym: f"(fun i j => {a[0]} i j {sym} {a[1]} i j)",
            scalar=lambda a, sym=sym: f"({a[0]} {sym} {a[1]})",
            box=Unsupported(
                f"no `Box` for `{sym}` on `Int` in `Relational.Box`."
            ),
        )
        for name, sym in (("Div", "/"), ("Mod", "%"))
    ),
    Op(
        "Neg",
        ALL,
        mat=lambda a: f"(-{a[0]})",
        scalar=lambda a: f"(-{a[0]})",
        box="Box.neg",
    ),
    Op(
        "UDiv",
        BV_ONLY,
        scalar=_SEE_MAT,
        box=_SEE_MAT,
    ),
    Op(
        "SDiv",
        BV_ONLY,
        scalar=_SEE_MAT,
        box=_SEE_MAT,
    ),
    Op(
        "UMod",
        BV_ONLY,
        mat=lambda a: f"(fun i j => BitVec.umod ({a[0]} i j) ({a[1]} i j))",
        scalar=lambda a: f"(BitVec.umod {a[0]} {a[1]})",
        box="Box.bvUMod",
    ),
    Op(
        "SMod",
        BV_ONLY,
        mat=lambda a: f"(fun i j => BitVec.smod ({a[0]} i j) ({a[1]} i j))",
        scalar=lambda a: f"(BitVec.smod {a[0]} {a[1]})",
        box="Box.bvSMod",
    ),
    # ── comparisons ───────────────────────────────────────────────────
    Op(
        "Lt",
        LIA_LRA,
        mat=lambda a: f"(fun _ _ => decide ({a[0]} 0 0 < {a[1]} 0 0))",
        scalar=lambda a: f"(decide ({a[0]} < {a[1]}))",
        box="Box.lt",
    ),
    Op(
        "Le",
        LIA_LRA,
        mat=lambda a: f"(fun _ _ => decide ({a[0]} 0 0 ≤ {a[1]} 0 0))",
        scalar=lambda a: f"(decide ({a[0]} ≤ {a[1]}))",
        box="Box.le",
    ),
    Op(
        "Gt",
        LIA_LRA,
        mat=lambda a: f"(fun _ _ => decide ({a[1]} 0 0 < {a[0]} 0 0))",
        scalar=lambda a: f"(decide ({a[1]} < {a[0]}))",
        box="Box.gt",
    ),
    Op(
        "Ge",
        LIA_LRA,
        mat=lambda a: f"(fun _ _ => decide ({a[1]} 0 0 ≤ {a[0]} 0 0))",
        scalar=lambda a: f"(decide ({a[1]} ≤ {a[0]}))",
        box="Box.ge",
    ),
    Op(
        "Eq",
        LIA_LRA,
        mat=lambda a: f"(fun _ _ => decide ({a[0]} 0 0 = {a[1]} 0 0))",
        scalar=lambda a: f"(decide ({a[0]} = {a[1]}))",
        box="Box.eq",
    ),
    Op(
        "Ne",
        LIA_LRA,
        mat=lambda a: f"(fun _ _ => decide ({a[0]} 0 0 ≠ {a[1]} 0 0))",
        scalar=lambda a: f"(decide ({a[0]} ≠ {a[1]}))",
        box="Box.neq",
    ),
    Op(
        "Eq",
        BV_ONLY,
        mat=lambda a: f"(fun _ _ => if {a[0]} 0 0 = {a[1]} 0 0 then 1 else 0)",
        scalar=lambda a: f"(if {a[0]} = {a[1]} then 1 else 0)",
        box="Box.bvEq",
    ),
    Op(
        "Ne",
        BV_ONLY,
        mat=lambda a: f"(fun _ _ => if {a[0]} 0 0 = {a[1]} 0 0 then 0 else 1)",
        scalar=lambda a: f"(if {a[0]} = {a[1]} then 0 else 1)",
        box="Box.bvNe",
    ),
    # BV compares unsigned and signed separately, and writes the answer to a
    # `BV<1>` wire rather than a Bool one.
    *(
        Op(
            name,
            BV_ONLY,
            scalar=_SEE_MAT,
            box=_SEE_MAT,
        )
        for name in ("ULt", "ULe", "UGt", "UGe", "SLt", "SLe", "SGt", "SGe")
    ),
    # ── reductions and matrix ops ─────────────────────────────────────
    # `Min`/`Max` are unary reductions — one read wire of any shape, a
    # `Mat t 1 1` written — folded row-major from element (0,0).
    Op(
        "Min",
        LIA_LRA,
        mat=lambda a: f"(matMin {a[0]})",
        scalar=ViaMat("`matMin` folds a matrix; there is no bare-scalar fold"),
        box="Box.min",
    ),
    Op(
        "Max",
        LIA_LRA,
        mat=lambda a: f"(matMax {a[0]})",
        scalar=ViaMat("`matMax` folds a matrix; there is no bare-scalar fold"),
        box="Box.max",
    ),
    Op(
        "MatMul",
        BV_ONLY,
        mat=lambda a: f"MatMul {a[0]} {a[1]}",
        scalar=lambda a: f"({a[0]} * {a[1]})",
        box="Box.mul",
    ),
    # `Get`'s element is baked into the op, as `Linear`'s matrices are.
    Op(
        "Get",
        ALL,
        mat=Inline("codegen.terms._get_expr"),
        scalar=Inline("codegen.terms._get_expr, or the element's flat slot"),
        box=Unsupported("no Box in `Relational.Box` reads one element of a matrix."),
    ),
    Op(
        "Transpose",
        LIA_LRA,
        mat=lambda a: f"(MatTranspose {a[0]})",
        scalar=ViaMat("`MatTranspose` is a matrix rearrangement"),
        box="Box.transpose",
    ),
    Op(
        "ReLU",
        LIA_LRA,
        mat=lambda a: f"ReLu {a[0]}",
        # `ReLu` is a matrix op; on a scalar-bound wire it has to be the
        # bare `max 0 x`. Lifting the operand and projecting the result
        # instead leaves `ReLu (fun _ _ => x) 0 0` with unconstrained
        # dimensions, so this is not a `ViaMat` cell.
        scalar=lambda a: f"(Max.max 0 {a[0]})",
        box="Box.relu",
    ),
    # `Linear`'s A and B are baked into the op rather than read from wires,
    # and `Argmax` picks its form from the read wire's shape — neither fits
    # a cell taking argument strings.
    Op(
        "Linear",
        LIA_LRA,
        mat=Inline("codegen.terms._linear_expr"),
        scalar=Inline("codegen.terms._linear_expr, lifted and projected"),
        box=Inline("codegen.boxes._linear_box"),
    ),
    Op(
        "Argmax",
        LIA_LRA,
        mat=Inline("codegen.terms._argmax_expr"),
        scalar=Inline("native: the `argmax1d_scalar_*` axiom, else _argmax_expr"),
        box=Inline("codegen.boxes._argmax_box"),
    ),
    # ── bit-width and conversion ──────────────────────────────────────
    Op(
        "BVToBool",
        BV_ONLY,
        scalar=_SEE_MAT,
        box=_SEE_MAT,
    ),
    *(
        Op(
            name,
            BV_ONLY,
            mat=Unsupported(reason),
            scalar=_SEE_MAT,
            box=_SEE_MAT,
        )
        for name, reason in (
            (
                "BitSelect",
                "width-changing: `BitSelect[high:low]` writes a narrower "
                "wire than it reads (theory/src/bv.rs). No backend emits it.",
            ),
            (
                "Extend",
                "width-changing: `Extend(+extra)` writes a wider wire than "
                "it reads (theory/src/bv.rs). No backend emits it.",
            ),
            (
                "Abs",
                "modelled as `Ite(x < 0, Neg x, x)` in theory/src/bv.rs, "
                "which leaves `Abs(BV_MIN)` open there (XXX). No backend "
                "emits it.",
            ),
        )
    ),
    # ── casts between the theories' sorts ─────────────────────────────
    *(
        Op(
            name,
            ("Cast",),
            mat=lambda a, one=one: f"(fun i j => {one(f'({a[0]} i j)')})",
            scalar=lambda a, one=one: one(a[0]),
            box=Unsupported("no Box in `Relational.Box` casts between sorts."),
        )
        for name, one in (
            ("IntToReal", lambda x: f"(({x} : Int) : Real)"),
            ("RealToInt", lambda x: f"⌊{x}⌋"),
            ("BVToInt", lambda x: f"((BitVec.toNat {x} : Nat) : Int)"),
            ("BitToBool", lambda x: f"(decide ({x} = 1))"),
            ("BoolToBit", lambda x: f"(if {x} then 1 else 0)"),
        )
    ),
    # ── nondeterminism and symbols ────────────────────────────────────
    *(
        Op(
            name,
            theories,
            mat=reason,
            scalar=_SEE_MAT,
            box=_SEE_MAT,
        )
        for name, theories, reason in (
            ("AnyBool", LIA_LRA, _NONDET),
            ("AnyInt", ("LIA",), _NONDET),
            ("AnyReal", ("LRA",), _NONDET),
            ("Havoc", BV_ONLY, _NONDET),
            ("Uninterpreted", ALL, _UNINTERPRETED),
            ("BoolZerograd", ("LRA",), _ZEROGRAD),
            ("RealZerograd", ("LRA",), _ZEROGRAD),
        )
    ),
    # ── dead ──────────────────────────────────────────────────────────
    # `IType`-era names no theory defines any more. Kept until it is
    # decided whether to remove them (TODO.md); unreachable either way,
    # since dispatch goes through `itype_name`.
    Op(
        "Tensor",
        NONE,
        mat=Inline("codegen.literals.constant_expr"),
        scalar=Inline("codegen.terms._constant_expr_scalar"),
        box=Inline("circ: @Box.const"),
    ),
    Op(
        "TensorGet",
        NONE,
        mat=lambda a: f"({a[0]} 0 0)",
        scalar=lambda a: a[0],
        box=Unsupported("never had a Box"),
    ),
    Op(
        "ToUnsigned",
        NONE,
        mat=lambda a: f"(fun _ _ => Int.toNat ({a[0]} 0 0))",
        scalar=lambda a: f"(Int.toNat {a[0]})",
        box=Unsupported("never had a Box"),
    ),
)


# Rows are keyed by (theory, variant); a dead row answers to no theory and
# is keyed by ("", name), which is what `theory_of` returns for an op
# carrying no theory prefix.
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


def theory_of(itype) -> str:
    """The theory an op came from: "LIA", "LRA", "BV" or "Cast".

    PyO3 exports the variants as `LIA_Add`, `LRA_Real`, `BV_MatMul`,
    `Cast_BVToInt`; an op
    with no such prefix belongs to no theory and gets "".
    """
    name = type(itype).__name__
    for theory in THEORIES:
        if name.startswith(theory + "_"):
            return theory
    return ""


def op_for(itype) -> Op:
    """The table row for this op, or a `ValueError` naming what is missing."""
    theory = theory_of(itype)
    name = itype_name(itype)
    row = _BY_KEY.get((theory, name))
    if row is None:
        raise ValueError(
            f"no op-table row for {theory or '<no theory>'}.{name}; "
            "add one in reactor/lean/ops.py"
        )
    return row


def _reason(row: Op, cell) -> str:
    """What a non-emitter cell has to say for itself.

    `_SEE_MAT` carries no reason of its own: the matrix cell holds the one
    that covers all three Lean columns.
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

    Asked of all three Lean columns, not of `mat` alone. No row in the table
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
    if any(not isinstance(cell, (Unsupported, ViaMat))
           for cell in (row.scalar, row.box)):
        return None
    return _reason(row, row.mat)


def lean_gap(itype) -> "str | None":
    """:func:`gap_of` for an operator, looked up by theory and variant.

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
    """The `Mat`-form Lean emitter, or a `ValueError` carrying the reason."""
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


def box_for(itype) -> str:
    """The `Box` combinator, or a `ValueError` carrying the reason."""
    row = op_for(itype)
    return _lean_emitter(itype, row, row.box, "Box")


# ══════════════════════════════════════════════════════════════════════
#  The matrix, printed
# ══════════════════════════════════════════════════════════════════════

COLUMNS = ("mat", "scalar", "box")


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
    """(theory, row) for every variant, theory-major then alphabetical."""
    order = {t: i for i, t in enumerate(THEORIES)}
    order[""] = len(order)
    return sorted(
        ((theory, op) for (theory, _), op in _BY_KEY.items()),
        key=lambda pair: (order[pair[0]], pair[1].name),
    )


def format_matrix() -> str:
    """The whole table as text: one line per variant, then every gap.

    Gaps are grouped by reason — the same one covers several variants, and
    reading it once per group is the point of having it written down.
    """
    head = f"{'theory':7} {'op':14} " + " ".join(f"{c:8}" for c in COLUMNS)
    lines = [head, "-" * len(head)]
    gaps: dict[str, list[str]] = {}
    for theory, op in rows():
        marks = " ".join(f"{cell_mark(cell_of(op, c)):8}" for c in COLUMNS)
        lines.append(f"{theory or '(dead)':7} {op.name:14} {marks}")
        for column in COLUMNS:
            cell = cell_of(op, column)
            if isinstance(cell, Unsupported) and cell is not _SEE_MAT:
                gaps.setdefault(cell.reason, []).append(
                    f"{theory or '(dead)'}.{op.name} [{column}]"
                )
    lines.append("\ngaps:")
    for reason, sites in gaps.items():
        lines.append(f"  * {', '.join(sites)}")
        lines.append(f"      {reason}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(format_matrix())
