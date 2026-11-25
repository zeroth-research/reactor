"""One row per op, one column per backend: the op table.

An op is one class of one theory: the namespace of `reactor.tensor` it is in
(`theory_of`: `"int"` for `tensor.int.Ne([1, 1])`) and the name of its class
(`itype_name`: `"Ne"`). Two backends turn that pair into a Lean expression:

| column   | backend                                | a cell holds                    |
|----------|----------------------------------------|---------------------------------|
| `mat`    | `codegen.terms.translate_terms_lean`   | args -> Lean expr (on `Matrix`) |
| `scalar` | `codegen.terms.translate_terms_scalar` | args -> Lean expr (bare)        |

A row is one op of one or more theories, and it must say something about
every column. A cell is either an emitter or one of

* `ViaMat(...)` — scalar column only: the matrix emitter, with scalar reads
  lifted to 1x1 and `0 0` projected back off the result;
* `Inline(...)` — the backend emits this op from a branch of its own,
  because the argument strings are not enough (constants need the registry,
  `Argmax`, `Linear` and `Scale` need shapes and the baked tensors);
* `Unsupported(...)` — no emitter, and the reason it is missing rather than
  forgotten. That reason is what the caller is told.

A row's two columns are both emitters or both refusals, so that a module
no encoding can write is refused before any is generated
(`codegen.terms.lean_gaps`), not half way through a project.

`scripts/op_table.py` prints the matrix, gaps and all.
`tests/test_lean_ops.py` checks it against `reactor.tensor` in both
directions, so an op added to a theory fails the suite until it has a row,
and a row no theory backs fails it too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

from ..errors import Refused
from .sorts import itype_name, mat_mk, theory_of


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

    `entrywise`: the cells are forms of 1×1 operands, read at `0 0`. The op
    of a larger shape is its scalar form applied entry by entry
    (`codegen.terms`, `entrywise_shape`).
    """

    name: str
    theories: tuple[str, ...]
    mat: MatCell
    scalar: ScalarCell
    entrywise: bool = False


# ══════════════════════════════════════════════════════════════════════
#  Reasons shared by several rows
# ══════════════════════════════════════════════════════════════════════

# The two columns share one reason whenever an op is missing from both;
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


def _word_with_no_lean_form(lean_form: str) -> Unsupported:
    """A word op of the theory that no Lean backend emits, and why;
    `codegen.terms.lean_gaps` reports it up front."""
    return Unsupported(
        f"no Lean form. {lean_form} A module using it is refused at project "
        "generation."
    )


# ══════════════════════════════════════════════════════════════════════
#  The table
# ══════════════════════════════════════════════════════════════════════

OPS: tuple[Op, ...] = (
    # ── structural ────────────────────────────────────────────────────
    Op("Skip", STRUCTURAL, mat=lambda a: a[0], scalar=lambda a: a[0]),
    Op("Havoc", STRUCTURAL, mat=_NONDET, scalar=_SEE_MAT),
    Op("Zero", STRUCTURAL, mat=_ZERO, scalar=_SEE_MAT),
    # ── constants ─────────────────────────────────────────────────────
    # The element type is the written wire's; scalar-vs-matrix is decided by
    # the wire's shape.
    Op(
        "Const",
        ("bool", "int", "real", "word"),
        mat=Inline("codegen.literals.constant_expr"),
        scalar=Inline("codegen.terms._constant_expr_scalar"),
    ),
    # ── propositional logic ───────────────────────────────────────────
    # The matrix forms are of 1×1 operands, read at `0 0`; a larger op is
    # the scalar form entry by entry (`entrywise`).
    Op(
        "Not",
        ("bool",),
        mat=lambda a: mat_mk(f"!({a[0]} 0 0)"),
        scalar=lambda a: f"(!{a[0]})",
        entrywise=True,
    ),
    Op(
        "And",
        ("bool",),
        mat=lambda a: mat_mk(f"({a[0]} 0 0 && {a[1]} 0 0)"),
        scalar=lambda a: f"({a[0]} && {a[1]})",
        entrywise=True,
    ),
    Op(
        "Or",
        ("bool",),
        mat=lambda a: mat_mk(f"({a[0]} 0 0 || {a[1]} 0 0)"),
        scalar=lambda a: f"({a[0]} || {a[1]})",
        entrywise=True,
    ),
    Op(
        "Xor",
        ("bool",),
        mat=Unsupported("no Lean form; nothing has asked for one."),
        scalar=_SEE_MAT,
    ),
    Op(
        "Implies",
        ("bool",),
        mat=lambda a: mat_mk(f"(!({a[0]} 0 0) || {a[1]} 0 0)"),
        scalar=lambda a: f"(!{a[0]} || {a[1]})",
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
    ),
    Op(
        "Ite",
        ("bool", "int", "real", "word"),
        mat=lambda a: f"(if {a[0]} 0 0 then {a[1]} else {a[2]})",
        scalar=lambda a: f"(if {a[0]} then {a[1]} else {a[2]})",
    ),
    # ── bitwise ───────────────────────────────────────────────────────
    # BitVec's own operators, mapped over the matrix.
    Op(
        "Not",
        WORD,
        mat=lambda a: mat_mk(f"~~~({a[0]} i j)", "i", "j"),
        scalar=lambda a: f"(~~~{a[0]})",
    ),
    Op(
        "And",
        WORD,
        mat=lambda a: mat_mk(f"({a[0]} i j) &&& ({a[1]} i j)", "i", "j"),
        scalar=lambda a: f"({a[0]} &&& {a[1]})",
    ),
    Op(
        "Or",
        WORD,
        mat=lambda a: mat_mk(f"({a[0]} i j) ||| ({a[1]} i j)", "i", "j"),
        scalar=lambda a: f"({a[0]} ||| {a[1]})",
    ),
    Op(
        "Xor",
        WORD,
        mat=lambda a: mat_mk(f"({a[0]} i j) ^^^ ({a[1]} i j)", "i", "j"),
        scalar=lambda a: f"({a[0]} ^^^ {a[1]})",
    ),
    *(
        Op(
            name,
            WORD,
            mat=_word_with_no_lean_form(
                f"The matrix form would be `~~~(a {sym} b)`."
            ),
            scalar=_SEE_MAT,
        )
        for name, sym in (
            ("Nand", "&&&"),
            ("Nor", "|||"),
            ("Xnor", "^^^"),
        )
    ),
    # ── arithmetic ────────────────────────────────────────────────────
    Op(
        "Add",
        ("int", "real", "word"),
        mat=lambda a: f"({a[0]} + {a[1]})",
        scalar=lambda a: f"({a[0]} + {a[1]})",
    ),
    Op(
        "Sub",
        ("int", "real", "word"),
        mat=lambda a: f"({a[0]} - {a[1]})",
        scalar=lambda a: f"({a[0]} - {a[1]})",
    ),
    # `*` on two `Matrix`es is their matrix product (Mathlib's), so the
    # pointwise product is spelled out entry by entry.
    Op(
        "Mul",
        ("int", "real", "word"),
        mat=lambda a: mat_mk(f"{a[0]} i j * {a[1]} i j", "i", "j"),
        scalar=lambda a: f"({a[0]} * {a[1]})",
    ),
    # Lean's `/` and `%` on `Int` are Euclidean, as SMT-LIB's `div` and
    # `mod` are, and by zero give what the theory gives (0, the dividend).
    *(
        Op(
            name,
            ("int",),
            mat=lambda a, sym=sym: mat_mk(f"{a[0]} i j {sym} {a[1]} i j", "i", "j"),
            scalar=lambda a, sym=sym: f"({a[0]} {sym} {a[1]})",
        )
        for name, sym in (("Div", "/"), ("Mod", "%"))
    ),
    # Lean's `x / 0` on `Real` is 0.
    Op(
        "Div",
        ("real",),
        mat=lambda a: mat_mk(f"{a[0]} i j / {a[1]} i j", "i", "j"),
        scalar=lambda a: f"({a[0]} / {a[1]})",
    ),
    Op(
        "Neg",
        ("int", "real", "word"),
        mat=lambda a: f"(-{a[0]})",
        scalar=lambda a: f"(-{a[0]})",
    ),
    Op(
        "Abs",
        NUM,
        mat=Unsupported("no Lean form; nothing has asked for one. The "
                        "matrix form would be `|a i j|`."),
        scalar=_SEE_MAT,
    ),
    Op(
        "Abs",
        SIGNED,
        mat=_word_with_no_lean_form(
            "The matrix form would negate the entries below zero."
        ),
        scalar=_SEE_MAT,
    ),
    Op(
        "Div",
        UNSIGNED,
        mat=_word_with_no_lean_form(
            "The matrix form would be `BitVec.udiv`."
        ),
        scalar=_SEE_MAT,
    ),
    Op(
        "Div",
        SIGNED,
        mat=_word_with_no_lean_form(
            "The matrix form would be `BitVec.sdiv`."
        ),
        scalar=_SEE_MAT,
    ),
    Op(
        "Rem",
        UNSIGNED,
        mat=lambda a: mat_mk(f"BitVec.umod ({a[0]} i j) ({a[1]} i j)", "i", "j"),
        scalar=lambda a: f"(BitVec.umod {a[0]} {a[1]})",
    ),
    Op(
        "Rem",
        SIGNED,
        mat=_word_with_no_lean_form(
            "The matrix form would be `BitVec.srem`."
        ),
        scalar=_SEE_MAT,
    ),
    Op(
        "Mod",
        SIGNED,
        mat=lambda a: mat_mk(f"BitVec.smod ({a[0]} i j) ({a[1]} i j)", "i", "j"),
        scalar=lambda a: f"(BitVec.smod {a[0]} {a[1]})",
    ),
    # ── comparisons: write Bool ───────────────────────────────────────
    *(
        Op(
            name,
            NUM,
            mat=lambda a, f=f: mat_mk(f"decide ({f(f'{a[0]} 0 0', f'{a[1]} 0 0')})"),
            scalar=lambda a, f=f: f"(decide ({f(a[0], a[1])}))",
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
            mat=_word_with_no_lean_form(
                "The matrix form would compare with `BitVec`'s own "
                f"{order} ordering."
            ),
            scalar=_SEE_MAT,
        )
        for theories, order in ((UNSIGNED, "unsigned"), (SIGNED, "signed"))
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
        )
        for name in ("Min", "Max")
    ),
    Op(
        "MatMul",
        NUM,
        mat=lambda a: f"({a[0]} * {a[1]})",
        # A 1×1 product need not have 1×1 operands: a row times a column.
        scalar=ViaMat("`*` is the matrix product, of operands that need "
                      "not be 1×1 when the product is"),
    ),
    # `Get`'s entry is baked into the op, as `Linear`'s matrices are.
    Op(
        "Get",
        ("bool", "int", "real", "word"),
        mat=Inline("codegen.terms._get_expr"),
        scalar=Inline("codegen.terms._get_expr, or the entry's flat slot"),
    ),
    Op(
        "Transpose",
        NUM,
        mat=lambda a: f"(Matrix.transpose {a[0]})",
        scalar=ViaMat("`Matrix.transpose` is a matrix rearrangement"),
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
    ),
    Op(
        "Scale",
        NUM,
        mat=Inline("codegen.terms._linear_expr, as `diag(c)·X`"),
        scalar=Inline("codegen.terms._linear_expr, lifted and projected"),
    ),
    Op(
        "Argmax",
        NUM,
        mat=Inline("codegen.terms._argmax_expr"),
        scalar=Inline("codegen.terms: over one row, `argmax1d_scalar_*`; "
                      "else _argmax_expr"),
    ),
    # ── shifts, rotations and resizing of words ───────────────────────
    *(
        Op(
            name,
            theories,
            mat=_word_with_no_lean_form(f"The matrix form would be `{lean}`."),
            scalar=_SEE_MAT,
        )
        for name, theories, lean in (
            ("Shl", WORD, "a <<< b"),
            ("Shr", UNSIGNED, "a >>> b"),
            ("Shr", SIGNED, "BitVec.sshiftRight' a b"),
            ("RotateLeft", WORD, "BitVec.rotateLeft a i"),
            ("RotateRight", WORD, "BitVec.rotateRight a i"),
            ("Repeat", WORD, "BitVec.replicate i a"),
            ("Extract", WORD, "BitVec.extractLsb i j a"),
            ("Extend", UNSIGNED, "BitVec.zeroExtend (w + i) a"),
            ("Extend", SIGNED, "BitVec.signExtend (w + i) a"),
            ("Concat", WORD, "a ++ b"),
        )
    ),
    # ── casts between the theories' sorts ─────────────────────────────
    *(
        Op(
            name,
            ("cast",),
            mat=lambda a, one=one: mat_mk(one(f"({a[0]} i j)"), "i", "j"),
            scalar=lambda a, one=one: one(a[0]),
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
    that covers both columns.
    """
    if cell is _SEE_MAT:
        cell = row.mat
    if isinstance(cell, Unsupported):
        return cell.reason
    if isinstance(cell, Inline):
        return f"handled outside the table, by {cell.site}"
    return f"the matrix form serves it ({cell.note})"


def gap_of(row: Op) -> "str | None":
    """Why no Lean encoding can write this row's operator, or None if one
    can: the matrix cell's reason, when it has no matrix form.

    The matrix form decides alone. The functional encoding is generated
    for every module, from the matrix column only, so a row with no matrix
    form refuses the module whatever its other columns hold; checked any
    later, it does so half way through the project.
    """
    if isinstance(row.mat, Unsupported):
        return _reason(row, row.mat)
    return None


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
