"""The op table: every op covered, every gap named.

`reactor_lean/lowering/ops.py` holds one row per op and one column per backend
(`mat`, `scalar`, `box`). These tests are the other half of that: they
check the table against what the namespaces of `reactor.tensor` actually
expose, in both directions.

The direction that matters is *missing*. Separate dicts could be checked for
keys no theory defines — a dead key silently disables the op it was meant to
serve — but nothing could see an op with no emitter at all, because there was
no one place that knew which ops there are. That is how every unsigned and
signed bit-vector comparison came to have no Lean backend, noticed only when a
module using one failed at project generation.

So: an op added to a theory fails `test_every_op_has_a_row` until someone
writes its row, and writing that row means saying what all three backends do
with it — an emitter, or the reason there is none.
"""

import torch
import pytest
from reactor import Wire, Term, Module, Bool, Int, Word, Var, X, tensor
from reactor.tensor import Havoc, Skip
from reactor_lean import ModuleToLean4
from reactor_lean.lowering.sorts import THEORIES, itype_name, theory_of
from reactor_lean.codegen.context import LeanContext
from reactor_lean.codegen.terms import lean_gaps
from reactor_lean.lowering import ops
from testdata import all_encodings
from reactor_lean.lowering.ops import (
    COLUMNS,
    OPS,
    Inline,
    Op,
    Unsupported,
    ViaMat,
    box_for,
    cell_of,
    mat_emitter,
    op_for,
    scalar_emitter,
)

S = [1, 1]


def _namespace(theory: str):
    ns = tensor
    for part in theory.split("."):
        ns = getattr(ns, part)
    return ns


def _op_names(theory: str) -> set[str]:
    """Every op the theory's namespace exposes, as `itype_name` sees it: its
    classes, not the namespaces nested in it (`word.signed`)."""
    if theory == "":
        return {"Havoc", "Skip", "Zero"}
    ns = _namespace(theory)
    return {n for n, v in vars(ns).items() if isinstance(v, type)}


def _row(theory: str, name: str) -> Op:
    return next(op for op in OPS if op.name == name and theory in op.theories)


# ══════════════════════════════════════════════════════════════════════
#  Completeness, both directions
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("theory", [*THEORIES, ""])
def test_every_op_has_a_row(theory):
    """The guard separate tables could not give: no op goes unanswered."""
    covered = {op.name for op in OPS if theory in op.theories}
    missing = _op_names(theory) - covered
    assert not missing, (
        f"{theory or 'the structural ops'} expose {sorted(missing)} with no row "
        "in reactor_lean/lowering/ops.py. Add one, saying for each of "
        f"{list(COLUMNS)} either an emitter or `Unsupported(<why not>)`."
    )


def test_no_row_outlives_its_theory():
    """A row keyed to an op no theory defines can never fire."""
    for op in OPS:
        assert op.theories, f"lowering/ops.py has a row for {op.name} of no theory"
        for theory in op.theories:
            assert op.name in _op_names(theory), (
                f"lowering/ops.py has a row for {theory}.{op.name}, "
                "which that theory does not define"
            )


def test_a_row_is_reachable_from_its_op():
    """`op_for` resolves an op to its row, theory included: the classes of two
    theories share names, so the class, not its name, picks the row."""
    assert op_for(tensor.int.Ne(S)) is _row("int", "Ne")
    assert op_for(tensor.word.signed.Lt(8, S)) is _row("word.signed", "Lt")
    assert op_for(tensor.word.unsigned.Lt(8, S)) is _row("word.unsigned", "Lt")
    assert op_for(tensor.word.signed.Lt(8, S)) is not op_for(tensor.word.unsigned.Lt(8, S))
    assert op_for(tensor.int.Div(S)) is not op_for(tensor.real.Div(S))
    assert theory_of(Skip(Int(S))) == ""


def test_an_op_of_a_backend_is_of_its_theory():
    """The constants, products and affine maps holding JAX arrays, or either
    tensor, are of the theory their torch counterparts are."""
    from reactor.reactor import mixed

    assert theory_of(mixed.int.Const(torch.tensor([[1]]))) == "int"
    assert theory_of(mixed.real.Scale(torch.tensor([[1.0]]))) == "real"
    jnp = pytest.importorskip("jax.numpy")
    from reactor.reactor import jax

    assert theory_of(jax.word.Const(jnp.array([[1]]), 8)) == "word"


# ══════════════════════════════════════════════════════════════════════
#  What a cell may say
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("column", COLUMNS)
def test_every_cell_is_an_emitter_or_says_why(column):
    for op in OPS:
        cell = cell_of(op, column)
        if isinstance(cell, Unsupported):
            assert cell.reason.strip(), f"{op.name}.{column}: empty reason"
        elif isinstance(cell, Inline):
            assert cell.site.strip(), f"{op.name}.{column}: empty site"
        elif isinstance(cell, ViaMat):
            assert cell.note.strip(), f"{op.name}.{column}: empty note"
        elif column == "box":
            assert isinstance(cell, str) and cell.startswith("Box.")
        else:
            assert callable(cell), f"{op.name}.{column}: not an emitter"


def test_the_lean_columns_are_all_emitters_or_all_refusals():
    """A module is refused before any of its encodings is generated
    (`lean_gaps`) only if no row has one Lean column refusing while another
    writes: otherwise the circuit encoding would fail half way through a
    project the functional one was generated for. `All` alone has no Box
    where it has the other forms: the circuit encoding is optional, and of a
    module with an `All` it is a file saying why."""
    for op in OPS:
        lean = [cell_of(op, c) for c in ("mat", "scalar", "box")]
        refusals = [isinstance(c, Unsupported) for c in lean]
        if op.name == "All":
            assert refusals == [False, False, True]
            continue
        assert all(refusals) or not any(refusals), f"{op.theories} {op.name}"


def test_the_scalar_column_falls_back_only_where_there_is_something_to_fall_back_to():
    """`ViaMat` is a claim about the matrix cell.

    `translate_terms_scalar` reaches for the matrix form whenever the
    scalar cell is not an emitter, so a `ViaMat` cell needs a matrix
    emitter behind it."""
    for op in OPS:
        if isinstance(op.scalar, ViaMat):
            assert callable(op.mat), f"{op.name}: ViaMat with no matrix form"


# ══════════════════════════════════════════════════════════════════════
#  The gaps, pinned
# ══════════════════════════════════════════════════════════════════════

_LEAN = ("mat", "scalar", "box")

# Every (theory, op, column) with no emitter. Closing one of these is a line
# removed here; opening a new one is a line added, which is the point —
# `lowering/ops.py` records *why* each is open, and this says how many there
# are.
_KNOWN_GAPS = {
    # Nondeterministic choice, and the zero rate of change: every encoding
    # here is a function of the read wires, in discrete time.
    *(("", name, column) for name in ("Havoc", "Zero") for column in COLUMNS),
    # `Xor` has no Lean emitter.
    *(("bool", "Xor", c) for c in _LEAN),
    # No `Box` in `Relational.Circ`: the pointwise `Abs`, and the reduction
    # `All`, which has the other two Lean forms.
    *((t, "Abs", c) for t in ("int", "real") for c in _LEAN),
    ("bool", "All", "box"),
    # Words with no Lean form.
    *(("word", name, c)
      for name in ("Concat", "Extract", "Nand", "Nor", "Repeat", "RotateLeft",
                   "RotateRight", "Shl", "Xnor")
      for c in _LEAN),
    *(("word.signed", name, c)
      for name in ("Abs", "Div", "Extend", "Ge", "Gt", "Le", "Lt", "Rem", "Shr")
      for c in _LEAN),
    *(("word.unsigned", name, c)
      for name in ("Div", "Extend", "Ge", "Gt", "Le", "Lt", "Shr")
      for c in _LEAN),
}


def test_the_coverage_gaps_are_the_known_ones():
    gaps = {
        (theory, op.name, column)
        for theory, op in ops.rows()
        for column in COLUMNS
        if isinstance(cell_of(op, column), Unsupported)
    }
    assert gaps == _KNOWN_GAPS, (
        "the coverage matrix moved: "
        f"new gaps {sorted(gaps - _KNOWN_GAPS)}, "
        f"closed gaps {sorted(_KNOWN_GAPS - gaps)}. "
        "Run `python -m reactor_lean.lowering.ops` to see the whole matrix."
    )


def test_the_matrix_prints():
    """`python -m reactor_lean.lowering.ops` is how the table is meant to be
    read."""
    text = ops.format_matrix()
    assert "gaps:" in text
    for theory in THEORIES:
        assert theory in text


# ══════════════════════════════════════════════════════════════════════
#  A gap, end to end
# ══════════════════════════════════════════════════════════════════════


def _word_ult_module() -> Module:
    """Two variables, one an unsigned comparison of words, which Lean has
    not."""
    s = Var(Word(8, S))
    flag = Var(Bool(S))
    init = [
        Term(tensor.word.Const(torch.tensor([[3]]), 8), [X(s)]),
        Term(tensor.bool.Const(torch.tensor([[False]])), [X(flag)]),
    ]
    update = [
        Term(Skip(Word(8, S)), [X(s)], [s]),
        Term(tensor.word.unsigned.Lt(8, S), [X(flag)], [s, s]),
    ]
    return Module.sequential([s, flag], init, update)


@pytest.mark.parametrize(
    "encoding", ["to_lean_functional", "to_lean_scalar", "to_lean_circ"]
)
def test_a_missing_lean_form_is_refused_with_the_recorded_reason(encoding):
    """The cell says whether the op was forgotten, is unimplementable, or is
    handled somewhere else, and that reason is what reaches the caller."""
    m2l = ModuleToLean4(_word_ult_module())
    with pytest.raises(ValueError) as excinfo:
        getattr(m2l, encoding)()
    message = str(excinfo.value)
    assert message.startswith("No Lean expression mapping for: Lt")
    assert "no Lean form" in message


def test_an_op_no_backend_has_names_its_own_reason():
    with pytest.raises(ValueError, match="nondeterministic"):
        mat_emitter(Havoc(Int(S)))


# ══════════════════════════════════════════════════════════════════════
#  The same gap, asked of a whole module before it is generated
# ══════════════════════════════════════════════════════════════════════


def _havoc_module(read_it: bool = True):
    """A lone `Havoc` write, read by the update when `read_it`."""
    x = Var(Int(S))
    y = Wire(Int(S))
    init = [Term(tensor.int.Const(torch.tensor([[0]])), [X(x)])]
    update = [Term(Havoc(Int(S)), [y]), Term(Skip(Int(S)), [X(x)], [y if read_it else x])]
    return Module.sequential([x], init, update)


def test_a_module_with_no_lean_form_says_which_wire_and_why():
    """The reason the table records, about the module rather than the table:
    which block, which wire, and why."""
    gaps = lean_gaps(LeanContext(_havoc_module()))
    assert len(gaps) == 1
    assert "`update` block" in gaps[0] and "`Havoc`" in gaps[0]
    assert "wire #" in gaps[0] and "nondeterministic" in gaps[0]


def test_a_module_that_generates_reports_no_gap():
    """The check cannot refuse what codegen accepts, so it says so here."""
    assert lean_gaps(LeanContext(_word_mod_module(tensor.word.unsigned.Rem(8, S)))) == []


def test_a_term_nothing_reads_is_a_gap_too():
    """Nothing reads `y`, but the functional encoding translates every
    term, so it is reported up front rather than met half way through."""
    ctx = LeanContext(_havoc_module(read_it=False))
    gaps = lean_gaps(ctx)
    assert len(gaps) == 1 and "`Havoc`" in gaps[0]
    with pytest.raises(ValueError, match="Havoc"):
        ModuleToLean4(ctx).to_lean_functional()


def _cells(mat, scalar, box) -> ops.Op:
    """A row of the shape under test. No theory claims it, so nothing can
    reach it through `op_for`; `gap_of` is asked about it directly."""
    return ops.Op("Fake", (), mat=mat, scalar=scalar, box=box)


def test_a_gap_is_asked_of_every_lean_column_not_only_the_matrix_one():
    """Every row with no matrix form has no other Lean form either, so
    reading `mat` alone gets the right answer for all of them. That is a
    fact about the table's contents, and this check refuses a *module* on
    the strength of it -- so the day a row has a `Box` and no matrix form,
    reading `mat` alone would refuse a module the circuit encoding builds.
    """
    unsupported = ops.Unsupported("no matrix form")
    see_mat = ops.Unsupported("see the matrix column")

    assert ops.gap_of(_cells(unsupported, see_mat, see_mat)) == "no matrix form"
    assert ops.gap_of(_cells(unsupported, see_mat, "Box.fake")) is None
    assert ops.gap_of(_cells(unsupported, lambda a: a[0], see_mat)) is None
    # `ViaMat` is served by the matrix form, which is the gap -- so it is
    # not an escape from one, unlike `Inline`, whose branch is reached
    # before the table is asked at all.
    assert ops.gap_of(_cells(unsupported, ops.ViaMat("folds to 1x1"), see_mat)) \
        == "no matrix form"
    assert ops.gap_of(_cells(unsupported, ops.Inline("codegen.terms._special"), see_mat)) is None


# ══════════════════════════════════════════════════════════════════════
#  Dispatch
# ══════════════════════════════════════════════════════════════════════


def test_ne_is_dispatchable():
    """`!=` is spelled `Ne` in every theory."""
    op = tensor.int.Ne(S)
    assert itype_name(op) == "Ne"
    assert callable(mat_emitter(op))
    assert callable(scalar_emitter(op))
    assert box_for(op) == "Box.neq"


def _make_ne_module():
    """Sequential module whose update compares two counters with `!=`."""
    x = Var(Int(S))
    y = Var(Int(S))
    differ = Var(Bool(S))

    init = [
        Term(tensor.int.Const(torch.tensor([[0]])), [X(x)]),
        Term(tensor.int.Const(torch.tensor([[1]])), [X(y)]),
        Term(tensor.bool.Const(torch.tensor([[False]])), [X(differ)]),
    ]
    update = [
        Term(Skip(Int(S)), [X(x)], [x]),
        Term(Skip(Int(S)), [X(y)], [y]),
        Term(tensor.int.Ne(S), [X(differ)], [x, y]),
    ]
    return Module.sequential([x, y, differ], init, update)


def test_ne_emits_in_functional_encoding():
    lean = ModuleToLean4(_make_ne_module()).to_lean_functional()
    assert "≠" in lean, "Ne did not reach the functional encoding"


def test_ne_emits_in_scalar_encoding():
    lean = ModuleToLean4(_make_ne_module()).to_lean_scalar()
    assert "≠" in lean, "Ne did not reach the scalar encoding"


def test_ne_emits_in_circuit_encoding():
    lean = ModuleToLean4(_make_ne_module()).to_lean_circ()
    assert "Box.neq" in lean, "Ne did not reach the circuit (Box) encoding"


# ──────────────────────────────────────────────────────────────
# Words: BitVec operators for the bitwise ops, Bool for the rest
# ──────────────────────────────────────────────────────────────

# The bitwise ops of words have rows of their own, with BitVec's operators;
# a comparison of words writes a Bool and an `Ite` of words reads one, so
# those share the rows of the other theories.
_WORD_BITWISE = ["Not", "And", "Or", "Xor"]


@pytest.mark.parametrize("name", _WORD_BITWISE)
def test_word_bitwise_ops_get_their_own_row(name):
    """A row shared with `bool` would emit `!`/`&&`/`||` against a `BitVec`."""
    row = _row("word", name)
    assert row is not _row("bool", name)
    for column in _LEAN:
        assert not isinstance(cell_of(row, column), Unsupported)
    assert box_for(getattr(tensor.word, name)(8, S)).startswith("Box.bv")


@pytest.mark.parametrize("name", ["Ite", "Eq", "Ne"])
def test_words_choose_and_compare_through_bool(name):
    """A word comparison writes a Bool and a word `Ite` reads one: the rows
    of the other theories serve them."""
    assert _row("word", name) is _row("bool", name) is _row("int", name)


def test_word_comparison_and_choice_read_and_write_bool():
    """Pins the premise of the shared rows."""
    w = Word(8, S)
    Term(tensor.word.Eq(8, S), [Wire(Bool(S))], [Wire(w), Wire(w)])
    Term(tensor.word.Ite(8, S), [Wire(w)], [Wire(Bool(S)), Wire(w), Wire(w)])
    with pytest.raises(Exception):
        Term(tensor.word.Ite(8, S), [Wire(w)], [Wire(Word(1, S)), Wire(w), Wire(w)])


def _word_bits_module():
    """Two 1-bit words: `b0` toggles, and `b1` takes `b0 & ~b1`."""
    b0, b1 = Var(Word(1, S)), Var(Word(1, S))
    n0, n1 = Wire(Word(1, S)), Wire(Word(1, S))
    init = [Term(tensor.word.Const(torch.tensor([[0]]), 1), [X(b)]) for b in (b0, b1)]
    update = [
        Term(tensor.word.Not(1, S), [n0], [b0]),
        Term(Skip(Word(1, S)), [X(b0)], [n0]),
        Term(tensor.word.Not(1, S), [n1], [b1]),
        Term(tensor.word.And(1, S), [X(b1)], [b0, n1]),
    ]
    return Module.sequential([b0, b1], init, update)


def test_word_module_emits_bitvec_operators_not_boolean_ones():
    """`!`/`&&`/`||` do not elaborate on `BitVec 1`."""
    lean = ModuleToLean4(_word_bits_module()).to_lean_functional()
    assert "~~~" in lean, "word Not did not use BitVec complement"
    assert "&&&" in lean, "word And did not use BitVec conjunction"
    for boolean in (" && ", " || ", "=> !("):
        assert boolean not in lean, f"Boolean form {boolean!r} emitted for words"


def test_word_module_emits_bitvec_boxes():
    lean = ModuleToLean4(_word_bits_module()).to_lean_circ()
    assert "Box.bvNot" in lean and "Box.bvAnd" in lean
    # Only the composed layers matter: `_LAYER_SIMP` lists every Box name as a
    # simp lemma regardless of which ones this module uses.
    layers = [l for l in lean.splitlines() if "⊗" in l or ("Box." in l and ":=" in l)]
    assert layers, "no composed layers emitted"
    for line in layers:
        for boolean in ("Box.not", "Box.and", "Box.or"):
            assert f"{boolean} " not in line and not line.rstrip().endswith(boolean), (
                f"Boolean box {boolean} emitted for words in: {line.strip()[:90]}"
            )


def test_simp_circ_unfolds_every_box_a_circuit_holds():
    """Each Box in a layer is one `simp_circ` unfolds: every module of the
    corpus, and the word one above, whose `bvNot` and `bvAnd` the list
    once missed."""
    import re
    from agree import CORPUS
    from reactor_lean.codegen.encodings.circuit import _LAYER_SIMP

    builds = {**CORPUS, "word_bits": _word_bits_module}
    seen = set()
    for name, build in builds.items():
        try:
            lean = ModuleToLean4(LeanContext(build())).to_lean_circ()
        except ValueError:
            continue                    # no circuit: `lean_gaps`, `_box`
        used = set(re.findall(r"Box\.\w+", lean)) - {"Box.seq"}
        assert used <= set(_LAYER_SIMP), (name, used - set(_LAYER_SIMP))
        seen |= used
    assert {"Box.bvNot", "Box.bvAnd", "Box.get", "Box.linear"} <= seen


@pytest.mark.parametrize(
    "op,box",
    [(tensor.word.unsigned.Rem(8, S), "Box.bvUMod"), (tensor.word.signed.Mod(8, S), "Box.bvSMod")],
)
def test_word_modulo_is_dispatchable(op, box):
    """Words have an unsigned remainder and a signed modulo, each with a Lean
    form; `int`'s Euclidean `Mod` is a row of its own."""
    assert callable(mat_emitter(op)) and callable(scalar_emitter(op))
    assert box_for(op) == box
    assert _row("int", "Mod") is not op_for(op)


def test_add_is_one_row_for_every_number():
    assert box_for(tensor.word.Add(8, S)) == "Box.add"
    assert _row("word", "Add") is _row("int", "Add") is _row("real", "Add")


def _word_mod_module(op):
    """Words whose update takes a modulo of the state."""
    s = Var(Word(8, S))
    d = Var(Word(8, S))
    init = [
        Term(tensor.word.Const(torch.tensor([[7]]), 8), [X(s)]),
        Term(tensor.word.Const(torch.tensor([[3]]), 8), [X(d)]),
    ]
    update = [Term(Skip(Word(8, S)), [X(d)], [d]), Term(op, [X(s)], [s, d])]
    return Module.sequential([s, d], init, update)


@pytest.mark.parametrize(
    "op,expected",
    [(tensor.word.unsigned.Rem(8, S), "BitVec.umod"), (tensor.word.signed.Mod(8, S), "BitVec.smod")],
)
def test_word_modulo_emits_the_right_lean(op, expected):
    lean = ModuleToLean4(_word_mod_module(op)).to_lean_functional()
    assert expected in lean


# ──────────────────────────────────────────────────────────────
# `Ite` picks whole matrices; `Scale` is an affine map
# ──────────────────────────────────────────────────────────────


def _ite_module(n: int, *, by_all: bool = False):
    """An `n`-vector of counters, all reset to 0 once the first is past 3,
    or, `by_all`, once every one is."""
    sh = [n, 1]
    v = Var(Int(sh))
    zero, one, inc = (Wire(Int(sh)) for _ in range(3))
    over = Wire(Bool([1, 1]))

    def c(k, shape):
        return tensor.int.Const(torch.full(shape, k, dtype=torch.int64))

    if by_all:
        three, each = Wire(Int(sh)), Wire(Bool(sh))
        test = [
            Term(c(3, sh), [three]),
            Term(tensor.int.Gt(sh), [each], [v, three]),
            Term(tensor.bool.All(sh), [over], [each]),
        ]
    else:
        first, three = Wire(Int([1, 1])), Wire(Int([1, 1]))
        test = [
            Term(tensor.int.Get(0, 0, sh), [first], [v]),
            Term(c(3, [1, 1]), [three]),
            Term(tensor.int.Gt([1, 1]), [over], [first, three]),
        ]
    init = [Term(c(0, sh), [X(v)])]
    update = [
        *test,
        Term(c(0, sh), [zero]), Term(c(1, sh), [one]),
        Term(tensor.int.Add(sh), [inc], [v, one]),
        Term(tensor.int.Ite(sh), [X(v)], [over, zero, inc]),
    ]
    return Module.sequential([v], init, update)


@pytest.mark.parametrize("n", [1, 2])
def test_an_ite_picks_whole_matrices_by_a_1x1_condition(n):
    """The form, and the `Box`, of an `Ite`, whatever the shape of its
    branches."""
    m2l = ModuleToLean4(_ite_module(n))
    assert " 0 0 then " in m2l.to_lean_functional()
    assert "Box.ite" in m2l.to_lean_circ()


def test_all_reduces_a_comparison_of_vectors_to_the_condition_of_an_ite():
    """The comparison is written entry by entry, and `All` decides that
    every entry holds."""
    lean = ModuleToLean4(_ite_module(2, by_all=True)).to_lean_functional()
    assert "(decide (" in lean and " i j) < (" in lean
    assert "decide (∀ p : _ × _, " in lean
    assert " 0 0 then " in lean


def test_all_has_no_box():
    """Of the booleans of a variable, so that no comparison is refused first."""
    b, holds = Var(Bool([2, 1])), Var(Bool([1, 1]))
    init = [Term(tensor.bool.Const(torch.zeros(s, dtype=torch.bool)), [X(v)])
            for v, s in ((b, (2, 1)), (holds, (1, 1)))]
    update = [
        Term(Skip(Bool([2, 1])), [X(b)], [b]),
        Term(tensor.bool.All([2, 1]), [X(holds)], [b]),
    ]
    m2l = ModuleToLean4(Module.sequential([b, holds], init, update))
    assert "decide (∀ p : _ × _, " in m2l.to_lean_functional()
    with pytest.raises(ValueError, match=r"All \(Box\) — no `Box` in `Relational.Circ`"):
        m2l.to_lean_circ()


def test_a_comparison_wider_than_1x1_compares_every_entry():
    """Not the entry `0 0` repeated: `decide ((a i j) = (b i j))`."""
    a, b = Var(Int([3, 1])), Var(Int([3, 1]))
    e = Var(Bool([3, 1]))
    init = [Term(tensor.int.Const(torch.zeros(3, 1, dtype=torch.int64)), [X(v)]) for v in (a, b)]
    init.append(Term(tensor.bool.Const(torch.zeros(3, 1, dtype=torch.bool)), [X(e)]))
    update = [
        Term(Skip(Int([3, 1])), [X(a)], [a]),
        Term(Skip(Int([3, 1])), [X(b)], [b]),
        Term(tensor.int.Eq([3, 1]), [X(e)], [a, b]),
    ]
    lean = ModuleToLean4(Module.sequential([a, b, e], init, update)).to_lean_functional()
    assert "Matrix.mk fun i j => (decide (" in lean and " i j) = (" in lean
    assert " 0 0 = " not in lean


def test_scale_is_written_as_the_affine_map_diag_c():
    x = Var(Int([2, 1]))
    init = [Term(tensor.int.Const(torch.ones(2, 1, dtype=torch.int64)), [X(x)])]
    update = [Term(tensor.int.Scale(torch.tensor([[2], [3]])), [X(x)], [x])]
    lean = all_encodings(Module.sequential([x], init, update))
    assert "matVecAffine 2 ([[(2 : Int), (0 : Int)], [(0 : Int), (3 : Int)]]" in lean
    assert "Box.linear 2" in lean


def test_scale_by_rows_of_two_numbers_has_no_lean_form():
    x = Var(Int([1, 2]))
    init = [Term(tensor.int.Const(torch.ones(1, 2, dtype=torch.int64)), [X(x)])]
    update = [Term(tensor.int.Scale(torch.tensor([[2, 3]])), [X(x)], [x])]
    with pytest.raises(ValueError, match="no `diag"):
        ModuleToLean4(Module.sequential([x], init, update)).to_lean_functional()
