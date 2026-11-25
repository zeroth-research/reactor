"""Tests for the eager Expression (`reactor.expr`).

Covers `expr()`, the `IExpr`/`RExpr`/`BExpr`/`SExpr`/`UExpr` split, explicit-sort literals, signed/unsigned
word ops, no-implicit-promotion (+ `cast`), the `collecting()` term collector, and eval.
"""

import torch
import pytest

import reactor
from reactor import Wire, T, Var, TheoryError
from reactor.matrix.sort import Bool, Int, Real, Word
from reactor import matrix
from reactor.eval import eval_itype
from reactor.expr import expr, cast, ite, collecting, IExpr, RExpr, BExpr, SExpr, UExpr, X

INT = Int([1, 1])
REAL = Real([1, 1])
BV32 = Word(32, [1, 1])


@pytest.fixture(autouse=True)
def _collector():
    # Building a Term requires an active collector; wrap every test in one. Tests that
    # need the captured terms open their own nested `with collecting() as terms:`.
    with collecting():
        yield


# --- helpers ----------------------------------------------------------------


def _var(sort):
    return Var(sort)


def _varexpr(sort, signed=False):
    return expr(_var(sort), signed=signed)


def _run(terms, state):
    for t in terms:
        read = [state[w] for w in t.read]
        out_sort = t.write[0].dtype if len(t.write) else None
        for w, val in zip(t.write, eval_itype(t.itype, read, out_sort)):
            state[w] = val
    return state


def _int(n):
    return torch.tensor([[n]], dtype=torch.int64)


# --- expr() factory & guards ------------------------------------------------


def test_factory_picks_subclass_from_sort():
    assert isinstance(expr(_var(INT)), IExpr)
    assert isinstance(expr(_var(REAL)), RExpr)
    assert isinstance(expr(True, sort=Bool), BExpr)
    assert isinstance(expr(_var(BV32)), UExpr)
    assert isinstance(expr(_var(BV32), signed=True), SExpr)


def test_expr_is_not_idempotent():
    e = expr(True)
    with pytest.raises(TypeError):
        expr(e)


def test_numeric_literal_needs_explicit_sort():
    with pytest.raises(TypeError):
        expr(3)
    assert isinstance(expr(3, sort=Real), RExpr)
    assert isinstance(expr(3, sort=Int), IExpr)
    assert isinstance(expr(True), BExpr)  # bool is exempt


def tensor_of_term(term):
    match term.itype:
        case matrix.real.Const(t) | matrix.int.Const(t) | matrix.bool.Const(t) | matrix.word.Const(t):
            return t
        case _:
            return None


def test_literals_take_the_element_kind_of_their_sort():
    # the constant operations require it: floating point for reals, int64 for integers
    with collecting() as terms:
        expr(3, sort=REAL)  # scalar int as Real -> float
        assert tensor_of_term(terms[0]).dtype == torch.float

    with pytest.raises(TypeError):
        expr(True, sort=INT)  # scalar bool as Int -> not a number

    with pytest.raises(TypeError):
        expr(True, sort=REAL)  # scalar bool as Real -> not a number

    with collecting() as terms:
        expr(torch.tensor([[2]]), sort=REAL)  # tensor -> no conversion
        assert tensor_of_term(terms[0]).dtype == torch.int64


def test_lossless_value_conversions_are_not_happening():
    with collecting() as terms:
        expr(3, sort=REAL)  # int as Real -> float
        assert tensor_of_term(terms[0]).dtype == torch.float

    with pytest.raises(TypeError):
        expr(True, sort=INT)  # bool as Int -> not a number

    with pytest.raises(TypeError):
        expr(True, sort=REAL)  # bool as Real -> not a number

    with collecting() as terms: # tensor -> no conversion
        expr(torch.tensor([[2]]), sort=REAL)
        assert tensor_of_term(terms[0]).dtype == torch.long

    # assert expr(True, sort=INT)._value.dtype is torch.int64  # bool -> Int
    # assert expr(True, sort=REAL)._value.dtype is torch.float32  # bool -> Real
    # # an int tensor on a Real wire is converted, not left as int
    # assert expr(torch.tensor([[2]]), sort=REAL)._value.dtype is torch.float32


def test_explicit_sort_wins_for_bool_value():
    # a bool is not a number: an explicit numeric sort is an error, not a conversion
    with pytest.raises(TypeError):
        expr(True, sort=Int([1, 1]))
    with pytest.raises(TypeError):
        expr([True, False], sort=Real)
    assert isinstance(expr(True), BExpr)  # no sort -> Bool


# --- variables & nxt --------------------------------------------------------


def test_variable_reads_latched_and_nxt():
    pair = _var(INT)
    x = expr(pair)
    assert x.wire is pair
    assert X(x).wire == reactor.X(pair)


def test_nxt_requires_a_variable():
    x = _varexpr(INT)
    with pytest.raises(TypeError):
        X(x + 1)


# --- result sorts -----------------------------------------------------------


def test_arith_and_compare_result_sorts():
    x, y = _varexpr(INT), _varexpr(INT)
    assert isinstance(x + y, IExpr) and (x + y).dtype == INT
    assert isinstance(x < y, BExpr) and (x < y).dtype == Bool([1, 1])
    assert isinstance(x == y, BExpr)


def test_real_and_word_result_sorts():
    xr = _varexpr(REAL)
    assert (xr + 1.0).dtype == REAL
    xb, yb = _varexpr(BV32), _varexpr(BV32)
    assert (xb + 1).dtype == BV32
    # word comparisons write booleans
    assert isinstance(xb < yb, BExpr) and (xb < yb).dtype == Bool([1, 1])


def test_ops_are_built_from_the_operands():
    x, y = _varexpr(Real([2, 3])), _varexpr(Real([2, 3]))
    with collecting() as terms:
        x + y
    assert terms[-1].itype == matrix.real.Add([Real([2, 3]), Real([2, 3])])
    w, v = _varexpr(Word(8, [1, 2])), _varexpr(Word(8, [1, 2]))
    with collecting() as terms:
        w + v
    assert terms[-1].itype == matrix.word.Add([Word(8, [1, 2]), Word(8, [1, 2])])


# --- equality builds predicates; truth-testing is refused -------------------


def test_equality_operators_build_predicates():
    x, y = _varexpr(INT), _varexpr(INT)
    with collecting() as terms:
        e = x == y
    assert isinstance(e, BExpr) and isinstance(terms[-1].itype, matrix.int.Eq)
    with collecting() as terms:
        e = x != y
    assert isinstance(e, BExpr) and isinstance(terms[-1].itype, matrix.int.Ne)


def test_boolean_equality():
    p, q = _varexpr(Bool([1, 1])), _varexpr(Bool([1, 1]))
    with collecting() as terms:
        e = p == q
    assert [type(t.itype) for t in terms] == [matrix.bool.Eq]
    t, f = torch.tensor([[True]]), torch.tensor([[False]])
    assert _run(terms, {p.wire: t, q.wire: t})[e.wire].item() is True
    assert _run(terms, {p.wire: t, q.wire: f})[e.wire].item() is False
    with collecting() as terms:
        e = p != q
    assert [type(t.itype) for t in terms] == [matrix.bool.Ne]
    assert _run(terms, {p.wire: t, q.wire: t})[e.wire].item() is False
    assert _run(terms, {p.wire: t, q.wire: f})[e.wire].item() is True


def test_equality_coerces_a_raw_operand():
    x = _varexpr(INT)
    with collecting() as terms:
        e = x == 3
    assert isinstance(e, BExpr) and isinstance(terms[-1].itype, matrix.int.Eq)


def test_truth_testing_raises():
    x, y = _varexpr(INT), _varexpr(INT)
    for thunk in (lambda: bool(x),
                  lambda: "t" if (x < y) else "f",
                  lambda: "t" if (x == y) else "f",
                  lambda: x and y,
                  lambda: x or y,
                  lambda: not x):
        with pytest.raises(TypeError, match="no truth value"):
            thunk()


def test_expr_is_not_hashable():
    x = _varexpr(INT)
    with pytest.raises(TypeError, match="unhashable"):
        {x: 1}


# --- no implicit promotion; cast is explicit --------------------------------


def test_mixed_sorts_raise():
    with pytest.raises(TypeError):
        _varexpr(INT) + expr(True)  # Int + Bool
    with pytest.raises(TypeError):
        ite(expr(True), _varexpr(INT), _varexpr(REAL))  # Int vs Real branches


def test_cast_identity_and_unsupported():
    x = _varexpr(INT)
    assert cast(x, Int) is x
    assert cast(x, INT) is x
    # a concrete sort must be of the shape of the expression
    with pytest.raises(TypeError):
        cast(x, Int([2, 2]))
    with pytest.raises(NotImplementedError):
        cast(x, Real)


# --- signed / unsigned word ops ---------------------------------------------


def test_word_signedness_picks_op():
    xs, ys = _varexpr(BV32, signed=True), _varexpr(BV32, signed=True)
    with collecting() as terms:
        xs < ys
    assert isinstance(terms[-1].itype, matrix.word.signed.Lt)

    xu, yu = _varexpr(BV32), _varexpr(BV32)
    with collecting() as terms:
        xu < yu
    assert isinstance(terms[-1].itype, matrix.word.unsigned.Lt)


def test_coercion_inherits_signedness_regardless_of_operand_order():
    # a coerced literal takes the variable's signedness, so `5 + xs` and `xs + 5`
    # both produce a *signed* comparison for a signed word `xs`.
    xs = _varexpr(BV32, signed=True)
    with collecting() as terms:
        (xs + 5) < 3
    assert isinstance(terms[-1].itype, matrix.word.signed.Lt)
    with collecting() as terms:
        (5 + xs) < 3
    assert isinstance(terms[-1].itype, matrix.word.signed.Lt)


def test_mixing_signed_and_unsigned_words_raises():
    xs = _varexpr(BV32, signed=True)
    yu = _varexpr(BV32, signed=False)
    with pytest.raises(TypeError, match="signed and an unsigned"):
        xs + yu


# --- collector --------------------------------------------------------------


def test_collecting_records_deps_first():
    x, y = _varexpr(INT), _varexpr(INT)
    with collecting() as terms:
        ite(x < y, x + 1, y)
    optypes = [type(t.itype) for t in terms]
    assert optypes == [matrix.int.Lt, matrix.int.Const, matrix.int.Add, matrix.int.Ite]


def test_shared_subexpression_recorded_once():
    x, y = _varexpr(INT), _varexpr(INT)
    with collecting() as terms:
        g = x < y
        ite(g, x + 1, x)
        ite(g, y, y)
    assert sum(isinstance(t.itype, matrix.int.Lt) for t in terms) == 1


# --- evaluation end-to-end --------------------------------------------------


def test_eval_arith():
    x = _varexpr(INT)
    with collecting() as terms:
        e = x + 1
    assert _run(terms, {x.wire: _int(5)})[e.wire].item() == 6


def test_eval_ite_both_branches():
    x, y = _varexpr(INT), _varexpr(INT)
    with collecting() as terms:
        e = ite(x < y, x + 1, y)
    assert _run(terms, {x.wire: _int(1), y.wire: _int(5)})[e.wire].item() == 2
    assert _run(terms, {x.wire: _int(5), y.wire: _int(1)})[e.wire].item() == 1


@pytest.mark.parametrize("n", [1, 2, 3])
def test_scaling_a_column_vector_by_a_constant(n):
    x = _varexpr(Int([n, 1]))
    with collecting() as terms:
        e = x * 3
    assert isinstance(terms[-1].itype, matrix.int.Scale)
    vals = torch.arange(1, n + 1, dtype=torch.int64).reshape(n, 1)
    got = _run(terms, {x.wire: vals})[e.wire]
    assert got.flatten().tolist() == [3 * v for v in range(1, n + 1)]


def test_mul_by_constant_folds_to_scale():
    x = _varexpr(INT)
    with collecting() as terms:
        e = x * 2
    assert isinstance(terms[-1].itype, matrix.int.Scale)
    assert _run(terms, {x.wire: _int(3)})[e.wire].item() == 6


@pytest.mark.parametrize("left", [False, True])
def test_mul_by_a_full_tensor_is_a_pointwise_scale(left):
    # a constant of the shape of the variable, with distinct entries: no `A·x` computes it
    x = _varexpr(Real([2, 3]))
    c = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with collecting() as terms:
        e = c * x if left else x * c
    assert len(terms) == 1
    assert isinstance(terms[0].itype, matrix.real.Scale)
    assert terms[0].itype.tensor.data_ptr() == c.data_ptr()  # shared, not copied
    assert e.dtype == Real([2, 3])
    vals = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    assert torch.equal(_run(terms, {x.wire: vals})[e.wire], c * vals)


def test_neg_and_abs():
    x = _varexpr(INT)
    with collecting() as terms:
        n, a = -x, abs(x)
    assert isinstance(terms[0].itype, matrix.int.Neg) and isinstance(terms[1].itype, matrix.int.Abs)
    state = _run(terms, {x.wire: _int(-3)})
    assert state[n.wire].item() == 3 and state[a.wire].item() == 3

    r = _varexpr(REAL)
    with collecting() as terms:
        m, b = -r, abs(r)
    assert isinstance(terms[0].itype, matrix.real.Neg) and isinstance(terms[1].itype, matrix.real.Abs)
    state = _run(terms, {r.wire: torch.tensor([[-2.5]])})
    assert state[m.wire].item() == 2.5 and state[b.wire].item() == 2.5


def test_real_operations_take_the_degree_of_their_operands():
    x = expr(Var(Real([1, 1])))
    dx = expr(Wire(T(Real([1, 1]))))
    with collecting() as terms:
        s = dx + 1  # the literal takes the degree of `dx`
        assert tensor_of_term(terms[-2]) is not None and terms[-2].itype.degree == 1
        assert terms[-1].itype.degree == 1 and s.dtype == T(Real([1, 1]))
        p = x * dx  # degrees add up
        assert list(terms[-1].itype.degrees) == [0, 1] and p.dtype == T(Real([1, 1]))
        q = dx / x  # degrees are taken off
        assert list(terms[-1].itype.degrees) == [1, 0] and q.dtype == T(Real([1, 1]))
        c = dx < 1  # compared at their degree
        assert terms[-1].itype.degree == 1 and c.dtype == Bool([1, 1])
        h = abs(dx)
        assert terms[-1].itype.degree == 1 and h.dtype == T(Real([1, 1]))


def test_division_by_a_real_of_a_higher_degree_is_not_defined():
    x = expr(Var(Real([1, 1])))
    dx = expr(Wire(T(Real([1, 1]))))
    with pytest.raises(ValueError, match="negative degree"):
        x / dx


def test_neg_keeps_the_degree_of_reals():
    x = expr(Var(Real([1, 1])))
    d1 = expr(Wire(T(Real([1, 1]))))
    with collecting() as terms:
        n, m = -x, -d1
    assert terms[0].itype.degree == 0 and n.dtype == Real([1, 1])
    assert terms[1].itype.degree == 1 and m.dtype == T(Real([1, 1]))


# theory errors are deprecated - the python API has become permissive
# def test_mul_of_two_variables_is_nonlinear():
#     x, y = _varexpr(INT), _varexpr(INT)
#     with pytest.raises(TheoryError):
#         _ = x * y


def test_mul_of_two_variables():
    x, y = _varexpr(INT), _varexpr(INT)
    with collecting() as terms:
        e = x * y
    assert isinstance(terms[-1].itype, matrix.int.Mul)
    assert _run(terms, {x.wire: _int(3), y.wire: _int(4)})[e.wire].item() == 12


def test_matmul():
    a, b = _varexpr(Real([2, 3])), _varexpr(Real([3, 4]))
    with collecting() as terms:
        e = a @ b
    assert terms[-1].itype == matrix.real.MatMul([Real([2, 3]), Real([3, 4])])
    assert e.dtype == Real([2, 4])
    av, bv = torch.rand(2, 3), torch.rand(3, 4)
    assert torch.allclose(_run(terms, {a.wire: av, b.wire: bv})[e.wire], av @ bv)


# --- single entry point: const / var / wire all go through expr() -----------


def test_single_entry_point_spec():
    """The one `expr()` handles every leaf; there is no separate var/const/pair API,
    and no string distinguishes a variable. (Spec from the `marco/dsl` review.)"""
    BOOL = Bool([1, 1])

    # a wire pair is not a leaf
    with pytest.raises(TypeError):
        expr((Wire(BOOL), Wire(BOOL)))

    # variable, bool literal and bare wire -> all BExpr (Bool sort)
    x = expr(Var(BOOL))
    y = expr(True)
    w = expr(Wire(BOOL))
    assert isinstance(x, BExpr) and isinstance(y, BExpr) and isinstance(w, BExpr)

    # boolean composition stays BExpr; a raw operand coerces through expr()
    assert isinstance(x & y & w, BExpr)
    assert isinstance(x & y & w & True, BExpr)

    # a numeric literal needs an explicit sort; with one it is an RExpr
    with pytest.raises(TypeError):
        expr(0.3)
    b = expr(0.3, sort=Real)
    assert isinstance(b, RExpr)
    assert isinstance(b + 1, RExpr)  # raw operand coerces

    # list / tensor constants; a raw operand of matmul is coerced to my sort, shape
    # included, so a factor of another shape is built explicitly
    t = expr([0.1, 1.0], sort=Real)
    assert isinstance(t, RExpr)
    with pytest.raises(ValueError):  # the constant rejects the codomain of my shape
        t @ torch.tensor([[0.2], [0.5]])
    u = t @ expr(torch.tensor([[0.2], [0.5]]), sort=Real)
    assert isinstance(u, RExpr) and u.dtype == Real([1, 1])

    # not idempotent: passing an Expr back into expr() is an error
    with pytest.raises(TypeError):
        expr(u)


def test_tag_is_a_label_with_no_logical_meaning():
    # `tag` is optional, purely a display label; it does not affect the class or ops.
    p = Var(Bool([1, 1]))
    x = expr(p, tag="x")
    assert x.tag == "x"
    assert isinstance(x, BExpr)
    assert "x" in repr(x)
    # untagged leaves and derived expressions simply have no tag
    assert expr(True).tag is None
    assert (x & True).tag is None
