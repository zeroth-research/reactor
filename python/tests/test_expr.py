"""Tests for the eager Expression (`reactor.expr`).

Covers `expr()`, the `IExpr`/`RExpr`/`BExpr`/`SExpr`/`UExpr` split, explicit-sort literals, signed/unsigned
word ops, no-implicit-promotion (+ `cast`), the `collecting()` term collector, and eval.
"""

import torch
import pytest

import reactor
from reactor import Wire, T, Var, TheoryError
import reactor as rr
from reactor.bool import Bool
from reactor.int import Int
from reactor.real import Real
from reactor.word import Word
from reactor import real, word
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
        case real.Num(t) | rr.int.Num(t) | rr.bool.Truth(t) | word.Num(t):
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
    assert terms[-1].itype == real.Add([Real([2, 3]), Real([2, 3])])
    w, v = _varexpr(Word(8, [1, 2])), _varexpr(Word(8, [1, 2]))
    with collecting() as terms:
        w + v
    assert terms[-1].itype == word.Add([Word(8, [1, 2]), Word(8, [1, 2])])


# --- equality builds predicates; truth-testing is refused -------------------


def test_equality_operators_build_predicates():
    x, y = _varexpr(INT), _varexpr(INT)
    with collecting() as terms:
        e = x == y
    assert isinstance(e, BExpr) and isinstance(terms[-1].itype, rr.int.Eq)
    with collecting() as terms:
        e = x != y
    assert isinstance(e, BExpr) and isinstance(terms[-1].itype, rr.int.Ne)


def test_boolean_equality():
    p, q = _varexpr(Bool([1, 1])), _varexpr(Bool([1, 1]))
    with collecting() as terms:
        e = p == q
    assert [type(t.itype) for t in terms] == [rr.bool.Eq]
    t, f = torch.tensor([[True]]), torch.tensor([[False]])
    assert _run(terms, {p.wire: t, q.wire: t})[e.wire].item() is True
    assert _run(terms, {p.wire: t, q.wire: f})[e.wire].item() is False
    with collecting() as terms:
        e = p != q
    assert [type(t.itype) for t in terms] == [rr.bool.Ne]
    assert _run(terms, {p.wire: t, q.wire: t})[e.wire].item() is False
    assert _run(terms, {p.wire: t, q.wire: f})[e.wire].item() is True


def test_all_and_any_reduce_a_boolean_matrix_along_an_axis():
    p = _varexpr(Bool([2, 3]))
    x = torch.tensor([[True, False, True], [True, True, True]])
    with collecting() as terms:
        a0, a1, a = p.all(0), p.all(1), p.all()
        o0, o1, o = p.any(0), p.any(1), p.any()
    assert all(isinstance(e, BExpr) for e in (a0, a1, a, o0, o1, o))
    assert (a0.shape, a1.shape, a.shape) == ([1, 3], [2, 1], [1, 1])
    assert (o0.shape, o1.shape, o.shape) == ([1, 3], [2, 1], [1, 1])
    assert terms[0].itype == rr.bool.reduce.And(0, Bool([2, 3]))
    assert terms[1].itype == rr.bool.reduce.And(1, Bool([2, 3]))
    values = _run(terms, {p.wire: x})
    assert values[a0.wire].tolist() == [[True, False, True]]
    assert values[a1.wire].tolist() == [[False], [True]]
    assert values[a.wire].tolist() == [[False]]
    assert values[o0.wire].tolist() == [[True, True, True]]
    assert values[o1.wire].tolist() == [[True], [True]]
    assert values[o.wire].tolist() == [[True]]
    with pytest.raises(ValueError, match="axis"):
        p.all(2)


def test_equality_coerces_a_raw_operand():
    x = _varexpr(INT)
    with collecting() as terms:
        e = x == 3
    assert isinstance(e, BExpr) and isinstance(terms[-1].itype, rr.int.Eq)


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
    assert isinstance(terms[-1].itype, word.signed.Lt)

    xu, yu = _varexpr(BV32), _varexpr(BV32)
    with collecting() as terms:
        xu < yu
    assert isinstance(terms[-1].itype, word.unsigned.Lt)


def test_coercion_inherits_signedness_regardless_of_operand_order():
    # a coerced literal takes the variable's signedness, so `5 + xs` and `xs + 5`
    # both produce a *signed* comparison for a signed word `xs`.
    xs = _varexpr(BV32, signed=True)
    with collecting() as terms:
        (xs + 5) < 3
    assert isinstance(terms[-1].itype, word.signed.Lt)
    with collecting() as terms:
        (5 + xs) < 3
    assert isinstance(terms[-1].itype, word.signed.Lt)


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
    assert optypes == [rr.int.Lt, rr.int.Num, rr.int.Add, rr.int.Ite]


def test_shared_subexpression_recorded_once():
    x, y = _varexpr(INT), _varexpr(INT)
    with collecting() as terms:
        g = x < y
        ite(g, x + 1, x)
        ite(g, y, y)
    assert sum(isinstance(t.itype, rr.int.Lt) for t in terms) == 1


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
    assert isinstance(terms[-1].itype, rr.int.Scale)
    vals = torch.arange(1, n + 1, dtype=torch.int64).reshape(n, 1)
    got = _run(terms, {x.wire: vals})[e.wire]
    assert got.flatten().tolist() == [3 * v for v in range(1, n + 1)]


def test_mul_by_constant_folds_to_scale():
    x = _varexpr(INT)
    with collecting() as terms:
        e = x * 2
    assert isinstance(terms[-1].itype, rr.int.Scale)
    assert _run(terms, {x.wire: _int(3)})[e.wire].item() == 6


@pytest.mark.parametrize("left", [False, True])
def test_mul_by_a_full_tensor_is_a_pointwise_scale(left):
    # a constant of the shape of the variable, with distinct entries: no `A·x` computes it
    x = _varexpr(Real([2, 3]))
    c = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with collecting() as terms:
        e = c * x if left else x * c
    assert len(terms) == 1
    assert isinstance(terms[0].itype, real.Scale)
    assert terms[0].itype.tensor.data_ptr() == c.data_ptr()  # shared, not copied
    assert e.dtype == Real([2, 3])
    vals = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    assert torch.equal(_run(terms, {x.wire: vals})[e.wire], c * vals)


def test_neg_and_abs():
    x = _varexpr(INT)
    with collecting() as terms:
        n, a = -x, abs(x)
    assert isinstance(terms[0].itype, rr.int.Neg) and isinstance(terms[1].itype, rr.int.Abs)
    state = _run(terms, {x.wire: _int(-3)})
    assert state[n.wire].item() == 3 and state[a.wire].item() == 3

    r = _varexpr(REAL)
    with collecting() as terms:
        m, b = -r, abs(r)
    assert isinstance(terms[0].itype, real.Neg) and isinstance(terms[1].itype, real.Abs)
    state = _run(terms, {r.wire: torch.tensor([[-2.5]])})
    assert state[m.wire].item() == 2.5 and state[b.wire].item() == 2.5


def test_real_operations_take_the_degree_of_their_operands():
    x = expr(Var(Real([1, 1])))
    dx = expr(Wire(T(Real([1, 1]))))
    with collecting() as terms:
        s = dx + 1  # the literal takes the degree of `dx`
        assert tensor_of_term(terms[-2]) is not None and terms[-2].itype.cod == (T(Real([1, 1])),)
        assert terms[-1].itype.dom == (T(Real([1, 1])),) * 2 and s.dtype == T(Real([1, 1]))
        p = x * dx  # degrees add up
        assert terms[-1].itype.dom == (Real([1, 1]), T(Real([1, 1]))) and p.dtype == T(Real([1, 1]))
        q = dx / x  # degrees are taken off
        assert terms[-1].itype.dom == (T(Real([1, 1])), Real([1, 1])) and q.dtype == T(Real([1, 1]))
        c = dx < 1  # compared at their degree
        assert terms[-1].itype.dom == (T(Real([1, 1])),) * 2 and c.dtype == Bool([1, 1])
        h = abs(dx)
        assert terms[-1].itype.dom == (T(Real([1, 1])),) and h.dtype == T(Real([1, 1]))


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
    assert terms[0].itype.dom == (Real([1, 1]),) and n.dtype == Real([1, 1])
    assert terms[1].itype.dom == (T(Real([1, 1])),) and m.dtype == T(Real([1, 1]))


# theory errors are deprecated - the python API has become permissive
# def test_mul_of_two_variables_is_nonlinear():
#     x, y = _varexpr(INT), _varexpr(INT)
#     with pytest.raises(TheoryError):
#         _ = x * y


def test_mul_of_two_variables():
    x, y = _varexpr(INT), _varexpr(INT)
    with collecting() as terms:
        e = x * y
    assert isinstance(terms[-1].itype, rr.int.Mul)
    assert _run(terms, {x.wire: _int(3), y.wire: _int(4)})[e.wire].item() == 12


def test_matmul():
    a, b = _varexpr(Real([2, 3])), _varexpr(Real([3, 4]))
    with collecting() as terms:
        e = a @ b
    assert terms[-1].itype == real.MatMul([Real([2, 3]), Real([3, 4])])
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


# --- enumerations and events ---------------------------------------------------

from reactor.enum import Enum
from reactor.event import Event
from reactor import enum as _en, event as _ev
from reactor.expr import NExpr, EExpr, recv, Havoc

MODE = Enum({"idle", "busy"})
COND = Bool([1, 1])


def test_a_string_is_a_constant_of_the_enumeration_made_of_it():
    with collecting() as terms:
        e = expr("idle")
    assert isinstance(e, NExpr) and e.dtype == Enum({"idle"}) and not hasattr(e, "shape")
    assert terms[0].itype == _en.Sym("idle", Enum({"idle"}))
    # the family leaves the enumeration to the constant, as for the other sorts
    assert expr("idle", sort=Enum).dtype == Enum({"idle"})
    # given an enumeration, of it, which must hold it
    assert expr("idle", sort=MODE).dtype == MODE
    with pytest.raises(ValueError):
        expr("done", sort=MODE)
    with pytest.raises(TypeError):
        expr(1, sort=MODE)
    with pytest.raises(TypeError):
        expr(1, sort=Enum)


def test_enumerations_compare_by_equality_only():
    x = expr(Var(MODE))
    with collecting() as terms:
        b = x == "idle"
    assert isinstance(b, BExpr) and b.dtype == COND
    # the constant is of the enumeration of `x`
    assert [t.itype for t in terms] == [_en.Sym("idle", MODE), _en.Eq([MODE, MODE])]
    assert (x != "busy").dtype == COND
    with pytest.raises(TypeError):
        x < "busy"
    with pytest.raises(TypeError):
        x + "busy"
    # a string is no number
    with pytest.raises(TypeError):
        expr(Var(INT)) == "idle"


def test_different_enumerations_compare_in_the_union_of_their_values():
    other = Enum({"busy", "done"})
    x, y = expr(Var(MODE)), expr(Var(other))
    union = Enum({"idle", "busy", "done"})
    with collecting() as terms:
        x == y
    assert [t.itype for t in terms] == [
        _en.Into(other, union),
        _en.Into(MODE, union),
        _en.Eq([union, union]),
    ]
    # a string is a constant of the union of the other operand's enumeration and itself
    with collecting() as terms:
        x == "done"
    assert [t.itype for t in terms] == [
        _en.Sym("done", union),
        _en.Into(MODE, union),
        _en.Eq([union, union]),
    ]


def test_cast_converts_an_enumeration_into_a_larger_one():
    x = expr(Var(MODE))
    large = Enum({"idle", "busy", "done"})
    assert cast(x, MODE) is x
    assert cast(x, large).dtype == large
    with pytest.raises(ValueError):
        cast(x, Enum({"idle"}))
    with pytest.raises(TypeError):
        cast(x, INT)


def test_ite_chooses_between_enumerations_by_a_scalar_condition():
    x = expr(Var(MODE))
    c = x == "idle"
    with collecting() as terms:
        r = ite(c, "busy", x)
    assert isinstance(r, NExpr) and r.dtype == MODE
    assert terms[-1].itype == _en.Ite([COND, MODE, MODE])
    # of different enumerations, in the union of their values
    y = expr(Var(Enum({"done"})))
    assert ite(c, x, y).dtype == Enum({"idle", "busy", "done"})
    # of two constants, in the enumeration of the two
    assert ite(c, "on", "off").dtype == Enum({"on", "off"})
    with pytest.raises(TypeError):
        ite(expr(Var(Bool([2, 2]))), x, x)


def test_an_event_is_received_when_it_changes():
    ev = Event()
    e = expr(Var(ev))
    assert isinstance(e, EExpr) and e.dtype == ev and not hasattr(e, "shape")
    with collecting() as terms:
        happens = recv(e)
    assert isinstance(happens, BExpr) and happens.dtype == COND
    # the difference of its next value and its latched one
    assert [t.itype for t in terms] == [_ev.Xor([ev, ev])]
    assert [w.id for w in terms[0].read] == [X(e).wire.id, e.wire.id]
    # an event has no constants
    for sort in [ev, Event]:
        with pytest.raises(TypeError):
            expr(True, sort=sort)
    with pytest.raises(TypeError):
        recv(expr(Var(COND)))


def test_an_event_is_shifted_by_a_boolean_in_either_order():
    ev = Event()
    e, c = expr(Var(ev)), expr(Var(COND))
    for build, dom in [
        (lambda: e ^ c, [ev, COND]),
        (lambda: c ^ e, [COND, ev]),
        (lambda: e ^ True, [ev, COND]),
        (lambda: False ^ e, [COND, ev]),
        (lambda: ~e, [ev, COND]),
    ]:
        with collecting() as terms:
            r = build()
        assert isinstance(r, EExpr) and r.dtype == ev
        assert terms[-1].itype == _ev.Xor(dom)
    # two values of it differ by a boolean
    with collecting() as terms:
        d = X(e) ^ e
    assert isinstance(d, BExpr) and d.dtype == COND
    assert [t.itype for t in terms] == [_ev.Xor([ev, ev])]


def test_an_event_meets_no_other_event_and_no_number():
    e, f = expr(Var(Event())), expr(Var(Event()))
    # the exclusive or rejects any other domain of events and booleans: another event, a
    # wider boolean
    for build in [
        lambda: e ^ f,
        lambda: e ^ expr(Var(Bool([2, 2]))),
        lambda: expr(Var(Bool([2, 2]))) ^ e,
    ]:
        with pytest.raises(ValueError):
            build()
    # a word is no sort of events, a number has no sort to take, and an event is neither a
    # condition, nor chosen between: it is shifted, `e ^ c`
    for build in [
        lambda: e ^ expr(Var(Word(8, [1, 1]))),
        lambda: e ^ 1,
        lambda: e & e,
        lambda: ite(e, "on", "off"),
        lambda: ite(recv(e), e, ~e),
        lambda: ite(recv(e), e, f),
    ]:
        with pytest.raises(TypeError):
            build()


def test_enumerations_and_events_do_not_mix():
    x, e = expr(Var(MODE)), expr(Var(Event()))
    with pytest.raises(TypeError):
        x == e
    with pytest.raises(TypeError):
        ite(x == "idle", x, e)


def test_ite_of_two_booleans_is_a_boolean_choice():
    c = expr(Var(COND))
    with collecting() as terms:
        r = ite(c, True, False)
    assert isinstance(r, BExpr) and r.dtype == COND
    assert terms[-1].itype == rr.bool.Ite([COND, COND, COND])
    # two numbers have no sort to take
    with pytest.raises(TypeError):
        ite(c, 1, 0)


def test_havoc_is_an_arbitrary_value_of_a_concrete_sort():
    with collecting() as terms:
        h = expr(Havoc, sort=INT)
    assert isinstance(h, IExpr) and h.dtype == INT
    assert [t.itype for t in terms] == [rr.Havoc(INT)]
    # of an event too, which has no constants
    ev = Event()
    assert expr(Havoc, sort=ev).dtype == ev
    # a family does not say which sort
    with pytest.raises(TypeError):
        expr(Havoc, sort=Int)
