import pytest
import torch
from reactor import Wire, Term, Block, Atom, Module, LIA, Bool, Int, LRA, Real, TReal, Zero, Var, X, d
from reactor import Combinatorial, Sequential, Differential
from reactor.blocks import boundary, block, discard, havoc, skip, zero, views


def _bool_t(v):
    """Helper: build a 2-D bool tensor."""
    return torch.tensor([[bool(v)]], dtype=torch.bool)


def test_wire_new():
    Wire(Bool([1, 1]))
    Wire(Int([1, 1]))
    Wire(Int([2, 3]))


def test_term_new():
    x = Wire(Int([2, 3]))
    y = Wire(Int([2, 3]))
    xn = Wire(Int([2, 3]))
    w4 = Wire(Int([1, 3]))
    w5 = Wire(Int([1, 3]))

    # test `function` ctor
    _ = Term.function(LIA.Add(2, 3), [xn], [x, y])
    _ = Term.function(LIA.Int(torch.tensor([[3, 4, 5]])), [w4], [])
    _ = Term.function(LIA.Int(torch.tensor([[3, 4, 6]])), [w5], [])

    # comparisons are pointwise -> Bool of the operand shape
    Term(LIA.Lt(1, 3), [Wire(Bool([1, 3]))], [w4, Wire(Int([1, 3]))])
    Term.constant(LIA.Int(torch.tensor([[3, 2, 1]])), [Wire(Int([1, 3]))])

    # the distinguished operations are terms too
    Term(Combinatorial.HAVOC(Int([1, 3])), [w4])
    Term(Sequential.SKIP(Int([1, 3])), [w5], [w4])
    Term(Differential.ZERO(Zero([1, 3])), [Wire(Zero([1, 3]))])
    Term(Combinatorial.DISCARD(Int([1, 3])), [], [w4])


def test_var_is_a_name_and_a_sort():
    v = Var(Real([2, 3]))
    u = Var(Real([1, 1]))

    # every variable gets a fresh name, in the order of creation
    assert v.dtype == Real([2, 3])
    assert v.name != u.name
    assert v < u and sorted([u, v]) == [v, u]

    # the views of a variable: the next value carries values, the derivative
    # the tangent
    assert X(v).dtype == Real([2, 3])
    assert d(v).dtype == TReal([2, 3], 1)
    assert d(Var(Int([2, 3]))).dtype == Zero([2, 3])
    assert X(v).var == v and X(v).next and d(v).derivative

    # views are values: equal views hash alike, and differ from the variable
    assert X(v) == X(v) and hash(X(v)) == hash(X(v))
    assert {X(v): "found"}[X(v)] == "found"
    assert v != X(v) and X(v) != d(v) and X(v) != X(u)

    # a variable is no wire
    with pytest.raises(TypeError):
        Term(LRA.IdReal(2, 3), [Wire(Real([2, 3]))], [v])


def _counter(x):
    """The atom counting `x` up from false: X(x) = x and X(x)."""
    inputs, outputs = boundary("init", ctrl=[x])
    init = block(inputs, outputs, [Term.constant(LIA.Bool(_bool_t(True)), [outputs[X(x)]])])
    inputs, outputs = boundary("update", ctrl=[x], read=[x])
    update = block(inputs, outputs, [Term(LIA.IdBool(1, 1), [outputs[X(x)]], [inputs[x]])])
    return Atom([x], [], [x], init, update, zero([x], read=[x]))


def test_module_sequential():
    x = Var(Bool([1, 1]))
    m = Module(_counter(x))
    assert m.closed() and m.intf == [x]


def test_module_combinatorial():
    x = Var(Bool([1, 1]))

    # the same assignment initialises and updates
    def assign():
        inputs, outputs = boundary("init", ctrl=[x])
        return block(inputs, outputs, [Term.constant(LIA.Bool(_bool_t(False)), [outputs[X(x)]])])

    m = Module(Atom([x], [], [], assign(), assign(), zero([x])))
    assert m.closed() and m.intf == [x]


def test_module_parallel():
    x = Var(Bool([1, 1]))
    y = Var(Bool([1, 1]))
    z = Var(Bool([1, 1]))
    w = Var(Bool([1, 1]))
    v = Var(Bool([1, 1]))
    false = Term.constant

    # p: X(x) = x and X(y), from false
    i, o = boundary("init", ctrl=[x], wait=[y])
    init = block(i, o, [false(LIA.Bool(_bool_t(False)), [o[X(x)]])])
    i, o = boundary("update", ctrl=[x], wait=[y], read=[x])
    update = block(i, o, [Term(LIA.And(1, 1), [o[X(x)]], [i[x], i[X(y)]])])
    p = Module(Atom([x], [y], [x], init, update, zero([x], [y], [x])))

    # q: X(v) = v and x, X(y) = x, from false, hiding v
    i, o = boundary("init", ctrl=[v, y])
    init = block(i, o, [false(LIA.Bool(_bool_t(False)), [o[X(v)]]),
                        false(LIA.Bool(_bool_t(False)), [o[X(y)]])])
    i, o = boundary("update", ctrl=[v, y], read=[v, x])
    update = block(i, o, [Term(LIA.And(1, 1), [o[X(v)]], [i[v], i[x]]),
                          Term(LIA.IdBool(1, 1), [o[X(y)]], [i[x]])])
    q = Module(Atom([v, y], [], [v, x], init, update, zero([v, y], read=[v, x])), hide=[v])

    # r: X(z) = X(y) or X(w), at every step
    def assign():
        i, o = boundary("init", ctrl=[z], wait=[y, w])
        return block(i, o, [Term(LIA.Or(1, 1), [o[X(z)]], [i[X(y)], i[X(w)]])])

    r = Module(Atom([z], [y, w], [], assign(), assign(), zero([z], [y, w])))

    m = Module.compose(p, q, r)

    for var in m.ctrl:
        print(var)
    for atom in m.atoms:
        print(atom)

    assert m.intf == [x, y, z]
    assert m.extl == [w]
    assert m.prvt == [v]


def test_interface():
    x = Wire(Bool([1, 1]))
    y = Wire(Bool([1, 1]))
    xn = Wire(Bool([1, 1]))
    f = Term(LIA.And(1, 1), [xn], [x, y])
    f2 = Term(LIA.And(1, 1), [xn], [x, y])

    w = f.write
    r = f.read
    r2 = f.read
    assert r is not r2
    assert r is not w
    assert r2 == r
    assert f2.read == r
    assert r == [x, y]
    assert [x, y] == r

    for wire in r:
        print("-->", wire)

    for i in range(len(w)):
        print("-->", w[i])


def test_block_boundary():
    x, y, xn = Wire(Int([1, 1])), Wire(Int([1, 1])), Wire(Int([1, 1]))

    # the block lists its boundary in order
    b = Block([y, x], [xn], [Term(LIA.Add(1, 1), [xn], [x, y])])
    assert b.read == [y, x] and b.write == [xn]
    assert len(b) == 1 and b[0].write == [xn]
    assert [t.write for t in b] == [[xn]]

    # every input is read: an unused one is discarded
    with pytest.raises(Exception, match="not read"):
        Block([x, y], [xn], [Term(LIA.IdInt(1, 1), [xn], [x])])
    b = Block([x, y], [xn], [Term(LIA.IdInt(1, 1), [xn], [x]), discard(y)])
    assert len(b) == 2

    # every output is written, and no input is
    with pytest.raises(Exception, match="not written"):
        Block([x], [xn], [discard(x)])
    with pytest.raises(Exception, match="written"):
        Block([x, y], [xn], [Term(LIA.IdInt(1, 1), [x], [y]), Term(LIA.IdInt(1, 1), [xn], [y])])

    # `block` discards the inputs no term reads
    b = block([x, y], [xn], [Term(LIA.IdInt(1, 1), [xn], [x])])
    assert len(b) == 2 and b.read == [x, y]


def test_views_follow_the_order_of_names():
    x, e = Var(Int([1, 1])), Var(Bool([1, 1]))

    # the interfaces are sets in the order of the names
    assert views("init", ctrl=[x], wait=[e]) == ([X(e)], [X(x)])
    assert views("update", ctrl=[x], wait=[e], read=[e, x, x]) == ([x, e, X(e)], [X(x)])
    assert views("delay", ctrl=[x], wait=[e], read=[e, x]) == ([x, e, d(e)], [d(x)])

    # each view gets a wire of its sort
    inputs, outputs = boundary("delay", ctrl=[x], wait=[e], read=[x])
    assert [w.dtype for w in inputs.values()] == [Int([1, 1]), Zero([1, 1])]
    assert [w.dtype for w in outputs.values()] == [Zero([1, 1])]

    with pytest.raises(ValueError):
        views("jump", ctrl=[x])


def _stateful():
    """Helper: an atom controlling and reading two variables, with its blocks."""
    zero_t = torch.tensor([[0.0]])
    x = Var(Real([1, 1]))
    p = Var(Real([1, 1]))

    i, o = boundary("init", ctrl=[x, p])
    init = block(i, o, [Term(LRA.Real(zero_t), [o[X(x)]]), Term(LRA.Real(zero_t), [o[X(p)]])])
    i, o = boundary("update", ctrl=[x, p], read=[x, p])
    update = block(i, o, [Term(LRA.IdReal(1, 1), [o[X(x)]], [i[x]]),
                          Term(LRA.IdReal(1, 1), [o[X(p)]], [i[p]])])
    i, o = boundary("delay", ctrl=[x, p], read=[x, p])
    delay = block(i, o, [Term(LRA.RealZerograd(1, 1, 1), [o[d(x)]]),
                         Term(LRA.RealZerograd(1, 1, 1), [o[d(p)]])])
    return x, p, init, update, delay


def test_module_of_an_atom():
    x, p, init, update, delay = _stateful()
    atom = Atom([x, p], [], [x, p], init, update, delay)

    m = Module(atom)
    assert m.closed()
    assert m.intf == [x, p]

    # the atom keeps the blocks as given; the delay discards the latched x
    # and p, which only the update reads
    atom = m.atoms[0]
    assert len(atom.init) == 2 and len(atom.update) == 2 and len(atom.delay) == 2 + 2
    assert atom.ctrl == [x, p] and atom.read == [x, p] and len(atom.wait) == 0


def test_module_hide_accepts_predicate_set_and_iterable():
    x, p, init, update, delay = _stateful()
    atom = Atom([x, p], [], [x, p], init, update, delay)

    # hide accepts a callable predicate over variables ...
    m = Module(atom, hide=lambda v: v == p)
    assert m.intf == [x]
    assert m.prvt == [p]

    m = Module(atom, hide={p})
    assert m.intf == [x]
    assert m.prvt == [p]

    S = {p}
    m = Module(atom, hide=S.__contains__)
    assert m.intf == [x]
    assert m.prvt == [p]

    m = Module(atom, hide=[p])
    assert m.intf == [x]
    assert m.prvt == [p]

    m = Module(atom, hide=iter([p]))
    assert m.intf == [x]
    assert m.prvt == [p]

    # hiding nothing yields a fully observable module
    m = Module(atom, hide=lambda v: False)
    assert m.intf == [x, p] and m.prvt == []

    # and so does hiding afterwards
    m = Module(atom).hide([p])
    assert m.intf == [x] and m.prvt == [p]


def test_module_hide_predicate_errors_propagate():
    x, p, init, update, delay = _stateful()
    atom = Atom([x, p], [], [x, p], init, update, delay)

    def broken(v):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        Module(atom, hide=broken)


def test_module_new_positional_is_parallel():
    x, p, init, update, delay = _stateful()
    y = Var(Real([1, 1]))

    p1 = Module(Atom([x, p], [], [x, p], init, update, delay))
    i, o = boundary("init", ctrl=[y])
    p2 = Module(Atom([y], [], [y],
                     block(i, o, [Term(LRA.Real(torch.tensor([[0.0]])), [o[X(y)]])]),
                     skip([y], read=[y]),
                     zero([y], read=[y])))

    m = Module(p1, p2)
    assert m.intf == [x, p, y]

    # module composition with hiding
    m1 = Module(p1, p2, hide=[p])
    assert p in m1.prvt
    assert m1.intf == [x, y]

    # and the product composes too
    assert (p1 * p2).intf == [x, p, y]


def test_compose_hiding_coupled_variables():
    zero_t = torch.tensor([[0.0]])
    x = Var(Real([1, 1]))
    y = Var(Real([1, 1]))
    z = Var(Real([1, 1]))
    w = Var(Real([1, 1]))

    # m1 controls x and y, reading the shared input w; m2 reads y and w and
    # controls z: y couples the two, w is external on both sides
    i, o = boundary("init", ctrl=[x, y])
    init = block(i, o, [Term(LRA.Real(zero_t), [o[X(x)]]), Term(LRA.Real(zero_t), [o[X(y)]])])
    i, o = boundary("update", ctrl=[x, y], read=[y, w])
    update = block(i, o, [Term(LRA.IdReal(1, 1), [o[X(x)]], [i[w]]),
                          Term(LRA.IdReal(1, 1), [o[X(y)]], [i[y]])])
    m1 = Module(Atom([x, y], [], [y, w], init, update, zero([x, y], read=[y, w])))

    i, o = boundary("init", ctrl=[z])
    init = block(i, o, [Term(LRA.Real(zero_t), [o[X(z)]])])
    i, o = boundary("update", ctrl=[z], read=[y, w])
    update = block(i, o, [Term(LRA.Add(1, 1), [o[X(z)]], [i[y], i[w]])])
    m2 = Module(Atom([z], [], [y, w], init, update, zero([z], read=[y, w])))

    assert y in m1.intf and y in m2.extl
    assert w in m1.extl and w in m2.extl

    # hide exactly the coupling variables: interface on one side, external
    # on the other (in either direction) — not the shared external w
    def coupled(v):
        return (v in m1.intf and v in m2.extl) or (v in m1.extl and v in m2.intf)

    m = Module.compose(m1, m2, hide=coupled)

    assert m.prvt == [y]
    assert m.intf == [x, z]
    assert m.extl == [w]

    # the coupling took place regardless: hiding only restricts visibility
    assert y not in m.obs and y not in m.extl


def test_module_new_positional_atoms():
    x, p, init, update, delay = _stateful()
    atom = Atom([x, p], [], [x, p], init, update, delay)

    # a sequence of atoms builds an observable module...
    m = Module(atom)
    assert m.closed() and m.intf == [x, p]

    # ...and hides the `hide` variables when given
    m = Module(atom, hide=[p])
    assert m.intf == [x] and m.prvt == [p]

    # several atoms compose into one module
    y = Var(Real([1, 1]))
    i, o = boundary("init", ctrl=[y])
    init = block(i, o, [Term(LRA.Real(torch.tensor([[0.0]])), [o[X(y)]])])
    other = Atom([y], [], [y], init, skip([y], read=[y]), zero([y], read=[y]))
    m = Module(atom, other)
    assert m.intf == [x, p, y]
    assert len(m.atoms) == 2


def test_module_new_rejects_bad_declarations():
    x, p, init, update, delay = _stateful()

    # modules are built from atoms or modules, not from blocks
    with pytest.raises(TypeError):
        Module(init=init, update=update)

    # an empty module hides nothing
    with pytest.raises(TypeError):
        Module(hide=[x])

    assert Module().closed() and len(Module().atoms) == 0


def test_synthesised_blocks():
    x, p = Var(Real([1, 1])), Var(Real([1, 1]))
    t = Var(Real([1, 1]))

    # havoc: an arbitrary next value per controlled variable, discarding the
    # awaited ones
    init = havoc([x, p], wait=[t])
    assert len(init) == 2 + 1 and len(init.read) == 1 and len(init.write) == 2

    # skip: the latched value per controlled variable, which it reads
    update = skip([x, p], wait=[t], read=[x, p])
    assert len(update) == 2 + 1
    with pytest.raises(ValueError):
        skip([x, p], read=[x])

    # zero: the zero derivative per controlled variable
    delay = zero([x, p], wait=[t], read=[x, p])
    assert len(delay) == 2 + 3 and [w.dtype for w in delay.write] == [TReal([1, 1], 1)] * 2

    # a hold atom: every variable an arbitrary constant
    m = Module(Atom([x, p], [], [x, p], havoc([x, p]), skip([x, p], read=[x, p]),
                    zero([x, p], read=[x, p])))
    assert m.closed() and m.intf == [x, p]


def test_atom_awaits_stated_variables():
    zero_t = torch.tensor([[0.0]])
    x = Var(Real([1, 1]))
    t = Var(Real([1, 1]))

    # x' = X(t): the atom awaits t rather than controlling it
    i, o = boundary("init", ctrl=[x], wait=[t])
    init = block(i, o, [Term(LRA.Real(zero_t), [o[X(x)]])])
    i, o = boundary("update", ctrl=[x], wait=[t])
    update = block(i, o, [Term(LRA.IdReal(1, 1), [o[X(x)]], [i[X(t)]])])
    atom = Atom([x], [t], [], init, update, zero([x], wait=[t]))

    assert atom.ctrl == [x]
    assert atom.wait == [t]
    assert len(atom.read) == 0


def test_atom_rejects_ill_formed_blocks():
    zero_t = torch.tensor([[0.0]])
    x = Var(Real([1, 1]))
    t = Var(Int([1, 1]))

    i, o = boundary("init", ctrl=[x])
    init = block(i, o, [Term(LRA.Real(zero_t), [o[X(x)]])])

    # the update writes the latched value instead of the next value
    i, o = boundary("update", ctrl=[x], read=[x])
    with pytest.raises(Exception, match="written"):
        Block(list(i.values()), list(o.values()), [Term(LRA.IdReal(1, 1), [i[x]], [o[X(x)]])])

    # the update takes inputs other than the interfaces say
    i, o = boundary("update", ctrl=[x], read=[x, t])
    update = block(i, o, [Term(LRA.IdReal(1, 1), [o[X(x)]], [i[x]])])
    with pytest.raises(Exception, match="boundary"):
        Atom([x], [], [x], init, update, zero([x], read=[x]))

    # a variable both controlled and awaited
    with pytest.raises(Exception, match="controlled and awaited"):
        Atom([x], [x], [], havoc([x], [x]), skip([x], [x], [x]), zero([x], [x]))


def test_show_named_rendering():
    x, p, init, update, delay = _stateful()
    m = Module(Atom([x, p], [], [x, p], init, update, delay))

    # named variables render by name, in interfaces and wire positions
    out = m.with_varnames({x: "x", p: "p"})
    # scalar [1, 1] shapes are omitted from the sort rendering
    assert "x" in out and "Real" in out
    assert "X(x)" in out and "X(p)" in out

    # unnamed variables fall back to question mark
    # this test is meant to fail when this convention changes, or errors are raised
    # if any other part of the code relies on the fallback, it is bad
    partial = m.with_varnames({x: "x"})
    assert f"#" in partial

    # the atom renders with the same naming
    out = m.atoms[0].show({x: "x", p: "p"})
    assert "controls" in out and "X(x)" in out

    print(m.with_varnames({x: "x"}))


def test_heterogeneous_composition():
    x = Var(Real([1, 1]))
    y = Var(Real([1, 1]))
    z = Var(Real([1, 1]))

    zero_t = torch.tensor([[0]])

    # P: x' = x from 0
    i, o = boundary("init", ctrl=[x])
    init = block(i, o, [Term(LRA.Real(zero_t), [o[X(x)]])])
    i, o = boundary("update", ctrl=[x], read=[x])
    update = block(i, o, [Term(LRA.IdReal(1, 1), [o[X(x)]], [i[x]])])
    P = Module(Atom([x], [], [x], init, update, zero([x], read=[x])))

    # Q: dy = 0 from 0
    i, o = boundary("init", ctrl=[y])
    init = block(i, o, [Term(LRA.Real(zero_t), [o[X(y)]])])
    i, o = boundary("delay", ctrl=[y], read=[y])
    flow = block(i, o, [Term(LRA.RealZerograd(1, 1, 1), [o[d(y)]])])
    Q = Module(Atom([y], [], [y], init, skip([y], read=[y]), flow))

    # R: z = x + y at every step
    def comb():
        i, o = boundary("init", ctrl=[z], wait=[x, y])
        return block(i, o, [Term(LRA.Add(1, 1), [o[X(z)]], [i[X(x)], i[X(y)]])])

    R = Module(Atom([z], [x, y], [], comb(), comb(), zero([z], wait=[x, y])))

    S = Module.compose(P, Q, R)
    assert S.closed()
