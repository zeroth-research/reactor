"""The terms, atoms and modules over JAX arrays, `reactor.jax`, next to those over torch
tensors at the top of `reactor`: their operations hold JAX arrays as given, checked as
torch tensors are, and evaluate through torch. Skipped unless JAX is installed (the `jax`
dependency group)."""

import numpy as np
import pytest
import torch
import z3

import reactor.jax as rjax
from reactor import Int, Real, Term, Var, Wire, X, d
from reactor import tensor
from reactor.eval import eval_itype, execute_init, execute_update
from reactor.smt.z3 import interpret

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def test_operations_hold_jax_arrays_as_given():
    a = jnp.array([[1.0, 2.0]])
    const = rjax.real.Const(a)
    assert const.tensor is a
    term = rjax.Term(const, [Wire(Real([1, 2]))])
    match term.itype:
        case rjax.real.Const(held):
            assert held is a


def test_jax_arrays_are_checked_as_torch_tensors_are():
    # integers are reals too, kept as given, but not booleans
    rjax.real.Const(jnp.array([[1]]))
    with pytest.raises(ValueError, match="floating-point or integer elements, got bool"):
        rjax.real.Const(jnp.array([[True]]))
    with pytest.raises(ValueError, match="floating-point elements, got int32"):
        rjax.real.Scale(jnp.array([[1]]))
    with pytest.raises(ValueError, match="integer"):
        rjax.int.Scale(jnp.eye(2))
    with pytest.raises(ValueError, match="integer"):
        rjax.int.Linear(jnp.eye(2), jnp.zeros((2, 1)))
    with pytest.raises(ValueError, match="expected a matrix"):
        rjax.real.Const(jnp.zeros(3))
    with pytest.raises(ValueError, match=r"\[m, k\]"):
        rjax.real.Linear(jnp.eye(2), jnp.zeros((3, 1)))


def test_each_package_holds_its_own_tensors():
    with pytest.raises(TypeError, match="jax.Array"):
        rjax.real.Const(torch.ones(1, 1))
    with pytest.raises(TypeError, match="torch.Tensor"):
        tensor.real.Const(jnp.ones((1, 1)))


def test_terms_type_check_jax_arrays_by_their_shape():
    const = rjax.real.Const(jnp.array([[1.0, 2.0]]))
    with pytest.raises(Exception, match=r"to write Real\(1, 2\), got Real\(2, 1\)"):
        rjax.Term(const, [Wire(Real([2, 1]))])


def test_the_operations_holding_no_tensors_are_shared():
    x, y = Wire(Real([2, 1])), Wire(Real([2, 1]))
    term = rjax.Term(tensor.real.Add([2, 1]), [Wire(Real([2, 1]))], [x, y])
    assert isinstance(term.itype, tensor.real.Add)


def _counter():
    """A counter starting at a JAX 0 and counting by a JAX map of a JAX 1."""
    x = Var(Int([1, 1]))
    one, two = Wire(Int([1, 1])), Wire(Int([1, 1]))
    init = [rjax.Term(rjax.int.Const(jnp.array([[0]])), [X(x)])]
    update = [
        rjax.Term(rjax.int.Const(jnp.array([[1]])), [one]),
        rjax.Term(rjax.int.Linear(jnp.array([[2]]), jnp.array([[0]])), [two], [one]),
        rjax.Term(tensor.int.Add([1, 1]), [X(x)], [x, two]),
    ]
    return rjax.Module.sequential([x], init, update), x, init, update


def test_a_module_evaluates_with_jax_arrays():
    # JAX makes `int32` arrays, which evaluate alongside the `int64` values
    counter, x, _, _ = _counter()
    state = {}
    execute_init(state, counter.atoms)
    for _ in range(3):
        state[x] = state[X(x)]
        execute_update(state, counter.atoms)
    assert state[X(x)].dtype is torch.int64
    assert state[X(x)].item() == 6


def test_atoms_and_modules_take_the_terms_of_their_package():
    counter, x, init, update = _counter()
    atom = rjax.Atom(vars=[x], init=init, update=update)
    assert len(rjax.Module(atom).atoms) == 1
    assert len(counter.atoms[0].update) == 3

    # the terms at the top of `reactor` hold torch tensors
    torch_init = [Term(tensor.int.Const(torch.zeros(1, 1, dtype=torch.int64)), [X(x)])]
    with pytest.raises(TypeError, match="expected the terms of `reactor.jax`"):
        rjax.Module.sequential([x], torch_init, update)


def test_an_operation_evaluates_with_jax_arrays():
    linear = rjax.real.Linear(jnp.eye(2), jnp.ones((2, 1)))
    (out,) = eval_itype(linear, [torch.tensor([[1.0], [2.0]])])
    assert torch.equal(out, torch.tensor([[2.0], [3.0]]))


def test_jax_operations_on_rates_keep_their_degree():
    # `ẋ = 2·ṫ`: a product with a constant, of reals of degree 1
    t = Var(Real([1, 1]))
    rate = Wire(Real([1, 1], 1))
    scale = rjax.real.Scale(jnp.array([[2.0]]), degree=1)
    assert scale.degree == 1
    rjax.Term(scale, [rate], [d(t)])
    with pytest.raises(Exception, match=r"to read Real\(1, 1\), got T\^1\(Real\(1, 1\)\)"):
        rjax.Term(rjax.real.Scale(jnp.array([[2.0]])), [rate], [d(t)])
    (out,) = eval_itype(scale, [torch.tensor([[3.0]])])
    assert torch.equal(out, torch.tensor([[6.0]]))
    assert rjax.real.Const(jnp.array([[1.0]]), 1).degree == 1
    assert rjax.real.Linear(jnp.eye(1), jnp.zeros((1, 1)), degree=1).degree == 1


def test_jax_operations_export_to_z3():
    (value,) = interpret(rjax.int.Const(jnp.array([[3]])))().flat
    assert value.as_long() == 3
    linear = interpret(rjax.int.Linear(jnp.array([[2]]), jnp.array([[1]])))
    (value,) = linear(np.array([[z3.IntVal(4)]], dtype=object)).flat
    assert z3.simplify(value).as_long() == 9
    scale = interpret(rjax.int.Scale(jnp.array([[3]])))
    (value,) = scale(np.array([[z3.IntVal(4)]], dtype=object)).flat
    assert z3.simplify(value).as_long() == 12
