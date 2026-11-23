"""The translation of terms to Z3 expressions, for the operations the other
suites do not reach."""

import numpy as np
import z3

from reactor import tensor
from reactor.smt.z3 import interpret


def test_implies_get_and_casts_export_to_z3():
    """`Implies`, `Get` and the casts."""
    a, b = z3.Bools("a b")
    got = interpret(tensor.bool.Implies([1, 1]))(np.array([[a]], dtype=object), np.array([[b]], dtype=object))
    assert z3.simplify(got[0, 0] == z3.Implies(a, b))
    x = np.array([[z3.Int(f"x{i}{j}") for j in range(3)] for i in range(2)], dtype=object)
    got = interpret(tensor.int.Get(1, 2, [2, 3]))(x)
    assert got.shape == (1, 1) and got[0, 0].eq(z3.Int("x12"))
    i = z3.Int("i")
    got = interpret(tensor.cast.ToReal([1, 1]))(np.array([[i]], dtype=object))
    assert got[0, 0].sort() == z3.RealSort()
    r = z3.Real("r")
    got = interpret(tensor.cast.ToInt([1, 1]))(np.array([[r]], dtype=object))
    assert got[0, 0].sort() == z3.IntSort()
