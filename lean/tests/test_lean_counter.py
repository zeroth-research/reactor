"""Tests for the Module-to-Lean4 functional converter."""

import torch
from reactor import Wire, Term, Module, Bool, Int, Var, X, tensor
from reactor_lean.codegen.context import LeanContext
from reactor_lean.project.scaffold import create_project
from reactor_lean.proof.render import lean_certificate
from reactor_lean.proof.render import certificate_files
from reactor_lean.parse import Certificate, Claim
from testdata import TESTS

from .test_lean_diagram import _make_matrix_module


def _make_counter():
    """Counter module with a 3×1 vector state, transitions via Linear.

    Python semantics (state = (x, y, z) as Matrix (Fin 3) (Fin 1) Int, extl = (y0, z0)):
        init(y0, z0) = (0, y0, z0)       = [[0,0],[1,0],[0,1]] · (y0, z0)
        update(x, y, z):
            if x < y or x < z: (x+1, y, z)   = I·s + (1,0,0)ᵀ
            else:              (0, y, z)      = [[0,0,0],[0,1,0],[0,0,1]] · s
    """
    state = Var(Int([3, 1]))
    extl = Var(Int([2, 1]))

    zero31 = torch.zeros((3, 1), dtype=torch.int64)
    zero11 = torch.zeros((1, 1), dtype=torch.int64)

    # init: state' = A · extl,  A = [[0,0],[1,0],[0,1]]
    A = torch.tensor([[0, 0], [1, 0], [0, 1]], dtype=torch.int64)
    init = [
        Term(tensor.int.Linear(A, zero31), [X(state)], [X(extl)]),
    ]

    # update
    x = Wire(Int([1, 1]))
    y = Wire(Int([1, 1]))
    z = Wire(Int([1, 1]))
    x_lt_y = Wire(Bool([1, 1]))
    x_lt_z = Wire(Bool([1, 1]))
    cond = Wire(Bool([1, 1]))
    result_true = Wire(Int([3, 1]))
    result_false = Wire(Int([3, 1]))

    # scalar extraction via row·state (a 1×3 constant times the 3×1 state)
    row_x = torch.tensor([[1, 0, 0]], dtype=torch.int64)
    row_y = torch.tensor([[0, 1, 0]], dtype=torch.int64)
    row_z = torch.tensor([[0, 0, 1]], dtype=torch.int64)
    e1 = torch.tensor([[1], [0], [0]], dtype=torch.int64)
    diag_yz = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.int64)

    update = [
        Term(tensor.int.Linear(row_x, zero11), [x], [state]),
        Term(tensor.int.Linear(row_y, zero11), [y], [state]),
        Term(tensor.int.Linear(row_z, zero11), [z], [state]),
        # cond = x < y ∨ x < z
        Term(tensor.int.Lt([1, 1]), [x_lt_y], [x, y]),
        Term(tensor.int.Lt([1, 1]), [x_lt_z], [x, z]),
        Term(tensor.bool.Or([1, 1]), [cond], [x_lt_y, x_lt_z]),
        # true branch: state + (1,0,0)ᵀ  = I·state + e1
        Term(tensor.int.Linear(torch.eye(3, dtype=torch.int64), e1), [result_true], [state]),
        # false branch: zero out x  = diag(0,1,1)·state
        Term(tensor.int.Linear(diag_yz, zero31), [result_false], [state]),
        Term(tensor.int.Ite([3, 1]), [X(state)], [cond, result_true, result_false]),
    ]

    return Module.sequential([state, extl], init, update)


def test_inf_counter_generates_lean():
    m = _make_matrix_module()

    _ = create_project(
        output_dir=TESTS / "LeanTests",
        module=m,
        project_name="CounterInfCertificate",
        executable=True,
        certificate=certificate_files(LeanContext(m)),
    )


def test_counter_generates_lean():
    m = _make_counter()

    # Certificate predicates over the 3×1 vector state s (x=s[0][0], y=s[1][0],
    # z=s[2][0]) and the 2×1 external input e (y0=e[0][0], z0=e[1][0]), written
    # in the Python-expression DSL with the `v[i][j]` matrix-indexing sugar.
    _ = create_project(
        output_dir=TESTS / "LeanTests",
        module=m,
        project_name="Counter",
        executable=True,
        certificate=certificate_files(LeanContext(m), lean_certificate(
            m,
            Claim(
                prp="s[0][0] == 0",
                init_pre="And(e[0][0] >= 0, e[1][0] >= 0)",
            ),
            Certificate(
                inv="And(s[0][0] >= 0, Or(s[0][0] <= s[1][0], s[0][0] <= s[2][0]))",
                ranking="Ite(s[0][0] == 0, 0, (Ite(s[1][0] >= s[2][0], s[1][0], s[2][0]) - s[0][0]) + 1)",
            ),
        )),
    )
