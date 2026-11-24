"""``reactor.casadi``: translate CasADi functions into Reactor terms and modules.

    import casadi as cs
    from reactor.casadi import sequential, differential, euler, check_equivalence

    x, u = cs.SX.sym("x", 2), cs.SX.sym("u")
    f = cs.Function("f", [x, u], [cs.vertcat(x[1], -x[0] + u)], ["x", "u"], ["xdot"])
    step = euler(f, dt=0.01)                 # discretize in CasADi
    m = sequential(step, init=[0.0, 0.0])    # m.module, m.state, m.inputs
    c = differential(f, init=[0.0, 0.0])     # or keep continuous time: d(x) = f(x, u)
    print(check_equivalence(step))           # miter: CasADi vs. its translation

The translation targets real arithmetic (``reactor.tensor.real``): polynomial and
rational CasADi graphs, comparisons, ``if_else``, ``fmin``/``fmax``, ``fabs``,
``sign``, integer powers, and anything constant. Transcendental functions of
variables raise ``UnsupportedOp``. See ``reactor.casadi.translate`` for the exact
semantics.

Requires the ``casadi`` package (it is not a dependency of ``reactor`` itself).
"""

from .translate import (
    CasadiModule,
    UnsupportedOp,
    differential,
    Terms,
    Value,
    as_real,
    euler,
    fold,
    op_name,
    rk4,
    scalar_vars,
    sequential,
    to_terms,
    translate,
)
from .validate import (
    EntryResult,
    Equivalence,
    casadi_to_z3,
    check_equivalence,
    check_numeric,
    terms_to_z3,
)

__all__ = [
    "CasadiModule",
    "UnsupportedOp",
    "differential",
    "Terms",
    "Value",
    "as_real",
    "euler",
    "fold",
    "op_name",
    "rk4",
    "scalar_vars",
    "sequential",
    "to_terms",
    "translate",
    "EntryResult",
    "Equivalence",
    "casadi_to_z3",
    "check_equivalence",
    "check_numeric",
    "terms_to_z3",
]
