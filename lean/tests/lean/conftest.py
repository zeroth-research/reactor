"""
Session-scoped fixtures that prepare the tests/lean/ lake project before any
test in this directory runs. The Lean library the files import,
``Relational.*``, is ReactLib's: the lakefile requires it from the checkout
``$REACTLIB`` (``lake.prepare``).

``generate_lean_files`` generates:
     • Certs/ScalarEnc*.lean  (functional + scalar encodings)
     • Certs/RelEnc*/     (those two + ScalarRel, one module each)
     • Certs/ArgmaxScalar.lean  (the scalar Argmax variants)
"""
from functools import partial

import pytest
import torch

from reactor import Module, Wire, Term, Int, Real, Bool, Var, X, tensor
from reactor_lean.project.names import LINTERS_OFF, Layout
from reactor_lean.project.scaffold import ENCODINGS, write_encoding
from reactor_lean.codegen.encodings.scalar import _argmax_scalar_def_lines
from lake import lake_build
from reactor_lean.parse import load
from testdata import LAKE_PROJECT, MODS, fixture

_CERTS_DIR = LAKE_PROJECT / "Certs"


# ──────────────────────────────────────────────────────────────
# Modules: the svcomp fixtures (as test_lean_svcomp.py uses them), and
# the ones only these files are about
# ──────────────────────────────────────────────────────────────


_make_countdown = partial(fixture, "svcomp_countdown")
_make_twovars = partial(fixture, "svcomp_twovars")


def _make_word_ops() -> Module:
    """`limits/mods/m_word_ops.py`: every word op that the circuit encoding
    once stood in for, at 1×1 and entry by entry."""
    return load(MODS / "m_word_ops.py").module


def _make_counter() -> Module:
    """3×1 vector-state counter with Linear transitions (matrix cert).

    state = (x, y, z); init = (0, y0, z0); update increments x while x < y or
    x < z, else resets x to 0. Exercises the Linear affine map plus tuple-select
    (`s[i][j]`) predicates end-to-end through zeroth_hammer.
    """
    state = Var(Int([3, 1]))
    extl = Var(Int([2, 1]))
    zero31 = torch.zeros((3, 1), dtype=torch.int64)
    zero11 = torch.zeros((1, 1), dtype=torch.int64)

    A = torch.tensor([[0, 0], [1, 0], [0, 1]], dtype=torch.int64)
    init = [Term(tensor.int.Linear(A, zero31), [X(state)], [X(extl)])]

    x, y, z = Wire(Int([1, 1])), Wire(Int([1, 1])), Wire(Int([1, 1]))
    x_lt_y, x_lt_z, cond = Wire(Bool([1, 1])), Wire(Bool([1, 1])), Wire(Bool([1, 1]))
    result_true, result_false = Wire(Int([3, 1])), Wire(Int([3, 1]))
    row_x = torch.tensor([[1, 0, 0]], dtype=torch.int64)
    row_y = torch.tensor([[0, 1, 0]], dtype=torch.int64)
    row_z = torch.tensor([[0, 0, 1]], dtype=torch.int64)
    e1 = torch.tensor([[1], [0], [0]], dtype=torch.int64)
    diag_yz = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.int64)
    update = [
        Term(tensor.int.Linear(row_x, zero11), [x], [state]),
        Term(tensor.int.Linear(row_y, zero11), [y], [state]),
        Term(tensor.int.Linear(row_z, zero11), [z], [state]),
        Term(tensor.int.Lt([1, 1]), [x_lt_y], [x, y]),
        Term(tensor.int.Lt([1, 1]), [x_lt_z], [x, z]),
        Term(tensor.bool.Or([1, 1]), [cond], [x_lt_y, x_lt_z]),
        Term(tensor.int.Linear(torch.eye(3, dtype=torch.int64), e1), [result_true], [state]),
        Term(tensor.int.Linear(diag_yz, zero31), [result_false], [state]),
        Term(tensor.int.Ite([3, 1]), [X(state)], [cond, result_true, result_false]),
    ]
    return Module.sequential([state, extl], init, update)


def _make_countdown_vec(n: int) -> Module:
    """Countdown embedded in an n-vector state (only s[0] is live; s[1..n-1] carried).

    Every transition is an n-wide MatMul contraction (row·s, Iₙ·s, diag·s). Because
    the codegen pre-contracts constant Linear ops (each output element becomes an
    explicit O(nnz) linear combination), the generated defs and proofs stay small
    and `omega`-friendly regardless of n — this is the scaling regression test.
    """
    s = Var(Int([n, 1]))
    z11 = torch.zeros((1, 1), dtype=torch.int64)

    init_vec = torch.zeros((n, 1), dtype=torch.int64)
    init_vec[0][0] = 100
    init = [Term(tensor.int.Const(init_vec), [X(s)])]

    x = Wire(Int([1, 1]))
    zc = Wire(Int([1, 1]))
    cond = Wire(Bool([1, 1]))
    reset = Wire(Int([n, 1]))
    dec = Wire(Int([n, 1]))

    row0 = torch.zeros((1, n), dtype=torch.int64)
    row0[0][0] = 1  # 1×n: extract s[0]
    diag_keep = torch.eye(n, dtype=torch.int64)
    diag_keep[0][0] = 0  # zero s[0], keep the rest
    b100 = torch.zeros((n, 1), dtype=torch.int64)
    b100[0][0] = 100
    In = torch.eye(n, dtype=torch.int64)
    bneg = torch.zeros((n, 1), dtype=torch.int64)
    bneg[0][0] = -1

    update = [
        Term(tensor.int.Linear(row0, z11), [x], [s]),            # x = s[0]
        Term(tensor.int.Const(torch.tensor([[0]])), [zc]),
        Term(tensor.int.Eq([1, 1]), [cond], [x, zc]),            # cond = (x == 0)
        Term(tensor.int.Linear(diag_keep, b100), [reset], [s]),  # reset: s[0] := 100
        Term(tensor.int.Linear(In, bneg), [dec], [s]),           # dec:   s[0] := s[0] - 1
        Term(tensor.int.Ite([n, 1]), [X(s)], [cond, reset, dec]),
    ]
    return Module.sequential([s], init, update)


def _make_mixed() -> Module:
    """ctrl = [x : 1x1, v : 3x1] — the state whose wires and elements disagree.

    Four state elements against two ctrl wires, so wire 1's data starts at
    flat slot 1. Everything else here has either all-1x1 wires (slot i *is*
    wire i) or a single wire (the slice is the whole tuple), and neither
    catches a per-wire projection of a per-element tuple. Both transitions
    read both wires so the slices cannot be independently wrong.
    """
    x = Var(Int([1, 1]))
    v = Var(Int([3, 1]))
    z11 = torch.zeros((1, 1), dtype=torch.int64)
    v0 = torch.tensor([[1], [2], [3]], dtype=torch.int64)

    init = [
        Term(tensor.int.Const(torch.zeros((1, 1), dtype=torch.int64)), [X(x)]),
        Term(tensor.int.Const(v0), [X(v)]),
    ]

    head, zc = Wire(Int([1, 1])), Wire(Int([1, 1]))
    cond = Wire(Bool([1, 1]))
    reset, dec = Wire(Int([3, 1])), Wire(Int([3, 1]))
    row0 = torch.tensor([[1, 0, 0]], dtype=torch.int64)
    zero33 = torch.zeros((3, 3), dtype=torch.int64)
    bneg = torch.tensor([[-1], [0], [0]], dtype=torch.int64)

    update = [
        Term(tensor.int.Linear(row0, z11), [head], [v]),          # head = v[0]
        Term(tensor.int.Add([1, 1]), [X(x)], [x, head]),          # x' = x + v[0]
        Term(tensor.int.Const(torch.zeros((1, 1), dtype=torch.int64)), [zc]),
        Term(tensor.int.Eq([1, 1]), [cond], [head, zc]),          # cond = (v[0] == 0)
        Term(tensor.int.Linear(zero33, v0), [reset], [v]),        # reset: v := (1,2,3)
        Term(tensor.int.Linear(torch.eye(3, dtype=torch.int64), bneg), [dec], [v]),
        Term(tensor.int.Ite([3, 1]), [X(v)], [cond, reset, dec]),
    ]
    return Module.sequential([x, v], init, update)



def _make_real_half() -> Module:
    """A Real countdown by halves: x = 5, then x - 1/2 while x > 0, else 5.

    `1/2` is not a whole number, so it is printed `((1 : Real) / 2)`, and a
    definition holding it has to be `noncomputable`: the hybrid benchmarks
    all failed to build on exactly that, while every Real module here had
    whole constants only.
    """
    x = Var(Real([1, 1]))
    half, zero, five = Wire(Real([1, 1])), Wire(Real([1, 1])), Wire(Real([1, 1]))
    cond, dec = Wire(Bool([1, 1])), Wire(Real([1, 1]))
    update = [
        Term(tensor.real.Const(torch.tensor([[0.5]])), [half]),
        Term(tensor.real.Const(torch.tensor([[0.0]])), [zero]),
        Term(tensor.real.Const(torch.tensor([[5.0]])), [five]),
        Term(tensor.real.Gt([1, 1]), [cond], [x, zero]),
        Term(tensor.real.Sub([1, 1]), [dec], [x, half]),
        Term(tensor.real.Ite([1, 1]), [X(x)], [cond, dec, five]),
    ]
    return Module.sequential(
        [x], [Term(tensor.real.Const(torch.tensor([[5.0]])), [X(x)])], update)


def _make_dot_product() -> Module:
    """d = [1 1 1]·v, a 1×1 product of a 1×3 row and the 3×1 state.

    The scalar encoding binds the 1×1 `d` to a bare `Int`, and it used to
    print the product as a bare `x0 * _m0` too, which is a 1×1 `Matrix`.
    """
    v, d = Var(Int([3, 1])), Var(Int([1, 1]))
    row = Wire(Int([1, 3]))
    init = [
        Term(tensor.int.Const(torch.tensor([[1], [2], [3]])), [X(v)]),
        Term(tensor.int.Const(torch.tensor([[0]])), [X(d)]),
    ]
    update = [
        Term(tensor.int.Const(torch.tensor([[1, 1, 1]])), [row]),
        Term(tensor.int.MatMul([[1, 3], [3, 1]]), [X(d)], [row, v]),
        Term(tensor.Skip(Int([3, 1])), [X(v)], [v]),
    ]
    return Module.sequential([v, d], init, update)


def _make_argmax(m: int, n: int) -> Module:
    """i = argmax s, over an m×n state that counts up from 0.

    The scalar encoding unrolls a one-row Argmax over its entries and leaves
    any other to the matrix form; a 2×2 one used to be unrolled too, and its
    `update_scalar_eq` did not close. A row of 12 is where the unrolled form
    used to run out of heartbeats.
    """
    s, i = Var(Int([m, n])), Var(Int([1, 1]))
    init = [
        Term(tensor.int.Const(torch.arange(m * n).reshape(m, n)), [X(s)]),
        Term(tensor.int.Const(torch.tensor([[0]])), [X(i)]),
    ]
    update = [
        Term(tensor.Skip(Int([m, n])), [X(s)], [s]),
        Term(tensor.int.Argmax([m, n]), [X(i)], [s]),
    ]
    return Module.sequential([s, i], init, update)


def _make_abs() -> Module:
    """x = |x - 3| on a 1×1 Int, v = |v - 1| on a 2×1 one and r = |r - 1/2|
    on a Real: `Abs` on a bare scalar, entry by entry, and over each number."""
    x, v, r = Var(Int([1, 1])), Var(Int([2, 1])), Var(Real([1, 1]))
    three, one, half = Wire(Int([1, 1])), Wire(Int([2, 1])), Wire(Real([1, 1]))
    dx, dv, dr = Wire(Int([1, 1])), Wire(Int([2, 1])), Wire(Real([1, 1]))
    init = [
        Term(tensor.int.Const(torch.tensor([[0]])), [X(x)]),
        Term(tensor.int.Const(torch.tensor([[-2], [5]])), [X(v)]),
        Term(tensor.real.Const(torch.tensor([[0.0]])), [X(r)]),
    ]
    update = [
        Term(tensor.int.Const(torch.tensor([[3]])), [three]),
        Term(tensor.int.Const(torch.tensor([[1], [1]])), [one]),
        Term(tensor.real.Const(torch.tensor([[0.5]])), [half]),
        Term(tensor.int.Sub([1, 1]), [dx], [x, three]),
        Term(tensor.int.Sub([2, 1]), [dv], [v, one]),
        Term(tensor.real.Sub([1, 1]), [dr], [r, half]),
        Term(tensor.int.Abs([1, 1]), [X(x)], [dx]),
        Term(tensor.int.Abs([2, 1]), [X(v)], [dv]),
        Term(tensor.real.Abs([1, 1]), [X(r)], [dr]),
    ]
    return Module.sequential([x, v, r], init, update)


def _make_three_wires() -> Module:
    """x, v, y = -x, -v, -y over a 1×1, a 2×1 and a 1×1 Int.

    `ScalarRel`'s `pack` round trip split the state with one `Prod.ext`,
    which leaves the wires after the first as one pair; a state of three or
    more wires with a multi-element one among them did not build. `Mixed`
    and `TwoVars` have two wires, where one split is all there is.
    """
    shapes = ([1, 1], [2, 1], [1, 1])
    vs = [Var(Int(sh)) for sh in shapes]
    init = [Term(tensor.int.Const(torch.full(sh, k, dtype=torch.int64)), [X(v)])
            for k, (v, sh) in enumerate(zip(vs, shapes), 1)]
    update = [Term(tensor.int.Neg(sh), [X(v)], [v]) for v, sh in zip(vs, shapes)]
    return Module.sequential(vs, init, update)


# Widths that exercise the scalar encoding's flattening: all-1x1, then
# multi-element state, up to the 32 that stresses the index enumeration;
# then the ops whose 1×1 result is made of more than 1×1 operands, `Abs`,
# and the word ops.
_SCALAR_ENC_SPECS = [
    ("Scalar", _make_countdown),
    ("Vec6", lambda: _make_countdown_vec(6)),
    ("Counter", _make_counter),
    ("Vec32", lambda: _make_countdown_vec(32)),
    ("RealHalf", _make_real_half),
    ("DotProduct", _make_dot_product),
    ("Argmax2x2", lambda: _make_argmax(2, 2)),
    ("Argmax1x12", lambda: _make_argmax(1, 12)),
    ("Abs", _make_abs),
    ("WordOps", _make_word_ops),
]


# The relational encodings sit on top of the scalar one, so they need every
# width the scalar specs cover, plus the multi-wire states: `TwoVars`,
# where each wire is one element so wire index and slot index coincide (what
# the per-wire projection assumed), `Mixed`, where they do not, and
# `ThreeWires`, where the state is more than one pair.
_REL_ENC_SPECS = _SCALAR_ENC_SPECS + [
    ("TwoVars", _make_twovars),
    ("Mixed", _make_mixed),
    ("ThreeWires", _make_three_wires),
]


# (elem_ty, n) pairs for the generated scalar Argmax variants. n = 1 is the
# degenerate single-element case; the rest exercise the fold.
# Int and Real at the same width sit together on purpose: the variant
# names used to be built from `n` alone, so these two collided. From 12 on,
# the `let` chain the variants used to be ran out of heartbeats.
_ARGMAX_SCALAR_SPECS = [
    ("Int", 1),
    ("Int", 2),
    ("Int", 4),
    ("Real", 3),
    ("Real", 4),
    ("Int", 12),
    ("Real", 12),
    ("Int", 32),
]


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def relational_mat() -> None:
    """`Relational.Mat` built, once a session: what the files the tests
    compile with `lake.lean_outputs` import."""
    built = lake_build("Relational.Mat", timeout=1800)
    assert built.returncode == 0, built.stdout[-2000:] + built.stderr[-2000:]


@pytest.fixture(scope="session", autouse=True)
def generate_lean_files() -> None:
    """Generate Certs/*.lean for the lake project."""
    _CERTS_DIR.mkdir(exist_ok=True)
    # Certs/ArgmaxScalar.lean — the scalar Argmax variants exactly as the
    # generator emits them. Each carries an `argmax1d_scalar_n_eq` theorem
    # proving it equal to Relational.Mat.argmax_1d, so building this file is what
    # keeps the unrolled scalar form and the matrix definition in step. No
    # certificate module currently uses Argmax, so nothing else compiles it.
    # Certs/ScalarEnc.lean — the functional and scalar encodings together, the
    # way a generated project arranges them (System/System.lean +
    # System/Scalar.lean). The `_scalar_eq` theorems only elaborate if the
    # flattening lines up, and no certificate carries a Scalar section, so
    # this is the only thing that compiles them.
    from reactor_lean.codegen.context import LeanContext
    from reactor_lean.codegen.encodings import ModuleToLean4

    for name, make_module in _SCALAR_ENC_SPECS:
        t = ModuleToLean4(LeanContext(make_module()))
        (_CERTS_DIR / f"ScalarEnc{name}.lean").write_text(
            f"import Relational.Basic\n\n{LINTERS_OFF}\n\n"
            + t.to_lean_functional()
            + "\n\n"
            + t.to_lean_scalar()
            + "\n"
        )

    # Certs/RelEnc*/ — functional, scalar and ScalarRel, one module each,
    # written by `create_project`'s own `write_encoding`: Scalar imports the
    # functional encoding (`Base` here) and ScalarRel imports Scalar.
    # `ScalarRel.effect_i` is per wire while the state tuple it is compared to
    # is per element, so these are what keep the slice and the codomain in
    # agreement; no certificate carries a ScalarRel section.
    #
    # The split is load-bearing, not cosmetic. These used to be concatenated
    # into one file, which is a weaker test than separate modules: a `match`
    # elaborates to an auxiliary matcher that is reused within a module but
    # regenerated across module boundaries, so an `effect_i_eq` saw one
    # matcher constant on both sides here and two different ones in a real
    # project, and every generated project with a multi-element ctrl wire
    # failed to compile while this test passed.
    for name, make_module in _REL_ENC_SPECS:
        t = ModuleToLean4(LeanContext(make_module()))
        layout = Layout(base="Base", prefix=f"Certs.RelEnc{name}.",
                        directory=_CERTS_DIR / f"RelEnc{name}")
        layout.directory.mkdir(exist_ok=True)
        for enc in ENCODINGS:
            if enc.suffix in ("", "Scalar", "ScalarRel"):
                write_encoding(enc, t, layout, name)
        (_CERTS_DIR / f"RelEnc{name}.lean").write_text(
            f"import {layout.lean_module('ScalarRel')}\n")

    argmax_lines = ["import Relational.Mat", ""]
    for elem_ty, n in _ARGMAX_SCALAR_SPECS:
        argmax_lines.extend(_argmax_scalar_def_lines(elem_ty, n))
    (_CERTS_DIR / "ArgmaxScalar.lean").write_text("\n".join(argmax_lines))
