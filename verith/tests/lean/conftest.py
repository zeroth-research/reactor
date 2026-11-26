"""
Session-scoped fixtures that prepare the tests/lean/ lake project before any
test in this directory runs: `reactor_lean`'s encodings and `verith`'s proofs
of them, built against ReactLib (`$REACTLIB`, linked at `.lake/ReactLib`)
and the Lake package cache (`$REACTOR_LAKE_CACHE`), as a generated project
is built.

``generate_lean_files`` generates:
     • ZerothHammer.lean  (the zeroth_hammer tactic, standalone)
     • Certs/<Name>.lean  (self-contained certificates, inline module)
     • Certs/LearnedCountdown.lean  (the certificate `--infer nuterm` finds)
     • Certs/Bridge*.lean  (the `--fbk-proveit` model, proved the module)

The certificates import ZerothHammer for the tactic and define their own
module-specific ``simp_mat`` / ``simp_defs`` macros.

The tests get `lake` through fixtures (``lake_build``, ``lean_outputs``):
`tests/` is no package and puts nothing on the path.
"""
from __future__ import annotations

import functools
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import pytest
import torch

from reactor import Module, Wire, Term, Int, Bool, Var, X, tensor
from reactor_lean.codegen.context import LeanContext
from reactor_lean.codegen.encodings import ModuleToLean4
from reactor_lean.core.certificate import Certificate, Claim
from reactor_lean.inputs.python import load_module
from reactor_lean.pipeline.certificate import render
from reactor_lean.project.lake_cache import lake_cache, use_lake_cache
from reactor_lean.project.lakefile import (FIXED_REQUIRES, REACTLIB_ENV,
                                           Require, reactlib_checkout)
from reactor_lean.project.names import LINTERS_OFF
from reactor_lean.proof.render import (generate_data_lean,
                                       generate_standalone_cert_lean)
from verith.proofs import generate_zeroth_hammer_lean, prove

PROJECT = Path(__file__).resolve().parent
CERTS = PROJECT / "Certs"
FIXTURES = PROJECT.parent / "fixtures"
REACTLIB = Require("ReactLib", path=".lake/ReactLib")


# ──────────────────────────────────────────────────────────────
# lake in this project
# ──────────────────────────────────────────────────────────────


@functools.cache
def _prepare() -> None:
    """The project ready for `lake`, once a session; skips the test without
    a ReactLib checkout to require."""
    checkout = reactlib_checkout()
    if checkout is None:
        pytest.skip(f"no ReactLib checkout: set ${REACTLIB_ENV}")
    link = PROJECT / REACTLIB.path
    if link.is_symlink():
        link.unlink()
    link.parent.mkdir(exist_ok=True)
    link.symlink_to(checkout / "ReactLib", target_is_directory=True)
    cache = lake_cache()
    if cache is not None:
        use_lake_cache(PROJECT, (*FIXED_REQUIRES, REACTLIB), name="VerithTests",
                       cache=cache)


def _lake_build(*targets: str, timeout: int = 600) -> subprocess.CompletedProcess:
    """`lake build *targets`, its output captured."""
    _prepare()
    return subprocess.run(["lake", "build", *targets], cwd=PROJECT,
                          capture_output=True, text=True, timeout=timeout)


def _lean_outputs(paths: "list[Path]") -> list[str]:
    """What `lake env lean` prints for each of `paths`, in order: each file
    compiled on its own, in parallel, against what the project has built."""
    _prepare()

    def lean(path: Path) -> str:
        return subprocess.run(["lake", "env", "lean", str(path)], cwd=PROJECT,
                              capture_output=True, text=True, timeout=1800).stdout

    with ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool:
        return list(pool.map(lean, paths))


@pytest.fixture(scope="session")
def lake_build():
    """`lake build *targets` in this project."""
    return _lake_build


@pytest.fixture(scope="session")
def lean_outputs():
    """`lake env lean` on each of a list of files, in this project."""
    return _lean_outputs


@pytest.fixture(scope="session")
def relational_mat() -> None:
    """`Relational.Mat` built, once a session: what the files the tests
    compile with `lean_outputs` import."""
    built = _lake_build("Relational.Mat", timeout=1800)
    assert built.returncode == 0, built.stdout[-2000:] + built.stderr[-2000:]


# ──────────────────────────────────────────────────────────────
# Modules: the svcomp fixtures, and the ones only these files are about
# ──────────────────────────────────────────────────────────────


def _fixture(name: str) -> Module:
    return load_module(FIXTURES / f"{name}.py")


_make_countdown = partial(_fixture, "svcomp_countdown")
_make_twovars = partial(_fixture, "svcomp_twovars")
_make_collatz = partial(_fixture, "svcomp_collatz_bounded")


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


_COUNTDOWN_CERT = (
    Claim(prp="s[0][0] == 0"),
    Certificate(
        inv="And(s[0][0] >= 0, s[0][0] <= 100)",
        ranking="Ite(s[0][0] == 0, 0, s[0][0])",
    ),
)


# (name, module, safety property) for the `--fbk-proveit` equivalence proof.
# One per shape class the route accepts: a scalar Int state, a mixed Bool/Int
# state, and a module whose transition is a net (`Linear` -> `ReLU` ->
# `Linear`), which is the case that needs the matrix simp set.
_BRIDGE_SPECS = [
    ("Countdown", _make_countdown, "(and (>= s0 0) (<= s0 100))"),
    ("TwoVars", _make_twovars, "(and (>= s0 0) (<= s0 s1) (= s1 10))"),
]


_CERT_SPECS = [
    (
        "BigCounter",
        lambda: _make_countdown_vec(6),
        _COUNTDOWN_CERT,
    ),
    (
        "HugeCounter",
        lambda: _make_countdown_vec(32),
        _COUNTDOWN_CERT,
    ),
    (
        "Counter",
        _make_counter,
        (
            Claim(
                prp="s[0][0] == 0",
                init_pre="And(e[0][0] >= 0, e[1][0] >= 0)",
            ),
            Certificate(
                inv="And(s[0][0] >= 0, Or(s[0][0] <= s[1][0], s[0][0] <= s[2][0]))",
                ranking="Ite(s[0][0] == 0, 0, (Ite(s[1][0] >= s[2][0], s[1][0], s[2][0]) - s[0][0]) + 1)",
            ),
        ),
    ),
    (
        "Countdown",
        _make_countdown,
        (
            Claim(prp="(= s0 0)"),
            Certificate(
                inv="(and (>= s0 0) (<= s0 100))",
                ranking="(ite (= s0 0) 0 s0)",
            ),
        ),
    ),
    (
        # The safety shape: `rule_globally`, an invariant that implies the
        # property, and no ranking function anywhere in the file. Nothing
        # else in this suite compiles a `--safety` certificate.
        "CountdownSafe",
        _make_countdown,
        (
            Claim(prp="(<= s0 100)", kind="safety"),
            Certificate(inv="(and (>= s0 0) (<= s0 100))"),
        ),
    ),
    (
        "TwoVars",
        _make_twovars,
        (
            Claim(prp="(= s0 s1)"),
            Certificate(
                inv="(and (>= s0 0) (<= s0 s1) (= s1 10))",
                ranking="(ite (= s0 s1) 0 (- s1 s0))",
            ),
        ),
    ),
    (
        "Collatz",
        _make_collatz,
        (
            Claim(prp="(= s0 1)"),
            Certificate(
                inv="(and (>= s0 1) (<= s0 8))",
                ranking="(ite (= s0 1) 0 (- s0 1))",
            ),
        ),
    ),
]


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


def proved_cert_lean(module: Module, claim: Claim,
                     cert: Certificate = Certificate()) -> str:
    """The standalone certificate of `claim` and `cert`, with the proofs
    written for it."""
    ctx = LeanContext(module)
    lean = render(module, claim, cert)
    return generate_standalone_cert_lean(
        ctx, lean, prove(module, ctx, claim, cert, lean))


def _strip(src: str) -> str:
    """`src` without its doc comments and imports, to be inlined."""
    out, in_doc = [], False
    for line in src.splitlines():
        if line.startswith("/-"):
            in_doc = True
        if in_doc:
            if "-/" in line:
                in_doc = False
            continue
        if line.startswith("import "):
            continue
        out.append(line)
    return "\n".join(out)


@pytest.fixture(scope="session", autouse=True)
def generate_lean_files() -> None:
    """Generate ZerothHammer.lean and Certs/*.lean for the lake project."""
    CERTS.mkdir(exist_ok=True)

    # ZerothHammer.lean — standalone tactic
    (PROJECT / "ZerothHammer.lean").write_text(generate_zeroth_hammer_lean())

    # Certs/ — one self-contained certificate per module
    for name, make_module, (claim, cert) in _CERT_SPECS:
        (CERTS / f"{name}.lean").write_text(
            proved_cert_lean(make_module(), claim, cert))

    # Certs/LearnedCountdown.lean — the certificate `--infer nuterm` produces,
    # rather than one written by hand. Its ranking function is a sum of ReLU
    # units, `(+ (* c (ite (> <affine> 0) <affine> 0)) ...)`, which no
    # hand-written certificate in this suite has: whether `zeroth_hammer`
    # closes `hrank` for that shape is a question only the learner's own
    # output asks. The route proves the obligations before emitting them, so
    # a failure here is the Lean side, not the certificate. Without cvc5
    # there is nothing to generate.
    try:
        import cvc5  # noqa: F401
    except ImportError:
        pass
    else:
        from verith.nuterm.engine import NutermEngine

        module = _make_countdown()
        claim = Claim(prp="(= s0 0)", kind="buchi")
        learned = NutermEngine(module, log=lambda *a: None).infer(claim)
        # Generated from the SMT-LIB the route answered with, as `verith`
        # does it.
        (CERTS / "LearnedCountdown.lean").write_text(
            proved_cert_lean(module, claim, learned))

    # Certs/Bridge*.lean — the `--fbk-proveit` route's model, the module's
    # own encodings, and the proof that the first is the second. The bridge
    # normally imports `System.*` and the NA model from a generated project;
    # inlining all of them in the order the project has them is the same
    # elaboration problem, and it runs in this warm build instead of a cold
    # `lake update` per case.
    from verith.fbk_proveit.bridge import atom_to_lean_fbk_bridge
    from verith.fbk_proveit.engine import property_to_bool_lean
    from verith.fbk_proveit.na import atom_to_lean_na

    for name, make_module, prop in _BRIDGE_SPECS:
        module = make_module()
        ctx = LeanContext(module)
        t = ModuleToLean4(module)
        cert = render(module, Claim(prp=prop, kind="safety"))
        (CERTS / f"Bridge{name}.lean").write_text(
            "\n".join([
                "import Relational.ReactiveTS", "import Relational.ProofRules",
                "import Smt", "", LINTERS_OFF, "",
                t.to_lean_functional(), "",
                t.to_lean_scalar(), "",
                _strip(generate_data_lean(ctx, cert)), "",
                _strip(atom_to_lean_na(ctx, property_to_bool_lean(module, prop),
                                       module_name=name)), "",
                _strip(atom_to_lean_fbk_bridge(ctx, na_module=name)),
            ]) + "\n"
        )
