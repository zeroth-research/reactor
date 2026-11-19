"""The fbk-proveit route's Lean against `reactor.lean`'s.

The NA model `lean2vmt` reads (`verith.fbk_proveit.na`) and the bridge that
proves it is the module (`verith.fbk_proveit.bridge`) are this project's;
what they are proved against -- the project's layout, its scalar encoding,
the flat element order every encoding shares, the certificate's simp set --
is `reactor.lean`'s. These are the checks that the two agree. They moved
here with the route from `reactor`'s own tests (`test_lean_flat_layout.py`,
`test_lean_names.py`, `test_lean_templates.py`), which keep the halves that
are `reactor`'s alone.
"""

import re
from pathlib import Path

import pytest
import torch

from reactor import Bool, Int, LIA, Module, Real, Term, Var, Wire, X
from reactor.lean.codegen.context import LeanContext
from reactor.lean.codegen.encodings.scalar import _unpack_body
from reactor.lean.inputs.python import load_module
from reactor.lean.lowering.elements import flat_layout, flat_size
from reactor.lean.project.names import CERT_LIB, SYSTEM_LAYOUT
from reactor.lean.project.scaffold import create_project
from reactor.lean.proof.render import certificate_files, generate_proofs_lean
from verith.fbk_proveit.bridge import atom_to_lean_fbk_bridge
from verith.fbk_proveit.na import atom_to_lean_na, check_na_supported, slot_accessors
from verith.proofs.scripts import proof_scripts

# The CLI fixtures are `reactor`'s, in the `python/` project beside this one.
FIXTURES = Path(__file__).resolve().parents[2] / "python" / "tests" / "fixtures"

# As `reactor`'s `test_lean_flat_layout.py` has them: one wire list per case,
# the last three the ones no fixture has -- a genuinely 2-D wire, and a mixed
# list where the wire index and the slot index disagree in both directions.
WIRE_LISTS = {
    "one scalar": [Int([1, 1])],
    "two scalars": [Int([1, 1]), Bool([1, 1])],
    "row vector": [Int([1, 3])],
    "column vector": [Int([3, 1])],
    "two vectors": [Int([2, 1]), Real([3, 1])],
    "2x3": [Int([2, 3])],
    "3x2": [Int([3, 2])],
    "scalar then 2x3": [Bool([1, 1]), Int([2, 3])],
    "2x3 then scalar": [Int([2, 3]), Bool([1, 1])],
    "2x3 then 3x2": [Int([2, 3]), Int([3, 2])],
}

# The reductions `Relational.Mat` folds, which every simp set that meets a
# matrix has to unfold (`reactor`'s `test_lean_templates.py`).
REDUCTIONS = ("matMin", "matMax", "argmax_1d", "argmax")


@pytest.fixture(params=list(WIRE_LISTS.values()), ids=list(WIRE_LISTS))
def wires(request):
    """Fresh `Var`s each time: a layout is keyed by wire identity."""
    return [Var(dt) for dt in request.param]


def _tuple_parts(body: str, n: int) -> "list[str]":
    """Split `build_tuple`'s output back into its `n` components.

    `n` has to be told, not inferred: `build_tuple` of one element is that
    element. The split is depth-aware, as a reconstruction's `match i, j
    with` arms are full of commas.
    """
    if n == 1:
        return [body]
    depth, cur, parts = 0, "", []
    for ch in body[1:-1]:
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
            continue
        depth += (ch == "(") - (ch == ")")
        cur += ch
    parts.append(cur.strip())
    assert len(parts) == n, f"split {body!r} into {len(parts)}, expected {n}"
    return parts


def _matrix_module() -> Module:
    """`x : Int 2x3`, `x' = x + 1`, from a matrix whose entries are all different.

    The distinct entries are the point: an `init_k` list that is right up to a
    transposition still matches on a constant matrix of equal elements.
    """
    x = Var(Int([2, 3]))
    ones = Wire(Int([2, 3]))
    start = torch.tensor([[1, 2, 3], [4, 5, 6]])
    init = [Term(LIA.Int(start), [X(x)])]
    update = [
        Term(LIA.Int(torch.ones((2, 3), dtype=torch.int64)), [ones]),
        Term(LIA.Add(), [X(x)], [x, ones]),
    ]
    return Module.sequential([x], init, update)


# ── the flat order ──────────────────────────────────────────────────────────


def test_na_state_variables_are_numbered_in_slot_order(wires):
    """The NA model's `var_k` and the flat tuple are the same numbering."""
    layout = flat_layout(wires)
    acc = slot_accessors(wires)
    assert list(acc) == [f"s{i}" for i in range(len(wires))]
    seen = []
    for i, w in enumerate(wires):
        reads = acc[f"s{i}"]
        assert len(reads) == flat_size(w)
        seen += [int(re.match(r"\(var_(\d+) state\)", r).group(1)) for r in reads]
    assert seen == list(range(layout.total))


def test_na_model_init_is_the_matrix_read_row_major():
    """`Definition.init_k` for a 2x3 ctrl wire, against the source tensor.

    The whole chain in one assertion: the Python tensor, `smt.encode`'s cvc5
    tuple, `fbk_proveit.slots.scalar_element`'s selection, and the slot numbering
    `lowering.elements` owns. Transpose any one of them and the constants
    come back permuted.
    """
    ctx = LeanContext(_matrix_module())
    src = atom_to_lean_na(ctx, "true", module_name="matrix")
    init = dict(re.findall(r"abbrev init_(\d+) : Int :=\n  \(?(-?\d+)", src))
    assert [init[str(k)] for k in range(6)] == ["1", "2", "3", "4", "5", "6"]


def test_bridge_reads_the_state_exactly_as_unpack_does():
    """`Bridge.toSlots` against `Scalar.unpack_ctrl`, arm by arm.

    These are the two halves of the FBK equivalence: `toSlots` says which
    model slot an element is, `unpack_ctrl` says which flat component it is.
    They used to be two functions written to match, and a disagreement makes
    `bridge_k` an obligation about the wrong element.
    """
    ctx = LeanContext(_matrix_module())
    src = atom_to_lean_fbk_bridge(ctx, na_module="NA")
    arms = re.findall(r"^  \| (?:_ \+ )?(\d+) => (.+)$", src, re.M)
    to_slots = [body for k, body in arms[:6]]
    assert to_slots == _tuple_parts(_unpack_body("s", ctx.ctrl_next), 6)


# ── the project the bridge imports from ─────────────────────────────────────


def test_every_project_module_the_bridge_imports_is_a_file_the_project_has(
        tmp_path):
    """The bridge imports `System.*` and `Certificate.*` of the project it
    is installed in: each is a file `create_project` writes."""
    module = load_module(FIXTURES / "counter.py")
    project = create_project(tmp_path, module, "Rea",
                             certificate=certificate_files(LeanContext(module)))
    ctx = LeanContext(module)
    bridge = atom_to_lean_fbk_bridge(ctx, na_module="ReaNA",
                                     bodies=check_na_supported(ctx))
    mods = [m for m in re.findall(r"^import (\S+)", bridge, re.M)
            if m.startswith(SYSTEM_LAYOUT.prefix) or m.startswith(f"{CERT_LIB}.")]
    assert mods, "the bridge imports nothing of the project"
    for mod in mods:
        path = project / (mod.replace(".", "/") + ".lean")
        assert path.exists(), f"the bridge imports {mod}, which is not written"


def test_the_fbk_bridge_simp_set_agrees_with_the_certificate_s():
    """The bridge proves the NA model is the module, over the same terms the
    certificate does: its matrix simp set unfolds the same reductions."""
    ctx = LeanContext(load_module(FIXTURES / "counter.py"))
    cert = generate_proofs_lean(None, proof_scripts(ctx))
    simp_mat = [ln for ln in cert.splitlines()
                if "simp [" in ln and "add_Matrix_apply" in ln]
    assert simp_mat, "the certificate's simp_mat macro moved"
    bridge = atom_to_lean_fbk_bridge(ctx, na_module="NA",
                                     bodies=check_na_supported(ctx))
    cascade = [ln for ln in bridge.splitlines() if "add_Matrix_apply" in ln]
    assert cascade, "the bridge's cascade has no matrix simp set"
    for name in REDUCTIONS:
        assert name in simp_mat[0], f"{name} is not in simp_mat"
        assert name in cascade[0], f"{name} is not in the fbk bridge's simp set"
