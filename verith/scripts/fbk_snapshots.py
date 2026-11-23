"""The fbk-proveit NA model and bridge, over every fixture module.

The route transcripts (`route_transcripts.py`) cover the files a run leaves,
which for `--infer fbk-proveit` needs ic3ia and the proof-prototyping
checkout; this writes the two Lean texts the route generates from the module
alone, so a change to the route's codegen -- or to the encodings beneath
it, which `lean/scripts/encoding_snapshots.py` records -- is checked the
same way: record at the commit before and after, then `diff -r`. Run from
`verith/`:

    uv run python scripts/fbk_snapshots.py OUTDIR

OUTDIR/<fixture>/ gets `NA.lean` and `Bridge.lean` where
`check_na_supported` takes the module (`property_lean` "true"). A stage
that refuses writes `<stage>.error` with the refusal instead. Every file
name ends in `.txt`, which the repository ignores.
"""

import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1] / "tests"
FIXTURES = [
    *sorted((TESTS / "fixtures").glob("*.py")),
    *sorted((TESTS / "limits" / "mods").glob("m_*.py")),
    *sorted((TESTS / "smv_fixtures").glob("*.smv")),
]


def _write(out: Path, name: str, produce) -> None:
    try:
        text = produce()
    except Exception as e:                  # a refusal is part of the record
        (out / f"{name}.error.txt").write_text(f"{type(e).__name__}: {e}\n")
        return
    (out / f"{name}.txt").write_text(text)


def snapshot(path: Path, out: Path) -> None:
    from reactor_lean import inputs
    from reactor_lean.codegen.context import LeanContext
    from verith.fbk_proveit.bridge import atom_to_lean_fbk_bridge
    from verith.fbk_proveit.na import atom_to_lean_na, check_na_supported

    out.mkdir(parents=True, exist_ok=True)
    try:
        ctx = LeanContext(inputs.load(path).module)
        bodies = check_na_supported(ctx)
    except Exception as e:
        (out / "NA.error.txt").write_text(f"{type(e).__name__}: {e}\n")
        return
    _write(out, "NA.lean", lambda: atom_to_lean_na(
        ctx, "true", module_name="NA", bodies=bodies))
    _write(out, "Bridge.lean", lambda: atom_to_lean_fbk_bridge(
        ctx, na_module="NA", bodies=bodies))


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    root = Path(sys.argv[1])
    for path in FIXTURES:
        snapshot(path, root / f"{path.parent.name}.{path.stem}")
        print(path.name, file=sys.stderr)


if __name__ == "__main__":
    main()
