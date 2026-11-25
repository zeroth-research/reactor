"""Every Lean text reactor generates for a module, over every fixture module.

The goldens (`tests/test_pipeline_golden.py`) cover the certificate files,
not the `System/*.lean` encodings or the `--cert-file` certificate of a bare
module. This writes those, so a codegen refactor is checked the same way:
record at the commit before and after, then `diff -r`. Run from `lean/`:

    uv run python scripts/encoding_snapshots.py OUTDIR

OUTDIR/<fixture>/ gets one file per `project.scaffold.ENCODINGS` row (`System.lean`,
`SystemRel.lean`, ...; printed by one `ModuleToLean4`, in table order, as
`create_project` does), each row again from a fresh `ModuleToLean4`
(`alone/`, which is what shows state carried between producers), and
`Cert.lean` (the standalone certificate with no claim). A stage that refuses
writes `<stage>.error` with the refusal instead. Every file name ends in
`.txt`, which the repository ignores.
"""

import sys
import traceback
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1] / "tests"
FIXTURES = [
    *sorted((TESTS / "fixtures").glob("*.py")),
    *sorted((TESTS / "limits" / "mods").glob("m_*.py")),
]


def _write(out: Path, name: str, produce) -> None:
    # `.txt` on everything: the repository ignores it, so a snapshot kept
    # beside the checkout stays out of `git status`.
    try:
        text = produce()
    except Exception as e:                  # a refusal is part of the record
        (out / f"{name}.error.txt").write_text(
            f"{type(e).__name__}: {e}\n" if not isinstance(e, AssertionError)
            else traceback.format_exc())
        return
    (out / f"{name}.txt").write_text(text)


def snapshot(path: Path, out: Path) -> None:
    from reactor_lean.parse import load
    from reactor_lean.codegen.context import LeanContext
    from reactor_lean.codegen.context import lean_gaps
    from reactor_lean.project.scaffold import ENCODINGS
    from reactor_lean.proof.render import generate_standalone_cert_lean
    from reactor_lean.codegen.encodings import ModuleToLean4

    out.mkdir(parents=True, exist_ok=True)
    try:
        module = load(path).module
        ctx = LeanContext(module)
        gaps = lean_gaps(ctx)
    except Exception as e:
        (out / "load.error.txt").write_text(f"{type(e).__name__}: {e}\n")
        return
    if gaps:
        (out / "load.error.txt").write_text("no Lean form: " + "; ".join(gaps) + "\n")
        return

    shared = ModuleToLean4(ctx)
    (out / "alone").mkdir(exist_ok=True)
    for enc in ENCODINGS:
        name = f"System{enc.suffix}.lean"
        _write(out, name, lambda enc=enc: enc.producer(shared))
        _write(out / "alone", name,
               lambda enc=enc: enc.producer(ModuleToLean4(LeanContext(module))))
    _write(out, "Cert.lean", lambda: generate_standalone_cert_lean(LeanContext(module)))


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    root = Path(sys.argv[1])
    for path in FIXTURES:
        snapshot(path, root / f"{path.parent.name}.{path.stem}")
        print(path.name, file=sys.stderr)


if __name__ == "__main__":
    main()
