"""The project on disk, written by `create_project`: its directories, the
lakefile (which requires the Lean libraries from ReactLib) and
toolchain, the module's encodings (`ENCODINGS`), and the certificate's
files, which it is handed as text (`proof.render.certificate_files`) and
writes as they are.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from ..codegen.context import LeanContext
from ..codegen.encodings import ModuleToLean4
from .names import CERT_LIB, LINTERS_OFF, SYSTEM_LAYOUT, Layout
from .templates import render
from .lake_cache import Packages, project_packages, use_lake_cache
from .lakefile import LEAN_TOOLCHAIN, generate_lakefile
from .main_lean import generate_main_lean


# ══════════════════════════════════════════════════════════════════════════
# The emitted encodings
#
# One row per Lean file `ModuleToLean4` prints, and one writer for all of
# them. Both routes that emit encodings -- the project (`create_project`)
# and the standalone `--cert-file` -- walk this same table, so the header,
# the imports and the producer agree across the two by construction rather
# than by inspection. They differ only in how a row becomes a path and a
# Lean module name, which is what `Layout` supplies.
#
# Row order is emission order only: each producer prints its definitions
# and the theorems about them in one call, so no row reads state another
# left behind. Rows are coupled by Lean names -- `ScalarRel` rewrites by
# `Scalar`'s `init_scalar_eq`, every row quotes the base's `init`/`update`
# -- and `Encoding.needs`, which becomes each file's imports, records that.
# ══════════════════════════════════════════════════════════════════════════




@dataclass(frozen=True)
class Encoding:
    """One Lean file: what names it, what it says, what it needs, who prints it."""

    suffix: str                 # appended to the layout's base; "" is the base
    title: str                  # header comment, "<title> of reactive module `X`"
    core: str                   # the `Relational` module it is written against
    needs: tuple[str, ...]      # suffixes of the sibling encodings it imports
    producer: Callable[[ModuleToLean4], str]    # prints it
    cert_file: bool = True      # also written by the `--cert-file` route


ENCODINGS: tuple[Encoding, ...] = (
    # The functional encoding is the base every other row imports. The
    # `--cert-file` route gets its own copy inlined into the standalone
    # certificate by `generate_standalone_cert_lean`, so it is not written
    # again there.
    Encoding("", "Functional encoding", "Relational.Mat", (),
             ModuleToLean4.to_lean_functional, cert_file=False),
    Encoding("Rel", "Matrix-domain relational encoding", "Relational.Basic", ("",),
             ModuleToLean4.to_lean_mat_rel),
    Encoding("Scalar", "Scalar encoding", "Relational.Basic", ("",),
             ModuleToLean4.to_lean_scalar),
    Encoding("ScalarRel", "Scalar-relational encoding", "Relational.Basic",
             ("Scalar",), ModuleToLean4.to_lean_rel),
)


def write_encoding(
    enc: Encoding, m2l: "ModuleToLean4", layout: Layout, module_name: str
) -> Path:
    """Print one encoding and write it where `layout` says it goes."""
    imports = [enc.core] + [layout.lean_module(s) for s in enc.needs]
    path = layout.path(enc.suffix)
    path.write_text(
        f"/- {enc.title} of reactive module `{module_name}` -/\n"
        + "".join(f"import {i}\n" for i in imports)
        + f"\n{LINTERS_OFF}\n\n{enc.producer(m2l)}\n"
    )
    return path


def generate_root() -> str:
    """Root module file that imports the `System.*` encodings.

    The import list is the encoding table, so a row added there reaches the
    root without a second edit.
    """
    return render(
        "project/Root.lean.j2",
        imports=[SYSTEM_LAYOUT.lean_module(e.suffix) for e in ENCODINGS],
    )


def write_files(project_dir: Path, files: Mapping[str, str]) -> None:
    """Write whole files by project-relative path, each directory made."""
    for rel, text in files.items():
        path = project_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"Wrote {path}")


def write_lakefile(project_dir: Path, project_name: str, packages: Packages,
                   *, executable: bool) -> Path:
    """The project's `lakefile.toml`, requiring `packages.requires`.

    `create_project` writes it. With a cache, the project's manifest is
    written with it, for the same requires (`use_lake_cache`).
    """
    cache = packages.cache
    path = project_dir / "lakefile.toml"
    path.write_text(generate_lakefile(project_name, packages.requires,
                                      executable=executable))
    print(f"Wrote {path}")
    if cache is not None and use_lake_cache(
            project_dir, packages.requires, name=project_name, cache=cache):
        print(f"Wrote {project_dir / 'lake-manifest.json'}, "
              f"its packages {cache / 'packages'}")
    return path


def create_project(
    output_dir: Path,
    ctx: LeanContext,
    project_name: str,
    executable: bool = False,
    *,
    certificate: Mapping[str, str],
    source: str | None = None,
    packages: Packages | None = None,
) -> Path:
    """
    Create a full Lean4 project.

    Args:
        `output_dir`:      Where to create the project folder.
        `ctx`:             The module's `LeanContext`, as
                           `codegen.context.lean_context` checks it.
        `project_name`:    The name of the Lean package, and of its folder.
        `executable`:      If True, generate Main.lean and add [[lean_exe]] to lakefile.
        `certificate`:     The certificate's files by project-relative path,
                           as `proof.render.certificate_files` renders them.
        `source`:          The text of the input the module was read from,
                           written to `dbg/system.py` as a debug artifact.
        `packages`:        What the lakefile requires, and the cache it
                           builds against; `project_packages()` if not given.
    """
    if packages is None:
        packages = project_packages()
    project_dir = output_dir / project_name
    src_dir = project_dir / SYSTEM_LAYOUT.base
    dbg_dir = project_dir / "dbg"

    # Create directory structure
    src_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created project directory: `{project_dir}`")

    # Debug artifacts: module representation and the input it was read from
    dbg_dir.mkdir(parents=True, exist_ok=True)
    (dbg_dir / "system.txt").write_text(str(ctx.module))
    print("Wrote system.txt")
    if source is not None:
        (dbg_dir / "system.py").write_text(source)
        print("Wrote system.py")

    # Render and write project-level files from templates
    write_lakefile(project_dir, project_name, packages, executable=executable)

    toolchain = project_dir / "lean-toolchain"
    toolchain.write_text(
        render("project/lean-toolchain.j2", lean_toolchain=LEAN_TOOLCHAIN)
    )
    print(f"Wrote {toolchain}")

    (project_dir / f"{CERT_LIB}.lean").write_text(render("project/Certificate.lean.j2"))

    # ----------------------------------------------------------
    # Generate reactive module (init and update)
    # ----------------------------------------------------------
    m2l = ModuleToLean4(ctx)

    root_lean = project_dir / f"{SYSTEM_LAYOUT.base}.lean"
    root_lean.write_text(generate_root())
    print(f"Wrote root module {root_lean}")

    # Every encoding the project carries, in table order.
    layout = replace(SYSTEM_LAYOUT, directory=src_dir)
    for enc in ENCODINGS:
        print(f"++ Generated {write_encoding(enc, m2l, layout, project_name)} ++")

    # The certificate: its definitions, and the proofs it was handed.
    write_files(project_dir, certificate)

    # ----------------------------------------------------------
    # Generate Main.lean for executable
    # ----------------------------------------------------------
    if executable:
        main_lean = project_dir / "Main.lean"
        main_lean.write_text(generate_main_lean(ctx))
        print(f"Wrote {main_lean}")

    print()
    print(f"DONE: Project created at: {project_dir}")
    return project_dir
