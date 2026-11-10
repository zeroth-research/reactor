"""What a run writes: the project (`generate`), a standalone certificate
(`standalone`, `--cert-file`), the hammer tactic alone (`hammer_only`,
`--hammer-file`), and the `lake build` `--build-cert` asks for
(`build_cert`).
"""

from __future__ import annotations

from pathlib import Path

from ..cli import Settings
from ..codegen.context import LeanContext
from ..codegen.encodings import ModuleToLean4
from ..core.errors import Refused
from .. import inputs
from ..project.build import LakeBuildError, build_certificate
from ..project.lakefile import LakefileParts
from ..project.names import CERT_LIB
from ..project.scaffold import (
    ENCODINGS,
    Layout,
    create_project,
    generate_zeroth_hammer_lean,
    write_encoding,
)
from ..proof.render import (
    LeanCertificate,
    certificate_files,
    generate_standalone_cert_lean,
)


def build_cert(project_dir: Path, lean_cert: "LeanCertificate | None") -> None:
    """`--build-cert`, once the certificate is written.

    `lean_cert` is what the project's certificate was generated from, or
    `None` when its files came from elsewhere. The predicates are re-checked
    here rather than only at parse time because the parse-time check sees
    the flags, not the certificate a caller hands over: one with `inv` or
    `ranking` unset would build as a proof of `sorry`.
    """
    if lean_cert is not None:
        defs = lean_cert.defs
        # A safety certificate has no ranking function to be missing.
        required = [("property", defs.prp), ("invariant", defs.inv)]
        if not defs.is_safety:
            required.append(("ranking", defs.ranking))
        missing = [name for name, value in required if not value]
        if missing:
            raise SystemExit(
                f"error: --build-cert: the certificate has no "
                f"{', '.join(missing)}, so every obligation it carries is "
                "`sorry`."
            )
    print(f".. Building the certificate in {project_dir}")
    try:
        build_certificate(project_dir)
    except LakeBuildError as e:
        raise SystemExit(f"error: --build-cert: {e}") from e


def hammer_only(hammer_file: str) -> None:
    """`--hammer-file`: write ZerothHammer.lean and return. No module needed."""
    out = Path(hammer_file)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(generate_zeroth_hammer_lean())
    except OSError as e:
        raise SystemExit(f"error: --hammer-file: {e}") from e
    print(f"Wrote {out}")


def standalone(cert_file: str, ctx: LeanContext, lean_cert) -> None:
    """`--cert-file`: a self-contained certificate and its sibling encodings.

    No project: the predicates are the ones supplied.
    """
    out = Path(cert_file)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(generate_standalone_cert_lean(ctx, lean_cert))
    except (Refused, OSError) as e:
        raise SystemExit(f"error: --cert-file: {e}") from e
    print(f"Wrote standalone certificate: {out}")
    # The same encoding table the project walks, so these files and
    # the ones in a generated project carry the same headers and the same
    # imports. The functional encoding is already inlined in the standalone
    # certificate above, which is the base they import.
    m2l = ModuleToLean4(ctx)
    layout = Layout(base=out.stem, flat=True, directory=out.parent)
    for enc in ENCODINGS:
        if not enc.cert_file:
            continue
        print(f"Wrote {enc.title.lower()}: {write_encoding(enc, m2l, layout, out.stem)}")


def generate(settings: Settings, module, ctx: LeanContext,
             lean_cert: "LeanCertificate | None",
             source: "inputs.SourceText", *,
             lakefile: "LakefileParts | None" = None) -> Path:
    """The Lean project, with the certificate as it stands.

    `lakefile` is what a caller adds to the lakefile; the lakefile is
    written here, with the rest of the project.
    """
    print(".. Generating lean code")
    certificate = certificate_files(ctx, lean_cert)
    try:
        return create_project(
            output_dir=settings.output_dir,
            module=module,
            project_name=settings.project_name,
            executable=settings.executable,
            certificate=certificate,
            source=source,
            lakefile=lakefile,
            remove=stale_certificate_files(
                settings.output_dir / settings.project_name, certificate),
            ctx=ctx,
        )
    except (Refused, OSError) as e:
        raise SystemExit(f"error: {e}") from e


def stale_certificate_files(project_dir: Path,
                            certificate: "dict[str, str]") -> tuple[str, ...]:
    """The files under `Certificate/` this run will not write.

    `Certificate/` is globbed by the build, so a file an earlier run in this
    `-o` left there would be built against this run's module. What wrote it
    is not the point: anything there that is not this run's certificate is a
    leftover. Directories beside `Certificate/` are left alone: nothing
    globs them until the lakefile declares a lean_lib for them.
    """
    root = project_dir / CERT_LIB
    if not root.is_dir():
        return ()
    return tuple(sorted(
        rel for rel in (p.relative_to(project_dir).as_posix()
                        for p in root.rglob("*.lean"))
        if rel not in certificate
    ))
