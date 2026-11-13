"""What a run writes: the project (`generate`), or a standalone
certificate (`standalone`, `--cert-file`). Both prove the obligations by
`sorry` unless a caller hands over the proofs (`ProofScripts`).
"""

from __future__ import annotations

from pathlib import Path

from ..cli import Settings
from ..codegen.context import LeanContext
from ..codegen.encodings import ModuleToLean4
from ..core.errors import Refused
from .. import inputs
from ..project.lakefile import LakefileParts
from ..project.names import CERT_LIB
from ..project.scaffold import ENCODINGS, Layout, create_project, write_encoding
from ..proof.render import (
    SORRY,
    LeanCertificate,
    ProofScripts,
    certificate_files,
    generate_standalone_cert_lean,
)


def standalone(cert_file: str, ctx: LeanContext, lean_cert,
               proofs: ProofScripts = SORRY) -> None:
    """`--cert-file`: a self-contained certificate and its sibling encodings.

    No project: the predicates are the ones supplied.
    """
    out = Path(cert_file)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(generate_standalone_cert_lean(ctx, lean_cert, proofs))
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
             lakefile: "LakefileParts | None" = None,
             proofs: ProofScripts = SORRY) -> Path:
    """The Lean project, with the certificate as it stands, proved by
    `proofs`.

    `lakefile` is what a caller adds to the lakefile; the lakefile is
    written here, with the rest of the project.
    """
    print(".. Generating lean code")
    certificate = certificate_files(ctx, lean_cert, proofs)
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
