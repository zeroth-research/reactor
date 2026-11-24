"""What a run writes: the project (`generate`), or a standalone
certificate (`standalone`, `--cert-file`). Both prove the obligations by
`sorry`.
"""

from __future__ import annotations

from pathlib import Path

from .settings import Settings
from ..codegen.context import LeanContext
from ..codegen.encodings import ModuleToLean4
from ..core.errors import Refused
from ..project.scaffold import ENCODINGS, Layout, create_project, write_encoding
from ..proof.render import (
    LeanCertificate,
    certificate_files,
    generate_standalone_cert_lean,
)


def standalone(cert_file: str, ctx: LeanContext, lean_cert) -> None:
    """`--cert-file`: a self-contained certificate and its sibling encodings.

    No project: the predicates are the ones supplied.
    """
    out = Path(cert_file)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(generate_standalone_cert_lean(ctx, lean_cert))
        print(f"Wrote standalone certificate: {out}")
        # The same encoding table the project walks, so these files and
        # the ones in a generated project carry the same headers and the
        # same imports. The functional encoding is already inlined in the
        # standalone certificate above, which is the base they import.
        m2l = ModuleToLean4(ctx)
        layout = Layout(base=out.stem, flat=True, directory=out.parent)
        for enc in ENCODINGS:
            if enc.cert_file:
                path = write_encoding(enc, m2l, layout, out.stem)
                print(f"Wrote {enc.title.lower()}: {path}")
    except (Refused, OSError) as e:
        raise SystemExit(f"error: --cert-file: {e}") from e


def generate(settings: Settings, module, ctx: LeanContext,
             lean_cert: "LeanCertificate | None",
             source: str) -> Path:
    """The Lean project, with the certificate as it stands; `source` is
    the input's text, which the project keeps in `dbg/`."""
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
            ctx=ctx,
        )
    except (Refused, OSError) as e:
        raise SystemExit(f"error: {e}") from e
