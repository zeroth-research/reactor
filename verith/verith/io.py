"""What a route is given and what it hands back.

`ProjectHandle` is the generated project, as much of it as the route's row
declared; `InferInput` is everything a route is run with; `InferResult` is
its answer. What a previous run left a route to resume from is read in
`verith.resume`.

`verith` run without `--project` hands a route a handle with no project
behind it (`dir` is `None`, and so is `artifacts`): it refuses the routes
whose row reads or writes the project before one runs, and to the others
there is nothing to resume from.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from reactor.lean.artifacts import ARTIFACTS_DIR, Artifact, ArtifactStore
from reactor.lean.core.certificate import Certificate, Claim, LeanAnswer
from reactor.lean.core.errors import Refused

if TYPE_CHECKING:
    from reactor import Module

    from reactor.lean.proof.render import LeanCertificate, ProofScripts
    from reactor.lean.codegen.context import LeanContext

    from .common.encoding import Encoding
    from .common.limits import SmtBudget


# ══════════════════════════════════════════════════════════════════════════
# What a route is given
# ══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ProjectHandle:
    """The generated project, as much of it as one route declared.

    A route is given this instead of a directory: ``reads`` and ``owns`` on
    its row are its whole access to the project, so "modifications are
    confined to the certificate's files" is enforced for the
    routes that say so, and a route that needs more of the project says
    which more -- by name, in its row, where the project builder can also
    read it.
    """

    dir: Path | None            # `None`: `verith` run without `--project`
    name: str
    ctx: "LeanContext"          # built by `create_project`; not built again
    # What the route declared (`declared`); nothing without a route.
    reads: frozenset[str] = frozenset()    # ENCODINGS suffixes
    owns: tuple[str, ...] = ()             # project-relative paths
    resumes: frozenset[str] = frozenset()  # artifact roles
    # `artifacts/` belongs to no route and to every route -- one run's
    # leftovers are the next run's starting point, and the run that wrote one
    # is usually not the run that uses it. So it is not an `owns` entry, it is
    # a field, and `path()` refuses it below so that the only way in is the
    # store's own typed writes.
    artifacts: ArtifactStore | None = None

    # --- reading --------------------------------------------------------

    def encoding(self, suffix: str = "") -> str:
        """The Lean of one emitted encoding.  ``""`` is the functional one.

        That is the file the LLM routes prompt on, and asking for it by the
        encoding it is removes the layout arithmetic -- ``System/System.lean``
        is the functional row's path only because `SYSTEM_LAYOUT` says so.
        """
        from reactor.lean.project.scaffold import ENCODINGS, SYSTEM_LAYOUT

        if suffix not in {e.suffix for e in ENCODINGS}:
            raise Refused(f"there is no {suffix!r} encoding to read")
        if suffix not in self.reads:
            raise Refused(
                f"this route did not declare that it reads the {suffix!r} "
                f"encoding (its row says: {sorted(self.reads)})"
            )
        layout = replace(SYSTEM_LAYOUT, directory=self.dir / "System")
        return layout.path(suffix).read_text()

    def resume(
        self,
        role: str,
        *,
        languages: "tuple[str, ...] | None" = None,
        statuses: "tuple[str, ...] | None" = None,
    ) -> tuple[Artifact, ...]:
        """What a previous run left of `role`, newest first.

        Declared access, like `encoding()`: the row's `reads_artifacts` is
        what a route may resume from. The filtering is the store's -- it is
        the half that knows what a stale artifact is, and it reports what it
        dropped, which a filter applied out here would do silently.
        """
        if role not in self.resumes:
            raise Refused(
                f"this route did not declare that it resumes from {role!r} "
                f"artifacts (its row says: {sorted(self.resumes)})"
            )
        if self.artifacts is None:
            return ()
        return self.artifacts.usable(role, languages=languages, statuses=statuses)

    # --- the ordinary write --------------------------------------------
    #
    # The only write most runs want: after inference the predicates are new,
    # so the definitions (`Data.lean`) and the proofs written from their
    # shape (`Proofs.lean`) are both restated, as
    # `proof.render.certificate_files` renders them -- with `Module.lean`,
    # which is the same text whatever the predicates, and the certificate
    # (`Certificate.lean`), which is the same text unless the kind of the
    # property changed -- and the hammer the proofs call goes in beside them.

    def write_certificate(self, lean_cert: "LeanCertificate",
                          proofs: "ProofScripts") -> None:
        """The certificate's four files, rewritten from `lean_cert` and
        proved by `proofs`, and the hammer."""
        from reactor.lean.project.scaffold import write_files
        from reactor.lean.proof.render import certificate_files

        from .proofs import hammer_files

        write_files(self.dir, {**certificate_files(self.ctx, lean_cert, proofs),
                               **hammer_files()})

    # --- writing more than that -----------------------------------------

    def path(self, rel: str) -> Path:
        """A file this route declared it owns, with its parent made.

        Refuses anything undeclared.  The point is not to police a route --
        it could import `pathlib` -- but to make the declaration the truth:
        the pipeline can then clean up what no route in *this* run owns
        (`create_project` unlinks what every row owns before the project is
        written), instead of hardcoding one route's filenames.
        """
        if rel == ARTIFACTS_DIR or rel.startswith(ARTIFACTS_DIR + "/"):
            raise Refused(
                f"{ARTIFACTS_DIR}/ is written through `project.artifacts`, not "
                f"`project.path`: an artifact with no role, status or `why` is "
                f"a file the next run can do nothing with but read"
            )
        if not self.owns_path(rel):
            raise Refused(
                f"{rel} is not among this route's declared paths "
                f"({', '.join(self.owns) or 'none'})"
            )
        out = self.dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    def owns_path(self, rel: str) -> bool:
        """Whether `rel` is one of the declared paths, or under a declared
        directory."""
        return any(
            rel == o or rel.startswith(o.rstrip("/") + "/") for o in self.owns
        )

    def write_files(self, files: Mapping[str, str]) -> None:
        """Write whole files a route handed back in `InferResult.files`, in
        the mapping's order, each through `path()`."""
        for rel, text in files.items():
            out = self.path(rel)
            out.write_text(text)
            print(f"Wrote {out}")


def declared(route) -> dict:
    """What `route`'s row declares of the project, as `ProjectHandle`'s
    fields: nothing without a route."""
    if route is None:
        return {}
    return dict(reads=frozenset(route.reads), owns=route.owns,
                resumes=frozenset(route.reads_artifacts))


@dataclass(frozen=True)
class InferInput:
    """Everything a route is run with.

    One object rather than a constructor signature per route: each row's
    `run` takes from it what its engine needs -- the Lean text, the module,
    the budget, the artifact store -- and no engine is asked for what it
    does not use.
    """

    project: ProjectHandle
    module: "Module"
    claim: Claim
    # What the route starts from: `--invariant` / `--ranking`, or an
    # invariant `taking_inv` resumed from `artifacts/`.
    given: Certificate
    opts: Mapping[str, Any]             # this route's own options, by dest
    # What the row's `resolve` and `precheck` returned, each of a type
    # only that row and its engine know.
    config: object | None = None
    prechecked: object | None = None
    # This run's `--smt-timeout` / `--smt-budget`, for a route that asks cvc5
    # a question of its own. The obligation queries have always been on it
    # (`common.limits.SmtBudget`); a route that *searches* has to be, for the
    # same reason and more so -- a synthesis with no limit hangs generation,
    # and `tlimit-per` is the option that actually stops one.
    budget: "SmtBudget | None" = None
    log: Callable = print
    # The run's one cvc5 encoding of `module`, built on first call.
    smt: "Callable[[], Encoding] | None" = None


# ══════════════════════════════════════════════════════════════════════════
# What a route hands back
# ══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class InferResult:
    """What a route answered: SMT-LIB in `cert`, or Lean in `lean` for the
    one row that answers in Lean (`returns="lean"`). A route whose answer is
    a certificate it produced whole hands back `files` instead and no
    predicates (`returns="files"`).

    `files` are whole files by project-relative path, which the pipeline
    writes; each has to be a path the row owns.
    """

    cert: Certificate = Certificate()
    lean: LeanAnswer | None = None
    files: Mapping[str, str] = field(default_factory=dict)
