#!/usr/bin/env python3
"""Run the verith limit probe: one full Lean project per case, shared .lake.

    uv run python run_limits.py [name ...]

Every project gets `.lake` symlinked to a single shared build dir whose
`packages` points at the already-built Mathlib/cslib/lean-smt of the Lake
package cache (`$REACTOR_LAKE_CACHE`). ZerothHammer is byte-identical
across projects, and ReactLib's packages are path requires on one checkout
(`$REACTLIB`), so they are built once and hit the cache afterwards; only
System/* and Certificate/* recompile per case. `just vr-run` sets both
variables.

Verdicts land in $VERITH_LIMITS_WORK/results.json (default
/tmp/verith-limits); a table is printed as it goes.

Two environment switches pick *what* is measured, so the same plumbing --
one project per case against a shared `.lake` -- serves more than the one
question the matrix was written for:

    VERITH_SUITE=limits|fbk    which case table (default `limits`)
    VERITH_ROUTE=supplied|nuterm   where the certificate comes from

`VERITH_SUITE=fbk` reads `verith/tests/fbk/probes.py`, whose probes are
`--safety` properties over the same modules; `run_fbk.py` puts them through
ic3ia, and this puts them through verith's own certificate.

`VERITH_ROUTE=nuterm` drops each case's supplied `inv`/`rank`/`pre` and
passes `--infer nuterm` instead. The matrix varies the supplied predicates
over a much smaller set of (module, property) pairs, so an inferring route
is asked each *distinct* pair once and the case names that share it are
listed together. A non-default suite or route writes its own results file,
`results-<suite>-<route>.json`, rather than overwriting the baseline's.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

SP = Path(__file__).parent                  # verith/tests/limits
REPO = SP.parent.parent.parent
MODS = SP / "mods"

# Generated projects and their shared build dir are build artifacts, so they
# live outside the tree. Override with VERITH_LIMITS_WORK to put them
# elsewhere -- see the single-writer warning in README.md.
#
# The `.noindex` suffix is load-bearing on macOS: Spotlight will happily
# index tens of thousands of freshly written `.olean` files, and
# `spotlightknowledged` pinning a core turns a 9 s case into a 928 s one.
# A directory whose name ends in `.noindex` is skipped. Keep the suffix on
# any path passed through VERITH_LIMITS_WORK too.
WORK = Path(os.environ.get("VERITH_LIMITS_WORK", "/tmp/verith-limits.noindex"))
PROJECTS = WORK / "projects"
SHARED = WORK / "shared_lake"
WORK.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SP))
from cases import CASES  # noqa: E402
from reactor_lean.inputs import to_surface  # noqa: E402
from reactor_lean.project.lake_cache import LAKE_CACHE_ENV, lake_cache  # noqa: E402

# The Lake package cache every project builds against, and the ReactLib
# checkout the projects require by path: both from the environment
# (`$REACTOR_LAKE_CACHE`, `$REACTLIB`; `just vr-run` sets them), and both
# inherited by the `certify` runs below, which write each project's
# manifest against the cache.
CACHE = lake_cache()

DEFAULT_TIMEOUT = 900

SUITE = os.environ.get("VERITH_SUITE", "limits")
ROUTE = os.environ.get("VERITH_ROUTE", "supplied")


def load_cases() -> list:
    """The case table `VERITH_SUITE` names, as this runner's case dicts.

    An inferring route reads none of the supplied predicates, so the cases
    that differ only in those are one problem: they are collapsed onto their
    (module, property, precondition) key and `shared` lists the names that
    fell together, so a verdict is not counted as many times as the matrix
    happens to spell it."""
    if SUITE == "limits":
        cases = list(CASES)
    elif SUITE == "fbk":
        # The fbk probe table is the route's, beside this directory.
        sys.path.insert(0, str(SP.parent / "fbk"))
        from probes import PROBES                        # noqa: PLC0415
        cases = [dict(name=n, group="fbk", mod=m, S=prop, expect=e)
                 for n, m, prop, e in PROBES]
    else:
        raise SystemExit(f"error: VERITH_SUITE={SUITE!r} is not `limits` or `fbk`")

    if ROUTE == "supplied":
        return cases
    problems: dict = {}
    for c in cases:
        if not (c.get("P") or c.get("S")):
            continue          # nothing to infer against: the case has no property
        key = (c["mod"], c.get("P"), c.get("S"), c.get("pre") or "")
        problems.setdefault(key, []).append(c)
    out = []
    for group in problems.values():
        first = dict(group[0])
        first["shared"] = [c["name"] for c in group]
        out.append(first)
    return out


def module_path(mod: str) -> Path:
    if mod.startswith("TESTS/"):
        return SP.parent / "fixtures" / f"{mod.split('/', 1)[1]}.py"
    return MODS / f"{mod}.py"


def run_verith(case) -> dict:
    out = PROJECTS / case["name"]
    if out.exists():
        shutil.rmtree(out)
    # `certify` (the `verith/` project) generates the project and proves it:
    # from the case's own predicates, or, in the nuterm column, from the
    # ones it infers. `reactor` alone leaves every obligation `sorry`.
    cmd = ["uv", "run", "certify", str(module_path(case["mod"]))]
    if case.get("P"):
        # A limit-matrix property is a Buchi one: `P` comes with a ranking.
        cmd += ["--buchi", to_surface(case["P"])]
    if case.get("S"):
        cmd += ["--safety", to_surface(case["S"])]
    if ROUTE == "nuterm":
        # The route computes the whole certificate and refuses a supplied
        # predicate or a precondition, so this is the case stripped to its
        # question: this module, this property, find the rest.
        cmd += ["--infer", "nuterm"]
    else:
        if case.get("inv"):
            cmd += ["--invariant", to_surface(case["inv"])]
        if case.get("rank"):
            cmd += ["--ranking", to_surface(case["rank"])]
        if case.get("pre"):
            cmd += ["--pre", to_surface(case["pre"])]
    cmd += ["-o", str(out), "-p", "Rea"]
    cmd += os.environ.get("VERITH_EXTRA", "").split()
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=REPO / "verith", capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return dict(ok=False, secs=time.time() - t0, err="verith timed out (600s)")
    if r.returncode != 0:
        tail = (r.stderr or r.stdout).strip().splitlines()
        err = " | ".join(ln.strip() for ln in tail[-3:])
        return dict(ok=False, secs=time.time() - t0, err=err)
    return dict(ok=True, secs=time.time() - t0, err="")


_ERR = re.compile(r"^(?:error|warning): (?P<file>[^:]+\.lean):(\d+):(\d+): (?P<msg>.*)$")
_FAILED_TGT = re.compile(r"^✖ \[[\d/]+\] Built (?P<t>\S+)")


def ensure_shared_lake() -> None:
    """Point the shared build dir's `packages` at the cache's.

    Without this every project tries to fetch and build Mathlib for itself,
    which shows up as `external command 'git' exited with code 128` on every
    case at once. `packages` is only ever read, so several build dirs may
    point at it -- unlike the build dir itself, which takes one writer.
    """
    if CACHE is None:
        raise SystemExit(f"error: no Lake package cache: set ${LAKE_CACHE_ENV} "
                         f"(`just vr-run` sets it)")
    warm = CACHE / "packages"
    SHARED.mkdir(parents=True, exist_ok=True)
    link = SHARED / "packages"
    if link.is_symlink():
        if link.readlink() == warm:
            return
        link.unlink()
    elif link.exists():
        raise SystemExit(f"error: {link} exists and is not a symlink")
    link.symlink_to(warm)


def link_lake(proj: Path, shared: Path) -> None:
    """Point `proj`'s `.lake` at `shared`, the one build dir. `certify`
    made it a directory holding the link to the cache's packages."""
    lake = proj / ".lake"
    if lake.is_symlink():
        lake.unlink()
    elif lake.is_dir():
        shutil.rmtree(lake)
    lake.symlink_to(shared)


def run_lake(case) -> dict:
    proj = PROJECTS / case["name"] / "Rea"
    link_lake(proj, SHARED)

    t0 = time.time()
    try:
        r = subprocess.run(
            ["lake", "build"], cwd=proj, capture_output=True, text=True,
            timeout=case.get("timeout", DEFAULT_TIMEOUT),
        )
    except subprocess.TimeoutExpired:
        return dict(ok=False, secs=time.time() - t0, targets=["<timeout>"],
                    errors=[f"lake build timed out ({case.get('timeout', DEFAULT_TIMEOUT)}s)"],
                    sorries=[])
    out = r.stdout + "\n" + r.stderr
    lines = out.splitlines()

    failed = [m.group("t") for ln in lines if (m := _FAILED_TGT.match(ln))]
    errors, sorries = [], []
    for i, ln in enumerate(lines):
        if ln.startswith("error: ") and ".lean:" in ln:
            m = _ERR.match(ln)
            if m:
                # first message line, plus the next line if the message is empty
                msg = m.group("msg").strip()
                if not msg and i + 1 < len(lines):
                    msg = lines[i + 1].strip()
                errors.append(f"{Path(m.group('file')).name}: {msg}")
            else:
                errors.append(ln[7:])
        elif ln.startswith("error: ") and "Lean exited" not in ln:
            errors.append(ln[7:])
        if "declaration uses 'sorry'" in ln:
            m = _ERR.match(ln)
            sorries.append(
                f"{Path(m.group('file')).name}:{m.group(2)}" if m else ln.strip()
            )
    return dict(ok=r.returncode == 0 and not sorries, raw_ok=r.returncode == 0,
                secs=time.time() - t0, targets=failed,
                errors=errors[:6], sorries=sorries[:8])


def verdict(case, gen, build) -> str:
    if not gen["ok"]:
        # An inferring route that finds nothing is not a generation bug; it
        # is the measurement. `main` wraps its refusal under this prefix.
        if "error: --infer:" in gen["err"]:
            return "NO-CERT"
        return "GEN-FAIL"
    if build.get("sorries") and build.get("raw_ok"):
        return "SORRY"
    if not build["ok"]:
        return "PROOF-FAIL" if not build["targets"] or "Certificate" in " ".join(build["targets"]) else "BUILD-FAIL"
    return "VERIFIED"


def main() -> None:
    only = set(sys.argv[1:])
    PROJECTS.mkdir(parents=True, exist_ok=True)
    ensure_shared_lake()
    stem = ("results" if (SUITE, ROUTE) == ("limits", "supplied")
            else f"results-{SUITE}-{ROUTE}")
    results_path = WORK / f"{stem}.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    todo = [c for c in load_cases() if not only or c["name"] in only]
    print(f"{len(todo)} cases  (suite {SUITE}, route {ROUTE})\n")
    for n, case in enumerate(todo, 1):
        name = case["name"]
        print(f"[{n}/{len(todo)}] {name:<16} ", end="", flush=True)
        gen = run_verith(case)
        build = (dict(ok=False, secs=0.0, targets=[], errors=["(not built: generation failed)"], sorries=[])
                 if not gen["ok"] else run_lake(case))
        v = verdict(case, gen, build)
        results[name] = dict(case={k: v2 for k, v2 in case.items() if k != "timeout"},
                             gen=gen, build=build, verdict=v)
        results_path.write_text(json.dumps(results, indent=1))
        # `expect` records what the *supplied* predicates were meant to show,
        # so it says nothing about a route that replaces them.
        flag = ("" if ROUTE != "supplied" or (v == "VERIFIED") == (case["expect"] == "ok")
                else "  <-- unexpected")
        print(f"{v:<11} gen {gen['secs']:5.1f}s  build {build['secs']:6.1f}s{flag}")
        if len(case.get("shared", [])) > 1:
            print(f"      shares this problem: {', '.join(case['shared'][1:])}")
        if v in ("GEN-FAIL", "NO-CERT"):
            print(f"      {gen['err'][:200]}")
        for e in (build["errors"] or [])[:2]:
            print(f"      {e[:150]}")
    print("\nwrote", results_path)


if __name__ == "__main__":
    main()
