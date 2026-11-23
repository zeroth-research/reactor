"""`lake` in the tests' Lake project (`testdata.LAKE_PROJECT`).

The project requires ReactLib's `ReactLib` package at `.lake/ReactLib`, a
link to the checkout `$REACTLIB` that `prepare` makes.
"""

from __future__ import annotations

import functools
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from reactor_lean.project.lakefile import REACTLIB_ENV, Require, reactlib_checkout
from testdata import LAKE_PROJECT

REACTLIB = Require("ReactLib", path=".lake/ReactLib")


@functools.cache
def prepare() -> None:
    """The project ready for `lake`, once a session; skips the test without
    a ReactLib checkout to require."""
    checkout = reactlib_checkout()
    if checkout is None:
        pytest.skip(f"no ReactLib checkout: set ${REACTLIB_ENV}")
    link = LAKE_PROJECT / REACTLIB.path
    if link.is_symlink():
        link.unlink()
    link.parent.mkdir(exist_ok=True)
    link.symlink_to(checkout / "ReactLib", target_is_directory=True)


def lake_build(*targets: str, timeout: int = 600) -> subprocess.CompletedProcess:
    """`lake build *targets`, its output captured."""
    prepare()
    return subprocess.run(["lake", "build", *targets], cwd=LAKE_PROJECT,
                          capture_output=True, text=True, timeout=timeout)


def lean_outputs(paths: "list[Path]") -> list[str]:
    """What `lake env lean` prints for each of `paths`, in order: each file
    compiled on its own, in parallel, against what the project has built."""
    prepare()

    def lean(path: Path) -> str:
        return subprocess.run(["lake", "env", "lean", str(path)], cwd=LAKE_PROJECT,
                              capture_output=True, text=True, timeout=1800).stdout

    with ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool:
        return list(pool.map(lean, paths))
