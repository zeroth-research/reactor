"""`lake` in the tests' Lake project (`testdata.LAKE_PROJECT`)."""

from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from testdata import LAKE_PROJECT


def lake_build(*targets: str, timeout: int = 600) -> subprocess.CompletedProcess:
    """`lake build *targets`, its output captured."""
    return subprocess.run(["lake", "build", *targets], cwd=LAKE_PROJECT,
                          capture_output=True, text=True, timeout=timeout)


def lean_outputs(paths: "list[Path]") -> list[str]:
    """What `lake env lean` prints for each of `paths`, in order: each file
    compiled on its own, in parallel, against what the project has built."""

    def lean(path: Path) -> str:
        return subprocess.run(["lake", "env", "lean", str(path)], cwd=LAKE_PROJECT,
                              capture_output=True, text=True, timeout=1800).stdout

    with ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool:
        return list(pool.map(lean, paths))
