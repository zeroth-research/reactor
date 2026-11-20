# torch must load before the reactor C-extension (libtorch rpath); importing it
# here ensures the right order regardless of a test module's own import order.
import torch  # noqa: F401

import os
from pathlib import Path

# The projects the tests generate require ReactLib's packages by path
# from the checkout beside this repository, when there is one, so a test
# that builds one builds it against the Lean libraries as they are on disk.
# (As `lean/tests/conftest.py` does for `reactor_lean`'s tests.)
_REACTLIB = Path(__file__).resolve().parents[3] / "ReactLib"
if "REACTLIB" not in os.environ and _REACTLIB.is_dir():
    os.environ["REACTLIB"] = str(_REACTLIB)
