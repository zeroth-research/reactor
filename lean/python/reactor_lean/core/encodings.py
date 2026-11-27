"""The module's Lean encodings by name: what `--encodings` selects, and
what each row of `project.scaffold.ENCODINGS` is called. A leaf, so the
command line can name them without importing the generators;
`tests/test_encodings_select.py` holds the names and the rows together.
"""

from __future__ import annotations

# The encoding every other one, and the certificate, is stated over: always
# written, whatever is selected.
FUNCTIONAL = "functional"

# The module as a `Relational.Module`, its rounds proved to be the
# functional encoding's.
MODULE = "module"

# In the table's order.
ENCODING_NAMES: tuple[str, ...] = (
    FUNCTIONAL, MODULE, "relational", "scalar", "scalar-relational",
)
