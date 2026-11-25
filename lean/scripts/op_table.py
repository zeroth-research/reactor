"""The op table (`reactor_lean.lowering.ops`) as text: one line per op of
each theory, a word per backend column, then every gap, grouped by the
reason the table records. Run from `lean/`:

    uv run python scripts/op_table.py
"""

from reactor_lean.lowering.ops import OPS, Inline, Unsupported, ViaMat, _SEE_MAT
from reactor_lean.lowering.sorts import THEORIES

COLUMNS = ("mat", "scalar", "box")


def mark(cell) -> str:
    """One word for a cell."""
    if isinstance(cell, Unsupported):
        return "-"
    if isinstance(cell, Inline):
        return "inline"
    if isinstance(cell, ViaMat):
        return "via mat"
    return "yes"


def rows():
    """(theory, row) for every op, theory-major then alphabetical; the
    structural ops, of no theory, last."""
    order = {t: i for i, t in enumerate(THEORIES)}
    order[""] = len(order)
    pairs = [(theory, op) for op in OPS for theory in op.theories]
    return sorted(pairs, key=lambda pair: (order[pair[0]], pair[1].name))


def table() -> str:
    head = f"{'theory':13} {'op':14} " + " ".join(f"{c:8}" for c in COLUMNS)
    lines = [head, "-" * len(head)]
    gaps: dict[str, list[str]] = {}
    for theory, op in rows():
        marks = " ".join(f"{mark(getattr(op, c)):8}" for c in COLUMNS)
        lines.append(f"{theory or '(struct)':13} {op.name:14} {marks}")
        for column in COLUMNS:
            cell = getattr(op, column)
            # `_SEE_MAT` defers to the matrix cell, which says why.
            if isinstance(cell, Unsupported) and cell is not _SEE_MAT:
                gaps.setdefault(cell.reason, []).append(
                    f"{theory or '(struct)'}.{op.name} [{column}]")
    lines.append("\ngaps:")
    for reason, sites in gaps.items():
        lines.append(f"  * {', '.join(sites)}")
        lines.append(f"      {reason}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(table())
