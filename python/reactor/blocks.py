"""The blocks of atoms, over the views of their variables.

An atom binds its variables to the boundaries of its blocks by position, in the
order of their names:

* ``init`` maps ``X(wait)`` to ``X(ctrl)``;
* ``update`` maps ``read`` followed by ``X(wait)`` to ``X(ctrl)``;
* ``delay`` maps ``read`` followed by ``d(wait)`` to ``d(ctrl)``.

A variable stands for its latched value, ``X(v)`` for its next value and ``d(v)``
for its derivative. The helpers here state these boundaries, give each of their
views a fresh wire, and build the blocks the atoms take::

    inputs, outputs = boundary("update", ctrl=[x], read=[x])
    update = block(inputs, outputs, [Term(op, [outputs[X(x)]], [inputs[x]])])
    atom = Atom([x], [], [x], havoc([x]), update, zero([x], read=[x]))
"""

from .reactor import Block, Term, Wire, X, d, Combinatorial, Sequential, Differential

INIT = "init"
UPDATE = "update"
DELAY = "delay"
KINDS = (INIT, UPDATE, DELAY)


def _sorted(vars) -> list:
    """The variables in the order of their names, each once."""
    return sorted(set(vars))


def views(kind, ctrl=(), wait=(), read=()) -> tuple[list, list]:
    """The views the block of the ``kind`` of an atom takes and gives, in order."""
    ctrl, wait, read = _sorted(ctrl), _sorted(wait), _sorted(read)
    match kind:
        case "init":
            return [X(v) for v in wait], [X(v) for v in ctrl]
        case "update":
            return read + [X(v) for v in wait], [X(v) for v in ctrl]
        case "delay":
            return read + [d(v) for v in wait], [d(v) for v in ctrl]
    raise ValueError(f"no block of the kind {kind!r}, expected one of {KINDS}")


def boundary(kind, ctrl=(), wait=(), read=()) -> tuple[dict, dict]:
    """The views the block of the ``kind`` of an atom takes and gives, each with
    a fresh wire of its sort, in order."""
    inputs, outputs = views(kind, ctrl, wait, read)
    return {v: Wire(v.dtype) for v in inputs}, {v: Wire(v.dtype) for v in outputs}


def discard(wire) -> Term:
    """The term consuming the wire, and giving nothing."""
    return Term(Combinatorial.DISCARD(wire.dtype), [], [wire])


def _wires(wires) -> list:
    return list(wires.values()) if isinstance(wires, dict) else list(wires)


def block(read, write, terms) -> Block:
    """The block of the terms with the inputs ``read`` and the outputs ``write``
    (wires, or views mapped to wires, in order), discarding the inputs no term
    reads."""
    read, write, terms = _wires(read), _wires(write), list(terms)
    used = {w for term in terms for w in term.read}
    terms += [discard(w) for w in read if w not in used]
    return Block(read, write, terms)


def build(kind, terms, ctrl=(), wait=(), read=()) -> Block:
    """The block of the ``kind`` of an atom, of the terms ``terms`` gives over the
    wires of the views of its boundary (a mapping from the views to the wires),
    discarding the inputs no term reads::

        update = build("update", lambda w: [Term(op, [w[X(x)]], [w[x]])], ctrl=[x], read=[x])
    """
    inputs, outputs = boundary(kind, ctrl, wait, read)
    return block(inputs, outputs, terms({**inputs, **outputs}))


def havoc(ctrl, wait=()) -> Block:
    """The init leaving the next values of the controlled variables arbitrary."""
    inputs, outputs = boundary(INIT, ctrl, wait)
    terms = [Term(Combinatorial.HAVOC(w.dtype), [w]) for w in outputs.values()]
    return block(inputs, outputs, terms)


def skip(ctrl, wait=(), read=()) -> Block:
    """The update keeping the values of the controlled variables, all of which
    it reads."""
    inputs, outputs = boundary(UPDATE, ctrl, wait, read)
    missing = [v for v in _sorted(ctrl) if v not in inputs]
    if missing:
        raise ValueError(f"skip keeps the variables it reads, but does not read {missing}")
    terms = [Term(Sequential.SKIP(v.dtype), [w], [inputs[v.var]]) for v, w in outputs.items()]
    return block(inputs, outputs, terms)


def zero(ctrl, wait=(), read=()) -> Block:
    """The delay keeping the controlled variables constant: their derivatives
    are zero."""
    inputs, outputs = boundary(DELAY, ctrl, wait, read)
    terms = [Term(Differential.ZERO(w.dtype), [w]) for w in outputs.values()]
    return block(inputs, outputs, terms)
