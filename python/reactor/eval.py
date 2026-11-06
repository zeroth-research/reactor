"""Torch interpreter for theory_pyo3 terms.

Dispatch is by `match` on the op (per-theory classes), not op-name strings.
LIA/LRA ops with identical torch semantics share a case via OR-patterns; BV
diverges (mod-2^width wraparound, bitwise ops, 1-bit-BV comparisons), so it
needs the output Sort for its width — threaded in as `out_sort`.
"""

import torch
from .reactor import LRA, LIA, BV, Sequential
from .sort import BitVec
from .blocks import views


def _bv_width(sort):
    match sort:
        case BitVec(bw, _):
            return bw
    return None


def _wrap(t, sort):
    """Mask a BV result to its bit-width (arithmetic modulo 2^width)."""
    bw = _bv_width(sort)
    if bw is None or bw >= 64:
        return t
    return t & ((1 << bw) - 1)


def _linear(weight, bias, x):
    """A·x (+ b). An empty bias tensor means no bias (matmul-as-Linear)."""
    if bias is None or bias.numel() == 0:
        return weight @ x
    return weight @ x + bias


def eval_itype(itype, read, out_sort=None):
    """Evaluate a single op with the given input tensors.

    `out_sort` is the write wire's Sort; needed for BV width-masking.
    """
    r = read
    match itype:
        # --- identity / control flow ---
        case (LRA.IdReal() | LRA.IdBool() | LRA.TId() | LIA.IdInt() | LIA.IdBool() | BV.Id()
              | Sequential.SKIP()):
            return [r[0].clone()]
        case (LRA.IteReal() | LRA.IteBool() | LRA.TIte() | LIA.IteInt() | LIA.IteBool()
              | BV.Ite()):
            return [torch.where(r[0].bool(), r[1], r[2])]

        # --- constants (payload bound from the op) ---
        case LRA.Bool(t) | LRA.Real(t) | LIA.Bool(t) | LIA.Int(t) | BV.Const(t):
            return [t.clone()]

        # --- arithmetic: LIA/LRA (exact, no wraparound) ---
        case LRA.Add() | LRA.TAdd() | LIA.Add():
            return [r[0] + r[1]]
        case LRA.Sub() | LRA.TSub() | LIA.Sub():
            return [r[0] - r[1]]

        # --- arithmetic: BV (modulo 2^width) ---
        case BV.Add():
            return [_wrap(r[0] + r[1], out_sort)]
        case BV.Sub():
            return [_wrap(r[0] - r[1], out_sort)]
        case BV.Mul():
            return [_wrap(r[0] * r[1], out_sort)]
        case BV.Neg():
            return [_wrap(-r[0], out_sort)]
        case BV.Abs():
            return [r[0].abs()]
        case BV.UDiv():
            return [r[0].div(r[1], rounding_mode="floor")]
        case BV.SDiv():
            return [r[0].div(r[1], rounding_mode="trunc")]
        case BV.UMod():
            return [r[0].remainder(r[1])]
        case BV.SMod():
            return [r[0].fmod(r[1])]
        case BV.MatMul():
            return [_wrap(r[0] @ r[1], out_sort)]

        # --- Linear (LIA/LRA): A, B constants carried in the op ---
        case LRA.Linear(weight, bias) | LRA.TLinear(weight, bias) | LIA.Linear(weight, bias):
            return [_linear(weight, bias, r[0])]

        # --- comparisons: LIA/LRA produce a boolean tensor ---
        case LRA.Eq() | LIA.Eq():
            return [r[0].eq(r[1])]
        case LRA.Ne() | LIA.Ne():
            return [r[0].ne(r[1])]
        case LRA.Lt() | LIA.Lt():
            return [r[0].lt(r[1])]
        case LRA.Le() | LIA.Le():
            return [r[0].le(r[1])]
        case LRA.Gt() | LIA.Gt():
            return [r[0].gt(r[1])]
        case LRA.Ge() | LIA.Ge():
            return [r[0].ge(r[1])]

        # --- comparisons: BV produce a 1-bit BV (0/1 int) ---
        # TODO: distinguish signed vs unsigned once BV values carry signedness;
        # for non-negative storage signed == unsigned.
        case BV.ULt() | BV.SLt():
            return [(r[0] < r[1]).to(r[0].dtype)]
        case BV.ULe() | BV.SLe():
            return [(r[0] <= r[1]).to(r[0].dtype)]
        case BV.UGt() | BV.SGt():
            return [(r[0] > r[1]).to(r[0].dtype)]
        case BV.UGe() | BV.SGe():
            return [(r[0] >= r[1]).to(r[0].dtype)]
        case BV.Eq():
            return [(r[0] == r[1]).to(r[0].dtype)]
        case BV.Ne():
            return [(r[0] != r[1]).to(r[0].dtype)]

        # --- logical: LIA/LRA boolean ops ---
        case LRA.And() | LIA.And():
            return [r[0].logical_and(r[1])]
        case LRA.Or() | LIA.Or():
            return [r[0].logical_or(r[1])]
        case LRA.Not() | LIA.Not():
            return [r[0].logical_not()]
        case LRA.Xor() | LIA.Xor():
            return [r[0].logical_xor(r[1])]

        # --- bitwise: BV (width-preserving) ---
        case BV.And():
            return [r[0] & r[1]]
        case BV.Or():
            return [r[0] | r[1]]
        case BV.Xor():
            return [r[0] ^ r[1]]
        case BV.Not():
            return [_wrap(~r[0], out_sort)]
        case BV.BVToBool():
            return [(r[0] != 0).to(r[0].dtype)]
        case BV.BitSelect(high=high, low=low):
            return [(r[0] >> low) & ((1 << (high - low + 1)) - 1)]
        case BV.Extend(extra=_):
            return [r[0]]  # zero-extend: value unchanged, width grows

        # --- neural / aggregate (LIA/LRA) ---
        case LRA.ReLU() | LIA.ReLU():
            return [r[0].relu()]
        case LRA.Argmax() | LIA.Argmax():
            return [r[0].argmax()]
        case LRA.Min() | LIA.Min():
            return [torch.minimum(r[0], r[1])]
        case LRA.Max() | LIA.Max():
            return [torch.maximum(r[0], r[1])]

        # --- matrix ---
        case LRA.TransposeReal() | LRA.TransposeBool() | LIA.TransposeInt() | LIA.TransposeBool():
            return [r[0].transpose(0, 1)]

        # --- uninterpreted ---
        case (LRA.UninterpretedReal(name) | LRA.UninterpretedBool(name)
              | LIA.UninterpretedInt(name) | LIA.UninterpretedBool(name) | BV.Uninterpreted(name)):
            raise RuntimeError(f"cannot evaluate uninterpreted function '{name}'")

    raise RuntimeError(f"cannot evaluate op: {itype}")


# ============================================================================
# Interpreter helpers (shared by reactor.gym.Wrapper, reactor.gym.Env)
# ============================================================================


def _eval_term(term, read):
    """The values the term writes, from the values it reads, as torch tensors."""
    return eval_itype(term.itype, read, term.write[0].dtype)


def execute_block(state, atom, kind, evaluate=_eval_term):
    """Evaluate the block of the ``kind`` of the atom.

    ``state`` maps the views of the variables (a variable for its latched value,
    ``X(v)`` for its next value, ``d(v)`` for its derivative) to their values. The
    block takes the values of the views of its boundary inputs, by position, and
    gives those of its outputs, which are stored back into ``state``.

    ``evaluate(term, read)`` gives the values a term writes from the values it reads:
    torch tensors by default, or e.g. Z3 expressions with ``reactor.z3.eval``::

        execute_block(state, atom, "update", lambda t, read: z3.eval(t.itype, read))
    """
    block = getattr(atom, kind)
    inputs, outputs = views(kind, atom.ctrl, atom.wait, atom.read)
    wires = {w: view for w, view in zip(block.read, inputs)}
    values = {w: state[view] for w, view in wires.items() if view in state}

    def value(w):
        if w not in values:
            raise KeyError(f"no value for {wires.get(w, w)} in the {kind} of {atom}")
        return values[w]

    for term in block:
        # a term that writes nothing (a discard of an unused input) has no effect
        if not len(term.write):
            continue
        read = [value(w) for w in term.read]
        results = evaluate(term, read)
        for w, val in zip(term.write, results):
            values[w] = val

    for w, view in zip(block.write, outputs):
        state[view] = values[w]


def _execute_block(state, atoms, kind):
    """Evaluate the block of the ``kind`` of each atom, in order."""
    for atom in atoms:
        execute_block(state, atom, kind)


def execute_init(state, atoms):
    """Evaluate the init block of all atoms."""
    _execute_block(state, atoms, "init")


def execute_update(state, atoms):
    """Evaluate the update block of all atoms."""
    _execute_block(state, atoms, "update")


def read_wire(state, wire):
    """Read a wire value from state."""
    return state[wire].detach().clone()


def getattr_wire(self, name):
    """__getattr__ helper for named wire access."""
    wire_names = object.__getattribute__(self, "_wire_names")
    if name in wire_names:
        state = object.__getattribute__(self, "_state")
        wire = wire_names[name]  # read from latched wire
        if wire in state:
            val = state[wire]
            return val.item() if val.numel() == 1 else val.detach().clone()
    raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
