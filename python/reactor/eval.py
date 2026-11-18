"""Torch interpreter for the operations of `reactor.op`.

Dispatch is by `match` on the op's class, one per operation of each theory. Ops of
different theories with the same torch semantics share a case via OR-patterns; word
ops carry their width `w`, which masks their results to `w` bits (arithmetic modulo
`2^w`), and words are stored as their unsigned value, which the signed ops read in
two's complement.
"""

import torch
from . import op


def _wrap(t, w):
    """Mask a word result to its width (arithmetic modulo 2^w)."""
    if w >= 64:
        return t
    return t & ((1 << w) - 1)


def _signed(t, w):
    """The unsigned `w`-bit words `t` read in two's complement."""
    return t - (((t >> (w - 1)) & 1) << w)


def _linear(weight, bias, x):
    """A·x + B."""
    return weight @ x + bias


def _euclidean_div(x, y):
    """The quotient `q` of `x = q·y + r` with `0 ≤ r < |y|`, as SMT-LIB2's `div`."""
    q = torch.div(x, y, rounding_mode="floor")
    return torch.where((y < 0) & (x - q * y != 0), q + 1, q)


def _rotate_left(t, i, w):
    i %= w
    return _wrap((t << i) | (t >> (w - i)), w) if i else t


def eval_itype(itype, read):
    """Evaluate a single op with the given input tensors."""
    r = read
    match itype:
        # --- identity / control flow ---
        case op.real.Id() | op.real.TId() | op.int.Id() | op.bool.Id() | op.word.Id():
            return [r[0].clone()]
        case op.real.Ite() | op.real.TIte() | op.int.Ite() | op.bool.Ite() | op.word.Ite():
            return [torch.where(r[0].bool(), r[1], r[2])]

        # --- constants (payload bound from the op) ---
        case op.real.Const(t) | op.int.Const(t) | op.bool.Const(t) | op.word.Const(t, _):
            return [t.clone()]
        case op.real.Zero(m, n, _):
            return [torch.zeros(m, n)]
        case op.bool.Zero():
            return [torch.zeros(())]

        # --- arithmetic: reals and integers (exact, no wraparound) ---
        case op.real.Add() | op.real.TAdd() | op.int.Add():
            return [r[0] + r[1]]
        case op.real.Sub() | op.real.TSub() | op.int.Sub():
            return [r[0] - r[1]]
        case op.real.Mul() | op.int.Mul():
            return [r[0] * r[1]]
        case op.real.Div():
            return [r[0] / r[1]]
        case op.int.Div():
            return [_euclidean_div(r[0], r[1])]
        case op.int.Mod():
            return [r[0] - _euclidean_div(r[0], r[1]) * r[1]]
        case op.real.MatMul() | op.int.MatMul():
            return [r[0] @ r[1]]

        # --- Linear: A, B constants carried in the op ---
        case op.real.Linear(weight, bias) | op.real.TLinear(weight, bias, _) | op.int.Linear(weight, bias):
            return [_linear(weight, bias, r[0])]

        # --- arithmetic: words (modulo 2^w) ---
        case op.word.Add(w, _, _):
            return [_wrap(r[0] + r[1], w)]
        case op.word.Sub(w, _, _):
            return [_wrap(r[0] - r[1], w)]
        case op.word.Mul(w, _, _):
            return [_wrap(r[0] * r[1], w)]
        case op.word.Neg(w, _, _):
            return [_wrap(-r[0], w)]
        case op.word.UDiv():
            return [r[0].div(r[1], rounding_mode="floor")]
        case op.word.URem():
            return [r[0].remainder(r[1])]
        case op.word.SDiv(w, _, _):
            return [_wrap(_signed(r[0], w).div(_signed(r[1], w), rounding_mode="trunc"), w)]
        case op.word.SRem(w, _, _):
            return [_wrap(_signed(r[0], w).fmod(_signed(r[1], w)), w)]
        case op.word.SMod(w, _, _):
            return [_wrap(_signed(r[0], w).remainder(_signed(r[1], w)), w)]

        # --- comparisons, which write booleans ---
        case op.real.Eq() | op.int.Eq() | op.word.Eq():
            return [r[0].eq(r[1])]
        case op.real.Ne() | op.int.Ne() | op.word.Ne():
            return [r[0].ne(r[1])]
        case op.real.Lt() | op.int.Lt() | op.word.ULt():
            return [r[0].lt(r[1])]
        case op.real.Le() | op.int.Le() | op.word.ULe():
            return [r[0].le(r[1])]
        case op.real.Gt() | op.int.Gt() | op.word.UGt():
            return [r[0].gt(r[1])]
        case op.real.Ge() | op.int.Ge() | op.word.UGe():
            return [r[0].ge(r[1])]
        case op.word.SLt(w, _, _):
            return [_signed(r[0], w).lt(_signed(r[1], w))]
        case op.word.SLe(w, _, _):
            return [_signed(r[0], w).le(_signed(r[1], w))]
        case op.word.SGt(w, _, _):
            return [_signed(r[0], w).gt(_signed(r[1], w))]
        case op.word.SGe(w, _, _):
            return [_signed(r[0], w).ge(_signed(r[1], w))]

        # --- logical: booleans ---
        case op.bool.And():
            return [r[0].logical_and(r[1])]
        case op.bool.Or():
            return [r[0].logical_or(r[1])]
        case op.bool.Not():
            return [r[0].logical_not()]
        case op.bool.Xor():
            return [r[0].logical_xor(r[1])]

        # --- bitwise: words (width-preserving) ---
        case op.word.And():
            return [r[0] & r[1]]
        case op.word.Or():
            return [r[0] | r[1]]
        case op.word.Xor():
            return [r[0] ^ r[1]]
        case op.word.Not(w, _, _):
            return [_wrap(~r[0], w)]
        case op.word.Nand(w, _, _):
            return [_wrap(~(r[0] & r[1]), w)]
        case op.word.Nor(w, _, _):
            return [_wrap(~(r[0] | r[1]), w)]
        case op.word.Xnor(w, _, _):
            return [_wrap(~(r[0] ^ r[1]), w)]
        case op.word.Shl(w, _, _):
            return [torch.where(r[1] < w, _wrap(r[0] << r[1].clamp(max=w), w), 0)]
        case op.word.LShr(w, _, _):
            return [torch.where(r[1] < w, r[0] >> r[1].clamp(max=w), 0)]
        case op.word.AShr(w, _, _):
            return [_wrap(_signed(r[0], w) >> r[1].clamp(max=w), w)]
        case op.word.RotateLeft(i, w, _, _):
            return [_rotate_left(r[0], i, w)]
        case op.word.RotateRight(i, w, _, _):
            return [_rotate_left(r[0], w - i % w, w)]
        case op.word.Concat(_, v, _, _):
            return [(r[0] << v) | r[1]]
        case op.word.Extract(high, low, _, _, _):
            return [(r[0] >> low) & ((1 << (high - low + 1)) - 1)]
        case op.word.Repeat(i, w, _, _):
            out = torch.zeros_like(r[0])
            for _ in range(i):
                out = (out << w) | r[0]
            return [out]
        case op.word.ZeroExtend():
            return [r[0]]  # value unchanged, width grows
        case op.word.SignExtend(i, w, _, _):
            return [_wrap(_signed(r[0], w), w + i)]

        # --- neural / aggregate (reals and integers) ---
        case op.real.ReLU() | op.int.ReLU():
            return [r[0].relu()]
        case op.real.Argmax() | op.int.Argmax():
            return [r[0].argmax()]
        case op.real.Min() | op.int.Min():
            return [torch.minimum(r[0], r[1])]
        case op.real.Max() | op.int.Max():
            return [torch.maximum(r[0], r[1])]

        # --- matrix ---
        case op.real.Transpose() | op.real.TTranspose() | op.int.Transpose():
            return [r[0].transpose(0, 1)]

        # --- uninterpreted ---
        case (op.real.Uninterpreted(name) | op.int.Uninterpreted(name)
              | op.bool.Uninterpreted(name) | op.word.Uninterpreted(name)):
            raise RuntimeError(f"cannot evaluate uninterpreted function '{name}'")

    raise RuntimeError(f"cannot evaluate op: {itype!r}")


# ============================================================================
# Interpreter helpers (shared by reactor.gym.Wrapper, reactor.gym.Env)
# ============================================================================


def _execute_block(state, atoms, get_block):
    """Evaluate a block from each atom."""
    for atom in atoms:
        for term in get_block(atom):
            read = [state[w] for w in term.read]
            results = eval_itype(term.itype, read)
            for w, val in zip(term.write, results):
                state[w] = val


def execute_init(state, atoms):
    """Evaluate the init block of all atoms."""
    _execute_block(state, atoms, lambda a: a.init)


def execute_update(state, atoms):
    """Evaluate the update block of all atoms."""
    _execute_block(state, atoms, lambda a: a.update)


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
