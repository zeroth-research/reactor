"""Torch interpreter for terms.

Dispatch is by `match` on the operation's class, binding positionally its parameters,
then its domain and codomain, a single sort bare and several as a list
(`case word.Extract(i, j, Word(w)):`). Values are tensors: booleans as `torch.bool`,
reals as floating point, integers and words as `int64`. A word of width `w` is stored
as its unsigned value in `[0, 2^w)`; the signed operations read it in two's complement,
and every word result is reduced modulo `2^w`. Division by zero follows SMT-LIB2.
"""

import torch

from .matrix import Havoc, Skip
from .reactor import Zero
from .matrix.sort import Real, Int, Bool, Word

from .reactor import matrix


def _wrap(t, w):
    """`t` modulo `2^w`: the word of width `w` it stands for."""
    return t & ((1 << w) - 1) if w < 64 else t


def _signed(t, w):
    """The words `t` of width `w` read in two's complement."""
    return torch.where(t >= (1 << (w - 1)), t - (1 << w), t) if w < 64 else t


def _zeros(sort):
    match sort:
        case Bool(s):
            return torch.zeros(s, dtype=torch.bool)
        case Real(s):
            return torch.zeros(s, dtype=torch.float32)
        case Int(s) | Word(_, s):
            return torch.zeros(s, dtype=torch.int64)
    raise RuntimeError(f"no value of sort {sort}")


def _euclid(x, y):
    """Euclidean division of integers, as in SMT-LIB2 Ints: the remainder `r` is
    `0 <= r < |y|`, and the quotient `q` is `(x - r) / y`. Division by zero is
    unspecified there; here it gives `q = 0` and `r = x`."""
    nz = torch.where(y == 0, torch.ones_like(y), y)
    r = x.remainder(nz.abs())
    q = (x - r).div(nz, rounding_mode="trunc")
    return torch.where(y == 0, torch.zeros_like(q), q), torch.where(y == 0, x, r)


def _udiv(x, y, w):
    q = torch.where(y == 0, torch.ones_like(x), x).div(torch.where(y == 0, torch.ones_like(y), y), rounding_mode="floor")
    return torch.where(y == 0, torch.full_like(x, (1 << w) - 1), q)


def _urem(x, y):
    return torch.where(y == 0, x, x.remainder(torch.where(y == 0, torch.ones_like(y), y)))


def _sdiv(x, y, w):
    a, b = _signed(x, w), _signed(y, w)
    q = a.div(torch.where(b == 0, torch.ones_like(b), b), rounding_mode="trunc")
    q = torch.where(b == 0, torch.where(a < 0, torch.ones_like(a), -torch.ones_like(a)), q)
    return _wrap(q, w)


def _srem(x, y, w):
    a, b = _signed(x, w), _signed(y, w)
    r = a.fmod(torch.where(b == 0, torch.ones_like(b), b))
    return _wrap(torch.where(b == 0, a, r), w)


def _smod(x, y, w):
    a, b = _signed(x, w), _signed(y, w)
    r = a.remainder(torch.where(b == 0, torch.ones_like(b), b))
    return _wrap(torch.where(b == 0, a, r), w)


def _shl(x, y, w):
    return torch.where(y >= w, torch.zeros_like(x), _wrap(x << y.clamp(max=w), w))


def _lshr(x, y, w):
    return torch.where(y >= w, torch.zeros_like(x), x >> y.clamp(max=w))


def _ashr(x, y, w):
    return _wrap(_signed(x, w) >> y.clamp(max=w - 1), w)


def _rotl(x, i, w):
    i %= w
    return _wrap((x << i) | (x >> (w - i)), w) if i else x


def _repeat(x, i, w):
    out = x
    for _ in range(i - 1):
        out = (out << w) | x
    return out


def eval_itype(itype, read, out_sort=None):
    """Evaluate a single op with the given input tensors.

    `out_sort` is the write wire's sort, needed by the structural operations that
    write a value of it from nothing (`Havoc`).
    """
    r = read
    match itype:
        # --- structural ---
        case Skip():
            return [r[0].clone()]
        case Havoc(sort):
            # an arbitrary value: zeros
            return [_zeros(sort)]
        case Zero():
            # the singleton sort is the rate of change of discrete values, which are
            # never integrated: nothing evaluates it
            raise RuntimeError("Zero writes the singleton sort, which is never evaluated")
        case matrix.real.Zero(Real(shape)):
            # the zero matrix, of any degree: the zero rate of change
            return [torch.zeros(shape, dtype=torch.float32)]

        # --- constants, linear and affine maps (tensors carried by the op) ---
        case matrix.real.Num(t):
            # a real constant may hold integers, kept as given; its value is a real, of
            # floating point, so that it neither wraps around nor mixes with integers
            return [t.clone() if t.is_floating_point() else t.to(torch.get_default_dtype())]
        case matrix.int.Num(t) | matrix.bool.Truth(t) | matrix.word.Num(t):
            return [t.clone()]
        case matrix.real.Linear(a) | matrix.int.Linear(a):
            return [a @ r[0]]
        case matrix.real.Affine(a, b) | matrix.int.Affine(a, b):
            return [a @ r[0] + b]
        case matrix.real.Scale(c) | matrix.int.Scale(c):
            return [c * r[0]]

        # --- choice ---
        case (
            matrix.real.Ite()
            | matrix.int.Ite()
            | matrix.bool.Ite()
            | matrix.word.Ite()
            | matrix.real.broadcast.Ite()
            | matrix.int.broadcast.Ite()
            | matrix.bool.broadcast.Ite()
            | matrix.word.broadcast.Ite()
        ):
            # pointwise, or by a scalar condition, broadcast to the whole matrix
            return [torch.where(r[0], r[1], r[2])]

        # --- real and integer arithmetic (exact) ---
        case matrix.real.Add() | matrix.int.Add():
            return [r[0] + r[1]]
        case matrix.real.Sub() | matrix.int.Sub():
            return [r[0] - r[1]]
        case matrix.real.Mul() | matrix.int.Mul():
            return [r[0] * r[1]]
        case matrix.real.Div():
            return [r[0] / r[1]]
        case matrix.int.Div():
            return [_euclid(r[0], r[1])[0]]
        case matrix.int.Mod():
            return [_euclid(r[0], r[1])[1]]
        case matrix.real.Neg() | matrix.int.Neg():
            return [-r[0]]
        case matrix.real.Abs() | matrix.int.Abs():
            return [r[0].abs()]
        case matrix.real.ReLU() | matrix.int.ReLU():
            return [r[0].relu()]
        case matrix.real.Min() | matrix.int.Min():
            return [torch.minimum(r[0], r[1])]
        case matrix.real.Max() | matrix.int.Max():
            return [torch.maximum(r[0], r[1])]
        case matrix.real.Argmax() | matrix.int.Argmax():
            return [r[0].argmax().reshape(1, 1).to(r[0].dtype)]
        case matrix.real.Transpose() | matrix.int.Transpose():
            return [r[0].transpose(0, 1)]
        case matrix.real.MatMul() | matrix.int.MatMul():
            return [r[0] @ r[1]]

        # --- comparisons: write booleans ---
        case matrix.real.Eq() | matrix.int.Eq() | matrix.bool.Eq() | matrix.word.Eq():
            return [r[0].eq(r[1])]
        case matrix.real.Ne() | matrix.int.Ne() | matrix.bool.Ne() | matrix.word.Ne():
            return [r[0].ne(r[1])]
        case matrix.real.Lt() | matrix.int.Lt() | matrix.word.unsigned.Lt():
            return [r[0].lt(r[1])]
        case matrix.real.Le() | matrix.int.Le() | matrix.word.unsigned.Le():
            return [r[0].le(r[1])]
        case matrix.real.Gt() | matrix.int.Gt() | matrix.word.unsigned.Gt():
            return [r[0].gt(r[1])]
        case matrix.real.Ge() | matrix.int.Ge() | matrix.word.unsigned.Ge():
            return [r[0].ge(r[1])]
        case matrix.word.signed.Lt([Word(w), Word(u)]):
            return [_signed(r[0], w).lt(_signed(r[1], u))]
        case matrix.word.signed.Le([Word(w), Word(u)]):
            return [_signed(r[0], w).le(_signed(r[1], w))]
        case matrix.word.signed.Gt([Word(w), Word(u)]):
            return [_signed(r[0], w).gt(_signed(r[1], w))]
        case matrix.word.signed.Ge([Word(w), Word(u)]):
            return [_signed(r[0], w).ge(_signed(r[1], w))]

        # --- propositional logic ---
        case matrix.bool.And():
            return [r[0].logical_and(r[1])]
        case matrix.bool.Or():
            return [r[0].logical_or(r[1])]
        case matrix.bool.Xor():
            return [r[0].logical_xor(r[1])]
        case matrix.bool.Not():
            return [r[0].logical_not()]

        # --- words: arithmetic modulo 2^w ---
        case matrix.word.Add(_, Word(w)):
            return [_wrap(r[0] + r[1], w)]
        case matrix.word.Sub(_, Word(w)):
            return [_wrap(r[0] - r[1], w)]
        case matrix.word.Mul(_, Word(w)):
            return [_wrap(r[0] * r[1], w)]
        case matrix.word.Neg(_, Word(w)):
            return [_wrap(-r[0], w)]
        case matrix.word.signed.Abs(_, Word(w)):
            return [_wrap(_signed(r[0], w).abs(), w)]
        case matrix.word.unsigned.Div(_, Word(w)):
            return [_udiv(r[0], r[1], w)]
        case matrix.word.unsigned.Rem():
            return [_urem(r[0], r[1])]
        case matrix.word.signed.Div(_, Word(w)):
            return [_sdiv(r[0], r[1], w)]
        case matrix.word.signed.Rem(_, Word(w)):
            return [_srem(r[0], r[1], w)]
        case matrix.word.signed.Mod(_, Word(w)):
            return [_smod(r[0], r[1], w)]

        # --- words: bitwise ---
        case matrix.word.And():
            return [r[0] & r[1]]
        case matrix.word.Or():
            return [r[0] | r[1]]
        case matrix.word.Xor():
            return [r[0] ^ r[1]]
        case matrix.word.Not(_, Word(w)):
            return [_wrap(~r[0], w)]
        case matrix.word.Nand(_, Word(w)):
            return [_wrap(~(r[0] & r[1]), w)]
        case matrix.word.Nor(_, Word(w)):
            return [_wrap(~(r[0] | r[1]), w)]
        case matrix.word.Xnor(_, Word(w)):
            return [_wrap(~(r[0] ^ r[1]), w)]

        # --- words: shifts, rotations and resizing ---
        case matrix.word.Shl(_, Word(w)):
            return [_shl(r[0], r[1], w)]
        case matrix.word.unsigned.Shr(_, Word(w)):
            return [_lshr(r[0], r[1], w)]
        case matrix.word.signed.Shr(_, Word(w)):
            return [_ashr(r[0], r[1], w)]
        case matrix.word.RotateLeft(i, _, Word(w)):
            return [_rotl(r[0], i, w)]
        case matrix.word.RotateRight(i, _, Word(w)):
            return [_rotl(r[0], w - i % w, w)]
        case matrix.word.Concat([Word(u), Word(v)]):
            return [(r[0] << v) | r[1]]
        case matrix.word.Extract(i, j, _, Word(v)):
            return [(r[0] >> j) & ((1 << v) - 1)]
        case matrix.word.Repeat(i, Word(w)):
            return [_repeat(r[0], i, w)]
        case matrix.word.unsigned.Extend():
            return [r[0].clone()]
        case matrix.word.signed.Extend(i, Word(w), Word(v)):
            return [_wrap(_signed(r[0], w), v)]

    raise RuntimeError(f"cannot evaluate op: {itype}")


# ============================================================================
# Interpreter helpers (shared by reactor.gym.Wrapper, reactor.gym.Env)
# ============================================================================


def _execute_block(state, atoms, get_block):
    """Evaluate a block from each atom."""
    for atom in atoms:
        for term in get_block(atom):
            read = [state[w] for w in term.read]
            out_sort = term.write[0].dtype if len(term.write) else None
            results = eval_itype(term.itype, read, out_sort)
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
