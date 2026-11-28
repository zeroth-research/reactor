"""Torch interpreter for terms.

Dispatch is by `match` on the operation's class, binding positionally its parameters,
then its domain and codomain, each a tuple of sorts, `(x,)` for one
(`case word.Extract(i, j, [Word(w)]):`). Values are tensors: booleans as `torch.bool`,
reals as floating point, integers and words as `int64`. A word of width `w` is stored
as its unsigned value in `[0, 2^w)`; the signed operations read it in two's complement,
and every word result is reduced modulo `2^w`. Division by zero follows SMT-LIB2.
A value of an enumeration is the position of its symbolic constant among the values
of the enumeration, sorted, as an `int64` of the shape `[1, 1]`, and an event a
`torch.bool` of the shape `[1, 1]` that flips when it happens.

Disclaimer: the evaluator is not faithful to the semantics of the theories, which is
that of SMT-LIB2, and so it is potentially unsound: a run it computes may be no run of
the system, and a property that holds of every run it computes may fail of the system.
The gap comes from the machine numbers that stand for the values of the theories:

- reals are simulated as floating point numbers, whose operations round, overflow to
  infinities and produce NaNs, where the reals of SMT-LIB2 are exact and unbounded;
- integers are `int64`, which wrap around past 64 bits, where the integers of SMT-LIB2
  are unbounded;
- words are `int64` too, which fail to capture words longer than 64 bits: a word of
  width `w >= 64` is not reduced modulo `2^w`, and one of width 64 is stored in two's
  complement, so that its unsigned operations read it as signed.

Its results are exact only as long as no value leaves the range that these numbers
represent exactly. Use an SMT solver, such as through `reactor.smt`, for exact answers.
"""

import torch

from .reactor import Havoc, Skip
from .real import Real
from .int import Int
from .bool import Bool
from .word import Word
from .enum import Enum
from .event import Event
from . import real as _rl
from .real import broadcast as _rb
from . import int as _ii
from .int import broadcast as _ib
from . import bool as _bl
from .bool import broadcast as _bb
from .bool import reduce as _br
from . import word as _ww
from .word import broadcast as _wb
from .word import signed as _ws
from .word import unsigned as _wu
from . import zero as _zz
from . import enum as _en
from . import event as _ev

__all__ = ["eval_itype", "execute_init", "execute_update", "read_wire", "getattr_wire"]


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
        case Enum():
            # the first of its values
            return torch.zeros([1, 1], dtype=torch.int64)
        case Event():
            # an arbitrary state: its state alone means nothing
            return torch.zeros([1, 1], dtype=torch.bool)
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
        case Havoc([sort]):
            # an arbitrary value: zeros
            return [_zeros(sort)]
        case _zz.Zero():
            # the singleton sort is the rate of change of discrete values, which are
            # never integrated: nothing evaluates it
            raise RuntimeError("Zero writes the singleton sort, which is never evaluated")
        case _rl.Zero([Real(shape)]):
            # the zero matrix, of any degree: the zero rate of change
            return [torch.zeros(shape, dtype=torch.float32)]

        # --- constants, linear and affine maps (tensors carried by the op) ---
        case _rl.Num(t):
            # a real constant may hold integers, kept as given; its value is a real, of
            # floating point, so that it neither wraps around nor mixes with integers
            return [t.clone() if t.is_floating_point() else t.to(torch.get_default_dtype())]
        case _ii.Num(t) | _bl.Truth(t) | _ww.Num(t):
            return [t.clone()]
        case _rl.Linear(a) | _ii.Linear(a):
            return [a @ r[0]]
        case _rl.Affine(a, b) | _ii.Affine(a, b):
            return [a @ r[0] + b]
        case _rl.Scale(c) | _ii.Scale(c):
            return [c * r[0]]

        # --- choice ---
        case (
            _rl.Ite()
            | _ii.Ite()
            | _bl.Ite()
            | _ww.Ite()
            | _rb.Ite()
            | _ib.Ite()
            | _bb.Ite()
            | _wb.Ite()
        ):
            # pointwise, or by a scalar condition, broadcast to the whole matrix
            return [torch.where(r[0], r[1], r[2])]

        # --- real and integer arithmetic (exact) ---
        case _rl.Add() | _ii.Add():
            return [r[0] + r[1]]
        case _rl.Sub() | _ii.Sub():
            return [r[0] - r[1]]
        case _rl.Mul() | _ii.Mul():
            return [r[0] * r[1]]
        case _rl.Div():
            return [r[0] / r[1]]
        case _ii.Div():
            return [_euclid(r[0], r[1])[0]]
        case _ii.Mod():
            return [_euclid(r[0], r[1])[1]]
        case _rl.Neg() | _ii.Neg():
            return [-r[0]]
        case _rl.Abs() | _ii.Abs():
            return [r[0].abs()]
        case _rl.ReLU() | _ii.ReLU():
            return [r[0].relu()]
        case _rl.Min() | _ii.Min():
            return [torch.minimum(r[0], r[1])]
        case _rl.Max() | _ii.Max():
            return [torch.maximum(r[0], r[1])]
        case _rl.Argmax() | _ii.Argmax():
            return [r[0].argmax().reshape(1, 1).to(r[0].dtype)]
        case _rl.Transpose() | _ii.Transpose():
            return [r[0].transpose(0, 1)]
        case _rl.MatMul() | _ii.MatMul():
            return [r[0] @ r[1]]

        # --- comparisons: write booleans ---
        case _rl.Eq() | _ii.Eq() | _bl.Eq() | _ww.Eq():
            return [r[0].eq(r[1])]
        case _rl.Ne() | _ii.Ne() | _bl.Ne() | _ww.Ne():
            return [r[0].ne(r[1])]
        case _rl.Lt() | _ii.Lt() | _wu.Lt():
            return [r[0].lt(r[1])]
        case _rl.Le() | _ii.Le() | _wu.Le():
            return [r[0].le(r[1])]
        case _rl.Gt() | _ii.Gt() | _wu.Gt():
            return [r[0].gt(r[1])]
        case _rl.Ge() | _ii.Ge() | _wu.Ge():
            return [r[0].ge(r[1])]
        case _ws.Lt([Word(w), Word(u)]):
            return [_signed(r[0], w).lt(_signed(r[1], u))]
        case _ws.Le([Word(w), Word(u)]):
            return [_signed(r[0], w).le(_signed(r[1], w))]
        case _ws.Gt([Word(w), Word(u)]):
            return [_signed(r[0], w).gt(_signed(r[1], w))]
        case _ws.Ge([Word(w), Word(u)]):
            return [_signed(r[0], w).ge(_signed(r[1], w))]

        # --- propositional logic ---
        case _bl.And():
            return [r[0].logical_and(r[1])]
        case _bl.Or():
            return [r[0].logical_or(r[1])]
        case _bl.Xor():
            return [r[0].logical_xor(r[1])]
        case _bl.Not():
            return [r[0].logical_not()]
        case _br.And(axis):
            return [r[0].all(dim=axis, keepdim=True)]
        case _br.Or(axis):
            return [r[0].any(dim=axis, keepdim=True)]

        # --- words: arithmetic modulo 2^w ---
        case _ww.Add(_, [Word(w)]):
            return [_wrap(r[0] + r[1], w)]
        case _ww.Sub(_, [Word(w)]):
            return [_wrap(r[0] - r[1], w)]
        case _ww.Mul(_, [Word(w)]):
            return [_wrap(r[0] * r[1], w)]
        case _ww.Neg(_, [Word(w)]):
            return [_wrap(-r[0], w)]
        case _ws.Abs(_, [Word(w)]):
            return [_wrap(_signed(r[0], w).abs(), w)]
        case _wu.Div(_, [Word(w)]):
            return [_udiv(r[0], r[1], w)]
        case _wu.Rem():
            return [_urem(r[0], r[1])]
        case _ws.Div(_, [Word(w)]):
            return [_sdiv(r[0], r[1], w)]
        case _ws.Rem(_, [Word(w)]):
            return [_srem(r[0], r[1], w)]
        case _ws.Mod(_, [Word(w)]):
            return [_smod(r[0], r[1], w)]

        # --- words: bitwise ---
        case _ww.And():
            return [r[0] & r[1]]
        case _ww.Or():
            return [r[0] | r[1]]
        case _ww.Xor():
            return [r[0] ^ r[1]]
        case _ww.Not(_, [Word(w)]):
            return [_wrap(~r[0], w)]
        case _ww.Nand(_, [Word(w)]):
            return [_wrap(~(r[0] & r[1]), w)]
        case _ww.Nor(_, [Word(w)]):
            return [_wrap(~(r[0] | r[1]), w)]
        case _ww.Xnor(_, [Word(w)]):
            return [_wrap(~(r[0] ^ r[1]), w)]

        # --- words: shifts, rotations and resizing ---
        case _ww.Shl(_, [Word(w)]):
            return [_shl(r[0], r[1], w)]
        case _wu.Shr(_, [Word(w)]):
            return [_lshr(r[0], r[1], w)]
        case _ws.Shr(_, [Word(w)]):
            return [_ashr(r[0], r[1], w)]
        case _ww.RotateLeft(i, _, [Word(w)]):
            return [_rotl(r[0], i, w)]
        case _ww.RotateRight(i, _, [Word(w)]):
            return [_rotl(r[0], w - i % w, w)]
        case _ww.Concat([Word(u), Word(v)]):
            return [(r[0] << v) | r[1]]
        case _ww.Extract(i, j, _, [Word(v)]):
            return [(r[0] >> j) & ((1 << v) - 1)]
        case _ww.Repeat(i, [Word(w)]):
            return [_repeat(r[0], i, w)]
        case _wu.Extend():
            return [r[0].clone()]
        case _ws.Extend(i, [Word(w)], [Word(v)]):
            return [_wrap(_signed(r[0], w), v)]

        # --- enumerations: positions of the values, sorted ---
        case _en.Sym(value, [Enum(values)]):
            return [torch.tensor([[values.index(value)]], dtype=torch.int64)]
        case _en.Into([Enum(small)], [Enum(large)]):
            # each position in the smaller enumeration, to that of its value in the larger
            return [torch.tensor([large.index(v) for v in small], dtype=torch.int64)[r[0]]]
        case _en.Eq():
            return [r[0] == r[1]]
        case _en.Ne():
            return [r[0] != r[1]]
        case _en.Ite():
            return [torch.where(r[0], r[1], r[2])]

        # --- events: flipped when they happen; the shift of an event by a boolean, and the
        # difference of two values of it, are both the exclusive or of their booleans ---
        case _ev.Xor():
            return [r[0] ^ r[1]]

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
