"""Torch interpreter for terms.

Dispatch is by `match` on the operation's class, binding its sizes positionally
(`case word.Extract(i, j, w, _):`). The tensors that the operations hold may be torch
tensors, or JAX arrays (the operations of `reactor.jax`, and of `reactor.mixed`, which
holds either), taken as torch tensors. Values are tensors: booleans as `torch.bool`,
reals as floating point, integers and words as `int64`. A word of width `w` is stored
as its unsigned value in `[0, 2^w)`; the signed operations read it in two's complement,
and every word result is reduced modulo `2^w`. Division by zero follows SMT-LIB2.
"""

import torch

from .reactor.tensor import Havoc, Skip, Zero
from .arrays import as_torch
from .sort import Real, Int, Bool, Word, shape as _shape

from .reactor import tensor
from .reactor import jax as _jax
from .reactor import mixed as _mixed


def _wrap(t, w):
    """`t` modulo `2^w`: the word of width `w` it stands for."""
    return t & ((1 << w) - 1) if w < 64 else t


def _signed(t, w):
    """The words `t` of width `w` read in two's complement."""
    return torch.where(t >= (1 << (w - 1)), t - (1 << w), t) if w < 64 else t


def _held(t):
    """A tensor that an operation holds, as a value: a JAX array is taken as a torch
    tensor, and integers widen to `int64`, as JAX makes `int32` ones by default."""
    t = as_torch(t)
    if t.dtype is torch.bool or t.is_floating_point():
        return t
    return t.to(torch.int64)


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
    write a value of it from nothing (`Havoc`, `Zero`).
    """
    r = read
    match itype:
        # --- structural ---
        case Skip(_):
            return [r[0].clone()]
        case Havoc(sort) | Zero(sort):
            # an arbitrary value, and the zero rate of change: zeros
            return [_zeros(sort)]

        # --- constants and affine maps (tensors carried by the op) ---
        case (tensor.real.Const(t) | tensor.int.Const(t) | tensor.bool.Const(t) | tensor.word.Const(t, _)
              | _jax.real.Const(t) | _jax.int.Const(t) | _jax.bool.Const(t) | _jax.word.Const(t, _)
              | _mixed.real.Const(t) | _mixed.int.Const(t) | _mixed.bool.Const(t)
              | _mixed.word.Const(t, _)):
            return [_held(t).clone()]
        case (tensor.real.Linear(a, b) | tensor.int.Linear(a, b)
              | _jax.real.Linear(a, b) | _jax.int.Linear(a, b)
              | _mixed.real.Linear(a, b) | _mixed.int.Linear(a, b)):
            return [_held(a) @ r[0] + _held(b)]
        case (tensor.real.Scale(c) | tensor.int.Scale(c) | _jax.real.Scale(c) | _jax.int.Scale(c)
              | _mixed.real.Scale(c) | _mixed.int.Scale(c)):
            return [_held(c) * r[0]]

        # --- choice: of a whole matrix, by the one boolean of a [1, 1] condition, which
        # `where` broadcasts over the branches ---
        case tensor.real.Ite(_) | tensor.int.Ite(_) | tensor.bool.Ite(_) | tensor.word.Ite(_, _):
            return [torch.where(r[0], r[1], r[2])]

        # --- real and integer arithmetic (exact) ---
        case tensor.real.Add(_) | tensor.int.Add(_):
            return [r[0] + r[1]]
        case tensor.real.Sub(_) | tensor.int.Sub(_):
            return [r[0] - r[1]]
        case tensor.real.Mul(_) | tensor.int.Mul(_):
            return [r[0] * r[1]]
        case tensor.real.Div(_):
            return [r[0] / r[1]]
        case tensor.int.Div(_):
            return [_euclid(r[0], r[1])[0]]
        case tensor.int.Mod(_):
            return [_euclid(r[0], r[1])[1]]
        case tensor.real.Neg(_) | tensor.int.Neg(_):
            return [-r[0]]
        case tensor.real.Abs(_) | tensor.int.Abs(_):
            return [r[0].abs()]
        # transcendental functions of reals, as torch computes them: NaN or an
        # infinity where they are undefined, which the theory leaves unspecified
        case tensor.real.Sqrt(_):
            return [r[0].sqrt()]
        case tensor.real.Exp(_):
            return [r[0].exp()]
        case tensor.real.Log(_):
            return [r[0].log()]
        case tensor.real.Sin(_):
            return [r[0].sin()]
        case tensor.real.Cos(_):
            return [r[0].cos()]
        case tensor.real.Tan(_):
            return [r[0].tan()]
        case tensor.real.Asin(_):
            return [r[0].asin()]
        case tensor.real.Acos(_):
            return [r[0].acos()]
        case tensor.real.Atan(_):
            return [r[0].atan()]
        case tensor.real.Atan2(_):
            return [torch.atan2(r[0], r[1])]
        case tensor.real.Sinh(_):
            return [r[0].sinh()]
        case tensor.real.Cosh(_):
            return [r[0].cosh()]
        case tensor.real.Tanh(_):
            return [r[0].tanh()]
        case tensor.real.ReLU(_) | tensor.int.ReLU(_):
            return [r[0].relu()]
        # the least and the greatest entry, as a [1, 1] matrix
        case tensor.real.Min(_) | tensor.int.Min(_):
            return [r[0].min().reshape(1, 1)]
        case tensor.real.Max(_) | tensor.int.Max(_):
            return [r[0].max().reshape(1, 1)]
        case tensor.real.Argmax(_) | tensor.int.Argmax(_):
            return [r[0].argmax().reshape(1, 1).to(r[0].dtype)]
        case tensor.real.Transpose(_) | tensor.int.Transpose(_):
            return [r[0].transpose(0, 1)]
        case tensor.real.MatMul(_) | tensor.int.MatMul(_):
            return [r[0] @ r[1]]

        # --- comparisons: write booleans ---
        case tensor.real.Eq(_) | tensor.int.Eq(_) | tensor.bool.Eq(_) | tensor.word.Eq(_, _):
            return [r[0].eq(r[1])]
        case tensor.real.Ne(_) | tensor.int.Ne(_) | tensor.bool.Ne(_) | tensor.word.Ne(_, _):
            return [r[0].ne(r[1])]
        case tensor.real.Lt(_) | tensor.int.Lt(_) | tensor.word.unsigned.Lt(_, _):
            return [r[0].lt(r[1])]
        case tensor.real.Le(_) | tensor.int.Le(_) | tensor.word.unsigned.Le(_, _):
            return [r[0].le(r[1])]
        case tensor.real.Gt(_) | tensor.int.Gt(_) | tensor.word.unsigned.Gt(_, _):
            return [r[0].gt(r[1])]
        case tensor.real.Ge(_) | tensor.int.Ge(_) | tensor.word.unsigned.Ge(_, _):
            return [r[0].ge(r[1])]
        case tensor.word.signed.Lt(w, _):
            return [_signed(r[0], w).lt(_signed(r[1], w))]
        case tensor.word.signed.Le(w, _):
            return [_signed(r[0], w).le(_signed(r[1], w))]
        case tensor.word.signed.Gt(w, _):
            return [_signed(r[0], w).gt(_signed(r[1], w))]
        case tensor.word.signed.Ge(w, _):
            return [_signed(r[0], w).ge(_signed(r[1], w))]

        # --- propositional logic ---
        case tensor.bool.And(_):
            return [r[0].logical_and(r[1])]
        case tensor.bool.Or(_):
            return [r[0].logical_or(r[1])]
        case tensor.bool.Xor(_):
            return [r[0].logical_xor(r[1])]
        case tensor.bool.Not(_):
            return [r[0].logical_not()]
        case tensor.bool.All(_):
            return [r[0].all().reshape(1, 1)]
        case tensor.bool.Implies(_):
            return [r[0].logical_not().logical_or(r[1])]

        # --- an entry of a matrix ---
        case (tensor.real.Get(i, j, _) | tensor.int.Get(i, j, _) | tensor.bool.Get(i, j, _)
              | tensor.word.Get(i, j, _, _)):
            return [r[0][i:i + 1, j:j + 1].clone()]

        # --- casts between the theories' sorts ---
        case tensor.cast.ToReal(_):
            return [r[0].to(torch.float32)]
        case tensor.cast.ToInt(_):
            return [r[0].floor().to(torch.int64)]
        case tensor.cast.UbvToInt(_, _):
            return [r[0].clone()]  # words are stored unsigned

        # --- words: arithmetic modulo 2^w ---
        case tensor.word.Add(w, _):
            return [_wrap(r[0] + r[1], w)]
        case tensor.word.Sub(w, _):
            return [_wrap(r[0] - r[1], w)]
        case tensor.word.Mul(w, _):
            return [_wrap(r[0] * r[1], w)]
        case tensor.word.Neg(w, _):
            return [_wrap(-r[0], w)]
        case tensor.word.signed.Abs(w, _):
            return [_wrap(_signed(r[0], w).abs(), w)]
        case tensor.word.unsigned.Div(w, _):
            return [_udiv(r[0], r[1], w)]
        case tensor.word.unsigned.Rem(_, _):
            return [_urem(r[0], r[1])]
        case tensor.word.signed.Div(w, _):
            return [_sdiv(r[0], r[1], w)]
        case tensor.word.signed.Rem(w, _):
            return [_srem(r[0], r[1], w)]
        case tensor.word.signed.Mod(w, _):
            return [_smod(r[0], r[1], w)]

        # --- words: bitwise ---
        case tensor.word.And(_, _):
            return [r[0] & r[1]]
        case tensor.word.Or(_, _):
            return [r[0] | r[1]]
        case tensor.word.Xor(_, _):
            return [r[0] ^ r[1]]
        case tensor.word.Not(w, _):
            return [_wrap(~r[0], w)]
        case tensor.word.Nand(w, _):
            return [_wrap(~(r[0] & r[1]), w)]
        case tensor.word.Nor(w, _):
            return [_wrap(~(r[0] | r[1]), w)]
        case tensor.word.Xnor(w, _):
            return [_wrap(~(r[0] ^ r[1]), w)]

        # --- words: shifts, rotations and resizing ---
        case tensor.word.Shl(w, _):
            return [_shl(r[0], r[1], w)]
        case tensor.word.unsigned.Shr(w, _):
            return [_lshr(r[0], r[1], w)]
        case tensor.word.signed.Shr(w, _):
            return [_ashr(r[0], r[1], w)]
        case tensor.word.RotateLeft(i, w, _):
            return [_rotl(r[0], i, w)]
        case tensor.word.RotateRight(i, w, _):
            return [_rotl(r[0], w - i % w, w)]
        case tensor.word.Concat([_, v], _):
            return [(r[0] << v) | r[1]]
        case tensor.word.Extract(i, j, _, _):
            return [(r[0] >> j) & ((1 << (i - j + 1)) - 1)]
        case tensor.word.Repeat(i, w, _):
            return [_repeat(r[0], i, w)]
        case tensor.word.unsigned.Extend(_, _, _):
            return [r[0].clone()]
        case tensor.word.signed.Extend(i, w, _):
            return [_wrap(_signed(r[0], w), w + i)]

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
