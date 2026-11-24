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
from .reactor import X, d as _d
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


def execute_delay(state, atoms):
    """Evaluate the delay block of all atoms: write the rate of change ``d(v)`` of each
    variable they control, from the values in `state` (the latched variables and the
    inputs). A discrete variable's rate is the zero of `Zero`, which writes nothing of
    use; the flowing variables are the reals, whose rates are reals of degree 1."""
    _execute_block(state, atoms, lambda a: a.delay)


def flowing(atoms) -> list:
    """The variables the atoms control that flow: the reals, whose delay block writes a
    rate of change (every other sort, and a real held by a `Zero` rate, has rate zero)."""
    out = []
    for atom in atoms:
        # the rates written by an operation other than `Zero` (the rate of a variable
        # that does not change, which a sequential module's delay block holds)
        writes = {w for term in atom.delay if not isinstance(term.itype, Zero) for w in term.write}
        for v in atom.ctrl:
            if isinstance(v.dtype, Real) and _d(v) in writes:
                out.append(v)
    return out


def rates(state, atoms, variables=None) -> list:
    """The rates of change ``d(v)`` of `variables` (default: `flowing(atoms)`) at the
    values in `state`, which is left as it was."""
    variables = flowing(atoms) if variables is None else variables
    env = dict(state)
    execute_delay(env, atoms)
    return [env[_d(v)] for v in variables]


def execute_flow(state, atoms, dt: float, method: str = "rk4"):
    """Let the atoms flow for `dt`: advance every flowing variable (`flowing`) in
    `state` by one step of `method` for the ODE their delay blocks define, holding the
    other values in `state` (the inputs, and the discrete variables) over the step.

    `method` is ``"euler"``, ``x + dt·f(x)``, or ``"rk4"``, the classical Runge-Kutta
    step, written in the operation order of `reactor.casadi.rk4`, so a differential
    module translated from CasADi flows exactly as the CasADi step of it computes."""
    variables = flowing(atoms)
    x = [state[v] for v in variables]

    def f(values):
        env = dict(state)
        env.update(zip(variables, values))
        execute_delay(env, atoms)
        return [env[_d(v)] for v in variables]

    if method == "euler":
        new = [xi + dt * ki for xi, ki in zip(x, f(x))]
    elif method == "rk4":
        k1 = f(x)
        k2 = f([xi + dt / 2 * ki for xi, ki in zip(x, k1)])
        k3 = f([xi + dt / 2 * ki for xi, ki in zip(x, k2)])
        k4 = f([xi + dt * ki for xi, ki in zip(x, k3)])
        new = [xi + dt / 6 * (a + 2 * b + 2 * c + e) for xi, a, b, c, e in zip(x, k1, k2, k3, k4)]
    else:
        raise ValueError(f"unknown method {method!r}: use 'euler' or 'rk4'")
    for v, value in zip(variables, new):
        state[v] = value


# ============================================================================
# Compiled evaluation: a block resolved once, then run many times
# ============================================================================
#
# `execute_*` interpret a block afresh at every call: they read each term's
# operation, read and write wires from the extension, look the values up by wire
# (hashed in Rust), and dispatch on the operation in `eval_itype`. For a module run
# for thousands of steps that overhead dominates (about 9 µs per term, against ~2 µs
# for the torch operation itself). A `Program` does all of it once: it numbers the
# wires (slots), and binds each term to a kernel, a function of the read values.


def _kernel(itype, out_sort):
    """A function computing `itype` from the list of its read values, with the data it
    holds extracted once. The common operations get a direct kernel; any other falls
    back to `eval_itype`, so both always compute the same."""
    match itype:
        case Skip(_):
            return lambda r: [r[0]]  # values are never mutated in place: no copy needed
        case (tensor.real.Const(t) | tensor.int.Const(t) | tensor.bool.Const(t)
              | _jax.real.Const(t) | _jax.int.Const(t) | _jax.bool.Const(t)
              | _mixed.real.Const(t) | _mixed.int.Const(t) | _mixed.bool.Const(t)):
            value = _held(t).clone()
            return lambda r: [value]
        case (tensor.real.Scale(c) | tensor.int.Scale(c) | _jax.real.Scale(c) | _jax.int.Scale(c)
              | _mixed.real.Scale(c) | _mixed.int.Scale(c)):
            c = _held(c)
            return lambda r: [c * r[0]]
        case (tensor.real.Linear(a, b) | tensor.int.Linear(a, b) | _jax.real.Linear(a, b)
              | _jax.int.Linear(a, b) | _mixed.real.Linear(a, b) | _mixed.int.Linear(a, b)):
            a, b = _held(a), _held(b)
            return lambda r: [a @ r[0] + b]
        case tensor.real.Add(_) | tensor.int.Add(_):
            return lambda r: [r[0] + r[1]]
        case tensor.real.Sub(_) | tensor.int.Sub(_):
            return lambda r: [r[0] - r[1]]
        case tensor.real.Mul(_) | tensor.int.Mul(_):
            return lambda r: [r[0] * r[1]]
        case tensor.real.Div(_):
            return lambda r: [r[0] / r[1]]
        case tensor.real.Neg(_) | tensor.int.Neg(_):
            return lambda r: [-r[0]]
        case tensor.real.Ite(_) | tensor.int.Ite(_) | tensor.bool.Ite(_):
            return lambda r: [torch.where(r[0], r[1], r[2])]
    return lambda r: eval_itype(itype, r, out_sort)


class Program:
    """A sequence of terms compiled against a table of slots (see `Runner`): each term
    is a kernel, the slots it reads and the slots it writes."""

    def __init__(self, terms, slot):
        self.ops = []
        for term in terms:
            write = list(term.write)
            out_sort = write[0].dtype if write else None
            self.ops.append((_kernel(term.itype, out_sort), [slot(w) for w in term.read],
                             [slot(w) for w in write]))

    def run(self, values: list) -> None:
        """Evaluate the terms in order, reading and writing `values`, indexed by slot."""
        for kernel, reads, writes in self.ops:
            results = kernel([values[i] for i in reads])
            for j, value in zip(writes, results):
                values[j] = value

    def __len__(self) -> int:
        return len(self.ops)


class Runner:
    """A module (or its atoms) compiled once, to be run many times: its init, update
    and delay blocks as `Program`s over one table of slots, one per wire.

    Values are set and read by wire or variable (``runner[v] = tensor``). Then:

    - `init()` runs the init block, and latches the controlled variables (``v ← X(v)``);
    - `update()` runs the update block and latches them: one discrete step;
    - `flow(dt, method)` lets the flowing variables flow for `dt`, as `execute_flow`.

    It computes exactly what `execute_init`, `execute_update` and `execute_flow` compute,
    the same tensor operations in the same order, several times faster.
    """

    def __init__(self, module_or_atoms):
        atoms = list(getattr(module_or_atoms, "atoms", module_or_atoms))
        self._slots: dict = {}
        slot = self._slot
        self.init_program = Program([t for a in atoms for t in a.init], slot)
        self.update_program = Program([t for a in atoms for t in a.update], slot)
        self.delay_program = Program([t for a in atoms for t in a.delay], slot)
        self.ctrl = [v for a in atoms for v in a.ctrl]
        self._latch = [(slot(X(v)), slot(v)) for v in self.ctrl]
        self.flowing = flowing(atoms)
        self._flow = [slot(v) for v in self.flowing]
        self._rate = [slot(_d(v)) for v in self.flowing]
        self.values: list = [None] * len(self._slots)

    def _slot(self, wire) -> int:
        if wire not in self._slots:
            self._slots[wire] = len(self._slots)
        return self._slots[wire]

    def __getitem__(self, wire):
        return self.values[self._slots[wire]]

    def __setitem__(self, wire, value) -> None:
        if wire not in self._slots:
            raise KeyError(f"{wire!r} is neither read nor written by the module")
        self.values[self._slots[wire]] = value

    def latch(self) -> None:
        """``v ← X(v)`` for every controlled variable whose next value was written."""
        for nxt, cur in self._latch:
            if self.values[nxt] is not None:
                self.values[cur] = self.values[nxt]

    def init(self) -> None:
        self.init_program.run(self.values)
        self.latch()

    def update(self) -> None:
        self.update_program.run(self.values)
        self.latch()

    def rates(self) -> list:
        """The rates of change of the flowing variables at the current values (left as
        they are)."""
        values = self.values.copy()
        self.delay_program.run(values)
        return [values[j] for j in self._rate]

    def flow(self, dt: float, method: str = "rk4") -> None:
        """One step of `method` (``"euler"`` or ``"rk4"``) of the flow of the delay
        blocks, holding the other values; as `execute_flow`."""
        x = [self.values[i] for i in self._flow]

        def f(xs):
            values = self.values.copy()
            for i, xi in zip(self._flow, xs):
                values[i] = xi
            self.delay_program.run(values)
            return [values[j] for j in self._rate]

        if method == "euler":
            new = [xi + dt * ki for xi, ki in zip(x, f(x))]
        elif method == "rk4":
            k1 = f(x)
            k2 = f([xi + dt / 2 * ki for xi, ki in zip(x, k1)])
            k3 = f([xi + dt / 2 * ki for xi, ki in zip(x, k2)])
            k4 = f([xi + dt * ki for xi, ki in zip(x, k3)])
            new = [xi + dt / 6 * (a + 2 * b + 2 * c + e) for xi, a, b, c, e in zip(x, k1, k2, k3, k4)]
        else:
            raise ValueError(f"unknown method {method!r}: use 'euler' or 'rk4'")
        for i, value in zip(self._flow, new):
            self.values[i] = value


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
