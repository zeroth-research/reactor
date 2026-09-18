"""Shared helpers for the ranking-function training tutorials (PTQ / QAT).

Not part of the framework — this is *usage* code: an integer program, a small ReLU
ranking network, a training loop, a `quantise` step that turns a trained real (LRA)
ranking module into the equivalent integer (LIA) one, and the LIA verification.

The program is a counter that reaches a target (`x == y` or `x == z`); `y`, `z` are
constant. `R(x, y, z)` is a liveness certificate: away from a target it stays
non-negative and strictly decreases, so a target is always reached.
"""

import numpy as np
import torch
import torch.nn as nn
import z3

from zrth import LIA, Int, Var, X
from zrth import sugar
from zrth.sugar import ite
from zrth.torch import Module
from zrth import z3 as zz3

Y, Z = 30, 50                         # the (constant) targets
INT = Int([1, 1])


class SugarCounter(sugar.Module):
    """x increments while x < y or x < z, resets to 0 otherwise; y, z held constant."""

    def init(self):
        return 0, 0, 0

    def update(self, x, y, z):
        loop = (x < y) | (x < z)
        return ite(loop, x + 1, 0), y, z


class RankingNN(nn.Module):
    """R(x, y, z) -> scalar;  [3] -> 4 -> [1]."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, s):
        return self.fc2(torch.relu(self.fc1(s)))


def make_program():
    """The integer program (LIA) and its ordered state variables."""
    x, y, z = Var(INT), Var(INT), Var(INT)
    program = SugarCounter(theory=LIA, ctrl=(x, y, z))
    return program, [x, y, z]


def _succ(s):
    x, y, z = s
    return (x + 1.0 if (x < y or x < z) else 0.0, y, z)


def train(real_module, *, margin=1.0, epochs=400, lr=0.01):
    """Train a real (LRA) ranking module in place on the reachable in-loop states."""
    opt = torch.optim.Adam(real_module.parameters(), lr=lr)
    states = [(float(x), float(Y), float(Z)) for x in range(max(Y, Z))]
    for _ in range(epochs):
        loss = torch.tensor(0.0)
        for s in states:
            sn = _succ(s)
            r = real_module(torch.tensor(s).float().unsqueeze(0)).squeeze()
            rn = real_module(torch.tensor(sn).float().unsqueeze(0)).squeeze()
            loss = loss + torch.relu(-r) + torch.relu(rn - r + margin)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return real_module


class _RoundSTE(torch.autograd.Function):
    """Round on the forward pass, identity gradient on the backward pass (straight-through)."""

    @staticmethod
    def forward(ctx, t):
        return t.round()

    @staticmethod
    def backward(ctx, g):
        return g


def _fake_quant(w, scale):
    """Quantise-then-dequantise: the net computes at real scale but 'sees' rounded weights."""
    return _RoundSTE.apply(w * scale) / scale


class QATRanking(nn.Module):
    """Same [3] -> 4 -> [1] net, but `forward` fake-quantises the weights (STE) at `scale`,
    so training already accounts for the rounding used at verification time."""

    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, s):
        import torch.nn.functional as F
        h = F.relu(F.linear(s, _fake_quant(self.fc1.weight, self.scale),
                            _fake_quant(self.fc1.bias, self.scale)))
        return F.linear(h, _fake_quant(self.fc2.weight, self.scale),
                        _fake_quant(self.fc2.bias, self.scale))


def quantise(real_module, scale):
    """Turn a trained real (LRA) ranking module into the equivalent integer (LIA) one:
    same architecture, Int instead of Real, weights fixed-point-rounded to round(scale*w).
    Verify the result with delta' = scale * delta.

    The int module is rebuilt as a plain `RankingNN` (standard forward) so the analyzer can
    extract it — this is why it works for a QAT net's fake-quant forward too."""
    src = real_module.inner
    qnet = RankingNN()
    with torch.no_grad():
        for dst, s in ((qnet.fc1, src.fc1), (qnet.fc2, src.fc2)):
            dst.weight = nn.Parameter((s.weight * scale).round().to(torch.int64), requires_grad=False)
            dst.bias = nn.Parameter((s.bias * scale).round().to(torch.int64), requires_grad=False)
    return Module(qnet, theory=LIA, combinatorial=True)


def verify(program, ranking_module, state_vars, *, delta):
    """Check the ranking obligation in LIA over the in-loop domain:
    returns (V >= 0 holds, V decreases by >= delta holds)."""
    s = {v: np.array([[z3.FreshInt()]]) for v in state_vars}
    for atom in program.atoms:
        for t in atom.update:
            s.update(zip(t.write, zz3.eval(t.itype, [s[w] for w in t.read])))
    cur = [s[v].item() for v in state_vars]
    nxt = [s[X(v)].item() for v in state_vars]

    ext, itf = list(ranking_module.extl), list(ranking_module.intf)

    def R(vec):
        r = {X(ext[0]): np.array([vec])}
        for atom in ranking_module.atoms:
            for t in atom.update:
                r.update(zip(t.write, zz3.eval(t.itype, [r[w] for w in t.read])))
        return r[X(itf[0])].item()

    dom = [cur[1] == Y, cur[2] == Z, z3.And(cur[0] >= 0, z3.Or(cur[0] < cur[1], cur[0] < cur[2]))]

    def holds(negation):
        solver = z3.Solver()
        solver.add(*dom)
        solver.add(negation)
        return solver.check() == z3.unsat

    return holds(R(cur) < 0), holds(R(nxt) >= R(cur) - delta)
