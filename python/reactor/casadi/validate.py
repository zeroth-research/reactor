"""Check a CasADi -> Reactor translation, numerically and symbolically.

``check_numeric`` evaluates the translated terms with Reactor's torch evaluator and
compares them with CasADi's own evaluation on random inputs.

``check_equivalence`` builds a miter: the CasADi function is encoded into Z3
directly (``casadi_to_z3``, an independent walker over the SX instructions) and the
translated terms through ``reactor.smt.z3``; both read the same Z3 input variables.
For each output entry it first compares sum-of-monomials normal forms and then
asks the solver for an input on which the two differ; with products of variables the
query is nonlinear, which Z3 decides (nlsat) but may need time for. ``proved`` means the entry
is equal for **all real inputs**.

Transcendental functions other than the square root are uninterpreted functions in
Z3, one per name, shared by both sides (``reactor.smt.z3.transcendental``): the
check then proves the translation applies the same function to the same arguments,
which is what it must do, without reasoning about the functions themselves.

Both sides use the same constant semantics, which is the translation's contract
(see ``translate``): constant subexpressions are folded in float64 and division by
a constant is multiplication by its float64 reciprocal. Z3 reads each float from
its shortest decimal representation on both sides.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import casadi as cs
import numpy as np
import torch
import z3

from ..eval import eval_itype
from ..smt import z3 as rz3
from .translate import (
    DTYPE, TRANSCENDENTAL, Terms, _as_sx, _div, fold, integer_exponent, op_name, to_terms,
)

__all__ = [
    "casadi_to_z3",
    "terms_to_z3",
    "check_numeric",
    "check_equivalence",
    "Equivalence",
    "EntryResult",
]


# ---------------------------------------------------------------------------
# Reference: CasADi -> Z3, following CasADi's semantics literally
# (comparisons are the reals 0/1, a condition is "nonzero").
# ---------------------------------------------------------------------------


def _rv(c: float) -> z3.ArithRef:
    return z3.RealVal(float(c))


def _z3_unary(op: int, a: z3.ArithRef) -> z3.ArithRef:
    match op:
        case cs.OP_NEG:
            return -a
        case cs.OP_TWICE:
            return 2 * a
        case cs.OP_SQ:
            return a * a
        case cs.OP_INV:
            return _rv(1.0) / a
        case cs.OP_NOT:
            return z3.If(a == 0, _rv(1.0), _rv(0.0))
        case cs.OP_FABS:
            return z3.If(a >= 0, a, -a)
        case cs.OP_SIGN:
            return z3.If(a > 0, _rv(1.0), z3.If(a < 0, _rv(-1.0), _rv(0.0)))
        case cs.OP_SQRT:
            return z3.Sqrt(a)
    if op in TRANSCENDENTAL:  # the uninterpreted functions of `reactor.smt.z3`
        return rz3.transcendental(TRANSCENDENTAL[op])(a)
    raise NotImplementedError(f"reference: no Z3 encoding for '{op_name(op)}' of a non-constant")


def _z3_binary(op: int, a, b) -> z3.ArithRef:
    one, zero = _rv(1.0), _rv(0.0)
    match op:
        case cs.OP_ADD:
            return a + b
        case cs.OP_SUB:
            return a - b
        case cs.OP_MUL:
            return a * b
        case cs.OP_DIV:
            if isinstance(b, float):  # the translation's contract: float64 reciprocal
                return a * _rv(_div(1.0, b))
            return a / b
        case cs.OP_LT:
            return z3.If(a < b, one, zero)
        case cs.OP_LE:
            return z3.If(a <= b, one, zero)
        case cs.OP_EQ:
            return z3.If(a == b, one, zero)
        case cs.OP_NE:
            return z3.If(a != b, one, zero)
        case cs.OP_AND:
            return z3.If(z3.And(a != 0, b != 0), one, zero)
        case cs.OP_OR:
            return z3.If(z3.Or(a != 0, b != 0), one, zero)
        case cs.OP_IF_ELSE_ZERO:
            return z3.If(a != 0, b, zero)
        case cs.OP_FMIN:
            return z3.If(a <= b, a, b)
        case cs.OP_FMAX:
            return z3.If(a >= b, a, b)
        case cs.OP_ATAN2:
            return rz3.transcendental("atan2", 2)(a, b)
        case cs.OP_HYPOT:
            return z3.Sqrt(a * a + b * b)
        case cs.OP_POW | cs.OP_CONSTPOW if isinstance(b, float) and integer_exponent(b) is not None:
            n = integer_exponent(b)
            return a**n if n >= 0 else _rv(1.0) / a ** (-n)
    raise NotImplementedError(f"reference: no Z3 encoding for '{op_name(op)}' of a non-constant")


def _lift(v):
    return _rv(v) if isinstance(v, float) else v


def casadi_to_z3(f: cs.Function, inputs: Sequence[Sequence[z3.ArithRef]]) -> list[list]:
    """Encode `f` into Z3 over `inputs` (per input, numel Z3 reals, column-major).

    Returns per output ``numel`` Z3 real terms (column-major). Constant
    subexpressions are folded in float64, as in the translation.
    """
    f = _as_sx(f)
    nz_in = [[inputs[i][k] for k in f.sparsity_in(i).find()] for i in range(f.n_in())]
    outs = [[_rv(0.0)] * f.numel_out(j) for j in range(f.n_out())]
    nz_out = [f.sparsity_out(j).find() for j in range(f.n_out())]
    reg: dict = {}
    for k in range(f.n_instructions()):
        op = f.instruction_id(k)
        i, o = f.instruction_input(k), f.instruction_output(k)
        if op == cs.OP_INPUT:
            reg[o[0]] = nz_in[i[0]][i[1]]
        elif op == cs.OP_OUTPUT:
            outs[o[0]][nz_out[o[0]][o[1]]] = _lift(reg[i[0]])
        elif op == cs.OP_CONST:
            reg[o[0]] = float(f.instruction_constant(k))
        elif op == cs.OP_ASSIGN:
            reg[o[0]] = reg[i[0]]
        else:
            args = [reg[j] for j in i]
            if all(isinstance(a, float) for a in args):
                reg[o[0]] = fold(op, args)
            elif len(args) == 1:
                reg[o[0]] = _z3_unary(op, args[0])
            elif op in (cs.OP_DIV, cs.OP_POW, cs.OP_CONSTPOW):  # keep a constant 2nd operand
                reg[o[0]] = _z3_binary(op, _lift(args[0]), args[1])
            else:
                reg[o[0]] = _z3_binary(op, *map(_lift, args))
    return outs


# ---------------------------------------------------------------------------
# Candidate: translated terms -> Z3 through Reactor's own encoding
# ---------------------------------------------------------------------------


def terms_to_z3(t: Terms, inputs: Sequence[Sequence[z3.ArithRef]]) -> list[list]:
    """Encode translated terms into Z3 with ``reactor.smt.z3``."""
    env = {}
    for wires, vals in zip(t.inputs, inputs):
        for w, v in zip(wires, vals):
            env[w] = np.array([[v]], dtype=object)
    for term in t.terms:
        results = rz3.eval(term.itype, [env[w] for w in term.read])
        for w, r in zip(term.write, results):
            env[w] = r
    return [[env[w][0, 0] for w in ws] for ws in t.outputs]


# ---------------------------------------------------------------------------
# Numerical check
# ---------------------------------------------------------------------------


def _eval_terms(t: Terms, inputs: Sequence[Sequence[float]]) -> list[list[float]]:
    env = {}
    for wires, vals in zip(t.inputs, inputs):
        for w, v in zip(wires, vals):
            env[w] = torch.tensor([[v]], dtype=DTYPE)
    for term in t.terms:
        out_sort = term.write[0].dtype if len(term.write) else None
        results = eval_itype(term.itype, [env[w] for w in term.read], out_sort)
        for w, r in zip(term.write, results):
            env[w] = r
    return [[float(env[w].reshape(-1)[0]) for w in ws] for ws in t.outputs]


def check_numeric(
    f: cs.Function,
    samples: int = 100,
    *,
    low: float | Sequence[np.ndarray] = -1.0,
    high: float | Sequence[np.ndarray] = 1.0,
    seed: int = 0,
    terms: Terms | None = None,
) -> float:
    """Largest absolute difference between `f` and its translation on random inputs.

    Inputs are drawn uniformly from ``[low, high]`` (scalars, or one array of
    ``numel`` bounds per input).
    """
    f = _as_sx(f)
    t = terms if terms is not None else to_terms(f)
    rng = np.random.default_rng(seed)

    def bounds(b, i):
        if isinstance(b, (int, float)):
            return np.full(f.numel_in(i), float(b))
        return np.asarray(b[i], dtype=float).reshape(-1)

    worst = 0.0
    for _ in range(samples):
        dense = [rng.uniform(bounds(low, i), bounds(high, i)) for i in range(f.n_in())]
        for i in range(f.n_in()):  # structural zeros stay zero
            mask = np.zeros(f.numel_in(i), dtype=bool)
            mask[f.sparsity_in(i).find()] = True
            dense[i][~mask] = 0.0
        args = [
            cs.DM(f.sparsity_in(i), dense[i][f.sparsity_in(i).find()]) for i in range(f.n_in())
        ]
        ref = f.call(args)
        ref = [np.asarray(cs.densify(r)).reshape(-1, order="F") for r in ref]
        got = _eval_terms(t, [d.tolist() for d in dense])
        for r, g in zip(ref, got):
            worst = max(worst, float(np.max(np.abs(r - np.asarray(g)), initial=0.0)))
    return worst


# ---------------------------------------------------------------------------
# Symbolic check (miter)
# ---------------------------------------------------------------------------


@dataclass
class EntryResult:
    output: str  # output name
    index: int  # column-major entry index
    status: str  # "proved" | "counterexample" | "unknown"
    method: str = ""  # "normal-form" | "solver"
    counterexample: dict[str, float] = field(default_factory=dict)


@dataclass
class Equivalence:
    entries: list[EntryResult]

    @property
    def proved(self) -> bool:
        return all(e.status == "proved" for e in self.entries)

    def failures(self) -> list[EntryResult]:
        return [e for e in self.entries if e.status != "proved"]

    def __str__(self) -> str:
        n = len(self.entries)
        ok = sum(e.status == "proved" for e in self.entries)
        lines = [f"{ok}/{n} output entries proved equivalent"]
        for e in self.failures():
            lines.append(f"  {e.output}[{e.index}]: {e.status} {e.counterexample or ''}".rstrip())
        return "\n".join(lines)


def _model_value(m: z3.ModelRef, v: z3.ArithRef) -> float:
    val = m.eval(v, model_completion=True)
    if z3.is_rational_value(val):
        return val.numerator_as_long() / val.denominator_as_long()
    return float(val.approx(20).as_decimal(20).rstrip("?"))


def check_equivalence(
    f: cs.Function,
    *,
    terms: Terms | None = None,
    assumptions: Sequence[z3.BoolRef] | None = None,
    inputs: Sequence[Sequence[z3.ArithRef]] | None = None,
    timeout_ms: int = 10_000,
) -> Equivalence:
    """Prove that the translation of `f` equals `f` for all real inputs.

    Args:
        f: The CasADi function.
        terms: Its translation; computed with ``to_terms`` if not given.
        assumptions: Extra constraints over `inputs` (e.g. a box, or ‖q‖ = 1).
        inputs: Z3 reals for the inputs (per input, numel, column-major); fresh
            ones named ``<input>[k]`` if not given.
        timeout_ms: Solver timeout per output entry.
    """
    f = _as_sx(f)
    t = terms if terms is not None else to_terms(f)
    if inputs is None:
        inputs = [
            [z3.Real(f"{f.name_in(i)}[{k}]") for k in range(f.numel_in(i))]
            for i in range(f.n_in())
        ]
    ref = casadi_to_z3(f, inputs)
    cand = terms_to_z3(t, inputs)
    flat_inputs = [v for vs in inputs for v in vs]

    entries = []
    for j in range(f.n_out()):
        for k, (r, c) in enumerate(zip(ref[j], cand[j])):
            name = f.name_out(j)
            r, c = _lift(r), _lift(c)
            if not assumptions and z3.eq(z3.simplify(r - c, som=True), _rv(0.0)):
                entries.append(EntryResult(name, k, "proved", "normal-form"))
                continue
            s = z3.Solver()
            s.set("timeout", timeout_ms)
            s.add(*(assumptions or []))
            s.add(r != c)
            res = s.check()
            if res == z3.unsat:
                entries.append(EntryResult(name, k, "proved", "solver"))
            elif res == z3.sat:
                m = s.model()
                cex = {str(v): _model_value(m, v) for v in flat_inputs}
                entries.append(EntryResult(name, k, "counterexample", "solver", cex))
            else:
                entries.append(EntryResult(name, k, "unknown", "solver"))
    return Equivalence(entries)
