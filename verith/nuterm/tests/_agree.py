"""Random runs of a module, and what its encodings say about them: the
check `lean/tests/agree.py` makes over `reactor_lean`'s own modules, made
here over the SV-COMP corpus (`benchmarks.svcomp`), which is nuterm's.

The reference is `reactor.eval`: the interpreter `reactor.gym` runs a module with.
A run draws each round's awaited inputs at random; the rounds are
`execute_init` and then `execute_update`, with the next wires latched in
between, exactly as `reactor.gym` steps a module. An encoding is then asked
about each round *from the reference's own state*, so a disagreement is
reported at the round it happens in and does not compound.

`test_svcomp_agree.py` asks the SMT encoding and
`verith/tests/lean/test_lean_svcomp_agree.py` the Lean one, about the same
runs.
"""

from __future__ import annotations

import random
from fractions import Fraction

import torch

from reactor import Bool, BitVec, Int, Real, X
from reactor.eval import execute_init, execute_update
from reactor_lean.lowering.sorts import wire_shape

RUNS = 6            # runs per module
ROUNDS = 5          # rounds per run after the initial one


class Skip(Exception):
    """The module is outside what one side can be asked: the reference
    cannot evaluate it (an uninterpreted or a nondeterministic op), or the
    encoding refuses it. Each says why."""


# ── the corpus ──────────────────────────────────────────────────────────────


def _corpus() -> dict:
    """Name -> a function building the module, for every SV-COMP benchmark."""
    from nuterm.tests.benchmarks.svcomp import discover

    return {f"svcomp/{b.name}": (lambda b=b: b.build()[0]) for b in discover()}


CORPUS = _corpus()


# ── runs ────────────────────────────────────────────────────────────────────


def _draw(wire, rng: random.Random) -> torch.Tensor:
    """A random value for `wire`, shaped as `reactor.eval` keeps it."""
    dt, shape = wire.dtype, wire_shape(wire)
    if isinstance(dt, Bool):
        vals, kind = [rng.random() < 0.5 for _ in range(shape.total)], torch.bool
    elif isinstance(dt, Int):
        vals, kind = [rng.randint(-8, 8) for _ in range(shape.total)], torch.int64
    elif isinstance(dt, Real):
        vals = [rng.randint(-16, 16) / 4 for _ in range(shape.total)]
        kind = torch.float64
    elif isinstance(dt, BitVec):
        vals = [rng.randrange(1 << dt._0) for _ in range(shape.total)]
        kind = torch.int64
    else:                                   # pragma: no cover
        raise Skip(f"no random values for sort {dt}")
    return torch.tensor(vals, dtype=kind).reshape(shape.m, shape.n)


def runs(module, *, seed: int = 0, count: int = RUNS) -> list:
    """`count` runs of `module`: each a list of rounds `(inputs, state)`,
    where `inputs` maps each input variable to the value awaited that round
    and `state` lists the ctrl values after it, in `module.ctrl` order. The
    first runs of a seed are the same whatever `count` is."""
    if len(list(module.atoms)) != 1:
        raise Skip("more than one atom; the encodings take one")
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        st: dict = {}
        run = []
        for k in range(ROUNDS + 1):
            inputs = {e: _draw(e, rng) for e in module.extl}
            if k:
                for c in module.ctrl:
                    st[c] = st[X(c)]
                for e in module.extl:
                    st[e] = st[X(e)]
            for e in module.extl:
                st[X(e)] = inputs[e]
            try:
                (execute_update if k else execute_init)(st, module.atoms)
            except RuntimeError as e:
                raise Skip(f"the interpreter cannot run it: {e}") from e
            run.append((inputs, [st[X(c)] for c in module.ctrl]))
        out.append(run)
    return out


def flat(wire, tensor) -> list:
    """`tensor` as `wire`'s elements, row-major, as plain Python values."""
    vals = tensor.reshape(-1).tolist()
    dt = wire.dtype
    if isinstance(dt, Bool):
        return [bool(v) for v in vals]
    if isinstance(dt, Real):
        return [Fraction(v) for v in vals]
    return vals


def same(wire, want: list, got: list) -> bool:
    """Whether `got` is `want`, elementwise. A Real is compared to within
    rounding, since the reference computes in floats."""
    if len(want) != len(got):
        return False
    if isinstance(wire.dtype, Real):
        tol = Fraction(1, 10**9)
        return all(abs(Fraction(a) - Fraction(b)) <= tol * (1 + abs(Fraction(a)))
                   for a, b in zip(want, got))
    # `True == 1` in Python; a Bool wire read as a number is a disagreement.
    return all(a == b and isinstance(a, bool) == isinstance(b, bool)
               for a, b in zip(want, got))


# ── the SMT encoding ────────────────────────────────────────────────────────


def _value(term) -> object:
    """A cvc5 value as a Python one; a tuple as its elements."""
    if term.isTupleValue():
        return [_value(x) for x in term.getTupleValue()]
    srt = term.getSort()
    if srt.isBoolean() and term.isBooleanValue():
        return term.getBooleanValue()
    if srt.isInteger() and term.isIntegerValue():
        return term.getIntegerValue()
    if srt.isReal() and term.isRealValue():
        return term.getRealValue()
    if srt.isBitVector() and term.isBitVectorValue():
        return int(term.getBitVectorValue(10))
    raise ValueError(f"not a value: {term}")


def disagreements_smt(module, *, seed: int = 0) -> list[str]:
    """Each round of `runs(module)` where the encoded round, evaluated by
    cvc5's rewriter at the same state and inputs, lands somewhere else."""
    import cvc5

    from reactor_lean.smt.encode import tensor_const
    from reactor_lean.smt.module import ModuleSMT, keep_alive

    rounds = runs(module, seed=seed)
    tm = cvc5.TermManager()
    keep_alive(tm)
    try:
        msmt = ModuleSMT(tm=tm, module=module)
        s0, el0, en0 = (msmt.fresh_ctrl(), msmt.fresh_extl_l(),
                        msmt.fresh_extl_n())
        init = msmt.init_state(en0)
        update = msmt.update_state(s0, el0, en0)
    except (ValueError, RuntimeError) as e:
        raise Skip(f"no SMT encoding: {e}") from e
    rewriter = cvc5.Solver(tm)
    rewriter.setLogic("ALL")
    keep_alive(rewriter)

    def lits(wires, values):
        return [tensor_const(tm, w, v) for w, v in zip(wires, values)]

    def at(terms, old, new):
        """The round translated once, at the constants `new`."""
        return [t.substitute(old, new) for t in terms] if old else list(terms)

    bad = []
    for r, run in enumerate(rounds):
        for k, ((inputs, state), before) in enumerate(zip(run, [None, *run])):
            en = lits([X(e) for e in module.extl], [inputs[e] for e in module.extl])
            if before is None:
                terms = at(init, en0, en)
            else:
                prev_inputs, prev_state = before
                s = lits(module.ctrl, prev_state)
                el = lits(module.extl, [prev_inputs[e] for e in module.extl])
                terms = at(update, s0 + el0 + en0, s + el + en)
            for c, want_t, term in zip(module.ctrl, state, terms):
                want = flat(c, want_t)
                try:
                    got = _value(rewriter.simplify(term))
                except ValueError as e:
                    bad.append(f"run {r} round {k}: {c.id} does not evaluate: {e}")
                    continue
                got = got if isinstance(got, list) else [got]
                if not same(c, want, got):
                    bad.append(f"run {r} round {k}: wire {c.id} ({c.dtype}) is "
                               f"{want} by the interpreter and {got} by cvc5")
    return bad


# ── the Lean encoding ───────────────────────────────────────────────────────


def _lean_value(wire, tensor) -> str:
    """`tensor` as a Lean `Matrix` literal of `wire`'s sort, in the form the
    encoding prints its own constants in
    (`lowering.elements.mat_from_scalars`)."""
    from reactor_lean.lowering.elements import mat_from_scalars
    from reactor_lean.codegen.literals import lean_literal

    elems = [lean_literal(wire.dtype, v) for v in flat(wire, tensor)]
    return mat_from_scalars(elems, wire_shape(wire))


def _lean_shown(wire, element: str) -> str:
    """`element` of `wire`'s sort, printed as `flat` prints the reference:
    a BitVec as the natural number it holds."""
    if isinstance(wire.dtype, BitVec):
        return f"{{toString ({element}).toNat}}"
    return f"{{toString ({element})}}"


def lean_script(module, rounds: list, ns: str) -> "tuple[str, dict[str, str]]":
    """Lean that `#eval`s `module`'s functional encoding on `rounds`, and
    the line each `#eval` should print. The text imports nothing: a file
    of several states `import Relational.Circ` once, above them all.

    Each round is a `def` in namespace `ns` -- `init` at the round's awaited
    inputs, or `update` at the reference's state before it -- and one
    printed line, `<tag>|<wire>|<wire>...`, each wire its elements
    row-major. Raises `Skip` for a module Lean cannot run: a Real is
    noncomputable, and a module with no Lean form has nothing to run.
    """
    from reactor_lean.lowering.elements import flat_layout
    from reactor_lean.codegen.context import LeanContext
    from reactor_lean.codegen.terms import build_tuple, lean_gaps
    from reactor_lean.codegen.encodings import ModuleToLean4

    wires = list(module.ctrl) + list(module.extl)
    if any(isinstance(w.dtype, Real) for w in wires):
        raise Skip("a Real is noncomputable in Lean, so there is nothing to #eval")
    ctx = LeanContext(module)
    gaps = lean_gaps(ctx)
    if gaps:
        raise Skip(f"no Lean form: {gaps[0]}")

    ctrl, extl = list(module.ctrl), list(module.extl)
    layout = flat_layout(ctrl)
    lines = [f"namespace {ns}", ModuleToLean4(ctx).to_lean_functional()]
    evals, want = [], {}
    for r, run in enumerate(rounds):
        for k, ((inputs, state), before) in enumerate(zip(run, [None, *run])):
            en = build_tuple([_lean_value(e, inputs[e]) for e in extl])
            if before is None:
                call = f"init {en}"
            else:
                prev_inputs, prev_state = before
                s = build_tuple([_lean_value(c, v) for c, v in zip(ctrl, prev_state)])
                el = build_tuple([_lean_value(e, prev_inputs[e]) for e in extl])
                call = f"update {s} {el} {en}"
            tag = f"{ns} r{r} k{k}"
            name = f"r{r}k{k}"
            lines.append(f"def {name} := {call}")
            shown, expected = [], []
            for p, c in enumerate(ctrl):
                shown.append(" ".join(
                    _lean_shown(c, layout.wire_accessor(f"{ns}.{name}", s))
                    for s in layout.slots_of(c)))
                expected.append(" ".join(
                    str(v).lower() if isinstance(v, bool) else str(int(v))
                    for v in flat(c, state[p])))
            evals.append(f'#eval IO.println s!"{tag}|{"|".join(shown)}"')
            want[tag] = f"{tag}|{'|'.join(expected)}"
    lines.append(f"end {ns}")
    return "\n".join(["", *lines, "", *evals, ""]), want
