"""Random runs of a module, and what its encodings say about them.

The reference is `reactor.eval`: the interpreter `reactor.gym` runs a module with.
A run draws each round's awaited inputs at random; the rounds are
`execute_init` and then `execute_update`, with the next wires latched in
between, exactly as `reactor.gym` steps a module. An encoding is then asked
about each round *from the reference's own state*, so a disagreement is
reported at the round it happens in and does not compound.

`tests/test_encodings_agree.py` asks the SMT encoding and
`tests/lean/test_lean_agree.py` the Lean one, about the same runs.
"""

from __future__ import annotations

import random
from fractions import Fraction

import torch

from reactor import Bool, Int, Real, Word, X
from reactor.eval import execute_init, execute_update
from reactor_lean.lowering.sorts import wire_shape
from testdata import FIXTURES, MODS

RUNS = 6            # runs per module
ROUNDS = 5          # rounds per run after the initial one


class Skip(Exception):
    """The module is outside what one side can be asked: the reference
    cannot evaluate it (an uninterpreted or a nondeterministic op), or the
    encoding refuses it. Each says why."""


# ── the corpus ──────────────────────────────────────────────────────────────


def _corpus() -> dict:
    """Name -> a function building the module: loaded the way `reactor`
    loads a Python input, so a fixture returning a `Named` module is the
    module it names."""
    from reactor_lean.inputs.python import load_module as _load

    out = {}
    for p in sorted(MODS.glob("m_*.py")):
        out[f"limits/{p.stem}"] = (lambda p=p: _load(p))
    for p in sorted(FIXTURES.glob("*.py")):
        if "def module(" in p.read_text():
            out[f"fixtures/{p.stem}"] = (lambda p=p: _load(p))
    return out


def _one_op(op, ins: list, out) -> object:
    """A module whose one variable is `op` of fresh inputs, every round: so
    each round asks the op alone, at random arguments."""
    from reactor import Module, Term, Var

    es = [Var(s) for s in ins]
    x = Var(out)
    block = [Term(op, [X(x)], [X(e) for e in es])]
    return Module.sequential([x, *es], block, list(block))


def _ops() -> dict:
    """One-op modules for every op the corpus never reaches but some
    encoding encodes, so that each encoded op is held to the interpreter.
    Words at width 4, where a random draw is negative as often as not.
    Real `Div` is not among them: the interpreter's `x / 0` is infinite,
    and the encodings' is 0."""
    from reactor import Word, tensor

    w, sc = 4, [1, 1]
    bv, real, boolean, integer = Word(w, sc), Real(sc), Bool(sc), Int(sc)
    word, signed, unsigned = tensor.word, tensor.word.signed, tensor.word.unsigned
    out = {}
    for name in ("Add", "Sub", "Mul", "And", "Or", "Xor", "Nand", "Nor", "Xnor", "Shl"):
        out[f"word.{name}"] = (getattr(word, name)(w, sc), [bv, bv], bv)
    for ns, tag in ((unsigned, "word.unsigned"), (signed, "word.signed")):
        for name in ("Div", "Rem", "Shr"):
            out[f"{tag}.{name}"] = (getattr(ns, name)(w, sc), [bv, bv], bv)
        for name in ("Lt", "Le", "Gt", "Ge"):
            out[f"{tag}.{name}"] = (getattr(ns, name)(w, sc), [bv, bv], boolean)
        out[f"{tag}.Extend"] = (ns.Extend(2, w, sc), [bv], Word(w + 2, sc))
    out["word.signed.Mod"] = (signed.Mod(w, sc), [bv, bv], bv)
    out["word.signed.Abs"] = (signed.Abs(w, sc), [bv], bv)
    for name in ("Neg", "Not"):
        out[f"word.{name}"] = (getattr(word, name)(w, sc), [bv], bv)
    out["Skip.word"] = (tensor.Skip(bv), [bv], bv)
    for name in ("Eq", "Ne"):
        out[f"word.{name}"] = (getattr(word, name)(w, sc), [bv, bv], boolean)
    out["word.Ite"] = (word.Ite(w, sc), [boolean, bv, bv], bv)
    out["word.Get"] = (word.Get(0, 2, w, [1, 3]), [Word(w, [1, 3])], bv)
    for name in ("RotateLeft", "RotateRight"):
        out[f"word.{name}"] = (getattr(word, name)(1, w, sc), [bv], bv)
    out["word.Extract"] = (word.Extract(2, 1, w, sc), [bv], Word(2, sc))
    out["word.Repeat"] = (word.Repeat(2, w, sc), [bv], Word(2 * w, sc))
    out["word.Concat"] = (word.Concat([w, 2], sc), [bv, Word(2, sc)], Word(w + 2, sc))
    for name in ("And", "Or", "Xor", "Implies", "Eq", "Ne"):
        out[f"bool.{name}"] = (getattr(tensor.bool, name)(sc), [boolean, boolean], boolean)
    out["bool.Not"] = (tensor.bool.Not(sc), [boolean], boolean)
    out["bool.Ite"] = (tensor.bool.Ite(sc), [boolean] * 3, boolean)
    out["bool.Get"] = (tensor.bool.Get(1, 0, [2, 2]), [Bool([2, 2])], boolean)
    out["int.Neg"] = (tensor.int.Neg([1, 2]), [Int([1, 2])], Int([1, 2]))
    out["real.Neg"] = (tensor.real.Neg(sc), [real], real)
    out["int.Mul"] = (tensor.int.Mul([2, 2]), [Int([2, 2])] * 2, Int([2, 2]))
    out["real.Mul"] = (tensor.real.Mul(sc), [real, real], real)
    out["int.Abs"] = (tensor.int.Abs([1, 2]), [Int([1, 2])], Int([1, 2]))
    out["real.Abs"] = (tensor.real.Abs(sc), [real], real)
    out["int.Get"] = (tensor.int.Get(1, 2, [2, 3]), [Int([2, 3])], integer)
    out["real.Get"] = (tensor.real.Get(0, 1, [1, 2]), [Real([1, 2])], real)
    out["cast.ToReal"] = (tensor.cast.ToReal([1, 2]), [Int([1, 2])], Real([1, 2]))
    out["cast.ToInt"] = (tensor.cast.ToInt(sc), [real], integer)
    out["cast.UbvToInt"] = (tensor.cast.UbvToInt(w, [1, 2]), [Word(w, [1, 2])], Int([1, 2]))
    for name in ("Div", "Mod"):
        out[f"int.{name}"] = (getattr(tensor.int, name)([1, 2]), [Int([1, 2])] * 2,
                              Int([1, 2]))
    for name in ("Ge", "Le", "Ne"):
        out[f"real.{name}"] = (getattr(tensor.real, name)(sc), [real, real], boolean)
    for ns, tag, s, one in ((tensor.int, "int", Int([1, 3]), integer),
                            (tensor.real, "real", Real([1, 3]), real)):
        for name in ("Max", "Min"):
            out[f"{tag}.{name}"] = (getattr(ns, name)([1, 3]), [s], one)
    out["real.Linear"] = (tensor.real.Linear(torch.tensor([[0.5, -2.0]]),
                                             torch.tensor([[0.25]])),
                          [Real([2, 1])], real)
    out["int.Scale"] = (tensor.int.Scale(torch.tensor([[3], [-2]])), [Int([2, 1])], Int([2, 1]))
    out["real.Scale"] = (tensor.real.Scale(torch.tensor([[0.5, 0.5]])), [Real([1, 2])],
                         Real([1, 2]))
    out["int.MatMul"] = (tensor.int.MatMul([[1, 2], [2, 1]]), [Int([1, 2]), Int([2, 1])],
                         integer)
    out["int.Ite.vector"] = (tensor.int.Ite([2, 1]),
                             [boolean, Int([2, 1]), Int([2, 1])], Int([2, 1]))
    out["bool.All"] = (tensor.bool.All([2, 1]), [Bool([2, 1])], boolean)
    return {f"ops/{k}": (lambda a=a: _one_op(*a)) for k, a in out.items()}


CORPUS = {**_corpus(), **_ops()}


# ── runs ────────────────────────────────────────────────────────────────────


def _draw(wire, rng: random.Random) -> torch.Tensor:
    """A random value for `wire`, shaped as `reactor.eval` keeps it."""
    dt, shape = wire.dtype, wire_shape(wire)
    if isinstance(dt, Bool):
        vals, kind = [rng.random() < 0.5 for _ in range(shape.total)], torch.bool
    elif isinstance(dt, Int):
        vals, kind = [rng.randint(-8, 8) for _ in range(shape.total)], torch.int64
    elif isinstance(dt, Real):
        # quarters, exact in the interpreter's float32
        vals = [rng.randint(-16, 16) / 4 for _ in range(shape.total)]
        kind = torch.float32
    elif isinstance(dt, Word):
        vals = [rng.randrange(1 << dt.width) for _ in range(shape.total)]
        kind = torch.int64
    else:                                   # pragma: no cover
        raise Skip(f"no random values for sort {dt}")
    return torch.tensor(vals, dtype=kind).reshape(shape.m, shape.n)


def runs(module, *, seed: int = 0, count: int = RUNS) -> list:
    """`count` runs of `module`: each a list of rounds `(inputs, state)`,
    where `inputs` maps each input variable to the value awaited that round
    and `state` lists the ctrl values after it, in `module.ctrl` order. The
    first runs of a seed are the same whatever `count` is."""
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
    a word as the natural number it holds."""
    if isinstance(wire.dtype, Word):
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
