"""An LTL formula over a net's variables as a module: the deterministic automaton that
reads the net's state round by round.

A formula is a function of the variables it is over, written like a module's block, and
`Automaton` makes the module of it:

    from zrth.omega import Automaton, G, F, U

    def bounded(n):
        return G(~(n >= 3))

    monitor = Automaton(bounded, extl=(queue.count,), theory=SPN)
    run = simulate(compose(net, monitor), horizon=50.0, seed=1)
    run.series(monitor.accepting[0])      # True while the queue has stayed below 3

The atoms are the Bool expressions over the variables, written as in any block (`n >= 3`,
`q == 1`, `e != 0`); `G`, `F`, `X` (the LTL next, not the sugar's), `U` and `~`, `&`, `|`
combine them and whole subformulas. The automaton reads one letter per round, the state of
that round: the atoms are evaluated on the variables' next values, in `init` too, so the
initial state is the first letter and the state after a round judges that round's marking.
Readers of the automaton's variables take them latched.

Spot translates the formula into a deterministic and complete automaton with state-based
acceptance (parity when no Büchi one exists). The module is of `theory`, the one the
variables' modules are of, which needs an `Enum` sort. Its state `q` is an `Enum` over
the automaton's states, and `accepting[i]` is "q is in acceptance set i"; `acceptance`
spells the condition over the sets (`Inf(0)`: set 0 visited infinitely often), which is
what a solver needs for a liveness formula, while for a safety formula `accepting[0]` is
the running verdict, false once the sink is reached.
"""

from .expr import Expr, Term, Wire, _emit, collecting, expr, ite
from .sort import Bool, Enum
from .sugar import Module
from .zrth import Var, X as base_X


class LTL:
    def __and__(self, other):
        return _Bin("&", self, _ltl(other))

    def __rand__(self, other):
        return _Bin("&", _ltl(other), self)

    def __or__(self, other):
        return _Bin("|", self, _ltl(other))

    def __ror__(self, other):
        return _Bin("|", _ltl(other), self)

    def __invert__(self):
        return _Un("!", self)

    def __bool__(self):
        raise TypeError("a formula has no truth value: combine with ~ & | and G F X U")


class Atom(LTL):
    def __init__(self, e: Expr):
        self.expr = e


class _Un(LTL):
    def __init__(self, op, sub):
        self.op, self.sub = op, sub


class _Bin(LTL):
    def __init__(self, op, left, right):
        self.op, self.left, self.right = op, left, right


def _ltl(p) -> LTL:
    if isinstance(p, LTL):
        return p
    if isinstance(p, Expr):
        if p.dtype != Bool([1, 1]):
            raise TypeError(f"an atom is a Bool expression, got {p.dtype}")
        return Atom(p)
    raise TypeError(f"a formula is built from Bool expressions and formulas, got {p!r}")


def G(p):
    return _Un("G", _ltl(p))


def F(p):
    return _Un("F", _ltl(p))


def X(p):
    return _Un("X", _ltl(p))


def U(p, q):
    return _Bin("U", _ltl(p), _ltl(q))


def _spot(phi: LTL, names: dict, made: dict) -> str:
    """The formula in Spot's syntax. `~`, `&` and `|` inside the atoms are Spot's own
    operators; what remains (the tests on the variables) are the atoms, named `p0, p1,
    ...` in order of first use, in `names` by wire id."""
    match phi:
        case Atom(expr=e):
            return _boolean(e.wire, names, made)
        case _Un(op=op, sub=sub):
            return f"{op}({_spot(sub, names, made)})"
        case _Bin(op=op, left=left, right=right):
            return f"({_spot(left, names, made)}) {op} ({_spot(right, names, made)})"
    raise TypeError(phi)


def _boolean(w, names: dict, made: dict) -> str:
    t = made.get(w.id)
    op = type(t.itype).__name__.rsplit("_", 1)[-1] if t is not None else None
    if op == "Not":
        return f"!({_boolean(t.read[0], names, made)})"
    if op in ("And", "Or"):
        left, right = (_boolean(r, names, made) for r in t.read)
        return f"({left}) {'&' if op == 'And' else '|'} ({right})"
    return names.setdefault(w.id, (f"p{len(names)}", w))[0]


def _replay(terms, names, by_id, theory) -> dict:
    """Build the atoms' terms again in the open block, reading the variables through the
    wires in `by_id` (their next wires) in place of the latched ones; returns the atoms'
    new expressions by name."""
    new = dict(by_id)
    for t in terms:
        write = [Wire(w.dtype) for w in t.write]
        term = Term(t.itype, write, [new.get(w.id, w) for w in t.read])
        _emit(term)
        for old, w in zip(t.write, write):
            new[old.id] = w
    return {name: expr(new.get(w.id, w), theory=theory) for name, w in names.values()}


def _cond(f, label: dict, theory) -> Expr:
    """A Spot Boolean formula over the atoms as a Bool expression."""
    import spot

    kind = f.kind()
    if kind == spot.op_ap:
        return label[f.ap_name()]
    if kind == spot.op_tt:
        return expr(True, theory=theory)
    if kind == spot.op_ff:
        return expr(False, theory=theory)
    if kind == spot.op_Not:
        return ~_cond(f[0], label, theory)
    if kind in (spot.op_And, spot.op_Or):
        parts = [_cond(c, label, theory) for c in f]
        out = parts[0]
        for p in parts[1:]:
            out = out & p if kind == spot.op_And else out | p
        return out
    raise ValueError(f"unexpected operator in an edge label: {f}")


class Automaton(Module):
    """The automaton of the formula `formula(*variables)` returns, over the variables in
    `extl`, a module of `theory`: `q` its state, `accepting[i]` whether `q` is in
    acceptance set i, `acceptance` the condition over the sets, `text` the formula as
    Spot reads it."""

    def __new__(cls, formula, *, extl=(), theory=None, name=None):
        import spot

        if theory is None:
            raise TypeError(f"{cls.__name__}: `theory` is a required constructor arg")
        if not callable(formula):
            raise TypeError(f"a formula is a function of its variables, got {formula!r}")
        extl = tuple(extl)
        for v in extl:
            if not isinstance(v, Var):
                raise TypeError(f"a formula is over variables, got {v!r}")
        if len(set(extl)) != len(extl):
            raise ValueError("a variable appears twice in extl")

        with collecting() as terms:
            phi = _ltl(formula(*(expr(v, theory=theory) for v in extl)))
        names = {}
        text = _spot(phi, names, {w.id: t for t in terms for w in t.write})

        aut = spot.translate(spot.formula(text), "det", "generic", "complete", "sbacc")
        if not aut.is_deterministic() or not spot.is_complete(aut):
            raise RuntimeError(f"Spot gave no deterministic complete automaton for {text}")
        n = aut.num_states()
        sets = aut.acc().num_sets()
        edges = {
            s: [(spot.bdd_to_formula(e.cond, aut.get_dict()), e.dst) for e in aut.out(s)]
            for s in range(n)
        }
        members = [
            frozenset(s for s in range(n) if i in aut.state_acc_sets(s).sets())
            for i in range(sets)
        ]
        q = Var(Enum(n))
        accepting = tuple(Var(Bool([1, 1])) for _ in range(sets))

        def literal(s):
            return expr(s, theory=theory, sort=Enum(n))

        def step(s, label):
            out = edges[s]
            r = literal(out[-1][1])
            for f, dst in reversed(out[:-1]):
                r = ite(_cond(f, label, theory), literal(dst), r)
            return r

        def within(new_q, states):
            if len(states) == n:
                return expr(True, theory=theory)
            r = expr(False, theory=theory)
            for s in sorted(states):
                r = r | (new_q == s)
            return r

        def read(variables):
            return _replay(terms, names, {v.id: base_X(v) for v in variables}, theory)

        def init(self, *variables):
            new_q = step(aut.get_init_state_number(), read(extl))
            return new_q, *[within(new_q, m) for m in members]

        def next(self, q, *rest):
            label = read(extl)
            new_q = step(n - 1, label)
            for s in reversed(range(n - 1)):
                new_q = ite(q == s, step(s, label), new_q)
            return new_q, *[within(new_q, m) for m in members]

        self = Module.__new__(
            type(cls.__name__, (cls,), {"init": init, "next": next}),
            theory=theory,
            ctrl=(q, *accepting),
            extl=extl,
        )
        self.name, self.text, self.q, self.accepting = name, text, q, accepting
        self.acceptance, self.spot = str(aut.get_acceptance()), aut
        return self

    def __repr__(self):
        return f"Automaton({self.name or self.text}, acceptance={self.acceptance})"
