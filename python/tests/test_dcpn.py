"""Tests for LIA-colored Petri nets (``reactor.dcpn``).

Covers:

- place and color declarations, and ``obj`` instantiation;
- token modules: initial state, resident updates, and same-round jumps
  depending on the ``transition_round`` and ``in_transition`` flags;
- the n-ary Boolean helpers ``_all`` and ``_any``;
- validation of net specifications.
"""

from dataclasses import FrozenInstanceError, fields
from itertools import product
from types import SimpleNamespace
from typing import get_type_hints

import pytest
import torch

from reactor import Bool, Int, LIA, LRA, Module, NonLinearError, Real, Var, X
from reactor.dcpn import DCPN, GuardedTransition, ImmediateTransition, Place, TokenModule, obj
from reactor.dcpn.lowering import _all, _any
from reactor.eval import execute_init, execute_update
from reactor.expr import collecting, expr
from reactor.reactor import Wire
from reactor.sugar import Module as SugarModule


INT = Int([1, 1])
BOOL = Bool([1, 1])

# Evaluation state for `reactor.eval.execute_init` / `execute_update`: latched
# values are keyed by their `Var`, updated values and intermediate results by
# their `Wire`.
type State = dict[Var | Wire, torch.Tensor]


def _init(module: Module) -> State:
    state: State = {}
    execute_init(state, module.atoms)
    return state


def _latch_updated_ctrl(module: Module, state: State) -> State:
    """Use updated controlled values as latched values for the next round.

    ``state`` is the completed round's evaluation environment, mapping wires
    to tensors; it may contain both latched and updated values and intermediate
    results. Return a new dictionary mapping each controlled variable to its
    updated value at ``X(var)``.
    """
    return {var: state[X(var)] for var in module.ctrl}


def _inject_transition_state(
    token: TokenModule, state: State, *,
    participating_in_transition=False, transition_round=None, new_place=0, new_colors=None,
) -> State:
    """Populate transition values in ``state`` and return the same dictionary.

    Set the updated ``in_transition`` to ``participating_in_transition``, and
    the updated ``transition_round`` to ``transition_round``; if omitted, it
    defaults to ``participating_in_transition`` (a participating token implies a
    transition round). Set the updated ``new_place`` and the ``new_colors``;
    omitted color values default to zero or False. Preserve all unrelated
    entries in ``state``.
    """
    if transition_round is None:
        transition_round = participating_in_transition
    # a participating token implies a transition round
    assert transition_round or not participating_in_transition
    state.update({
        X(token.transition_round): torch.tensor([[transition_round]]),
        X(token.in_transition): torch.tensor([[participating_in_transition]]),
        X(token.new_place): torch.tensor([[new_place]]),
    })
    new_colors = {} if new_colors is None else new_colors
    for place, record in token.new_colors.items():
        for field in fields(record):
            var = getattr(record, field.name)
            dtype = torch.bool if isinstance(var.dtype, Bool) else torch.int64
            match var.dtype:
                case Int(shape) | Bool(shape):
                    value = new_colors.get((place, field.name), torch.zeros(list(shape), dtype=dtype))
            state[X(var)] = torch.as_tensor(value, dtype=dtype).reshape(list(shape))
    return state


def _step(
    token: TokenModule, previous_round_state: State, *,
    external: State | None = None, **transition_state,
) -> State:
    """Evaluate one round starting from the previous round's updated values.

    Latch the previous round's updated controlled values, populate transition state
    according to ``_inject_transition_state`` and latch ``external`` wires.
    Then run `update` and return the new environment.

    Return a new evaluation environment containing latched and updated values.
    """
    state = _latch_updated_ctrl(token, previous_round_state)
    state = _inject_transition_state(token, state, **transition_state)
    state.update({} if external is None else external)
    execute_update(state, token.atoms)
    return state


def _net(*places, transitions=()):
    return DCPN(theory=LIA, places=places, transitions=transitions)


# Declarations ----------------------------------------------------------------


def test_place_color_declarations():
    class BasePlace(Place):
        x = Var(INT)
        ready = Var(BOOL)
        increment = 2

        def update(self, x, ready):
            return x + self.increment, ~ready

    class ChildPlace(BasePlace):
        x = Var(INT)

    # controlled variables are present
    assert ChildPlace.ctrl == (ChildPlace.x, BasePlace.ready)
    # `Color` struct has the expected fields
    assert [f.name for f in fields(ChildPlace.Color)] == ["x", "ready"]

    # test `Color` construction
    color = ChildPlace.Color(x=3, ready=True)
    with pytest.raises(FrozenInstanceError):
        color.x = 4
    with pytest.raises(TypeError):
        ChildPlace.Color()

    # Construct one token and do init -> transition -> update
    child = ChildPlace()
    token, = _net(child).to_token_modules(token_capacity=1)
    state = _step(token, _init(token), participating_in_transition=True, new_place=1)
    state = _step(token, state)
    # `x` was incremented once by 2, and `ready` was flipped
    assert state[X(token.colors[child].x)].item() == 2
    assert state[X(token.colors[child].ready)].item() is True


def test_multiple_inheritance_follows_mro():
    class Base(Place):
        x = Var(INT)

    class Left(Base):
        x = Var(BOOL)
        left = Var(INT)

    class Right(Base):
        right = Var(INT)

    class Child(Left, Right):
        last = Var(INT)

    assert Child.ctrl == (Left.x, Right.right, Left.left, Child.last)
    assert [f.name for f in fields(Child.Color)] == ["x", "right", "left", "last"]


def test_obj_and_explicit_instantiation():
    external = Var(INT)

    @obj
    class Empty(Place):
        pass

    @obj(extl=(external,))
    class Colored(Place):
        x = Var(INT)

    # implicit instantiation of Empty and Colored
    assert isinstance(Empty, Place)
    assert Empty.extl == ()
    assert isinstance(Colored, Place)
    assert Colored.extl == (external,)

    # explicit instantiation is still possible
    ColoredClass = type(Colored)
    assert ColoredClass() is not Colored
    assert ColoredClass(extl=(external,)).extl == (external,)


def test_net_contains_correct_places_and_transitions():
    # [Source] --> (Work) ==(2x)=> [Complete (guarded)] ==-> (Finished)
    #                 ^-----------------------------------|
    @obj
    class Work(Place):
        progress = Var(INT)

    @obj
    class Finished(Place):
        pass

    @obj
    class Source(ImmediateTransition):
        outputs = (Work,)

        def produce(self):
            return (Work.Color(progress=0),)

    @obj
    class Complete(GuardedTransition):
        # `Complete` fires when two tokens reach the progress threshold:
        # One token is promoted to `Finished` with an empty color,
        # the other token returns to `Work` with the combined progress.
        inputs = (Work, Work)
        outputs = (Finished, Work)
        threshold = 3

        def guard(self, a, b):
            assert isinstance(a, Work.Color)
            assert isinstance(b, Work.Color)
            assert a.progress.wire != b.progress.wire
            return (a.progress >= self.threshold) & (b.progress >= self.threshold)

        def produce(self, a, b):
            return Finished.Color(), Work.Color(progress=a.progress + b.progress)

    net = _net(Work, Finished, transitions=(Source, Complete))
    assert net.transitions == (Source, Complete)
    assert net.place_ids == {Work: 1, Finished: 2}
    tokens = net.to_token_modules(token_capacity=2)
    assert len(tokens) == 2
    # all produced modules are TokenModules; they are open due to the event externals
    assert all(isinstance(t, TokenModule) and t.open() for t in tokens)


def test_token_modules_use_fresh_vars():
    @obj
    class Work(Place):
        x = Var(INT)
        ready = Var(BOOL)

    net = _net(Work)
    a, b = net.to_token_modules(token_capacity=2)
    c, = net.to_token_modules(token_capacity=1)

    # `to_token_modules` produces fresh `Var`s
    assert len({a.place, b.place, c.place}) == 3
    assert len({Work.x, a.colors[Work].x, b.colors[Work].x, c.colors[Work].x}) == 4
    assert len({a.in_transition, b.in_transition, c.in_transition}) == 3
    # tokens of one `to_token_modules` call share `transition_round`
    assert a.transition_round is b.transition_round
    assert a.transition_round is not c.transition_round

    for token in (a, b, c):
        # all produced modules are open due to the event externals
        assert token.open()
        # check `ctrl` and `extl`
        assert set(token.ctrl) == {token.place, token.colors[Work].x, token.colors[Work].ready}
        assert set(token.extl) == {
            token.transition_round, token.in_transition, token.new_place,
            token.new_colors[Work].x, token.new_colors[Work].ready,
        }

    # their composition is still open
    assert Module(a, b, c).open()


def test_transition_inputs_are_awaited():
    @obj
    class Work(Place):
        x = Var(INT)
        ready = Var(BOOL)

    token, = _net(Work).to_token_modules(token_capacity=1)
    awaited = set(token.atoms[0].wait)
    latched = set(token.atoms[0].read)

    # The token uses only the next values of the transition flags and payloads,
    # set in the same round; it never reads their stale latched values.
    event_payload = (token.new_place, token.new_colors[Work].x, token.new_colors[Work].ready)
    for var in (token.transition_round, token.in_transition, *event_payload):
        assert var in awaited
        assert var not in latched


# Token state and same-round transition events -----------------------------------------


def test_init_state():
    @obj
    class Work(Place):
        x = Var(INT)
        ready = Var(BOOL)

    net = _net(Work)
    a, b = net.to_token_modules(token_capacity=2)
    c, = net.to_token_modules(token_capacity=1)

    for token in (a, b, c):
        # all vars are initially zeroed
        state = _init(token)
        assert state[X(token.place)].item() == 0
        assert state[X(token.colors[Work].x)].item() == 0
        assert state[X(token.colors[Work].ready)].item() is False


def test_production_update_removal():
    @obj
    class Work(Place):
        x = Var(INT)
        ready = Var(BOOL)

        def update(self, x, ready):
            return x + 1, ~ready

    @obj
    class Other(Place):
        y = Var(INT)

        def update(self, y):
            return (y + 10,)

    net = _net(Work, Other)
    token, = net.to_token_modules(token_capacity=1)
    x, ready, y = token.colors[Work].x, token.colors[Work].ready, token.colors[Other].y

    # initially all zeroed, token is absent
    state = _step(token, _init(token))
    assert state[X(token.place)].item() == 0    # absent
    assert state[X(x)].item() == 0
    assert state[X(y)].item() == 0
    assert state[X(ready)].item() == False

    # token is absent -> no evolution
    state = _step(token, state)
    assert state[X(token.place)].item() == 0    # absent
    assert state[X(x)].item() == 0
    assert state[X(y)].item() == 0
    assert state[X(ready)].item() == False

    # token becomes present in `Work` with color `(x=5, ready=True)`
    state = _step(token, state, participating_in_transition=True, new_place=1,
                  new_colors={(Work, "x"): 5, (Work, "ready"): True})
    assert state[X(token.place)].item() == 1
    assert state[X(x)].item() == 5
    assert state[X(ready)].item() == True
    assert state[X(y)].item() == 0

    # evolution in `Work`; `Other` color doesn't evolve
    state = _step(token, state)
    assert state[X(token.place)].item() == 1
    assert state[X(x)].item() == 6
    assert state[X(ready)].item() is False
    assert state[X(y)].item() == 0

    # jump token to `Other` with color `(y=20)`
    state = _step(token, state, participating_in_transition=True, new_place=2, new_colors={(Other, "y"): 20})
    assert state[X(token.place)].item() == 2
    assert state[X(x)].item() == 6
    assert state[X(ready)].item() is False
    assert state[X(y)].item() == 20

    # evolution in `Other`
    state = _step(token, state)
    assert state[X(token.place)].item() == 2
    assert state[X(x)].item() == 6
    assert state[X(ready)].item() is False
    assert state[X(y)].item() == 30

    # remove token
    state = _step(token, state, participating_in_transition=True, new_place=0)
    assert state[X(token.place)].item() == 0
    assert state[X(x)].item() == 6
    assert state[X(ready)].item() is False
    assert state[X(y)].item() == 30

    # evolve absent token stutters
    state = _step(token, state)
    assert state[X(token.place)].item() == 0
    assert state[X(x)].item() == 6
    assert state[X(ready)].item() is False
    assert state[X(y)].item() == 30

    # produce token in `Work`
    state = _step(token, state, participating_in_transition=True, new_place=1, new_colors={(Work, "x"): 90})
    assert state[X(token.place)].item() == 1
    assert state[X(x)].item() == 90  # production replaces stale colours
    assert state[X(ready)].item() is False
    assert state[X(y)].item() == 30

    # same-place transition also suppresses `update`
    state = _step(token, state, participating_in_transition=True, new_place=1, new_colors={(Work, "x"): 7})
    assert state[X(token.place)].item() == 1
    assert state[X(x)].item() == 7  # same-place transition events also suppress updates
    assert state[X(ready)].item() is False
    assert state[X(y)].item() == 30


def test_missing_update_stutters():
    @obj
    class Hold(Place):
        x = Var(INT)

    @obj
    class Empty(Place):
        pass

    token, = _net(Hold, Empty).to_token_modules(token_capacity=1)
    state = _step(token, _init(token), participating_in_transition=True, new_place=1, new_colors={(Hold, "x"): 42})

    state = _step(token, state)
    assert state[X(token.place)].item() == 1
    assert state[X(token.colors[Hold].x)].item() == 42
    assert fields(token.colors[Empty]) == ()

    state = _step(token, state, participating_in_transition=True, new_place=2)
    assert state[X(token.place)].item() == 2
    assert state[X(token.colors[Hold].x)].item() == 42
    assert fields(token.colors[Empty]) == ()


def test_matrix_colors():
    amount = Var(Int([1, 2]))

    @obj(extl=(amount,))
    class Moving(Place):
        position = Var(Int([1, 2]))

        def update(self, position, amount):
            return (position + amount,)

    token, = _net(Moving).to_token_modules(token_capacity=1)
    assert amount in token.extl
    assert amount in token.atoms[0].read

    state = _step(token, _init(token), participating_in_transition=True, new_place=1,
                  new_colors={(Moving, "position"): [[1, 2]]},
                  external={amount: torch.tensor([[3, 4]])})
    assert state[X(token.colors[Moving].position)].tolist() == [[1, 2]]

    state = _step(token, state, external={amount: torch.tensor([[3, 4]])})
    assert state[X(token.colors[Moving].position)].tolist() == [[4, 6]]


def test_in_transition_flag():
    @obj
    class Work(Place):
        x = Var(INT)

        def update(self, x):
            return (x + 1,)

    token, = _net(Work).to_token_modules(token_capacity=1)
    state = _init(token)

    # The token jumps to the payload when `in_transition` holds.
    state = _step(token, state, participating_in_transition=True, new_place=1, new_colors={(Work, "x"): 10})
    assert state[X(token.place)].item() == 1
    assert state[X(token.colors[Work].x)].item() == 10

    # The token can participate in consecutive transition rounds; each round
    # jumps it to that round's payload.
    state = _step(token, state, participating_in_transition=True, new_place=1, new_colors={(Work, "x"): 20})
    assert state[X(token.place)].item() == 1
    assert state[X(token.colors[Work].x)].item() == 20

    # When `in_transition` does not hold, payloads are ignored and the token
    # does a normal update round.
    state = _step(token, state, new_place=0, new_colors={(Work, "x"): 99})
    assert state[X(token.place)].item() == 1
    assert state[X(token.colors[Work].x)].item() == 21

    # A transition round in which the token does not participate: it stutters.
    state = _step(token, state, transition_round=True, new_place=0, new_colors={(Work, "x"): 99})
    assert state[X(token.place)].item() == 1
    assert state[X(token.colors[Work].x)].item() == 21


def test_transition_event_controller_composes():
    @obj
    class Work(Place):
        x = Var(INT)

        def update(self, x):
            return (x + 1,)

    token, = _net(Work).to_token_modules(token_capacity=1)

    # The token by itself is an open system.
    assert token.open()

    class TransitionController(SugarModule):
        def init(self):
            return False, False, 0, 0

        def update(self, transition_round, in_transition, new_place, payload):
            # Step the payload by 10, unlike the resident update's +1, so that
            # each round's color shows that the token jumped.
            return True, True, 1, payload + 10

    controller = TransitionController(
        theory=LIA,
        ctrl=(token.transition_round, token.in_transition, token.new_place, token.new_colors[Work].x),
    )
    # Compose in reverse dependency order: the token awaits the controllers's outputs.
    system = Module(token, controller)

    # Plugging the transition controller results in a closed system.
    assert system.closed()
    assert set(token.atoms[0].wait) == set(controller.ctrl)

    # Init the composed system.
    state = _init(system)
    assert state[X(token.place)].item() == 0
    assert state[X(token.colors[Work].x)].item() == 0

    for step in range(1, 4):
        state = _latch_updated_ctrl(system, state)
        execute_update(state, system.atoms)
        assert state[X(token.in_transition)].item() is True
        assert state[X(token.place)].item() == 1
        assert state[X(token.colors[Work].x)].item() == 10 * step


def test_empty_net_and_capacity():
    net = _net()
    token, = net.to_token_modules(token_capacity=1)
    assert _init(token)[X(token.place)].item() == 0
    for capacity in (-1, 0, True, 1.5):
        with pytest.raises(ValueError, match="token_capacity"):
            net.to_token_modules(token_capacity=capacity)


# Boolean folding helpers -----------------------------------------------------


def _evaluate(terms, result, assignment):
    """Evaluate collected ``terms`` under ``assignment`` and return ``result``'s value."""
    state: State = {var: torch.tensor([[value]]) for var, value in assignment.items()}
    execute_update(state, [SimpleNamespace(update=terms)])
    return state[result.wire].item()


def test_all_empty_is_true():
    with collecting() as terms:
        result = _all([])
    assert result.dtype == BOOL
    assert result.theory is LIA
    assert _evaluate(terms, result, {}) is True


def test_any_empty_is_false():
    with collecting() as terms:
        result = _any([])
    assert result.dtype == BOOL
    assert result.theory is LIA
    assert _evaluate(terms, result, {}) is False


def test_all_single_returns_operand():
    with collecting() as terms:
        operand = expr(Var(BOOL), theory=LIA)
        assert _all([operand]) is operand
    assert not any(isinstance(term.itype, LIA.And) for term in terms)


def test_any_single_returns_operand():
    with collecting() as terms:
        operand = expr(Var(BOOL), theory=LIA)
        assert _any([operand]) is operand
    assert not any(isinstance(term.itype, LIA.Or) for term in terms)


def test_all_truth_table():
    a, b, c = Var(BOOL), Var(BOOL), Var(BOOL)
    with collecting() as terms:
        # a generator, as used by the transition controller
        result = _all(expr(var, theory=LIA) for var in (a, b, c))
    assert result.dtype == BOOL
    for values in product([False, True], repeat=3):
        assert _evaluate(terms, result, dict(zip((a, b, c), values))) is all(values)


def test_any_truth_table():
    a, b, c = Var(BOOL), Var(BOOL), Var(BOOL)
    with collecting() as terms:
        # a generator, as used by the transition controller
        result = _any(expr(var, theory=LIA) for var in (a, b, c))
    assert result.dtype == BOOL
    for values in product([False, True], repeat=3):
        assert _evaluate(terms, result, dict(zip((a, b, c), values))) is any(values)


# Validation ------------------------------------------------------------------


def test_unsupported_theory_and_sorts():
    with pytest.raises(NotImplementedError, match="only LIA"):
        DCPN(theory=LRA)

    class RealPlace(Place):
        x = Var(Real([1, 1]))

    # Declarations do not choose a theory; sort support is a net-level check.
    real_place = RealPlace()
    real_input = Place(extl=(Var(Real([1, 1])),))
    with pytest.raises(TypeError, match="Int or Bool"):
        _net(real_place)
    with pytest.raises(TypeError, match="Int or Bool"):
        _net(real_input)


def test_place_constructor_validates_local_inputs_and_methods():
    class Flow(Place):
        def flow(self):
            return ()

    class BadUpdate(Place):
        update = 1

    with pytest.raises(TypeError, match="external inputs must be Vars"):
        Place(extl=(1,))
    with pytest.raises(NotImplementedError, match="flow"):
        Flow()
    with pytest.raises(TypeError, match="update must be callable"):
        BadUpdate()


def test_color_declarations_are_validated_at_class_definition():
    with pytest.raises(TypeError, match="reserved color name"):
        class Reserved(Place):
            ctrl = Var(INT)

    class Base(Place):
        x = Var(INT)

    with pytest.raises(TypeError, match="cannot override an inherited color field"):
        class BadOverride(Base):
            x = 0


def test_symbolic_methods_run_only_when_constructing_the_net():
    calls = []

    class Work(Place):
        x = Var(INT)
        increment = 1

        def advance(self, x):
            calls.append("advance")
            return x + self.increment

        def update(self, x):
            calls.append("update")
            return (self.advance(x),)

    work = Work()

    class Complete(GuardedTransition):
        inputs = (work,)
        outputs = (work,)

        def guard(self, token):
            calls.append("guard")
            return token.x >= 0

        def produce(self, token):
            calls.append("produce")
            return (work.Color(x=token.x + 1),)

    transition = Complete()

    assert calls == []
    _net(work, transitions=(transition,))
    assert calls == ["update", "advance", "guard", "produce"]


def test_update_signature_arity_and_nonlinearity():
    class WrongSignature(Place):
        x = Var(INT)

        def update(self):
            return 0

    class WrongArity(Place):
        x = Var(INT)

        def update(self, x):
            return x, x

    class Nonlinear(Place):
        x = Var(INT)

        def update(self, x):
            return x * x

    with pytest.raises(TypeError):
        _net(WrongSignature())
    with pytest.raises(ValueError, match="return 1 values"):
        _net(WrongArity())
    with pytest.raises(NonLinearError):
        _net(Nonlinear())


@pytest.mark.parametrize("value", [True, 1.5, [[1, 2]]])
def test_update_requires_exact_sort_and_shape(value):
    class Bad(Place):
        x = Var(INT)

        def update(self, x):
            return (value,)

    with pytest.raises(TypeError):
        _net(Bad())


def test_net_membership():
    p, other = Place(), Place()
    with pytest.raises(TypeError, match="Place objects"):
        _net(Place)
    with pytest.raises(ValueError, match="twice"):
        _net(p, p)

    class Foreign(ImmediateTransition):
        inputs = (other,)

        def produce(self, token):
            return ()

    transition = Foreign()
    with pytest.raises(ValueError, match="endpoints"):
        _net(p, transitions=(transition,))
    with pytest.raises(TypeError, match="transition objects"):
        _net(p, transitions=(object(),))


def test_transition_constructor_validates_endpoints_and_required_methods():
    p = Place()

    class BadEndpoint(ImmediateTransition):
        inputs = (object(),)

    class MissingProduce(ImmediateTransition):
        outputs = (p,)

    class BadProduce(ImmediateTransition):
        produce = 1

    with pytest.raises(TypeError, match="Place objects"):
        BadEndpoint()
    with pytest.raises(TypeError, match="produce"):
        MissingProduce()
    with pytest.raises(TypeError, match="produce must be callable"):
        BadProduce()
    with pytest.raises(TypeError, match="guard"):
        GuardedTransition()


def test_sink_transitions_without_produce():
    p = Place()

    class Sink(ImmediateTransition):
        inputs = (p,)

    class GuardedSink(GuardedTransition):
        inputs = (p,)

        def guard(self, token):
            return True

    sink, guarded = Sink(), GuardedSink()
    net = _net(p, transitions=(sink, guarded))
    assert net.transitions == (sink, guarded)


def test_transition_signature_and_guard_sort():
    p = Place()

    class WrongSignature(ImmediateTransition):
        inputs = (p,)

        def produce(self):
            return ()

    class WrongGuard(GuardedTransition):
        def guard(self):
            return 1

        def produce(self):
            return ()

    with pytest.raises(TypeError):
        _net(p, transitions=(WrongSignature(),))
    with pytest.raises(TypeError):
        _net(p, transitions=(WrongGuard(),))


def test_produce_alignment_and_destination_color():
    p, other = Place(), Place()

    class WrongLength(ImmediateTransition):
        outputs = (p,)

        def produce(self):
            return ()

    # Different place classes have different Color types; instances of a shared
    # class intentionally share a colour schema.
    class Other(Place):
        pass

    class WrongColor(ImmediateTransition):
        outputs = (p,)

        def produce(self):
            return (Other.Color(),)

    with pytest.raises(ValueError, match="aligned"):
        _net(p, transitions=(WrongLength(),))
    with pytest.raises(TypeError, match="Color"):
        _net(p, other, transitions=(WrongColor(),))


@pytest.mark.parametrize("colored", [False, True])
def test_none_output_is_rejected(colored):
    class Colored(Place):
        x = Var(INT)

    place = Colored() if colored else Place()

    class Bad(ImmediateTransition):
        outputs = (place,)

        def produce(self):
            return (None,)

    with pytest.raises(TypeError, match="produce must return"):
        _net(place, transitions=(Bad(),))


def test_unsupported_transition_features():
    class UnsupportedRate(ImmediateTransition):
        def produce(self):
            return ()

        def rate(self):
            return 1

    with pytest.raises(NotImplementedError):
        UnsupportedRate()


@pytest.mark.parametrize("value", [True, 2.5, [[1, 2]]])
def test_produced_fields_require_exact_sort_and_shape(value):
    @obj
    class Work(Place):
        x = Var(INT)

    @obj
    class Bad(ImmediateTransition):
        outputs = (Work,)

        def produce(self):
            return (Work.Color(x=value),)

    with pytest.raises(TypeError):
        _net(Work, transitions=(Bad,))


def test_guard_must_be_scalar_boolean():
    @obj
    class Work(Place):
        ready = Var(Bool([1, 2]))

    @obj
    class Bad(GuardedTransition):
        inputs = (Work,)

        def guard(self, token):
            return token.ready

        def produce(self, token):
            return ()

    with pytest.raises(TypeError, match="expected LIA value"):
        _net(Work, transitions=(Bad,))


def test_symbolic_update_sort_must_match_declaration():
    class Bad(Place):
        x = Var(INT)

        def update(self, x):
            return (x > 0,)

    with pytest.raises(TypeError, match="expected LIA value"):
        _net(Bad())
