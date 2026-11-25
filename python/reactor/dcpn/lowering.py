"""Lower LIA Petri-net specifications to bounded reactive modules.

Each TokenModule allocates fresh state for one token slot: an integer place ID
and a separate color vector for every place. Tokens start absent (place ID zero),
with integer colors zero and Boolean colors False. Ordinary updates evolve only
the resident place's colors; inactive place colors stutter.

A shared external Boolean transition_round signals that some transition fires in
this round; non-pariticipating tokens do not perform the resident place update
(i.e., they must stutter).
The per-token external Boolean in_transition signals that a token participates
in the current round's firing and must change its state within the same round
(it "jumps"). Both flags are levels, not toggles: the controller must assign
them in every round.
Participating tokens await their new place and color by reading the new_place
and new_colors payloads' next wires, and must apply them in the same round
instead of performing a resident place update.

A TransitionControllerModule implements the net's transition semantics: in
each round, at most one (transition, binding) pair fires, chosen in transition
definition order and then in lexicographic slot order. As resident token
updates happen only in rounds where no transition fires, transitions are
urgent. The controller's controlled variables are exactly the token modules'
external transition variables, so composing all token modules of a net with
their transition controller yields a closed system.
"""

from dataclasses import fields
from itertools import permutations
from typing import NamedTuple

import torch

from ..reactor import LIA, Module, Term, Var, X
from ..sort import Bool, Int, tensor_for
from ..expr import Expr, collecting, expr, ite


BOOL = Bool([1, 1])


def _value(value, sort):
    """Wrap ``value`` in a LIA expression of ``sort``.

    An existing ``Expr`` is checked and returned unchanged.
    A ``Var`` is wrapped in an expression that reads it.
    A literal or tensor becomes a constant expression.

    Requires an active collector when constructing constant expressions.
    """
    if not isinstance(value, Expr):
        if not isinstance(value, Var):
            value = torch.as_tensor(value)
            if isinstance(sort, Int) and value.dtype == torch.bool:
                raise TypeError("an integer color requires integers, not Booleans")
            value = tensor_for(value, sort)
        value = expr(value, theory=LIA, sort=sort)
    if value.theory is not LIA or value.dtype != sort:
        raise TypeError(f"expected LIA value of sort {sort}, got {value.theory}: {value.dtype}")
    return value


def _update(place, colors, external):
    """Return symbolic next-color expressions for a place.

    Call ``place.update`` with ``colors`` followed by ``external``.
    Its results must align with ``place.ctrl``;

    If ``update`` is omitted, return the current colors unchanged to represent
    stuttering.

    Requires an active collector for expressions built by the update method
    or by ``_value``.
    """
    method = getattr(place, "update", None)

    # if `update` is not present, state is stuttered
    if method is None:
        return colors

    # symbolically call `update`
    values = method(*colors, *external)
    if len(values) != len(place.ctrl):
        raise ValueError(
            f"{type(place).__name__}.update must return {len(place.ctrl)} values, "
            f"got {len(values)}"
        )
    return tuple(_value(value, var.dtype) for value, var in zip(values, place.ctrl))


def _allocate_color_records(places):
    """Allocate a fresh color-variable record for each place.

    Return a dictionary mapping each place to its ``Color`` record. Each field
    contains a new ``Var`` with the same sort as its class-level declaration.
    """
    return {
        place: place.Color(*(Var(var.dtype) for var in place.ctrl))
        for place in places
    }


def _color_record_to_tuple(color):
    return tuple(getattr(color, field.name) for field in fields(color))


def _zero_init(variables):
    """Return init terms setting each of ``variables`` to zero or False.

    Requires an active collector.
    """
    terms = []
    for var in variables:
        match var.dtype:
            case Int(shape):
                dtype = torch.int64
            case Bool(shape):
                dtype = torch.bool
            case _:
                raise TypeError(
                    f"LIA colors and external inputs must be Int or Bool, got {var.dtype}"
                )
        zero = expr(torch.zeros(list(shape), dtype=dtype), theory=LIA, sort=var.dtype)
        terms.append(Term(LIA.Id(), [X(var)], [zero.wire]))
    return terms


def _all(conditions):
    """Conjoin Boolean expressions; the empty conjunction is True."""
    result = None
    for condition in conditions:
        result = condition if result is None else result & condition
    return _value(True, BOOL) if result is None else result


def _any(conditions):
    """Disjoin Boolean expressions; the empty disjunction is False."""
    result = None
    for condition in conditions:
        result = condition if result is None else result | condition
    return _value(False, BOOL) if result is None else result


class _Write(NamedTuple):
    """A possible write of a transition controller to one token slot."""

    condition: Expr  # condition for the write
    place_id: int    # the token's new place ID (0: the token is consumed)
    color: object    # the new place's Color record, or None if consumed


class TokenModule(Module):
    """A reactive module owning one token's ``place`` and per-place ``colors``.

    The token is driven by a transition controller through four external
    variables. It awaits the controller's variables, so a firing takes
    effect immediately.

    ``transition_round`` is shared by all tokens of a net and indicates if
    any transition fires this round. If none does, the token runs its resident
    update as usual.

    ``in_transition`` indicates if this particular token participates in the
    firing. If a transition fires but the token is not part of it, the token
    keeps its state unchanged.

    Both flags are plain Booleans, not toggles, so the controller has to set
    them in every round.

    ``new_place`` and ``new_colors`` hold the token's new state when it takes
    part in a firing. A new place of zero marks the token as absent from the net.
    """

    def __new__(cls, net, transition_round):
        """Create a token module for one slot of ``net``.

        Args:
            net: The ``DCPN`` whose places the token can reside in.
            transition_round: External Boolean ``Var``; its next value holds
                iff some transition fires in this round, including transitions
                that do not involve this token. In such a round, an uninvolved
                token stutters instead of performing its resident place update
                (an involved token jumps to its new state), so time does not
                advance while transitions fire. Thus, a token cannot evolve
                past a state that enables a transition. When composed with
                other tokens and a transition controller, all tokens of a net
                must share the same ``Var``.
        """
        # current place and color
        current_place = Var(Int([1, 1]))
        current_colors = _allocate_color_records(net.places)
        # transition trigger and payload
        in_transition = Var(BOOL)
        new_place = Var(Int([1, 1]))
        new_colors = _allocate_color_records(net.places)

        # Construct the module's controlled and external variables
        controlled = [current_place]
        external = [transition_round, in_transition, new_place]
        for place in net.places:
            controlled.extend(_color_record_to_tuple(current_colors[place]))
            external.extend(_color_record_to_tuple(new_colors[place]))
            external.extend(place.extl)
        variables = controlled + external

        # Initialize the token as absent, with zeroed per-place color components:
        # (current_place', *current_colors') := 0
        with collecting() as init:
            init.extend(_zero_init(controlled))

        with collecting() as update:
            expr_Xtransition_round = expr(X(transition_round), theory=LIA)
            expr_Xin_transition = expr(X(in_transition), theory=LIA)

            # I. The token's place: a participating token moves to `new_place`;
            #    otherwise its place is unchanged.
            # current_place' := ite(in_transition', new_place', current_place)
            expr_current_place = expr(current_place, theory=LIA)
            expr_Xnew_place = expr(X(new_place), theory=LIA)
            next_place = ite(expr_Xin_transition, expr_Xnew_place, expr_current_place)
            update.append(Term(LIA.Id(), [X(current_place)], [next_place.wire]))

            # II. The token's colors, one color vector per place.
            for place in net.places:
                vars = _color_record_to_tuple(current_colors[place])

                if not vars:
                    # place is uncolored, nothing left to do
                    continue

                current = tuple(expr(var, theory=LIA) for var in vars)
                extl = tuple(expr(var, theory=LIA) for var in place.extl)
                evolved = _update(place, current, extl)

                event_payload = _color_record_to_tuple(new_colors[place])
                # jumps_here := the token participates in the firing and moves to `place`
                jumps_here = expr_Xin_transition & (net.place_ids[place] == expr_Xnew_place)
                # evolves_here := no transition fires and the token resides in `place`
                evolves_here = ~expr_Xtransition_round & (net.place_ids[place] == expr_current_place)

                # Process one per-place color component `current_colors[place][i]`
                # at a time:
                # A. the token participates in this round's firing AND the place
                #    we're processing is the token's new place
                #    -> set the component to the corresponding payload component
                #       `new_colors[place][i]'` (the token jumps)
                # B. NO transition fires in this round AND the place we're
                #    processing is the token's current place
                #    -> update according to `update` (stutters if `update` is omitted)
                # C. otherwise
                #    -> stutter the component
                for var, old, updated, new_color in zip(vars, current, evolved, event_payload):
                    # ite(in_transition' & <place_id> == X(new_place),
                    #     new_colors[place][i]',
                    #     ite(~transition_round' & <place_id> == current_place,
                    #         update(current_colors[place], extl)[i],
                    #         current_colors[place][i]))
                    new_value = ite(jumps_here,
                        expr(X(new_color), theory=LIA),
                        ite(evolves_here, updated, old))

                    # current_colors[place][i]' := ite(...)
                    update.append(Term(LIA.Id(), [X(var)], [new_value.wire]))

        self = super().__new__(cls, vars=variables, init=init, update=update)
        self.net = net
        self.place = current_place
        self.colors = current_colors
        self.transition_round = transition_round
        self.in_transition = in_transition
        self.new_place = new_place
        self.new_colors = new_colors
        return self


class TransitionControllerModule(Module):
    """A reactive module firing at most one Petri net transition per round.

    The controller reads the tokens' latched ``place`` and ``colors`` to
    evaluate transition guards and in turn communicates firing information to
    the tokens. To this end, it controls the shared ``transition_round``, each
    token's ``in_transition``, ``new_place`` and ``new_colors``, and a sticky
    ``overflow`` flag. ``in_transition`` and ``transition_round`` are flags
    (they are assigned in each round, and are not reactive module events).

    Evaluating guards:
        A binding is an ordered tuple of distinct slots ``(s_1, ..., s_k)``
        whose tokens reside in the transition's input places, in ``inputs``
        order.
        A candidate is a pair of a transition and a binding. A candidate is
        enabled if the transition's guard holds on the binding's latched colors
        (immediate transitions have guard True).

    Transition firing:
        Candidates are ordered by transition definition order, then
        lexicographically by token slot. The first candidate that can fire,
        fires. ``transition_round`` is True iff some candidate fires.

    Output tokens:
        For a candidate with outputs ``o_1, ..., o_m``, output ``o_j`` with ``j
        <= k`` reuses slot ``s_j``. Extra outputs ``o_j`` with ``k < j <= m``
        take the lowest absent slots, in order. Consumed slots ``s_j`` with ``m
        < j <= k`` become absent. An enabled candidate lacking absent slots for
        its extra outputs cannot fire and sets ``overflow``.
    """

    def __new__(cls, net, tokens):
        # validate tokens and their `transition_round` flag
        tokens = tuple(tokens)
        if not tokens:
            raise ValueError("a transition controller requires at least one token")
        transition_round = tokens[0].transition_round
        if any(token.transition_round is not transition_round for token in tokens):
            raise ValueError("tokens must share the same transition_round variable")

        # Construct the module's controlled and external variables; the token state
        # is external and read latched only
        overflow = Var(BOOL)
        controlled = [transition_round, overflow]
        external = []
        for token in tokens:
            controlled.extend((token.in_transition, token.new_place))
            external.append(token.place)
            for place in net.places:
                controlled.extend(_color_record_to_tuple(token.new_colors[place]))
                external.extend(_color_record_to_tuple(token.colors[place]))
        variables = controlled + external

        # Initialize with no transition firing, no overflow, and zeroed payloads:
        # (transition_round', overflow', *in_transition', *new_place', *new_colors') := 0
        with collecting() as init:
            init.extend(_zero_init(controlled))

        with collecting() as update:
            # The tokens' latched state, per slot: the place ID, one Color record of
            # per-place color components for each place, and whether the slot is absent
            places = [expr(token.place, theory=LIA) for token in tokens]
            colors = [
                {
                    place: place.Color(*(
                        expr(var, theory=LIA)
                        for var in _color_record_to_tuple(token.colors[place])
                    ))
                    for place in net.places
                }
                for token in tokens
            ]

            # Pick the lowest absent slot for each extra output token, in order.
            # Output tokens paired with an input token reuse its slot, so only
            # output tokens beyond the number of input tokens need a fresh slot. The picks
            # depend only on which slots are absent, not on the transition or
            # binding: a bound slot is occupied whenever its candidate is enabled.
            # So we compute them once, for the largest number of extra outputs,
            # and a transition with e extra outputs uses the first e.
            #
            # Implementation note: each pick could instead be a LIA slot index
            # (the lowest absent slot above the previous pick), but per-slot
            # boolean conditions keep slot allocation in the propositional fragment.
            max_extra_tokens = max(
                (len(t.outputs) - len(t.inputs) for t in net.transitions), default=0
            )
            remaining: list[Expr] = [place == 0 for place in places]  # remaining[s]: slot s is absent and not picked yet
            picks: list[list[Expr]] = []  # picks[j][s]: extra output j takes slot s
            enough: list[Expr] = []  # enough[j]: some absent slot remained for output j
            for _ in range(max_extra_tokens):
                pick = []
                # mark the lowest slot that is available as picked
                picked_a_slot = _value(False, BOOL)
                for is_free in remaining:
                    # pick[s] := slot s is available and no lower slot was picked
                    pick.append(is_free & ~picked_a_slot)
                    # picked_a_slot := some slot up to and including s was available,
                    #                  so a slot has already been picked
                    picked_a_slot = picked_a_slot | is_free
                picks.append(pick)
                # picked_a_slot is true at this point iff we found an empty slot
                enough.append(picked_a_slot)
                # mark the picked slot as no longer remaining
                remaining = [r & ~p for r, p in zip(remaining, pick)]

            # Enumerate candidates in transition definition and slot order; the
            # first candidate that can fire wins. Each candidate adds its possible
            # writes to the slots it touches, conditioned on its selection.
            writes: list[list[_Write]] = [[] for _ in tokens]  # writes[s]: possible writes to slot s
            overflows: list[Expr] = []  # one per transition with extra outputs
            picked_a_candidate = _value(False, BOOL)  # some earlier candidate was picked to fire
            for transition in net.transitions:
                n_inputs, n_outputs = len(transition.inputs), len(transition.outputs)
                n_extra_tokens = max(n_outputs - n_inputs, 0)
                guard = getattr(transition, "guard", None)
                produce = getattr(transition, "produce", None)
                # An enabled candidate fires only if every extra output finds a
                # slot; otherwise it overflows and later candidates are considered.
                # has_room := every extra output finds a slot (True if n_extra_tokens == 0)
                has_room = _all(enough[:n_extra_tokens])
                enabled_bindings = []
                for binding in permutations(range(len(tokens)), n_inputs):
                    # Check if the binding is enabled and fires.
                    input_colors = tuple(
                        colors[slot][place] for slot, place in zip(binding, transition.inputs)
                    )
                    conditions = [
                        places[slot] == net.place_ids[place]
                        for slot, place in zip(binding, transition.inputs)
                    ]
                    if guard is not None:
                        conditions.append(_value(guard(*input_colors), BOOL))
                    # enabled := every bound token is in its input place and the guard
                    #            holds on their colors
                    enabled = _all(conditions)
                    enabled_bindings.append(enabled)
                    # candidate_fires := no earlier candidate was picked, and this one
                    #                    is enabled and has room
                    candidate_fires = ~picked_a_candidate & (enabled & has_room)
                    # picked_a_candidate := some candidate up to this one fires
                    picked_a_candidate = picked_a_candidate | (enabled & has_room)

                    # Produce the output colors.
                    output_colors = produce(*input_colors) if produce is not None else ()

                    # Assign slots to the output tokens.
                    #
                    # We record a guarded write for each slot that this candidate may change.
                    # Paired and leftover (= to-be-removed) input tokens are in bound slots,
                    # so each of them results in one write to a known, concrete slot.
                    # For leftover output tokens, the picked slot is only known symbolically,
                    # so every slot receives one guarded write per leftover output token,
                    # guarded by whether that slot is the one picked for it.
                    #  When the candidate fires, at most one write per slot takes effect.

                    # Paired case: the output token replaces the input token in the same
                    # slot, with the produced color (`zip` stops at the shorter
                    # side; leftovers on either side are handled below)
                    for slot, place, color in zip(binding, transition.outputs, output_colors):
                        writes[slot].append(_Write(candidate_fires, net.place_ids[place], color))

                    # Leftover input tokens: consumed, their slot becomes absent (place ID 0).
                    for slot in binding[n_outputs:]:
                        writes[slot].append(_Write(candidate_fires, 0, None))

                    # Leftover output tokens: each goes to its picked fresh slot.
                    for pick, place, color in zip(picks, transition.outputs[n_inputs:], output_colors[n_inputs:]):
                        for slot, slot_picked in enumerate(pick):
                            writes[slot].append(
                                _Write(candidate_fires & slot_picked, net.place_ids[place], color))

                # the transition overflows if some binding is enabled but lacks room for its extra outputs
                if n_extra_tokens:
                    overflows.append(_any(enabled_bindings) & ~has_room)

            # transition_round' := some candidate fires
            update.append(Term(LIA.Id(), [X(transition_round)], [picked_a_candidate.wire]))
            # overflow' := overflow | some transition overflows in this round
            sticky_overflow = _any([expr(overflow, theory=LIA), *overflows])
            update.append(Term(LIA.Id(), [X(overflow)], [sticky_overflow.wire]))

            # Turn each slot's recorded writes into the token's transition inputs.
            # At most one write per slot takes effect in a round, so the result is
            # independent of the nested `ite` ordering below.
            for token, slot_writes in zip(tokens, writes):
                # in_transition' := some write to this slot takes effect
                touched = _any(write.condition for write in slot_writes)
                update.append(Term(LIA.Id(), [X(token.in_transition)], [touched.wire]))

                # new_place' := the place ID of the write that takes effect:
                #   ite(cond_1, place_1, ite(cond_2, place_2, ... 0))
                # The default 0 is never used: without a write, in_transition' is
                # False and the token ignores new_place'.
                new_place = _value(0, Int([1, 1]))
                # Build the chain from the inside out; reversed makes the first write
                # (in candidate order) the outermost ite, for readability.
                for write in reversed(slot_writes):
                    new_place = ite(write.condition, write.place_id, new_place)
                update.append(Term(LIA.Id(), [X(token.new_place)], [new_place.wire]))

                # new_colors[place]' := the color of the write that takes effect, component-wise:
                #   ite(cond_1, color_1[i], ite(cond_2, color_2[i], ... new_colors[place][i]))
                # The default (the latched value) is never used: without a write
                # to `place`, the token does not move to `place` and ignores
                # new_colors[place]'.
                #
                # chains[place, field]: the ite chain built so far for payload
                # component new_colors[place][field]; it starts at the latched value
                chains = {
                    (place, field.name):
                        expr(getattr(token.new_colors[place], field.name), theory=LIA)
                    for place in net.places
                    for field in fields(place.Color)
                }
                # reversed: the first write is the outermost ite (for readability)
                for write in reversed(slot_writes):
                    if write.color is None:
                        # the write consumes the token: there is no color to pass
                        continue
                    # wrap each component of the write's new place in one more ite
                    place = net.places_by_id[write.place_id]
                    for field, var in zip(fields(place.Color), place.ctrl):
                        color = _value(getattr(write.color, field.name), var.dtype)
                        key = (place, field.name)
                        chains[key] = ite(write.condition, color, chains[key])
                # new_colors[place][i]' := ite(cond_1, color_1[i], ...)
                for (place, name), value in chains.items():
                    target = getattr(token.new_colors[place], name)
                    update.append(Term(LIA.Id(), [X(target)], [value.wire]))

        self = super().__new__(cls, vars=variables, init=init, update=update)
        self.net = net
        self.tokens = tokens
        self.transition_round = transition_round
        self.overflow = overflow
        return self


class DCPNReactiveModules(NamedTuple):
    """The reactive modules of a bounded net.

    Compose them as ``Module(*lowered.tokens, lowered.controller)`` to obtain a
    closed system.
    """

    tokens: tuple[TokenModule, ...]
    controller: TransitionControllerModule
