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

        # construct the modules controlled and external variables
        controlled = [current_place]
        external = [transition_round, in_transition, new_place]
        for place in net.places:
            controlled.extend(_color_record_to_tuple(current_colors[place]))
            external.extend(_color_record_to_tuple(new_colors[place]))
            external.extend(place.extl)

        # Places can share external inputs; declare each variable once.
        variables = tuple(dict.fromkeys(controlled + external))

        with collecting() as init:
            # Initialize the token as absent, with zeroed per-place color components:
            # (current_place', *current_colors') := 0
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
                # the token participates in the firing and jumps into the currently processed place
                jumps_here = expr_Xin_transition & (net.place_ids[place] == expr_Xnew_place)
                # no transition fires and the token resides in the currently processed place
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
