"""Lower LIA Petri-net specifications to bounded token reactive modules.

Each TokenModule allocates fresh state for one token slot: an integer place ID
and a separate color block for every place. Tokens start absent (place ID zero),
with integer colors zero and Boolean colors False. Ordinary updates evolve only
the resident place's colors; inactive place color components stutter.

An external Boolean transition_event signals a firing by changing value within
a round. The token awaits new_place and new_colors payloads through their next
wires, applying them in that same round instead of its resident place update.
A new place ID of zero removes the token from the net.

The generated modules are open: jumps must be initiated by an external
transition controller.
"""

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Self, cast

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

    Tokens react to transitions: ``transition_event`` is an external Boolean Var.
    A transition event fires when its latched and next values differ, i.e.,
    iff ``transition_event != X(transition_event)``. As long as
    ``transition_event == X(transition_event)``, resident evolution runs
    normally.
    ``new_place`` and ``new_colors`` are external payload Vars, read through X
    in the transition firing round. A transition event takes precedence over a
    resident update, even when the new place is the current place. A new place
    of zero removes the token.
    """

    def __new__(cls, net):
        # current place and color
        current_place = Var(Int([1, 1]))
        current_colors = _allocate_color_records(net.places)
        # transition trigger and payload
        transition_event=Var(Bool([1, 1]))
        new_place=Var(Int([1, 1]))
        new_colors=_allocate_color_records(net.places)

        # construct the modules controlled and external variables
        controlled = [current_place]
        external = [transition_event, new_place]
        for place in net.places:
            controlled.extend(_color_record_to_tuple(current_colors[place]))
            external.extend(_color_record_to_tuple(new_colors[place]))
            external.extend(place.extl)

        # Places can share external inputs; declare each variable once.
        variables = tuple(dict.fromkeys(controlled + external))

        with collecting() as init:
            # Initialize the token as absent, with zeroed color fields:
            # (current_place', *current_colors') := 0
            for var in controlled:
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
                init.append(Term(LIA.Id(), [X(var)], [zero.wire]))

        with collecting() as update:
            # current_place' := ite(transition_event?, new_place', current_place)
            expr_current_place = expr(current_place, theory=LIA)
            expr_Xnew_place = expr(X(new_place), theory=LIA)
            expr_fired = expr(transition_event, theory=LIA) != expr(X(transition_event), theory=LIA)
            next_place = ite(expr_fired, expr_Xnew_place, expr_current_place)
            update.append(Term(LIA.Id(), [X(current_place)], [next_place.wire]))

            for place in net.places:
                vars = _color_record_to_tuple(current_colors[place])

                if not vars:
                    # place is uncolored, nothing left to do
                    continue

                current = tuple(expr(var, theory=LIA) for var in vars)
                extl = tuple(expr(var, theory=LIA) for var in place.extl)
                evolved = _update(place, current, extl)

                event_payload = _color_record_to_tuple(new_colors[place])

                # Process one color field at a time:
                # A. NO transition event happened in this round:
                #    1. the place we're processing is the token's current place
                #       -> update according to `update` (stutters if `update` is omitted)
                #    2. the place we're processing is NOT the token's current place
                #       -> stutter the color field
                # B. a transition event DID happen in this round:
                #    1. the place we're processing is the token's new place
                #       -> re-initialize (jump) the field from the event payload
                #    2. the place we're processing is NOT the token's new place
                #       -> stutter the color field
                for var, old, updated, new_color in zip(vars, current, evolved, event_payload):
                    # ite(transition_event?,
                    #     ite(<place_id> == X(new_place), new_colors[place][i]', current_colors[place][i]),
                    #     ite(<place_id> == current_place, update(current_colors[place], extl)[i], current_colors[place][i]))
                    value = ite(expr_fired,
                        ite(net.place_ids[place] == expr_Xnew_place, expr(X(new_color), theory=LIA), old),
                        ite(net.place_ids[place] == expr_current_place, updated, old))

                    # current_colors[place][i]' := ite(...)
                    update.append(Term(LIA.Id(), [X(var)], [value.wire]))

        self = super().__new__(cls, vars=variables, init=init, update=update)
        self.net = net
        self.place = current_place
        self.colors = current_colors
        self.transition_event = transition_event
        self.new_place = new_place
        self.new_colors = new_colors
        return self
