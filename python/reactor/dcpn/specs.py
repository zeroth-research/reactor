"""Python specifications for LIA-colored Petri nets.

Specifications describe a net independently of its token capacity. Places
declare color fields as class-level Vars and optionally define resident updates.
Each place class generates a frozen Color record for passing token colors to
transition guards and describing transition outputs. Immediate and guarded
transitions connect ordered input and output places.

The ``obj`` decorator optionally instantiates a declaration.

Validation is performed via declaration hooks and constructors to check local
structure; DCPN checks the theory, net connections, and symbolic method results.

``DCPN.to_modules`` delegates bounded construction to ``lowering``: it returns
a ``DCPNReactiveModules`` named tuple of token modules with empty initial
marking and a transition controller, which together form a closed system.
``DCPN.to_token_modules`` returns only the open token modules.
"""

from dataclasses import fields, make_dataclass

from ..reactor import LIA, Var
from ..sort import Bool, Int
from ..expr import collecting, expr
from .lowering import (
    DCPNReactiveModules, TokenModule, TransitionControllerModule, _update, _value,
)


def obj(cls=None, **kwargs):
    """Instantiate a declaration, optionally forwarding constructor keywords."""
    # support the form `@obj(param=...)`
    if cls is None:
        return lambda cls: cls(**kwargs)
    # plain `@obj`
    return cls(**kwargs)


class Place:
    """A colored Petri net place specification.

    Declare controlled colors as class-level ``Var`` objects.

    Optionally define ``update(self, *colors, *extl)`` for discrete evolution
    while a token resides here. Its arguments are: colors in declaration order,
    followed by external inputs in constructor order. Return color values in the
    same order, or a single value for one color. Omitting ``update`` leaves
    colors unchanged.

    Attributes:
        ctrl: Ordered tuple of controlled color declarations.
        extl: Ordered tuple of external input variables.
        Color: Generated frozen dataclass with one required field per color.
            Transitions return these records to specify produced token colors.

    Example:
        A place whose tokens advance by an externally supplied increment::

            from reactor import Int, Var
            from reactor.dcpn import Place

            increment = Var(Int([1, 1]))

            class Work(Place):
                progress = Var(Int([1, 1]))

                def update(self, progress, increment):
                    return progress + increment

            work = Work(extl=(increment,))
            color = work.Color(progress=0)  # a color record, not a token
    """

    ctrl: tuple[Var, ...] = ()
    Color = make_dataclass("Color", [], frozen=True)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        declarations = {}
        # Base declarations precede derived declarations; overriding a field
        # keeps its position. Normal Python MRO precedence applies to diamonds.
        for base in reversed(cls.__mro__):
            for name, value in vars(base).items():
                if isinstance(value, Var):
                    if name in {"ctrl", "extl", "Color"}:
                        raise TypeError(f"{cls.__name__}: reserved color name {name!r}")
                    declarations[name] = value
                elif name in declarations:
                    raise TypeError(
                        f"{cls.__name__}.{name}: cannot override an inherited color field "
                        f"with {type(value).__name__}; expected a Var"
                    )
        cls.ctrl = tuple(declarations.values())
        cls.Color = make_dataclass(
            f"{cls.__name__}Color",
            [(name, object) for name in declarations],
            frozen=True,
            namespace={"__module__": cls.__module__},
        )

    def __init__(self, *, extl=()):
        """Create a place specification.

        Args:
            extl: Ordered iterable of external variables. Defaults to no external inputs.

        Controlled color variables are declared on the class.
        Transitions supply initial colors when producing tokens.
        """
        self.extl = tuple(extl)

        # validate
        if not all(isinstance(var, Var) for var in self.extl):
            raise TypeError("external inputs must be Vars")
        if getattr(self, "flow", None) is not None:
            # TODO(thpani): implement LRA/flows
            raise NotImplementedError("flows are not supported yet")
        update = getattr(self, "update", None)
        if update is not None and not callable(update):
            raise TypeError("update must be callable")


class _Transition:
    inputs: tuple[Place, ...] = ()
    outputs: tuple[Place, ...] = ()

    def __init__(self):
        self.inputs = tuple(self.inputs)
        self.outputs = tuple(self.outputs)

        # validate
        if not all(isinstance(place, Place) for place in self.inputs + self.outputs):
            raise TypeError("transition inputs and outputs must be Place objects")
        if getattr(self, "rate", None) is not None:
            # TODO(thpani): implement stochastic transitions
            raise NotImplementedError("stochastic transitions are not supported yet")
        produce = getattr(self, "produce", None)
        if produce is None:
            if self.outputs:
                raise TypeError("a transition with output places must define produce")
        elif not callable(produce):
            raise TypeError("produce must be callable")


class ImmediateTransition(_Transition):
    """An immediate transition; define ``produce(*input_colors)`` if it has outputs."""


class GuardedTransition(_Transition):
    """Define ``guard(*input_colors)`` and, if it has outputs, ``produce(*input_colors)``."""

    def __init__(self):
        super().__init__()

        # validate
        if not callable(getattr(self, "guard", None)):
            raise TypeError("a guarded transition must define guard")


class DCPN:
    """An LIA net specification with empty initial marking.

    ``to_modules`` lowers the net to reactive modules: one per token, plus a
    transition controller. ``to_token_modules`` returns only the token modules.
    """

    def __init__(self, *, theory, places=(), transitions=()):
        self.theory = theory
        self.places = tuple(places)
        self.transitions = tuple(transitions)

        # validate the spec
        self._validate()

        # map places to place ids and back, starting at 1 (0 denotes an absent token)
        self.place_ids = {place: i for i, place in enumerate(self.places, 1)}
        self.places_by_id = {i: place for place, i in self.place_ids.items()}

    def _validate(self):
        # validate theory
        if self.theory is not LIA:
            raise NotImplementedError("DCPN currently supports only LIA")

        # validate places
        if not all(isinstance(place, Place) for place in self.places):
            raise TypeError("places must contain Place objects")
        if len(set(self.places)) != len(self.places):
            raise ValueError("a place occurs twice in the net")

        # validate transitions and their connections within this net
        for transition in self.transitions:
            if not isinstance(transition, (ImmediateTransition, GuardedTransition)):
                raise TypeError("transitions must be immediate or guarded transition objects")
            for place in transition.inputs + transition.outputs:
                if place not in self.places:
                    raise ValueError("transition endpoints must be places in the net")

        # Symbolic expressions require a collector; we discard these terms built only for
        # validation, since `to_token_modules` builds the actual module terms separately.
        with collecting():
            for place in self.places:
                # validate the place's variables against the net's theory
                for var in place.ctrl + place.extl:
                    if not isinstance(var.dtype, (Int, Bool)):
                        raise TypeError(
                            f"LIA colors and external inputs must be Int or Bool, got {var.dtype}"
                        )

                # symbolically call the place's `update` to validate it
                colors = tuple(expr(Var(var.dtype), theory=LIA) for var in place.ctrl)
                external = tuple(expr(var, theory=LIA) for var in place.extl)
                _update(place, colors, external)

            for transition in self.transitions:
                # generate symbolic variables for the color (i.e., each variable) of each input place
                args = tuple(
                    place.Color(*(expr(Var(var.dtype), theory=LIA) for var in place.ctrl))
                    for place in transition.inputs
                )

                # for guarded transitions, call `guard` to validate it
                if isinstance(transition, GuardedTransition):
                    guard = getattr(transition, "guard")
                    _value(guard(*args), Bool([1, 1]))

                # symbolically call `produce`, if provided, to validate its results
                produce = getattr(transition, "produce", None)
                if produce is None:
                    continue
                outputs = produce(*args)
                # validate arity
                if not isinstance(outputs, tuple) or len(outputs) != len(transition.outputs):
                    raise ValueError("produce must return a tuple aligned with outputs")
                # validate that each result from `produce` corresponds to the color of the output place
                for output_place, produced_color in zip(transition.outputs, outputs):
                    if not isinstance(produced_color, output_place.Color):
                        raise TypeError(f"produce must return {output_place.Color.__name__}")
                    for field, var in zip(fields(produced_color), output_place.ctrl):
                        _value(getattr(produced_color, field.name), var.dtype)

    def to_token_modules(self, *, token_capacity) -> tuple[TokenModule, ...]:
        """Return fresh open token modules sharing one ``transition_round`` input."""
        if type(token_capacity) is not int or token_capacity <= 0:
            raise ValueError("token_capacity must be a positive integer")
        transition_round = Var(Bool([1, 1]))
        return tuple(TokenModule(self, transition_round) for _ in range(token_capacity))

    def to_modules(self, *, token_capacity) -> DCPNReactiveModules:
        """Return the token modules and their transition controller.

        ``Module(*tokens, controller)`` is a closed system. The
        ``TransitionControllerModule`` fires at most one transition per round;
        check ``controller.overflow`` to detect firings blocked by insufficient
        ``token_capacity``.
        """
        tokens = self.to_token_modules(token_capacity=token_capacity)
        controller = TransitionControllerModule(self, tokens)
        return DCPNReactiveModules(tokens=tokens, controller=controller)
