import inspect
import torch.nn as nn

from ..reactor import Module as _BaseModule, X
from ..matrix.real import Real
from ..matrix.int import Int
from ..builder import builder_for, number_sort
from ..analyzer import convert_method, resolve_wire


def _numeric_sort(number, n):
    """The numeric row-vector sort of width `n`, of the sort family `number`."""
    return number_sort(number)([1, n])


def _validate_number_supports_nn(number):
    """A neural module compiles to `Linear`, `Transpose` and `ReLU`, which only reals and
    integers have; reject any other numeric sort with a clear error."""
    if number not in (Real, Int):
        name = getattr(number, "__name__", repr(number))
        raise NotImplementedError(
            f"neural modules need real or integer numbers, got {name}; use Real or Int"
        )


def _validate_weight_dtypes(live_layers, number):
    """Weights are used as-is (no coercion), so their dtype must match the numeric sort:
    floating-point for Real, integer for Int. Raise otherwise."""
    want_float = number is Real
    kind = "floating-point" if want_float else "integer"
    for name, layer in live_layers.items():
        for attr in ("weight", "bias"):
            t = getattr(layer, attr, None)
            if t is None:
                continue
            if t.is_floating_point() != want_float:
                raise TypeError(
                    f"{number.__name__} numbers need {kind} weights, but layer "
                    f"'{name or '<root>'}'.{attr} has dtype {t.dtype}; "
                    f"quantise/cast the network to match"
                )


def _extract_nn_module(nn_instance, number=Real, combinatorial=False, **kwargs):
    """Analyze an nn.Module instance and extract a symbolic Module.

    Uses live tensor references for weight/bias so that training updates
    flow through to the symbolic module automatically.

    ``combinatorial`` selects the kind of module built from ``forward``:
      * ``False`` (default) — a **sequential** module over the latched input,
        e.g. V(s): ``update`` reads the *latched* input and writes the next
        output; the initial output is havoced.
      * ``True`` — a **combinatorial** module: ``forward`` reads the awaited
        *next* input, so it computes the output in the same step, e.g. V(s').
    """
    nn_cls = type(nn_instance)

    user_extl = kwargs.pop("extl", None)
    user_ctrl = kwargs.pop("ctrl", None)
    if kwargs:
        raise TypeError(f"unknown arguments: {sorted(kwargs)}")

    obs_param = next(
        p for p in inspect.signature(nn_cls.forward).parameters if p != "self"
    )

    # Infer layer structure from the actual nn.Module instance
    layers = {}
    live_layers = {}
    for name, child in nn_instance.named_modules():
        if isinstance(child, nn.Linear):
            layers[name] = (child.in_features, child.out_features)
            live_layers[name] = child

    if not layers:
        raise ValueError("No nn.Linear layers found in the module")

    layer_list = list(layers.values())
    obs_size = layer_list[0][0]
    qval_size = layer_list[-1][1]

    _validate_number_supports_nn(number)
    _validate_weight_dtypes(live_layers, number)

    extl = resolve_wire("extl", _numeric_sort(number, obs_size), user_extl)
    ctrl = resolve_wire("ctrl", _numeric_sort(number, qval_size), user_ctrl)

    layer_out_features = {name: out for name, (_, out) in layers.items()}

    def _forward_terms(read_next: bool):
        # `forward` reads its input as the variable's *latched* slot (index 0). Put the
        # next wire there (read_next=True) to read s'; put the latched wire there
        # (read_next=False) to read s. The output is always the next intf wire.
        in_wire, other = (X(extl), extl) if read_next else (extl, X(extl))
        return convert_method(
            nn_cls.forward, {obs_param: [in_wire, other]}, [X(ctrl)], cls=nn_cls,
            layers=layer_out_features, live_layers=live_layers, builder=builder_for(number),
        )

    if combinatorial:
        # V(s'): memoryless, reads the awaited next input.
        assign = _forward_terms(True)
        return dict(init=assign, update=assign, vars=[ctrl, extl])
    else:
        # V(s): init havocs; update reads the latched input.
        return dict(update=_forward_terms(False), vars=[ctrl, extl])


class Module(_BaseModule, nn.Module):
    """An nn.Module backed by a symbolic Module with live tensor references.

    Inherits both Module (symbolic reactive module) and nn.Module (trainable).

    Usage:
        from reactor.torch import Module

        wrapped = Module(nn_module_instance)                    # sequential, V(s)
        wrapped = Module(nn_module_instance, number=Int)         # integer (e.g. quantised) net
        wrapped = Module(nn_module_instance, combinatorial=True) # combinatorial, V(s')
        wrapped.parameters()   # returns original nn.Module parameters
        wrapped(x)             # runs forward pass via original nn.Module
        wrapped.atoms          # symbolic module structure

    ``number`` selects the numeric sort of the interface and the weights, ``Real``
    (default) or ``Int``. ``combinatorial`` selects whether
    ``forward`` becomes a sequential atom over the latched input (``False``,
    default) or a combinatorial atom over the next input (``True``).

    Training the original nn.Module automatically updates the symbolic module
    because the Linear op holds a reference to the live weight tensors.
    """

    def __new__(cls, nn_module, number=Real, combinatorial=False, **kwargs):
        if not isinstance(nn_module, nn.Module):
            raise TypeError(f"Expected nn.Module, got {type(nn_module)}")
        parts = _extract_nn_module(
            nn_module, number=number, combinatorial=combinatorial, **kwargs
        )
        return _BaseModule.__new__(cls, **parts)

    def __init__(self, nn_module, number=Real, combinatorial=False, **kwargs):
        nn.Module.__init__(self)
        self.inner = nn_module

    def forward(self, x):
        return self.inner(x)
