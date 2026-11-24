"""A module's wires, in the order every encoding and every predicate
numbers them."""

from __future__ import annotations

from dataclasses import dataclass

from reactor import X


@dataclass(frozen=True)
class ModuleWires:
    """Each interface variable (`ctrl_latched`) and `X` of it, its value
    after the round (`ctrl_next`); each input as latched in the round
    before (`extl_latched`) and as awaited in the round (`extl_next`)."""

    ctrl_latched: list
    ctrl_next: list
    extl_latched: list
    extl_next: list

    @staticmethod
    def of(module) -> "ModuleWires":
        return ModuleWires(
            ctrl_latched=list(module.ctrl),
            ctrl_next=[X(v) for v in module.ctrl],
            extl_latched=list(module.extl),
            extl_next=[X(v) for v in module.extl],
        )
