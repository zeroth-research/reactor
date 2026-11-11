"""SMT wrapper around a reactive Module: symbolic input/output in cvc5.

Exposes `ModuleSMT` — given a `Module`, provides:
  * `fresh_ctrl` / `fresh_extl_l` / `fresh_extl_n` — mint cvc5 consts for
    the ctrl-latched and extl-latched/next input wires.
  * `init_state(extl_n)` / `update_state(ctrl, extl_l, extl_n)` — return
    the ctrl-next outputs as a list of cvc5 terms (one per component).

State is carried as a Python list of per-component cvc5 terms, matching
the left-nested tuple layout used elsewhere in the codegen.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import cvc5

from reactor import Wire, Var, Module, X
from .encode import translate_terms, wire_sort


# Everything cvc5 mints, kept alive for the run: the bindings segfault when
# a `TermManager` is freed before the solvers and terms made from it, and a
# GC cycle frees the two together in arbitrary order. One list, here,
# because the ordering it defends against is global -- a per-module list
# only orders the objects inside it, so two of them (`smt.precheck`'s
# queries and another caller's encodings) could still be finalised in the
# wrong order relative to each other. A run keeps a handful of entries and
# then exits.
#
# The list does not hold through the exit itself. At interpreter shutdown
# every module is garbage, this list with it, and one collection frees all
# of it in arbitrary order: a term the collector clears before freeing it
# releases its manager, and when that was the last reference the term is
# then freed after its manager, which crashes in
# `NodeValue::markForDeletion` (exit -11, after the run printed everything).
# A cycle that holds terms -- a kept exception's traceback did -- is enough.
# So a `TermManager` handed in is also pinned: never freed, whatever the order.
_LIVE: list = []


def keep_alive(*objs: object) -> None:
    """Retain cvc5 objects until the process exits, and every `TermManager`
    among them past it. See `_LIVE`."""
    _LIVE.extend(objs)
    for obj in objs:
        if isinstance(obj, cvc5.TermManager):
            _pin(obj)


_incref = ctypes.pythonapi.Py_IncRef
_incref.argtypes = [ctypes.py_object]
_incref.restype = None


def _pin(obj: object) -> None:
    """A reference to `obj` that is never released."""
    _incref(obj)


@dataclass
class ModuleSMT:
    tm: cvc5.TermManager
    module: Module

    def __post_init__(self):
        atoms = list(self.module.atoms)
        if len(atoms) != 1:
            raise ValueError(
                f"ModuleSMT: expected single-atom module, got {len(atoms)}"
            )
        self.atom = atoms[0]
        # Snapshot init/update terms immediately — the Rust-backed
        # `Atom.init` / `Atom.update` iterators are one-shot in some
        # wrappers, so other consumers (e.g. `LeanContext`) could
        # exhaust them before we get here.
        self.init_terms: list = list(self.atom.init)
        self.update_terms: list = list(self.atom.update)
        self.ctrl_latched: list[Var] = list(self.module.ctrl)
        self.ctrl_next: list[Wire] = [X(v) for v in self.module.ctrl]
        self.extl_latched: list[Var] = list(self.module.extl)
        self.extl_next: list[Wire] = [X(v) for v in self.module.extl]

    # --- input construction ---------------------------------------------

    def _consts_for(self, wires: list[Wire], prefix: str) -> list[cvc5.Term]:
        return [
            self.tm.mkConst(wire_sort(self.tm, w), f"{prefix}{i}")
            for i, w in enumerate(wires)
        ]

    def fresh_ctrl(self, prefix: str = "s") -> list[cvc5.Term]:
        return self._consts_for(self.ctrl_latched, prefix)

    def fresh_extl_l(self, prefix: str = "el") -> list[cvc5.Term]:
        return self._consts_for(self.extl_latched, prefix)

    def fresh_extl_n(self, prefix: str = "en") -> list[cvc5.Term]:
        return self._consts_for(self.extl_next, prefix)

    # --- symbolic init/update -------------------------------------------

    def init_state(self, extl_n: list[cvc5.Term]) -> list[cvc5.Term]:
        """Return ctrl-next outputs after running `init(extl_n)`."""
        bindings = {w.id: v for w, v in zip(self.extl_next, extl_n)}
        wt = translate_terms(self.tm, self.init_terms, bindings)
        return [wt[w.id] for w in self.ctrl_next]

    def update_state(
        self,
        ctrl: list[cvc5.Term],
        extl_l: list[cvc5.Term],
        extl_n: list[cvc5.Term],
    ) -> list[cvc5.Term]:
        """Return ctrl-next outputs after running `update(ctrl, extl_l, extl_n)`."""
        bindings: dict[int, cvc5.Term] = {}
        for w, v in zip(self.ctrl_latched, ctrl):
            bindings[w.id] = v
        for w, v in zip(self.extl_latched, extl_l):
            bindings[w.id] = v
        for w, v in zip(self.extl_next, extl_n):
            bindings[w.id] = v
        wt = translate_terms(self.tm, self.update_terms, bindings)
        return [wt[w.id] for w in self.ctrl_next]
