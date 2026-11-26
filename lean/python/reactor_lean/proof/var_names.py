"""The module's variables in Lean by the names its input gives them.

A claim names variables in the input's terms (`inputs.names`): `(X px)` for the state
variable `px` after the round, `(X w)` and `w` for an input read in the round and in the
round before. Without this module the certificate's Lean (`Data.lean`) reads them by
position only -- `(s.2.2.1 0 0)`, `(e.2.1 0 0)`. With it, `Data.lean` defines one
abbreviation per named variable, in the namespace `Var`, and the predicates are printed
through them:

    namespace Var
    @[simp] abbrev px (s : <state>) := s.1 0 0              -- state variable `px`
    @[simp] abbrev w (e : <inputs>) := e.2.1 0 0             -- input `w`, read in the round: `(X w)`
    @[simp] abbrev w.latched (e : <inputs>) := e.1.1 0 0     -- read in the round before: `w`
    end Var

    open Var in
    def inv : <state> → Prop := fun s => px s * px s ≤ 1

A name that is not a plain Lean identifier is quoted (`«a-b»`). A name that a Lean keyword,
a certificate definition, ReactLib or a common function already uses is printed qualified
(`Var.init s`), so `open Var` can never make a predicate ambiguous. The abbreviations are
`@[simp]` and `simp_names` lists them for the proofs' own `simp only` sets, so a proof
unfolds them as it unfolds `inv`, and it sees the positional terms it always did.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from reactor import Wire

from ..lowering.elements import product_accessor
from ..lowering.sorts import wire_shape

NAMESPACE = "Var"
LATCHED = "latched"

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_']*")
_KEYWORDS = frozenset("""
    abbrev at axiom by calc class def deriving do else end example exists export extends
    for forall fun have if import in inductive infix infixl infixr instance lemma let local
    macro match mut namespace noncomputable nomatch notation open opaque partial prefix
    private protected rec return section set_option show structure suffices syntax then
    theorem this universe unsafe variable where with Prop Sort Type
""".split())
# Root names a predicate might also mean: the certificate's definitions, the encoding's,
# ReactLib's (`Relational`, `Spec`) and Mathlib's that predicates print.
_ROOT = frozenset("""
    init update RM lts inv P ranking init_pre update_pre s e el
    affineLinear And AP argmax argmax_ Ax Ay Az colOf Finally Globally LTLFormula
    matMax matMin matrixOf matVecAffine Next Not Or reachable reachableSet ReactiveTS
    ReLu sem Shallow StateSet T toShallow toTS Trace traces TS Until
    max min abs id ite dite Matrix Real Int Nat Bool True False
""".split())


def lean_ident(name: str) -> str:
    """`name` as a Lean identifier: itself when it is one, else quoted, `«name»`."""
    if _IDENT.fullmatch(name) and name not in _KEYWORDS:
        return name
    if "«" in name or "»" in name:
        raise ValueError(f"`{name}` cannot be a Lean identifier")
    return f"«{name}»"


@dataclass(frozen=True)
class VarNames:
    """Source names by position, as `inputs.Names` holds them: `ctrl[i]` names state
    (interface) variable `i`, `extl[i]` external variable `i`, `None` where the input
    gives it none."""

    ctrl: tuple[str | None, ...]
    extl: tuple[str | None, ...]

    @classmethod
    def of(cls, names) -> "VarNames | None":
        """From `inputs.Names`; `None` when it names nothing."""
        if names is None or not any(n is not None for n in names.ctrl + names.extl):
            return None
        return cls(tuple(names.ctrl), tuple(names.extl))

    # -- how a predicate refers to them -------------------------------------

    @staticmethod
    def _ref(name: str) -> str:
        ident = lean_ident(name)
        return f"{NAMESPACE}.{ident}" if name in _ROOT or name in _KEYWORDS else ident

    def rename(self) -> dict[str, str]:
        """Internal variable (`s3`, `e1`, `el1`) → the abbreviation a predicate applies
        to its parameter (`px`, `w`, `w.latched`)."""
        out: dict[str, str] = {}
        for i, n in enumerate(self.ctrl):
            if n is not None:
                out[f"s{i}"] = self._ref(n)
        for i, n in enumerate(self.extl):
            if n is not None:
                out[f"e{i}"] = self._ref(n)
                out[f"el{i}"] = f"{self._ref(n)}.{LATCHED}"
        return out

    def simp_names(self) -> list[str]:
        """The abbreviations, qualified, for a proof's `simp only` set."""
        out = [f"{NAMESPACE}.{lean_ident(n)}" for n in self.ctrl if n is not None]
        for n in self.extl:
            if n is not None:
                out += [f"{NAMESPACE}.{lean_ident(n)}", f"{NAMESPACE}.{lean_ident(n)}.{LATCHED}"]
        return out

    # -- their definitions --------------------------------------------------

    def definitions(self, ctrl: list[Wire], extl: list[Wire], ctrl_type: str, extl_type: str) -> str:
        """The `namespace Var` block of abbreviations, over the state type `ctrl_type`
        (whose slots are `ctrl`) and the inputs' type `extl_type` (`(latched) ×
        (awaited)`, whose slots are `extl` on each side)."""

        def read(base: str, i: int, wires: list[Wire]) -> str:
            acc = f"{base}{product_accessor(i, len(wires))}"
            return f"{acc} 0 0" if wire_shape(wires[i]).is_scalar else acc

        lines = [f"namespace {NAMESPACE}", "noncomputable section", ""]
        for i, n in enumerate(self.ctrl):
            if n is not None:
                lines += [f"/-- `{n}`: state variable {i}. -/",
                          f"@[simp] abbrev {lean_ident(n)} (s : {ctrl_type}) := {read('s', i, ctrl)}"]
        for i, n in enumerate(self.extl):
            if n is not None:
                ident = lean_ident(n)
                lines += [f"/-- `{n}`: input {i}, read in the round (`(X {n})` in a claim). -/",
                          f"@[simp] abbrev {ident} (e : {extl_type}) := {read('e.2', i, extl)}",
                          f"/-- `{n}`: input {i}, read in the round before (`{n}` in a claim). -/",
                          f"@[simp] abbrev {ident}.{LATCHED} (e : {extl_type}) := {read('e.1', i, extl)}"]
        lines += ["", "end", f"end {NAMESPACE}"]
        return "\n".join(lines)
