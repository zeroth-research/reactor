"""`Main.lean`, which `--executable` adds: the module as a program that
reads its inputs from stdin and prints its state each round.
"""

from __future__ import annotations

from reactor import Bool, Int, Word

from ..errors import Refused
from ..lowering.sorts import lean_elem_type, mat_mk, wire_shape
from ..codegen.context import LeanContext
from ..codegen.terms import build_tuple, product_type
from .names import LINTERS_OFF


def check_runnable(ctx: LeanContext) -> None:
    """Refuse a module that `Main.lean` could not run: one that computes
    with `Real` anywhere, a state, an input or a wire between two `Int`s.
    Its `init` and `update` are `noncomputable` then, and a `Real` could
    not be read or printed either. Asked before any of the project is
    written."""
    if ctx.uses_real:
        raise Refused(
            "--executable: the module computes with `Real`, which is "
            "noncomputable in Lean, so there is no program to run it"
        )


def _elem_parser(sort) -> "tuple[str, str, str]":
    """`(bind, parser, push)` for reading one element of `sort` -- a
    `Bool`, an `Int` or a `Word`, since `check_runnable` refused a `Real`
    -- from a token.

    The emitted line is `let v <bind> <parser> tokens[...]!`, so `bind` is
    `:=` for a pure parser and `←` for one in IO; `push` is the expression
    stored into the array.
    """
    if isinstance(sort, Bool):
        return ":=", "parseBool", "v"
    if isinstance(sort, Int):
        return "←", "parseIntOrFail", "v"
    assert isinstance(sort, Word), sort
    return "←", "parseIntOrFail", f"(BitVec.ofInt {sort.width} v)"


def _elem_shower(sort) -> str:
    """Lean function rendering one element of `sort` (as `_elem_parser`
    reads it) as a String."""
    return "showBool" if isinstance(sort, Bool) else "toString"


def generate_main_lean(ctx: LeanContext) -> str:
    """Generate Main.lean source that runs init/update in a stdin/stdout
    loop, for a module `check_runnable` passed."""
    extl_next, ctrl_next = ctx.extl_next, ctx.ctrl_next

    # Each input is read as its elements, one token each.
    total_extl_tokens = sum(wire_shape(w).total for w in extl_next)

    lines: list[str] = []

    # Imports. `init`/`update` are emitted into System/System.lean, which the
    # `System` lean_lib's root module pulls in.
    lines.append("import System")
    lines.append("")
    lines.append(LINTERS_OFF)
    lines.append("")

    # Static helpers
    lines.append(
        'def parseBool (s : String) : Bool := s.trimAscii.toString == "true" || s.trimAscii.toString == "1"'
    )
    lines.append('def showBool (b : Bool) : String := if b then "1" else "0"')
    lines.append("")
    lines.append("def parseIntOrFail (s : String) : IO Int := do")
    lines.append("  match s.trimAscii.toString.toInt? with")
    lines.append("  | some n => pure n")
    lines.append(
        '  | none => throw (.userError s!"Invalid integer: {s.trimAscii.toString}")'
    )
    lines.append("")
    lines.append(
        "def showMatrix {t : Type} (m n : Nat) (f : t → String) (mat : Matrix (Fin m) (Fin n) t)"
        " : String :="
    )
    lines.append(
        "  let vals := (List.ofFn fun i => List.ofFn fun j => [f (mat i j)]).flatten.flatten"
    )
    lines.append('  String.intercalate " " vals')
    lines.append("")

    # parseExtl
    lines.append(
        f"def parseExtl (tokens : Array String) : IO ({product_type(extl_next)}) := do"
    )
    lines.append(
        f'  if tokens.size < {total_extl_tokens} then throw (.userError "Expected {total_extl_tokens} input values")'
    )

    offset = 0
    parse_vars: list[str] = []
    for i, w in enumerate(extl_next):
        elem = lean_elem_type(w.dtype)
        var = f"e{i}"
        m, n = wire_shape(w).m, wire_shape(w).n
        # Every wire is a matrix here, so parse m*n tokens into an array of
        # the wire's own element type -- reading them all as Int made the
        # body disagree with the ascribed return type for every other sort.
        bind, parser, push = _elem_parser(w.dtype)
        lines.append(f"  let mut arr{i} : Array {elem} := #[]")
        lines.append(f"  for k in List.range {m * n} do")
        lines.append(f"    let v {bind} {parser} tokens[{offset} + k]!")
        lines.append(f"    arr{i} := arr{i}.push {push}")
        entry = f"arr{i}[i.val * {n} + j.val]!"
        lines.append(f"  let {var} : Matrix (Fin {m}) (Fin {n}) {elem} := {mat_mk(entry, 'i', 'j')}")
        offset += m * n
        parse_vars.append(var)

    # `product_type` orders components as the wires are declared, and
    # `product_accessor` projects them the same way, so the literal follows suit.
    lines.append(f"  pure {build_tuple(parse_vars)}")
    lines.append("")

    # showCtrl
    lines.append(f"def showCtrl (v : {product_type(ctrl_next)}) : String :=")
    # Destructure
    destr_vars: list[str] = []
    for i in range(len(ctrl_next)):
        destr_vars.append(f"v{i}")
    # Same order here: v{i} must bind ctrl_next[i], since the formatters
    # below pick showBool/toString/showMatrix from that wire's type.
    lines.append(f"  let {build_tuple(destr_vars)} := v")

    # Format each variable by its element sort: `dtype_to_lean_type` is a
    # `Matrix ...` here whatever the sort.
    show_parts: list[str] = []
    for i, w in enumerate(ctrl_next):
        var = f"v{i}"
        m, n = wire_shape(w).m, wire_shape(w).n
        show_parts.append(f"showMatrix {m} {n} {_elem_shower(w.dtype)} {var}")

    if len(show_parts) == 1:
        lines.append(f"  {show_parts[0]}")
    else:
        # Use s!"..." interpolation to join with spaces
        interp = " ".join(f"{{{p}}}" for p in show_parts)
        lines.append(f'  s!"{interp}"')
    lines.append("")

    # main function. `update` is emitted curried as
    # `update (ctrl) (extl_l) (extl_n)`, so the three groups are applied
    # separately -- and `extl_l` is the *previous* step's input, which the
    # loop therefore has to carry.
    lines.append("""\
def main : IO Unit := do
  let stdin ← IO.getStdin
  let line0 ← stdin.getLine
  if line0.trimAscii.toString.isEmpty then return
  let extl0 ← parseExtl (line0.trimAscii.toString.splitOn " " |>.toArray)
  let mut state := init extl0
  let mut extlPrev := extl0
  IO.println (showCtrl state)
  repeat do
    let line ← stdin.getLine
    if line.trimAscii.toString.isEmpty then break
    let extl ← parseExtl (line.trimAscii.toString.splitOn " " |>.toArray)
    let state' := update state extlPrev extl
    IO.println (showCtrl state')
    state := state'
    extlPrev := extl
""")

    return "\n".join(lines)
