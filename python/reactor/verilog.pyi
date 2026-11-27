"""Type stubs for `reactor.verilog` (a submodule of the native extension):
SystemVerilog designs translated into modules by the `verilog` crate."""

from os import PathLike
from typing import Literal, Mapping, Sequence

from .reactor import Module, Var

type Path = str | PathLike[str]
type Role = Literal["input", "clock", "init", "blackbox", "output", "internal"]

class VerilogError(Exception): ...
class UserError(VerilogError): ...
class UnsupportedError(VerilogError): ...
class InternalError(VerilogError): ...
class VerilogWarning(UserWarning): ...

def translate(
    files: Path | Sequence[Path],
    top: str | None = None,
    params: Mapping[str, int | str] | None = None,
) -> tuple[Module, dict[str, Var], dict[str, Role]]: ...

def _simulate(
    files: Path | Sequence[Path],
    rounds: Sequence[Mapping[str, int]],
    top: str | None = None,
    params: Mapping[str, int | str] | None = None,
) -> list[dict[str, int]]: ...
