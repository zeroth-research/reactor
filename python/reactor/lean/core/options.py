"""A command-line option that belongs to one row of a registry -- an
input format (`inputs`), say -- rather than to the command, and the three
rules every registry's options follow. A leaf module, so any registry can
import it."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Opt:
    """One command-line option that belongs to a row, not to the command.

    Given without its row selected it is an error rather than a no-op,
    said once here rather than by hand for each option of each row.
    """

    flags: tuple[str, ...]
    kwargs: dict                # forwarded to `add_argument`
    required: bool = False      # the row cannot run without it

    @property
    def dest(self) -> str:
        d = self.kwargs.get("dest")
        return d or self.flags[-1].lstrip("-").replace("-", "_")

    @property
    def default(self):
        """What the row says the option is when it is not given."""
        return self.kwargs.get("default")


# A row's options are parsed with `default=None`, so that "given" can be told
# from "not given": an option passed the very value its row defaults to still
# counts as given, which is what makes "only meaningful with its row" honest.
# These read that convention back, for any row's options.


def given(opt: Opt, args) -> bool:
    """Whether the command line `args` gives `opt`."""
    return getattr(args, opt.dest, None) is not None


def missing(options: Iterable[Opt], args) -> list[Opt]:
    """The `required` options among `options` that `args` leaves out."""
    return [o for o in options
            if o.required and getattr(args, o.dest, None) in (None, "")]


def put_defaults(options: Iterable[Opt], args) -> None:
    """Each of `options` that `args` leaves out, set to its row's default."""
    for o in options:
        if not given(o, args) and "default" in o.kwargs:
            setattr(args, o.dest, o.default)
