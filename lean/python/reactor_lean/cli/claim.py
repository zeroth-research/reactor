"""What the run is about: the module, read from its input (`load`), and
the claim the flags ask (`claim_from_flags`).
"""

from __future__ import annotations

from .settings import Settings
from ..core.certificate import Certificate, Claim
from .. import inputs


def load(settings: Settings) -> "inputs.Problem":
    """The module the run is about, and the input it was read from.

    The loader refuses a path that is not there or cannot be read, a file
    that is not a module and a `-d` the file does not define, all as
    `InputError`. All of them are the user's input, so they are reported
    rather than raised. What the module file's own code raises is not a
    refusal, and keeps its traceback: an `AttributeError` from a typo in
    `module()` reads, as one `error:` line, exactly like "must define
    `module`", and an `OSError` from a file it opens like a missing input.
    """
    try:
        return inputs.load(settings.module_file, settings.input_opts,
                           format=settings.input_format)
    except (inputs.ClaimError, inputs.InputError) as e:
        raise SystemExit(f"error: {e}") from e


def claim_from_flags(
    settings: Settings, names: "inputs.Names",
) -> "tuple[Claim | None, Certificate]":
    """What the flags ask to be proved, and what of its certificate they
    supply. No claim when no predicate flag was passed.

    The flags are in the module's own terms (`inputs.names`); the claim
    is in the internal names every later stage reads, and each flag is
    printed in them once, since that is how the logs and
    any refusal will spell it."""
    if not (settings.prp or settings.pre or settings.invariant
            or settings.ranking):
        return None, Certificate()
    prp_flag = "--safety" if settings.is_safety else "--buchi"

    def internal(text: "str | None", flag: str) -> "str | None":
        if not text:
            return None
        try:
            spelled = inputs.to_internal(text, names, flag)
        except inputs.ClaimError as e:
            raise SystemExit(f"error: {e}") from e
        print(f".. {flag}, in internal names: {spelled}")
        return spelled

    prp = internal(settings.prp, prp_flag)
    pre = internal(settings.pre, "--pre")
    claim = Claim(prp=prp, kind=settings.kind, init_pre=pre, update_pre=pre)
    return claim, Certificate(internal(settings.invariant, "--invariant"),
                              internal(settings.ranking, "--ranking"))
