"""The language model the LLM routes (`--infer ai`, `--infer ai-cegis`) ask.

For Claude API: pip install verith[ai]
For local LLMs (Ollama, vLLM, etc.) and OpenAI-compatible providers
(OpenRouter, ...): pip install verith[ai-local]

Neither package is imported until a client is made: `cli` imports
`DEFAULT_MODEL`.
"""

import os

from reactor_lean.core.certificate import Claim
from reactor_lean.core.errors import Refused

DEFAULT_MODEL = "claude-sonnet-4-6"

# Room for the answer *after* the reasoning. The generate prompt asks for two
# lines, but a model reasons in prose first and the lines come last, so a cap
# sized for the answer cuts the reply before it: at 1024 tokens 56 of the
# bench matrix's `--infer ai` cells ended mid-sentence, some inside
# `INVARIANT: fun s => s`, and failed to parse. Kept below the size at which
# the Anthropic SDK insists on streaming.
MAX_TOKENS = 8192


def require(package: str, who: str, install: str | None = None):
    """`package`, imported -- or a refusal saying `who` needs it and how to
    install it: `install`, or the package itself.

    A missing package is about how the run was asked for, not about the
    module, and the `ImportError` from wherever it happened to be imported
    first says neither which flag wanted it nor what to do.
    """
    import importlib

    try:
        return importlib.import_module(package)
    except ImportError as e:
        raise Refused(
            f"{who} needs the `{package}` package, which is not importable. "
            f"Install with: uv pip install {install or package}",
            setup=True,
        ) from e


def reply_fields(text: str, *fields: str) -> dict[str, str]:
    """The `FIELD: value` lines of a reply, by field, values stripped.

    Both LLM routes ask for their answer as such lines after whatever the
    model reasons first, so a line is read wherever it is and the last of
    each field wins: a model that corrects itself writes the correction
    after the mistake. A field the reply does not give is absent.
    """
    out: dict[str, str] = {}
    for line in text.splitlines():
        s = line.strip()
        for f in fields:
            if s.startswith(f"{f}:"):
                out[f] = s[len(f) + 1:].strip()
    return out


def source_block(source: str) -> str:
    """The module's Lean encoding, as the opening block of a prompt."""
    return f"Source code:\n```lean\n{source}\n```\n\n"


def describe_preconditions(claim: Claim) -> str:
    """`claim`'s preconditions, as the line of a prompt that states them."""
    parts = []
    if claim.init_pre is not None:
        parts.append(f"Precondition on init inputs (init_pre): {claim.init_pre}")
    if claim.update_pre is not None:
        parts.append(f"Precondition on update inputs (update_pre): {claim.update_pre}")
    if not parts:
        return "No preconditions on inputs."
    return "\n".join(parts)


def make_client(base_url: str | None, model: str):
    """A chat callable, `(system, user) -> str`, that reaches the model on
    its first question rather than up front: a run that asks nothing -- a
    certificate supplied whole is verified once -- needs neither the LLM
    package nor a key."""
    made = None

    def chat(system: str, user: str) -> str:
        nonlocal made
        if made is None:
            made = _connect(base_url, model)
        return made(system, user)

    return chat


def _connect(base_url: str | None, model: str):
    """`make_client`'s chat callable, with its client made now."""
    if base_url is not None:
        openai = require("openai", "--base-url", install="verith[ai-local]")
        # Hosted OpenAI-compatible providers (e.g. OpenRouter) need a real
        # key; local servers (Ollama, vLLM) accept any placeholder.
        api_key = (
            os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "unused"
        )
        client = openai.OpenAI(base_url=base_url, api_key=api_key)

        def chat(system: str, user: str) -> str:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content

        return chat
    else:
        anthropic = require("anthropic", "An LLM route", install="verith[ai]")
        # Without a key the request itself fails with a TypeError about
        # header resolution, which says nothing about the flag that needs
        # one. A run that never asks the model never gets here.
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise Refused(
                "ANTHROPIC_API_KEY is not set, and the default model is "
                "reached through the Anthropic API. Set it, or point "
                "--base-url at an OpenAI-compatible server (a local one "
                "needs no key).",
                setup=True,
            )
        client = anthropic.Anthropic()

        def chat(system: str, user: str) -> str:
            resp = client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text

        return chat
