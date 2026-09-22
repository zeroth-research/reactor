"""Load a Petrinaut SDCPN export (YAML or JSON) as a reactive module.

Each place becomes an integer state variable, so the marking is the module's state, and each
transition a guarded update: enabled when every input place holds at least the arc weight
(fewer, for an inhibitor arc), firing moves the tokens. Which transition fires is a
nondeterministic external input ``fire`` holding the transition's index; a disabled or
out-of-range choice stutters.

Only plain nets are supported: uncoloured tokens, trivially true predicate lambdas, no
dynamics, capacities, subnets or code scenarios. Anything else raises ``NotImplementedError``.
"""

import re
from dataclasses import dataclass

import yaml

from .zrth import LIA, Module, Term, Var, X
from .sort import Int
from .expr import expr, ite, collecting

INT = Int([1, 1])

_TRUE_LAMBDA = re.compile(r"^\s*export default Lambda\(\(\)\s*=>\s*true\);?\s*$")


@dataclass
class PetriNet:
    module: Module
    places: dict  # place name -> Var
    transitions: list  # transition names, indexed by the value of ``fire``
    fire: Var
    control: dict  # transition name -> its ``metadata.control`` (or None)
    marking: dict  # place name -> initial token count of the loaded scenario
    pre: list  # per transition, tokens consumed: place name -> weight (standard arcs)
    post: list  # per transition, tokens produced: place name -> weight


def load_sdcpn(path, scenario=None) -> PetriNet:
    """Load the net at ``path``; ``scenario`` picks the initial marking by name or id
    (default: the first scenario, or an empty marking if the file has none)."""
    with open(path) as f:
        doc = yaml.safe_load(f)
    if doc.get("version") != 1:
        raise ValueError(f"unsupported SDCPN file format version {doc.get('version')!r}")
    _require_plain(doc)

    ids = [p["id"] for p in doc["places"]]
    names = {p["id"]: p["name"] for p in doc["places"]}
    marking = _initial_marking(doc, scenario)
    var = {pid: Var(INT) for pid in ids}
    fire = Var(INT)

    with collecting() as init:
        for pid in ids:
            e = expr(marking.get(pid, 0), theory=LIA, sort=INT)
            init.append(Term(LIA.Id(), [X(var[pid])], [e.wire]))

    with collecting() as update:
        tokens = {pid: expr(var[pid], theory=LIA) for pid in ids}
        chosen = expr(fire, theory=LIA)
        nxt = dict(tokens)
        for i, t in enumerate(doc["transitions"]):
            enabled = chosen == i
            delta = {}
            for arc in t["inputArcs"]:
                pid, w, kind = arc["placeId"], arc.get("weight", 1), arc.get("type", "standard")
                enabled = enabled & (tokens[pid] < w if kind == "inhibitor" else tokens[pid] >= w)
                if kind == "standard":
                    delta[pid] = delta.get(pid, 0) - w
            for arc in t["outputArcs"]:
                delta[arc["placeId"]] = delta.get(arc["placeId"], 0) + arc.get("weight", 1)
            for pid, d in delta.items():
                if d:
                    nxt[pid] = ite(enabled, nxt[pid] + d, nxt[pid])
        for pid in ids:
            update.append(Term(LIA.Id(), [X(var[pid])], [nxt[pid].wire]))

    module = Module(vars=[*var.values(), fire], init=init, update=update)
    return PetriNet(
        module=module,
        places={names[pid]: var[pid] for pid in ids},
        transitions=[t["name"] for t in doc["transitions"]],
        fire=fire,
        control={t["name"]: (t.get("metadata") or {}).get("control") for t in doc["transitions"]},
        marking={names[pid]: marking.get(pid, 0) for pid in ids},
        pre=[{names[a["placeId"]]: a.get("weight", 1) for a in t["inputArcs"] if a.get("type", "standard") == "standard"}
             for t in doc["transitions"]],
        post=[{names[a["placeId"]]: a.get("weight", 1) for a in t["outputArcs"]} for t in doc["transitions"]],
    )


def _require_plain(doc):
    for key in ("types", "differentialEquations", "subnets", "componentInstances"):
        if doc.get(key):
            raise NotImplementedError(f"SDCPN {key} are not supported")
    for p in doc["places"]:
        if p.get("colorId") is not None or p.get("dynamicsEnabled") or p.get("capacity") is not None:
            raise NotImplementedError(f"place {p['name']!r}: coloured, dynamic or bounded places are not supported")
    for t in doc["transitions"]:
        if t.get("lambdaType", "predicate") != "predicate" or not _TRUE_LAMBDA.match(t.get("lambdaCode", "")):
            raise NotImplementedError(f"transition {t['name']!r}: only trivially true predicate lambdas are supported")
        for arc in t["inputArcs"] + t["outputArcs"]:
            if "placeId" not in arc:
                raise NotImplementedError(f"transition {t['name']!r}: component-port arcs are not supported")


def _initial_marking(doc, scenario):
    scenarios = doc.get("scenarios") or []
    if not scenarios:
        return {}
    if scenario is None:
        chosen = scenarios[0]
    else:
        matches = [s for s in scenarios if scenario in (s["id"], s["name"])]
        if not matches:
            raise KeyError(f"no scenario {scenario!r}; available: {[s['name'] for s in scenarios]}")
        chosen = matches[0]
    state = chosen["initialState"]
    if state["type"] != "per_place":
        raise NotImplementedError(f"scenario {chosen['name']!r}: only per_place initial states are supported")
    return {pid: int(count) for pid, count in state["content"].items()}
