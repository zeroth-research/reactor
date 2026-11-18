# (S)DCPN Interface Design for Parameterized Verification

We describe an interface for constructing various flavors of (stochastic)
(dynamically-)coloured Petri nets. Its **goals** are to

1. translate Petri nets with **bounded token pools** into reactive modules for verification, as a prerequisite for
2. **parameterized verification** of inductive invariants via **small model theorems** (ideally, one theorem per flavor), which establish the token bounds sufficient for verification.

#### Non-goals

1. Immediately produce a frontend for every (future) (S)DCPN use case inside `reactor` (e.g., simulation, plots, ...).
1. Exact (S)DCPN semantics where they cannot be modeled in reactive modules; in particular the (S)DCPN transition firing rules R0-R4.
1. A final, fixed user-facing frontend for SDCPNs -- iterations on this are fine.


## Authoring

Places and transitions are Python specifications that compile down to reactive modules.  
Reuse (e.g., of shared color schemas) is possible via subclassing (inherts declarations and methods).  
The optional `@obj` helper instantiates the declaration and binds its name to the resulting object:

```python
from reactor import Int, LIA, Module, Var
from reactor.dcpn import (
    Place, ImmediateTransition, GuardedTransition, DCPN, obj,
)


@obj    # <- instantiate `Work` as singleton object `Work`
class Work(Place):
    progress = Var(Int([1, 1])) # <- controlled variable

    def update(self, progress): # <- updates, as for reactive modules
        return progress + 1


@obj
class Finished(Place):  # <- uncolored place
    pass


@obj
class Source(ImmediateTransition):
    inputs = ()
    outputs = (Work,)

    def produce(self):      # <- transition output
        # initialize one `Work`-colored token
        return (Work.Color(progress=0),)
                    # ^- `dataclass`-like interface synthesized by `Place`


@obj
class Complete(GuardedTransition):
    inputs = (Work,)
    outputs = (Finished,)

    def guard(self, token):
        return token.progress >= 3

    def produce(self, token):
        return (Finished.Color(),)


net = DCPN(
    theory=LIA,
    places=(Work, Finished),
    transitions=(Source, Complete),
)

# parameterized verification mock
for size in net.small_model_sizes():
    modules = net.to_modules(token_capacity=size)
    system = Module(*modules)
    verify(system)
```

`small_model_sizes()` returns the sizes sufficient for parameterized
verification under the applicable small-model theorem.

### Places and colours

#### Colour declarations

Controlled variables are declared as direct class-level `Var`s.

A small `__init_subclass__` hook collects declarations in definition order, and derives:

- An internal ordered `ctrl` tuple for positional method arguments/results.
- A frozen `Color` dataclass with the declared field names and no implicit
  defaults.

These are templates: compilation allocates fresh state for each token instance.
LIA accepts integer/Boolean colours; LRA accepts real/Boolean colours.

#### Discrete updates and flows

Conventions follow reactive modules:
- `update(*colours, *extl)` describes local discrete evolution while resident. Omission stutters the state.
- `flow(*colours, *extl)` describes dynamics; omission means zero flow.
-  Return values follow `ctrl` order.

Note that there is no `Place.init`: transitions supply produced tokens/colours.

#### External variables

External variables are constructor inputs. For example, a place moving at two
units per unit of external time:

```python
from reactor import LRA, Real
from reactor.sugar import d


t = Var(Real([1, 1]))


@obj(extl=(t,))
class Moving(Place):
    position = Var(Real([1, 1]))

    def flow(self, position, t):
        return 2 * d(t)
```

Use this place in a `DCPN(theory=LRA, ...)`; a source transition can supply `Moving.Color(position=...)`.

### Transitions

Connections are ordered lists of place objects. Our priority is to implement input/output arcs:

| Attribute    | Meaning                                     |
| ------------ | ------------------------------------------- |
| `inputs`     | Select and consume one token per occurrence |
| `outputs`    | Produce exactly one token per occurrence    |

and support the three flavors of SDCPN transitions:

| Transition class       | Required methods    |
| ---------------------- | ------------------- |
| `ImmediateTransition`  | `produce` if outputs are declared |
| `GuardedTransition`    | `guard`; `produce` if outputs are declared |
| `StochasticTransition` | `rate`; `produce` if outputs are declared |

`guard`, `produce`, and `rate` receive one generated `Color` record per selected
input token after `self`. Consumed-token records come first, in `inputs` order,
followed by enabling-token records in `enabling` order. Inhibiting places
contribute no arguments.

For example, `inputs = (Work, Work)` selects two distinct tokens, passed as
`token_a` and `token_b` to `guard(self, token_a, token_b)` (and likewise to
`produce` or `rate`).

#### Token production

`produce` is required only when `outputs` is nonempty. Without output places,
it may be omitted; if defined, it must return an empty tuple.

`produce` returns a tuple aligned with `outputs`. Each entry must be the
destination place's `Color` record and produces exactly one token. `None` entries
are not allowed. The empty record `Place.Color()` produces an uncoloured token.
Transitions with `outputs = ()` produce no tokens.

#### Enabling and inhibiting arcs

Enabling/inhibiting arcs are second priority:

| Attribute    | Meaning                               |
| ------------ | ------------------------------------- |
| `enabling`   | Select/read without consuming         |
| `inhibiting` | Require the listed places to be empty |

### Theories

We parameterize each net by the colour theory, and pursue implementation in this order:
1. LIA with discrete updates first, then
2. LRA with continuous flows.

### The `@obj` helper

We only write classes for grouping methods in their bodies (and occasional reuse), but
in principle we want all classes to be instantiated as singletons. The `@obj` helper:
- calls the class constructor, and binds the object to the class name
- does not enforce singleton identity
- does not compile anything
-  `@obj(extl=(...))` forwards constructor keywords
- explicit instantiation remains supported

## Translation

`DCPN` describes the net for an arbitrary number of tokens.
`to_modules(token_capacity=N)` lowers it to reactive modules, with **N token
modules** plus **modules for transition coordination**.

### Tokens and colours

Each token module owns its place as a discrete mode and a separate colour block
for every place:

```text
(place ∈ {⊥, P₁, …, Pₙ}, P₁.colours, …, Pₙ.colours)
```

where `⊥` is a special place indicating that the token is absent.

For now, we support only the empty initial marking: no tokens are present and
transitions populate the net. Thus,

- Initialize all slots at `⊥`, with zeroed or arbitrary colour blocks (will be overwritten by transition outputs).
- `update`/`flow` dispatch by the token's resident place; inactive places have stuttering / zero flow behavior.
- Transition events remove or produce tokens. Since the token modules have to await these events, transitions
  take precedence over a resident place update in that round.

### Transitions

The transition module(s) read token state and control firing events. Only token
modules write token state. Consumed token slots may be reused by produced
tokens; additional outputs use absent slots and surplus consumed slots become
absent.

Each firing is atomic: its events and destination/colour payloads must affect
all participating token modules in the same reactive-module round, without an
additional round of communication latency.

#### Open design questions

- One transition module for the whole Petri net, or one reactive module per transition (we can explore both options in the translation).
- Can multiple independent firings share one reactive module round?
- Triggering an error if produced tokens exceed available slots.

## Implementation

We pursue implementation in stages:

1. **LIA:** discrete `update`, immediate/guarded transitions, bounded translation. Reject unsupported concepts.
2. **LRA:** add real colours and continuous `flow` alongside `update`.
3. **Stochastic transitions:** add stochastic transitions in addition to immediate and guarded transitions.
4. **Enabling/inhibiting arcs:** read-only bindings and empty-place tests -- largely additions to the transition translation.
