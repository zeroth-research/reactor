# Tutorials

Jupyter notebooks that walk through `zrth` step by step:

1. **counter.ipynb**: wrapping environments and neural networks, training a ranking function, formal verification with Z3
2. **pendulum.ipynb**: module composition with shared wires, training a controller, closed-loop verification
3. **mountaincar.ipynb**: wrapping an unmodified gymnasium environment (`MountainCarContinuous-v0`) and matching its behavior exactly
4. **gym_and_sugar.ipynb**: one ranking function verified identically against a gym environment and an equivalent `sugar.Module` — the two front-ends produce the same transition

Tutorials 2 and 3 build on concepts from Tutorial 1, **do them in order**.

## Reinforcement learning

- **rl_corridor.ipynb**: a corridor environment authored directly as a `sugar.Module`, a policy trained with REINFORCE against it, and the learned policy verified with Z3 (safety of the environment, progress of the policy).

## Petri nets

- **petri_queue_reachability.ipynb**: a bounded-queue Petri net modelled monolithically as a `sugar.Module` (marking = state, the firing choice = a non-deterministic external input); bounded reachability with Z3, returning a witness firing sequence for reachable markings.

## Quantised ranking functions

Train a neural ranking function and verify it as an **integer** (`LIA`) certificate. Shared
helpers (program, training, `quantise`, LIA `verify`) live in `_ranking.py`.

- **quantise_after_training.ipynb**: train in the reals, then quantise (post-training) — shows why fixed-point **scaling** is needed (naive rounding degenerates).

## Termination examples

Worked examples that model a C program as a `sugar.Module`, load a ranking function, and verify termination with Z3 (each in its own folder):

- **cairo/**: `while (x != 0) x = x - 1` — disjunctive guard; terminates over the integers.
- **decrement_1d/**: the simplest terminating loop, `while (x > 0) x = x - 1`.
- **singapore-2/**: a two-variable loop; verification carries the loop invariant `x + y <= 0`.

## Setup

Run

```
just nb-build
```

Or simply

```
just nb-run
```

which will do the necessary setup and run jupyter notebook automatically.

### Manual

From the project root:

```
uv sync --group tutorials
```

This installs `zrth` (built via maturin) plus the tutorial dependencies (`ipykernel`, `notebook`, `matplotlib`, `pygame`, `z3-solver`). Requires Python 3.12–3.13 and a Rust toolchain.

## Running

### VS Code

Install the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter), open a notebook, and select the `.venv` kernel.

`zrth` is built in place under `python/` (not installed into site-packages), so if the editor flags `import zrth` as unresolved, create `.vscode/settings.json` with:

```json
{
    "python.analysis.extraPaths": ["${workspaceFolder}/python"]
}
```

### uv

From the root directory, run:

```
uv run jupyter notebook tutorials/
```

### Jupyter directly

In the root directory, activate the venv first and then run Jupyter:

```
source .venv/bin/activate
jupyter notebook tutorials/
```
