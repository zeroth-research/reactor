# Use bash for better scripting features
# set shell := ["bash", "-eu", "-o", "pipefail", "-c"]
#
# Recipes are named by target: `rs-` for Rust, `py-` for Python, `lean-` for
# the `lean` project (the `reactor` command), `nb-` for notebooks; an
# unprefixed name means "all" (e.g. `test` = `rs-test` + `py-test` +
# `lean-test` + `nb-test`),
# and a `-fix` suffix marks the recipes that modify files in the repository —
# everything else is read-only towards the sources.
#
# Prerequisites are declared as native dependencies (not `@just` calls in the
# bodies), so shared ones run at most once per invocation: `just test` builds
# the python crate once, although both `py-test` and `nb-test` require it.

# Default recipe (runs when `just` is invoked with no args)
default:
    @just --list

# -------------------------------------------------
# Variables
# -------------------------------------------------
# Allow overriding from the CLI: `just PROFILE=release build`

PROFILE := ""
FEATURES := ""
# Run cargo through `uv` so the build/test env uses the project's Python
# and PyTorch — `theory` transitively links libtorch (via pyo3-tch/torch-sys),
# and `.cargo/config.toml` sets LIBTORCH_USE_PYTORCH=1, so cargo must see the
# venv's torch. If neccessary, override with `just CARGO=cargo ...` to use
# a bare toolchain.
CARGO := "uv run cargo"

# The uv venv is uv's default `.venv` in the repo root; a machine sharing the checkout
# with another OS points UV_PROJECT_ENVIRONMENT elsewhere in its own environment.
#
# The target dir is asked to plain cargo (not `uv run`, which would create a venv
# first) from the repo root and exported as an absolute path: a relative
# CARGO_TARGET_DIR (e.g. `target-linux`, to keep per-OS builds apart in a shared
# checkout) then means the same dir in recipes that `cd` elsewhere.
TARGET_DIR := `cargo metadata --no-deps --format-version 1 --offline | sed 's/.*"target_directory":"\([^"]*\)".*/\1/'`
export CARGO_TARGET_DIR := TARGET_DIR
# Bytecode goes in the target dir too, not in `__pycache__` next to the sources: a .pyc
# records its source's absolute path, so one written by another host of a shared
# checkout breaks `inspect.getsource` here. Pytest's rewritten test modules honour it too.
export PYTHONPYCACHEPREFIX := env("PYTHONPYCACHEPREFIX", TARGET_DIR / "pycache")
NBSTRIPOUT := "uvx nbstripout@0.9.1 --max-size 16k --drop-empty-cells"

# Convert FEATURES into a flag only if set

profile_flag := if PROFILE == "" { "" } else { "--profile " + PROFILE }
features_flag := if FEATURES == "" { "" } else { "--features " + FEATURES }

# -------------------------------------------------
# All (unprefixed = Rust + Python + notebooks)
# -------------------------------------------------

# Build everything: the Rust workspace (all targets and features) and the Python crate
build: rs-build-all py-build

# Run the test suite that needs no `lake` (Rust + Python + lean + notebooks)
test: rs-test py-test lean-test nb-test

# Run everything, including the Lean tests that build proofs via `lake`
test-all: test test-lean

# Run the Lean tests that build proofs via `lake` (marked `slow`; require lake + mathlib)
test-lean *args: (lean-run "pytest" "-m" "slow" args)

# `cargo clean` needs no torch, and running it under `uv run` would trigger a
# sync that can fail — cleaning the repo works fine outside the virtual env.

# Clean the current build (the cargo target dir)
clean:
    cargo clean

# Full rebuild from scratch
rebuild: clean build

# -------------------------------------------------
# Rust (rs-)
# -------------------------------------------------

# Build the Rust workspace in the default mode (use PROFILE and FEATURES variables to adjust)
rs-build:
    {{ CARGO }} build {{ profile_flag }} {{ features_flag }}

# Build the Rust workspace with all its targets and features. Split like `rs-test`:
# `pyo3-tch` forces pyo3's `extension-module`, so no binary links libpython, and
# only `theory` (via its build.rs) links it into its test binaries — with
# `--all-features` the other crates' test binaries would pull in `torch` and fail to link.
rs-build-all:
    {{ CARGO }} build --all-targets {{ profile_flag }}
    {{ CARGO }} build --all-targets -p theory --features torch,jax {{ profile_flag }}

# The test binaries of `theory` with `torch` link libpython and libtorch (without
# features the workspace uses no pyo3). `uv run` does not set up the dynamic-linker path, so we
# add both libdirs to it here, per-OS. Only a search path: on macOS, injecting libpython
# with DYLD_INSERT_LIBRARIES would also reach the build scripts' `python -c "import torch"`
# (torch-sys, theory), whose interpreter then crashes with a second libpython.
# The tests that embed Python import `torch` from the venv, so its site-packages go on
# PYTHONPATH: an embedded interpreter finds the venv on Linux through the `python3` on
# PATH, but on macOS it starts from the test binary's own path and never sees the venv.

# Run the Rust test suite (all features)
rs-test:
    #!/usr/bin/env bash
    set -euo pipefail
    TORCH_LIB="$(uv run python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
    PY_LIB="$(uv run python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
    export PYTHONPATH="$(uv run python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')${PYTHONPATH:+:$PYTHONPATH}"
    if [ "$(uname)" = "Darwin" ]; then
        export DYLD_FALLBACK_LIBRARY_PATH="$TORCH_LIB:$PY_LIB${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
    else
        export LD_LIBRARY_PATH="$TORCH_LIB:$PY_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    # Split by feature set: the `torch` and `jax` features are exclusive to `theory`, the
    # rest of the workspace is exercised without them. The JAX tests need JAX (the `jax`
    # dependency group), and test nothing without it.
    {{ CARGO }} test {{ profile_flag }}
    {{ CARGO }} test -p theory --features torch,jax {{ profile_flag }}

# Run clippy on the workspace on all targets and with all features (warnings are errors)
rs-clippy:
    {{ CARGO }} clippy --all-targets --all-features -- -D warnings

# `cargo fmt` needs no torch, so both fmt recipes skip the uv wrapper.

# Check formatting without modifying files (used by CI)
rs-fmt:
    cargo fmt --all -- --check

# Format the sources in place
rs-fmt-fix:
    cargo fmt --all

# -------------------------------------------------
# Python (py-)
# -------------------------------------------------

# Build the Python crate (maturin builds the workspace crates it depends on itself)
py-build:
    cd python && uv run maturin develop

# Build Python (from scratch) with all possible bindings
py-rebuild: clean rs-build-all py-build
    #DISABLED: build the `torch` crate and make sure it uses the libtorch that will
    #DISABLED:  be used also by the `python` crate
    #DISABLED: cd python && uv sync --no-build-package reactor --no-install-project
    #DISABLED: cd python && source .venv/bin/activate  && LIBTORCH_USE_PYTORCH=1 {{ CARGO }} build --package torch
    #DISABLED: cd python && uv run maturin develop  --features enable-torch
    @echo "Now you can go into the \`python\` directory and use \`uv run <script.py>\`"\
          "(or \`uv run python\` to get Python interpreter with \`reactor\` available)"

# Run all or a concrete python test (rebuilding the Python crate first)
py-test *args: (py-run "pytest" args)

# Run a command inside the `python` crate (with rebuilding the Python crate). The command given is executed from *within* the `python` crate, i.e., with paths relative to the root of the crate.
py-run *args: py-build
    cd python && uv run {{ args }}

# -------------------------------------------------
# lean (lean-)
# -------------------------------------------------

# Run all or a concrete `lean` test (rebuilding the Python crate first),
# excluding the Lean tests that need `lake` (marked `slow`; see `just test-lean`)
lean-test *args: (lean-run "pytest" "-m" "'not slow'" args)

# Run a command inside the `lean` project (with rebuilding the Python crate it builds on). The command given is executed from *within* `lean/`, i.e., with paths relative to it.
lean-run *args: py-build
    cd lean && uv run {{ args }}

# -------------------------------------------------
# Notebooks (nb-)
# -------------------------------------------------

# Install the tutorials dependency group
nb-sync:
    uv sync --group tutorials

# Build code and prepare for running tutorials
nb-build: build nb-sync

# Open the tutorial notebooks in Jupyter
nb-run: nb-build
    uv run jupyter notebook tutorials/

# mountaincar and pendulum are parked: they need non-linear ops (sin/cos/
# tanh/pow) and Stack, which no current theory provides — the analyzer
# raises on them by design.
# (REACTOR_NO_BROWSER keeps reactor.visual.show from opening browser pages during the run)

# Execute the tutorial notebooks against the current python crate without modifying them; fails on the first cell that errors
nb-test *notebooks="tutorials/counter.ipynb tutorials/gym_and_sugar.ipynb tutorials/decrement_1d/decrement_1d.ipynb tutorials/cairo/cairo.ipynb tutorials/singapore-2/Singapore-2.ipynb": py-build nb-sync
    REACTOR_NO_BROWSER=1 uv run jupyter nbconvert --to notebook --execute --stdout \
        --ExecutePreprocessor.timeout=600 {{ notebooks }} > /dev/null

# Fail when a committed notebook carries outputs the stripper would remove (used by CI)
nb-strip:
    {{ NBSTRIPOUT }} --verify $(git ls-files '*.ipynb')

# Strip bulky generated notebook outputs in place; text outputs under 16k (verification results, module prints) survive
nb-strip-fix:
    {{ NBSTRIPOUT }} $(git ls-files '*.ipynb')
