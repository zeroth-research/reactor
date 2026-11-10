# Benchmarks: coverage of v2rm

How much of public SystemVerilog benchmark sets v2rm accepts, why it rejects the rest, and
whether the accepted designs agree with Yosys (`v2rm --yosys`).

Results: [`results/SUMMARY.md`](results/SUMMARY.md) (tables), `results/<set>.tsv` (one row per
case: raw / rewritten / Yosys result and bucket), `results/skipped-<set>.tsv` (every file not
measured, with its reason). They are committed, so the effect of a new feature shows as a diff.

## Run

```sh
./fetch.sh                                                         # sets.conf → sets/ (~45 MB)
CARGO_TARGET_DIR=~/.cache/rm-target cargo build --release -p verilog --bin v2rm
V2RM=~/.cache/rm-target/release/v2rm python3 -I tools/measure.py    # all sets, ~30 s on 8 cores
python3 -I tools/measure.py --sets verilog-eval --limit 10          # a subset
```

`V2RM` defaults to `~/.cache/rm-target/release/v2rm`. Yosys and z3 are found as by `v2rm --yosys`
(plus `~/tools/oss-cad-suite/bin` on the `PATH`).

The sets are **not** in this repository: [`sets.conf`](sets.conf) pins each one (URL + commit) and
[`fetch.sh`](fetch.sh) shallow-fetches them into `sets/` (git-ignored). They are third-party data
under their own licenses (sv-tests, sby, riscv-formal: ISC; verilog-eval: MIT; hw-cbmc: BSD-style)
and are only read: `tools/measure.py` passes their paths to `sed`, `v2rm`, Yosys and z3, and never
builds or executes anything from them. Run it with `python3 -I`, so Python does not import modules
from the current directory. Generated inputs (concatenated or rewritten files) go to `work/`
(git-ignored).

## What is measured

For each case:
- **raw**: `v2rm [--top T] FILE`. Several files (SBY jobs) are concatenated into one, since v2rm
  reads one file; nothing else changes.
- **rewritten**: the same after [`tools/rewrite.sed`](tools/rewrite.sed): `reg` → `logic`,
  `always @(posedge/negedge …)` → `always_ff`, `always @(*)` / `always @*` → `always_comb`. These
  only change spelling the subset rejects; the typing still checks the rest (only `<=` in
  `always_ff`, …).
- **Yosys**: `v2rm --yosys` on the first accepted variant (raw, else rewritten): `Equal` = every
  controlled variable equal to Yosys's in the init and the update round.
- Failures are classified by v2rm's **first** error only (v2rm stops there): the message is
  normalized (locations, names, numbers stripped) and mapped to a coarse bucket (`BUCKETS` in the
  script). Generic messages that only name an sv-parser node (`unsupported: ModuleCommonItem`,
  `syntax error`) get the first keyword of the source line added (``[at `initial`]``). So a bucket
  counts the blocker hit *first*; a design rejected for `initial` may need more features after it.
  To see what a feature unlocks, rerun after it lands.

## Selection (what is skipped, and why)

| set | measured | skipped |
|---|---|---|
| **VerilogEval v2** (`dataset_spec-to-rtl`) | all 156 reference solutions `*_ref.sv`, top `RefModule` | the 156 `*_test.sv` testbenches |
| **sv-tests** (`tests/`) | every `.sv` with a `module` that passes the filters | `:should_fail_because:` (should-fail tests), `:unsynthesizable: 1`, `:type:` without parsing/elaboration/simulation (preprocessor-only), `:defines:` (v2rm has no `-D`), no `module` (package/class/interface snippets), `tests/uvm/` and `tests/testbenches/` (infrastructure). Top: `:top_module:` if given, else v2rm's choice |
| **EBMC regression** (`hw-cbmc/regression/verilog`, `regression/ebmc`) | every `*.desc` marked `CORE` with a `.v`/`.sv` input; top from `--module`/`--top` | non-`CORE` (`KNOWNBUG`), non-Verilog inputs (SMV, AIGER, …), tests needing `-I`/`-D`/`-y`, and **duplicates**: several `.desc` run the same file and top with other engines/bounds; each (file, top) is measured once |
| **SymbiYosys examples** (`sby/docs/examples`) | every `.sby` job: the Verilog files of its `[files]` section, concatenated in order, top from `prep -top`/`hierarchy -top` | jobs without Verilog files (VHDL). `[tasks]` variants and `read -define` are ignored, `FORMAL` is **not** defined: `` `ifdef FORMAL `` blocks (the properties of most SBY examples) are dropped by the preprocessor, so an accepted SBY design is measured without its properties |

**Not measured:**
- **riscv-formal** (fetched): case-study size (RISC-V cores checked against the RVFI interface
  with SVA properties). A later step, once the subset handles a whole core (arrays, functions,
  generate, assertions).
- **RTLLM**: not chosen (an early probe accepted 1 of 50 designs: old Verilog style).
- `hw-cbmc/regression/{smv,vhdl,hw-cbmc,vlindex,rajdeep}`: other input languages, C
  co-verification, indexer tests.

## Caveats

- sv-tests are per-feature conformance snippets, many of them simulation tests built around
  `initial` + `$display` without being marked unsynthesizable: low coverage there mostly means
  "not a synthesizable design", not "missing feature". The per-bucket table is the useful part.
- EBMC tests mix in EBMC's own syntax (`always assert p: e`), counted as assertions.
- The Yosys column says nothing for designs Yosys itself rejects (e.g. `` `pragma ``); see the
  "Yosys check: not Equal" tables.
