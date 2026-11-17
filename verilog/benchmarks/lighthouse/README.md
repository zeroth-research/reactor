# Case study: the Lighthouse deck FPGA

A real, complete hardware design translated as a whole: the FPGA of the Bitcraze
[Lighthouse positioning deck](https://github.com/bitcraze/lighthouse-fpga) for the Crazyflie
quadcopter (Lattice iCE40UP5K). It receives the light sweeps of Lighthouse base stations
through four TS4231 light sensors. For each pulse it measures the timestamp and width, decodes
the beam's data bits, identifies the base station by an LFSR polynomial search, finds the offset
to the station's sync event (ROM lookup) and sends a 12-byte UART frame to the Crazyflie.

Results: [`../results/lighthouse.md`](../results/lighthouse.md) (table) and
`../results/lighthouse.tsv` (one row per module: v2rm result and time, Yosys result and time).
Times are rough: the Yosys checks run in parallel, so theirs depend on the machine's load.

## Files

| file | what |
|---|---|
| `generate.sh` | generates the Verilog from the SpinalHDL sources at a pinned commit, in Bitcraze's builder image (pinned by digest), into `../sets/lighthouse/` (git-ignored); `--check` only compares existing files with `expected.sha256` |
| `expected.sha256` | hashes of the generated files (`LighthouseTopLevel.v` without line 2, the generation date; the rest byte for byte), so a regeneration can be checked |
| `cells.v` | blackbox stubs of the vendor cells `pll`, `SB_IO` and `SB_WARMBOOT` (ports only, see below) |
| `measure.sh` | runs v2rm on every module as top (and on `ts4231Configurator.v` alone); with `--yosys` also the Yosys equivalence check of every accepted module; writes the results |

## Run

```sh
./generate.sh                  # needs git, docker and network; ~10 min (sbt)
CARGO_TARGET_DIR=~/.cache/rm-target cargo build --release -p verilog --bin v2rm
./measure.sh                   # acceptance only, ~1 min
./measure.sh --yosys           # plus the Yosys check: hours (the whole top ~2 h, mostly z3)
```

## Third-party code

The design is Bitcraze's code under their (L)GPL-3 license and is **not** in this repository:
`generate.sh` produces it locally. The SpinalHDL build runs only inside the container (nothing
of it is executed on the host); v2rm, Yosys and z3 only read the generated files.

## What the translation assumes

The files are used **as generated**, without edits, together with `cells.v`. A module marked
`(* blackbox *)` declares only ports and parameters; v2rm treats each output of a blackbox
instance as a free value in every time step, like a top-level input, and reads none of its
inputs (and warns once per instance); Yosys does the same with `cutpoint -blackbox`. This is
a sound over-approximation of the cells:
- `pll` (12 MHz in, 48 MHz out): its output is `Core_clk`, which thus becomes an input clock of
  its own; the 4:1 frequency ratio to `io_clk12MHz` is not modelled. `Slow_clk` is a register
  toggled on `Core_clk`, so a clock derived from it (related, not a crossing).
- `SB_IO` ×8, one per `inout` sensor pin (registered DDR input/output, tristate pin): the samples of the sensor pins
  (`D_IN_0` at the rising, `D_IN_1` at the falling edge) are free; the `inout` pins themselves
  are outside the translation.
- `SB_WARMBOOT`: inputs only (reboots the FPGA into another image).

Further assumptions, as for every design (see the crate documentation): one round of the
reactive module is one time step (a clock edge takes a round, so a clock cycle takes two);
data inputs meet setup and hold; at a derived clock's edge registers sample the old value
(hold timing met); the asynchronous reset release meets recovery/removal. The design's initial
values (SpinalHDL's power-on reset) fix the initial state. One variable bit select,
`config_value[15-config_bit_counter]` in `ts4231Configurator`, may go out of range in
principle (v2rm warns; it reads 0 there); it stays in range in every state reachable from
power-on.

## Results in short

All modules but one are accepted, including the whole top `LighthouseTopLevel` (about 1200
atoms, a few seconds), and every accepted module is equal to Yosys's reading of it from the
power-on state. `StreamCCByToggle` is rejected only **on its own**: there its two clocks are
unrelated inputs, and it passes data between them (a clock-domain crossing, not supported yet);
inside the top its second clock is derived from the first, and it is translated and checked as
part of the top.
