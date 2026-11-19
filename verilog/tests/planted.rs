//! The planted-bug suite: every mutant of every example (see
//! `mutate::mutants`) is translated by us and by Yosys and compared with
//! the original by the miter in both directions, and by random simulation.
//!
//! It takes about two minutes, so it is marked `#[ignore]` and left out of
//! a plain `cargo test -p verilog`. The full run includes it:
//!
//! ```text
//! cargo test -p verilog -- --include-ignored
//! ```
//!
//! The mutation tool itself (`tests/mutate/mod.rs`) has direct tests here,
//! which always run. The suite needs Yosys and z3: it is skipped with a
//! message without them, and fails instead with `V2RM_REQUIRE_TOOLS=1`.

mod common;
mod mutate;

use common::{Rand, TempDir, examples, load, seed, stem, tools};
use mutate::{Mutant, mutants};
use std::path::Path;
use verilog::ast::Ty;
use verilog::ast_interp::{Interp, Values};
use verilog::elab::{self, Inst};
use verilog::lower::{Translation, lower_inst};
use verilog::miter::{self, Side, Verdict};
use verilog::yosys;

/// The first difference random simulation finds between the reference
/// semantics of `inst` and of `mutant`, on the same random initial
/// registers (those without initial values) and inputs (clocks at random levels): the values of every
/// signal in every round, including the first.
fn simulated_difference(inst: &Inst, mutant: &Inst, rand: &mut Rand) -> Option<String> {
    for _ in 0..10 {
        let init = rand.registers(inst);
        let mut original = Interp::new(inst, &init).unwrap();
        let mut edited = match Interp::new(mutant, &init) {
            Ok(i) => i,
            Err(e) => return Some(format!("the mutant does not start: {e}")),
        };
        let inputs: Vec<(String, Ty)> = original.inputs().to_vec();
        for k in 0..30 {
            let ins: Values = inputs
                .iter()
                .map(|(n, ty)| (n.clone(), rand.bits(ty.width)))
                .collect();
            let a = original.step(&ins).unwrap();
            let b = match edited.step(&ins) {
                Ok(b) => b,
                Err(e) => return Some(format!("round {k}: {e}")),
            };
            if a != b {
                let name = a.keys().chain(b.keys()).find(|n| a.get(*n) != b.get(*n));
                return Some(format!(
                    "round {k}: `{}`, from {init:?} with inputs {ins:?}",
                    name.unwrap()
                ));
            }
        }
    }
    None
}

/// What one direction of a mutant gives.
#[derive(Debug, PartialEq)]
enum Found {
    /// `build` rejects the pair: the registers or names differ.
    Mismatch,
    /// These variables differ, each replayed as the expected side.
    Differs(Vec<String>),
    Equal,
}

/// The miter of `t` and `ys`, with a counterexample and replay against
/// the reference on `inst` for every differing variable, which must be
/// attributed to `side`.
fn direction(
    z3: &Path,
    inst: &Inst,
    t: &Translation,
    ys: &yosys::Smt2,
    side: Side,
) -> Result<Found, String> {
    let Ok(m) = miter::build(t, ys) else {
        return Ok(Found::Mismatch);
    };
    let mut differs = vec![];
    for (phase, name, verdict) in miter::check(z3, &m)? {
        match verdict {
            Verdict::Equal => {}
            Verdict::Unknown => return Err(format!("`{name}` in {phase:?}: unknown")),
            Verdict::Differs => {
                let q = m
                    .queries
                    .iter()
                    .find(|q| q.phase == phase && q.name == name);
                let q = q.unwrap();
                let cex = miter::counterexample(z3, &m, q)?;
                let r = miter::replay(inst, t, &cex)?;
                if r.side != side {
                    return Err(format!("{cex}\n  {r}"));
                }
                differs.push(format!("{name} in {phase:?}"));
            }
        }
    }
    Ok(if differs.is_empty() {
        Found::Equal
    } else {
        Found::Differs(differs)
    })
}

/// What a mutant gives.
#[derive(Debug, PartialEq)]
enum Outcome {
    /// Our frontend or typing rejects it.
    Rejected,
    /// Both directions find no difference, and neither does simulation.
    Equivalent,
    /// Both directions find a difference (or mismatching registers).
    Caught,
}

/// Plants mutant `m` of the example at `path` (hierarchy `inst`, our
/// translation `t`, Yosys's `ys`) on each side in turn:
/// - in ours: `t` is lowered from the mutant and compared with `ys`; every
///   difference must replay as Yosys's side being right (`Theirs`);
/// - in Yosys's: Yosys translates the mutant and is compared with `t`;
///   every difference must replay as ours being right (`Ours`).
///
/// The reference always runs the original. Both directions must agree on
/// whether there is a difference, and so must random simulation of the
/// original against the mutant, an oracle independent of the miter. Fails
/// with a description if not.
fn plant(
    (yosys, z3): (&Path, &Path),
    dir: &Path,
    path: &Path,
    (inst, t, ys): (&Inst, &Translation, &yosys::Smt2),
    m: &Mutant,
    rand: &mut Rand,
) -> Result<Outcome, String> {
    let Ok(edited) = elab::load(&m.source, path) else {
        return Ok(Outcome::Rejected);
    };
    let Ok(ours) = lower_inst(&edited) else {
        return Ok(Outcome::Rejected);
    };
    let file = dir.join(format!(
        "mutant-{}",
        path.file_name().unwrap().to_string_lossy()
    ));
    std::fs::write(&file, &m.source).unwrap();
    let theirs = yosys::run(yosys, &file, &t.name, &[]).map_err(|e| format!("Yosys: {e}"))?;

    let a = direction(z3, inst, &ours, ys, Side::Theirs).map_err(|e| format!("in ours: {e}"))?;
    let b = direction(z3, inst, t, &theirs, Side::Ours).map_err(|e| format!("in Yosys's: {e}"))?;
    let simulated = simulated_difference(inst, &edited, rand);
    match (a == Found::Equal, b == Found::Equal, simulated) {
        (true, true, None) => Ok(Outcome::Equivalent),
        (false, false, _) => Ok(Outcome::Caught),
        (true, true, Some(d)) => Err(format!("missed in both directions: {d}")),
        _ => Err(format!(
            "the directions disagree: in ours {a:?}, in Yosys's {b:?}"
        )),
    }
}

/// The mutants of the examples that change nothing the miter can see,
/// each with the reason.
const EQUIVALENT: &[(&str, &str, &str)] = &[
    (
        "addc.sv",
        "9:22 `9` -> `8`",
        "`8'(a)` keeps all of the 8-bit `a`, extended in the 9-bit sum anyway",
    ),
    (
        "addc.sv",
        "9:30 `9` -> `8`",
        "`8'(b)` keeps all of the 8-bit `b`, extended in the 9-bit sum anyway",
    ),
    (
        "addc.sv",
        "9:38 `9` -> `8`",
        "`8'(cin)` extends `cin` with zeros, as the 9-bit sum does anyway",
    ),
    (
        "arinst.sv",
        "12:15 `4` -> `'d4`",
        "`W` is an `int`: the override takes the value 4 either way",
    ),
    (
        "arinst.sv",
        "13:15 `6` -> `'d6`",
        "`W` is an `int`: the override takes the value 6 either way",
    ),
    (
        "arinst.sv",
        "17:21 `4` -> `3`",
        "every instance of `rcount` overrides this default of `W`",
    ),
    (
        "arinst.sv",
        "17:21 `4` -> `'d4`",
        "every instance of `rcount` overrides this default of `W`",
    ),
    (
        "arinst.sv",
        "18:21 `-` -> `+`",
        "a wider `INIT` holds the same value, truncated when assigned to `q`",
    ),
    (
        "arinst.sv",
        "18:22 `1` -> `0`",
        "a wider `INIT` holds the same value, truncated when assigned to `q`",
    ),
    (
        "arinst.sv",
        "18:37 `0` -> `'d0`",
        "the cast `W'(…)` of 0 is 0 either way",
    ),
    (
        "arinst.sv",
        "23:22 `-` -> `+`",
        "a wider `ONE` is still 1, the sum truncated when assigned to `q`",
    ),
    (
        "arinst.sv",
        "23:23 `1` -> `0`",
        "a wider `ONE` is still 1, the sum truncated when assigned to `q`",
    ),
    (
        "arinst.sv",
        "23:37 `1` -> `'d1`",
        "the cast `W'(…)` of 1 is 1 either way",
    ),
    (
        "arlow.sv",
        "18:15 `0` -> `'d0`",
        "an unsized based index `'dN` is a variable index of the same value as the constant `N`, so the same bits are read",
    ),
    (
        "arlow.sv",
        "18:21 `1` -> `'d1`",
        "an unsized based index `'dN` is a variable index of the same value as the constant `N`, so the same bits are read",
    ),
    (
        "barrel.sv",
        "18:22 `>>>` -> `>>`",
        "`y` is unsigned: `>>>` shifts in zeros",
    ),
    (
        "bbchild.sv",
        "6:38 `1` -> `0`",
        "the default of a blackbox parameter, which has no effect",
    ),
    (
        "bbchild.sv",
        "6:38 `1` -> `'d1`",
        "the default of a blackbox parameter, which has no effect",
    ),
    (
        "bbchild.sv",
        "17:35 `8` -> `'d8`",
        "`filter`'s default `W` as an unsigned 32-bit value: the same widths and casts",
    ),
    (
        "bbchild.sv",
        "24:18 `4` -> `3`",
        "the value of a blackbox parameter, which has no effect",
    ),
    (
        "bbchild.sv",
        "24:18 `4` -> `'d4`",
        "the value of a blackbox parameter, which has no effect",
    ),
    (
        "bbchild.sv",
        "47:15 `10` -> `'d10`",
        "the same parameter value, unsigned",
    ),
    (
        "bbchild.sv",
        "51:20 `10` -> `9`",
        "`9'(b)` keeps all of the 8-bit unsigned `b`, extended to 10 bits anyway",
    ),
    (
        "bbtop.sv",
        "16:25 `5` -> `4`",
        "the type of a blackbox parameter, which has no effect",
    ),
    (
        "bbtop.sv",
        "16:41 `6'b000000` -> `6'd1`",
        "the default of a blackbox parameter, which has no effect",
    ),
    (
        "bbtop.sv",
        "27:43 `1` -> `0`",
        "the width of a blackbox input, which nothing reads",
    ),
    (
        "bbtop.sv",
        "43:19 `6'b110000` -> `6'd49`",
        "the value of a blackbox parameter, which has no effect",
    ),
    (
        "bbtop.sv",
        "44:42 `0` -> `1`",
        "a blackbox input, which nothing reads",
    ),
    (
        "bbtop.sv",
        "44:42 `0` -> `'d0`",
        "a blackbox input, which nothing reads",
    ),
    (
        "bbtop.sv",
        "44:51 `1'b0` -> `1'd1`",
        "a blackbox input, which nothing reads",
    ),
    (
        "bbtop.sv",
        "46:19 `6'b110000` -> `6'd49`",
        "the value of a blackbox parameter, which has no effect",
    ),
    (
        "bbtop.sv",
        "47:39 `&` -> `|`",
        "a blackbox input, which nothing reads",
    ),
    (
        "bbtop.sv",
        "49:22 `==` -> `!=`",
        "a blackbox input, which nothing reads",
    ),
    (
        "bbtop.sv",
        "49:25 `4'hf` -> `4'd0`",
        "a blackbox input, which nothing reads",
    ),
    (
        "bcd.sv",
        "16:38 `3` -> `2`",
        "`h` is at most 2, so its bit 3 is 0, the zero the assignment extends with",
    ),
    (
        "bytes.sv",
        "15:24 `7` -> `'d7`",
        "an unsized based index `'dN` is a variable index of the same value as the constant `N`, so the same bits are read",
    ),
    (
        "bytes.sv",
        "16:18 `16` -> `15`",
        "sign-extending to 15 bits, then to 16, is sign-extending to 16",
    ),
    (
        "bytes.sv",
        "17:17 `$signed` -> ``",
        "`==` of two 16-bit operands does not depend on their signedness",
    ),
    (
        "bytes.sv",
        "17:35 `16` -> `15`",
        "sign-extending to 15 bits, then to 16, is sign-extending to 16",
    ),
    (
        "carry.sv",
        "15:16 `9` -> `'d9`",
        "the comparison with the unsigned `cnt` is unsigned either way: a nonnegative constant: unsigned or `int`, the value at the width it is used at is the same",
    ),
    (
        "carry.sv",
        "15:26 `0` -> `'d0`",
        "a nonnegative constant: unsigned or `int`, the value at the width it is used at is the same",
    ),
    (
        "carry.sv",
        "16:23 `1` -> `'d1`",
        "`cnt + 1` is unsigned either way: a nonnegative constant: unsigned or `int`, the value at the width it is used at is the same",
    ),
    ("castparam.sv", "8:14 `3` -> `2`", "`S0` is 0 at any width"),
    (
        "castparam.sv",
        "8:24 `0` -> `'d0`",
        "a nonnegative constant: unsigned or `int`, the value at the width it is used at is the same",
    ),
    (
        "castparam.sv",
        "9:31 `1` -> `'d1`",
        "`M` is typed: `-'d1` truncated to 8 bits is eight ones, as `-1` is",
    ),
    (
        "castparam.sv",
        "19:17 `16` -> `15`",
        "the sum of two 8-bit values fits in 15 bits",
    ),
    (
        "castparam.sv",
        "20:27 `1` -> `'d1`",
        "the shift amount is unsigned anyway",
    ),
    (
        "consts.sv",
        "10:17 `9` -> `'d9`",
        "`N` is used only in a range, where the unsigned 8 gives the same width",
    ),
    (
        "consts.sv",
        "10:32 `1` -> `'d1`",
        "`N` is used only in a range, where the unsigned 8 gives the same width",
    ),
    (
        "consts.sv",
        "11:23 `16` -> `'d16`",
        "`W` is typed: it takes the value 16 either way",
    ),
    (
        "consts.sv",
        "12:18 `3` -> `'d3`",
        "`S` is used only in `1 << S` and `S > 2`, which give the same values for the unsigned 3",
    ),
    (
        "consts.sv",
        "13:18 `1` -> `'d1`",
        "`L` is used only in a range, where the unsigned 8 gives the same width",
    ),
    (
        "consts.sv",
        "14:20 `>` -> `>=`",
        "`S` is 3: `S >= 2` is true as `S > 2` is",
    ),
    (
        "consts.sv",
        "14:22 `2` -> `1`",
        "`S` is 3: `S > 1` is true as `S > 2` is",
    ),
    (
        "consts.sv",
        "14:22 `2` -> `'d2`",
        "`K` is used only in ranges, and the comparison gives the same bit",
    ),
    (
        "consts.sv",
        "14:26 `1` -> `'d1`",
        "`K` is used only in ranges, where the unsigned 1 gives the same width",
    ),
    (
        "consts.sv",
        "14:30 `0` -> `1`",
        "`S > 2` holds, so the else arm is never taken",
    ),
    (
        "consts.sv",
        "14:30 `0` -> `'d0`",
        "`S > 2` holds, so the else arm is never taken",
    ),
    (
        "consts.sv",
        "35:16 `65536` -> `'d65536`",
        "the operands are positive: unsigned, the 64-bit product is 2^32 too",
    ),
    (
        "consts.sv",
        "35:24 `65536` -> `'d65536`",
        "the operands are positive: unsigned, the 64-bit product is 2^32 too",
    ),
    (
        "consts.sv",
        "36:16 `2` -> `'d2`",
        "the base is positive: unsigned, the 64-bit power is 2^31 too",
    ),
    (
        "consts.sv",
        "36:21 `31` -> `'d31`",
        "the exponent is self-determined and positive: the power is the same",
    ),
    (
        "consts.sv",
        "38:25 `3` -> `'d3`",
        "`-'d3` has the bits of -3 at 40 bits, and the power keeps 40 bits either way",
    ),
    (
        "consts.sv",
        "38:31 `3` -> `'d3`",
        "the exponent is self-determined and positive: the power is the same",
    ),
    (
        "ctxwidth.sv",
        "20:30 `1` -> `'d1`",
        "the shift amount is unsigned anyway",
    ),
    (
        "ctxwidth.sv",
        "21:29 `1` -> `'d1`",
        "the shift amount is unsigned anyway",
    ),
    (
        "decode.sv",
        "16:13 `3'd0` -> `3'd1`",
        "both labels are shadowed by earlier items",
    ),
    (
        "ffblock.sv",
        "24:14 `0` -> `'d0`",
        "an unsized based index `'dN` is a variable index of the same value as the constant `N`, so the same bits are read",
    ),
    (
        "fsmint.sv",
        "12:21 `0` -> `'d0`",
        "the states are compared with the unsigned `st` and assigned to it: a nonnegative constant: unsigned or `int`, the value at the width it is used at is the same",
    ),
    (
        "fsmint.sv",
        "12:30 `1` -> `'d1`",
        "the states are compared with the unsigned `st` and assigned to it: a nonnegative constant: unsigned or `int`, the value at the width it is used at is the same",
    ),
    (
        "fsmint.sv",
        "12:40 `2` -> `'d2`",
        "the states are compared with the unsigned `st` and assigned to it: a nonnegative constant: unsigned or `int`, the value at the width it is used at is the same",
    ),
    (
        "fsmint.sv",
        "17:13 `3` -> `'d3`",
        "the labels are compared with the unsigned `st`: a nonnegative constant: unsigned or `int`, the value at the width it is used at is the same",
    ),
    (
        "fsmint.sv",
        "22:7 `4'd9` -> `4'd10`",
        "the 3-bit `k` matches neither label",
    ),
    (
        "fsmint.sv",
        "22:19 `1` -> `0`",
        "the label `4'd9` never matches the 3-bit `k`",
    ),
    (
        "fsmint.sv",
        "22:19 `1` -> `'d1`",
        "the label `4'd9` never matches the 3-bit `k`",
    ),
    (
        "fsmint.sv",
        "23:7 `1` -> `'d1`",
        "the labels are compared with the unsigned `k`: a nonnegative constant: unsigned or `int`, the value at the width it is used at is the same",
    ),
    (
        "fsmint.sv",
        "23:10 `2` -> `'d2`",
        "the labels are compared with the unsigned `k`: a nonnegative constant: unsigned or `int`, the value at the width it is used at is the same",
    ),
    (
        "fsmint.sv",
        "24:22 `0` -> `'d0`",
        "a nonnegative constant: unsigned or `int`, the value at the width it is used at is the same",
    ),
    (
        "initinst.sv",
        "4:36 `4` -> `'d4`",
        "the default width is the same as an unsigned 32-bit value",
    ),
    (
        "initinst.sv",
        "4:57 `-` -> `+`",
        "the type of `INIT` only resizes the always overridden value (`4'd3` and `6'd60` fit)",
    ),
    (
        "initinst.sv",
        "4:58 `1` -> `0`",
        "the type of `INIT` only resizes the always overridden value (`4'd3` and `6'd60` fit)",
    ),
    (
        "initinst.sv",
        "4:70 `'0` -> `'1`",
        "the default `INIT` is always overridden",
    ),
    (
        "initinst.sv",
        "4:93 `0` -> `1`",
        "the default `ID` is always overridden",
    ),
    (
        "initinst.sv",
        "4:93 `0` -> `'d0`",
        "the default `ID` is always overridden",
    ),
    (
        "initinst.sv",
        "21:16 `6` -> `'d6`",
        "a width of 6 either way",
    ),
    (
        "initinst.sv",
        "21:38 `5` -> `'d5`",
        "5 either way, assigned to the 4-bit `id`",
    ),
    (
        "initinst.sv",
        "22:30 `9` -> `'d9`",
        "9 either way, assigned to the 4-bit `id`",
    ),
    (
        "initrst.sv",
        "24:36 `0` -> `'d0`",
        "an unsized based index `'dN` is a variable index of the same value as the constant `N`, so the same bits are read",
    ),
    (
        "mac.sv",
        "13:23 `16'sd0` -> `16'd0`",
        "zero is zero at any signedness",
    ),
    (
        "mac.sv",
        "14:23 `16` -> `15`",
        "sign-extending to 15 bits, then to 16, is sign-extending to 16",
    ),
    (
        "mac.sv",
        "14:32 `16` -> `15`",
        "sign-extending to 15 bits, then to 16, is sign-extending to 16",
    ),
    (
        "memcomb.sv",
        "14:17 `4` -> `'d4`",
        "an array size: the unsigned 4 gives the same size",
    ),
    (
        "memcomb.sv",
        "25:13 `1` -> `'d1`",
        "a constant element index: the unsigned 1 picks the same element",
    ),
    (
        "memcomb.sv",
        "25:19 `0` -> `'d0`",
        "a constant element index: the unsigned 0 picks the same element",
    ),
    (
        "memcomb.sv",
        "26:18 `0` -> `'d0`",
        "an unsized based index `'dN` is a variable index of the same value as the constant `N`, so the same bits are read",
    ),
    (
        "meminst.sv",
        "4:23 `4` -> `3`",
        "every instance of `mem` overrides this default of `DEPTH`",
    ),
    (
        "meminst.sv",
        "4:23 `4` -> `'d4`",
        "every instance of `mem` overrides this default of `DEPTH`",
    ),
    (
        "meminst.sv",
        "5:19 `4` -> `'d4`",
        "`W` is used only in ranges, where the unsigned 4 gives the same width",
    ),
    (
        "meminst.sv",
        "28:16 `8` -> `'d8`",
        "`DEPTH` is used only in `$clog2(DEPTH)` and as an array size, the same for the unsigned 8",
    ),
    (
        "meminst.sv",
        "28:23 `6` -> `'d6`",
        "`W` is used only in ranges, where the unsigned 6 gives the same width",
    ),
    (
        "meminst.sv",
        "36:16 `2` -> `'d2`",
        "`DEPTH` is used only in `$clog2(DEPTH)` and as an array size, the same for the unsigned 2",
    ),
    (
        "meminst.sv",
        "39:15 `0` -> `'d0`",
        "an unsized based index `'dN` is a variable index of the same value as the constant `N`, so the same bits are read",
    ),
    (
        "meminst.sv",
        "40:15 `0` -> `'d0`",
        "an unsized based index `'dN` is a variable index of the same value as the constant `N`, so the same bits are read",
    ),
    (
        "memoor.sv",
        "16:32 `0` -> `'d0`",
        "an unsized based index `'dN` is a variable index of the same value as the constant `N`, so the same bits are read",
    ),
    (
        "modcount.sv",
        "7:21 `10` -> `'d10`",
        "`N` is an `int`: it takes the value 10 either way",
    ),
    (
        "modcount.sv",
        "14:42 `1` -> `'d1`",
        "`N - 'd1` is positive: cast to `W` bits it has the same bits as `N - 1`",
    ),
    (
        "modcount.sv",
        "14:56 `0` -> `'d0`",
        "the cast `W'(…)` of 0 is 0 either way",
    ),
    (
        "modcount.sv",
        "14:69 `1` -> `'d1`",
        "the cast `W'(…)` of 1 is 1 either way",
    ),
    (
        "mux4.sv",
        "21:21 `-:` -> `+:`",
        "a select of width 1 is the same bit either way",
    ),
    (
        "mux4.sv",
        "24:18 `15` -> `'d15`",
        "an unsized based index `'dN` is a variable index of the same value as the constant `N`, so the same bits are read",
    ),
    (
        "nest.sv",
        "6:32 `3` -> `2`",
        "the values `mid` gives `K` (2 and 7) fit in 3 bits",
    ),
    (
        "nest.sv",
        "6:41 `4'd1` -> `4'd2`",
        "every instance of `leaf` overrides this default of `K`",
    ),
    (
        "nest.sv",
        "15:31 `3` -> `2`",
        "the values of `K` (2 and 7) fit in 3 bits",
    ),
    (
        "nibreg.sv",
        "13:12 `0` -> `'d0`",
        "an unsized based index `'dN` is a variable index of the same value as the constant `N`, so the same bits are read",
    ),
    (
        "nibreg.sv",
        "14:12 `1` -> `'d1`",
        "an unsized based index `'dN` is a variable index of the same value as the constant `N`, so the same bits are read",
    ),
    (
        "nibreg.sv",
        "19:18 `7` -> `'d7`",
        "an unsized based index `'dN` is a variable index of the same value as the constant `N`, so the same bits are read",
    ),
    (
        "onehot.sv",
        "19:13 `0` -> `'d0`",
        "an unsized based index `'dN` is a variable index of the same value as the constant `N`, so the same bits are read",
    ),
    (
        "oorread.sv",
        "12:16 `15` -> `'d15`",
        "`'d15 - c` has the same 32 bits as `15 - c`, and an index is read as unsigned either way",
    ),
    (
        "oorread.sv",
        "19:12 `15` -> `'d15`",
        "`'d15 - c` has the same 32 bits as `15 - c`, and an index is read as unsigned either way",
    ),
    (
        "oorwr.sv",
        "18:11 `1` -> `'d1`",
        "`i` is unsigned, so `i - 1` is an unsigned 32-bit subtraction either way",
    ),
    (
        "oorwr.sv",
        "19:11 `0` -> `'d0`",
        "an unsized based index `'dN` is a variable index of the same value as the constant `N`, so the same bits are read",
    ),
    (
        "oorwr.sv",
        "19:30 `1` -> `'d1`",
        "an unsized based index `'dN` is a variable index of the same value as the constant `N`, so the same bits are read",
    ),
    (
        "parcase.sv",
        "8:21 `0` -> `'d0`",
        "`'dN` is a 32-bit unsigned constant of the same value as `N`: the labels compare equal at 32 bits and assignments keep the same low bits",
    ),
    (
        "parcase.sv",
        "9:21 `1` -> `'d1`",
        "`'dN` is a 32-bit unsigned constant of the same value as `N`: the labels compare equal at 32 bits and assignments keep the same low bits",
    ),
    (
        "parcase.sv",
        "10:20 `2` -> `'d2`",
        "`'dN` is a 32-bit unsigned constant of the same value as `N`: the labels compare equal at 32 bits and assignments keep the same low bits",
    ),
    (
        "parcase.sv",
        "11:21 `3` -> `'d3`",
        "`'dN` is a 32-bit unsigned constant of the same value as `N`: the labels compare equal at 32 bits and assignments keep the same low bits",
    ),
    (
        "parcase.sv",
        "12:21 `4` -> `'d4`",
        "`'dN` is a 32-bit unsigned constant of the same value as `N`: the labels compare equal at 32 bits and assignments keep the same low bits",
    ),
    (
        "pipe.sv",
        "5:34 `8` -> `'d8`",
        "`W` is an `int`: it takes the value 8 either way",
    ),
    (
        "pipe.sv",
        "24:14 `4` -> `'d4`",
        "`W` is an `int`: the override takes the value 4 either way",
    ),
    (
        "plainff.sv",
        "19:12 `0` -> `'d0`",
        "a constant bit index: `0` and `'d0` select the same bit",
    ),
    (
        "portw.sv",
        "24:15 `5` -> `'d5`",
        "truncated to the 8 bits of the port: a nonnegative constant: unsigned or `int`, the value at the width it is used at is the same",
    ),
    (
        "powerup.sv",
        "25:13 `1` -> `'d1`",
        "1 either way, assigned to the 1-bit `armed`",
    ),
    (
        "ramload.sv",
        "13:21 `4` -> `'d4`",
        "an array size: the unsigned 4 gives the same size",
    ),
    (
        "ramload.sv",
        "14:10 `7` -> `6`",
        "the ROM's values are below 128, so 7-bit elements hold them",
    ),
    (
        "regfile.sv",
        "18:24 `8` -> `'d8`",
        "an array size: the unsigned 8 gives the same size",
    ),
    (
        "regfile.sv",
        "23:7 `0` -> `'d0`",
        "the address 0 either way",
    ),
    ("regfile.sv", "23:13 `8'sd0` -> `8'd0`", "zero either way"),
    (
        "ripple.sv",
        "19:15 `0` -> `'d0`",
        "an unsized based index `'dN` is a variable index of the same value as the constant `N`, so the same bits are read",
    ),
    (
        "ripple.sv",
        "19:25 `0` -> `'d0`",
        "an unsized based index `'dN` is a variable index of the same value as the constant `N`, so the same bits are read",
    ),
    (
        "ripple.sv",
        "20:15 `1` -> `'d1`",
        "an unsized based index `'dN` is a variable index of the same value as the constant `N`, so the same bits are read",
    ),
    (
        "ripple.sv",
        "20:25 `1` -> `'d1`",
        "an unsized based index `'dN` is a variable index of the same value as the constant `N`, so the same bits are read",
    ),
    (
        "ripple.sv",
        "21:15 `2` -> `'d2`",
        "an unsized based index `'dN` is a variable index of the same value as the constant `N`, so the same bits are read",
    ),
    (
        "ripple.sv",
        "21:25 `2` -> `'d2`",
        "an unsized based index `'dN` is a variable index of the same value as the constant `N`, so the same bits are read",
    ),
    (
        "ripple.sv",
        "22:15 `3` -> `'d3`",
        "an unsized based index `'dN` is a variable index of the same value as the constant `N`, so the same bits are read",
    ),
    (
        "ripple.sv",
        "22:25 `3` -> `'d3`",
        "an unsized based index `'dN` is a variable index of the same value as the constant `N`, so the same bits are read",
    ),
    (
        "rom.sv",
        "13:19 `4` -> `'d4`",
        "an array size: the unsigned 4 gives the same size",
    ),
    (
        "rom.sv",
        "21:42 `8` -> `7`",
        "the 5-bit element is zero-extended to 7 bits, then to 8, or to 8 at once",
    ),
    (
        "shiftw.sv",
        "11:20 `2` -> `'d2`",
        "the shift amount is unsigned anyway",
    ),
    (
        "shiftw.sv",
        "13:16 `1` -> `'d1`",
        "truncated to 8 bits: a nonnegative constant: unsigned or `int`, the value at the width it is used at is the same",
    ),
    (
        "signmix.sv",
        "24:17 `+` -> `-`",
        "adding 0 is subtracting 0",
    ),
    (
        "signmix.sv",
        "25:18 `+` -> `-`",
        "adding 0 is subtracting 0",
    ),
    (
        "signmix.sv",
        "29:22 `1` -> `'d1`",
        "the shift amount is unsigned anyway",
    ),
    (
        "signmix.sv",
        "30:21 `>>>` -> `>>`",
        "a concatenation is unsigned: `>>>` shifts in zeros",
    ),
    (
        "signmix.sv",
        "30:25 `1` -> `'d1`",
        "the shift amount is unsigned anyway",
    ),
    (
        "signmix.sv",
        "31:20 `>>>` -> `>>`",
        "`(s >>> 1) + u` is unsigned, so the shift is too",
    ),
    (
        "signmix.sv",
        "31:24 `1` -> `'d1`",
        "the shift amount is unsigned anyway",
    ),
    (
        "signmix.sv",
        "38:19 `1'b1` -> `1'd0`",
        "the label `8'd255` never matches the zero-extended 4-bit `t`",
    ),
    (
        "traffic.sv",
        "10:21 `2` -> `1`",
        "the timer limits 3, 0 and 2 fit in 2 bits, extended in the comparisons",
    ),
    (
        "unsized.sv",
        "16:20 `1` -> `'d1`",
        "the sum with the unsigned `q` is unsigned either way: a nonnegative constant: unsigned or `int`, the value at the width it is used at is the same",
    ),
    (
        "unsized.sv",
        "17:24 `1` -> `'d1`",
        "the sum with the unsigned `q` is unsigned either way: a nonnegative constant: unsigned or `int`, the value at the width it is used at is the same",
    ),
    (
        "unsized.sv",
        "17:29 `0` -> `'d0`",
        "the comparison is unsigned either way: a nonnegative constant: unsigned or `int`, the value at the width it is used at is the same",
    ),
    (
        "unsized.sv",
        "24:42 `2` -> `'d2`",
        "the sum with the unsigned `'h7fffffff` is unsigned either way: a nonnegative constant: unsigned or `int`, the value at the width it is used at is the same",
    ),
    (
        "wrinst.sv",
        "3:19 `4` -> `'d4`",
        "`W` is used only in `$clog2(W)` and a range, the same for the unsigned 4",
    ),
    (
        "wrinst.sv",
        "20:13 `8` -> `'d8`",
        "`W` is used only in `$clog2(W)` and a range, the same for the unsigned 8",
    ),
    (
        "xstore.sv",
        "27:20 `3` -> `2`",
        "a target of an assignment of x, which has no effect (`wide` is assigned again on every path)",
    ),
    (
        "xstore.sv",
        "27:22 `0` -> `1`",
        "a target of an assignment of x, which has no effect (`wide` is assigned again on every path)",
    ),
    (
        "xstore.sv",
        "36:11 `0` -> `'d0`",
        "an unsized based index `'dN` is a variable index of the same value as the constant `N`, so the same bits are read",
    ),
];

/// What the mutants of one example give: descriptions of the failures,
/// the equivalent mutants, the number caught and rejected.
type Outcomes = (Vec<String>, Vec<(String, String)>, usize, usize);

/// Plants every mutant of the example at `path`, written to `dir`.
fn plant_all((yosys, z3): (&Path, &Path), dir: &Path, path: &Path) -> Outcomes {
    let mut rand = Rand(seed(0x1405_7b7e_f767_814f, &stem(path)));
    let (inst, t) = load(path);
    let ys = yosys::run(yosys, path, &t.name, &[]).unwrap();
    let example = path.file_name().unwrap().to_string_lossy().to_string();
    let source = std::fs::read_to_string(path).unwrap();
    let mut out: Outcomes = Default::default();
    for m in mutants(&source) {
        match plant((yosys, z3), dir, path, (&inst, &t, &ys), &m, &mut rand) {
            Ok(Outcome::Rejected) => out.3 += 1,
            Ok(Outcome::Caught) => out.2 += 1,
            Ok(Outcome::Equivalent) => out.1.push((example.clone(), m.what)),
            Err(e) => out.0.push(format!("{example} {}: {e}", m.what)),
        }
    }
    out
}

/// The planted-bug suite: every mutant of every example (see [`mutants`])
/// is planted on our side and on Yosys's ([`plant`]). Each one our typing
/// accepts must be caught in both directions, with every difference
/// attributed to the mutated side, unless it is listed in [`EQUIVALENT`];
/// and the list must be exactly the mutants found equivalent. The
/// examples run in parallel, each with its own seed (from its name).
/// Slow: run with `cargo test -p verilog -- --include-ignored`.
#[test]
#[ignore = "slow (about two minutes); the full run is `cargo test -p verilog -- --include-ignored`"]
fn planted_bugs() {
    let Some((yosys, z3)) = tools() else { return };
    let paths = examples();
    // the mutants, with the files of `$readmemb`/`$readmemh` next to them
    let dir = TempDir::new("mutants");
    for data in std::fs::read_dir(paths[0].parent().unwrap()).unwrap() {
        let data = data.unwrap().path();
        if data.extension().is_some_and(|e| e != "sv") {
            std::fs::copy(&data, dir.join(data.file_name().unwrap())).unwrap();
        }
    }
    let results: Vec<Outcomes> = std::thread::scope(|scope| {
        let handles: Vec<_> = paths
            .iter()
            .map(|path| {
                let tools = (yosys.as_path(), z3.as_path());
                let dir = dir.path();
                scope.spawn(move || plant_all(tools, dir, path))
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let (mut failures, mut equivalent, mut caught, mut rejected) = Outcomes::default();
    for (f, e, c, r) in results {
        failures.extend(f);
        equivalent.extend(e);
        caught += c;
        rejected += r;
    }
    eprintln!(
        "planted bugs: {caught} caught, {} equivalent, {rejected} rejected by typing",
        equivalent.len()
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    let listed: Vec<(String, String)> = EQUIVALENT
        .iter()
        .map(|(e, w, _)| (e.to_string(), w.to_string()))
        .collect();
    assert_eq!(
        equivalent, listed,
        "mutants found equivalent (left), listed (right)"
    );
}

