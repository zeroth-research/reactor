/*!
# Simulating a design with iverilog

Icarus Verilog is a simulator independent of everything in this crate:
its own parser, elaboration and event-driven semantics. The tests compare
the reference semantics ([`ast_interp`](crate::ast_interp)) with it on
random traces, and [`miter`](crate::miter) asks it for a second opinion
on a counterexample. This module writes the testbenches for both and
runs them.

A testbench drives the inputs of the design round by round (see
[`testbench`]) and prints, after each round, the values of the signals of
[`shown`] in hex, one line per round, as [`hex_line`] prints ours. A
simulator gives `x` to bits read outside a signal, where the round
semantics reads 0, and runs the blocks of a derived clock after the
updates that make its edge ([`simulator_differs`]); otherwise the two
agree on the designs the translation accepts.
*/

use crate::BitVec;
use crate::ast::{Dir, Item, Name, Ty};
use crate::ast_interp::Values;
use crate::elab::Inst;
use crate::lower::Translation;
use num_bigint::BigUint;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

/// The directory of the binaries `iverilog` and `vvp`: `$IVERILOG_BIN`,
/// else `~/tools/oss-cad-suite/bin`, if it holds `iverilog`; else the
/// empty path (the `PATH`) if `iverilog -V` runs; else `None`.
pub fn find() -> Option<PathBuf> {
    let candidates = [
        std::env::var_os("IVERILOG_BIN").map(PathBuf::from),
        std::env::var_os("HOME").map(|h| Path::new(&h).join("tools/oss-cad-suite/bin")),
    ];
    let found = candidates
        .into_iter()
        .flatten()
        .find(|dir| dir.join("iverilog").exists());
    found.or_else(|| {
        let ok = Command::new("iverilog").arg("-V").output().is_ok();
        ok.then(PathBuf::new)
    })
}

/// Compiles the testbench `tb` (module `tb`) with the design `files`
/// (with `SYNTHESIS` defined, as the frontend does), runs it in the
/// directory of the first file (where `$readmemb`/`$readmemh` find their
/// files) and returns its output lines, without the one `$finish` prints.
/// `bin` is the directory of the binaries (see [`find`]).
pub fn simulate(bin: &Path, files: &[&Path], tb: &str) -> Result<Vec<String>, String> {
    let tmp = crate::EmptyDir::new("v2rm-iverilog")
        .map_err(|e| format!("no temporary directory for iverilog: {e}"))?;
    let file = tmp.0.join("tb.sv");
    let out = tmp.0.join("tb.vvp");
    std::fs::write(&file, tb).map_err(|e| format!("{}: {e}", file.display()))?;
    let failed = |what: &str, e: std::io::Error| format!("{what} did not run: {e}");
    let compile = Command::new(bin.join("iverilog"))
        .args(["-g2012", "-DSYNTHESIS", "-s", "tb", "-o"])
        .arg(&out)
        .arg(&file)
        .args(files)
        .output()
        .map_err(|e| failed("iverilog", e))?;
    if !compile.status.success() {
        return Err(format!(
            "iverilog failed:\n{}",
            String::from_utf8_lossy(&compile.stderr).trim_end()
        ));
    }
    let dir = match files.first().and_then(|f| f.parent()) {
        Some(d) if !d.as_os_str().is_empty() => d,
        _ => Path::new("."),
    };
    let run = Command::new(bin.join("vvp"))
        .arg("-n")
        .arg(&out)
        .current_dir(dir)
        .output()
        .map_err(|e| failed("vvp", e))?;
    if !run.status.success() {
        return Err(format!(
            "vvp failed:\n{}",
            String::from_utf8_lossy(&run.stderr).trim_end()
        ));
    }
    let stdout = String::from_utf8_lossy(&run.stdout);
    Ok(stdout
        .lines()
        .filter(|l| !l.contains("$finish"))
        .map(String::from)
        .collect())
}

/// The inputs of a design, with their types and their names in a
/// testbench: the inputs of the top module, then the signals connected to
/// outputs of blackboxes, which take any value in every round as inputs
/// do (written in a testbench through the output of the blackbox stub,
/// `dut.g.clk_out`, a variable that only the testbench assigns).
pub fn design_inputs(inst: &Inst) -> Vec<(Name, Ty, String)> {
    let naming: BTreeMap<Name, Name> = inst.naming().into_iter().collect();
    let mut out: Vec<(Name, Ty, String)> = (inst.module.ports.iter())
        .filter(|p| p.dir == Dir::Input)
        .map(|p| (p.decl.name.clone(), p.decl.ty, p.decl.name.clone()))
        .collect();
    let mut walk = vec![(inst, String::new())];
    while let Some((i, prefix)) = walk.pop() {
        for u in i.module.instances() {
            for (c, f) in u.conns(Dir::Output).filter(|(_, f)| f.blackbox) {
                if let Some(y) = c.shared() {
                    let name = naming[&format!("{prefix}{y}")].clone();
                    out.push((name, f.ty, format!("dut.{prefix}{}.{}", u.name, c.port)));
                }
            }
        }
        for (u, child) in i.children.iter().rev() {
            walk.push((child, format!("{prefix}{u}.")));
        }
    }
    out
}

/// The declarations of a testbench for the top module of `inst`: the
/// ports and the instance `dut`; the inputs in `start` start with these
/// values (before time 0, so their first assignment makes no edge).
fn tb_header(inst: &Inst, start: &Values) -> String {
    let m = &inst.module;
    let mut tb = String::from("module tb;\n");
    for p in &m.ports {
        let kind = if p.dir == Dir::Input { "logic" } else { "wire" };
        let x = &p.decl.name;
        let value = match start.get(x) {
            Some(v) => format!(" = {}'h{:x}", v.width(), v.unsigned()),
            None => String::new(),
        };
        tb += &format!("  {kind} [{}:0] {x}{value};\n", p.decl.ty.width - 1);
    }
    let conns: Vec<String> = m
        .ports
        .iter()
        .map(|p| format!(".{0}({0})", p.decl.name))
        .collect();
    tb += &format!("  {} dut({});\n  initial begin\n", m.name, conns.join(", "));
    tb
}

/// `    name = value;` for each of `values` (inputs of the design) whose
/// name `pick` accepts, with `name` as the testbench writes it (see
/// [`design_inputs`]).
fn tb_set(inst: &Inst, values: &Values, pick: impl Fn(&str) -> bool) -> String {
    let paths: BTreeMap<Name, String> = design_inputs(inst)
        .into_iter()
        .map(|(name, _, path)| (name, path))
        .collect();
    let mut out = String::new();
    for (name, value) in values.iter().filter(|(n, _)| pick(n)) {
        let path = &paths[name];
        out += &format!("    {path} = {}'h{:x};\n", value.width(), value.unsigned());
    }
    out
}

/// `    dut.q = value;` for every register.
fn tb_init(init: &Values) -> String {
    let mut out = String::new();
    for (q, value) in init {
        out += &format!("    dut.{q} = {}'h{:x};\n", value.width(), value.unsigned());
    }
    out
}

/// Sets the registers to `init` twice: setting a register may make an edge
/// of a clock derived from it (from x), whose block then overwrites
/// registers, so they are set again once those updates are done (no edge
/// the second time); except the registers with an asynchronous reset,
/// which a reset that is a register may have reset (their clocks are
/// inputs in the examples).
fn tb_init_twice(t: &Translation, init: &Values) -> String {
    let mut again = init.clone();
    again.retain(|q, _| !t.reset_registers.contains(q));
    format!("{}    #1;\n{}", tb_init(init), tb_init(&again))
}

/// `$display` of the signals compared, in hex.
fn tb_display(inst: &Inst) -> String {
    let shown: Vec<String> = shown(inst).iter().map(|x| format!("dut.{x}")).collect();
    let format = vec!["%h"; shown.len()].join(" ");
    format!("$display(\"{format}\", {});\n", shown.join(", "))
}

/// A testbench that runs `rounds` (a value for every input, clocks
/// included) and prints all signals but the inputs in hex, one line per
/// round.
///
/// A round sets the clocks, waits, then sets the other inputs, waits and
/// prints. So the blocks whose edge happens sample the values of the
/// previous round, as registers read latched values (data inputs meet
/// setup and hold), and no assignment races with an edge. The first round
/// then sets the registers to `init` (by path), by hierarchical assignment
/// (`dut.q = …`; iverilog accepts this although the standard forbids
/// writing an `always_ff` variable from another process), overriding
/// whatever an edge from x of a derived clock did (input clocks start at
/// their level in the first round, so that registers with an initial value
/// see no edge from x either), and
/// only then sets the other inputs: an asynchronous reset active in the
/// first round then resets its registers (its change from x is an edge
/// of the reset). `t.reset_registers` are set only once: an edge from x
/// of a reset that is a register can have reset them already.
pub fn testbench(inst: &Inst, t: &Translation, init: &Values, rounds: &[Values]) -> String {
    let is_clock = |n: &str| t.clocks.iter().any(|c| c == n);
    let mut start = rounds.first().cloned().unwrap_or_default();
    start.retain(|n, _| is_clock(n));
    let mut tb = tb_header(inst, &start);
    for (k, v) in rounds.iter().enumerate() {
        tb += &tb_set(inst, v, is_clock);
        tb += "    #1;\n";
        if k == 0 {
            tb += &tb_init_twice(t, init);
        }
        tb += &tb_set(inst, v, |n| !is_clock(n));
        tb += &format!("    #1 {}", tb_display(inst));
    }
    tb += "    $finish;\n  end\nendmodule\n";
    tb
}

/// A testbench of the cycle model: it sets the registers to `init` (as
/// [`testbench`] says), then for each of `vectors` (a value for every
/// input but the clock of `t`, if any) sets the inputs, prints all signals
/// but the inputs in hex (one line per vector) and pulses the clock. The
/// clock starts at 0, since a change from x to 1 would be a rising edge.
pub fn cycle_testbench(inst: &Inst, t: &Translation, init: &Values, vectors: &[Values]) -> String {
    let clk = t.clocks.first().map(String::as_str);
    let level = |b: u8| {
        let v = Values::from([(clk.unwrap().to_string(), BitVec::new(1, b).unwrap())]);
        tb_set(inst, &v, |_| true)
    };
    let mut tb = tb_header(inst, &Values::new());
    if clk.is_some() {
        tb += &level(0);
    }
    tb += &tb_init_twice(t, init);
    for v in vectors {
        tb += &tb_set(inst, v, |n| Some(n) != clk);
        tb += &format!("    #1 {}", tb_display(inst));
        match clk {
            Some(_) => tb += &format!("{}    #1\n{}", level(1), level(0)),
            None => tb += "    #1;\n",
        }
    }
    tb += "    $finish;\n  end\nendmodule\n";
    tb
}

/// The signals a testbench prints: all but the inputs of the design (see
/// [`design_inputs`]), by name (which is a hierarchical name, see
/// [`Inst::naming`]).
pub fn shown(inst: &Inst) -> Vec<Name> {
    let inputs: BTreeSet<Name> = design_inputs(inst)
        .into_iter()
        .map(|(name, ..)| name)
        .collect();
    let mut out: Vec<Name> = Vec::new();
    for (_, name) in inst.naming() {
        if !inputs.contains(&name) && !out.contains(&name) {
            out.push(name);
        }
    }
    out
}

/// The signals of [`shown`] in `values`, as one line of hex numbers, as a
/// testbench prints them.
pub fn hex_line(inst: &Inst, values: &Values) -> String {
    shown(inst)
        .into_iter()
        .map(|x| {
            let v = &values[&x];
            format!("{:0w$x}", v.unsigned(), w = v.width().div_ceil(4))
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Why iverilog may disagree with the round semantics on a design, if it
/// may. A simulator runs the blocks of a derived clock after the updates
/// that make its edge, so they read the new values of the registers
/// updated in the same time step (and the new level of their clock),
/// where the round semantics reads the values from before the edge, as
/// hardware meeting its hold times does. Found statically: a block whose
/// clock is not an input of the top module reads a signal that depends on
/// a register of another clock (one of its clock's own chain included).
pub fn simulator_differs(inst: &Inst) -> Option<String> {
    let naming: BTreeMap<Name, Name> = inst.naming().into_iter().collect();
    // the clock of every register, what every combinational signal reads,
    // and every block with its clock and reads, by name
    let mut clock: BTreeMap<Name, Name> = BTreeMap::new();
    let mut comb: Vec<(Name, Name)> = Vec::new();
    let mut blocks: Vec<(Name, Name, Vec<Name>)> = Vec::new();
    let mut walk = vec![(inst, String::new())];
    while let Some((i, prefix)) = walk.pop() {
        let canon = |x: &str| naming[&format!("{prefix}{x}")].clone();
        for item in &i.module.items {
            let reads = || item.reads().into_iter().map(|(x, _)| canon(x));
            match item {
                Item::AlwaysFf { clock: c, .. } => {
                    for q in item.targets() {
                        clock.insert(canon(q), canon(c));
                    }
                    let q = canon(item.targets()[0]);
                    blocks.push((q, canon(c), reads().collect()));
                }
                Item::Assign { .. } | Item::AlwaysComb { .. } => {
                    for y in item.targets() {
                        comb.extend(reads().map(|x| (canon(y), x)));
                    }
                }
                Item::Instance(u) => {
                    for (c, _) in u.conns(Dir::Input).filter(|(_, f)| !f.blackbox) {
                        let Some(e) = &c.actual else { continue };
                        let mut reads = Vec::new();
                        e.reads(&mut reads);
                        let y = canon(&format!("{}.{}", u.name, c.port));
                        comb.extend(reads.into_iter().map(|(x, _)| (y.clone(), canon(x))));
                    }
                }
            }
        }
        for (u, child) in &i.children {
            walk.push((child, format!("{prefix}{u}.")));
        }
    }
    // the registers each signal depends on
    let mut deps: BTreeMap<Name, BTreeSet<Name>> = clock
        .keys()
        .map(|q| (q.clone(), BTreeSet::from([q.clone()])))
        .collect();
    loop {
        let mut changed = false;
        for (y, x) in &comb {
            let add = deps.get(x).cloned().unwrap_or_default();
            let d = deps.entry(y.clone()).or_default();
            let before = d.len();
            d.extend(add);
            changed |= d.len() != before;
        }
        if !changed {
            break;
        }
    }
    let inputs: BTreeSet<Name> = design_inputs(inst)
        .into_iter()
        .map(|(name, ..)| name)
        .collect();
    for (q, c, reads) in blocks {
        if inputs.contains(&c) {
            continue;
        }
        for x in reads {
            if let Some(r) = deps.get(&x).into_iter().flatten().find(|r| clock[*r] != c) {
                return Some(format!(
                    "`{q}` (clocked by the derived clock `{c}`) reads `{x}`, which depends on \
                     `{r}` (clocked by `{}`)",
                    clock[r]
                ));
            }
        }
    }
    None
}

/// Runs `rounds` from the registers `init` (by path) in iverilog, with
/// [`testbench`] on the design `files` (whose hierarchy is `inst`, lowered
/// to `t`). Returns the values of the signals of [`shown`] after each
/// round, as numbers: `None` for a value iverilog gives with x or z bits.
pub fn run(
    bin: &Path,
    files: &[&Path],
    inst: &Inst,
    t: &Translation,
    init: &Values,
    rounds: &[Values],
) -> Result<Vec<BTreeMap<Name, Option<BigUint>>>, String> {
    let tb = testbench(inst, t, init, rounds);
    let lines = simulate(bin, files, &tb)?;
    if lines.len() != rounds.len() {
        return Err(format!(
            "iverilog printed {} lines for {} rounds:\n{}",
            lines.len(),
            rounds.len(),
            lines.join("\n")
        ));
    }
    let names = shown(inst);
    lines
        .iter()
        .map(|line| {
            let digits: Vec<&str> = line.split(' ').collect();
            if digits.len() != names.len() {
                return Err(format!("iverilog printed `{line}`"));
            }
            let parse = |d: &str| BigUint::parse_bytes(d.as_bytes(), 16);
            Ok(names
                .iter()
                .cloned()
                .zip(digits.into_iter().map(parse))
                .collect())
        })
        .collect()
}
