/*!
# Running Yosys

An independent translation of a design into SMT-LIB, to compare ours with.
Yosys is looked up as `$YOSYS_BIN` (the binary itself), then
`~/tools/oss-cad-suite/bin/yosys`, then `yosys` on the `PATH`. [`run`]
gives it the script of [`script`]:

```text
read_verilog -sv FILE; hierarchy -top TOP; proc; flatten; clk2fflogic; write_smt2 -wires -
```

`flatten` makes the whole hierarchy one module, whose wires have our
instance path names (`u.q`); without `opt`, every register stays one
flip-flop. `clk2fflogic` makes the clocks ordinary signals (inputs, or
wires for clocks derived from registers), sampled at every step of an
implicit global clock, so that one transition is one time step, as one of
our rounds is (see below).

## The output

[`Smt2::text`] is the script Yosys writes (its `write_smt2` help describes
it). For the top module `T` it declares a state sort `|T_s|`, a transition
predicate `(|T_t| s s')`, a hierarchy predicate `(|T_h| s)` that must hold in
every state, and an accessor `(|T_n v| s)` for every port, register and wire
`v` (with `-wires`, every named wire). Each accessor follows a comment
`; yosys-smt2-wire v W` (W = width); ports and registers additionally get
`; yosys-smt2-input v W`, `; yosys-smt2-output v W` or
`; yosys-smt2-register v W`. [`parse`] collects these four kinds into
[`Smt2`] and ignores the others (`clock`, `witness`, …).

An accessor of width 1 returns `Bool`, wider ones `(_ BitVec W)`;
[`Smt2::value`] always gives a bit vector, as our rounds use
(see [`smt`]).

## Clocks

After `clk2fflogic`, every flip-flop `q` clocked by `c` is three Yosys
registers, sampling values of the previous step:

- `$auto$clk2fflogic.cc:N:sample_control_edge$/c#sampled$K` (1 bit), the
  clock level (`c` is the clock's name in the flip-flop's module, a path
  after `flatten`);
- `$auto$clk2fflogic.cc:N:sample_data$D#sampled$K`, the flip-flop's data
  input, with `D` the name of a wire (`/s1` when `q <= s1`) or an anonymous
  cell output (`$procmux$4_Y`);
- `$auto$clk2fflogic.cc:N:sample_data$/q#sampled$K`, the flip-flop's
  output;

and the wire `q` becomes `ite({c#sampled, c} == 01, D#sampled,
q#sampled)` (`10` for a falling edge): the sampled input if the edge
happened between the previous step and this one, else the old output. A
flip-flop with an asynchronous reset to the constant `v` has a fourth
register `$auto$clk2fflogic.cc:N:sample_control$R#sampled$K` (1 bit), the
reset's active level of the previous step, with `R` an anonymous cell
(`$auto$rtlil.cc:N:ReduceOr$6`, `…:Not$36` for an active-low reset), and
its wire `q` is `ite(r, v, ite(r#sampled, v, …))` with `r` the active
level now: the reset value while the reset is active and in the step it
is released, even if an edge happens then. So
the wire `q` in a state is the register's value after that step, and
`|T_t|` relates two consecutive time steps. The initial-state predicate
`(|T_i| s)` sets the clock samples so that no edge happens in the first
step, the reset samples to 0 (inactive) and the data input samples to 0; it constrains an output sample only
by `(= (bvand q #b0…) #b0…)` (always true) or not at all. A data input
sample can carry the name of a register (`/s1#sampled` above) with its
width, so names alone do not tell the output sample of a register; see
[`miter::registers`](crate::miter::registers). The names contain line
numbers of Yosys's source: the format was checked with Yosys 0.69+173.

## Limits

Parameter overrides of the top module (`v2rm -G N=4`) are not passed to
Yosys yet: [`run`] rejects them, so the designs compared are always those
with the default parameter values.
*/

use crate::ast::Override;
use crate::smt;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The Yosys binary, or `None` if there is none.
pub fn find() -> Option<PathBuf> {
    let candidates = [
        std::env::var_os("YOSYS_BIN").map(PathBuf::from),
        std::env::var_os("HOME").map(|h| Path::new(&h).join("tools/oss-cad-suite/bin/yosys")),
    ];
    let found = candidates.into_iter().flatten().find(|p| p.exists());
    found.or_else(|| {
        let ok = Command::new("yosys").arg("-V").output().is_ok();
        ok.then(|| PathBuf::from("yosys"))
    })
}

/// The Yosys script translating the module `top` of `file` (with all
/// instances below it) into SMT-LIB on standard output. Fails on a path
/// Yosys cannot read inside a script (one with `"` or a line break).
pub fn script(file: &Path, top: &str) -> Result<String, String> {
    let file = file.to_string_lossy();
    if file.contains(['"', '\n', '\r']) {
        return Err(format!("{file}: path not usable in a Yosys script"));
    }
    Ok(format!(
        "read_verilog -sv \"{file}\"; hierarchy -top {top}; proc; flatten; clk2fflogic; \
         write_smt2 -wires -"
    ))
}

/// Runs the Yosys binary `bin` on the module `top` of `file` and parses
/// its output. Fails if Yosys cannot be started, reports an error, or
/// `overrides` is not empty (not supported yet, see the module docs).
pub fn run(bin: &Path, file: &Path, top: &str, overrides: &[Override]) -> Result<Smt2, String> {
    if let Some(o) = overrides.first() {
        return Err(format!(
            "parameter overrides are not supported by the Yosys comparison yet (`{}`)",
            o.name
        ));
    }
    let out = Command::new(bin)
        .args(["-q", "-p", &script(file, top)?])
        .output()
        .map_err(|e| format!("{}: {e}", bin.display()))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let log = String::from_utf8_lossy(&out.stdout);
        return Err(format!("yosys failed on {}:\n{err}{log}", file.display()));
    }
    parse(&String::from_utf8_lossy(&out.stdout))
}

/// What a Yosys register of a design after `clk2fflogic` samples (see the
/// module docs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Sample<'n> {
    /// The level of the clock with this name.
    Edge(&'n str),
    /// The level of an asynchronous reset (an anonymous cell output).
    Control,
    /// The wire with this name (the data input or the output of a
    /// flip-flop), or an anonymous cell output (`None`).
    Data(Option<&'n str>),
}

/// What the Yosys register `name` samples, or `None` if it is not a
/// register made by `clk2fflogic`.
pub fn sample(name: &str) -> Option<Sample<'_>> {
    let rest = name.strip_prefix("$auto$clk2fflogic.cc:")?;
    let (line, rest) = rest.split_once(':')?;
    if line.is_empty() || !line.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let (rest, k) = rest.rsplit_once("#sampled$")?;
    if k.is_empty() || !k.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    if let Some(c) = rest.strip_prefix("sample_control_edge$/") {
        return Some(Sample::Edge(c));
    }
    if rest.starts_with("sample_control$") {
        return Some(Sample::Control);
    }
    let wire = rest.strip_prefix("sample_data$")?;
    Some(Sample::Data(wire.strip_prefix('/')))
}

/// What `write_smt2` produced for a flattened design.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Smt2 {
    /// The top module, the prefix of every symbol.
    pub top: String,
    /// The whole SMT-LIB output.
    pub text: String,
    /// Widths by name, from the `; yosys-smt2-…` comments.
    pub inputs: BTreeMap<String, usize>,
    pub outputs: BTreeMap<String, usize>,
    pub registers: BTreeMap<String, usize>,
    /// Every named wire, including ports and registers.
    pub wires: BTreeMap<String, usize>,
    /// The conjuncts of the initial-state predicate `|T_i|`, by the
    /// register each constrains.
    pub initial: BTreeMap<String, String>,
}

impl Smt2 {
    /// Whether `|T_i|` leaves the register `name` free: it has no
    /// conjunct for it, or `(= (bvand (… state) #b0…0) #b0…0)`, which
    /// always holds.
    pub fn free_initially(&self, name: &str) -> bool {
        let Some(expr) = self.initial.get(name) else {
            return true;
        };
        let zeros =
            |t: &str| t.len() > 2 && t.starts_with("#b") && t[2..].bytes().all(|b| b == b'0');
        let Some(rest) = expr.strip_prefix("(= (bvand (|") else {
            return false;
        };
        let Some((_, rest)) = rest.split_once("| state) ") else {
            return false;
        };
        let Some((mask, rest)) = rest.split_once(") ") else {
            return false;
        };
        let Some(value) = rest.strip_suffix(')') else {
            return false;
        };
        zeros(mask) && zeros(value)
    }
}

/// Parses the output of `write_smt2`. Fails if it has no top module or
/// several modules (the design was not flattened), or if a name is listed
/// twice with different widths.
pub fn parse(text: &str) -> Result<Smt2, String> {
    let mut modules = Vec::new();
    let mut top = None;
    let mut maps: [BTreeMap<String, usize>; 4] = Default::default();
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("; yosys-smt2-") else {
            continue;
        };
        let (kind, arg) = rest.split_once(' ').unwrap_or((rest, ""));
        let map = match kind {
            "module" => {
                modules.push(arg.to_string());
                continue;
            }
            "topmod" => {
                top = Some(arg.to_string());
                continue;
            }
            "input" => &mut maps[0],
            "output" => &mut maps[1],
            "register" => &mut maps[2],
            "wire" => &mut maps[3],
            _ => continue,
        };
        // the width is the last word: a name could contain spaces
        let (name, width) = arg
            .rsplit_once(' ')
            .and_then(|(n, w)| Some((n, w.parse::<usize>().ok()?)))
            .ok_or_else(|| format!("malformed Yosys comment: {line}"))?;
        if let Some(old) = map.insert(name.to_string(), width)
            && old != width
        {
            return Err(format!(
                "Yosys lists `{name}` with widths {old} and {width}"
            ));
        }
    }
    let top = top.ok_or("no `yosys-smt2-topmod` in the Yosys output")?;
    if modules != [top.clone()] {
        return Err(format!(
            "expected only module `{top}`, Yosys output has {modules:?}"
        ));
    }
    let [inputs, outputs, registers, wires] = maps;
    let initial = initial(text, &top)?;
    Ok(Smt2 {
        top,
        text: text.to_string(),
        inputs,
        outputs,
        registers,
        wires,
        initial,
    })
}

/// The conjuncts of `(|T_i| s)`, by the register each one names in its
/// comment.
fn initial(text: &str, top: &str) -> Result<BTreeMap<String, String>, String> {
    let head = format!("(define-fun {} ((state ", smt::symbol(&format!("{top}_i")));
    let mut lines = text.lines().skip_while(|l| !l.starts_with(&head));
    let Some(first) = lines.next() else {
        return Err(format!("no `{top}_i` in the Yosys output"));
    };
    let mut out = BTreeMap::new();
    if first.ends_with(" Bool true)") {
        return Ok(out);
    }
    if !first.ends_with(" Bool (and") {
        return Err(format!("unexpected Yosys line: {first}"));
    }
    for line in lines {
        if line == "))" {
            return Ok(out);
        }
        let (expr, name) = line
            .trim()
            .split_once(" ; ")
            .ok_or_else(|| format!("unexpected Yosys line: {line}"))?;
        out.insert(name.to_string(), expr.to_string());
    }
    Err(format!("`{top}_i` does not end in the Yosys output"))
}

impl Smt2 {
    /// The sort of states, `|T_s|`.
    pub fn state_sort(&self) -> String {
        smt::symbol(&format!("{}_s", self.top))
    }

    /// `(|T_t| s next)`: `next` follows `s` after one time step.
    pub fn transition(&self, s: &str, next: &str) -> String {
        format!("({} {s} {next})", smt::symbol(&format!("{}_t", self.top)))
    }

    /// `(|T_h| s)`, to be asserted for every state.
    pub fn hierarchy(&self, s: &str) -> String {
        format!("({} {s})", smt::symbol(&format!("{}_h", self.top)))
    }

    /// `(|T_i| s)`: `s` is an initial state.
    pub fn initial(&self, s: &str) -> String {
        format!("({} {s})", smt::symbol(&format!("{}_i", self.top)))
    }

    /// The value of wire or register `name` in state `s` as a
    /// `(_ BitVec W)`, or `None` if there is no such wire or register.
    /// Width-1 accessors return `Bool` and are converted with `ite`.
    pub fn value(&self, name: &str, s: &str) -> Option<String> {
        let width = *self.wires.get(name).or_else(|| self.registers.get(name))?;
        let get = format!("({} {s})", smt::symbol(&format!("{}_n {name}", self.top)));
        Some(if width == 1 {
            format!("(ite {get} #b1 #b0)")
        } else {
            get
        })
    }
}
