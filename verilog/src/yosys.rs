/*!
# Running Yosys

An independent translation of a design into SMT-LIB, to compare ours with.
Yosys is looked up as `$YOSYS_BIN` (the binary itself), then
`~/tools/oss-cad-suite/bin/yosys`, then `yosys` on the `PATH`. [`run`]
gives it the script of [`script`]:

```text
read_verilog -sv -noblackbox FILE…; hierarchy -top TOP; proc -norom;
setattr -set keep 1 w:[!$]*; opt_clean; flatten; check; cutpoint -blackbox;
memory_collect; memory_map; clk2fflogic; write_smt2 -wires -
```

(all files of the design in one `read_verilog`, so that a macro defined in
one is defined in the following ones, as in our frontend).

Yosys must read the same text as our frontend. [`run`] starts it in an
empty directory of its own, with the files' absolute paths: Yosys looks
for a file of `$readmemb`/`$readmemh` with a relative name first in its
working directory and then next to the source file, ours only next to the
source file, so in an empty directory both find the same file.
`read_verilog` defines the macros `SYNTHESIS`, as our frontend does, and
`YOSYS` in every file, which our frontend does not (Yosys-specific code
under `` `ifdef YOSYS `` is usually beyond the language we translate) and
which cannot be undefined for Yosys. So [`run`] refuses a design whose
text after preprocessing depends on `YOSYS`
([`frontend::depends_on_macro`]); a mere mention, in a comment or in a
branch skipped either way, is fine.

`flatten` makes the whole hierarchy one module, whose wires have our
instance path names (`u.q`); without `opt`, every register stays one
flip-flop. `-noblackbox`: by default `read_verilog` marks an empty module
as a blackbox, which `flatten` keeps as an instance and `write_smt2` then
refers to without declaring it. `-norom`: by default `proc` turns a `case`
that only selects constants into a ROM, which `write_smt2` writes with SMT
arrays; without it the `case` stays multiplexers, as in our translation.
`opt_clean` removes cells whose outputs nothing uses: `read_verilog` lowers
an assignment to a select with a variable index in a clocked block through
helper variables (`$bitselwrite$pos…`, `$lookahead…`), for which `proc`
makes flip-flops that nothing reads. Every wire with a source name is
marked `keep` first (the pattern `[!$]*` leaves out Yosys's own names,
which start with `$`): `opt_clean` would otherwise also remove unused
source names that it considers internal (`_zz_1_`, as SpinalHDL generates
them). It runs before `flatten`, while every port of a child is still a
port: afterwards it would also remove a child's output that the parent
leaves unconnected. `check` reports problems of the netlist; [`run`] fails
on conflicting drivers (it passes `-e` so that Yosys stops at that
warning, and also looks for the warning's text). The comparison relies on
Yosys's transition accepting every input value, and a wire with two
drivers breaks that: Yosys (0.69, still 0.71) reads the target `{f, r[n]} = {e[2], e[0]}`
backwards and drives the input bit `e[0]`, which in SMT-LIB defines the
input as a function of other signals, so the comparison could find no
difference where Yosys is wrong. [`parse`] checks this on the output too,
whatever the cause (see below).
The other problems `check` reports do not hide differences: a used wire
without a driver is free in every step (Yosys's side only gets more
behaviors; a register bit that is never written, which for us keeps its
initial value, is one), and `write_smt2` itself fails on a combinational
loop. `cutpoint -blackbox` replaces each instance of a module marked
`(* blackbox *)` (which `-noblackbox` leaves marked) by a fresh value per
step on each of its outputs, under the name of the wire it drives, and
drops the outputs nothing is connected to: the same abstraction as ours, so
the outputs are compared like inputs. A pin (`inout` port) only connected
to blackboxes is listed as an input and an output of the top module; we
have no variable for it. `memory_collect; memory_map` turns every unpacked array into one
flip-flop per element, named by its address (`m[3]`) as our element
signals are, and the read ports into multiplexers (a ROM loaded by
`$readmemb` into constants); `write_smt2` would otherwise write memories
with SMT arrays. `clk2fflogic` makes the clocks ordinary signals (inputs, or
wires for clocks derived from registers), sampled at every step of an
implicit global clock, so that one transition is one time step, as one of
our rounds is (see below).

`write_smt2` has no x: it encodes an x bit as 0, such as the value of an
out-of-range read. That is why Yosys agrees with us on out-of-range reads with
a variable index, which read 0 in our semantics; it is no independent check
of that rule.

## The output

[`Smt2::text`] is the script Yosys writes (its `write_smt2` help describes
it). For the top module `T` it declares a state sort `|T_s|`, a transition
predicate `(|T_t| s s')`, a hierarchy predicate `(|T_h| s)` that must hold in
every state, and an accessor `(|T_n v| s)` for every port, register and wire
`v` (with `-wires`, every named wire). Each accessor follows a comment
`; yosys-smt2-wire v W` (W = width); ports and registers additionally get
`; yosys-smt2-input v W`, `; yosys-smt2-output v W` or
`; yosys-smt2-register v W`. A fresh value per step (an `$anyseq` cell,
as `cutpoint` makes them) is a function `|T#k|` of the state, after a
comment `; yosys-smt2-anyseq T#k W CELL`. [`parse`] collects these five
kinds into [`Smt2`] and ignores the others (`clock`, `witness`, …).

The values Yosys leaves free in every step are uninterpreted functions of
the state, declared with `declare-fun`; registers too, but the transition
constrains them in the next state (`(|T#k| next_state)`), and wires are
defined with `define-fun`. An input's accessor is such a symbol, or, when
`write_smt2` declares the input in pieces or in another bit order, the
pieces put together with `concat` and `extract` (and `(ite b #b1 #b0)` for
a `Bool` piece). [`parse`] fails unless every input's accessor and every
`$anyseq` symbol is built in this way from declared symbols that are no
registers, each bit of them used only once: otherwise
Yosys's predicates constrain an input (as a conflicting driver does), and
a comparison would miss every difference outside the constraint.

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
[`miter::registers`](crate::miter::registers). That is the only place
that depends on Yosys's internal names: it recognises the samples by the
patterns above (`N` and `K` are wildcards), checked with Yosys 0.71+128
([`CHECKED_VERSION`]). `v2rm --yosys` warns with any other version; if that
version names the samples differently, the comparison refuses the design
instead of giving a wrong result.

## Limits

Parameter overrides of the top module (`v2rm -G N=4`) are not passed to
Yosys yet: [`run`] rejects them, so the designs compared are always those
with the default parameter values.
*/

use crate::ast::Override;
use crate::frontend;
use crate::miter::{Sexp, sexp, tokens};
use crate::smt;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

/// The Yosys binary, or `None` if there is none.
pub fn find() -> Option<PathBuf> {
    crate::find_tool("YOSYS_BIN", "yosys", "-V")
}

/// The Yosys version the naming pattern of the flip-flop samples of
/// `clk2fflogic` (see the module docs) was checked with. Another version
/// may name them differently; then the comparison refuses the design.
pub const CHECKED_VERSION: &str = "0.71+128";

/// The version of the Yosys binary `bin`, as `yosys -V` gives it (`0.71+128`
/// from `Yosys 0.71+128 (git sha1 …)`). Fails if Yosys cannot be started or
/// its answer has no version.
pub fn version(bin: &Path) -> Result<String, String> {
    let out = Command::new(bin)
        .arg("-V")
        .output()
        .map_err(|e| format!("{}: {e}", bin.display()))?;
    let text = String::from_utf8_lossy(&out.stdout);
    parse_version(&text).ok_or_else(|| {
        format!(
            "{}: no version in the answer to `-V`: {}",
            bin.display(),
            text.trim()
        )
    })
}

/// The version in Yosys's answer `text` to `-V`: the word after `Yosys`.
pub fn parse_version(text: &str) -> Option<String> {
    let mut words = text.split_whitespace();
    words.find(|w| *w == "Yosys")?;
    words.next().map(str::to_string)
}

/// A warning if `version` (of [`version`]) is not [`CHECKED_VERSION`], or
/// could not be found. The comparison runs all the same: a version that
/// names the flip-flop samples differently makes it refuse the design, it
/// cannot make it report a wrong result.
pub fn version_warning(version: &Result<String, String>) -> Option<String> {
    let checked = format!(
        "the comparison recognises Yosys's flip-flops by a naming pattern checked only \
         with Yosys {CHECKED_VERSION}; if this version names them differently, the \
         comparison refuses the design instead of giving a wrong result"
    );
    match version {
        Ok(v) if v == CHECKED_VERSION => None,
        Ok(v) => Some(format!("Yosys {v}: {checked}")),
        Err(e) => Some(format!("the Yosys version is unknown ({e}): {checked}")),
    }
}

/// The Yosys script translating the module `top` of the design in `files`
/// (with all instances below it) into SMT-LIB on standard output. Fails on
/// a path Yosys cannot read inside a script (one with `"` or a line break).
pub fn script<P: AsRef<Path>>(files: &[P], top: &str) -> Result<String, String> {
    let mut quoted = Vec::new();
    for file in files {
        let file = file.as_ref().to_string_lossy();
        if file.contains(['"', '\n', '\r']) {
            return Err(format!("{file}: path not usable in a Yosys script"));
        }
        quoted.push(format!("\"{file}\""));
    }
    Ok(format!(
        "read_verilog -sv -noblackbox {}; hierarchy -top {top}; proc -norom; \
         setattr -set keep 1 w:[!$]*; opt_clean; flatten; check; cutpoint -blackbox; \
         memory_collect; memory_map; clk2fflogic; write_smt2 -wires -",
        quoted.join(" ")
    ))
}

/// Runs the Yosys binary `bin` on the module `top` of the design in
/// `files` and parses its output. Fails if Yosys cannot be started,
/// reports an error or a wire with conflicting drivers (see the module
/// docs), `overrides` is not empty (not supported yet), or the design
/// depends on the macro `YOSYS` (see the module docs).
pub fn run<P: AsRef<Path>>(
    bin: &Path,
    files: &[P],
    top: &str,
    overrides: &[Override],
) -> Result<Smt2, String> {
    let file = files
        .iter()
        .map(|f| f.as_ref().display().to_string())
        .collect::<Vec<_>>()
        .join(" ");
    if let Some(o) = overrides.first() {
        return Err(format!(
            "parameter overrides are not supported by the Yosys comparison yet (`{}`)",
            o.name
        ));
    }
    if frontend::depends_on_macro(files, "YOSYS").map_err(|e| format!("{file}: {e}"))? {
        return Err(
            "the design tests the macro `YOSYS`, which Yosys defines and we do not, \
                    so Yosys would read another text than ours: not compared"
                .into(),
        );
    }
    // absolute paths, for Yosys runs in an empty directory of its own
    let absolute = files
        .iter()
        .map(|f| std::path::absolute(f).map_err(|e| format!("{}: {e}", f.as_ref().display())))
        .collect::<Result<Vec<_>, _>>()?;
    let dir =
        crate::EmptyDir::new("v2rm-yosys").map_err(|e| format!("a directory for Yosys: {e}"))?;
    // `-e`: Yosys stops with an error instead of the warning
    let out = Command::new(bin)
        .args(["-q", "-e", CONFLICT, "-p", &script(&absolute, top)?])
        .current_dir(&dir.0)
        .output()
        .map_err(|e| format!("{}: {e}", bin.display()))?;
    drop(dir);
    let err = String::from_utf8_lossy(&out.stderr);
    // the warning's text, in case it does not stop Yosys
    if err.contains(CONFLICT) {
        return Err(format!(
            "Yosys's netlist of {} has conflicting drivers, so a comparison with it would \
             not be meaningful:\n{err}",
            file
        ));
    }
    if !out.status.success() {
        let log = String::from_utf8_lossy(&out.stdout);
        return Err(format!("yosys failed on {file}:\n{err}{log}"));
    }
    parse(&String::from_utf8_lossy(&out.stdout))
}

/// Yosys's warning about a wire with several drivers.
const CONFLICT: &str = "multiple conflicting drivers";

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
    /// The free values Yosys draws in every step (`$anyseq` cells, which
    /// `cutpoint` puts on the outputs of blackboxes), by symbol (`T#k`),
    /// with their widths.
    pub anyseq: BTreeMap<String, usize>,
    /// The conjuncts of the initial-state predicate `|T_i|`, by the
    /// register each constrains.
    pub init_conjuncts: BTreeMap<String, String>,
}

impl Smt2 {
    /// Whether `|T_i|` leaves the register `name` free: it has no
    /// conjunct for it, or `(= (bvand (… state) #b0…0) #b0…0)`, which
    /// always holds.
    pub fn free_initially(&self, name: &str) -> bool {
        let Some(expr) = self.init_conjuncts.get(name) else {
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

/// `name`, or an error if it contains `|` or `\`, which an SMT-LIB quoted
/// symbol cannot ([`smt::symbol`]). `write_smt2` writes `\` in a name as
/// `/`, but keeps `|` (and then writes invalid SMT-LIB itself); names get
/// a `|` only from escaped identifiers, which the frontend rejects, or from
/// the path of a source file in the name of an internal wire.
fn symbol_safe(name: &str) -> Result<&str, String> {
    if name.contains(['|', '\\']) {
        return Err(format!(
            "Yosys's name `{name}` contains `|` or `\\`, which an SMT-LIB symbol cannot"
        ));
    }
    Ok(name)
}

/// Parses the output of `write_smt2`. Fails if it has no top module or
/// several modules (the design was not flattened), if a name is listed
/// twice with different widths or cannot be an SMT-LIB symbol, or if an
/// input or a fresh value is not free in every step (see the module docs).
pub fn parse(text: &str) -> Result<Smt2, String> {
    let mut modules = Vec::new();
    let mut top = None;
    let mut maps: [BTreeMap<String, usize>; 5] = Default::default();
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
                top = Some(symbol_safe(arg)?.to_string());
                continue;
            }
            "input" => &mut maps[0],
            "output" => &mut maps[1],
            "register" => &mut maps[2],
            "wire" => &mut maps[3],
            "anyseq" => &mut maps[4],
            _ => continue,
        };
        // the width is the last word: a name could contain spaces; an
        // `anyseq` comment has the symbol, the width and the cell name
        let arg = match kind {
            "anyseq" => arg.splitn(3, ' ').take(2).collect::<Vec<_>>().join(" "),
            _ => arg.to_string(),
        };
        let (name, width) = arg
            .rsplit_once(' ')
            .and_then(|(n, w)| Some((n, w.parse::<usize>().ok()?)))
            .ok_or_else(|| format!("malformed Yosys comment: {line}"))?;
        symbol_safe(name)?;
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
    let [inputs, outputs, registers, wires, anyseq] = maps;
    free(text, &top, &inputs, &anyseq)?;
    let init_conjuncts = init_conjuncts(text, &top)?;
    Ok(Smt2 {
        top,
        text: text.to_string(),
        inputs,
        outputs,
        registers,
        wires,
        anyseq,
        init_conjuncts,
    })
}

/// Checks that Yosys's predicates constrain no input and no `$anyseq`
/// value (see the module docs): each `$anyseq` symbol and every leaf of
/// an input's accessor is declared with `declare-fun`, is no register (a
/// symbol applied to `next_state`, which the transition constrains), and
/// each of its bits is used once only, by one of them. The accessor of an
/// input may only put such symbols together with `concat`, `extract`,
/// `(ite b #b1 #b0)` and `(= x #b1)`, as `write_smt2` does for an input
/// whose bits it declares in pieces or in another order.
fn free(
    text: &str,
    top: &str,
    inputs: &BTreeMap<String, usize>,
    anyseq: &BTreeMap<String, usize>,
) -> Result<(), String> {
    let declared: BTreeMap<&str, usize> = (text.lines())
        .filter_map(|l| {
            let (symbol, rest) = l.strip_prefix("(declare-fun |")?.split_once('|')?;
            let (_, sort) = rest.split_once(") ")?;
            let width = match sort.split_once(')').map_or(sort, |(s, _)| s) {
                s if s.starts_with("Bool") => 1,
                s => s.strip_prefix("(_ BitVec ")?.parse().ok()?,
            };
            Some((symbol, width))
        })
        .collect();
    let registers: BTreeSet<&str> = (text.match_indices("| next_state)"))
        .filter_map(|(i, _)| {
            let head = &text[..i];
            Some(&head[head.rfind("(|")? + 2..])
        })
        .collect();
    let mut used: BTreeSet<(String, usize)> = BTreeSet::new();
    let mut fresh = |symbol: &str, bits: Option<(usize, usize)>| {
        let Some(&width) = declared.get(symbol) else {
            return false;
        };
        let (hi, lo) = bits.unwrap_or((width.saturating_sub(1), 0));
        !registers.contains(symbol)
            && hi < width
            && (lo..=hi).all(|b| used.insert((symbol.to_string(), b)))
    };
    for symbol in anyseq.keys() {
        if !fresh(symbol, None) {
            return Err(format!(
                "Yosys's fresh value `{symbol}` is not a declared symbol of its own, so a \
                 comparison would not be meaningful"
            ));
        }
    }
    let state = smt::symbol(&format!("{top}_s"));
    for name in inputs.keys() {
        let head = format!(
            "(define-fun {} ((state {state})) ",
            smt::symbol(&format!("{top}_n {name}"))
        );
        let body = (text.lines())
            .find(|l| l.starts_with(&head))
            .and_then(|l| accessor_body(&l[head.len()..]));
        let mut leaves = vec![];
        let built = body.is_some_and(|b| leaves_of(&b, None, &mut leaves));
        if !built || !leaves.iter().all(|(s, bits)| fresh(s, *bits)) {
            return Err(format!(
                "Yosys's input `{name}` is not free in every step: its accessor is not built \
                 from declared symbols only, each bit used once, so a comparison would not \
                 be meaningful"
            ));
        }
    }
    Ok(())
}

/// The body of an accessor from the text after its parameters: the sort,
/// the body and the closing parenthesis.
fn accessor_body(rest: &str) -> Option<Sexp> {
    let tokens = tokens(&format!("({rest}"));
    let mut pos = 0;
    match sexp(&tokens, &mut pos)? {
        Sexp::List(items) if pos == tokens.len() => match <[Sexp; 2]>::try_from(items) {
            Ok([_, body]) => Some(body),
            Err(_) => None,
        },
        _ => None,
    }
}

/// A leaf `(|S| state)` of an accessor: the symbol `S` and the bits
/// `(hi, lo)` of it that are used, `None` for all.
type Leaf = (String, Option<(usize, usize)>);

/// Collects the leaves of `term` into `leaves`, of which only the bits
/// `bits` (`(hi, lo)` of the value of `term`, `None` for all) are used, or
/// returns false if `term` is not built from leaves with `concat`,
/// `extract`, `(ite b #b1 #b0)` and `(= x #b1)`.
fn leaves_of(term: &Sexp, bits: Option<(usize, usize)>, leaves: &mut Vec<Leaf>) -> bool {
    let Sexp::List(items) = term else {
        return false;
    };
    let atom = |s: &Sexp, a: &str| matches!(s, Sexp::Atom(t) if t == a);
    let number = |s: &Sexp| match s {
        Sexp::Atom(t) => t.parse::<usize>().ok(),
        Sexp::List(_) => None,
    };
    match items.as_slice() {
        [Sexp::Atom(f), arg] if atom(arg, "state") => {
            match f.strip_prefix('|').and_then(|f| f.strip_suffix('|')) {
                Some(symbol) => {
                    leaves.push((symbol.to_string(), bits));
                    true
                }
                None => false,
            }
        }
        [f, a, b] if atom(f, "concat") && bits.is_none() => {
            leaves_of(a, None, leaves) && leaves_of(b, None, leaves)
        }
        [f, b, one, zero] if atom(f, "ite") && atom(one, "#b1") && atom(zero, "#b0") => {
            bits.is_none() && leaves_of(b, None, leaves)
        }
        [f, x, one] if atom(f, "=") && atom(one, "#b1") => {
            bits.is_none() && leaves_of(x, None, leaves)
        }
        [Sexp::List(f), x] => match f.as_slice() {
            [u, e, hi, lo] if atom(u, "_") && atom(e, "extract") => {
                let (Some(hi), Some(lo)) = (number(hi), number(lo)) else {
                    return false;
                };
                let bits = match bits {
                    None => (hi, lo),
                    Some((h, l)) => (lo + h, lo + l),
                };
                bits.0 >= bits.1 && bits.0 <= hi && leaves_of(x, Some(bits), leaves)
            }
            _ => false,
        },
        _ => false,
    }
}

/// The conjuncts of `(|T_i| s)`, by the register each one names in its
/// comment.
fn init_conjuncts(text: &str, top: &str) -> Result<BTreeMap<String, String>, String> {
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
    /// The `$anyseq` symbol (`T#k`) whose value the wire `name` is in every
    /// state, or `None` if its accessor is anything else: a wire that is a
    /// fresh value in every step, of the same width. `write_smt2` defines
    /// such an accessor as `(|T#k| state)`, or for width 1 as
    /// `(= ((_ extract 0 0) (|T#k| state)) #b1)`.
    pub fn free_value(&self, name: &str) -> Option<&str> {
        let width = *self.wires.get(name)?;
        let head = format!(
            "(define-fun {} ((state {})) ",
            smt::symbol(&format!("{}_n {name}", self.top)),
            self.state_sort()
        );
        let line = self.text.lines().find(|l| l.starts_with(&head))?;
        let body = line[head.len()..].strip_suffix(')')?;
        let body = match width {
            1 => body
                .strip_prefix("Bool (= ((_ extract 0 0) ")?
                .strip_suffix(") #b1)")?,
            w => body.strip_prefix(&format!("(_ BitVec {w}) "))?,
        };
        let symbol = body.strip_prefix("(|")?.strip_suffix("| state)")?;
        (self.anyseq.get(symbol) == Some(&width)).then_some(symbol)
    }

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
    pub fn is_initial(&self, s: &str) -> String {
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
