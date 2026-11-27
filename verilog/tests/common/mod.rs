//! Helpers shared by the integration tests: finding the external tools,
//! temporary directories, translating sources and checking errors, the
//! examples, random values and traces, and the checks every accepted
//! design must pass (the translated module against the AST semantics on
//! random traces, and against Yosys's translation with the miter).
//!
//! Every test binary includes this module and uses a part of it.
#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use num_bigint::BigUint;
use verilog::ast::{Edge, Error, ErrorKind, Item, Ty};
use verilog::ast_interp::{Interp, Values};
use verilog::elab::{self, Inst};
use verilog::lower::{Translation, init_name, lower_inst};
use verilog::miter::{self, Verdict};
use verilog::rm_sim::{Sim, Valuation};
use verilog::{BitVec, any, qfbv, typing, yosys, z3};

// ---------------------------------------------------------------------
// external tools

/// Whether the environment variable `V2RM_REQUIRE_TOOLS` is `1`: then a
/// test that needs an external tool (Yosys, z3, iverilog) fails when the
/// tool is not found, instead of skipping its work. For CI and for runs
/// whose result must cover the comparisons with the tools.
pub fn tools_required() -> bool {
    std::env::var_os("V2RM_REQUIRE_TOOLS").is_some_and(|v| v == "1")
}

/// Reports that the running test leaves out `what` (all of its work, or a
/// part) because `tool` was not found, once per test. With
/// `V2RM_REQUIRE_TOOLS=1` (see [`tools_required`]) it panics instead, so
/// that the test fails.
///
/// The message is written to the standard error stream directly, not with
/// `eprintln!`: the test harness captures `eprintln!` (shown only for
/// failed tests or with `--nocapture`), but not writes to the stream
/// itself, so a skip shows in every run of `cargo test`.
pub fn missing(tool: &str, what: &str) {
    static REPORTED: Mutex<BTreeSet<(String, String, String)>> = Mutex::new(BTreeSet::new());
    let thread = std::thread::current();
    let test = thread.name().unwrap_or("?");
    if tools_required() {
        panic!("{tool} not found, needed for {what} (V2RM_REQUIRE_TOOLS=1)");
    }
    let key = (test.to_string(), tool.to_string(), what.to_string());
    if !REPORTED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(key)
    {
        return;
    }
    let _ = writeln!(
        std::io::stderr(),
        "SKIPPED in {test}: {what}: {tool} not found (V2RM_REQUIRE_TOOLS=1 turns this into a failure)"
    );
}

/// Yosys and z3, or `None` (with a message, see [`missing`]) if one is
/// missing; `what` is the work left out.
pub fn tools_for(what: &str) -> Option<(PathBuf, PathBuf)> {
    let (y, z) = (yosys::find(), z3::find());
    if y.is_none() {
        missing("yosys", what);
    }
    if z.is_none() {
        missing("z3", what);
    }
    Some((y?, z?))
}

/// Yosys and z3, or `None` if one is missing (the whole test is skipped).
pub fn tools() -> Option<(PathBuf, PathBuf)> {
    tools_for("the whole test")
}

// ---------------------------------------------------------------------
// temporary files

/// A new empty directory below the test target's temporary directory,
/// named after `label`, the process and a counter, so that runs at the
/// same time (two `cargo test`, or tests of one run) never share files.
/// Removed with its contents when dropped, unless the test fails (then it
/// stays for inspection).
pub struct TempDir(PathBuf);

impl TempDir {
    pub fn new(label: &str) -> Self {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = Path::new(env!("CARGO_TARGET_TMPDIR"));
        std::fs::create_dir_all(root).unwrap();
        loop {
            let k = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = root.join(format!("{label}-{}-{k}", std::process::id()));
            match std::fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("{}: {e}", path.display()),
            }
        }
    }

    pub fn path(&self) -> &Path {
        &self.0
    }

    pub fn join(&self, name: impl AsRef<Path>) -> PathBuf {
        self.0.join(name)
    }

    /// Writes `text` to the file `name` in the directory; its path.
    pub fn write(&self, name: &str, text: &str) -> PathBuf {
        let path = self.0.join(name);
        std::fs::write(&path, text).unwrap();
        path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

// ---------------------------------------------------------------------
// translating sources

/// The translation of the design `src` (its top module and every
/// instance below it), from a file `test.sv`.
pub fn translate(src: &str) -> Result<Translation, Error> {
    let inst = elab::load(src, "test.sv")?;
    lower_inst(&inst)
}

/// Translates `src` and expects an error at the first occurrence of
/// `needle` in line `line` of `src`, with a message containing `msg`.
pub fn rejects(src: &str, line: usize, needle: &str, msg: &str) {
    let err = match translate(src) {
        Ok(_) => panic!("accepted:\n{src}"),
        Err(e) => e,
    };
    assert!(err.msg.contains(msg), "{src}\n{err}\nexpected: {msg}");
    let text = src.lines().nth(line - 1).unwrap();
    let col = text
        .find(needle)
        .unwrap_or_else(|| panic!("`{needle}` not in `{text}`"))
        + 1;
    assert_eq!(
        (err.span.unwrap().line, err.span.unwrap().col),
        (line, col),
        "{src}\n{err}\nexpected at `{needle}`"
    );
}

/// A module `name` with the port list `ports` and the body `body`; the
/// body starts in line 2.
pub fn module(name: &str, ports: &str, body: &str) -> String {
    format!("module {name}({ports});\n{body}\nendmodule\n")
}

// ---------------------------------------------------------------------
// the examples

/// The example designs `examples/*.sv`, sorted.
pub fn examples() -> Vec<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "sv"))
        .collect();
    paths.sort();
    assert!(!paths.is_empty());
    paths
}

/// The file name of `path` without its extension.
pub fn stem(path: &Path) -> String {
    path.file_stem().unwrap().to_string_lossy().into_owned()
}

/// The hierarchy and the translation of the design in the file `path`.
pub fn load(path: &Path) -> (Inst, Translation) {
    let inst = elab::load_file(path).unwrap_or_else(|e| panic!("{}:{e}", path.display()));
    let t = lower_inst(&inst).unwrap_or_else(|e| panic!("{}:{e}", path.display()));
    (inst, t)
}

/// Every register of a hierarchy, by path, with its type.
pub fn registers(inst: &Inst) -> BTreeMap<String, Ty> {
    inst.registers()
        .into_iter()
        .map(|path| {
            let (prefix, q) = match path.rsplit_once('.') {
                Some((p, q)) => (format!("{p}."), q.to_string()),
                None => (String::new(), path.clone()),
            };
            let env = typing::env(&inst.at(&prefix).module).unwrap();
            let ty = env[&q].ty;
            (path, ty)
        })
        .collect()
}

/// The registers of a hierarchy without an initial value (those whose
/// power-on value is free), by path, with their types.
pub fn uninitialized(inst: &Inst) -> BTreeMap<String, Ty> {
    let initialized = inst.initialized();
    registers(inst)
        .into_iter()
        .filter(|(p, _)| !initialized.contains(p))
        .collect()
}

/// `name=hex …` as values of the signals' widths.
pub fn values(text: &str, width: impl Fn(&str) -> usize) -> Values {
    text.split_whitespace()
        .map(|kv| {
            let (k, v) = kv.split_once('=').unwrap();
            let n = BigUint::parse_bytes(v.as_bytes(), 16).unwrap();
            (k.to_string(), BitVec::new(width(k), n).unwrap())
        })
        .collect()
}

pub use verilog::iverilog::design_inputs;

/// The asynchronous resets of the hierarchy, by name (see
/// `Inst::naming`), with their edges (`posedge` for an active-high reset).
pub fn resets(inst: &Inst) -> BTreeMap<String, Edge> {
    let naming: BTreeMap<String, String> = inst.naming().into_iter().collect();
    let mut out = BTreeMap::new();
    let mut walk = vec![(inst, String::new())];
    while let Some((i, prefix)) = walk.pop() {
        for item in &i.module.items {
            if let Item::AlwaysFf { reset: Some(r), .. } = item {
                out.insert(naming[&format!("{prefix}{}", r.name)].clone(), r.edge);
            }
        }
        for (u, child) in &i.children {
            walk.push((child, format!("{prefix}{u}.")));
        }
    }
    out
}

// ---------------------------------------------------------------------
// random values

/// A seed for the randomness of one design, from `base` and the design's
/// name (FNV-1a): each design gets its own sequence, which adding or
/// removing another design does not change.
pub fn seed(base: u64, name: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in name.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    base ^ h
}

/// A pseudo-random generator (64-bit LCG).
pub struct Rand(pub u64);

impl Rand {
    pub fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 16
    }

    /// A number below `n`.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// A value of the given width, biased towards the edge cases 0, 1, all
    /// ones, the most negative and the largest signed value, and small
    /// values, to hit equalities, carries and overflows.
    pub fn bits(&mut self, width: usize) -> BitVec {
        match self.below(10) {
            0 => BitVec::zero(width),
            1 => BitVec::wrap(width, 1u8),
            2 => BitVec::from_signed(width, -1),
            3 => BitVec::wrap(width, BigUint::from(1u8) << (width - 1)),
            4 => BitVec::wrap(width, (BigUint::from(1u8) << (width - 1)) - 1u8),
            5 => BitVec::wrap(width, self.below(4)),
            _ => {
                let mut v = BigUint::from(0u8);
                for _ in 0..width.div_ceil(48) {
                    v = (v << 48) | BigUint::from(self.next());
                }
                BitVec::wrap(width, v)
            }
        }
    }

    /// A value for every input of the design (see [`design_inputs`]),
    /// clocks included.
    pub fn inputs(&mut self, inst: &Inst) -> Values {
        design_inputs(inst)
            .into_iter()
            .map(|(name, ty, _)| (name, self.bits(ty.width)))
            .collect()
    }

    /// A value for every register without an initial value, by path.
    pub fn registers(&mut self, inst: &Inst) -> Values {
        uninitialized(inst)
            .into_iter()
            .map(|(q, ty)| (q, self.bits(ty.width)))
            .collect()
    }

    /// `rounds` values for every input of the top of `inst`, with the
    /// clocks `clocks` driven as `gen` says. With a generator, the inputs
    /// that are asynchronous resets are active in the first two rounds and
    /// then in one round of eight on average (random values would keep
    /// them active half the time).
    pub fn trace(
        &mut self,
        inst: &Inst,
        clocks: &[String],
        clocking: Gen,
        rounds: usize,
    ) -> Vec<Values> {
        let resets = resets(inst);
        // per clock: its period (in rounds per level) and first level
        let shapes: Vec<(usize, u64)> = clocks
            .iter()
            .map(|_| {
                let period = match clocking {
                    Gen::Toggle => 1,
                    _ => 1 + (self.next() % 3) as usize,
                };
                (period, self.next() % 2)
            })
            .collect();
        (0..rounds)
            .map(|k| {
                let mut v = self.inputs(inst);
                if clocking != Gen::Random {
                    for (c, (period, first)) in clocks.iter().zip(&shapes) {
                        let level = (first + (k / period) as u64) % 2;
                        v.insert(c.clone(), BitVec::new(1, level as u128).unwrap());
                    }
                    for (r, edge) in &resets {
                        if let Some(w) = v.get(r).map(BitVec::width) {
                            let active = k < 2 || self.next().is_multiple_of(8);
                            let level = active == (*edge == Edge::Pos);
                            v.insert(r.clone(), BitVec::new(w, level as u128).unwrap());
                        }
                    }
                }
                v
            })
            .collect()
    }
}

/// How a trace drives the clocks.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Gen {
    /// A random level in every round, like every other input.
    Random,
    /// Every clock toggles in every round, all at once.
    Toggle,
    /// Every clock toggles with its own period of 1 to 3 rounds.
    Periods,
}

impl Gen {
    /// The generators, in turn.
    pub fn nth(k: usize) -> Self {
        [Gen::Random, Gen::Toggle, Gen::Periods][k % 3]
    }
}

// ---------------------------------------------------------------------
// checks of accepted designs

/// The translated module `t` of `inst` computes the same values as the
/// AST semantics, from the same initial register values (given to
/// `q@init` in the first round, and random values without effect in later
/// rounds), on `runs` random traces of `rounds` rounds, with clocks at
/// random levels or driven by generators (resets then in pulses). `label`
/// names the design in a failure.
pub fn rm_matches_ast(
    label: &str,
    (inst, t): (&Inst, &Translation),
    rand: &mut Rand,
    runs: usize,
    rounds: usize,
) {
    for run in 0..runs {
        let init = rand.registers(inst);
        let mut interp = Interp::new(inst, &init).unwrap();
        let mut sim = Sim::new(&t.module);
        let trace = rand.trace(inst, &t.clocks, Gen::nth(run), rounds);
        for (k, inputs) in trace.iter().enumerate() {
            let expected = interp.step(inputs).unwrap();
            let regs = if k == 0 {
                init.clone()
            } else {
                rand.registers(inst)
            };
            let extl: Valuation = inputs
                .iter()
                .map(|(n, v)| (t.var(n), v.clone()))
                .chain(regs.iter().map(|(q, v)| (t.var(&init_name(q)), v.clone())))
                .filter(|(var, _)| t.module.extl().contains(var))
                .collect();
            let got = sim.step(&extl).unwrap();
            let mut compared = 0;
            for (var, value) in &got {
                let name = t.name(var);
                if name.contains('@') {
                    continue;
                }
                assert_eq!(
                    value, &expected[name],
                    "{label}: `{name}` in round {k} from {init:?} with inputs {inputs:?}"
                );
                compared += 1;
            }
            assert_eq!(compared, expected.len() - t.unread_inputs.len(), "{label}");
        }
    }
}

/// Whether the miter of our translation `t` of the file `path` and
/// Yosys's proves the init round and one update round equal on every
/// controlled variable; if not, the queries not proved equal.
pub fn miter_equal(
    (yosys, z3): (&Path, &Path),
    path: &Path,
    t: &Translation,
) -> Result<(), String> {
    let ys = yosys::run(yosys, &[path], &t.name, &[]).map_err(|e| format!("Yosys: {e}"))?;
    let m = miter::build(t, &ys)?;
    let differ: Vec<String> = miter::check(z3, &m)?
        .into_iter()
        .filter(|(_, _, v)| *v != Verdict::Equal)
        .map(|(phase, name, v)| format!("`{name}` in {phase:?}: {v:?}"))
        .collect();
    if differ.is_empty() {
        Ok(())
    } else {
        Err(differ.join("\n"))
    }
}

/// Rewrites the module of `t` into the SMT-LIB bit-vector op set
/// ([`qfbv::bridge`]) and that into the ops of `theory::bv`
/// ([`any::desugar`]), and checks with z3 that the init and update rounds
/// of each compute what those of the one before compute; then exports the
/// last into `theory::any` and checks that it is the same module
/// ([`exported_equal`]). Returns the bridged and the desugared module, or `None` if `t` has
/// words too wide for the op sets; skips the z3 checks (with a message, see
/// [`missing`]) if z3 is missing. `label` names the design in a failure.
pub fn bridge_equal(label: &str, t: &Translation) -> Option<(qfbv::Bridged, any::Desugared)> {
    let b = match qfbv::bridge(t) {
        Ok(b) => b,
        Err(e) if e.kind == ErrorKind::Unsupported => return None,
        Err(e) => panic!("{label}bridging: {e}"),
    };
    let d = desugar_equal(label, &b);
    if let Some(z3) = z3::find() {
        let differ = qfbv::bridge::check(&z3, t, &b).unwrap_or_else(|e| panic!("{label}z3: {e}"));
        assert!(
            differ.is_empty(),
            "{label}the bridged module differs: {differ:?}"
        );
    }
    exported_equal(label, &d);
    Some((b, d))
}

/// Desugars the `qfbv` module `b` ([`any::desugar`]) and checks with z3
/// that its rounds compute what those of `b` compute (skipped with a
/// message if z3 is missing).
pub fn desugar_equal(label: &str, b: &qfbv::Bridged) -> any::Desugared {
    let d = any::desugar(b).unwrap_or_else(|e| panic!("{label}desugaring: {e}"));
    let Some(z3) = z3::find() else {
        missing("z3", "the check of the bridged and desugared modules");
        return d;
    };
    let differ = any::desugar::check(&z3, b, &d).unwrap_or_else(|e| panic!("{label}z3: {e}"));
    assert!(
        differ.is_empty(),
        "{label}the desugared module differs: {differ:?}"
    );
    d
}

/// Exports `d` into `theory::any` (constants as `Uninterpreted` ops named
/// by their value: this crate cannot make tensors) and checks that the
/// result is `d` renamed: the same variables (by width and class), atoms,
/// blocks and terms, each term with the op of the same name on wires of the
/// same widths, a wire written or read in two places in one of the same in
/// the other.
pub fn exported_equal(label: &str, d: &any::Desugared) -> any::Exported {
    use theory::any::Sort as S;
    use theory::bv::BV;
    let e = any::export(d, |v, w| BV::Uninterpreted(format!("{w}'d{v}")))
        .unwrap_or_else(|e| panic!("{label}export: {e}"));
    let (m, n) = (&d.module, &e.module);
    let width = |s: &S| match s {
        S::BitVec(w, [1, 1]) => *w,
        _ => panic!("{label}exported sort {s}"),
    };
    for (ours, theirs) in [
        (m.extl(), n.extl()),
        (m.intf(), n.intf()),
        (m.prvt(), n.prvt()),
    ] {
        let mapped: BTreeSet<_> = ours.iter().map(|v| e.vars[v]).collect();
        let theirs: BTreeSet<_> = theirs.iter().copied().collect();
        assert_eq!(mapped, theirs, "{label}exported variables");
    }
    for (v, x) in &e.vars {
        assert_eq!(
            v.dtype().width(),
            width(x.dtype()),
            "{label}width of a variable"
        );
    }
    assert_eq!(m.atoms().len(), n.atoms().len(), "{label}exported atoms");
    let mut wires = std::collections::HashMap::new();
    for (a, b) in m.atoms().iter().zip(n.atoms()) {
        let blocks = [
            (
                a.init().iter().collect::<Vec<_>>(),
                b.init()
                    .iter()
                    .map(|t| (t.itype().to_string(), t.write(), t.read()))
                    .collect::<Vec<_>>(),
            ),
            (
                a.update().iter().collect(),
                b.update()
                    .iter()
                    .map(|t| (t.itype().to_string(), t.write(), t.read()))
                    .collect(),
            ),
        ];
        for (ours, theirs) in blocks {
            assert_eq!(ours.len(), theirs.len(), "{label}exported block");
            for (t, (op, write, read)) in ours.iter().zip(theirs) {
                let name = match t.itype() {
                    any::Op::Const { value, width } => format!("Uninterpreted({width}'d{value})"),
                    op => op.to_string(),
                };
                assert_eq!(name, op, "{label}exported op");
                let pairs = t.write().iter().zip(write).chain(t.read().iter().zip(read));
                assert_eq!(t.write().len() + t.read().len(), write.len() + read.len());
                for (x, y) in pairs {
                    assert_eq!(x.dtype().width(), width(y.dtype()), "{label}wire width");
                    assert_eq!(
                        *wires.entry(*x).or_insert(*y),
                        *y,
                        "{label}wire renamed twice"
                    );
                }
            }
        }
    }
    e
}

/// How an accepted design is checked by [`accepted_with`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Check {
    /// Against the AST semantics and Yosys.
    All,
    /// Against the AST semantics only, for the reason given: Yosys reads
    /// the design otherwise than the standard, or the miter cannot compare
    /// what Yosys makes of it.
    NotYosys(&'static str),
}

/// Translates the accepted design `src` (panics if it is rejected) and
/// checks the translation: against the AST semantics on random traces,
/// and against Yosys's translation with the miter (skipped with a message
/// if Yosys or z3 is missing).
pub fn accepted(src: &str) -> Translation {
    accepted_with(src, Check::All)
}

/// [`accepted`], with the checks `check`.
pub fn accepted_with(src: &str, check: Check) -> Translation {
    let dir = TempDir::new("accepted");
    let path = dir.write("test.sv", src);
    let inst = elab::load(src, &path).unwrap_or_else(|e| panic!("{src}\n{e}"));
    let t = lower_inst(&inst).unwrap_or_else(|e| panic!("{src}\n{e}"));
    let label = format!("accepted design\n{src}\n");
    let mut rand = Rand(seed(0x9e37_79b9_7f4a_7c15, src));
    rm_matches_ast(&label, (&inst, &t), &mut rand, 3, 30);
    bridge_equal(&label, &t);
    if check == Check::All
        && let Some((yosys, z3)) = tools_for("the comparison of accepted designs with Yosys")
        && let Err(e) = miter_equal((&yosys, &z3), &path, &t)
    {
        panic!("{label}differs from Yosys:\n{e}");
    }
    t
}
