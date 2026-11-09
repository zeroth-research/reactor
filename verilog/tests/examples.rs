//! The example designs in `examples/`: printed modules against golden
//! files, structure, the AST semantics against the
//! translated module, the SMT-LIB rounds against the translated module (with
//! z3), and the AST semantics against iverilog, on random initial register
//! values and random input traces: clock levels chosen at random in every
//! round (any interleaving of edges, simultaneous edges, rounds without an
//! edge) or by generators (toggling every round, or with a period per
//! clock). Designs with one rising-edge clock are also compared with the
//! cycle model: iverilog pulsing the clock once per cycle against the AST
//! semantics with a toggling clock, at the rounds where the clock is 0.
//!
//! `UPDATE_GOLDEN=1 cargo test -p verilog --test examples` rewrites the
//! golden files in `tests/golden/`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use verilog::ast::{Dir, Edge, Item, Module, Ty};
use verilog::ast_interp::{Interp, Values};
use verilog::elab::{self, Inst};
use verilog::lower::{Translation, init_name, lower_inst};
use verilog::rm_sim::{Sim, Valuation};
use verilog::smt::{self, Phase};
use verilog::{BitVec, typing, z3};

fn examples() -> Vec<PathBuf> {
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

fn stem(path: &Path) -> String {
    path.file_stem().unwrap().to_string_lossy().into_owned()
}

fn load(path: &Path) -> (Inst, Translation) {
    let inst = elab::load_file(path).unwrap_or_else(|e| panic!("{}:{e}", path.display()));
    let t = lower_inst(&inst).unwrap_or_else(|e| panic!("{}:{e}", path.display()));
    (inst, t)
}

/// The registers of a hierarchy by path, with their types.
fn registers(inst: &Inst) -> Vec<(String, Ty)> {
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

/// The number of atoms of a hierarchy: one per item, and one per input
/// connected to an expression other than a name.
fn atoms(inst: &Inst) -> usize {
    let m = &inst.module;
    let own = m
        .items
        .iter()
        .filter(|i| !matches!(i, Item::Instance(_)))
        .count();
    let conns = m
        .instances()
        .flat_map(|i| i.conns(Dir::Input))
        .filter(|(c, _)| {
            c.actual
                .as_ref()
                .is_some_and(|e| !matches!(e.kind, verilog::ast::ExprKind::Var(_)))
        })
        .count();
    own + conns + inst.children.iter().map(|(_, c)| atoms(c)).sum::<usize>()
}

#[test]
fn golden() {
    let update = std::env::var_os("UPDATE_GOLDEN").is_some();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden");
    for path in examples() {
        let (_, t) = load(&path);
        let printed = t.print();
        let golden = dir.join(format!("{}.rm", stem(&path)));
        if update {
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(&golden, &printed).unwrap();
        } else {
            let expected = std::fs::read_to_string(&golden).unwrap_or_else(|_| {
                panic!("{} missing, run with UPDATE_GOLDEN=1", golden.display())
            });
            assert_eq!(printed, expected, "{}", path.display());
        }
    }
}

/// Inputs (clocks included) and the initial values of the registers are
/// external, outputs interface, all other signals of the hierarchy private;
/// an unread input is no variable at all. The update of a register atom
/// awaits only its clock, for the edge test.
#[test]
fn structure() {
    for path in examples() {
        let (inst, t) = load(&path);
        let m = &inst.module;
        let names = |intf: &base::Interface<verilog::Sort>| {
            let mut names: Vec<String> = intf.iter().map(|v| t.name(v).to_string()).collect();
            names.sort();
            names
        };
        let sorted = |mut v: Vec<String>| {
            v.sort();
            v
        };
        let ports = |dir| {
            m.ports
                .iter()
                .filter(|p| p.dir == dir && !t.unread_inputs.contains(&p.decl.name))
                .map(|p| p.decl.name.clone())
                .collect::<Vec<_>>()
        };
        let mut extl = ports(Dir::Input);
        extl.extend(t.registers.iter().map(|q| init_name(q)));
        let display = path.display();
        assert_eq!(names(t.module.extl()), sorted(extl), "{display}");
        assert_eq!(
            names(t.module.intf()),
            sorted(ports(Dir::Output)),
            "{display}"
        );
        let top: BTreeSet<&String> = m.ports.iter().map(|p| &p.decl.name).collect();
        let private: BTreeSet<String> = inst
            .naming()
            .into_iter()
            .map(|(_, n)| n)
            .filter(|n| !top.contains(n))
            .collect();
        assert_eq!(
            names(t.module.prvt()),
            private.into_iter().collect::<Vec<_>>(),
            "{display}"
        );
        assert_eq!(t.module.atoms().len(), atoms(&inst), "{display}");
        let naming: BTreeMap<String, String> = inst.naming().into_iter().collect();
        let regs: BTreeMap<&String, &String> =
            t.registers.iter().map(|q| (&naming[q], q)).collect();
        for atom in t.module.atoms() {
            let ctrl = names(atom.ctrl());
            if !regs.contains_key(&ctrl[0]) {
                continue;
            }
            let mut inits: Vec<String> = ctrl.iter().map(|q| init_name(regs[q])).collect();
            let wait = names(atom.wait());
            let clock: Vec<&String> = wait.iter().filter(|v| t.clocks.contains(v)).collect();
            assert_eq!(clock.len(), 1, "{display}: {ctrl:?} awaits {wait:?}");
            inits.push(clock[0].clone());
            let clock = t.var(clock[0]);
            assert_eq!(wait, sorted(inits), "{display}");
            for term in atom.update().iter() {
                for w in term.read() {
                    let awaited = t.vars.values().any(|v| base::var::X(*v) == *w);
                    assert!(
                        !awaited || *w == base::var::X(clock),
                        "{display}: update of {ctrl:?} awaits"
                    );
                }
            }
        }
    }
    let (_, t) = load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/arith.sv"));
    assert_eq!(t.unread_inputs, ["clk"]);
    assert!(t.clocks.is_empty());
    let (_, t) = load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/accum.sv"));
    assert_eq!(t.clocks, ["clk"]);
    assert!(t.unread_inputs.is_empty());
    assert_eq!(t.registers, ["acc", "overflow"]);
}

/// A pseudo-random generator (64-bit LCG).
struct Rand(u64);

impl Rand {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 16
    }

    /// A value of the given width, biased towards the edge cases 0, 1 and
    /// all ones, and a few small values, to hit equalities and carries.
    fn bits(&mut self, width: usize) -> BitVec {
        let ones = (1u128 << width) - 1;
        let v = match self.next() % 8 {
            0 => 0,
            1 => 1,
            2 => ones,
            3 => (self.next() % 4) as u128,
            _ => ((self.next() as u128) << 64 | self.next() as u128) & ones,
        };
        BitVec::new(width, v & ones).unwrap()
    }

    /// A value for every input, clocks included.
    fn inputs(&mut self, m: &Module) -> Values {
        m.ports
            .iter()
            .filter(|p| p.dir == Dir::Input)
            .map(|p| (p.decl.name.clone(), self.bits(p.decl.ty.width)))
            .collect()
    }

    /// `rounds` values for every input, with the clocks `clocks` driven as
    /// `gen` says.
    fn trace(
        &mut self,
        m: &Module,
        clocks: &[String],
        clocking: Gen,
        rounds: usize,
    ) -> Vec<Values> {
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
                let mut v = self.inputs(m);
                if clocking != Gen::Random {
                    for (c, (period, first)) in clocks.iter().zip(&shapes) {
                        let level = (first + (k / period) as u64) % 2;
                        v.insert(c.clone(), BitVec::new(1, level as u128).unwrap());
                    }
                }
                v
            })
            .collect()
    }

    /// A value for every register, by path.
    fn registers(&mut self, inst: &Inst) -> Values {
        registers(inst)
            .into_iter()
            .map(|(q, ty)| (q, self.bits(ty.width)))
            .collect()
    }
}

/// How a trace drives the clocks.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Gen {
    /// A random level in every round, like every other input.
    Random,
    /// Every clock toggles in every round, all at once.
    Toggle,
    /// Every clock toggles with its own period of 1 to 3 rounds.
    Periods,
}

impl Gen {
    /// The generators, in turn.
    fn nth(k: usize) -> Self {
        [Gen::Random, Gen::Toggle, Gen::Periods][k % 3]
    }
}

/// The translated module computes the same values as the AST semantics,
/// from the same initial register values (given to `q@init` in the first
/// round, and random values without effect in later rounds), with clocks
/// at random levels or driven by generators.
#[test]
fn rm_matches_ast() {
    let mut rand = Rand(0x2545_f491_4f6c_dd1d);
    for path in examples() {
        let (inst, t) = load(&path);
        let m = &inst.module;
        for run in 0..12 {
            let init = rand.registers(&inst);
            let mut interp = Interp::new(&inst, &init).unwrap();
            let mut sim = Sim::new(&t.module);
            let trace = rand.trace(m, &t.clocks, Gen::nth(run), 100);
            for (k, inputs) in trace.iter().enumerate() {
                let expected = interp.step(inputs).unwrap();
                let regs = if k == 0 {
                    init.clone()
                } else {
                    rand.registers(&inst)
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
                        value,
                        &expected[name],
                        "{}: `{name}` in round {k} from {init:?} with inputs {inputs:?}",
                        path.display()
                    );
                    compared += 1;
                }
                assert_eq!(compared, expected.len() - t.unread_inputs.len());
            }
        }
    }
}

/// The SMT-LIB rounds define the next values `rm_sim` computes: on random
/// traces, z3 proves that each round, given the declared values of the
/// trace (latched values from the previous round, next values of external
/// variables from this one), defines the next value of every controlled
/// variable as in the trace. The first round is the init round, the
/// others are update rounds. Skipped (with a message) if z3 is not found.
#[test]
fn smt_matches_rm() {
    let Some(bin) = z3::find() else {
        eprintln!("z3 not found: skipped");
        return;
    };
    let mut rand = Rand(0x5851_f42d_4c95_7f2d);
    for path in examples() {
        let (_, t) = load(&path);
        let m = &t.module;
        let rounds = [Phase::Init, Phase::Update].map(|phase| smt::round(&t, phase));
        let mut controlled: Vec<&str> = m.ctrl().iter().map(|v| t.name(v)).collect();
        controlled.sort();
        for r in &rounds {
            let mut next: Vec<&str> = r.next.iter().map(String::as_str).collect();
            next.sort();
            assert_eq!(next, controlled, "{}", path.display());
            assert!(r.inputs.iter().all(|n| m.extl().contains(&t.var(n))));
        }
        assert!(rounds[0].latched.is_empty());

        let mut script = String::from("(set-logic QF_BV)\n");
        let mut checks = 0;
        for _ in 0..5 {
            let mut sim = Sim::new(m);
            let mut last: Option<Valuation> = None;
            for k in 0..10 {
                let extl: Valuation = m
                    .extl()
                    .iter()
                    .map(|v| (*v, rand.bits(v.dtype().width())))
                    .collect();
                let got = sim.step(&extl).unwrap();
                let r = &rounds[usize::from(k > 0)];
                script += "(push)\n";
                script += &r.text;
                if let Some(last) = &last {
                    let eq = equal(&t, &r.latched, last, smt::latched_symbol);
                    script += &format!("(assert {eq})\n");
                }
                let eq = equal(&t, &r.inputs, &got, smt::next_symbol);
                script += &format!("(assert {eq})\n");
                let eq = equal(&t, &r.next, &got, smt::next_symbol);
                script += &format!("(assert (not {eq}))\n(check-sat)\n(pop)\n");
                checks += 1;
                last = Some(got);
            }
        }
        let out = z3::run(&bin, &script).unwrap();
        assert_eq!(out, vec!["unsat"; checks], "{}", path.display());
    }
}

/// The SMT-LIB formula saying that the constant `sym(n)` is the value of
/// variable `n` in `values`, for each of the `names`.
fn equal(t: &Translation, names: &[String], values: &Valuation, sym: fn(&str) -> String) -> String {
    let eqs: Vec<String> = names
        .iter()
        .map(|n| format!("(= {} {})", sym(n), smt::lit(&values[&t.var(n)])))
        .collect();
    format!("(and true {})", eqs.join(" "))
}

fn iverilog_bin() -> Option<PathBuf> {
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

/// The declarations of a testbench for the top module of `inst`: the
/// ports and the instance `dut`.
fn tb_header(inst: &Inst) -> String {
    let m = &inst.module;
    let mut tb = String::from("module tb;\n");
    for p in &m.ports {
        let kind = if p.dir == Dir::Input { "logic" } else { "wire" };
        tb += &format!("  {kind} [{}:0] {};\n", p.decl.ty.width - 1, p.decl.name);
    }
    let conns: Vec<String> = m
        .ports
        .iter()
        .map(|p| format!(".{0}({0})", p.decl.name))
        .collect();
    tb += &format!("  {} dut({});\n  initial begin\n", m.name, conns.join(", "));
    tb
}

/// `    name = value;` for each of `values` whose name `pick` accepts.
fn tb_set(values: &Values, pick: impl Fn(&str) -> bool) -> String {
    let mut out = String::new();
    for (name, value) in values.iter().filter(|(n, _)| pick(n)) {
        out += &format!("    {name} = {}'h{:x};\n", value.width(), value.unsigned());
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
/// then sets the registers to `init`, by hierarchical assignment
/// (`dut.q = …`; iverilog accepts this although the standard forbids
/// writing an `always_ff` variable from another process), overriding
/// whatever an edge from the unknown starting level of a clock did.
fn testbench(inst: &Inst, clocks: &[String], init: &Values, rounds: &[Values]) -> String {
    let is_clock = |n: &str| clocks.iter().any(|c| c == n);
    let mut tb = tb_header(inst);
    for (k, v) in rounds.iter().enumerate() {
        tb += &tb_set(v, is_clock);
        tb += "    #1;\n";
        tb += &tb_set(v, |n| !is_clock(n));
        if k == 0 {
            tb += "    #1;\n";
            tb += &tb_init(init);
        }
        tb += &format!("    #1 {}", tb_display(inst));
    }
    tb += "    $finish;\n  end\nendmodule\n";
    tb
}

/// A testbench of the cycle model: it sets the registers to `init`, then
/// for each of `vectors` (a value for every input but the clock `clk`, if
/// any) sets the inputs, prints all signals but the inputs in hex (one
/// line per vector) and pulses the clock. The clock starts at 0, since a
/// change from x to 1 would be a rising edge.
fn cycle_testbench(inst: &Inst, clk: Option<&str>, init: &Values, vectors: &[Values]) -> String {
    let mut tb = tb_header(inst);
    if let Some(clk) = clk {
        tb += &format!("    {clk} = 1'b0;\n");
    }
    tb += &tb_init(init);
    for v in vectors {
        tb += &tb_set(v, |n| Some(n) != clk);
        tb += &format!("    #1 {}", tb_display(inst));
        match clk {
            Some(clk) => tb += &format!("    {clk} = 1'b1;\n    #1 {clk} = 1'b0;\n"),
            None => tb += "    #1;\n",
        }
    }
    tb += "    $finish;\n  end\nendmodule\n";
    tb
}

/// One cycle of the cycle model in the AST semantics, with the clock
/// `clk` as a generator: a round with the clock at 0, whose values are
/// those of the cycle, then a round with the clock at 1, the rising edge.
fn cycle(interp: &mut Interp, clk: &str, inputs: &Values) -> Values {
    let mut v = inputs.clone();
    v.insert(clk.to_string(), BitVec::new(1, 0u8).unwrap());
    let values = interp.step(&v).unwrap();
    v.insert(clk.to_string(), BitVec::new(1, 1u8).unwrap());
    interp.step(&v).unwrap();
    values
}

/// The signals compared with iverilog: all but the inputs of the top
/// module, by name (which is a hierarchical name, see `elab`).
fn shown(inst: &Inst) -> Vec<String> {
    let inputs: BTreeSet<&String> = inst
        .module
        .ports
        .iter()
        .filter(|p| p.dir == Dir::Input)
        .map(|p| &p.decl.name)
        .collect();
    let mut out: Vec<String> = Vec::new();
    for (_, name) in inst.naming() {
        if !inputs.contains(&name) && !out.contains(&name) {
            out.push(name);
        }
    }
    out
}

/// The signals compared with iverilog in `values`, as one line of hex
/// numbers.
fn hex_line(inst: &Inst, values: &Values) -> String {
    shown(inst)
        .into_iter()
        .map(|x| {
            let v = &values[&x];
            format!("{:0w$x}", v.unsigned(), w = v.width().div_ceil(4))
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Compiles and runs a testbench for the design at `path`, and returns
/// its output lines.
fn run_iverilog(bin: &Path, path: &Path, name: &str, tb: &str) -> Vec<String> {
    let tmp = Path::new(env!("CARGO_TARGET_TMPDIR")).join("iverilog");
    std::fs::create_dir_all(&tmp).unwrap();
    let file = tmp.join(format!("tb_{name}.sv"));
    let out = tmp.join(format!("tb_{name}.vvp"));
    std::fs::write(&file, tb).unwrap();
    let compile = Command::new(bin.join("iverilog"))
        .args(["-g2012", "-o"])
        .args([&out, &file, &path.to_path_buf()])
        .output()
        .unwrap();
    assert!(
        compile.status.success(),
        "{}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let run = Command::new(bin.join("vvp"))
        .args(["-n"])
        .arg(&out)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&run.stdout);
    stdout
        .lines()
        .filter(|l| !l.contains("$finish"))
        .map(String::from)
        .collect()
}

/// The AST semantics agree with iverilog on random traces, round by round.
/// Skipped (with a message) if iverilog is not found in `$IVERILOG_BIN`,
/// `~/tools/oss-cad-suite/bin` or the `PATH`.
#[test]
fn ast_matches_iverilog() {
    let Some(bin) = iverilog_bin() else {
        eprintln!("iverilog not found: skipped");
        return;
    };
    let mut rand = Rand(0x9e37_79b9_7f4a_7c15);
    for (n, path) in examples().into_iter().enumerate() {
        let (inst, t) = load(&path);
        let m = &inst.module;
        let init = rand.registers(&inst);
        let rounds = rand.trace(m, &t.clocks, Gen::nth(n), 300);
        let mut interp = Interp::new(&inst, &init).unwrap();
        let expected: Vec<String> = rounds
            .iter()
            .map(|v| hex_line(&inst, &interp.step(v).unwrap()))
            .collect();
        let tb = testbench(&inst, &t.clocks, &init, &rounds);
        let got = run_iverilog(&bin, &path, &m.name, &tb);
        assert_eq!(got.len(), expected.len(), "{}: {got:?}", path.display());
        for (k, (g, e)) in got.iter().zip(&expected).enumerate() {
            assert_eq!(g, e, "{}: round {k} = {:?}", path.display(), rounds[k]);
        }
    }
}

/// The round semantics with a toggling clock is the cycle model: for
/// designs whose blocks all run at the rising edge of one clock, iverilog
/// pulsing the clock once per cycle agrees with the AST semantics at the
/// rounds where the clock is 0. Skipped like [`ast_matches_iverilog`].
#[test]
fn rounds_match_cycles() {
    let Some(bin) = iverilog_bin() else {
        eprintln!("iverilog not found: skipped");
        return;
    };
    let mut rand = Rand(0x1405_7b7e_f767_814f);
    let mut compared = 0;
    for path in examples() {
        let (inst, t) = load(&path);
        if t.clocks.len() > 1 || negedge(&inst) {
            continue;
        }
        let m = &inst.module;
        let clk = t.clocks.first().map(String::as_str);
        let init = rand.registers(&inst);
        let vectors: Vec<Values> = (0..200)
            .map(|_| {
                let mut v = rand.inputs(m);
                v.retain(|n, _| Some(n.as_str()) != clk);
                v
            })
            .collect();
        let mut interp = Interp::new(&inst, &init).unwrap();
        let expected: Vec<String> = vectors
            .iter()
            .map(|v| match clk {
                Some(clk) => hex_line(&inst, &cycle(&mut interp, clk, v)),
                None => hex_line(&inst, &interp.step(v).unwrap()),
            })
            .collect();
        let tb = cycle_testbench(&inst, clk, &init, &vectors);
        let got = run_iverilog(&bin, &path, &format!("{}_cycles", m.name), &tb);
        assert_eq!(got, expected, "{}", path.display());
        compared += 1;
    }
    assert!(compared > 20, "{compared}");
}

/// Whether some block of the hierarchy runs at a falling edge.
fn negedge(inst: &Inst) -> bool {
    let own = inst.module.items.iter().any(|i| {
        matches!(
            i,
            Item::AlwaysFf {
                edge: Edge::Neg,
                ..
            }
        )
    });
    own || inst.children.iter().any(|(_, c)| negedge(c))
}
