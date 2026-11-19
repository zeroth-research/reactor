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
//! Designs where a block on a derived clock reads registers changing at
//! its edge are not compared with iverilog, which gives their new values
//! there (see `simulator_differs`).
//!
//! `UPDATE_GOLDEN=1 cargo test -p verilog --test examples` rewrites the
//! golden files in `tests/golden/`. The comparisons with z3 and iverilog
//! are skipped with a message if the tool is not found, and fail instead
//! with `V2RM_REQUIRE_TOOLS=1` (see `common::missing`).

mod common;

use common::{Gen, Rand, TempDir, design_inputs, examples, load, seed, stem};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use verilog::ast::{Dir, Edge, Item};
use verilog::ast_interp::{Interp, Values};
use verilog::elab::{self, Inst};
use verilog::lower::{Translation, init_name};
use verilog::rm_sim::{Sim, Valuation};
use verilog::smt::{self, Phase};
use verilog::{BitVec, sizing, typing, z3};

/// The number of atoms of a hierarchy: one per item, and one per input
/// connected to an expression other than a name (blackboxes have none).
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
        .filter(|(c, f)| !f.blackbox && c.actual.as_ref().is_some_and(|_| c.shared().is_none()))
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

/// The sized source of every example (`v2rm --sized`): against golden
/// files, in the explicit-width fragment when read back (every module of
/// it, before any sizing), and simulated by iverilog like the original on
/// a random trace, so that the sizing pass is checked on its own.
#[test]
fn sized_source() {
    let update = std::env::var_os("UPDATE_GOLDEN").is_some();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden");
    let tmp = TempDir::new("sized");
    let bin = iverilog_bin();
    for path in examples() {
        let (inst, t) = load(&path);
        let mut rand = Rand(seed(0x51ed_270b_27a4_c19f, &stem(&path)));
        let printed = sizing::print(&inst).unwrap();
        let golden = dir.join(format!("{}.sized.sv", stem(&path)));
        if update {
            std::fs::write(&golden, &printed).unwrap();
        } else {
            let expected = std::fs::read_to_string(&golden).unwrap_or_else(|_| {
                panic!("{} missing, run with UPDATE_GOLDEN=1", golden.display())
            });
            assert_eq!(printed, expected, "{}", path.display());
        }
        let file = tmp.join(format!("{}.sv", stem(&path)));
        std::fs::write(&file, &printed).unwrap();
        // the files of `$readmemb`/`$readmemh`, next to the printed source
        for data in std::fs::read_dir(path.parent().unwrap()).unwrap() {
            let data = data.unwrap().path();
            if data.extension().is_some_and(|e| e != "sv") {
                std::fs::copy(&data, tmp.join(data.file_name().unwrap())).unwrap();
            }
        }
        let back = elab::load_file(&file).unwrap_or_else(|e| panic!("{}:{e}", file.display()));
        let mut walk = vec![&back];
        while let Some(i) = walk.pop() {
            let env = typing::check(&i.module).unwrap();
            typing::explicit(&env, &i.module).unwrap_or_else(|e| panic!("{}:{e}", file.display()));
            walk.extend(i.children.iter().map(|(_, c)| c));
        }
        let Some(bin) = &bin else { continue };
        if simulator_differs(&inst).is_some() {
            continue;
        }
        let init = rand.registers(&inst);
        let clocking = Gen::nth(rand.below(3));
        let rounds = rand.trace(&inst, &t.clocks, clocking, 100);
        let tb = testbench(&inst, &t, &init, &rounds);
        let name = &inst.module.name;
        // own file names: other tests run iverilog on the examples at the
        // same time
        let original = run_iverilog(bin, &path, &format!("{name}_original"), &tb);
        let sized = run_iverilog(bin, &file, &format!("{name}_sized"), &tb);
        assert_eq!(original.len(), rounds.len(), "{}", path.display());
        assert_eq!(original, sized, "{}", path.display());
    }
    if bin.is_none() {
        common::missing("iverilog", "the simulation of the sized sources");
    }
}

/// Inputs (clocks included) and the initial values of the registers are
/// external, outputs interface, all other signals of the hierarchy private;
/// an unread input is no variable at all. The update of a register atom
/// awaits only its clock, for the edge test, and its asynchronous reset.
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
        let free = t.registers.iter().filter(|q| !t.initialized.contains(q));
        extl.extend(free.map(|q| init_name(q)));
        // the outputs of blackboxes
        let outputs: Vec<String> = design_inputs(&inst)
            .into_iter()
            .skip(m.ports.iter().filter(|p| p.dir == Dir::Input).count())
            .map(|(name, ..)| name)
            .collect();
        extl.extend(outputs.iter().cloned());
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
            .filter(|n| !top.contains(n) && !outputs.contains(n))
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
        let clocking = clocking(&inst);
        for atom in t.module.atoms() {
            let ctrl = names(atom.ctrl());
            if !regs.contains_key(&ctrl[0]) {
                continue;
            }
            // the initial values, the clock (an input clock, or a register
            // or a copy of one: a derived clock) and the reset
            let mut inits: Vec<String> = (ctrl.iter())
                .filter(|q| !t.initialized.contains(regs[*q]))
                .map(|q| init_name(regs[q]))
                .collect();
            let wait = names(atom.wait());
            let (clock, reset) = &clocking[&ctrl[0]];
            let edges: Vec<&String> = [clock].into_iter().chain(reset).collect();
            inits.extend(edges.iter().map(|x| x.to_string()));
            assert_eq!(wait, sorted(inits), "{display}: {ctrl:?} awaits {wait:?}");
            for term in atom.update().iter() {
                for w in term.read() {
                    let awaited = t.vars.values().any(|v| base::var::X(*v) == *w);
                    assert!(
                        !awaited || edges.iter().any(|x| *w == base::var::X(t.var(x))),
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

/// The translated module computes the same values as the AST semantics,
/// from the same initial register values (given to `q@init` in the first
/// round, and random values without effect in later rounds), with clocks
/// at random levels or driven by generators (resets then in pulses).
#[test]
fn rm_matches_ast() {
    let paths = examples();
    std::thread::scope(|scope| {
        for path in &paths {
            scope.spawn(move || {
                let (inst, t) = load(path);
                let mut rand = Rand(seed(0x2545_f491_4f6c_dd1d, &stem(path)));
                let label = path.display().to_string();
                common::rm_matches_ast(&label, (&inst, &t), &mut rand, 12, 100);
            });
        }
    });
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
        common::missing("z3", "the whole test");
        return;
    };
    for path in examples() {
        let (_, t) = load(&path);
        let mut rand = Rand(seed(0x5851_f42d_4c95_7f2d, &stem(&path)));
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
    let paths: BTreeMap<String, String> = design_inputs(inst)
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
/// then sets the registers to `init`, by hierarchical assignment
/// (`dut.q = …`; iverilog accepts this although the standard forbids
/// writing an `always_ff` variable from another process), overriding
/// whatever an edge from x of a derived clock did (input clocks start at
/// their level in the first round, so that registers with an initial value
/// see no edge from x either), and
/// only then sets the other inputs: an asynchronous reset active in the
/// first round then resets its registers (its change from x is an edge
/// of the reset). `t.reset_registers` are set only once: an edge from x
/// of a reset that is a register can have reset them already.
fn testbench(inst: &Inst, t: &Translation, init: &Values, rounds: &[Values]) -> String {
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
/// [`tb_init_twice`] says), then for each of `vectors` (a value for every
/// input but the clock of `t`, if any) sets the inputs, prints all signals but the inputs in hex (one
/// line per vector) and pulses the clock. The clock starts at 0, since a
/// change from x to 1 would be a rising edge.
fn cycle_testbench(inst: &Inst, t: &Translation, init: &Values, vectors: &[Values]) -> String {
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

/// The signals compared with iverilog: all but the inputs of the design
/// (see [`design_inputs`]), by name (which is a hierarchical name, see
/// `elab`).
fn shown(inst: &Inst) -> Vec<String> {
    let inputs: BTreeSet<String> = design_inputs(inst)
        .into_iter()
        .map(|(name, ..)| name)
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

/// Whether iverilog's output line `got` agrees with the expected `line`
/// (both as [`hex_line`] prints them). In a signal whose entry of `x` is
/// set (see [`may_be_x`]), a hex digit that iverilog prints as x (wholly
/// or partly unknown) agrees with any digit: iverilog gives x to bits read
/// outside a signal, where we read 0. Other signals must agree exactly.
fn agrees(got: &str, line: &str, x: &[bool]) -> bool {
    let got: Vec<&str> = got.split(' ').collect();
    let line: Vec<&str> = line.split(' ').collect();
    got.len() == line.len()
        && got.iter().zip(&line).zip(x).all(|((g, e), &x)| {
            g == e
                || x && g.len() == e.len()
                    && g.chars()
                        .zip(e.chars())
                        .all(|(g, e)| g == e || matches!(g, 'x' | 'X'))
        })
}

/// For each signal of [`shown`], whether iverilog may give it x bits where
/// we compute values: whether it depends, through any item (registers
/// included, which keep an x once they latch it), on a read that may select
/// bits outside a signal (the reads of `t.out_of_range`). Each such read is
/// found by its position, in the instance its warning names. Dependence is
/// syntactic, per assignment: its targets depend on the reads of its
/// right-hand side, of the indices of its target and of the conditions and
/// `case` selectors around it.
fn may_be_x(inst: &Inst, t: &Translation) -> Vec<bool> {
    use verilog::ast::{Span, Stmt, StmtKind};
    let naming: BTreeMap<String, String> = inst.naming().into_iter().collect();
    // the out-of-range reads: instance prefix and position
    let mut unseen: BTreeSet<(String, Span)> = t
        .out_of_range
        .iter()
        .map(|w| {
            let prefix = match w.msg.rsplit_once(" (in instance `") {
                Some((_, p)) => format!("{}.", p.trim_end_matches("`)")),
                None => String::new(),
            };
            (prefix, w.span)
        })
        .collect();
    let oor = unseen.clone();
    // the assignments of a block: targets and reads (with positions)
    type Dep<'a> = (Vec<&'a String>, Vec<(&'a String, Span)>);
    fn deps<'a>(s: &'a Stmt, around: &mut Vec<(&'a String, Span)>, out: &mut Vec<Dep<'a>>) {
        match &s.kind {
            StmtKind::Block(ss) => ss.iter().for_each(|s| deps(s, around, out)),
            StmtKind::If(c, a, b) => {
                let n = around.len();
                c.reads(around);
                deps(a, around, out);
                deps(b, around, out);
                around.truncate(n);
            }
            StmtKind::Case {
                sel,
                items,
                default,
                ..
            } => {
                let n = around.len();
                sel.reads(around);
                for i in items {
                    i.labels.iter().for_each(|l| l.reads(around));
                }
                for i in items {
                    deps(&i.body, around, out);
                }
                if let Some(d) = default {
                    deps(d, around, out);
                }
                around.truncate(n);
            }
            StmtKind::NonBlocking { lhs, rhs } | StmtKind::Blocking { lhs, rhs } => {
                let mut targets = Vec::new();
                lhs.targets(&mut targets);
                let mut reads = around.clone();
                rhs.reads(&mut reads);
                lhs.reads(&mut reads);
                out.push((targets, reads));
            }
            StmtKind::Unknown { .. } => {}
        }
    }
    // every assignment, by name; the targets of the ones with an
    // out-of-range read
    let mut edges: Vec<(Vec<String>, Vec<String>)> = Vec::new();
    let mut x: BTreeSet<String> = BTreeSet::new();
    let mut walk = vec![(inst, String::new())];
    while let Some((i, prefix)) = walk.pop() {
        let canon = |n: &str| naming[&format!("{prefix}{n}")].clone();
        for item in &i.module.items {
            let mut assignments: Vec<Dep> = Vec::new();
            match item {
                Item::Instance(u) => {
                    for (c, _) in u.conns(Dir::Input).filter(|(_, f)| !f.blackbox) {
                        let Some(e) = &c.actual else { continue };
                        let mut reads = Vec::new();
                        e.reads(&mut reads);
                        let port = format!("{}.{}", u.name, c.port);
                        let hit = reads
                            .iter()
                            .any(|(_, s)| oor.contains(&(prefix.clone(), *s)));
                        if hit {
                            x.insert(canon(&port));
                        }
                        let read = reads.iter().map(|(n, _)| canon(n)).collect();
                        edges.push((vec![canon(&port)], read));
                        for (_, s) in reads {
                            unseen.remove(&(prefix.clone(), s));
                        }
                    }
                    continue;
                }
                Item::Assign { lhs, rhs, .. } => {
                    let mut targets = Vec::new();
                    lhs.targets(&mut targets);
                    let mut reads = Vec::new();
                    rhs.reads(&mut reads);
                    lhs.reads(&mut reads);
                    assignments.push((targets, reads));
                }
                Item::AlwaysFf { body, .. } | Item::AlwaysComb { body, .. } => {
                    deps(body, &mut Vec::new(), &mut assignments)
                }
            }
            for (targets, reads) in assignments {
                let targets: Vec<String> = targets.into_iter().map(|q| canon(q)).collect();
                if reads
                    .iter()
                    .any(|(_, s)| oor.contains(&(prefix.clone(), *s)))
                {
                    x.extend(targets.iter().cloned());
                }
                for (_, s) in &reads {
                    unseen.remove(&(prefix.clone(), *s));
                }
                let read = reads.iter().map(|(n, _)| canon(n)).collect();
                edges.push((targets, read));
            }
        }
        for (u, child) in &i.children {
            walk.push((child, format!("{prefix}{u}.")));
        }
    }
    assert!(
        unseen.is_empty(),
        "out-of-range reads in no item: {unseen:?}"
    );
    loop {
        let before = x.len();
        for (targets, read) in &edges {
            if read.iter().any(|r| x.contains(r)) {
                x.extend(targets.iter().cloned());
            }
        }
        if x.len() == before {
            break;
        }
    }
    shown(inst).iter().map(|s| x.contains(s)).collect()
}

/// Compiles and runs a testbench for the design at `path`, and returns
/// its output lines.
fn run_iverilog(bin: &Path, path: &Path, name: &str, tb: &str) -> Vec<String> {
    let tmp = TempDir::new("iverilog");
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
    // `$readmemb`/`$readmemh` files are found next to the design
    let run = Command::new(bin.join("vvp"))
        .args(["-n"])
        .arg(&out)
        .current_dir(path.parent().unwrap())
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
        common::missing("iverilog", "the whole test");
        return;
    };
    let mut skipped = Vec::new();
    for path in examples() {
        let (inst, t) = load(&path);
        let mut rand = Rand(seed(0x9e37_79b9_7f4a_7c15, &stem(&path)));
        if let Some(why) = simulator_differs(&inst) {
            eprintln!("{}: not compared with iverilog: {why}", path.display());
            skipped.push(stem(&path));
            continue;
        }
        let m = &inst.module;
        let init = rand.registers(&inst);
        let clocking = Gen::nth(rand.below(3));
        let rounds = rand.trace(&inst, &t.clocks, clocking, 300);
        let mut interp = Interp::new(&inst, &init).unwrap();
        let expected: Vec<String> = rounds
            .iter()
            .map(|v| hex_line(&inst, &interp.step(v).unwrap()))
            .collect();
        let tb = testbench(&inst, &t, &init, &rounds);
        let got = run_iverilog(&bin, &path, &m.name, &tb);
        assert_eq!(got.len(), expected.len(), "{}: {got:?}", path.display());
        let x = may_be_x(&inst, &t);
        for (k, (g, e)) in got.iter().zip(&expected).enumerate() {
            assert!(
                agrees(g, e, &x),
                "{}: round {k} = {:?}\n{g}\n{e}",
                path.display(),
                rounds[k]
            );
        }
    }
    assert_eq!(skipped, SIMULATOR_DIFFERS);
}

/// The round semantics with a toggling clock is the cycle model: for
/// designs whose blocks all run at the rising edge of one clock, iverilog
/// pulsing the clock once per cycle agrees with the AST semantics at the
/// rounds where the clock is 0. Skipped like [`ast_matches_iverilog`].
#[test]
fn rounds_match_cycles() {
    let Some(bin) = iverilog_bin() else {
        common::missing("iverilog", "the whole test");
        return;
    };
    let mut compared = 0;
    for path in examples() {
        let (inst, t) = load(&path);
        let mut rand = Rand(seed(0x1405_7b7e_f767_814f, &stem(&path)));
        if t.clocks.len() > 1 || negedge(&inst) || simulator_differs(&inst).is_some() {
            continue;
        }
        let m = &inst.module;
        let clk = t.clocks.first().map(String::as_str);
        let init = rand.registers(&inst);
        let vectors: Vec<Values> = (0..200)
            .map(|_| {
                let mut v = rand.inputs(&inst);
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
        let tb = cycle_testbench(&inst, &t, &init, &vectors);
        let got = run_iverilog(&bin, &path, &format!("{}_cycles", m.name), &tb);
        let x = may_be_x(&inst, &t);
        assert_eq!(got.len(), expected.len(), "{}", path.display());
        for (g, e) in got.iter().zip(&expected) {
            assert!(agrees(g, e, &x), "{}\n{g}\n{e}", path.display());
        }
        compared += 1;
    }
    assert!(compared > 20, "{compared}");
}

/// The clock and the asynchronous reset (if the block resets a register)
/// of every register of the hierarchy, by name (see `Inst::naming`).
fn clocking(inst: &Inst) -> BTreeMap<String, (String, Option<String>)> {
    let naming: BTreeMap<String, String> = inst.naming().into_iter().collect();
    let mut out = BTreeMap::new();
    let mut walk = vec![(inst, String::new())];
    while let Some((i, prefix)) = walk.pop() {
        let canon = |x: &str| naming[&format!("{prefix}{x}")].clone();
        for item in &i.module.items {
            if let Item::AlwaysFf {
                clock, reset, body, ..
            } = item
            {
                let reset = reset
                    .as_ref()
                    .filter(|_| !body.reset_values().is_empty())
                    .map(|r| canon(&r.name));
                for q in item.targets() {
                    out.insert(canon(q), (canon(clock), reset.clone()));
                }
            }
        }
        for (u, child) in &i.children {
            walk.push((child, format!("{prefix}{u}.")));
        }
    }
    out
}

/// Why iverilog may disagree with the round semantics on a design, if it
/// may. A simulator runs the blocks of a derived clock after the updates
/// that make its edge, so they read the new values of the registers
/// updated in the same time step (and the new level of their clock),
/// where the round semantics reads the values from before the edge, as
/// hardware meeting its hold times does. Found statically: a block whose
/// clock is not an input of the top module reads a signal that depends on
/// a register of another clock (one of its clock's own chain included).
fn simulator_differs(inst: &Inst) -> Option<String> {
    let naming: BTreeMap<String, String> = inst.naming().into_iter().collect();
    // the clock of every register, what every combinational signal reads,
    // and every block with its clock and reads, by name
    let mut clock: BTreeMap<String, String> = BTreeMap::new();
    let mut comb: Vec<(String, String)> = Vec::new();
    let mut blocks: Vec<(String, String, Vec<String>)> = Vec::new();
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
    let mut deps: BTreeMap<String, BTreeSet<String>> = clock
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
    let inputs: BTreeSet<String> = design_inputs(inst)
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

/// The examples iverilog is expected to disagree with (see
/// [`simulator_differs`]): the round semantics on these is checked by
/// the Yosys miter and the interpreters only.
const SIMULATOR_DIFFERS: [&str; 1] = ["divider"];

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
