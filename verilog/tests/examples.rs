//! The example designs in `examples/`: printed modules against golden
//! files, structure, the AST semantics against the translated module, and
//! the AST semantics against iverilog, on random initial register values
//! and random input traces.
//!
//! `UPDATE_GOLDEN=1 cargo test -p verilog --test examples` rewrites the
//! golden files in `tests/golden/`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use verilog::ast::{Dir, Item, Module, Ty};
use verilog::ast_interp::{Interp, Values};
use verilog::elab::{self, Inst};
use verilog::lower::{Translation, init_name, lower_inst};
use verilog::rm_sim::{Sim, Valuation};
use verilog::{BitVec, typing};

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

/// Inputs and the initial values of the registers are external, outputs
/// interface, all other signals of the hierarchy private; an unread input
/// and the clock are no variables at all. The update of a register atom
/// awaits nothing.
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
                .filter(|p| Some(&p.decl.name) != t.clock.as_ref())
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
            let inits: Vec<String> = ctrl.iter().map(|q| init_name(regs[q])).collect();
            assert_eq!(names(atom.wait()), sorted(inits), "{display}");
            for term in atom.update().iter() {
                for w in term.read() {
                    let awaited = t.vars.values().any(|v| base::var::X(*v) == *w);
                    assert!(!awaited, "{display}: update of {ctrl:?} awaits");
                }
            }
        }
    }
    let (_, t) = load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/arith.sv"));
    assert_eq!(t.unread_inputs, ["clk"]);
    assert_eq!(t.clock, None);
    let (_, t) = load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/accum.sv"));
    assert_eq!(t.clock.as_deref(), Some("clk"));
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

    /// A value for every input but the clock.
    fn inputs(&mut self, m: &Module) -> Values {
        m.ports
            .iter()
            .filter(|p| p.dir == Dir::Input && Some(&p.decl.name) != m.clock())
            .map(|p| (p.decl.name.clone(), self.bits(p.decl.ty.width)))
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

/// The translated module computes the same values as the AST semantics,
/// from the same initial register values (given to `q@init` in the first
/// round, and random values without effect in later rounds).
#[test]
fn rm_matches_ast() {
    let mut rand = Rand(0x2545_f491_4f6c_dd1d);
    for path in examples() {
        let (inst, t) = load(&path);
        let m = &inst.module;
        for _ in 0..10 {
            let init = rand.registers(&inst);
            let mut interp = Interp::new(&inst, &init).unwrap();
            let mut sim = Sim::new(&t.module);
            for k in 0..100 {
                let inputs = rand.inputs(m);
                let expected = interp.step(&inputs).unwrap();
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
                        "{}: `{name}` in cycle {k} from {init:?} with inputs {inputs:?}",
                        path.display()
                    );
                    compared += 1;
                }
                assert_eq!(compared, expected.len() - t.unread_inputs.len());
            }
        }
    }
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

/// A testbench that sets the registers to `init`, then for each of
/// `vectors` sets the inputs, prints all signals but the inputs in hex (one
/// line per vector) and pulses the clock.
///
/// The registers are set by hierarchical assignment (`dut.q = …`) before
/// the first edge; iverilog accepts this although the standard forbids
/// writing an `always_ff` variable from another process. The clock starts
/// at 0, since a change from x to 1 would be a rising edge.
fn testbench(inst: &Inst, init: &Values, vectors: &[Values]) -> String {
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
    if let Some(clk) = m.clock() {
        tb += &format!("    {clk} = 1'b0;\n");
    }
    for (q, value) in init {
        tb += &format!("    dut.{q} = {}'h{:x};\n", value.width(), value.unsigned());
    }
    let shown: Vec<String> = shown(inst).iter().map(|x| format!("dut.{x}")).collect();
    let format = vec!["%h"; shown.len()].join(" ");
    for v in vectors {
        for (name, value) in v {
            tb += &format!("    {name} = {}'h{:x};\n", value.width(), value.unsigned());
        }
        tb += &format!("    #1 $display(\"{format}\", {});\n", shown.join(", "));
        if let Some(clk) = m.clock() {
            tb += &format!("    {clk} = 1'b1;\n    #1 {clk} = 1'b0;\n");
        }
    }
    tb += "    $finish;\n  end\nendmodule\n";
    tb
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

/// The AST semantics agree with iverilog on random vectors. Skipped (with a
/// message) if iverilog is not found in `$IVERILOG_BIN`,
/// `~/tools/oss-cad-suite/bin` or the `PATH`.
#[test]
fn ast_matches_iverilog() {
    let Some(bin) = iverilog_bin() else {
        eprintln!("iverilog not found: skipped");
        return;
    };
    let tmp = Path::new(env!("CARGO_TARGET_TMPDIR")).join("iverilog");
    std::fs::create_dir_all(&tmp).unwrap();
    let mut rand = Rand(0x9e37_79b9_7f4a_7c15);
    for path in examples() {
        let (inst, _) = load(&path);
        let m = &inst.module;
        let init = rand.registers(&inst);
        let vectors: Vec<Values> = (0..300).map(|_| rand.inputs(m)).collect();
        let mut interp = Interp::new(&inst, &init).unwrap();
        let expected: Vec<String> = vectors
            .iter()
            .map(|v| {
                let values = interp.step(v).unwrap();
                shown(&inst)
                    .into_iter()
                    .map(|x| {
                        let v = &values[&x];
                        format!("{:0w$x}", v.unsigned(), w = v.width().div_ceil(4))
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect();

        let tb = tmp.join(format!("tb_{}.sv", m.name));
        let out = tmp.join(format!("tb_{}.vvp", m.name));
        std::fs::write(&tb, testbench(&inst, &init, &vectors)).unwrap();
        let compile = Command::new(bin.join("iverilog"))
            .args(["-g2012", "-o"])
            .args([&out, &tb, &path])
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
        let got: Vec<&str> = stdout.lines().filter(|l| !l.contains("$finish")).collect();
        assert_eq!(got.len(), expected.len(), "{}: {stdout}", path.display());
        for (k, (g, e)) in got.iter().zip(&expected).enumerate() {
            assert_eq!(g, e, "{}: vector {k} = {:?}", path.display(), vectors[k]);
        }
    }
}
