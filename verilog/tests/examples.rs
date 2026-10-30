//! The example designs in `examples/`: printed modules against golden
//! files, structure, the AST semantics against the translated module, and
//! the AST semantics against iverilog.
//!
//! `UPDATE_GOLDEN=1 cargo test -p verilog --test examples` rewrites the
//! golden files in `tests/golden/`.

use std::path::{Path, PathBuf};
use std::process::Command;
use verilog::ast::{Dir, Module};
use verilog::ast_interp::{Interp, Values};
use verilog::lower::{Translation, lower};
use verilog::rm_sim::{Sim, Valuation};
use verilog::{BitVec, frontend};

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

fn load(path: &Path) -> (Module, Translation) {
    let m = frontend::parse_file(path).unwrap_or_else(|e| panic!("{}:{e}", path.display()));
    let t = lower(&m).unwrap_or_else(|e| panic!("{}:{e}", path.display()));
    (m, t)
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

/// Inputs are external, outputs interface, internal signals private; an
/// unread input is no variable at all.
#[test]
fn structure() {
    for path in examples() {
        let (m, t) = load(&path);
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
                .collect()
        };
        assert_eq!(
            names(t.module.extl()),
            sorted(ports(Dir::Input)),
            "{}",
            path.display()
        );
        assert_eq!(
            names(t.module.intf()),
            sorted(ports(Dir::Output)),
            "{}",
            path.display()
        );
        let decls = m.decls.iter().map(|d| d.name.clone()).collect();
        assert_eq!(names(t.module.prvt()), sorted(decls), "{}", path.display());
        assert_eq!(t.module.atoms().len(), m.items.len());
    }
    let (_, t) = load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/arith.sv"));
    assert_eq!(t.unread_inputs, ["clk"]);
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

    fn inputs(&mut self, m: &Module) -> Values {
        m.ports
            .iter()
            .filter(|p| p.dir == Dir::Input)
            .map(|p| (p.decl.name.clone(), self.bits(p.decl.ty.width)))
            .collect()
    }
}

/// The translated module computes the same values as the AST semantics.
#[test]
fn rm_matches_ast() {
    let mut rand = Rand(0x2545_f491_4f6c_dd1d);
    for path in examples() {
        let (m, t) = load(&path);
        let mut interp = Interp::new(&m).unwrap();
        let mut sim = Sim::new(&t.module);
        for k in 0..500 {
            let inputs = rand.inputs(&m);
            let expected = interp.step(&inputs).unwrap();
            let extl: Valuation = inputs
                .iter()
                .map(|(n, v)| (t.var(n), v.clone()))
                .filter(|(var, _)| t.module.extl().contains(var))
                .collect();
            let got = sim.step(&extl).unwrap();
            for (var, value) in &got {
                let name = t.name(var);
                assert_eq!(
                    value,
                    &expected[name],
                    "{}: `{name}` in cycle {k} with inputs {inputs:?}",
                    path.display()
                );
            }
            assert_eq!(got.len(), expected.len() - t.unread_inputs.len());
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

/// A testbench driving the module with `vectors` and printing the outputs
/// in hex after each, one line per vector.
fn testbench(m: &Module, vectors: &[Values]) -> String {
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
    let outputs: Vec<&str> = m
        .ports
        .iter()
        .filter(|p| p.dir == Dir::Output)
        .map(|p| p.decl.name.as_str())
        .collect();
    let format = vec!["%h"; outputs.len()].join(" ");
    for v in vectors {
        for (name, value) in v {
            tb += &format!("    {name} = {}'h{:x};\n", value.width(), value.unsigned());
        }
        tb += &format!("    #1 $display(\"{format}\", {});\n", outputs.join(", "));
    }
    tb += "    $finish;\n  end\nendmodule\n";
    tb
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
        let (m, _) = load(&path);
        let vectors: Vec<Values> = (0..300).map(|_| rand.inputs(&m)).collect();
        let mut interp = Interp::new(&m).unwrap();
        let expected: Vec<String> = vectors
            .iter()
            .map(|v| {
                let values = interp.step(v).unwrap();
                m.ports
                    .iter()
                    .filter(|p| p.dir == Dir::Output)
                    .map(|p| {
                        let v = &values[&p.decl.name];
                        format!("{:0w$x}", v.unsigned(), w = v.width().div_ceil(4))
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect();

        let tb = tmp.join(format!("tb_{}.sv", m.name));
        let out = tmp.join(format!("tb_{}.vvp", m.name));
        std::fs::write(&tb, testbench(&m, &vectors)).unwrap();
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
