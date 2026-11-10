//! `v2rm [--top NAME] [-G NAME=VALUE]… [--instances | --sized | --smt | --smt-init | --yosys] FILE.sv`:
//! translates the top module of a SystemVerilog file (with its instances)
//! and prints the reactive module. `-G` sets a parameter of the top module
//! to a constant expression. `--instances` first prints the module of every
//! instance as it is composed into its parent, children before parents.
//! `--sized` prints the design instead with every implicit resize written
//! out, as SystemVerilog in the explicit-width fragment (see the `sizing`
//! module of the library). `--smt` prints an update round of the module as
//! an SMT-LIB script instead, `--smt-init` the init round.
//!
//! `--yosys` compares the translation with Yosys's instead (see the
//! `miter` module of the library): for every variable on which the two
//! differ, in the init or in the update round, it prints a counterexample
//! and the replay of it in both interpreters, saying which side the
//! reference semantics agrees with. Exits with failure if a variable
//! differs or the solver gives up. Needs Yosys and z3 (found as in the
//! `yosys` and `z3` modules), and does not take `-G` yet.

use std::collections::BTreeMap;
use std::process::ExitCode;
use verilog::ast::{Error, Override, Span};
use verilog::elab::Inst;
use verilog::lower::Translation;
use verilog::miter::{self, Verdict};
use verilog::smt::{self, Phase};
use verilog::{constant, elab, frontend, lower, sizing, yosys, z3};

const USAGE: &str = "usage: v2rm [--top NAME] [-G NAME=VALUE]... \
                     [--instances | --sized | --smt | --smt-init | --yosys] FILE.sv";

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let mut top = None;
    let mut params = Vec::new();
    let mut path = None;
    let mut instances = false;
    let mut sized = false;
    let mut smt = None;
    let mut compare = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--top" => top = args.next(),
            "--yosys" => compare = true,
            "--instances" => instances = true,
            "--sized" => sized = true,
            "--smt" => smt = Some(Phase::Update),
            "--smt-init" => smt = Some(Phase::Init),
            "-G" => match args.next() {
                Some(p) => params.push(p),
                None => return usage(),
            },
            _ if a.starts_with("-G") => params.push(a[2..].to_string()),
            _ if path.is_none() && !a.starts_with('-') => path = Some(a),
            _ => return usage(),
        }
    }
    let Some(path) = path else {
        return usage();
    };
    if [instances, sized, smt.is_some(), compare]
        .iter()
        .filter(|b| **b)
        .count()
        > 1
    {
        return usage();
    }
    let result = (|| {
        let d = frontend::parse_design_file(&path)?;
        let top = match top {
            Some(t) => t,
            None => d.top()?,
        };
        let mut overrides = Vec::new();
        for p in &params {
            let err = |msg: String| Error::new(Span::default(), format!("-G {p}: {msg}"));
            let (name, value) = p
                .split_once('=')
                .ok_or_else(|| err("expected NAME=VALUE".into()))?;
            let value = constant::eval(value, &BTreeMap::new()).map_err(|f| match f {
                constant::Failure::NotConstant(m) | constant::Failure::Error(m) => err(m),
            })?;
            overrides.push(Override {
                name: name.to_string(),
                value,
                span: Span::default(),
            });
        }
        let inst = elab::elaborate(&d, &top, &overrides)?;
        let t = lower::lower_inst(&inst)?;
        Ok::<_, Error>((inst, t, overrides))
    })();
    match result {
        Ok((inst, t, overrides)) => {
            for name in &t.unread_inputs {
                eprintln!("{path}: warning: input `{name}` is never read and is left out");
            }
            if compare {
                return match yosys_check(&path, &inst, &t, &overrides) {
                    Ok(true) => ExitCode::SUCCESS,
                    Ok(false) => ExitCode::FAILURE,
                    Err(e) => {
                        eprintln!("{path}: {e}");
                        ExitCode::FAILURE
                    }
                };
            }
            if sized {
                return match sizing::print(&inst) {
                    Ok(text) => {
                        print!("{text}");
                        ExitCode::SUCCESS
                    }
                    Err(e) => {
                        eprintln!("{path}:{e}");
                        ExitCode::FAILURE
                    }
                };
            }
            if let Some(phase) = smt {
                print!("(set-logic QF_BV)\n{}", smt::round(&t, phase).text);
                return ExitCode::SUCCESS;
            }
            if instances {
                for part in &t.instances {
                    println!("{}", t.print_part(part));
                }
            }
            print!("{}", t.print());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{path}:{e}");
            ExitCode::FAILURE
        }
    }
}

/// Compares `t`, lowered from `inst`, with Yosys's translation of the file
/// at `path`, printing every difference with its counterexample and replay.
/// `Ok(true)` if every variable is equal in both rounds.
fn yosys_check(
    path: &str,
    inst: &Inst,
    t: &Translation,
    overrides: &[Override],
) -> Result<bool, String> {
    let yosys = yosys::find().ok_or("yosys not found (set $YOSYS_BIN)")?;
    let z3 = z3::find().ok_or("z3 not found (set $Z3_BIN)")?;
    let ys = yosys::run(&yosys, path.as_ref(), &t.name, overrides)?;
    let m = miter::build(t, &ys)?;
    let verdicts = miter::check(&z3, &m)?;
    let round = |phase| match phase {
        Phase::Init => "init",
        Phase::Update => "update",
    };
    let mut equal = 0;
    for (phase, name, verdict) in &verdicts {
        match verdict {
            Verdict::Equal => equal += 1,
            Verdict::Unknown => println!("unknown: `{name}` in the {} round", round(*phase)),
            Verdict::Differs => {
                let q = m
                    .queries
                    .iter()
                    .find(|q| q.phase == *phase && &q.name == name);
                let cex = miter::counterexample(&z3, &m, q.expect("queried"))?;
                let replay = miter::replay(inst, t, &cex)?;
                println!("differs: `{name}` in the {} round", round(*phase));
                println!("  {cex}\n  {replay}");
            }
        }
    }
    println!(
        "{path}: {equal} of {} queries equal (every controlled variable, in the init and \
         the update round)",
        verdicts.len()
    );
    Ok(equal == verdicts.len())
}

fn usage() -> ExitCode {
    eprintln!("{USAGE}");
    ExitCode::FAILURE
}
