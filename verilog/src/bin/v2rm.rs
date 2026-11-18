//! `v2rm [--top NAME] [-G NAME=VALUE]… [--ignore-initializers] [--instances | --sized | --smt | --smt-init | --yosys [-t:MS]] FILE.sv…`:
//! translates the top module of a SystemVerilog design (with its
//! instances) and prints the reactive module. A design of several files is
//! read as if the files were one, in the order given (a file of blackbox
//! stubs, say); errors name the file they are in. Warnings (on standard
//! error) name the inputs nothing reads, every instance of a blackbox with
//! the signals it leaves free (see the crate docs, "The boundary of a
//! design"), and reads that may select bits outside a signal. `-G` sets a parameter of the top module
//! to a constant expression. `--instances` first prints the module of every
//! instance as it is composed into its parent, children before parents.
//! `--sized` prints the design instead with every implicit resize written
//! out, as SystemVerilog in the explicit-width fragment (see the `sizing`
//! module of the library). `--smt` prints an update round of the module as
//! an SMT-LIB script instead, `--smt-init` the init round.
//!
//! `--ignore-initializers` drops the initial values of registers
//! (`logic q = c;`, `initial q = c;`, files loaded into written arrays), so
//! that they start with any value, as on hardware that does not keep
//! initial values (ASICs); constants and ROMs keep their values.
//!
//! `--yosys` compares the translation with Yosys's instead (see the
//! `miter` module of the library): for every variable on which the two
//! differ, in the init or in the update round, it prints a counterexample
//! and the replay of it in both interpreters, saying which side the
//! reference semantics agrees with. Exits with failure if a variable
//! differs or the solver gives up. Needs Yosys and z3 (found as in the
//! `yosys` and `z3` modules), and does not take `-G` or
//! `--ignore-initializers` (Yosys keeps initial values). `-t:MS` gives z3
//! at most `MS` milliseconds for each query (no limit without it); a query
//! that runs out of time is reported as `unknown:`.

use std::collections::BTreeMap;
use std::process::ExitCode;
use verilog::ast::{Error, Override, Span};
use verilog::elab::Inst;
use verilog::lower::Translation;
use verilog::miter::{self, Verdict};
use verilog::smt::{self, Phase};
use verilog::{constant, elab, frontend, lower, sizing, yosys, z3};

const USAGE: &str = "usage: v2rm [--top NAME] [-G NAME=VALUE]... [--ignore-initializers] \
                     [--instances | --sized | --smt | --smt-init | --yosys [-t:MS]] FILE.sv...";

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let mut top = None;
    let mut params = Vec::new();
    let mut paths: Vec<String> = Vec::new();
    let mut instances = false;
    let mut sized = false;
    let mut smt = None;
    let mut compare = false;
    let mut ignore = false;
    let mut timeout = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--top" => top = args.next(),
            "--yosys" => compare = true,
            "--ignore-initializers" => ignore = true,
            "--instances" => instances = true,
            "--sized" => sized = true,
            "--smt" => smt = Some(Phase::Update),
            "--smt-init" => smt = Some(Phase::Init),
            "-G" => match args.next() {
                Some(p) => params.push(p),
                None => return usage(),
            },
            _ if a.starts_with("-G") => params.push(a[2..].to_string()),
            _ if a.starts_with("-t:") => match a[3..].parse::<u64>() {
                Ok(ms) if ms > 0 => timeout = Some(std::time::Duration::from_millis(ms)),
                _ => return usage(),
            },
            _ if !a.starts_with('-') => paths.push(a),
            _ => return usage(),
        }
    }
    let Some(path) = paths.first().cloned() else {
        return usage();
    };
    if [instances, sized, smt.is_some(), compare]
        .iter()
        .filter(|b| **b)
        .count()
        > 1
        || (compare && ignore)
        || (timeout.is_some() && !compare)
    {
        return usage();
    }
    // the design, once parsed, to locate errors in their files
    let mut design = None;
    let result = (|| {
        let d = design.insert(frontend::parse_design_files(&paths)?);
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
        let mut inst = elab::elaborate(d, &top, &overrides)?;
        if ignore {
            inst.ignore_initializers();
        }
        let t = lower::lower_inst(&inst)?;
        Ok::<_, Error>((inst, t, overrides))
    })();
    // `FILE:L:C: message`, with the file of the error
    let located = |e: Error| {
        let e = match &design {
            Some(d) => d.located(e),
            None => e,
        };
        let file = e
            .file
            .as_ref()
            .map_or(path.clone(), |f| f.display().to_string());
        format!("{file}:{e}")
    };
    match result {
        Ok((inst, t, overrides)) => {
            for name in &t.unread_inputs {
                eprintln!("{path}: warning: input `{name}` is never read and is left out");
            }
            for w in t.blackboxes.iter().chain(&t.out_of_range) {
                eprintln!(
                    "{}",
                    located(Error::new(w.span, format!("warning: {}", w.msg)))
                );
            }
            if compare {
                return match yosys_check(&paths, &inst, &t, &overrides, timeout) {
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
                        eprintln!("{}", located(e));
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
            eprintln!("{}", located(e));
            ExitCode::FAILURE
        }
    }
}

/// Compares `t`, lowered from `inst`, with Yosys's translation of the
/// files at `paths`, printing every difference with its counterexample and
/// replay, giving z3 at most `timeout` per query. `Ok(true)` if every
/// variable is equal in both rounds.
fn yosys_check(
    paths: &[String],
    inst: &Inst,
    t: &Translation,
    overrides: &[Override],
    timeout: Option<std::time::Duration>,
) -> Result<bool, String> {
    let yosys = yosys::find().ok_or("yosys not found (set $YOSYS_BIN)")?;
    let z3 = z3::find().ok_or("z3 not found (set $Z3_BIN)")?;
    let files: Vec<&std::path::Path> = paths.iter().map(|p| p.as_ref()).collect();
    let ys = yosys::run_files(&yosys, &files, &t.name, overrides)?;
    let mut m = miter::build(t, &ys)?;
    m.timeout = timeout;
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
        "{}: {equal} of {} queries equal (every controlled variable, in the init and \
         the update round)",
        paths.join(" "),
        verdicts.len()
    );
    Ok(equal == verdicts.len())
}

fn usage() -> ExitCode {
    eprintln!("{USAGE}");
    ExitCode::FAILURE
}
