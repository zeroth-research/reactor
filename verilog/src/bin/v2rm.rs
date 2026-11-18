//! `v2rm [--top NAME] [-G NAME=VALUE]… [--ignore-initializers] [--instances | --sized | --smt | --smt-init | --yosys [-t:MS]] FILE.sv…`:
//! translates the top module of a SystemVerilog design (with its
//! instances) and prints the reactive module. A design of several files is
//! read as if the files were one, in the order given (a file of blackbox
//! stubs, say); errors name the file they are in. Warnings (on standard
//! error) name the inputs nothing reads, every instance of a blackbox with
//! the signals it leaves free (see the crate docs, "The boundary of a
//! design"), and reads that may select bits outside a signal. `-G` sets a parameter of the top module
//! to a literal (`8`, `-3`, `4'hf`), taken as the parameter's declaration
//! takes a value. `--instances` first prints the module of every
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
//! reference semantics agrees with. Needs Yosys and z3 (found as in the
//! `yosys` and `z3` modules), and does not take `-G` or
//! `--ignore-initializers` (Yosys keeps initial values). It warns if Yosys
//! is not the version the format of its output was checked with. `-t:MS` gives z3
//! at most `MS` milliseconds for each query (no limit without it); a query
//! that runs out of time is reported as `unknown:`. A counterexample or
//! replay that cannot be made is reported under its query, and the other
//! queries are still reported.
//!
//! Exit status: 0 when the module was printed, or under `--yosys` when
//! every query is equal; 1 under `--yosys` when a variable differs or z3
//! gave up on a query; 2 on an error in the design, in running Yosys or
//! z3, or in the arguments (each problem with the arguments gets a message
//! of its own, then the usage). `-h`/`--help` prints the usage.

use std::process::ExitCode;
use verilog::ast::{Error, Override, Span};
use verilog::elab::Inst;
use verilog::lower::Translation;
use verilog::miter::{self, Verdict};
use verilog::smt::{self, Phase};
use verilog::{constant, elab, frontend, lower, sizing, yosys, z3};

const USAGE: &str = "usage: v2rm [--top NAME] [-G NAME=VALUE]... [--ignore-initializers] \
                     [--instances | --sized | --smt | --smt-init | --yosys [-t:MS]] FILE.sv...";

/// Exit status: the translation was printed, or every query of `--yosys`
/// is equal.
const EQUAL: u8 = 0;
/// Exit status of `--yosys` when a variable differs or z3 gave up on a query.
const DIFFERS: u8 = 1;
/// Exit status on an error in the design or in running the tools, and on
/// wrong arguments.
const ERROR: u8 = 2;

/// What `v2rm` prints.
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Print,
    Instances,
    Sized,
    Smt(Phase),
    Yosys,
}

impl Mode {
    fn flag(self) -> &'static str {
        match self {
            Mode::Print => "",
            Mode::Instances => "--instances",
            Mode::Sized => "--sized",
            Mode::Smt(Phase::Update) => "--smt",
            Mode::Smt(Phase::Init) => "--smt-init",
            Mode::Yosys => "--yosys",
        }
    }
}

/// The command line, checked.
struct Options {
    top: Option<String>,
    params: Vec<String>,
    paths: Vec<String>,
    mode: Mode,
    ignore: bool,
    timeout: Option<std::time::Duration>,
}

/// Reads the arguments (without the program name), or every problem with
/// them, one message each. `Ok(None)` for `--help`.
fn options(args: impl IntoIterator<Item = String>) -> Result<Option<Options>, Vec<String>> {
    let mut args = args.into_iter();
    let mut problems = Vec::new();
    let mut top: Option<String> = None;
    let mut params = Vec::new();
    let mut paths = Vec::new();
    let mut mode = Mode::Print;
    let mut ignore = false;
    let mut timeout = None;
    let mut time_flag = None;
    // an option's value: the next argument, unless it is missing or an option
    let value =
        |args: &mut dyn Iterator<Item = String>| args.next().filter(|v| !v.starts_with('-'));
    while let Some(a) = args.next() {
        let chosen = match a.as_str() {
            "--instances" => Some(Mode::Instances),
            "--sized" => Some(Mode::Sized),
            "--smt" => Some(Mode::Smt(Phase::Update)),
            "--smt-init" => Some(Mode::Smt(Phase::Init)),
            "--yosys" => Some(Mode::Yosys),
            _ => None,
        };
        if let Some(m) = chosen {
            if mode != Mode::Print && mode != m {
                problems.push(format!(
                    "`{}` and `{}` cannot be combined: each chooses what v2rm prints",
                    mode.flag(),
                    m.flag()
                ));
            } else {
                mode = m;
            }
            continue;
        }
        match a.as_str() {
            "-h" | "--help" => return Ok(None),
            "--top" => match value(&mut args) {
                Some(_) if top.is_some() => problems.push("`--top` given twice".into()),
                Some(t) => top = Some(t),
                None => problems.push("`--top` needs the name of a module".into()),
            },
            "--ignore-initializers" => ignore = true,
            "-G" => match value(&mut args) {
                Some(p) => params.push(p),
                None => problems.push("`-G` needs NAME=VALUE".into()),
            },
            _ if a.starts_with("-G") => params.push(a[2..].to_string()),
            _ if a.starts_with("-t:") => match a[3..].parse::<u64>() {
                Ok(ms) if ms > 0 => {
                    timeout = Some(std::time::Duration::from_millis(ms));
                    time_flag = Some(a);
                }
                _ => problems.push(format!(
                    "`{a}`: the time limit must be a positive number of milliseconds"
                )),
            },
            _ if !a.starts_with('-') => paths.push(a),
            _ => problems.push(format!("unknown option `{a}`")),
        }
    }
    if paths.is_empty() {
        problems.push("no input file".into());
    }
    if mode == Mode::Yosys && ignore {
        problems.push(
            "`--ignore-initializers` cannot be used with `--yosys`: Yosys keeps initial values"
                .into(),
        );
    }
    if let Some(t) = time_flag.filter(|_| mode != Mode::Yosys) {
        problems.push(format!("`{t}` limits z3's queries, it needs `--yosys`"));
    }
    if problems.is_empty() {
        Ok(Some(Options {
            top,
            params,
            paths,
            mode,
            ignore,
            timeout,
        }))
    } else {
        Err(problems)
    }
}

fn main() -> ExitCode {
    let Options {
        top,
        params,
        paths,
        mode,
        ignore,
        timeout,
    } = match options(std::env::args().skip(1)) {
        Ok(Some(o)) => o,
        Ok(None) => {
            println!("{USAGE}");
            return ExitCode::from(EQUAL);
        }
        Err(problems) => {
            for p in problems {
                eprintln!("v2rm: {p}");
            }
            eprintln!("{USAGE}");
            return ExitCode::from(ERROR);
        }
    };
    let path = paths[0].clone();
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
            let value = constant::parse(value).map_err(err)?;
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
            let code = match mode {
                Mode::Yosys => match yosys_check(&paths, &inst, &t, &overrides, timeout) {
                    Ok(true) => EQUAL,
                    Ok(false) => DIFFERS,
                    Err(e) => {
                        eprintln!("{path}: {e}");
                        ERROR
                    }
                },
                Mode::Sized => match sizing::print(&inst) {
                    Ok(text) => {
                        print!("{text}");
                        EQUAL
                    }
                    Err(e) => {
                        eprintln!("{}", located(e));
                        ERROR
                    }
                },
                Mode::Smt(phase) => {
                    print!("(set-logic QF_BV)\n{}", smt::round(&t, phase).text);
                    EQUAL
                }
                Mode::Instances | Mode::Print => {
                    if mode == Mode::Instances {
                        for part in &t.instances {
                            println!("{}", t.print_part(part));
                        }
                    }
                    print!("{}", t.print());
                    EQUAL
                }
            };
            ExitCode::from(code)
        }
        Err(e) => {
            eprintln!("{}", located(e));
            ExitCode::from(ERROR)
        }
    }
}

/// Compares `t`, lowered from `inst`, with Yosys's translation of the
/// files at `paths`, printing every difference with its counterexample and
/// replay, giving z3 at most `timeout` per query. `Ok(true)` if every
/// variable is equal in both rounds. A counterexample or replay that fails
/// is reported in the block of its query, and the other queries are still
/// reported; it does not change the result.
fn yosys_check(
    paths: &[String],
    inst: &Inst,
    t: &Translation,
    overrides: &[Override],
    timeout: Option<std::time::Duration>,
) -> Result<bool, String> {
    let yosys = yosys::find().ok_or("yosys not found (set $YOSYS_BIN)")?;
    if let Some(w) = yosys::version_warning(&yosys::version(&yosys)) {
        eprintln!("{}: warning: {w}", paths[0]);
    }
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
    // a message of several lines, indented under its query
    let indent = |e: String| e.replace('\n', "\n    ");
    let mut equal = 0;
    for (phase, name, verdict) in &verdicts {
        match verdict {
            Verdict::Equal => equal += 1,
            Verdict::Unknown => println!("unknown: `{name}` in the {} round", round(*phase)),
            Verdict::Differs => {
                println!("differs: `{name}` in the {} round", round(*phase));
                let cex = m
                    .queries
                    .iter()
                    .find(|q| q.phase == *phase && &q.name == name)
                    .ok_or_else(|| format!("the miter has no query of `{name}`"))
                    .and_then(|q| miter::counterexample(&z3, &m, q));
                match cex {
                    Err(e) => println!("  no counterexample: {}", indent(e)),
                    Ok(cex) => {
                        println!("  {cex}");
                        match miter::replay(inst, t, &cex) {
                            Ok(replay) => println!("  {replay}"),
                            Err(e) => println!("  no replay: {}", indent(e)),
                        }
                    }
                }
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
