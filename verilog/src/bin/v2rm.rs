//! `v2rm [--top NAME] [-G NAME=VALUE]… [--instances | --smt | --smt-init] FILE.sv`:
//! translates the top module of a SystemVerilog file (with its instances)
//! and prints the reactive module. `-G` sets a parameter of the top module
//! to a constant expression. `--instances` first prints the module of every
//! instance as it is composed into its parent, children before parents.
//! `--smt` prints an update round of the module as an SMT-LIB script
//! instead, `--smt-init` the init round.

use std::collections::BTreeMap;
use std::process::ExitCode;
use verilog::ast::{Error, Override, Span};
use verilog::smt::{self, Phase};
use verilog::{constant, elab, frontend, lower};

const USAGE: &str =
    "usage: v2rm [--top NAME] [-G NAME=VALUE]... [--instances | --smt | --smt-init] FILE.sv";

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let mut top = None;
    let mut params = Vec::new();
    let mut path = None;
    let mut instances = false;
    let mut smt = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--top" => top = args.next(),
            "--instances" => instances = true,
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
    if instances && smt.is_some() {
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
        lower::lower_inst(&inst)
    })();
    match result {
        Ok(t) => {
            for name in &t.unread_inputs {
                eprintln!("{path}: warning: input `{name}` is never read and is left out");
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

fn usage() -> ExitCode {
    eprintln!("{USAGE}");
    ExitCode::FAILURE
}
