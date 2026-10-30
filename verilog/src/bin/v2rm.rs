//! `v2rm FILE.sv`: translates a SystemVerilog module and prints the
//! reactive module.

use std::process::ExitCode;
use verilog::{frontend, lower};

fn main() -> ExitCode {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: v2rm FILE.sv");
        return ExitCode::FAILURE;
    };
    let result = frontend::parse_file(&path).and_then(|m| lower::lower(&m));
    match result {
        Ok(t) => {
            for name in &t.unread_inputs {
                eprintln!("{path}: warning: input `{name}` is never read and is left out");
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
