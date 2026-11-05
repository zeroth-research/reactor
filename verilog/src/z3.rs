/*!
# Running z3

Finds the z3 binary and runs it on an SMT-LIB script given as text. z3 is
looked up as `$Z3_BIN` (the binary itself), then
`~/tools/oss-cad-suite/bin/z3`, then `z3` on the `PATH`.
*/

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// The z3 binary, or `None` if there is none.
pub fn find() -> Option<PathBuf> {
    let candidates = [
        std::env::var_os("Z3_BIN").map(PathBuf::from),
        std::env::var_os("HOME").map(|h| Path::new(&h).join("tools/oss-cad-suite/bin/z3")),
    ];
    let found = candidates.into_iter().flatten().find(|p| p.exists());
    found.or_else(|| {
        let ok = Command::new("z3").arg("--version").output().is_ok();
        ok.then(|| PathBuf::from("z3"))
    })
}

/// Runs the z3 binary `bin` on `script` and returns its output lines
/// (`sat`, `unsat`, `unknown`, errors, values, …). Fails if z3 cannot be
/// started.
pub fn run(bin: &Path, script: &str) -> Result<Vec<String>, String> {
    let fail = |e: std::io::Error| format!("{}: {e}", bin.display());
    let mut child = Command::new(bin)
        .args(["-smt2", "-in"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(fail)?;
    let mut stdin = child.stdin.take().expect("piped");
    // a writer thread, so that z3 can write while it reads a long script
    let script = script.to_string();
    let writer = std::thread::spawn(move || stdin.write_all(script.as_bytes()));
    let out = child.wait_with_output().map_err(fail)?;
    writer.join().expect("writer").map_err(fail)?;
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::to_string)
        .collect())
}
