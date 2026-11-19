/*!
# Running z3

Finds the z3 binary and runs it on an SMT-LIB script given as text. z3 is
looked up as `$Z3_BIN` (the binary itself), then
`~/tools/oss-cad-suite/bin/z3`, then `z3` on the `PATH`.
*/

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

/// The z3 binary, or `None` if there is none.
pub fn find() -> Option<PathBuf> {
    crate::find_tool("Z3_BIN", "z3", "--version")
}

/// Runs the z3 binary `bin` on `script` and returns its output lines
/// (`sat`, `unsat`, `unknown`, errors, values, …), with no time limit.
/// See [`run_with`].
pub fn run(bin: &Path, script: &str) -> Result<Vec<String>, String> {
    run_with(bin, script, None)
}

/// Runs the z3 binary `bin` on `script` and returns its output lines.
/// With a `timeout`, z3 gives up on every single `check-sat` that takes
/// longer (z3's `-t:` option, in milliseconds) and answers `unknown` for
/// it; the other `check-sat`s of the script still run.
///
/// Fails if z3 cannot be started, if it stops before it has read the
/// whole script, or if a signal ends it; the message then gives z3's exit
/// status and what it wrote to standard error. A script with errors is no
/// failure: z3 reports them in its output and goes on.
pub fn run_with(
    bin: &Path,
    script: &str,
    timeout: Option<Duration>,
) -> Result<Vec<String>, String> {
    let fail = |e: std::io::Error| format!("{}: {e}", bin.display());
    let mut cmd = Command::new(bin);
    cmd.args(["-smt2", "-in"]);
    if let Some(t) = timeout {
        // z3 reads `-t:0` as no limit
        cmd.arg(format!("-t:{}", t.as_millis().max(1)));
    }
    let mut child = cmd
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(fail)?;
    let mut stdin = child.stdin.take().expect("piped");
    // a writer thread, so that z3 can write while it reads a long script
    let script = script.to_string();
    let writer = std::thread::spawn(move || stdin.write_all(script.as_bytes()));
    let out = child.wait_with_output().map_err(fail)?;
    let written = writer.join().expect("writer");
    let lines: Vec<String> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::to_string)
        .collect();
    let stopped = |why: String| {
        let stderr = String::from_utf8_lossy(&out.stderr);
        let mut msg = format!("{}: {why} ({})", bin.display(), out.status);
        for (what, text) in [
            ("standard error", stderr.trim()),
            ("output", &lines.join("\n")),
        ] {
            if !text.is_empty() {
                msg += &format!("\n{what}:\n{text}");
            }
        }
        msg
    };
    if let Err(e) = written {
        return Err(stopped(format!(
            "z3 stopped before reading the whole script: {e}"
        )));
    }
    if out.status.code().is_none() {
        return Err(stopped("z3 was ended by a signal".into()));
    }
    Ok(lines)
}
