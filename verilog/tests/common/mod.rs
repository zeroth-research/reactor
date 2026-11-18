//! Helpers shared by the integration tests.

use std::io::Write;

/// Whether the environment variable `V2RM_REQUIRE_TOOLS` is `1`: then a
/// test that needs an external tool (Yosys, z3, iverilog) fails when the
/// tool is not found, instead of skipping its work. For CI and for runs
/// whose result must cover the comparisons with the tools.
pub fn tools_required() -> bool {
    std::env::var_os("V2RM_REQUIRE_TOOLS").is_some_and(|v| v == "1")
}

/// Reports that the running test leaves out `what` (all of its work, or a
/// part) because `tool` was not found. With `V2RM_REQUIRE_TOOLS=1` (see
/// [`tools_required`]) it panics instead, so that the test fails.
///
/// The message is written to the standard error stream directly, not with
/// `eprintln!`: the test harness captures `eprintln!` (shown only for
/// failed tests or with `--nocapture`), but not writes to the stream
/// itself, so a skip shows in every run of `cargo test`.
pub fn missing(tool: &str, what: &str) {
    let thread = std::thread::current();
    let test = thread.name().unwrap_or("?");
    if tools_required() {
        panic!("{tool} not found, needed for {what} (V2RM_REQUIRE_TOOLS=1)");
    }
    let _ = writeln!(
        std::io::stderr(),
        "SKIPPED in {test}: {what}: {tool} not found (V2RM_REQUIRE_TOOLS=1 turns this into a failure)"
    );
}
