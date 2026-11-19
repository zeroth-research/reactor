//! The command line of `v2rm`: what each option prints, exit statuses and
//! the messages for wrong arguments. (`--yosys` is tested in `yosys.rs`.)

mod common;

use common::TempDir;
use verilog::smt::{self, Phase};
use verilog::{elab, lower};

/// Runs `v2rm` in the crate directory: exit status, standard output and
/// standard error.
fn v2rm(args: &[&str]) -> (i32, String, String) {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_v2rm"))
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .unwrap();
    let text = |b: Vec<u8>| String::from_utf8(b).unwrap();
    (
        out.status.code().unwrap(),
        text(out.stdout),
        text(out.stderr),
    )
}

/// 0 when the module is printed, 2 on an error in the design.
#[test]
fn exit_status() {
    for args in [
        &["examples/counter.sv"][..],
        &["--instances", "examples/nest.sv"],
        &["--sized", "examples/counter.sv"],
        &["--smt", "examples/counter.sv"],
        &["--smt-init", "examples/counter.sv"],
        &["--smt", "--smt", "examples/counter.sv"],
        &["-G", "N=5", "examples/modcount.sv"],
        &["-GN=5", "examples/modcount.sv"],
    ] {
        let (code, stdout, stderr) = v2rm(args);
        assert_eq!(code, 0, "{args:?}: {stderr}");
        assert!(!stdout.is_empty(), "{args:?}");
    }
    let (code, _, stderr) = v2rm(&["no-such-file.sv"]);
    assert_eq!(code, 2, "{stderr}");
    let (code, _, stderr) = v2rm(&["--top", "nothing", "examples/counter.sv"]);
    assert_eq!(code, 2, "{stderr}");
    let (code, _, stderr) = v2rm(&["-G", "N", "examples/modcount.sv"]);
    assert_eq!(code, 2);
    assert!(stderr.contains("-G N: expected NAME=VALUE"), "{stderr}");
    let (code, stdout, _) = v2rm(&["--help"]);
    assert_eq!(code, 0);
    assert!(stdout.starts_with("usage: v2rm "), "{stdout}");
}

/// Wrong arguments: one message per problem, then the usage, exit status 2.
#[test]
fn wrong_arguments() {
    let cases: &[(&[&str], &[&str])] = &[
        (
            &["--top"],
            &["`--top` needs the name of a module", "no input file"],
        ),
        (
            &["--top", "--yosys", "examples/counter.sv"],
            &["`--top` needs the name of a module"],
        ),
        (
            &["--top", "a", "--top", "b", "examples/counter.sv"],
            &["`--top` given twice"],
        ),
        (&["-G"], &["`-G` needs NAME=VALUE", "no input file"]),
        (
            &["--sized", "--yosys", "examples/counter.sv"],
            &["`--sized` and `--yosys` cannot be combined: each chooses what v2rm prints"],
        ),
        (
            &["--smt", "--smt-init", "examples/counter.sv"],
            &["`--smt` and `--smt-init` cannot be combined: each chooses what v2rm prints"],
        ),
        (
            &["--yosys", "--ignore-initializers", "examples/powerup.sv"],
            &["`--ignore-initializers` cannot be used with `--yosys`: Yosys keeps initial values"],
        ),
        (
            &["-t:1000", "examples/counter.sv"],
            &["`-t:1000` limits z3's queries, it needs `--yosys`"],
        ),
        (
            &["--yosys", "-t:0", "examples/counter.sv"],
            &["`-t:0`: the time limit must be a positive number of milliseconds"],
        ),
        (
            &["--yosys", "-t:x", "examples/counter.sv"],
            &["`-t:x`: the time limit must be a positive number of milliseconds"],
        ),
        (&["-x", "examples/counter.sv"], &["unknown option `-x`"]),
        (&[], &["no input file"]),
    ];
    for (args, problems) in cases {
        let (code, stdout, stderr) = v2rm(args);
        assert_eq!(code, 2, "{args:?}: {stderr}");
        assert!(stdout.is_empty(), "{args:?}: {stdout}");
        let mut lines: Vec<&str> = stderr.lines().collect();
        let usage = lines.pop().unwrap();
        assert!(usage.starts_with("usage: v2rm "), "{args:?}: {stderr}");
        let expected: Vec<String> = problems.iter().map(|p| format!("v2rm: {p}")).collect();
        assert_eq!(lines, expected, "{args:?}");
    }
}

/// The translation of the top module of an example, as the library gives
/// it.
fn translation(example: &str) -> lower::Translation {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(example);
    lower::lower_inst(&elab::load_file(path).unwrap()).unwrap()
}

/// The golden file `tests/golden/{name}`.
fn golden(name: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(name);
    std::fs::read_to_string(path).unwrap()
}

/// What each mode prints: the module (as in the golden files), the sized
/// design, the SMT-LIB rounds, the instances before the composed module.
#[test]
fn printed_forms() {
    let ok = |args: &[&str]| {
        let (code, stdout, stderr) = v2rm(args);
        assert_eq!(code, 0, "{args:?}: {stderr}");
        assert!(stderr.is_empty(), "{args:?}: {stderr}");
        stdout
    };
    assert_eq!(ok(&["examples/counter.sv"]), golden("counter.rm"));
    assert_eq!(
        ok(&["--sized", "examples/counter.sv"]),
        golden("counter.sized.sv")
    );
    let t = translation("examples/counter.sv");
    for (flag, phase) in [("--smt", Phase::Update), ("--smt-init", Phase::Init)] {
        assert_eq!(
            ok(&[flag, "examples/counter.sv"]),
            format!("(set-logic QF_BV)\n{}", smt::round(&t, phase).text),
            "{flag}"
        );
    }
    // every instance as composed into its parent, children first, then
    // the module itself
    let printed = ok(&["--instances", "examples/nest.sv"]);
    let headers: Vec<&str> = printed
        .lines()
        .filter(|l| l.starts_with("module "))
        .collect();
    assert_eq!(
        headers,
        [
            "module leaf #(.K(4'd2)) m1.l",
            "module mid #(.K(4'd2)) m1",
            "module leaf #(.K(4'd7)) m2.l",
            "module mid #(.K(4'd7)) m2",
            "module nest",
        ]
    );
    assert!(printed.ends_with(&golden("nest.rm")), "{printed}");
}

/// `--top` chooses the module to translate, with its instances; without
/// it, the one module no other instantiates.
#[test]
fn top_module() {
    let (code, stdout, _) = v2rm(&["--top", "leaf", "examples/nest.sv"]);
    assert_eq!(code, 0);
    assert!(stdout.starts_with("module leaf\n"), "{stdout}");
    // the default parameter values of `mid` (K = 2) reach its leaf
    let (code, stdout, _) = v2rm(&["--top", "mid", "--instances", "examples/nest.sv"]);
    assert_eq!(code, 0);
    assert!(
        stdout.starts_with("module leaf #(.K(4'd2)) l\n"),
        "{stdout}"
    );
    let (_, stdout, _) = v2rm(&["examples/nest.sv"]);
    assert!(stdout.starts_with("module nest\n"), "{stdout}");
    let (code, stdout, stderr) = v2rm(&["--top", "nothing", "examples/nest.sv"]);
    assert_eq!(code, 2);
    assert!(stdout.is_empty());
    assert!(stderr.contains("no module `nothing`"), "{stderr}");
}

/// `-G NAME=VALUE` (also written `-GNAME=VALUE`) sets a parameter of the
/// top module to a literal; a bad value or name is an error (status 2).
#[test]
fn parameters() {
    let default = v2rm(&["examples/modcount.sv"]).1;
    assert!(default.contains("    q : bv4\n"), "{default}");
    let (code, five, _) = v2rm(&["-G", "N=5", "examples/modcount.sv"]);
    assert_eq!(code, 0);
    assert!(five.contains("    q : bv3\n"), "{five}");
    assert!(five.contains("%1 := 3'd4"), "{five}");
    assert_eq!(v2rm(&["-GN=5", "examples/modcount.sv"]).1, five);
    assert_eq!(v2rm(&["-G", "N=32'sd5", "examples/modcount.sv"]).1, five);
    assert_eq!(v2rm(&["-G", "N=0_5", "examples/modcount.sv"]).1, five);
    for (value, message) in [
        ("N=abc", "-G N=abc: malformed literal `abc`"),
        ("N=", "-G N=: malformed literal ``"),
        ("N=4x'h3", "-G N=4x'h3: malformed literal `4x'h3`"),
        (
            "N=99999999999",
            "-G N=99999999999: integer `99999999999` too large",
        ),
        (
            "N=-8'd3",
            "-G N=-8'd3: `-8'd3`: only an unsized decimal can have a sign",
        ),
        (
            "N=4'hz",
            "-G N=4'hz: literal `4'hz` with x/z digits not supported",
        ),
        ("N", "-G N: expected NAME=VALUE"),
        ("=5", "-G =5: expected NAME=VALUE"),
        ("X=1", "`modcount` has no parameter `X`"),
        ("W=3", "parameter `W` of `modcount` is local"),
        // a value the module itself rejects
        ("N=-3", "`$clog2` of the negative value -3"),
    ] {
        let (code, stdout, stderr) = v2rm(&["-G", value, "examples/modcount.sv"]);
        assert_eq!(code, 2, "{value}: {stderr}");
        assert!(stdout.is_empty(), "{value}: {stdout}");
        assert!(
            stderr.starts_with("examples/modcount.sv:"),
            "{value}: {stderr}"
        );
        assert!(stderr.contains(message), "{value}: {stderr}");
        assert_eq!(stderr.lines().count(), 1, "{value}: {stderr}");
    }
}

/// Options that do not exist (also misspelt ones and `--opt=value`) are
/// errors, whatever else is given, and nothing is translated.
#[test]
fn unknown_options() {
    for flag in [
        "--frobnicate",
        "-y",
        "--yosy",
        "--top=counter",
        "--SIZED",
        "-",
    ] {
        let (code, stdout, stderr) = v2rm(&[flag, "examples/counter.sv"]);
        assert_eq!(code, 2, "{flag}: {stderr}");
        assert!(stdout.is_empty(), "{flag}");
        assert_eq!(
            stderr.lines().next(),
            Some(format!("v2rm: unknown option `{flag}`").as_str()),
            "{stderr}"
        );
    }
}

/// Warnings go to standard error and leave the status 0; errors in the
/// design give `FILE:LINE:COL: message` and status 2, with nothing on
/// standard output.
#[test]
fn warnings_and_errors() {
    let (code, stdout, stderr) = v2rm(&["examples/arith.sv"]);
    assert_eq!(code, 0);
    assert!(!stdout.is_empty());
    assert_eq!(
        stderr,
        "examples/arith.sv: warning: input `clk` is never read and is left out\n"
    );
    let (code, _, stderr) = v2rm(&["examples/oorread.sv"]);
    assert_eq!(code, 0);
    assert!(
        stderr.starts_with("examples/oorread.sv:12:14: warning: `v[(15 - c)]` may select bits"),
        "{stderr}"
    );
    let tmp = TempDir::new("cli");
    let dir = tmp.join("cli");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("bad.sv");
    std::fs::write(
        &path,
        "module bad(input logic a, output logic y);\n  assign y = a + b;\nendmodule\n",
    )
    .unwrap();
    let path = path.to_str().unwrap();
    for mode in ["--instances", "--sized", "--smt", "--smt-init"] {
        let (code, stdout, stderr) = v2rm(&[mode, path]);
        assert_eq!(code, 2, "{mode}");
        assert!(stdout.is_empty(), "{mode}: {stdout}");
        assert!(
            stderr.starts_with(&format!("{path}:2:18: ")),
            "{mode}: {stderr}"
        );
    }
}
