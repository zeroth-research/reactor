//! The command line of `v2rm`: exit statuses and the messages for wrong
//! arguments. (`--yosys` is tested in `yosys.rs`.)

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
