//! Blackboxes (modules marked `(* blackbox *)`), `inout` pins and designs
//! of several files.

use verilog::ast::Error;
use verilog::elab;
use verilog::lower::{Translation, lower_inst};

fn translate(src: &str) -> Result<Translation, Error> {
    let inst = elab::load(src, "test.sv")?;
    lower_inst(&inst)
}

/// Stubs used by the cases below.
const STUBS: &str = "
(* blackbox *)
module gen (input logic i, output logic o, output logic [7:0] w);
endmodule
(* blackbox *)
module pad #(parameter [5:0] KIND = 6'd0) (inout pin, input logic d, output logic q);
endmodule
";

/// A top module `top` with `body` (numbered from line 2), followed by the
/// stubs.
fn top(body: &str) -> String {
    format!(
        "module top(input logic clk, input logic [3:0] a, inout p, output logic [3:0] y);\n\
         {body}\nendmodule\n{STUBS}"
    )
}

/// Expects an error at the first occurrence of `needle` in line `line` of
/// `src`, with a message containing `msg`.
fn rejects(src: &str, line: usize, needle: &str, msg: &str) {
    let err = match translate(src) {
        Ok(_) => panic!("accepted:\n{src}"),
        Err(e) => e,
    };
    assert!(err.msg.contains(msg), "{src}\n{err}\nexpected: {msg}");
    let text = src.lines().nth(line - 1).unwrap();
    let col = text
        .find(needle)
        .unwrap_or_else(|| panic!("`{needle}` not in `{text}`"))
        + 1;
    assert_eq!(
        (err.span.line, err.span.col),
        (line, col),
        "{src}\n{err}\nexpected at `{needle}`"
    );
}

#[test]
fn accepted_blackboxes() {
    // the shape of an FPGA top: a generated clock, a pad on a pin
    let src = top("  logic c, d, u;\n  gen g (.i(clk), .o(c), .w());\n  \
         pad #(.KIND(6'd3)) x (.pin(p), .d(a[0] ^ u), .q(d));\n  \
         always_ff @(posedge c) u <= d;\n  assign y = {3'b000, u};");
    let t = translate(&src).unwrap();
    let printed = t.print();
    // the outputs of the blackboxes are external, like inputs
    assert!(
        printed.contains("  external\n    c : bv1\n    d : bv1\n"),
        "{printed}"
    );
    // the pin, the unconnected output `w` and the inputs of the blackboxes
    // are no variables
    for absent in ["p :", "w :", "g.", "x."] {
        assert!(!printed.contains(absent), "`{absent}` in\n{printed}");
    }
    // a blackbox output is an input clock
    assert_eq!(t.clocks, ["c"]);
    // read only by blackboxes: as good as unread
    assert_eq!(t.unread_inputs, ["clk", "a"]);
    // one warning per blackbox instance, at its name
    let warnings: Vec<String> = t.blackboxes.iter().map(|w| w.to_string()).collect();
    assert_eq!(
        warnings,
        [
            "3:7: `g` is an instance of the blackbox `gen`, whose behaviour is not modelled: \
             `c` takes any value in every time step, and nothing reads its inputs",
            "4:22: `x` is an instance of the blackbox `pad`, whose behaviour is not modelled: \
             `d` takes any value in every time step, and nothing reads its inputs",
        ]
    );

    // a blackbox inside an instantiated module: its outputs stay external
    let src = format!(
        "module top(input logic clk, output logic [7:0] y);\n  \
         sub s (.clk, .y);\nendmodule\n\
         module sub(input logic clk, output logic [7:0] y);\n  \
         logic [7:0] w;\n  gen g (.i(clk), .o(), .w(w));\n  \
         always_ff @(posedge clk) y <= w;\nendmodule\n{STUBS}"
    );
    let t = translate(&src).unwrap();
    let printed = t.print();
    assert!(
        printed.contains("  external\n    clk : bv1\n    s.w : bv8\n"),
        "{printed}"
    );
    assert_eq!(
        t.blackboxes[0].to_string(),
        "6:7: `g` is an instance of the blackbox `gen`, whose behaviour is not modelled: `s.w` \
         takes any value in every time step, and nothing reads its inputs (in instance `s`)"
    );
    // a blackbox without inputs and without connected outputs
    let src = format!(
        "{}(* blackbox *)\nmodule led (output logic on);\nendmodule\n",
        top("  led l (.on());\n  assign y = a;")
    );
    let t = translate(&src).unwrap();
    assert_eq!(
        t.blackboxes[0].msg,
        "`l` is an instance of the blackbox `led`, whose behaviour is not modelled"
    );
    // the reset of a register may come from a blackbox, any width
    let src = top(
        "  logic r;\n  logic [7:0] w;\n  gen g (.i(clk), .o(r), .w(w));\n  \
         always_ff @(posedge clk or posedge r)\n    if (r) y <= 4'd0; else y <= w[3:0];",
    );
    translate(&src).unwrap();
    // an unused pin, a blackbox without connected outputs
    let src = top("  pad x (.pin(), .d(clk), .q());\n  assign y = a;");
    translate(&src).unwrap();
}

#[test]
fn rejected_pins() {
    let msg = "`p` is an `inout` port: it may only be connected to an `inout` port of a blackbox";
    rejects(&top("  assign y = {3'b000, p};"), 2, "p}", msg);
    rejects(
        &top("  assign y = a;\n  always_comb if (p) begin end"),
        3,
        "p)",
        msg,
    );
    rejects(&top("  assign p = clk;\n  assign y = a;"), 2, "p =", msg);
    rejects(
        &top("  logic q;\n  pad x (.pin(p), .d(p), .q(q));\n  assign y = a;"),
        3,
        "p), .q",
        msg,
    );
    rejects(
        &top("  logic q;\n  pad x (.pin(p), .d(clk), .q(p));\n  assign y = a;"),
        3,
        "p));",
        msg,
    );
    // to a module that is no blackbox
    rejects(
        &format!(
            "{}module sub(input logic i, output logic o);\n  assign o = i;\nendmodule\n",
            top("  logic q;\n  sub s (.i(p), .o(q));\n  assign y = a;")
        ),
        3,
        "p), .o",
        msg,
    );
    // the pin of a blackbox takes a pin of the module
    rejects(
        &top("  logic q;\n  pad x (.pin(clk), .d(clk), .q(q));\n  assign y = a;"),
        3,
        "clk), .d",
        "`inout` port `pin` of `x` must be connected to an `inout` port of `top` (or left unconnected)",
    );
    rejects(
        &top("  logic q;\n  pad x (.pin(p ^ clk), .d(clk), .q(q));\n  assign y = a;"),
        3,
        "^ clk",
        "`inout` port `pin` of `x` must be connected to an `inout` port of `top`",
    );
    rejects(
        "module top(inout [1:0] p, output logic y);\n  \
         pad x (.pin(p), .d(1'b0), .q(y));\nendmodule\n\
         (* blackbox *)\nmodule pad (inout pin, input logic d, output logic q);\nendmodule\n",
        2,
        "p),",
        "`inout` port `pin` of `x` has 1 bit, `p` 2",
    );
    // only blackboxes and the top module have `inout` ports
    rejects(
        &format!(
            "{}module sub(inout q);\nendmodule\n",
            top("  sub s (.q(p));\n  assign y = a;")
        ),
        2,
        "s (",
        "`sub` has an `inout` port `q`: only blackboxes and the top module may have `inout` ports",
    );
    // a blackbox output is never a signal of a pin
    rejects(
        &top(
            "  logic q;\n  pad x (.pin(p), .d(clk), .q(q));\n  pad z (.pin(q), .d(clk), .q());\n  assign y = a;",
        ),
        4,
        "q), .d",
        "`inout` port `pin` of `z` must be connected to an `inout` port of `top`",
    );
}

#[test]
fn rejected_blackboxes() {
    rejects(
        &top("  logic q;\n  nothing x (.pin(p), .d(clk), .q(q));\n  assign y = a;"),
        3,
        "x (",
        "no module `nothing` (for a vendor cell or another module defined outside the design, \
         declare a stub with its ports, marked `(* blackbox *)`: its outputs are then free in \
         every time step)",
    );
    // a blackbox output acts like an input of the module, so it cannot be
    // an output of it
    rejects(
        &top("  logic [7:0] w;\n  gen g (.i(clk), .o(y[0]), .w(w));"),
        3,
        "y[0]",
        "output port `o` of `g` must be connected to a signal name",
    );
    rejects(
        "module top(input logic clk, output logic o);\n  \
         gen g (.i(clk), .o(o), .w());\nendmodule\n\
         (* blackbox *)\nmodule gen (input logic i, output logic o, output logic [7:0] w);\nendmodule\n",
        2,
        "o), .w",
        "`o` is an output of `top`, but port `o` of the blackbox `g` gives a free value in every \
         time step, like an input: connect an internal signal and `assign` it to `o`",
    );
    rejects(
        &top("  logic [3:0] w;\n  gen g (.i(clk), .o(), .w(w));\n  assign y = w;"),
        3,
        "w));",
        "port `w` of `g` has 8 bits, `w` 4",
    );
    rejects(
        &top(
            "  logic [7:0] w;\n  gen g (.i(clk), .o(), .w(w));\n  assign w = 8'd0;\n  assign y = a;",
        ),
        4,
        "w = 8",
        "`w` already driven",
    );
    rejects(
        &top("  logic [7:0] w;\n  gen g (.i(), .o(), .w(w));\n  assign y = w[3:0];"),
        3,
        "g (",
        "input `i` of `g` is not connected",
    );
    rejects(
        &top("  logic q;\n  pad #(.WIDTH(4)) x (.pin(p), .d(clk), .q(q));\n  assign y = a;"),
        3,
        "WIDTH",
        "`pad` has no parameter `WIDTH`",
    );
    rejects(
        &top("  logic q;\n  pad x (.pin(p), .d(clk), .q(q), .z(q));\n  assign y = a;"),
        3,
        "z(q",
        "`pad` has no port `z`",
    );
    // blackbox outputs as clocks: 1 bit, and unrelated to each other
    rejects(
        &top(
            "  logic [7:0] w;\n  logic c;\n  gen g (.i(clk), .o(), .w(w));\n  \
             assign c = w[0];\n  always_ff @(posedge c) y <= a;",
        ),
        6,
        "c)",
        "the clock `c` is a bit of `w`, which has 8 bits: input clocks must be 1-bit inputs \
         (or blackbox outputs)",
    );
    rejects(
        &top("  logic c1, c2, r;\n  gen g1 (.i(clk), .o(c1), .w());\n  \
             gen g2 (.i(clk), .o(c2), .w());\n  always_ff @(posedge c1) r <= a[0];\n  \
             always_ff @(posedge c2) y <= {3'b000, r};"),
        6,
        "r}",
        "`y` (clocked by `c2`) reads `r`, which depends on registers clocked by `c1`: \
         crossings between unrelated clocks are not supported yet",
    );
    // a stub has ports and parameters only
    let stub = |header: &str, body: &str| {
        format!(
            "module top(input logic clk, output logic y);\n  logic q;\n  \
             m u (.i(clk), .q(q));\n  assign y = q;\nendmodule\n\
             (* blackbox *)\nmodule m {header};\n{body}endmodule\n"
        )
    };
    rejects(
        &stub("(input logic i, output logic q)", "  assign q = i;\n"),
        8,
        "assign",
        "the blackbox `m` has a body: a blackbox declares only its ports and parameters",
    );
    rejects(
        &stub("(input logic i, output logic q)", "  logic t;\n"),
        8,
        "logic",
        "the blackbox `m` has a body",
    );
    rejects(
        &stub(
            "#(parameter int W = 1) (input logic i, output logic [W-1:0] q)",
            "",
        ),
        7,
        "W-1:0]",
        "the width of port `q` of the blackbox `m` depends on a parameter: the widths of \
         blackbox ports must be constants",
    );
    rejects(
        "module top(input logic clk, output logic y);\nendmodule\n\
         (* blackbox, keep *)\nmodule m (input logic i);\nendmodule\n",
        3,
        "keep",
        "attribute `keep` not supported",
    );
    rejects(
        "module top(input logic clk, output logic y);\nendmodule\n\
         (* blackbox = 1 *)\nmodule m (input logic i);\nendmodule\n",
        3,
        "blackbox",
        "`blackbox` takes no value",
    );
    rejects(
        "(* keep *)\nmodule top(input logic clk, output logic y);\n  assign y = clk;\nendmodule\n",
        1,
        "keep",
        "attribute `keep` not supported",
    );
    // a blackbox is never the top module
    let only = "(* blackbox *)\nmodule m (input logic i, output logic o);\nendmodule\n";
    let err = translate(only).err().expect("rejected");
    assert!(err.msg.contains("no top module"), "{err}");
    let d = verilog::frontend::parse_design(only, "test.sv").unwrap();
    let err = elab::elaborate(&d, "m", &[]).unwrap_err();
    assert!(
        err.msg
            .contains("`m` is a blackbox: it cannot be the top module"),
        "{err}"
    );
    // two tops besides the stubs
    let err = translate(&format!(
        "module a(input logic i);\nendmodule\nmodule b(input logic i);\nendmodule\n{STUBS}"
    ))
    .err()
    .expect("rejected");
    assert!(err.msg.contains("several top modules (`a`, `b`)"), "{err}");
}

/// Writes `files` (name, text) into a fresh directory and returns their
/// paths.
fn files(test: &str, files: &[(&str, &str)]) -> Vec<std::path::PathBuf> {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("blackbox-{test}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    files
        .iter()
        .map(|(name, text)| {
            let path = dir.join(name);
            std::fs::write(&path, text).unwrap();
            path
        })
        .collect()
}

#[test]
fn several_files() {
    // the design in one file, the stubs and a ROM in another directory
    let paths = files(
        "several",
        &[
            (
                "top.sv",
                "module top(input logic clk, inout p, output logic [7:0] y);\n  \
                 logic d;\n  pad x (.pin(p), .d(clk), .q(d));\n  rom r (.clk, .a(d), .y);\nendmodule\n",
            ),
            (
                "cells.v",
                "(* blackbox *)\nmodule pad (inout pin, input logic d, output logic q);\nendmodule\n\
                 module rom(input logic clk, input logic a, output logic [7:0] y);\n  \
                 logic [7:0] m [0:1];\n  initial $readmemh(\"rom.hex\", m);\n  \
                 always_ff @(posedge clk) y <= m[a];\nendmodule\n",
            ),
            ("rom.hex", "a5 5a\n"),
        ],
    );
    // the CLI names the file of a warning
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_v2rm"))
        .args(&paths[..2])
        .output()
        .unwrap();
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(out.status.success(), "{stderr}");
    assert!(
        stderr.contains(&format!(
            "{}:3:7: warning: `x` is an instance of the blackbox `pad`",
            paths[0].display()
        )),
        "{stderr}"
    );
    let d = verilog::frontend::parse_design_files(&paths[..2]).unwrap();
    assert_eq!(d.top().unwrap(), "top");
    let inst = elab::elaborate(&d, "top", &[]).unwrap();
    let t = lower_inst(&inst).unwrap();
    assert!(t.print().contains("8'd165"), "{}", t.print());

    // an error in the second file is located there
    let paths = files(
        "located",
        &[
            (
                "top.sv",
                "module top(input logic clk, output logic y);\n  m u (.clk, .y);\nendmodule\n",
            ),
            (
                "m.sv",
                "module m(input logic clk, output logic y);\n  assign y = z;\nendmodule\n",
            ),
        ],
    );
    let d = verilog::frontend::parse_design_files(&paths).unwrap();
    let err = elab::elaborate(&d, "top", &[])
        .and_then(|inst| lower_inst(&inst).map(|_| ()))
        .unwrap_err();
    let (file, span) = d.locate(err.span);
    assert_eq!(file, paths[1]);
    assert_eq!((span.line, span.col), (2, 14), "{err}");
    // a module declared in two files
    let paths = files(
        "twice",
        &[
            ("a.sv", "module m(input logic i);\nendmodule\n"),
            ("b.sv", "\nmodule m(input logic i);\nendmodule\n"),
        ],
    );
    let err = verilog::frontend::parse_design_files(&paths)
        .err()
        .expect("rejected");
    assert!(err.msg.contains("module `m` already declared"), "{err}");
    // the CLI takes several files and names the file of an error
    let paths_text: Vec<String> = paths.iter().map(|p| p.display().to_string()).collect();
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_v2rm"))
        .args(&paths_text)
        .output()
        .unwrap();
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(!out.status.success());
    assert!(
        stderr.starts_with(&format!(
            "{}:2:8: module `m` already declared at {}:1:8",
            paths_text[1], paths_text[0]
        )),
        "{stderr}"
    );
}

/// Positions that a message mentions besides its own (an earlier driver, a
/// gate) are given in their own files too, like the position of the error:
/// as `L:C` in the error's file, else with the path of their file.
#[test]
fn positions_in_messages_of_later_files() {
    let error = |paths: &[std::path::PathBuf]| {
        let d = verilog::frontend::parse_design_files(paths).unwrap();
        let err = elab::elaborate(&d, "top", &[])
            .and_then(|inst| lower_inst(&inst).map(|_| ()))
            .unwrap_err();
        d.located(err)
    };
    // both positions in the second file
    let paths = files(
        "mentions",
        &[
            (
                "a.sv",
                "module a(input logic x, output logic y);\n  assign y = x;\nendmodule\n",
            ),
            (
                "b.sv",
                "// second\nmodule top(input logic p, q, output logic y);\n  assign y = p;\n  \
                 assign y = q;\nendmodule\n",
            ),
        ],
    );
    let err = error(&paths);
    assert_eq!(err.file.as_ref(), Some(&paths[1]));
    assert_eq!(err.to_string(), "4:10: `y` already driven at 3:10");
    // the error in the first file, the gate in the second
    let paths = files(
        "mentions-other",
        &[
            (
                "top.sv",
                "module top(input logic a, b, output logic y);\n  logic g;\n  m u(.a, .b, .g);\n  \
                 always_ff @(posedge g) y <= 1;\nendmodule\n",
            ),
            (
                "m.sv",
                "// child\nmodule m(input logic a, b, output logic g);\n  assign g = a & b;\nendmodule\n",
            ),
        ],
    );
    let err = error(&paths);
    assert_eq!(err.file.as_ref(), Some(&paths[0]));
    assert_eq!(
        err.to_string(),
        format!(
            "4:23: the clock `g` comes from `u.g`, computed combinationally at {}:3:10: gated \
             clocks are not supported",
            paths[1].display()
        )
    );
    // the CLI prints the same
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_v2rm"))
        .args(&paths)
        .output()
        .unwrap();
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert_eq!(out.status.code(), Some(2), "{stderr}");
    assert!(
        stderr.starts_with(&format!("{}:4:23: ", paths[0].display()))
            && stderr.contains(&format!("at {}:3:10:", paths[1].display())),
        "{stderr}"
    );
}
