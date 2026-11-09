//! The frontend and typing: operator precedence, ports, and a located error
//! for every rejected construct.

use verilog::ast::{Dir, Edge, Item, Ty};
use verilog::lower::Translation;
use verilog::{frontend, lower};

/// The right-hand side of `assign y = <e>;`, fully parenthesized.
fn parse_expr(e: &str) -> String {
    let src = format!(
        "module m(input logic [7:0] a, b, c, d, e, output logic [7:0] y);\n  assign y = {e};\nendmodule\n"
    );
    let m = frontend::parse(&src, "test.sv").unwrap();
    match &m.items[..] {
        [Item::Assign { rhs, .. }] => rhs.to_string(),
        items => panic!("{items:?}"),
    }
}

#[test]
fn precedence() {
    let cases = [
        ("a * b + c", "((a * b) + c)"),
        ("a + b * c", "(a + (b * c))"),
        ("a - b - c", "((a - b) - c)"),
        ("a - b + c - d", "(((a - b) + c) - d)"),
        ("(a + b) * c", "((a + b) * c)"),
        ("-a * b", "((-a) * b)"),
        ("a | b ^ c & d", "(a | (b ^ (c & d)))"),
        ("a & b == c", "(a & (b == c))"),
        ("a ~^ b | c", "((a ~^ b) | c)"),
        ("a == b && c < d || e", "(((a == b) && (c < d)) || e)"),
        ("!a && b", "((!a) && b)"),
        ("a + b ? c : d", "((a + b) ? c : d)"),
        ("a ? b : c ? d : e", "(a ? b : (c ? d : e))"),
        ("a ? b ? c : d : e", "(a ? (b ? c : d) : e)"),
        ("a ? b + c : d", "(a ? (b + c) : d)"),
        ("a ? b : c + d", "(a ? b : (c + d))"),
        ("a || b ? c : d || e", "((a || b) ? c : (d || e))"),
        ("8'd3 + 8'shf_f", "(8'd3 + 8'sd255)"),
        ("a << b + c", "(a << (b + c))"),
        ("a >> b < c", "((a >> b) < c)"),
        ("a <<< b >>> c", "((a <<< b) >>> c)"),
    ];
    for (src, expected) in cases {
        assert_eq!(parse_expr(src), expected, "{src}");
    }
}

#[test]
fn ports_inherit_direction_and_type() {
    let src = "module m(input logic signed [7:0] a, b, input c, output wire [3:0] y, z);\n\
               logic t; wire [1:0] u, v;\nendmodule\n";
    let m = frontend::parse(src, "test.sv").unwrap();
    let ports: Vec<_> = m
        .ports
        .iter()
        .map(|p| (p.decl.name.as_str(), p.dir, p.decl.ty))
        .collect();
    let s8 = Ty {
        width: 8,
        signed: true,
    };
    assert_eq!(
        ports,
        [
            ("a", Dir::Input, s8),
            ("b", Dir::Input, s8),
            ("c", Dir::Input, Ty::BIT),
            ("y", Dir::Output, Ty::unsigned(4)),
            ("z", Dir::Output, Ty::unsigned(4)),
        ]
    );
    let decls: Vec<_> = m.decls.iter().map(|d| (d.name.as_str(), d.ty)).collect();
    assert_eq!(
        decls,
        [
            ("t", Ty::BIT),
            ("u", Ty::unsigned(2)),
            ("v", Ty::unsigned(2))
        ]
    );
}

const HEADER: &str = "module m(input logic [7:0] a, output logic [7:0] y);";

/// Translates `body` in a module with an 8-bit input `a` and output `y`
/// (if `header` is `None`), expects an error at the first occurrence of
/// `needle` in line `line`, with a message containing `msg`.
fn rejects(header: Option<&str>, body: &str, line: usize, needle: &str, msg: &str) {
    let src = format!("{}\n{body}\nendmodule\n", header.unwrap_or(HEADER));
    let err = match frontend::parse(&src, "test.sv").and_then(|m| lower::lower(&m)) {
        Ok(_) => panic!("accepted:\n{src}"),
        Err(e) => e,
    };
    let text = src.lines().nth(line - 1).unwrap();
    let col = text
        .find(needle)
        .unwrap_or_else(|| panic!("`{needle}` not in `{text}`"))
        + 1;
    assert!(err.msg.contains(msg), "{src}\n{err}\nexpected: {msg}");
    assert_eq!(
        (err.span.line, err.span.col),
        (line, col),
        "{src}\n{err}\nexpected at `{needle}`"
    );
}

#[test]
fn rejected_expressions() {
    let cases = [
        ("  assign y = a + 1;", "1;", "unsized literal"),
        ("  assign y = a + 'd1;", "'d1", "unsized literal"),
        ("  assign y = a + '1;", "'1", "unsized literal"),
        ("  assign y = 8'bx;", "8'bx", "x/z"),
        ("  assign y = 8'd256;", "8'd256", "does not fit"),
        (
            "  assign y = a ** a;",
            "**",
            "`**` only between integer constants",
        ),
        (
            "  assign y = $clog2(a);",
            "$clog2",
            "`$clog2` of an integer constant only",
        ),
        (
            "  assign y = $countones(a);",
            "$countones",
            "call `$countones(a)`",
        ),
        ("  assign y = b;", "b;", "`b` is not declared"),
        ("  assign y = a + 4'd1;", "+", "operands of `+` differ"),
        (
            "  assign y = a == a ? a : 8'sd0;",
            "?",
            "arms of `?:` differ",
        ),
        ("  assign y = 4'd1;", "y", "assigning 4 bits to 8 bits"),
        ("  assign y = a == a;", "y", "assigning 1 bits to 8 bits"),
    ];
    for (body, needle, msg) in cases {
        rejects(None, body, 2, needle, msg);
    }
    // operands differ in signedness only
    rejects(
        Some("module m(input logic [7:0] a, input logic signed [7:0] s, output logic y);"),
        "  assign y = a < s;",
        2,
        "<",
        "operands of `<` differ: `logic [7:0]` and `logic signed [7:0]`",
    );
}

#[test]
fn rejected_drivers() {
    rejects(
        None,
        "  assign a = y;\n  assign y = a;",
        2,
        "a =",
        "input `a` is assigned",
    );
    rejects(
        None,
        "  assign y = a;\n  assign y = a;",
        3,
        "y",
        "already driven at 2:10",
    );
    rejects(None, "", 1, "y)", "`y` is never driven");
    rejects(
        None,
        "  logic t;\n  assign y = a;",
        2,
        "t",
        "`t` is never driven",
    );
    rejects(None, "  logic [7:0] a;", 2, "a", "already declared at 1:28");
    rejects(
        None,
        "  assign y = y;",
        2,
        "y",
        "combinational loop: y -> y",
    );
    rejects(
        None,
        "  logic [7:0] t, u;\n  assign t = u;\n  assign u = t + a;\n  assign y = t;",
        3,
        "t",
        "combinational loop: t -> u -> t",
    );
}

#[test]
fn rejected_declarations() {
    let header = |h: &'static str| Some(h);
    rejects(
        header("module m(input reg [7:0] a, output logic [7:0] y);"),
        "",
        1,
        "reg",
        "`reg` not supported",
    );
    rejects(
        header("module m(input logic [8:1] a, output logic [7:0] y);"),
        "",
        1,
        "[8",
        "range [8:1] not supported",
    );
    rejects(
        header("module m(input logic [0:7] a, output logic [7:0] y);"),
        "",
        1,
        "[0",
        "range [0:7] not supported",
    );
    rejects(
        header("module m(input logic [7:0] a, inout logic [7:0] y);"),
        "",
        1,
        "inout",
        "unsupported",
    );
    rejects(
        header("module m(input int a, output logic [7:0] y);"),
        "",
        1,
        "int",
        "type `int` not supported",
    );
    rejects(None, "  wire [7:0] t = a;", 2, "t", "initializer");
    rejects(None, "  tri [7:0] t;", 2, "tri", "net type `tri`");
    rejects(None, "  logic [7:0] t [4];", 2, "t", "unpacked dimension");
}

#[test]
fn rejected_items() {
    rejects(None, "  assign #1 y = a;", 2, "#1", "delay");
    rejects(None, "  initial y = a;", 2, "initial", "unsupported");
    rejects(None, "  assign y = ;", 2, ";", "syntax error");
}

const HEADER_FF: &str = "module m(input logic clk, input logic [7:0] a, output logic [7:0] y);";

fn translate_ff(body: &str) -> Translation {
    let src = format!("{HEADER_FF}\n{body}\nendmodule\n");
    frontend::parse(&src, "test.sv")
        .and_then(|m| lower::lower(&m))
        .unwrap_or_else(|e| panic!("{src}\n{e}"))
}

/// `else if` chains nest to the right; a missing `else` is empty.
#[test]
fn if_else_chains() {
    let src = format!(
        "{HEADER_FF}\n  always_ff @(posedge clk)\n    if (a == 8'd0) y <= a; else if (a < 8'd9) y <= 8'd1; else if (|a) ;\nendmodule\n"
    );
    let m = frontend::parse(&src, "test.sv").unwrap();
    let [
        Item::AlwaysFf {
            clock, edge, body, ..
        },
    ] = &m.items[..]
    else {
        panic!("{:?}", m.items);
    };
    assert_eq!(clock, "clk");
    assert_eq!(*edge, Edge::Pos);
    assert_eq!(
        body.to_string(),
        "if ((a == 8'd0)) y <= a; else if ((a < 8'd9)) y <= 8'd1; else if ((|a)) begin end else begin end"
    );
}

#[test]
fn accepted_always_ff() {
    // several assignments to one register in a block: the last one wins
    let t = translate_ff("  always_ff @(posedge clk) begin y <= a; y <= y; end");
    assert_eq!(t.registers, ["y"]);
    assert_eq!(t.clocks, ["clk"]);
    // a loop through a register is no combinational loop
    let t =
        translate_ff("  logic [7:0] t;\n  assign t = y + a;\n  always_ff @(posedge clk) y <= t;");
    assert_eq!(t.module.atoms().len(), 2);
    // the clock is an external variable
    assert!(t.unread_inputs.is_empty());
    let clk = t.var("clk");
    assert!(t.module.extl().contains(&clk));
}

/// Several clocks, both edges, and clocks read as data.
#[test]
fn accepted_clocks() {
    let src = "module m(input logic clk, input logic [7:0] a, output logic [7:0] y);\n  \
               always_ff @(negedge clk) y <= a;\nendmodule\n";
    let m = frontend::parse(src, "test.sv").unwrap();
    let [Item::AlwaysFf { edge, .. }] = &m.items[..] else {
        panic!("{:?}", m.items);
    };
    assert_eq!(*edge, Edge::Neg);
    // both edges of one clock, in different blocks; a block on one edge
    // reads a register of the other
    let t = translate_ff(
        "  logic [7:0] n;\n  always_ff @(negedge clk) n <= a;\n  always_ff @(posedge clk) y <= n;",
    );
    assert_eq!(t.clocks, ["clk"]);
    // the clock read as data, in an `if` and through an `assign`
    let t = translate_ff("  always_ff @(posedge clk) if (clk) y <= a;");
    assert_eq!(t.registers, ["y"]);
    let t = translate_ff(
        "  logic [7:0] t;\n  assign t = clk ? a : y;\n  always_ff @(negedge clk) y <= t;",
    );
    assert_eq!(t.clocks, ["clk"]);
    // two unrelated clocks, each register reading inputs and its own
    // domain only; an `assign` may combine both domains
    let two = "module m(input logic ca, cb, input logic [7:0] a, output logic [7:0] y, z, s);";
    let src = format!(
        "{two}\n  always_ff @(posedge ca) y <= y + a;\n  always_ff @(negedge cb) z <= z + a;\n  \
         assign s = y + z;\nendmodule\n"
    );
    let t = frontend::parse(&src, "test.sv")
        .and_then(|m| lower::lower(&m))
        .unwrap();
    assert_eq!(t.clocks, ["ca", "cb"]);
    // a clock that is only read as data is no clock
    let src = format!(
        "{two}\n  always_ff @(posedge ca) y <= a;\n  always_ff @(posedge ca) z <= cb ? a : z;\n  \
         assign s = a;\nendmodule\n"
    );
    let t = frontend::parse(&src, "test.sv")
        .and_then(|m| lower::lower(&m))
        .unwrap();
    assert_eq!(t.clocks, ["ca"]);
}

#[test]
fn rejected_always() {
    let cases = [
        (
            "  always_ff @(posedge clk) y = a;",
            "y =",
            "blocking assignment",
        ),
        (
            "  always_ff @(edge clk) y <= a;",
            "edge",
            "`edge` not supported: use separate `posedge` and `negedge` blocks",
        ),
        (
            "  always_ff @(posedge clk, negedge clk) y <= a;",
            "posedge",
            "only one event",
        ),
        (
            "  always_ff @(posedge clk or posedge a) y <= a;",
            "posedge",
            "only one event",
        ),
        ("  always_ff @(clk) y <= a;", "clk)", "level-sensitive"),
        ("  always_ff @(posedge clk iff a) y <= a;", "iff", "`iff`"),
        ("  always_ff @* y <= a;", "@", "needs `@(posedge"),
        ("  always_ff y <= a;", "y <=", "needs `@(posedge"),
        (
            "  always @(posedge clk) y <= a;",
            "always",
            "plain `always` not supported, use `always_ff` or `always_comb`",
        ),
        (
            "  always_ff @(posedge clk) ;",
            "always_ff",
            "drives nothing",
        ),
        ("  always_latch y <= a;", "always_latch", "`always_latch`"),
        (
            "  always_ff @(posedge clk) y <= #1 a;",
            "#1",
            "intra-assignment delay",
        ),
        (
            "  always_ff @(posedge clk) begin : b y <= a; end",
            ": b",
            "block name",
        ),
        (
            "  always_ff @(posedge clk) begin logic [7:0] t; y <= a; end",
            "logic",
            "declarations in blocks",
        ),
        (
            "  always_ff @(posedge clk) unique if (a == a) y <= a;",
            "unique",
            "`unique` not supported",
        ),
        (
            "  always_ff @(posedge clk) @(posedge clk) y <= a;",
            "@(posedge clk) y",
            "unsupported",
        ),
        ("  always_ff @(posedge clk) y++;", "y++", "unsupported"),
    ];
    for (body, needle, msg) in cases {
        rejects(Some(HEADER_FF), body, 2, needle, msg);
    }
}

#[test]
fn rejected_clocks() {
    rejects(
        Some(HEADER_FF),
        "  always_ff @(posedge a) y <= a;",
        2,
        "a)",
        "the clock `a` must be a 1-bit input",
    );
    rejects(
        Some(HEADER_FF),
        "  logic c;\n  assign c = &a;\n  always_ff @(posedge c) y <= a;",
        4,
        "c)",
        "the clock `c` is computed combinationally: gated clocks are not supported",
    );
    rejects(
        Some(HEADER_FF),
        "  logic g;\n  always_comb g = clk & a[0];\n  always_ff @(negedge g) y <= a;",
        4,
        "g)",
        "the clock `g` is computed combinationally",
    );
    rejects(
        Some(HEADER_FF),
        "  logic s;\n  always_ff @(posedge clk) s <= !s;\n  always_ff @(posedge s) y <= a;",
        4,
        "s)",
        "the clock `s` is a register: clocks derived from registers are not supported yet",
    );
    rejects(
        Some(HEADER_FF),
        "  always_ff @(posedge y[0]) y <= a;",
        2,
        "y[0]",
        "the clock must be a signal name",
    );
    rejects(
        Some(HEADER_FF),
        "  always_ff @(posedge nope) y <= a;",
        2,
        "nope",
        "`nope`",
    );
    // reads across unrelated clocks, directly and through `assign`
    let two = Some("module m(input logic ca, cb, input logic [7:0] a, output logic [7:0] y, z);");
    rejects(
        two,
        "  always_ff @(posedge ca) y <= a;\n  always_ff @(posedge cb) z <= y;",
        3,
        "y;",
        "`z` (clocked by `cb`) reads `y`, which depends on registers clocked by `ca`: \
         crossings between unrelated clocks are not supported yet",
    );
    rejects(
        two,
        "  logic [7:0] t;\n  assign t = a + y;\n  always_ff @(negedge ca) y <= a;\n  \
         always_ff @(negedge cb) if (t[0]) z <= a;",
        5,
        "t[0]",
        "`z` (clocked by `cb`) reads `t`, which depends on registers clocked by `ca`",
    );
}

#[test]
fn rejected_registers() {
    let two = Some("module m(input logic clk, input logic [7:0] a, output logic [7:0] y, z);");
    rejects(
        two,
        "  always_ff @(posedge clk) begin y <= a; z <= a; end\n  always_ff @(posedge clk) y <= a;",
        3,
        "y <=",
        "`y` already driven at 2:34",
    );
    rejects(
        Some(HEADER_FF),
        "  assign y = a;\n  always_ff @(posedge clk) y <= a;",
        3,
        "y <=",
        "`y` already driven at 2:10",
    );
    rejects(
        Some(HEADER_FF),
        "  always_ff @(posedge clk) y <= a;\n  assign y = a;",
        3,
        "y",
        "`y` already driven at 2:28",
    );
    rejects(
        Some(HEADER_FF),
        "  always_ff @(posedge clk) begin a <= y; y <= a; end",
        2,
        "a <=",
        "input `a` is assigned",
    );
    rejects(
        Some(HEADER_FF),
        "  always_ff @(posedge clk) y <= 4'd1;",
        2,
        "y <=",
        "assigning 4 bits to 8 bits",
    );
    rejects(
        Some(HEADER_FF),
        "  always_ff @(posedge clk) z <= a;\n  assign y = a;",
        2,
        "z",
        "`z` is not declared",
    );
}

/// The statement of `always_comb <stmt>` in `body`, on one line.
fn comb_body(body: &str) -> String {
    let src = format!("{HEADER}\n{body}\nendmodule\n");
    let m = frontend::parse(&src, "test.sv").unwrap();
    let [Item::AlwaysComb { body, .. }] = &m.items[..] else {
        panic!("{:?}", m.items);
    };
    body.to_string()
}

/// `default` is kept apart wherever it is written; a missing one is
/// `None`.
#[test]
fn case_statements() {
    assert_eq!(
        comb_body(
            "  always_comb case (a) 8'd1, 8'd2: y = a; default: y = 8'd0; 8'd3: begin end endcase"
        ),
        "case (a) 8'd1, 8'd2: y = a; 8'd3: begin end default: y = 8'd0; endcase"
    );
    assert_eq!(
        comb_body("  always_comb begin y = a; case (a) 8'd1: y = 8'd0; endcase end"),
        "begin y = a; case (a) 8'd1: y = 8'd0; endcase end"
    );
}

fn translate(body: &str) -> Translation {
    let src = format!("{HEADER}\n{body}\nendmodule\n");
    frontend::parse(&src, "test.sv")
        .and_then(|m| lower::lower(&m))
        .unwrap_or_else(|e| panic!("{src}\n{e}"))
}

#[test]
fn accepted_always_comb() {
    // a default assignment first: no latch
    translate("  always_comb begin y = 8'd0; if (a == 8'd1) y = a; end");
    // a case without default after a default assignment
    translate("  always_comb begin y = a; case (a) 8'd1: y = 8'd0; endcase end");
    // reads of the block's own signal after its assignment
    let t = translate("  logic [7:0] t;\n  always_comb begin t = a + a; y = t; t = t + 8'd1; end");
    assert_eq!(t.module.atoms().len(), 1);
    // an awaited variable of an always_comb is no loop when the other block
    // reads only what does not depend on it
    let t = translate(
        "  logic [7:0] t, u;\n  always_comb begin t = a; u = t + a; end\n  assign y = u;",
    );
    assert_eq!(t.module.atoms().len(), 2);
}

#[test]
fn rejected_always_comb() {
    let cases: [(&str, usize, &str, &str); 14] = [
        (
            "  always_comb y <= a;",
            2,
            "y <=",
            "non-blocking assignment not allowed in `always_comb`",
        ),
        ("  always_comb y += a;", 2, "+=", "`+=` not supported"),
        ("  always_comb y = #1 a;", 2, "#1", "intra-assignment delay"),
        (
            "  always_comb begin end",
            2,
            "always_comb",
            "drives nothing",
        ),
        (
            "  always_comb if (a == 8'd0) y = a;",
            2,
            "if",
            "`y` is assigned on only some paths through this `if`",
        ),
        (
            "  always_comb\n    case (a) 8'd0: y = a; 8'd1: ; default: y = a; endcase",
            3,
            "case",
            "`y` is assigned on only some paths through this `case`",
        ),
        (
            "  always_comb case (a) 8'd0: y = a; endcase",
            2,
            "case",
            "without `default`, nothing is assigned",
        ),
        (
            "  always_comb begin y = y + a; end",
            2,
            "y + a",
            "`y` read before it is assigned",
        ),
        (
            "  always_comb begin if (a == 8'd0) y = a; y = y; end",
            2,
            "y; end",
            "`y` read before it is assigned",
        ),
        (
            "  always_comb case (a) 8'd1: y = a; 4'd2: y = a; default: y = a; endcase",
            2,
            "4'd2",
            "`case` label `logic [3:0]` differs from selector `logic [7:0]`",
        ),
        (
            "  always_comb case (a) default: y = a; default: y = a; endcase",
            2,
            "default: y = a; endcase",
            "second `default`",
        ),
        (
            "  always_comb casez (a) default: y = a; endcase",
            2,
            "casez",
            "`casez` not supported",
        ),
        (
            "  always_comb unique case (a) default: y = a; endcase",
            2,
            "unique",
            "`unique` not supported",
        ),
        (
            "  always_comb case (a) inside default: y = a; endcase",
            2,
            "inside",
            "`case … inside` not supported",
        ),
    ];
    for (body, line, needle, msg) in cases {
        rejects(None, body, line, needle, msg);
    }
    // drivers: one process per signal
    rejects(
        None,
        "  assign y = a;\n  always_comb y = a;",
        3,
        "y =",
        "already driven at 2:10",
    );
    rejects(
        None,
        "  always_comb y = a;\n  always_comb y = a;",
        3,
        "y =",
        "already driven at 2:15",
    );
}

/// A loop through an `always_comb` driving several signals, which the data
/// flow does not have, names the block to split; a real loop does
/// not.
#[test]
fn comb_loops() {
    rejects(
        Some("module m(input logic [7:0] a, output logic [7:0] x, y);"),
        "  logic [7:0] z;\n  always_comb begin x = a; y = z; end\n  always_comb z = x;",
        3,
        "always_comb begin",
        "combinational loop: x -> z -> x; in the `always_comb` at 3:3, `x` does not depend on `z`, \
         but the block is translated as a whole: split it",
    );
    rejects(
        Some("module m(input logic [7:0] a, output logic [7:0] x, y);"),
        "  logic [7:0] z;\n  always_comb begin x = z; y = a; end\n  always_comb z = x;",
        3,
        "always_comb begin",
        "combinational loop: x -> z -> x",
    );
    let src = "module m(input logic [7:0] a, output logic [7:0] x, y);\n  logic [7:0] z;\n  always_comb begin x = z; y = a; end\n  always_comb z = x;\nendmodule\n";
    let err = frontend::parse(src, "t.sv")
        .and_then(|m| lower::lower(&m))
        .err()
        .unwrap();
    assert!(!err.msg.contains("split"), "{err}");
    // control dependencies count: `x` depends on `z` through the condition
    let src = "module m(input logic [7:0] a, output logic [7:0] x, y);\n  logic [7:0] z;\n  always_comb begin y = a; if (z == a) x = a; else x = y; end\n  always_comb z = x;\nendmodule\n";
    let err = frontend::parse(src, "t.sv")
        .and_then(|m| lower::lower(&m))
        .err()
        .unwrap();
    assert!(err.msg.starts_with("combinational loop"), "{err}");
    assert!(!err.msg.contains("split"), "{err}");
}

/// Concatenation, replication, selects and casts as parsed (`$signed` is
/// `signed'`), and division at the precedence of `*`.
#[test]
fn sizing_syntax() {
    let cases = [
        ("{a, b[3:0], c[2]}", "{a, b[3:0], c[2]}"),
        ("{2{a, b}}", "{2{a, b}}"),
        ("{a}[0]", ""),
        ("a[b +: 4] | a[7 -: 2]", "(a[b +: 4] | a[7 -: 2])"),
        ("a[b + c]", "a[(b + c)]"),
        ("16'(a + b)", "16'((a + b))"),
        ("$signed(a) + signed'(b)", "(signed'(a) + signed'(b))"),
        (
            "$unsigned(a) + unsigned'(b)",
            "(unsigned'(a) + unsigned'(b))",
        ),
        ("a + b / c % d", "(a + ((b / c) % d))"),
        ("a * b / c", "((a * b) / c)"),
    ];
    for (src, expected) in cases {
        if expected.is_empty() {
            let src = format!(
                "module m(input logic [7:0] a, output logic y);\n  assign y = {src};\nendmodule\n"
            );
            let err = frontend::parse(&src, "test.sv").unwrap_err();
            assert!(err.msg.contains("select of a concatenation"), "{err}");
        } else {
            assert_eq!(parse_expr(src), expected, "{src}");
        }
    }
}

const HEADER_SIZING: &str = "module m(input logic [7:0] a, input logic signed [7:0] s, \
     input logic [2:0] i, input logic [3:0] j, input logic signed [2:0] k, \
     output logic [7:0] y, output logic [15:0] w, output logic o);";

fn translate_with(header: &str, body: &str) -> Translation {
    let src = format!("{header}\n{body}\nendmodule\n");
    frontend::parse(&src, "test.sv")
        .and_then(|m| lower::lower(&m))
        .unwrap_or_else(|e| panic!("{src}\n{e}"))
}

#[test]
fn accepted_sizing() {
    let ok = [
        // concatenation and selects are unsigned, of the width they say
        "  assign w = {a, a[7:4], a[3:0]};",
        "  assign w = {2{a}};",
        "  assign y = {8{a[0]}};",
        // a variable index that stays in range: 3 bits for 8 bits
        "  assign o = a[i];",
        "  assign y = w[i +: 8];",
        "  assign o = a[i -: 1];",
        // widening casts of context-free operands, narrowing of anything
        "  assign w = 16'(a) + 16'(s[7:0]);",
        "  assign w = 16'({a, a});",
        "  assign w = 16'(8'(a + a));",
        "  assign y = 8'(w + w);",
        "  assign y = $unsigned(s) + a;",
        // the divisor is a nonzero literal
        "  assign y = a / 8'd3 + a % 8'd7;",
        "  assign y = $unsigned(s / 8'shff);",
        // shifts
        "  assign y = a << i | a >> j | $unsigned(s >>> a);",
    ];
    for body in ok {
        // drive the outputs the case leaves alone
        let mut src = body.to_string();
        for (out, value) in [("y", "a"), ("w", "{a, a}"), ("o", "a[0]")] {
            if !body.contains(&format!("assign {out} ")) {
                src += &format!("\n  assign {out} = {value};");
            }
        }
        translate_with(HEADER_SIZING, &src);
    }
}

#[test]
fn rejected_sizing() {
    let cases = [
        (
            "  assign o = a[8];",
            "a[",
            "`a[8]` out of range: `a` has bits 7:0",
        ),
        ("  assign y = w[16 -: 8];", "w[", "out of range"),
        (
            "  assign y = {a[3:0], a[0:3]};",
            "a[0",
            "write the higher bit first",
        ),
        ("  assign o = a[0 -: 2];", "a[", "out of range"),
        (
            "  assign y = {a[i +: 4], a[3:0]};",
            "a[i",
            "may select beyond bit 7",
        ),
        ("  assign o = a[j];", "a[j", "the index has 4 bits"),
        ("  assign o = a[k];", "a[k", "must be unsigned"),
        ("  assign o = a[i -: 2];", "a[i", "use `+:`"),
        (
            "  assign w = 16'(a + a);",
            "16'",
            "cast its operands instead",
        ),
        (
            "  assign w = 16'(-a);",
            "16'",
            "would evaluate this operator expression at 16 bits",
        ),
        (
            "  assign y = s + a[7:0];",
            "+",
            "`logic signed [7:0]` and `logic [7:0]`",
        ),
        ("  assign y = s + {s};", "+", "operands of `+` differ"),
        (
            "  assign y = a / a;",
            "a;",
            "divisor must be a nonzero literal",
        ),
        (
            "  assign y = a % 8'd0;",
            "8'd0",
            "divisor must be a nonzero literal",
        ),
        (
            "  assign y = a / -8'd1;",
            "-8",
            "divisor must be a nonzero literal",
        ),
        ("  assign w = {0{a}};", "{0", "replication count 0"),
        ("  assign o = a[0 +: 0];", "a[", "of width 0"),
        (
            "  assign y = $signed(a, a);",
            "$signed",
            "takes one argument",
        ),
        ("  assign y = string'(a);", "string", "cast to `string`"),
        ("  assign y = {a, a};", "y", "assigning 16 bits to 8 bits"),
    ];
    for (body, needle, msg) in cases {
        rejects(Some(HEADER_SIZING), body, 2, needle, msg);
    }
}

/// Targets: selects and concatenations, in every kind of block.
#[test]
fn accepted_targets() {
    let header = "module m(input logic clk, input logic [7:0] a, input logic e, \
                  output logic [7:0] y, q, output logic c, output logic [6:0] d);";
    let t = translate_with(
        header,
        "  assign {c, d} = a;\n  assign y = a;\n  assign q = a;",
    );
    assert_eq!(t.module.atoms().len(), 3);
    translate_with(header, "  assign {y[7:4], y[3:0], q, c, d} = {a, a, a};");
    // in always_ff, unassigned bits keep their value
    translate_with(
        header,
        "  always_ff @(posedge clk) begin q[0] <= e; if (e) q[7:4] <= a[3:0]; end\n\
         \x20 assign {y, c, d} = {a, a};",
    );
    // in always_comb, every bit is assigned before it is read
    translate_with(
        header,
        "  always_comb begin y[7:4] = a[3:0]; y[3:0] = y[7:4]; {c, d} = y; end\n\
         \x20 assign q = a;",
    );
    translate_with(
        header,
        "  always_comb begin\n    if (e) y[7:1] = a[6:0]; else y[7:1] = a[7:1];\n    \
         y[0] = e; q = y;\n  end\n  assign {c, d} = a;",
    );
}

#[test]
fn rejected_targets() {
    let header = "module m(input logic clk, input logic [7:0] a, input logic [2:0] i, \
                  input logic e, output logic [7:0] y, q);";
    let cases: [(&str, usize, &str, &str); 9] = [
        (
            "  assign y[3:0] = a[3:0];\n  assign y[7:4] = a[7:4];\n  assign q = a;",
            2,
            "y[",
            "this `assign` leaves `y[7:4]` undriven",
        ),
        (
            "  assign {y[7:1], q} = {a[7:1], a};\n  assign y[0] = 1'b0;",
            2,
            "{y",
            "leaves `y[0]` undriven",
        ),
        (
            "  assign {y, y[0]} = {a, 1'b0};\n  assign q = a;",
            2,
            "y[0]",
            "`y[0]` assigned twice in one target",
        ),
        (
            "  always_comb begin y = a; y[i] = 1'b0; end\n  assign q = a;",
            2,
            "i]",
            "variable index in an assignment target",
        ),
        (
            "  assign y[i] = 1'b0;\n  assign q = a;",
            2,
            "i]",
            "variable index in an assignment target",
        ),
        (
            "  always_ff @(posedge clk) q[8] <= 1'b0;\n  assign y = a;",
            2,
            "q[8]",
            "out of range",
        ),
        (
            "  always_comb begin y[3:0] = a[3:0]; q = y; end",
            2,
            "y;",
            "`y[7:4]` read before it is assigned on every path",
        ),
        (
            "  always_comb begin y[3:0] = a[3:0]; if (e) y[7:4] = a[7:4]; q = a; end",
            2,
            "if",
            "`y[7:4]` is assigned on only some paths through this `if`",
        ),
        (
            "  always_comb begin y[6:2] = a[4:0]; q = a; end",
            2,
            "always_comb",
            "`y[7]`, `y[1:0]` is never assigned in this `always_comb`",
        ),
    ];
    for (body, line, needle, msg) in cases {
        rejects(Some(header), body, line, needle, msg);
    }
    // per whole signal, as for whole assignments: one driver
    rejects(
        Some(header),
        "  always_ff @(posedge clk) q[3:0] <= a[3:0];\n  always_ff @(posedge clk) q[7:4] <= a[7:4];\n  assign y = a;",
        3,
        "q[",
        "`q` already driven at 2:28",
    );
}
