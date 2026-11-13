//! The frontend and typing: operator precedence, ports, and a located error
//! for every rejected construct.

use verilog::ast::{Dir, Edge, Item, Ty};
use verilog::lower::Translation;
use verilog::{elab, frontend, lower};

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
        ("  assign y = 8'bx;", "8'bx", "x/z"),
        ("  assign y = 'x;", "'x", "x/z"),
        ("  assign y = 'hz;", "'hz", "x/z"),
        ("  assign y = a + 4294967296;", "4294967296", "too large"),
        (
            "  assign y = a + 'h1_0000_0000;",
            "'h1",
            "does not fit in 32 bits",
        ),
        (
            "  assign y = {a, 1};",
            "1}",
            "unsized literal `1` in a concatenation",
        ),
        (
            "  assign y = {'1};",
            "'1",
            "unsized literal `'1` in a concatenation",
        ),
        (
            "  assign y = {2{'d1}};",
            "'d1",
            "unsized literal `'d1` in a concatenation",
        ),
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
    ];
    for (body, needle, msg) in cases {
        rejects(None, body, 2, needle, msg);
    }
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
    rejects(None, "  tri [7:0] t;", 2, "tri", "net type `tri`");
    rejects(
        None,
        "  logic [7:0] t [4][2];",
        2,
        "[2]",
        "arrays with several unpacked dimensions not supported",
    );
}

#[test]
fn rejected_items() {
    rejects(None, "  assign #1 y = a;", 2, "#1", "delay");
    rejects(
        None,
        "  initial $display(a);\n  assign y = a;",
        2,
        "$display",
        "`initial` may only give signals initial values (`q = 8'd0;`) and load arrays",
    );
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

/// Clocks derived from registers, directly and as copies or bits.
#[test]
fn accepted_derived_clocks() {
    // a divider; the register on the derived clock reads one of the input
    // clock and the derived clock itself
    let t = translate_ff(
        "  logic s;\n  logic [7:0] q;\n  always_ff @(posedge s) y <= q + {7'd0, s};\n  \
         always_ff @(posedge clk) begin s <= !s; q <= a; end",
    );
    assert_eq!(t.clocks, ["clk"]);
    assert_eq!(t.registers, ["y", "s", "q"]);
    // the block on `s` comes after the one updating `s`, though written
    // first, and awaits `s`
    let atoms = t.module.atoms();
    let (s, y) = (t.var("s"), t.var("y"));
    let pos = |v| atoms.iter().position(|a| a.ctrl().contains(&v)).unwrap();
    assert!(pos(s) < pos(y));
    assert!(atoms[pos(y)].wait().contains(&s));
    // a bit of a counter, a copy of a copy, a copy of the input clock
    let t = translate_ff(
        "  logic [3:0] n;\n  logic c, d, k;\n  assign c = n[3];\n  assign d = c;\n  \
         assign k = clk;\n  always_ff @(negedge d) y <= a;\n  always_ff @(posedge k) n <= n + 4'd1;",
    );
    assert_eq!(t.clocks, ["clk"]);
    // two stages
    let t = translate_ff(
        "  logic s1, s2;\n  always_ff @(posedge s2) y <= y + 8'd1;\n  \
         always_ff @(negedge s1) s2 <= !s2;\n  always_ff @(posedge clk) s1 <= !s1;",
    );
    assert_eq!(t.clocks, ["clk"]);
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
            "clk)",
            "`clk` appears twice in the event list",
        ),
        (
            "  always_ff @(posedge clk or posedge a) y <= a;",
            "y <=",
            "a block with two events needs an asynchronous reset: its body must be a single `if`",
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
    // a clock must have 1 bit, and a bit of a wider input is no clock
    rejects(
        Some(HEADER_FF),
        "  always_ff @(posedge y) y <= a;",
        2,
        "y)",
        "the clock `y` must have 1 bit",
    );
    rejects(
        Some(HEADER_FF),
        "  logic c;\n  assign c = a[0];\n  always_ff @(posedge c) y <= a;",
        4,
        "c)",
        "the clock `c` is a bit of the input `a`: input clocks must be 1-bit inputs",
    );
    // a copy of a register made in an `always_comb`, a gated clock behind
    // a copy, a register clocked by a gated clock
    rejects(
        Some(HEADER_FF),
        "  logic s, c;\n  always_ff @(posedge clk) s <= !s;\n  always_comb c = s;\n  \
         always_ff @(posedge c) y <= a;",
        5,
        "c)",
        "the clock `c` is computed combinationally: gated clocks are not supported",
    );
    rejects(
        Some(HEADER_FF),
        "  logic g, c;\n  assign g = clk & a[0];\n  assign c = g;\n  always_ff @(posedge c) y <= a;",
        5,
        "c)",
        "the clock `c` comes from `g`, computed combinationally at 3:10: gated clocks are not \
         supported",
    );
    rejects(
        Some(HEADER_FF),
        "  logic g, s;\n  assign g = clk & a[0];\n  always_ff @(posedge g) s <= !s;\n  \
         always_ff @(posedge s) y <= a;",
        4,
        "g)",
        "the clock `g` is computed combinationally",
    );
    // clock loops: a register clocked by itself, directly, through a copy,
    // and through another register
    rejects(
        Some(HEADER_FF),
        "  logic s;\n  always_ff @(posedge s) s <= !s;\n  assign y = a;",
        3,
        "always_ff",
        "combinational loop: s -> s (through the clock of the `always_ff` at 3:3)",
    );
    rejects(
        Some(HEADER_FF),
        "  logic s, c;\n  assign c = s;\n  always_ff @(negedge c) s <= !s;\n  assign y = a;",
        3,
        "c =",
        "combinational loop: c -> s -> c (through the clock of the `always_ff` at 4:3)",
    );
    rejects(
        Some(HEADER_FF),
        "  logic s1, s2;\n  always_ff @(posedge s2) s1 <= !s1;\n  \
         always_ff @(posedge s1) s2 <= !s2;\n  assign y = a;",
        3,
        "always_ff",
        "combinational loop: s1 -> s2 -> s1 (through the clock of the `always_ff` at 3:3)",
    );
    rejects(
        Some(HEADER_FF),
        "  logic [3:0] n;\n  always_ff @(posedge n[2]) n <= n + 4'd1;\n  assign y = a;",
        3,
        "n[2]",
        "an event must be on a signal name",
    );
    rejects(
        Some(HEADER_FF),
        "  logic [3:0] n;\n  logic c;\n  assign c = n[2];\n  \
         always_ff @(posedge c) n <= n + 4'd1;\n  assign y = a;",
        4,
        "c =",
        "combinational loop: c -> n -> c (through the clock of the `always_ff` at 5:3)",
    );
    rejects(
        Some(HEADER_FF),
        "  always_ff @(posedge y[0]) y <= a;",
        2,
        "y[0]",
        "an event must be on a signal name",
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
    // a derived clock is in the domain of its input clock, both ways
    rejects(
        two,
        "  logic s;\n  always_ff @(posedge ca) s <= !s;\n  always_ff @(posedge s) y <= a;\n  \
         always_ff @(posedge cb) z <= y;",
        5,
        "y;",
        "`z` (clocked by `cb`) reads `y`, which depends on registers clocked by `ca` or by \
         clocks derived from it: crossings between unrelated clocks are not supported yet",
    );
    rejects(
        two,
        "  logic s;\n  always_ff @(negedge ca) s <= !s;\n  always_ff @(posedge s) y <= z;\n  \
         always_ff @(posedge cb) z <= a;",
        4,
        "z;",
        "`y` (clocked by `s`, derived from `ca`) reads `z`, which depends on registers clocked \
         by `cb`: crossings",
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
    let cases: [(&str, usize, &str, &str); 13] = [
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
    // and the index of a target: which bits of `x` are written depends on `z`
    let src = "module m(input logic [7:0] a, output logic [7:0] x, y);\n  logic [7:0] z;\n  always_comb begin y = a; x = a; x[z[2:0]] = 1'b0; end\n  always_comb z = x;\nendmodule\n";
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
        // implicit resizing: widening casts of operator expressions, mixed
        // signedness, truncation, constant divisors
        "  assign w = 16'(a + a);",
        "  assign w = 16'(-a);",
        "  assign y = s + a[7:0];",
        "  assign y = s + {s};",
        "  assign y = a / -8'd1 + a / 2 + a % (3 - 1);",
        "  assign y = {a, a};",
        "  assign o = a[i + 3'd1];",
        // only a literal itself is no item of a concatenation
        "  assign w = {a, 1 + 1};",
    ];
    for body in ok {
        translate_with(HEADER_SIZING, &drive_others(body));
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
            "  assign o = a[i + 1];",
            "a[",
            "the index has 32 bits, so every one of its values must be in range (narrow the \
             index); an unsized literal or an integer parameter makes the index 32 bits wide",
        ),
        (
            "  assign y = a / a;",
            "a;",
            "divisor must be a nonzero constant",
        ),
        (
            "  assign y = a % 8'd0;",
            "8'd0",
            "divisor must be a nonzero constant",
        ),
        (
            "  assign y = a / (2 - 2);",
            "- 2",
            "divisor must be a nonzero constant",
        ),
        ("  assign w = {0{a}};", "{0", "replication count 0"),
        ("  assign o = a[0 +: 0];", "a[", "of width 0"),
        (
            "  assign y = $signed(a, a);",
            "$signed",
            "takes one argument",
        ),
        ("  assign y = string'(a);", "string", "cast to `string`"),
    ];
    for (body, needle, msg) in cases {
        rejects(Some(HEADER_SIZING), &drive_others(body), 2, needle, msg);
    }
}

/// `body` with the outputs of [`HEADER_SIZING`] it leaves alone driven
/// (after it, so that its line stays the second).
fn drive_others(body: &str) -> String {
    let mut src = body.to_string();
    for (out, value) in [("y", "a"), ("w", "{a, a}"), ("o", "a[0]")] {
        if !body.contains(&format!("assign {out} ")) {
            src += &format!("\n  assign {out} = {value};");
        }
    }
    src
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

/// Targets with a variable index: in blocks, any unsigned index (bits
/// outside the signal are not written), alone in their signal in a
/// concatenation; in `always_comb` after the whole signal is assigned.
#[test]
fn accepted_variable_targets() {
    let header = "module m(input logic clk, input logic [7:0] a, input logic [2:0] i, \
                  input logic [5:0] k, input logic e, output logic [7:0] y, q, \
                  output logic c, output logic [6:0] d);";
    let others = "\n  assign {c, d} = a;";
    translate_with(
        header,
        &format!(
            "  always_ff @(posedge clk) begin q[i] <= e; q[k +: 3] <= a[2:0]; \
             q[k -: 5] <= a[4:0]; q[i + 1] <= e; end\n\
             \x20 always_comb begin y = a; y[i] = e; y[y[2:0]] = 1'b0; end{others}"
        ),
    );
    // a select of another signal in a concatenation, and a whole-width one
    translate_with(
        header,
        "  always_ff @(posedge clk) {c, q[i +: 2], d[k]} <= {e, a[1:0], e};\n\
         \x20 always_comb begin y = 8'd0; if (e) y[k[0] +: 8] = a; end",
    );
    // after a full assignment on every path
    translate_with(
        header,
        &format!("  always_comb begin if (e) y = a; else y = '1; y[i] = e; q = y; end{others}"),
    );
}

#[test]
fn rejected_targets() {
    let header = "module m(input logic clk, input logic [7:0] a, input logic [2:0] i, \
                  input logic signed [2:0] s, input logic e, output logic [7:0] y, q);";
    let cases: &[(&str, usize, &str, &str)] = &[
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
            "  always_comb begin y[i] = 1'b0; end\n  assign q = a;",
            2,
            "y[i]",
            "`y[i]` has a variable index, so it assigns no bit for sure and keeps the \
             others, but `y` may not be assigned yet (a latch): assign all of `y` before it \
             on every path, e.g. `y = '0;`",
        ),
        (
            "  always_comb begin y[3:0] = a[3:0]; y[i] = 1'b0; end\n  assign q = a;",
            2,
            "y[i]",
            "but `y[7:4]` may not be assigned yet",
        ),
        (
            "  always_comb begin if (e) y = a; y[i] = 1'b0; end\n  assign q = a;",
            2,
            "y[i]",
            "but `y` may not be assigned yet",
        ),
        (
            "  always_comb begin y = a; q[y[2:0]] = 1'b0; q = a; end",
            2,
            "q[y",
            "`q` may not be assigned yet",
        ),
        (
            "  always_comb begin q = a; y[q[2:0]] = 1'b0; y = a; end",
            2,
            "y[q",
            "`y` may not be assigned yet",
        ),
        (
            "  always_comb begin q = a; q[y[2:0]] = 1'b0; y = a; end",
            2,
            "y[2:0]",
            "`y[2:0]` read before it is assigned on every path",
        ),
        (
            "  assign y[i] = 1'b0;\n  assign q = a;",
            2,
            "i]",
            "a continuous assignment needs a constant select",
        ),
        (
            "  always_ff @(posedge clk) y[s] <= e;\n  assign q = a;",
            2,
            "y[s]",
            "the index of `y[s]` must be unsigned",
        ),
        (
            "  always_ff @(posedge clk) y[z] <= e;\n  assign q = a;",
            2,
            "z]",
            "`z` is not declared",
        ),
        (
            "  always_ff @(posedge clk) y[i +: 9] <= a;\n  assign q = a;",
            2,
            "y[i",
            "`y[i +: 9]` selects 9 bits of `y`, which has 8",
        ),
        (
            "  always_ff @(posedge clk) {y[i], y[0]} <= 2'd0;\n  assign q = a;",
            2,
            "y[0]",
            "`y` assigned twice in one target: a select or element of `y` with a variable \
             index may overlap its other pieces",
        ),
        (
            "  always_ff @(posedge clk) {y[7:4], y[i]} <= 5'd0;\n  assign q = a;",
            2,
            "y[i]",
            "`y` assigned twice in one target",
        ),
        (
            "  always_ff @(posedge clk) {y[i], y[i]} <= 2'd0;\n  assign q = a;",
            2,
            "y[i]}",
            "`y` assigned twice in one target",
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
    for &(body, line, needle, msg) in cases {
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

const HEADER_RST: &str = "module m(input logic clk, input logic rst, input logic rst_n, \
                          input logic [7:0] a, output logic [7:0] y);";

fn translate_rst(body: &str) -> Translation {
    let src = format!("{HEADER_RST}\n{body}\nendmodule\n");
    frontend::parse(&src, "test.sv")
        .and_then(|m| lower::lower(&m))
        .unwrap_or_else(|e| panic!("{src}\n{e}"))
}

/// The names of the variables the only register atom of `t` awaits,
/// sorted.
fn register_awaits(t: &Translation) -> Vec<&str> {
    let atom = t
        .module
        .atoms()
        .iter()
        .find(|a| a.ctrl().contains(&t.var("y")))
        .unwrap();
    let mut names: Vec<&str> = atom.wait().iter().map(|v| t.name(v)).collect();
    names.sort();
    names
}

/// Asynchronous resets: either order of the events, `or` or `,`, both
/// polarities, registers left out of the reset branch, constant reset
/// values, and resets that are registers or copies.
#[test]
fn accepted_resets() {
    let t =
        translate_rst("  always_ff @(posedge rst or posedge clk) if (rst) y <= 8'd0; else y <= a;");
    assert_eq!(t.clocks, ["clk"]);
    assert_eq!(t.reset_registers, ["y"]);
    assert_eq!(register_awaits(&t), ["clk", "rst", "y@init"]);
    let printed = t.print();
    assert!(
        printed.contains("X(y) := ITE (X(rst), %1, X(y@init))"),
        "{printed}"
    );
    // active low, `~`, parentheses, a comma, a register not reset, a
    // reset value computed from constants
    let t = translate_rst(
        "  logic [7:0] z;\n  always_ff @(negedge clk, negedge rst_n)\n    \
         if ((~rst_n)) z <= 8'd2 + 8'd3; else begin z <= a; y <= z; end",
    );
    assert_eq!(t.reset_registers, ["z"]);
    let printed = t.print();
    assert!(printed.contains("NOT X(rst_n)"), "{printed}");
    assert!(printed.contains("X(y) := ID X(y@init)"), "{printed}");
    // a reset branch resetting nothing: no reset in the atom
    let t = translate_rst("  always_ff @(posedge clk or posedge rst) if (rst) ; else y <= a;");
    assert!(t.reset_registers.is_empty());
    assert_eq!(register_awaits(&t), ["clk", "y@init"]);
    // a register as reset, through a copy; the atom of the register comes
    // first though written after
    let t = translate_rst(
        "  logic r, c;\n  assign c = r;\n  always_ff @(posedge clk or posedge c) \
         if (c) y <= 8'd0; else y <= y + 8'd1;\n  always_ff @(posedge clk) r <= rst;",
    );
    assert_eq!(register_awaits(&t), ["c", "clk", "y@init"]);
    let atoms = t.module.atoms();
    let pos = |v| atoms.iter().position(|a| a.ctrl().contains(&v)).unwrap();
    assert!(pos(t.var("r")) < pos(t.var("c")));
    assert!(pos(t.var("c")) < pos(t.var("y")));
    // a bit of a wider input, a reset also read as data
    let t = translate_rst(
        "  logic r;\n  assign r = a[7];\n  always_ff @(posedge clk or posedge r) \
         if (r) y <= 8'd1; else y <= {a[6:0], r};",
    );
    assert_eq!(register_awaits(&t), ["clk", "r", "y@init"]);
}

#[test]
fn rejected_resets() {
    let cases = [
        (
            "  always_ff @(posedge clk or posedge rst or negedge rst_n) y <= a;",
            "negedge rst_n",
            "at most two events supported: a clock and an asynchronous reset",
        ),
        (
            "  always_ff @(posedge clk or posedge rst) begin y <= a; if (rst) y <= 8'd0; end",
            "begin",
            "its body must be a single `if` testing one of the events",
        ),
        (
            "  always_ff @(posedge clk or posedge rst) if (rst == 1'b1) y <= 8'd0; else y <= a;",
            "if",
            "its body must be a single `if` testing one of the events",
        ),
        (
            "  always_ff @(posedge clk or posedge rst) if (!rst_n) y <= 8'd0; else y <= a;",
            "!rst_n",
            "its `if` must test `clk` or `rst`",
        ),
        (
            "  always_ff @(posedge clk or negedge rst_n) if (rst_n) y <= a; else y <= 8'd0;",
            "rst_n) y",
            "the block resets on `negedge rst_n`, so its `if` must test `!rst_n`",
        ),
        (
            "  always_ff @(posedge clk or posedge rst) if (!rst) y <= a; else y <= 8'd0;",
            "!rst",
            "the block resets on `posedge rst`, so its `if` must test `rst`",
        ),
        (
            "  always_ff @(posedge clk or posedge a) if (a) y <= 8'd0; else y <= a;",
            "a)",
            "the reset `a` must have 1 bit",
        ),
        (
            "  always_ff @(posedge clk or posedge rst) if (rst) y[0] <= 1'b0; else y <= a;",
            "y[0]",
            "the reset branch may only assign whole registers",
        ),
        (
            "  always_ff @(posedge clk or posedge rst) if (rst) y <= a; else y <= 8'd0;",
            "a;",
            "the reset value of `y` must be a constant, it reads `a`",
        ),
        (
            "  always_ff @(posedge clk or posedge rst) if (rst) begin if (rst_n) y <= 8'd0; end \
             else y <= a;",
            "if (rst_n)",
            "the reset branch may only assign constants to registers",
        ),
        (
            "  always_ff @(posedge clk or posedge rst) if (rst) begin y <= 8'd0; y <= 8'd1; end \
             else y <= a;",
            "y <= 8'd1",
            "`y` is reset twice",
        ),
    ];
    for (body, needle, msg) in cases {
        rejects(Some(HEADER_RST), body, 2, needle, msg);
    }
    // a computed reset
    rejects(
        Some(HEADER_RST),
        "  logic r;\n  assign r = rst & rst_n;\n  always_ff @(posedge clk or posedge r) \
         if (r) y <= 8'd0; else y <= a;",
        4,
        "r)",
        "the reset `r` is computed combinationally: computed asynchronous resets are not supported",
    );
    rejects(
        Some(HEADER_RST),
        "  logic r, c;\n  always_comb c = r;\n  always_ff @(posedge clk) r <= rst;\n  \
         always_ff @(posedge clk or posedge c) if (c) y <= 8'd0; else y <= a;",
        5,
        "c)",
        "the reset `c` is computed combinationally",
    );
    // a register reset by itself, directly or through a copy
    rejects(
        Some(HEADER_RST),
        "  logic r;\n  always_ff @(posedge clk or posedge r) if (r) r <= 1'b0; else r <= 1'b1;\n  \
         assign y = a;",
        3,
        "always_ff",
        "combinational loop: r -> r (through the reset of the `always_ff` at 3:3)",
    );
    rejects(
        Some(HEADER_RST),
        "  logic [7:0] n;\n  logic c;\n  assign c = n[7];\n  always_ff @(posedge clk or posedge c) \
         if (c) n <= 8'd0; else n <= n + 8'd1;\n  assign y = a;",
        4,
        "c =",
        "combinational loop: c -> n -> c (through the reset of the `always_ff` at 5:3)",
    );
    // a reset from another clock domain: releasing it is a crossing
    rejects(
        Some(
            "module m(input logic clk, input logic clk_b, input logic rst, \
             input logic [7:0] a, output logic [7:0] y);",
        ),
        "  logic r;\n  always_ff @(posedge clk_b) r <= rst;\n  \
         always_ff @(posedge clk or posedge r) if (r) y <= 8'd0; else y <= a;",
        4,
        "r) y",
        "crossings between unrelated clocks are not supported yet",
    );
}

const HEADER_ARRAY: &str = "module m(input logic clk, input logic [7:0] a, input logic [2:0] i, \
                            input logic [1:0] j, input logic signed [1:0] s, output logic [7:0] y);";

/// A file with `text` for `$readmemb`/`$readmemh`, by absolute path.
fn mem_file(name: &str, text: &str) -> String {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("readmem");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

#[test]
fn accepted_arrays() {
    // the three forms of the range, parameters in it and in indices,
    // selects of elements, reads of the block's own elements
    let t = translate_with(
        "module m #(parameter N = 4) (input logic clk, input logic [1:0] j, \
         input logic [7:0] a, output logic [7:0] y, output logic [3:0] z);",
        "  logic [7:0] p [0:N-1];\n  logic [7:0] q [N-1:0];\n  logic [7:0] r [N];\n\
         \x20 always_ff @(posedge clk) begin p[j] <= a; p[N-1][j] <= a[0]; end\n\
         \x20 always_comb begin q[0] = a; q[1] = p[j]; q[2] = q[1]; q[3] = '0; q[j][3:0] = a[3:0]; end\n\
         \x20 assign {r[3], r[2], r[1], r[0]} = {4{q[j]}};\n\
         \x20 assign y = r[j] ^ p[N-2] ^ r[2'd1];\n  assign z = q[j][j +: 4];",
    );
    let names: Vec<&str> = t.vars.keys().map(|k| k.as_str()).collect();
    for x in ["p[0]", "p[3]", "p[3]@init", "q[2]", "r[0]"] {
        assert!(names.contains(&x), "{names:?}");
    }
    // a ROM: no variables for its elements; addresses are hexadecimal
    let file = mem_file(
        "accepted.mem",
        "// header\n@1 0f /* skip */ 1_0\n@3\n3 4 5 6 7 8 9 a b c d e f\n@10 aa @0\nff\n",
    );
    let t = translate_with(
        HEADER_ARRAY,
        &format!(
            "  logic [7:0] t [17];\n  initial begin\n    $readmemh(\"{file}\", t);\n  end\n\
             \x20 assign y = t[j[0]] + t[16];"
        ),
    );
    assert!(!t.vars.keys().any(|k| k.starts_with("t[")));
}

#[test]
fn rejected_arrays() {
    let ok = mem_file("ok.mem", "1\n10\n11\n100\n");
    let cases: Vec<(String, usize, &str, &str)> = vec![
        (
            "  logic [7:0] t [4];\n  assign y = t;".into(),
            3,
            "t;",
            "`t` is an array: only its elements can be used",
        ),
        (
            "  logic [7:0] t [4];\n  assign y = t[1:0];".into(),
            3,
            "t[",
            "`t` is an array: only its elements can be used",
        ),
        (
            "  logic [7:0] t [1:4];\n  assign y = a;".into(),
            2,
            "[1:4]",
            "array range `[1:4]` not supported: addresses must start at 0",
        ),
        (
            "  logic [7:0] t [4];\n  logic t;\n  assign y = a;".into(),
            3,
            "t;",
            "`t` already declared at 2:15",
        ),
        (
            "  logic [7:0] t [4];\n  always_ff @(posedge clk) t[j] <= a;\n  assign y = t[i];"
                .into(),
            4,
            "t[i]",
            "`t[i]` may address beyond the last element of `t` (3): the index has 3 bits",
        ),
        (
            "  logic [7:0] t [4];\n  always_ff @(posedge clk) t[j] <= a;\n  assign y = t[s];"
                .into(),
            4,
            "t[s]",
            "the index of `t[s]` must be unsigned",
        ),
        (
            "  logic [7:0] t [4];\n  always_ff @(posedge clk) t[s] <= a;\n  assign y = t[0];"
                .into(),
            3,
            "t[s]",
            "the index of `t[s]` must be unsigned",
        ),
        (
            "  logic [7:0] t [4];\n  always_ff @(posedge clk) t[j] <= a;\n  assign y = t[4];"
                .into(),
            4,
            "t[4]",
            "`t[4]` out of range: `t` has addresses 0 to 3",
        ),
        (
            "  logic [7:0] t [4];\n  always_ff @(posedge clk) t[j] <= a;\n  assign y = t[j][8:1];"
                .into(),
            4,
            "t[j]",
            "`t[j][8:1]` out of range: `t[j]` has bits 7:0",
        ),
        (
            "  logic [7:0] t [4];\n  always_ff @(posedge clk) t[j] <= a;\n  assign t[0] = a;\n  \
             assign y = t[1];"
                .into(),
            4,
            "t[0]",
            "`t[0]`: the array `t` is already driven at 3:28 (an array has one driver)",
        ),
        (
            "  logic [7:0] t [4];\n  always_ff @(posedge clk) t[0] <= a;\n  assign y = t[1];"
                .into(),
            2,
            "t ",
            "`t[1]` is never driven",
        ),
        (
            "  logic [7:0] t [4];\n  assign t[j] = a;\n  assign y = t[1];".into(),
            3,
            "j]",
            "a continuous assignment needs a constant select",
        ),
        (
            "  logic [7:0] t [4];\n  always_ff @(posedge clk) {t[j], t[0]} <= {a, a};\n  \
             assign y = t[1];"
                .into(),
            3,
            "t[0]",
            "`t` assigned twice in one target: a select or element of `t` with a variable index",
        ),
        (
            "  logic [7:0] t [4];\n  always_comb begin t[0] = a; t[j] = a; end\n  assign y = t[1];"
                .into(),
            3,
            "t[j]",
            "`t[j]` has a variable index, so it assigns no element for sure and keeps the \
             others, but `t[1]`, `t[2]`, `t[3]` may not be assigned yet (a latch): assign \
             every element of `t` before it on every path",
        ),
        (
            format!(
                "  logic [7:0] t [4];\n  initial $readmemb(\"{ok}\", t);\n  \
                 initial t[2] = 8'd1;\n  assign y = t[1];"
            ),
            4,
            "t[2]",
            "`t` is loaded at 3:11, and nothing writes it: its elements are constants",
        ),
        (
            "  logic [7:0] t [4];\n  initial $readmemb(\"missing.mem\", t);\n  assign y = t[1];"
                .into(),
            3,
            "\"missing",
            "cannot read `missing.mem`",
        ),
        (
            format!(
                "  logic [7:0] t [4];\n  initial $readmemb(\"{}\", t);\n  assign y = t[1];",
                mem_file("xz.mem", "0\n1\n1x\n0\n")
            ),
            3,
            "$readmemb",
            "line 3: `1x` has x/z digits (not supported: values are 2-state)",
        ),
        (
            format!(
                "  logic [7:0] t [4];\n  initial $readmemb(\"{}\", t);\n  assign y = t[1];",
                mem_file("few.mem", "0\n1\n\n10\n")
            ),
            3,
            "$readmemb",
            "no value for `t[3]` (every element of an array that is never written needs one: it \
             is a ROM)",
        ),
        (
            format!(
                "  logic [7:0] t [4];\n  initial $readmemh(\"{}\", t);\n  assign y = t[1];",
                mem_file("many.mem", "0 1 2\n3 4\n")
            ),
            3,
            "$readmemh",
            "line 2: `4` would go to address 4, but `t` has addresses 0 to 3",
        ),
        (
            format!(
                "  logic [7:0] t [4];\n  initial $readmemh(\"{}\", t);\n  assign y = t[1];",
                mem_file("wide.mem", "0 1 2 1ff\n")
            ),
            3,
            "$readmemh",
            "line 1: `1ff` does not fit the 8 bits of an element of `t`",
        ),
        (
            format!("  initial $readmemb(\"{ok}\", y);\n  assign y = a;"),
            2,
            "y)",
            "`y` is not an array declared before",
        ),
        (
            format!(
                "  logic [7:0] t [4];\n  initial $readmemb(\"{ok}\", t, 0, 3);\n  assign y = t[1];"
            ),
            3,
            "$readmemb",
            "`$readmemb` with start and end addresses not supported",
        ),
        (
            format!(
                "  logic [7:0] t [4];\n  initial begin $readmemb(\"{ok}\", t); \
                 $readmemh(\"{ok}\", t); end\n  assign y = t[1];"
            ),
            3,
            "$readmemh",
            "`t` is already loaded at 3:17",
        ),
        (
            format!(
                "  logic [7:0] t [4];\n  initial begin $readmemb(\"{ok}\", t); y = a; end\n  \
                 assign y = t[1];"
            ),
            3,
            "a; end",
            "the initial value of `y` must be a constant, it reads `a`",
        ),
        (
            "  initial begin logic x; end\n  assign y = a;".into(),
            2,
            "logic x",
            "`initial` may only give signals initial values",
        ),
    ];
    for (body, line, needle, msg) in &cases {
        rejects(Some(HEADER_ARRAY), body, *line, needle, msg);
    }
    // arrays as ports
    rejects(
        Some("module m(input logic [7:0] a [2], output logic [7:0] y);"),
        "  assign y = a[0];",
        1,
        "[2]",
        "unpacked dimension not supported",
    );
}

/// The init action of the atom controlling `q`, printed.
fn init_of(t: &Translation, q: &str) -> String {
    let text = t.print();
    let start = text
        .match_indices("atom controls ")
        .map(|(k, _)| k)
        .find(|k| {
            let line = text[*k..].lines().next().unwrap();
            let ctrl = line["atom controls ".len()..]
                .split(" reads")
                .next()
                .unwrap();
            ctrl.split(" awaits")
                .next()
                .unwrap()
                .split(", ")
                .any(|x| x == q)
        })
        .unwrap_or_else(|| panic!("no atom of `{q}` in\n{text}"));
    let atom = &text[start..];
    let init = &atom[atom.find("init").unwrap()..];
    init[..init.find("update").unwrap_or(init.len())].to_string()
}

#[test]
fn accepted_initial_values() {
    // declaration, `initial`, both in `begin … end`, sized as assignments
    // (`-1` is extended by its sign); `x` everywhere is no initial value
    let t = translate_ff(
        "  logic [7:0] p = -1, r;\n  logic [3:0] s = 4'b1010;\n  logic [7:0] u;\n  \
         initial r = 8'd3;\n  initial begin u = 'x; y = 2 + 3; end\n  \
         always_ff @(posedge clk) begin p <= a; r <= p; s <= a[3:0]; u <= a; y <= {s, s}; end",
    );
    assert_eq!(t.initialized, ["p", "r", "s", "y"]);
    let names: Vec<&str> = t.vars.keys().map(|k| k.as_str()).collect();
    assert!(
        names.contains(&"u@init") && !names.contains(&"p@init"),
        "{names:?}"
    );
    let init = init_of(&t, "p");
    for line in [
        "X(p) := 8'd255",
        "X(r) := 8'd3",
        "X(s) := 4'd10",
        "X(u) := ID X(u@init)",
        "X(y) := 8'd5",
    ] {
        assert!(init.contains(line), "{line} in\n{init}");
    }
    // an initial value of an output; a 1-bit x literal on a 1-bit register
    let t = translate_with(
        "module m(input logic clk, output logic [3:0] y = 4'd7, output logic z);",
        "  initial z = 1'bx;\n  always_ff @(posedge clk) begin y <= y + 4'd1; z <= !z; end",
    );
    assert_eq!(t.initialized, ["y"]);
    assert!(t.vars.contains_key("z@init"));
    // with an asynchronous reset, the reset value wins in the first round
    let t = translate_ff(
        "  logic r = 1'b1;\n  logic [7:0] q = 8'd9;\n  \
         always_ff @(posedge clk or posedge r) if (r) q <= 8'd0; else q <= a;\n  \
         always_ff @(posedge clk) r <= 1'b0;\n  assign y = q;",
    );
    assert!(
        init_of(&t, "q").contains("X(q) := ITE (X(r), %2, %1)"),
        "{}",
        init_of(&t, "q")
    );
    // a constant: initialized and never written; also an output
    let t = translate_with(
        "module m(input logic [3:0] i, output logic b, output logic [3:0] k = 4'd5);",
        "  logic [15:0] c = {15'h392b, 1'b0};\n  assign b = c[i];",
    );
    assert!(t.registers.is_empty() && t.initialized.is_empty());
    assert!(t.print().contains("X(k) := 4'd5"), "{}", t.print());
    // `wire y = e;` is `assign y = e;`
    let t = translate_with(HEADER, "  wire [8:0] t = a + 9'd1;\n  assign y = t[8:1];");
    assert_eq!(t.module.atoms().len(), 2);
    // a RAM preloaded from a file (address 2 left out) and given element
    // values; with initial values ignored, every element is free
    let file = mem_file("preload.mem", "1 10\n@3 11\n");
    let body = format!(
        "  logic [1:0] t [4];\n  logic [7:0] v [2];\n  initial $readmemb(\"{file}\", t);\n  \
         initial v[1] = 8'd7;\n  \
         always_ff @(posedge clk) begin t[j] <= a[1:0]; v[j[0]] <= a; y <= {{v[i[0]], 6'd0}} | t[j]; end"
    );
    let t = translate_with(HEADER_ARRAY, &body);
    assert_eq!(t.initialized, ["t[0]", "t[1]", "t[3]", "v[1]"]);
    let src = format!("{HEADER_ARRAY}\n{body}\nendmodule\n");
    let mut inst = elab::load(&src, "test.sv").unwrap();
    assert_eq!(inst.initialized(), t.initialized);
    inst.ignore_initializers();
    let t = lower::lower_inst(&inst).unwrap();
    assert!(t.initialized.is_empty());
    assert!(t.vars.contains_key("t[0]@init"));
}

#[test]
fn rejected_initial_values() {
    let cases: &[(&str, usize, &str, &str)] = &[
        (
            "  logic [7:0] t = 8'd1;\n  assign t = a;\n  assign y = t;",
            2,
            "t =",
            "`t` is driven by an `assign` at 3:10, which computes it in every time step: an \
             initial value has no effect on it",
        ),
        (
            "  initial y = 8'd0;\n  always_comb y = a;",
            2,
            "y",
            "`y` is driven by an `always_comb` at 3:15",
        ),
        (
            "  initial a = 8'd0;\n  assign y = a;",
            2,
            "a =",
            "`a` is an input: only signals of the module have initial values",
        ),
        (
            "  logic [7:0] t = a;\n  always_ff @(posedge clk) t <= a;\n  assign y = t;",
            2,
            "a;",
            "the initial value of `t` must be a constant, it reads `a`",
        ),
        (
            "  logic [7:0] t = 8'd1;\n  initial t = 8'd2;\n  always_ff @(posedge clk) t <= a;\n  \
             assign y = t;",
            3,
            "t =",
            "`t` already has an initial value at 2:15",
        ),
        (
            "  logic [7:0] t = 'x;\n  initial t = 8'd2;\n  always_ff @(posedge clk) t <= a;\n  \
             assign y = t;",
            3,
            "t =",
            "`t` already has an initial value at 2:15",
        ),
        (
            "  initial y <= 8'd0;\n  always_ff @(posedge clk) y <= a;",
            2,
            "y <=",
            "`initial` gives initial values with blocking assignments (`q = 8'd0;`), not `<=`",
        ),
        (
            "  initial y[3:0] = 4'd0;\n  always_ff @(posedge clk) y <= a;",
            2,
            "y[3:0]",
            "initial values are given to whole signals and elements",
        ),
        (
            "  initial y = 4'bxxxx;\n  always_ff @(posedge clk) y <= a;",
            2,
            "y",
            "the initial value of `y` has 4 x bits, but `y` has 8 bits",
        ),
        (
            "  initial y = 8'b0000_xxxx;\n  always_ff @(posedge clk) y <= a;",
            2,
            "8'b",
            "x",
        ),
        (
            "  initial y += 8'd1;\n  always_ff @(posedge clk) y <= a;",
            2,
            "+=",
            "`+=` not supported",
        ),
        (
            "  initial begin if (a) y = 8'd1; end\n  always_ff @(posedge clk) y <= a;",
            2,
            "if",
            "`initial` may only give signals initial values",
        ),
    ];
    for (body, line, needle, msg) in cases {
        rejects(Some(HEADER_FF), body, *line, needle, msg);
    }
    let ok = mem_file("init.mem", "1 2 3 4\n");
    let few = mem_file("init_few.mem", "1 2\n");
    let cases: Vec<(String, usize, &str, &str)> = vec![
        (
            "  logic [7:0] t [4] = '{8'd0, 8'd1, 8'd2, 8'd3};\n  assign y = t[j];".into(),
            2,
            "'{",
            "initial values of whole arrays not supported",
        ),
        (
            "  logic [7:0] t [4];\n  initial t[j] = 8'd0;\n  always_ff @(posedge clk) t[j] <= a;\n  \
             assign y = t[1];"
                .into(),
            3,
            "t[j]",
            "the initial value of an element needs a constant address",
        ),
        (
            "  logic [7:0] t [4];\n  initial t[4] = 8'd0;\n  always_ff @(posedge clk) t[j] <= a;\n  \
             assign y = t[1];"
                .into(),
            3,
            "t[4]",
            "`t[4]` out of range: `t` has addresses 0 to 3",
        ),
        (
            "  logic [7:0] t [2];\n  initial begin t[0] = 8'd0; t[1] = 8'd1; end\n  \
             assign y = t[i[0]];"
                .into(),
            3,
            "t[0]",
            "`t[0]`: nothing writes the array `t`, so its elements are constants: load it from \
             a file",
        ),
        (
            format!(
                "  logic [7:0] t [4];\n  initial $readmemh(\"{ok}\", t);\n  initial t[1] = 8'd0;\n  \
                 always_ff @(posedge clk) t[j] <= a;\n  assign y = t[1];"
            ),
            4,
            "t[1]",
            "`t[1]` already has an initial value at 3:11",
        ),
        (
            format!(
                "  logic [7:0] t [4];\n  initial $readmemh(\"{few}\", t);\n  \
                 assign y = t[j];"
            ),
            3,
            "$readmemh",
            "no value for `t[2]` (every element of an array that is never written needs one",
        ),
        (
            format!(
                "  logic [7:0] t [4];\n  initial $readmemh(\"{ok}\", t);\n  \
                 always_comb begin t[0] = a; t[1] = a; t[2] = a; t[3] = a; end\n  \
                 assign y = t[j];"
            ),
            3,
            "$readmemh",
            "is driven by an `always_comb`",
        ),
    ];
    for (body, line, needle, msg) in &cases {
        rejects(Some(HEADER_ARRAY), body, *line, needle, msg);
    }
    rejects(
        Some("module m(input logic [7:0] a = 8'd0, output logic [7:0] y);"),
        "  assign y = a;",
        1,
        "8'd0",
        "default values of inputs not supported",
    );
}
