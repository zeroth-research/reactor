//! The frontend and typing: operator precedence, ports, and a located error
//! for every rejected construct.

use verilog::ast::{Dir, Item, Ty};
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
        ("  assign y = a / a;", "/", "operator `/`"),
        ("  assign y = a % a;", "%", "operator `%`"),
        ("  assign y = a ** a;", "**", "operator `**`"),
        ("  assign y = a << 8'd1;", "<<", "shift"),
        ("  assign y = a[3:0];", "[3", "bit and part selects"),
        ("  assign y = {a};", "{", "concatenation"),
        ("  assign y = {2{a}};", "{", "replication"),
        ("  assign y = $signed(a);", "$signed", "call `$signed(a)`"),
        ("  assign y = 8'(a);", "8'(", "casts"),
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
    rejects(
        None,
        "  always_comb y = a;",
        2,
        "always_comb",
        "always blocks",
    );
    rejects(None, "  assign #1 y = a;", 2, "#1", "delay");
    rejects(None, "  initial y = a;", 2, "initial", "unsupported");
    rejects(None, "  assign y = ;", 2, ";", "syntax error");
}
