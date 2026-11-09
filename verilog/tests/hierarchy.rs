//! Parameters, module instances and their errors.

use verilog::ast::{Error, Override};
use verilog::elab;
use verilog::lower::{Translation, lower_inst};
use verilog::{constant, frontend};

fn translate(src: &str) -> Result<Translation, Error> {
    let inst = elab::load(src, "test.sv")?;
    lower_inst(&inst)
}

/// A register and a combinational path, for instantiating.
const CHILD: &str = "
module child #(parameter int W = 4, parameter logic [3:0] K = 4'd1) (
  input  logic         clk,
  input  logic [W-1:0] a,
  output logic [W-1:0] q, y
);
  always_ff @(posedge clk) q <= a;
  assign y = a + W'(K);
endmodule
";

/// Translates `top` (a module, numbered from line 1) together with
/// `CHILD`, and expects an error at the first occurrence of `needle` in line
/// `line` of `top`, with a message containing `msg`.
fn rejects(top: &str, line: usize, needle: &str, msg: &str) {
    let src = format!("{top}{CHILD}");
    let err = match translate(&src) {
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

/// A top module with the usual ports and `body`.
fn top(body: &str) -> String {
    format!(
        "module top(input logic clk, input logic [3:0] a, output logic [3:0] q, y);\n{body}\nendmodule\n"
    )
}

#[test]
fn accepted_instances() {
    // shorthand connections, an omitted output, parameter values
    let t = translate(&format!(
        "{}{CHILD}",
        top("  child #(.K(4'd3)) u (.clk, .a, .q, .y);")
    ))
    .unwrap();
    let printed = t.print();
    assert!(printed.contains("u.q@init : bv4"), "{printed}");
    assert!(printed.contains("4'd3"), "{printed}");
    assert_eq!(t.registers, ["u.q"]);
    // `y` of the instance is unconnected: a private signal `u.y`
    let t = translate(&format!(
        "{}{CHILD}",
        top("  child u (.clk, .a, .q);\n  assign y = q;")
    ))
    .unwrap();
    let printed = t.print();
    assert!(printed.contains("  private\n    u.y : bv4"), "{printed}");
    // an input connected to an expression gets its own signal
    let t = translate(&format!(
        "{}{CHILD}",
        top("  child u (.clk, .a(~a), .q, .y);")
    ))
    .unwrap();
    let printed = t.print();
    assert!(printed.contains("X(u.a) := NOT X(a)"), "{printed}");
}

/// Derived clocks across instances: passed down to a clock port, made in
/// an instance and used by the parent, read across the related domains.
#[test]
fn accepted_derived_clocks() {
    // `u` runs on a divided clock, `v` on the input clock reads `u`'s
    // register
    let t = translate(&format!(
        "{}{CHILD}",
        top(
            "  logic s;\n  logic [3:0] w;\n  always_ff @(posedge clk) s <= !s;\n  \
             child u (.clk(s), .a, .q(w), .y);\n  child v (.clk, .a(w), .q, .y());"
        )
    ))
    .unwrap();
    assert_eq!(t.clocks, ["clk"]);
    // the divider is an instance; its output, a bit of it and a copy are
    // clocks of the parent
    let src = "module top(input logic clk, input logic [3:0] a, output logic [3:0] q, y);
  logic s, b, c;
  logic [1:0] n;
  div d (.c(clk), .s, .n);
  assign b = n[1];
  assign c = b;
  child u (.clk(s), .a, .q, .y());
  always_ff @(negedge c) y <= q;
endmodule
module div(input logic c, output logic s, output logic [1:0] n);
  always_ff @(posedge c) begin s <= !s; n <= n + 2'd1; end
endmodule
";
    let t = translate(&format!("{src}{CHILD}")).unwrap();
    assert_eq!(t.clocks, ["clk"]);
}

#[test]
fn overrides_from_outside() {
    let d = frontend::parse_design(CHILD, "test.sv").unwrap();
    let w = Override {
        name: "W".into(),
        value: constant::int_const(6),
        span: Default::default(),
    };
    let inst = elab::elaborate(&d, "child", &[w]).unwrap();
    let t = lower_inst(&inst).unwrap();
    assert!(t.print().contains("a : bv6"), "{}", t.print());
}

#[test]
fn rejected_instances() {
    let cases = [
        (
            "  child u (.clk, .a, .q, .y);\n  nothing v (.a);",
            "v (",
            "no module `nothing`",
        ),
        (
            "  child u (.clk, .a, .q, .y, .z(a));",
            "z(a",
            "`child` has no port `z`",
        ),
        (
            "  child u (.clk, .a, .a, .q, .y);",
            "a, .q",
            "port `a` connected twice",
        ),
        (
            "  child u (.clk, .q, .y);",
            "u (",
            "input `a` of `u` is not connected",
        ),
        (
            "  child u (.clk, .a(), .q, .y);",
            "u (",
            "input `a` of `u` is not connected",
        ),
        (
            "  child u (clk, a, q, y);",
            "clk, a",
            "positional port connections",
        ),
        ("  child u (.*);", ".*", "`.*` not supported"),
        (
            "  child #(4) u (.clk, .a, .q, .y);",
            "4)",
            "positional parameter values",
        ),
        (
            "  child u (.clk, .a, .q(y[3:0]), .y(q));",
            "y[3:0]",
            "must be connected to a signal name",
        ),
        (
            "  child u (.clk, .a(a[1:0]), .q, .y);",
            "a[1:0]",
            "port `a` of `u` has 4 bits, the signal connected 2",
        ),
        (
            "  child u (.clk, .a, .q, .y);\n  assign y = a;",
            "y = a",
            "`y` already driven",
        ),
        (
            "  child u (.clk, .a, .q, .y);\n  child v (.clk, .a, .q(y), .y(q));",
            "y), ",
            "`y` already driven",
        ),
        (
            "  logic [3:0] y2;\n  child u (.clk(a[0]), .a, .q, .y);",
            "a[0]",
            "must be connected to a clock input",
        ),
        (
            "  logic g;\n  assign g = clk & a[0];\n  child u (.clk(g), .a, .q, .y);",
            "g)",
            "the clock `g` is computed combinationally: gated clocks are not supported",
        ),
        (
            "  logic [3:0] w;\n  logic s;\n  assign s = w[0] ^ w[1];\n  child u (.clk, .a, .q(w), .y);\n  \
             child v (.clk(s), .a, .q, .y());",
            "s)",
            "the clock `s` is computed combinationally",
        ),
        (
            "  child #(.Z(1)) u (.clk, .a, .q, .y);",
            "Z(",
            "`child` has no parameter `Z`",
        ),
        (
            "  child #(.K(4'd1), .K(4'd2)) u (.clk, .a, .q, .y);",
            "K(4'd2",
            "parameter `K` given twice",
        ),
        (
            "  child #(.K(1)) u (.clk, .a, .q, .y);",
            "K(1",
            "parameter `K` of type `logic [3:0]` given a value of 32 bits",
        ),
        (
            "  child #(.W(4'd4)) u (.clk, .a, .q, .y);",
            "W(4'd4",
            "parameter `W` is an `int`",
        ),
        (
            "  defparam u.W = 4;\n  child u (.clk, .a, .q, .y);",
            "defparam",
            "`defparam` not supported",
        ),
        (
            "  child u [1:0] (.clk, .a, .q, .y);",
            "[1:0]",
            "instance array",
        ),
        (
            "  logic u;\n  child u (.clk, .a, .q, .y);",
            "u (",
            "`u` already declared",
        ),
    ];
    for (body, needle, msg) in cases {
        let line = 2 + body.lines().position(|l| l.contains(needle)).unwrap();
        rejects(&top(body), line, needle, msg);
    }
    // a clock driven by an instance output: followed into the instance
    rejects(
        "module top(input logic clk, input logic [3:0] a, output logic [3:0] q, y);
  logic g;
  one o (.y(g));
  child u (.clk(g), .a, .q, .y);
endmodule
module one(output logic y);
  assign y = 1'b1;
endmodule
",
        4,
        "g)",
        "the clock `g` comes from `o.y`, computed combinationally at 7:10: gated clocks are not \
         supported",
    );
    // a clock loop through two dividers (reported without the clock)
    rejects(
        "module top(input logic clk, input logic [3:0] a, output logic [3:0] q, y);
  logic s1, s2;
  div d1 (.c(s2), .s(s1));
  div d2 (.c(s1), .s(s2));
  child u (.clk, .a, .q, .y);
endmodule
module div(input logic c, output logic s);
  always_ff @(posedge c) s <= !s;
endmodule
",
        3,
        "d1",
        "combinational loop through instances: s1 -> s2 -> s1",
    );
    // a register of one clock domain read in another, through instances
    rejects(
        "module top(input logic clk, clk2, input logic [3:0] a, output logic [3:0] q, y);
  logic [3:0] w;
  child u (.clk, .a, .q(w), .y());
  child v (.clk(clk2), .a(w), .q, .y);
endmodule
",
        8 + 4,
        "a;",
        "`q` (clocked by `clk2`) reads `a`, which depends on registers clocked by `clk`: \
         crossings between unrelated clocks are not supported yet (in instance `v`)",
    );
}

#[test]
fn rejected_loops_and_recursion() {
    // a loop through an instance: y = u.y = a + K = x = y
    let src = format!(
        "{}{CHILD}",
        "module top(input logic clk, output logic [3:0] q, y);
  logic [3:0] x;
  assign x = y;
  child u (.clk, .a(x), .q, .y);
endmodule
"
    );
    let err = translate(&src).err().expect("rejected");
    assert!(
        err.msg.contains("combinational loop through instances"),
        "{err}"
    );
    assert_eq!(err.span.line, 4, "{err}");
    // an output fed back into an input of the same instance
    let src = format!("{}{CHILD}", top("  child u (.clk, .a(y), .q, .y);"));
    let err = translate(&src).err().expect("rejected");
    assert!(
        err.msg
            .contains("combinational loop: y -> y (in instance `u`)"),
        "{err}"
    );
    // recursion below the top module
    let src = "module top(input logic a, output logic y);
  r u (.a, .y);
endmodule
module r(input logic a, output logic y);
  r v (.a, .y);
endmodule
";
    let err = translate(src).err().expect("rejected");
    assert!(err.msg.contains("recursive instantiation of `r`"), "{err}");
    assert!(err.msg.contains("(in instance `u`)"), "{err}");
    assert_eq!((err.span.line, err.span.col), (5, 5));
    // two tops, duplicate modules
    let src = "module a(input logic x, output logic y); assign y = x; endmodule
module b(input logic x, output logic y); assign y = x; endmodule
";
    let err = translate(src).err().expect("rejected");
    assert!(err.msg.contains("several top modules (`a`, `b`)"), "{err}");
    let src = "module a(input logic x, output logic y); assign y = x; endmodule
module a(input logic x, output logic y); assign y = ~x; endmodule
";
    let err = translate(src).err().expect("rejected");
    assert!(
        err.msg.contains("module `a` already declared at 1:8"),
        "{err}"
    );
    assert_eq!(err.span.line, 2);
}

/// Translates a single module with header `header` and body `body`, and
/// expects an error at `needle` in line `line`.
fn rejects_module(header: &str, body: &str, line: usize, needle: &str, msg: &str) {
    let src = format!("{header}\n{body}\nendmodule\n");
    let err = match translate(&src) {
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
fn accepted_parameters() {
    let src =
        "module m #(parameter int N = 6, W = $clog2(N) + 1, parameter logic [W-1:0] K = W'(N - 1))
  (input logic [W-1:0] a, output logic [W-1:0] y, output logic [2*W:0] z, output logic b);
  localparam S = 2'd2;
  localparam int P = 2 ** (W - 1) % 5;
  assign y = a + K;
  assign z = {{(N / 2){a[W-1 -: 1]}}, a[W-2:0], {(W - 2){1'b0}}, a[N - 6]};
  assign b = S == $unsigned(2'(P));
endmodule
";
    let m = frontend::parse(src, "test.sv").unwrap();
    let params: Vec<(String, String)> = m
        .params
        .iter()
        .map(|p| (p.name.clone(), format!("{} {}", p.ty, p.value.unsigned())))
        .collect();
    let expected = [
        ("N", "logic signed [31:0] 6"),
        ("W", "logic signed [31:0] 4"),
        ("K", "logic [3:0] 5"),
        ("S", "logic [1:0] 2"),
        ("P", "logic signed [31:0] 3"),
    ];
    assert_eq!(
        params,
        expected
            .iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect::<Vec<_>>()
    );
    translate(src).unwrap();
}

#[test]
fn rejected_parameters() {
    let h = "module m #(parameter int N = 8) (input logic [7:0] a, output logic [7:0] y);";
    let cases = [
        (
            "  assign y = a == N - 1 ? a : 8'd0;",
            "==",
            "an integer constant is an `int` (32 bits): cast it, e.g. `8'(N - 1)`",
        ),
        (
            "  assign y = N;",
            "y",
            "assigning 32 bits to 8 bits (resize explicitly); an integer constant",
        ),
        ("  assign y = a + 1;", "1;", "unsized literal `1`"),
        (
            "  assign y = a + (2 + 3);",
            "2 +",
            "made of unsized literals only",
        ),
        (
            "  localparam K = 8'd1 + 8'd2;",
            "8'd1",
            "`+` on the sized constant",
        ),
        (
            "  localparam int K = N / (N - 8);",
            "N /",
            "division by zero",
        ),
        (
            "  localparam int K = N * 1000000000;",
            "N *",
            "overflows 32 bits",
        ),
        (
            "  localparam logic [3:0] K = 5;",
            "5",
            "given a value of 32 bits",
        ),
        (
            "  localparam int K = 4'd5;",
            "4'd5",
            "is an `int`, but the value",
        ),
        ("  localparam K = a;", "a;", "`a` is not a parameter"),
        ("  localparam N = 3;", "N =", "`N` already declared"),
        ("  logic N;", "N;", "`N` already declared"),
        ("  assign y = a[N];", "a[N]", "out of range"),
        ("  assign y = N[7:0];", "N[7:0]", "select of parameter `N`"),
        (
            "  assign y = a ** 8'd2;",
            "**",
            "`**` only between integer constants",
        ),
        ("  assign y = {(N - 9){a}};", "(N - 9", "is negative"),
        (
            "  localparam type T = logic;",
            "localparam",
            "type parameters",
        ),
        (
            "  localparam byte B = 1;",
            "byte",
            "parameter type `byte` not supported",
        ),
    ];
    for (body, needle, msg) in cases {
        rejects_module(h, body, 2, needle, msg);
    }
    rejects_module(
        "module m #(parameter N) (input logic a, output logic y);",
        "  assign y = a;",
        1,
        "N)",
        "parameter `N` has no value",
    );
    rejects_module(
        "module m (input logic [7:0] a, output logic [7:0] y);",
        "  generate if (1) assign y = a; endgenerate",
        2,
        "generate",
        "generate regions not supported",
    );
}

#[test]
fn instance_modules() {
    // every instance keeps its module before composition, children first,
    // titled with its parameter values and printed with the design's names
    let t = translate(&format!(
        "{}{CHILD}",
        top("  child #(.K(4'd3)) u (.clk, .a(~a), .q, .y);")
    ))
    .unwrap();
    let titles: Vec<&str> = t.instances.iter().map(|p| p.title.as_str()).collect();
    assert_eq!(titles, ["child #(.W(4), .K(4'd3))"]);
    let printed = t.print_part(&t.instances[0]);
    assert!(
        printed.starts_with("module child #(.W(4), .K(4'd3)) u\n"),
        "{printed}"
    );
    // the connection signal `u.a` is computed by the parent, so the instance
    // reads it; the clock is the parent's, and the edge test is named after
    // the instance's port
    assert!(
        printed.contains("  external\n    clk : bv1\n    u.a : bv4"),
        "{printed}"
    );
    assert!(
        printed.contains("      u.clk@pos := AND (%1, X(clk))\n"),
        "{printed}"
    );
}
