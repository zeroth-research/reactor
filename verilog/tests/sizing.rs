//! The width and signedness rules of IEEE 1800: values computed by hand
//! from the rules (and checked against iverilog when written) for the
//! example designs, which the AST semantics and the translated module must
//! give; and the sized form of expressions, rule by rule.

use std::collections::BTreeMap;
use std::path::Path;
use verilog::ast::Item;
use verilog::ast_interp::{Interp, Values};
use verilog::lower::{init_name, lower_inst};
use verilog::rm_sim::{Sim, Valuation};
use verilog::{BitVec, elab, sizing, typing};

/// Per example: rounds of input values and the expected output values,
/// in hex. Registers start at 0; clocks stay low (no edge).
const VECTORS: &[(&str, &[(&str, &str)])] = &[
    (
        "ctxwidth",
        &[
            (
                "a=c8 b=64 c=1",
                "sum16=012c sum4=c half16=0096 half8=16 red=0 land=1 cat=002c carry=1 inv=ff37",
            ),
            (
                "a=ff b=01 c=0",
                "sum16=0100 sum4=0 half16=0080 half8=00 red=0 land=0 cat=0000 carry=1 inv=ff00",
            ),
            (
                "a=f0 b=0f c=1",
                "sum16=00ff sum4=f half16=007f half8=7f red=1 land=1 cat=00ff carry=0 inv=ff0f",
            ),
        ],
    ),
    (
        "unsized",
        &[
            (
                "q=f x=ff c=0",
                "inc=0 wrapped=0 wrapped4=1 ones=ff zeros=00 pick=ff full=1 based=02 \
                 big=0100000000",
            ),
            (
                "q=3 x=12 c=1",
                "inc=4 wrapped=0 wrapped4=0 ones=ff zeros=00 pick=12 full=0 based=15 \
                 big=0100000000",
            ),
        ],
    ),
    (
        "signmix",
        &[
            (
                "s=f9 t=e u=c8 c=1",
                "su=01c1 s0=fff9 sd0=00f9 sg=ff90 lt=0 neg=1 ash=fffc cash=007c mash=0144 \
                 div=fffd cs=00f9 cw=fffe cc=01 m=0",
            ),
            (
                "s=05 t=f u=00 c=0",
                "su=0005 s0=0005 sd0=0005 sg=0000 lt=0 neg=0 ash=0002 cash=0002 mash=0002 \
                 div=0002 cs=0000 cw=0005 cc=02 m=0",
            ),
            (
                "s=f9 t=7 u=10 c=0",
                "su=0109 s0=fff9 sd0=00f9 sg=0020 lt=0 neg=1 ash=fffc cash=007c mash=008c \
                 div=fffd cs=0010 cw=fff9 cc=01 m=0",
            ),
        ],
    ),
    ("carry", &[("a=ff b=01 cin=1 clk=0", "sum=01 cout=1")]),
    (
        "fsmint",
        &[
            ("k=1 clk=0 go=0", "hit=1"),
            ("k=2 clk=0 go=0", "hit=1"),
            ("k=0 clk=0 go=0", "hit=0"),
            ("k=7 clk=0 go=0", "hit=0"),
        ],
    ),
    (
        "castparam",
        &[(
            "u=c8 a=ff b=03",
            "wide=0190 h=001 low=2 m=ff s=5 k=00000003",
        )],
    ),
    (
        "shiftw",
        &[
            ("a=c1 n=5", "sl=0304 sn=1820 one=20 amt=0010"),
            ("a=c1 n=7", "sl=0304 sn=6080 one=80 amt=0040"),
        ],
    ),
    (
        "consts",
        &[
            (
                "a=c8 b=0000000005",
                "w=015a l=37 k=0 c=08 h=a9 big=0000000100000000 pow=0000000080000000 \
                 least=ffffffff80000000 negpow=ffffffffea",
            ),
            (
                "a=ff b=0000000001",
                "w=0191 l=00 k=3 c=1f h=e0 big=0000000100000000 pow=0000000080000000 \
                 least=ffffffff80000000 negpow=ffffffffe6",
            ),
        ],
    ),
    (
        "portw",
        &[(
            "n4=a s4=a w16=1234 a=ff b=02",
            "y1=0a y2=fa y3=34 y4=01 y5=05",
        )],
    ),
];

/// `name=hex …` as values of the signals' widths.
fn values(text: &str, width: impl Fn(&str) -> usize) -> Values {
    text.split_whitespace()
        .map(|kv| {
            let (k, v) = kv.split_once('=').unwrap();
            let n = num_bigint::BigUint::parse_bytes(v.as_bytes(), 16).unwrap();
            (k.to_string(), BitVec::new(width(k), n).unwrap())
        })
        .collect()
}

#[test]
fn vectors() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
    for (name, rounds) in VECTORS {
        let inst = elab::load_file(dir.join(format!("{name}.sv"))).unwrap();
        let t = lower_inst(&inst).unwrap();
        let env = typing::check(&inst.module).unwrap();
        let width = |x: &str| env[x].ty.width;
        let init: Values = registers(&inst);
        for (inputs, expected) in rounds.iter() {
            let inputs = values(inputs, width);
            let expected = values(expected, width);
            // a fresh start for each round: the vectors are independent
            let mut interp = Interp::new(&inst, &init).unwrap();
            let got = interp.step(&inputs).unwrap();
            let mut sim = Sim::new(&t.module);
            let extl: Valuation = inputs
                .iter()
                .map(|(n, v)| (n.clone(), v.clone()))
                .chain(init.iter().map(|(q, v)| (init_name(q), v.clone())))
                .filter(|(n, _)| t.vars.contains_key(n))
                .map(|(n, v)| (t.var(&n), v))
                .filter(|(var, _)| t.module.extl().contains(var))
                .collect();
            let sim: BTreeMap<String, BitVec> = sim
                .step(&extl)
                .unwrap()
                .into_iter()
                .map(|(var, v)| (t.name(&var).to_string(), v))
                .collect();
            for (y, v) in &expected {
                assert_eq!(&got[y], v, "{name}: `{y}` with {inputs:?} (AST semantics)");
                assert_eq!(
                    &sim[y], v,
                    "{name}: `{y}` with {inputs:?} (translated module)"
                );
            }
        }
    }
}

/// Every register without an initial value at 0, by path.
fn registers(inst: &elab::Inst) -> Values {
    let mut out = Values::new();
    let mut walk = vec![(inst, String::new())];
    while let Some((i, prefix)) = walk.pop() {
        let env = typing::check(&i.module).unwrap();
        let inits: Vec<&String> = i.module.inits.iter().map(|i| &i.target).collect();
        for q in i
            .module
            .registers()
            .into_iter()
            .filter(|q| !inits.contains(q))
        {
            out.insert(format!("{prefix}{q}"), BitVec::zero(env[q].ty.width));
        }
        for (u, c) in &i.children {
            walk.push((c, format!("{prefix}{u}.")));
        }
    }
    out
}

/// The sized right-hand side of `assign y = <e>;` in a module with the
/// given ports, as `v2rm --sized` prints it.
fn sized(ports: &str, e: &str) -> String {
    let src = format!("module m({ports});\n  assign y = {e};\nendmodule\n");
    let inst = elab::load(&src, "test.sv").unwrap_or_else(|err| panic!("{src}\n{err}"));
    assert!(matches!(inst.module.items[..], [Item::Assign { .. }]));
    let text = sizing::print(&inst).unwrap();
    let line = text.lines().find(|l| l.contains("assign y = ")).unwrap();
    line.trim()
        .trim_start_matches("assign y = ")
        .trim_end_matches(';')
        .to_string()
}

const PORTS: &str = "input logic [7:0] a, b, input logic signed [7:0] s, input logic signed [3:0] t, \
                     input logic [3:0] q, input logic [2:0] n, input logic c";

/// Each rule on its own, as the sized expression shows it.
#[test]
fn rules() {
    let out = |w: &str| format!("{PORTS}, output logic {w} y");
    let cases = [
        // context-determined operands take the width of the target
        ("[15:0]", "a + b", "(16'(unsigned'(a)) + 16'(unsigned'(b)))"),
        // truncated after the operation, not before
        ("[3:0]", "a + b", "4'((a + b))"),
        // `~` passes the context down
        ("[15:0]", "~a", "(~16'(unsigned'(a)))"),
        // a self-determined operand keeps its width: reductions, `&&`,
        // concatenation items, the condition of `?:`, the shift amount
        ("", "&(a + b)", "(&(a + b))"),
        ("", "(a + b) && c", "((a + b) && c)"),
        ("[15:0]", "{a + b}", "16'(unsigned'({(a + b)}))"),
        (
            "[15:0]",
            "(a + b) >> n",
            "((16'(unsigned'(a)) + 16'(unsigned'(b))) >> n)",
        ),
        ("[7:0]", "a ? b : 8'd0", "(a ? b : 8'd0)"),
        // the operands of a comparison are sized together
        (
            "",
            "a + b > 9'd255",
            "((9'(unsigned'(a)) + 9'(unsigned'(b))) > 9'd255)",
        ),
        ("", "q + 1 == 0", "((32'(unsigned'(q)) + 32'd1) == 32'd0)"),
        // signedness: an operator is signed only if all its operands are,
        // and the operands are extended as the operator's type says
        ("[15:0]", "s + a", "(16'(unsigned'(s)) + 16'(unsigned'(a)))"),
        ("[15:0]", "s + t", "(16'(signed'(s)) + 16'(signed'(t)))"),
        ("[15:0]", "s + 0", "16'((32'(signed'(s)) + 32'sd0))"),
        ("[15:0]", "s + 'd0", "16'((32'(unsigned'(s)) + 32'd0))"),
        ("", "s < a", "(unsigned'(s) < a)"),
        ("", "s < t", "(s < 8'(signed'(t)))"),
        // `$signed` and concatenations: self-determined operands
        ("[15:0]", "$signed(a + b)", "16'(signed'(signed'((a + b))))"),
        ("[15:0]", "{s} >>> 1", "(16'(unsigned'({s})) >>> 32'sd1)"),
        ("[15:0]", "s >>> 1", "(16'(signed'(s)) >>> 32'sd1)"),
        // `?:` arms are sized together
        (
            "[15:0]",
            "c ? s : a",
            "(c ? 16'(unsigned'(s)) : 16'(unsigned'(a)))",
        ),
        (
            "[15:0]",
            "c ? t : s",
            "(c ? 16'(signed'(t)) : 16'(signed'(s)))",
        ),
        // unsized literals and fills
        ("[7:0]", "'1", "8'd255"),
        ("[7:0]", "c ? a : '1", "(c ? a : 8'd255)"),
        ("", "a == '1", "(a == 8'd255)"),
        ("[39:0]", "'h7fffffff + 'h7fffffff + 2", "40'd4294967296"),
        // a size cast is an assignment to its width
        (
            "[15:0]",
            "16'(a + b)",
            "16'((16'(unsigned'(a)) + 16'(unsigned'(b))))",
        ),
        (
            "[8:0]",
            "8'(a + b) >> 1",
            "(9'(unsigned'(8'((a + b)))) >> 32'sd1)",
        ),
        // constants are folded at the type of their context
        (
            "[39:0]",
            "a + -1",
            "(40'(unsigned'(a)) + 40'd1099511627775)",
        ),
        (
            "[31:0]",
            "a + (0 - 7) / 2",
            "(32'(unsigned'(a)) + 32'd2147483644)",
        ),
        ("[7:0]", "a / 2", "8'((32'(unsigned'(a)) / 32'd2))"),
    ];
    for (w, e, expected) in cases {
        assert_eq!(sized(&out(w), e), expected, "{e} into {w}");
    }
}

/// The selector and the labels of a `case` are sized together, and an
/// input port connection is an assignment to the port.
#[test]
fn case_and_ports() {
    let src = "module m(input logic [2:0] k, output logic y);
  always_comb case (k) 4'd9: y = 1; 1, 2: y = 1'b1; default: y = 0; endcase
endmodule
";
    let inst = elab::load(src, "test.sv").unwrap();
    let text = sizing::print(&inst).unwrap();
    assert!(
        text.contains(
            "case (32'(unsigned'(k))) 32'd9: y = 1'sd1; 32'd1, 32'd2: y = 1'd1; \
             default: y = 1'sd0; endcase"
        ),
        "{text}"
    );
    // signed only if the selector and all labels are
    let src = "module m(input logic signed [3:0] t, output logic y);
  always_comb case (t) 8'd255: y = 1'b1; default: y = 1'b0; endcase
endmodule
";
    let text = sizing::print(&elab::load(src, "test.sv").unwrap()).unwrap();
    assert!(text.contains("case (8'(unsigned'(t))) 8'd255:"), "{text}");
    let src = "module c(input logic [7:0] x, output logic [7:0] y); assign y = x; endmodule
module m(input logic [3:0] n, input logic signed [3:0] s, output logic [7:0] y, z);
  c u (.x(n), .y);
  c v (.x(s), .y(z));
endmodule
";
    let inst = elab::load(src, "test.sv").unwrap();
    let text = sizing::print(&inst).unwrap();
    assert!(
        text.contains("c u (.x(8'(unsigned'(n))), .y(y));"),
        "{text}"
    );
    assert!(text.contains("c v (.x(8'(signed'(s))), .y(z));"), "{text}");
    // the ports are signals of their own: `u.x` is not `n`
    let names: BTreeMap<_, _> = inst.naming().into_iter().collect();
    assert_eq!(names["u.x"], "u.x");
    assert_eq!(names["u.y"], "y");
}

/// `**` is evaluated at the type of its context like the other
/// arithmetic operators: the base is context-determined, the exponent
/// self-determined (IEEE 1800 Table 11-21), so a negative base in a wide
/// unsigned context is extended before the power is taken.
#[test]
fn powers_of_negative_bases() {
    let ok = |ports: &str, e: &str, expected: &str| assert_eq!(sized(ports, e), expected);
    ok(
        "input logic [39:0] a, output logic [39:0] y",
        "a + 3 ** 2",
        "(a + 40'd9)",
    );
    ok(
        "input logic [31:0] a, output logic [31:0] y",
        "a + (-3) ** 1",
        "(a + 32'd4294967293)",
    );
    ok(
        "input logic signed [39:0] a, output logic signed [39:0] y",
        "a + (-3) ** 1",
        "(a + 40'sd1099511627773)",
    );
    // all operands signed: a signed context, even with an unsigned target
    ok(
        "input logic [39:0] a, output logic [39:0] y",
        "(-3) ** 1",
        "40'sd1099511627773",
    );
    // the base extended to 40 bits first: -3 and -7, not 2^32 - 3
    ok(
        "input logic [39:0] a, output logic [39:0] y",
        "a + (-3) ** 1",
        "(a + 40'd1099511627773)",
    );
    ok(
        "input logic [39:0] a, output logic [39:0] y",
        "a + ((-2) ** 3 + 1)",
        "(a + 40'd1099511627769)",
    );
    // the exponent is self-determined: an unsigned exponent is never negative
    ok(
        "input logic [7:0] a, output logic [7:0] y",
        "a + 8'd2 ** 4'b1111",
        "(a + 8'd0)",
    );
    ok(
        "input logic [7:0] a, output logic [7:0] y",
        "a + 8'd3 ** 4'd3",
        "(a + 8'd27)",
    );
    // a negative exponent: 0 for bases other than 0, 1 and -1
    ok(
        "input logic [7:0] a, output logic [7:0] y",
        "a + 8'd2 ** -1",
        "(a + 8'd0)",
    );
    ok(
        "input logic [7:0] a, output logic [7:0] y",
        "a + 8'd1 ** -1",
        "(a + 8'd1)",
    );
    // the base -1 only in a signed context: unsigned, it is 255
    ok(
        "input logic signed [7:0] a, output logic signed [7:0] y",
        "a + (-8'sd1) ** -3",
        "(a + 8'sd255)",
    );
    ok(
        "input logic [7:0] a, output logic [7:0] y",
        "a + (-8'd1) ** -3",
        "(a + 8'd0)",
    );
}

/// The source of a module `m` with the parameters `params` and the body
/// `body`, its ports `ports`, translated; the error if it is rejected.
fn translate(params: &str, ports: &str, body: &str) -> Result<String, verilog::ast::Error> {
    let src = format!("module m #({params}) ({ports});\n{body}\nendmodule\n");
    elab::load(&src, "test.sv")
        .and_then(|i| lower_inst(&i))
        .map(|t| t.print())
}

/// Constant expressions are expressions like any other: comments inside
/// them, every operator, typed parameters. Integer constants are exact in
/// ranges and untyped parameters; in a value expression they are evaluated
/// at the width of the context, so `65536 * 65536` does not overflow there.
#[test]
fn constant_expressions() {
    let width = |params: &str, range: &str| {
        let ports = format!("input logic {range} a, output logic {range} y");
        translate(params, &ports, "  assign y = a;")
            .unwrap_or_else(|e| panic!("{params} {range}: {e}"))
    };
    let cases = [
        ("parameter N = 8 /* bits */ - 0", "[N /* w */ -1:0]", "bv8"),
        ("parameter N = 3", "[(1 << N) - 1:0]", "bv8"),
        ("parameter N = 3", "[N > 2 ? 1 : 0:0]", "bv2"),
        ("parameter [4:0] W = 8", "[W-1:0]", "bv8"),
        ("parameter [4:0] W = 8", "[$clog2(W):0]", "bv4"),
        ("parameter N = 8 // bits\n", "[N-1:0]", "bv8"),
    ];
    for (params, range, sort) in cases {
        let text = width(params, range);
        assert!(
            text.contains(&format!("a : {sort}")),
            "{params} {range}:\n{text}"
        );
    }
    let ok = |ports: &str, e: &str, expected: &str| assert_eq!(sized(ports, e), expected);
    ok("output logic [63:0] y", "65536 * 65536", "64'sd4294967296");
    ok("output logic [63:0] y", "2 ** 31", "64'sd2147483648");
    ok("output logic [63:0] y", "2**32 / 2", "64'sd2147483648");
    // the least `int`: the literal 2147483648 negated at once
    let text = translate(
        "localparam P = -2147483648",
        "output logic [63:0] y, output logic [31:0] z",
        "  assign y = P;\n  assign z = -2147483648;",
    )
    .unwrap();
    // 0xffffffff80000000 and 0x80000000
    assert!(text.contains("X(y) := 64'd18446744071562067968"), "{text}");
    assert!(text.contains("X(z) := 32'd2147483648"), "{text}");
    // a typed parameter is assigned its value: evaluated at its type
    let text = translate(
        "localparam int I = 65536 * 65536, localparam logic [63:0] Q = 65536 * 65536, \
         localparam [7:0] K = 8'd1 + 8'd2",
        "output logic [31:0] i, output logic [63:0] q, output logic [7:0] k",
        "  assign i = I;\n  assign q = Q;\n  assign k = K;",
    )
    .unwrap();
    assert!(text.contains("X(i) := 32'd0"), "{text}");
    assert!(text.contains("X(q) := 64'd4294967296"), "{text}");
    assert!(text.contains("X(k) := 8'd3"), "{text}");
    // integer constants stay exact where an integer is needed
    let rejected = [
        ("localparam P = 65536 * 65536", "[0:0]", "overflows 32 bits"),
        ("localparam P = 2147483648", "[0:0]", "too large"),
        ("localparam P = 1", "[65536 * 65536:0]", "overflows 32 bits"),
        ("localparam P = 0 ** -1", "[0:0]", "0 ** -1"),
        // a typed parameter of 0 minus 1 is a large unsigned number
        ("parameter [3:0] W = 0", "[W-1:0]", "is 4294967295"),
    ];
    for (params, range, msg) in rejected {
        let ports = format!("input logic {range} a, output logic {range} y");
        let err = translate(params, &ports, "  assign y = a;")
            .err()
            .unwrap_or_else(|| panic!("accepted: {params} {range}"));
        assert!(err.msg.contains(msg), "{params} {range}: {err}");
    }
}
