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
