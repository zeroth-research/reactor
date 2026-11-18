//! Unit tests of the reference semantics on the AST (`ast_interp`) and of
//! the names `elab` gives the signals and registers of a hierarchy: small
//! designs with hand-computed traces, and the errors for wrong arguments.
//! (On the examples, `ast_interp` is checked against the translation and
//! against iverilog in `examples.rs`.)

use verilog::BitVec;
use verilog::ast_interp::{Interp, Values};
use verilog::elab::{self, Inst};

fn load(src: &str) -> Inst {
    elab::load(src, "test.sv").unwrap_or_else(|e| panic!("{e}\n{src}"))
}

fn bv(width: usize, v: u64) -> BitVec {
    BitVec::new(width, v).unwrap()
}

fn values(vs: &[(&str, usize, u64)]) -> Values {
    vs.iter()
        .map(|(n, w, v)| (n.to_string(), bv(*w, *v)))
        .collect()
}

const COUNTER: &str = "
module counter(input logic clk, input logic en, output logic [3:0] q, output logic full);
  always_ff @(posedge clk) if (en) q <= q + 4'd1;
  assign full = &q;
endmodule
";

/// A round is a time step: the first round has no edge, a block runs in a
/// round where its clock rose since the previous round, on the values of
/// the previous round; combinational signals follow at once.
#[test]
fn rounds_and_edges() {
    let inst = load(COUNTER);
    let mut it = Interp::new(&inst, &values(&[("q", 4, 14)])).unwrap();
    assert_eq!(
        it.registers(),
        [(&"q".to_string(), verilog::ast::Ty::unsigned(4))]
    );
    let inputs: Vec<&str> = it.inputs().iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(inputs, ["clk", "en"]);
    // (clk, en) per round, then the expected q and full
    let rounds = [
        // the first round: no edge, although the clock is 1
        ((1, 1), 14, 0),
        // falling: no edge for `posedge`
        ((0, 1), 14, 0),
        // rising, with `en` = 1 in the previous round: counts
        ((1, 0), 15, 1),
        // high again: no edge
        ((1, 1), 15, 1),
        ((0, 1), 15, 1),
        // rising: wraps around
        ((1, 1), 0, 0),
        ((0, 0), 0, 0),
        // rising, `en` was 0 in the previous round: keeps
        ((1, 1), 0, 0),
    ];
    for (k, ((clk, en), q, full)) in rounds.into_iter().enumerate() {
        let out = it.step(&values(&[("clk", 1, clk), ("en", 1, en)])).unwrap();
        assert_eq!(out["q"], bv(4, q), "q in round {k}");
        assert_eq!(out["full"], bv(1, full), "full in round {k}");
        assert_eq!(out["clk"], bv(1, clk), "inputs are in the values");
    }
}

/// `negedge` blocks run when their clock falls; an asynchronous reset acts
/// in every round where it is active, the first included;
/// initializers fix the register's first value.
#[test]
fn falling_edges_resets_and_initializers() {
    let inst = load(
        "
module m(input logic clk, input logic rst, output logic [1:0] n, output logic [1:0] r,
         output logic [1:0] i);
  initial i = 2'd2;
  always_ff @(negedge clk) n <= n + 2'd1;
  always_ff @(posedge clk or posedge rst)
    if (rst) r <= 2'd3; else r <= r + 2'd1;
  always_ff @(posedge clk) i <= i + 2'd1;
endmodule
",
    );
    let mut it = Interp::new(&inst, &values(&[("n", 2, 0), ("r", 2, 0)])).unwrap();
    // (clk, rst), then n, r, i
    // (the body reads the reset, like any data input, as it was in the
    // previous round)
    let rounds = [
        ((0, 1), 0, 3, 2),
        // the body still sees the reset active
        ((1, 0), 0, 3, 3),
        ((0, 0), 1, 3, 3),
        ((1, 1), 1, 3, 0),
        ((0, 0), 2, 3, 0),
        ((1, 0), 2, 0, 1),
    ];
    for (k, ((clk, rst), n, r, i)) in rounds.into_iter().enumerate() {
        let out = it
            .step(&values(&[("clk", 1, clk), ("rst", 1, rst)]))
            .unwrap();
        let got = [&out["n"], &out["r"], &out["i"]];
        assert_eq!(got, [&bv(2, n), &bv(2, r), &bv(2, i)], "round {k}");
    }
}

/// Wrong initial values and inputs are errors, not panics.
#[test]
fn wrong_arguments() {
    let inst = load(COUNTER);
    let err = |init: Values| Interp::new(&inst, &init).err().unwrap().msg;
    assert_eq!(err(Values::new()), "no initial value for register `q`");
    assert_eq!(err(values(&[("q", 3, 0)])), "register `q` given 3 bits");
    assert_eq!(
        err(values(&[("q", 4, 0), ("full", 1, 0)])),
        "`full` is not a register"
    );
    let init = load(
        "module m(input logic clk, output logic q = 1'b1);\n  \
         always_ff @(posedge clk) q <= !q;\nendmodule\n",
    );
    let e = Interp::new(&init, &values(&[("q", 1, 0)])).err().unwrap();
    assert_eq!(e.msg, "register `q` has an initializer");
    assert!(Interp::new(&init, &Values::new()).is_ok());
    let mut it = Interp::new(&inst, &values(&[("q", 4, 0)])).unwrap();
    assert_eq!(
        it.step(&values(&[("clk", 1, 0)])).unwrap_err(),
        "no value for input `en`"
    );
    assert_eq!(
        it.step(&values(&[("clk", 1, 0), ("en", 2, 0)]))
            .unwrap_err(),
        "input `en` given 2 bits"
    );
}

const HIERARCHY: &str = "
module leaf(input logic clk, input logic [1:0] d, output logic [1:0] q);
  logic [1:0] t;
  assign t = d;
  always_ff @(posedge clk) q <= t;
endmodule
module mid(input logic clk, input logic [1:0] i, output logic [1:0] o);
  logic [1:0] w;
  leaf a(.clk, .d(i), .q(w));
  leaf b(.clk, .d(w), .q(o));
endmodule
module top(input logic clk, input logic [1:0] x, output logic [1:0] y);
  logic r;
  mid m(.clk, .i(x + 2'd1), .o(y));
  always_ff @(posedge clk) r <= ^y;
endmodule
";

/// Every signal path has a name: a module's own signals before its
/// instances', ports connected to a whole signal take that signal's name,
/// other ports keep their path. Registers are listed by path, a module's
/// before its instances'.
#[test]
fn naming_and_registers() {
    let inst = load(HIERARCHY);
    let naming = inst.naming();
    let pairs: Vec<(&str, &str)> = naming
        .iter()
        .map(|(p, n)| (p.as_str(), n.as_str()))
        .collect();
    assert_eq!(
        pairs,
        [
            ("clk", "clk"),
            ("x", "x"),
            ("y", "y"),
            ("r", "r"),
            ("m.clk", "clk"),
            // an expression, not a signal: the port keeps its path
            ("m.i", "m.i"),
            ("m.o", "y"),
            ("m.w", "m.w"),
            ("m.a.clk", "clk"),
            ("m.a.d", "m.i"),
            ("m.a.q", "m.w"),
            ("m.a.t", "m.a.t"),
            ("m.b.clk", "clk"),
            ("m.b.d", "m.w"),
            ("m.b.q", "y"),
            ("m.b.t", "m.b.t"),
        ]
    );
    assert_eq!(inst.registers(), ["r", "m.a.q", "m.b.q"]);
    assert!(inst.initialized().is_empty());
    assert_eq!(inst.at("m.b.").module.name, "leaf");
    // the interpreter takes registers by path and gives signals by name
    let init = values(&[("r", 1, 0), ("m.a.q", 2, 1), ("m.b.q", 2, 2)]);
    let mut it = Interp::new(&inst, &init).unwrap();
    let out = it.step(&values(&[("clk", 1, 0), ("x", 2, 3)])).unwrap();
    assert_eq!(out["m.i"], bv(2, 0));
    assert_eq!(out["m.w"], bv(2, 1));
    assert_eq!(out["y"], bv(2, 2));
    assert_eq!(out["m.b.t"], bv(2, 1));
    assert!(!out.contains_key("m.b.q"));
    let out = it.step(&values(&[("clk", 1, 1), ("x", 2, 3)])).unwrap();
    assert_eq!(
        (&out["m.w"], &out["y"], &out["r"]),
        (&bv(2, 0), &bv(2, 1), &bv(1, 1))
    );
}

/// Runs `rounds` (inputs, clocks included) in both the reference
/// semantics and the translated module from the register values `init`,
/// and returns the values of the signals `shown` in each round, after
/// checking that both agree on them.
fn both(inst: &Inst, init: &Values, rounds: &[Values], shown: &[&str]) -> Vec<Vec<u64>> {
    use num_traits::ToPrimitive;
    use verilog::lower::{init_name, lower_inst};
    use verilog::rm_sim::{Sim, Valuation};
    let t = lower_inst(inst).unwrap();
    let mut interp = Interp::new(inst, init).unwrap();
    let mut sim = Sim::new(&t.module);
    let mut out = Vec::new();
    for (k, inputs) in rounds.iter().enumerate() {
        let expected = interp.step(inputs).unwrap();
        let extl: Valuation = inputs
            .iter()
            .map(|(n, v)| (t.var(n), v.clone()))
            .chain(init.iter().map(|(q, v)| (t.var(&init_name(q)), v.clone())))
            .filter(|(var, _)| t.module.extl().contains(var))
            .collect();
        let got = sim.step(&extl).unwrap();
        let mut row = Vec::new();
        for x in shown {
            assert_eq!(got[&t.var(x)], expected[*x], "`{x}` in round {k}");
            row.push(expected[*x].unsigned().to_u64().unwrap());
        }
        out.push(row);
    }
    out
}

/// A reset asserted in the round of an edge of the data clock acts after
/// the edge: the registers it resets end the round at their reset values,
/// and a block running at that edge reads the value from before the edge,
/// not the reset value. A register on an edge the reset makes (a derived
/// clock it resets) changes in the reset's round, and a block on another
/// clock of the same root that reads it sees the change from its next edge
/// on.
#[test]
fn resets_at_an_edge() {
    let inst = load(
        "
module m(input logic clk, input logic rst, output logic [1:0] q, output logic [1:0] y);
  always_ff @(posedge clk or posedge rst)
    if (rst) q <= 2'd0; else q <= q + 2'd1;
  always_ff @(posedge clk) y <= q;
endmodule
",
    );
    let round = |clk, rst| values(&[("clk", 1, clk), ("rst", 1, rst)]);
    let rounds = [
        round(0, 0),
        round(1, 0),
        round(0, 0),
        round(1, 1),
        round(0, 0),
        round(1, 0),
    ];
    let init = values(&[("q", 2, 2), ("y", 2, 0)]);
    // q, y per round
    let expected = [[2, 0], [3, 2], [3, 2], [0, 3], [0, 3], [1, 0]];
    assert_eq!(both(&inst, &init, &rounds, &["q", "y"]), expected);

    let inst = load(
        "
module m(input logic ca, input logic rb, output logic s, output logic t, output logic y);
  always_ff @(posedge ca or posedge rb) if (rb) s <= 1'b1; else s <= !s;
  always_ff @(posedge s) t <= !t;
  always_ff @(posedge ca) y <= t;
endmodule
",
    );
    let round = |ca, rb| values(&[("ca", 1, ca), ("rb", 1, rb)]);
    // the reset makes a rising edge of `s` in round 1, where `t` toggles;
    // `y`, on the clock `s` is derived from, sees it at the next edge of
    // `ca` (round 3)
    let rounds = [round(0, 0), round(0, 1), round(0, 0), round(1, 0)];
    let init = values(&[("s", 1, 0), ("t", 1, 0), ("y", 1, 0)]);
    let expected = [[0, 0, 0], [1, 1, 0], [1, 1, 0], [0, 1, 1]];
    assert_eq!(both(&inst, &init, &rounds, &["s", "t", "y"]), expected);
}
