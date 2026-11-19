//! Scaling of the translation and of both interpreters on generated
//! designs that grow in one direction each: a wide state machine (one
//! `case` with many labels), a long chain of `assign`s (written in reverse
//! order, so the await order must sort all of them), a deep hierarchy of
//! distinct modules, a wide hierarchy of instances of one module, and one
//! expression with very many operands (the depth of the expression tree).
//!
//! Each design is translated, then run for a few rounds by the AST
//! semantics and by the translated module, which must agree; the time of
//! each phase is printed (`cargo test --test scaling -- --nocapture`):
//! parsing, elaboration, the checks of every instance, the clock domains,
//! the whole translation (which runs the checks again), the AST semantics
//! (its setup checks the instances again), and both interpreters compared.
//! The default sizes keep the test quick; `V2RM_SCALE=k` multiplies them
//! by `k`, and the ignored test `scaling_large` runs the large sizes
//! (about 10 000 operands in one expression).

mod common;

use std::fmt::Write as _;
use std::time::Instant;

use common::{Gen, Rand, rm_matches_ast};
use verilog::ast_interp::Interp;
use verilog::{elab, frontend, typing};
use verilog::lower::lower_inst;

/// A state machine with `n` states: each state goes to the next one or
/// back to the first, by an input bit; one output per state range.
fn fsm(n: usize) -> String {
    let w = (usize::BITS - n.leading_zeros()) as usize + 1;
    let mut s = format!(
        "module top(input logic clk, input logic [7:0] in, output logic [{}:0] s, output logic odd);\n\
         logic [{}:0] state, next;\n\
         always_comb begin\n  odd = 1'b0;\n  case (state)\n",
        w - 1,
        w - 1
    );
    for k in 0..n {
        let _ = writeln!(
            s,
            "    {w}'d{k}: begin next = in[{}] ? {w}'d{} : {w}'d0; odd = 1'b{}; end",
            k % 8,
            (k + 1) % n,
            k % 2
        );
    }
    s.push_str(
        "    default: next = '0;\n  endcase\nend\n\
         always_ff @(posedge clk) state <= next;\n\
         assign s = state;\nendmodule\n",
    );
    s
}

/// `n` wires, each one more than the previous, assigned in reverse order.
fn chain(n: usize) -> String {
    let mut s = String::from("module top(input logic [7:0] a, output logic [7:0] y);\n");
    for k in 0..=n {
        let _ = writeln!(s, "logic [7:0] w{k};");
    }
    for k in (1..=n).rev() {
        let _ = writeln!(s, "assign w{k} = w{} + 8'd1;", k - 1);
    }
    s.push_str("assign w0 = a;\nassign y = w");
    let _ = writeln!(s, "{n};\nendmodule");
    s
}

/// A hierarchy `n` modules deep: module `m<k>` has a register and an
/// instance of `m<k+1>`.
fn deep(n: usize) -> String {
    let mut s = String::from(
        "module top(input logic clk, input logic [7:0] a, output logic [7:0] y);\n\
         m0 u(.clk(clk), .a(a), .y(y));\nendmodule\n",
    );
    for k in 0..n {
        let _ = write!(
            s,
            "module m{k}(input logic clk, input logic [7:0] a, output logic [7:0] y);\n\
             logic [7:0] q, t;\nalways_ff @(posedge clk) q <= a + 8'd{};\n",
            k % 256
        );
        if k + 1 < n {
            let _ = writeln!(s, "m{} u(.clk(clk), .a(q), .y(t));", k + 1);
        } else {
            s.push_str("assign t = q;\n");
        }
        s.push_str("assign y = t ^ q;\nendmodule\n");
    }
    s
}

/// `n` instances of one module, in a chain.
fn wide(n: usize) -> String {
    let mut s = String::from(
        "module stage #(parameter int W = 8)(input logic clk, input logic [W-1:0] a, output logic [W-1:0] y);\n\
         logic [W-1:0] q;\nalways_ff @(posedge clk) q <= a + 1'b1;\nassign y = q ^ a;\nendmodule\n\
         module top(input logic clk, input logic [7:0] a, output logic [7:0] y);\n",
    );
    for k in 0..=n {
        let _ = writeln!(s, "logic [7:0] w{k};");
    }
    s.push_str("assign w0 = a;\n");
    for k in 0..n {
        let _ = writeln!(s, "stage u{k}(.clk(clk), .a(w{k}), .y(w{}));", k + 1);
    }
    let _ = writeln!(s, "assign y = w{n};\nendmodule");
    s
}

/// One expression with `n` operands.
fn expr(n: usize) -> String {
    let mut s = String::from(
        "module top(input logic [7:0] a, input logic [7:0] b, output logic [7:0] y);\nassign y = a",
    );
    for k in 1..n {
        s.push_str(if k % 2 == 0 { " + a" } else { " ^ b" });
    }
    s.push_str(";\nendmodule\n");
    s
}

/// Translates and runs `src`, printing the time of each phase.
fn measure(label: &str, size: usize, src: &str) {
    let t0 = Instant::now();
    let d = frontend::parse_design(src, "scaling.sv").unwrap_or_else(|e| panic!("{label}: {e}"));
    let tp = Instant::now();
    let top = d.top().unwrap_or_else(|e| panic!("{label}: {e}"));
    let inst = elab::elaborate(&d, &top, &[]).unwrap_or_else(|e| panic!("{label}: {e}"));
    let t1 = Instant::now();
    for (_, i) in inst.iter().filter(|(_, i)| !i.module.blackbox) {
        typing::checked(i).unwrap_or_else(|e| panic!("{label}: {e}"));
    }
    let tc = Instant::now();
    typing::domains(&inst).unwrap_or_else(|e| panic!("{label}: {e}"));
    let td = Instant::now();
    let t = lower_inst(&inst).unwrap_or_else(|e| panic!("{label}: {e}"));
    let t2 = Instant::now();
    let mut rand = Rand(common::seed(0x5ca1_ab1e, label));
    let init = rand.registers(&inst);
    let trace = rand.trace(&inst, &t.clocks, Gen::Toggle, 4);
    let mut interp = Interp::new(&inst, &init).unwrap();
    for inputs in &trace {
        interp.step(inputs).unwrap();
    }
    let t3 = Instant::now();
    rm_matches_ast(label, (&inst, &t), &mut rand, 1, 4);
    let t4 = Instant::now();
    let phases = [
        ("parse", tp - t0),
        ("elab", t1 - tp),
        ("checks", tc - t1),
        ("domains", td - tc),
        ("lower", t2 - td),
        ("interp", t3 - t2),
        ("both", t4 - t3),
    ];
    let mut line = format!("{label:>6} n={size:>6}:");
    for (name, d) in phases {
        let _ = write!(line, " {name} {:>8.1}", d.as_secs_f64() * 1e3);
    }
    eprintln!("{line} (ms)");
}

/// The factor `V2RM_SCALE` (1 by default).
fn scale() -> usize {
    std::env::var("V2RM_SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
}

/// Runs every design at the given sizes, on a thread with a stack of
/// `stack` bytes: the parser recurses once per operand of an operator
/// chain, with about 10 KB per level.
fn run(sizes: [usize; 5], stack: usize) {
    std::thread::Builder::new()
        .stack_size(stack)
        .spawn(move || {
            let [f, c, d, w, e] = sizes;
            measure("fsm", f, &fsm(f));
            measure("chain", c, &chain(c));
            measure("deep", d, &deep(d));
            measure("wide", w, &wide(w));
            measure("expr", e, &expr(e));
        })
        .unwrap()
        .join()
        .unwrap();
}

/// Small sizes, quick in a debug build (times printed with
/// `--nocapture`).
#[test]
fn scaling() {
    let k = scale();
    run([32 * k, 100 * k, 10 * k, 20 * k, 200 * k], 1 << 30);
}

/// Large sizes (run in a release build:
/// `cargo test --release --test scaling -- --ignored --nocapture`).
#[test]
#[ignore = "large designs; slow in a debug build"]
fn scaling_large() {
    run([1000, 2000, 200, 500, 10_000], 1 << 30);
}
