//! The rewriting into the SMT-LIB bit-vector op set (`qfbv::bridge`): each
//! operator of the accepted subset at the widths where its bridge changes
//! (1 bit, odd widths, 64 bits; shift amounts narrower, as wide and wider
//! than the value), checked by z3 against the original rounds; the shape of
//! the bridged module; and the refusal of words over 64 bits. (Every
//! example and every accepted test design is checked the same way, see
//! `common::bridge_equal`.)

mod common;

use common::{bridge_equal, examples, load, module, translate};
use std::collections::BTreeSet;
use verilog::ast::ErrorKind;
use verilog::qfbv::{self, Op, Sort};

/// Translates the combinational design with inputs `a`, `b`, `c`, `k`
/// (declared by `ports`) and the body `body`, and checks the bridged module.
fn check(ports: &str, body: &str) {
    let src = module("m", ports, body);
    let t = translate(&src).unwrap_or_else(|e| panic!("{src}\n{e}"));
    let label = format!("{src}\n");
    assert!(bridge_equal(&label, &t).is_some(), "{label}refused");
}

#[test]
fn reductions() {
    for w in [1, 2, 3, 5, 8, 17, 63, 64] {
        check(
            &format!("input logic [{}:0] a, output logic x, y, z", w - 1),
            "assign x = &a;\nassign y = |a;\nassign z = ^a;",
        );
    }
}

#[test]
fn comparisons_and_conditions() {
    for w in [1, 8, 64] {
        let ports = format!(
            "input logic [{h}:0] a, b, input logic signed [{h}:0] sa, sb, input logic c, \
             output logic [5:0] x, output logic [{h}:0] y, z",
            h = w - 1
        );
        check(
            &ports,
            "assign x = {a == b, a != b, a < b, a <= b, sa < sb, sa >= sb};\n\
             assign y = c ? a : b;\n\
             assign z = (a > b) ? a : sa;",
        );
    }
}

#[test]
fn shifts() {
    for (w, k) in [
        (8, 3),
        (8, 8),
        (3, 8),
        (1, 1),
        (1, 64),
        (64, 64),
        (64, 7),
        (5, 64),
        (63, 64),
    ] {
        let ports = format!(
            "input logic [{}:0] a, input logic signed [{}:0] s, input logic [{}:0] k, \
             output logic [{}:0] x, y, output logic signed [{}:0] z",
            w - 1,
            w - 1,
            k - 1,
            w - 1,
            w - 1
        );
        check(
            &ports,
            "assign x = a << k;\nassign y = a >> k;\nassign z = s >>> k;",
        );
    }
}

#[test]
fn arithmetic_and_bits() {
    for w in [1, 7, 32, 64] {
        let ports = format!(
            "input logic [{h}:0] a, b, input logic signed [{h}:0] s, output logic [{h}:0] x, y, \
             output logic signed [{h}:0] z",
            h = w - 1
        );
        // a concatenation of at most 64 bits
        let cat = if w <= 32 {
            "{a, b}".to_string()
        } else {
            format!("{{a[0], b[{}:1]}}", w - 1)
        };
        check(
            &ports,
            &format!(
                "assign x = (a + b) * (a - b) ^ ~a & -b | a / 3;\n\
                 assign y = a % 3 + {cat};\n\
                 assign z = s / 3 + s % 3;"
            ),
        );
    }
}

#[test]
fn extensions_and_constants() {
    check(
        "input logic [6:0] a, input logic signed [6:0] s, output logic [63:0] x, \
         output logic signed [63:0] y, output logic [12:0] z",
        "assign x = a ^ 64'hFFFF_FFFF_FFFF_FFFF;\n\
         assign y = s + 64'sh8000_0000_0000_0000;\n\
         assign z = {a[6:1], a};",
    );
}

#[test]
fn registers_and_reset() {
    check(
        "input logic clk, rst_n, input logic [3:0] d, output logic [3:0] q, r",
        "always_ff @(posedge clk or negedge rst_n)\n\
           if (!rst_n) q <= 4'd5; else q <= q + d;\n\
         always_ff @(negedge clk) r <= ^q ? d : r;",
    );
}

/// The bridged module has the same variables in the same roles, the same
/// number of atoms, and every atom controls, awaits and reads the same
/// variables.
#[test]
fn same_shape() {
    for path in examples() {
        let (_, t) = load(&path);
        let b = qfbv::bridge(&t).unwrap();
        let names = |vs: &mut dyn Iterator<Item = String>| vs.collect::<BTreeSet<String>>();
        let (m, n) = (&t.module, &b.module);
        let ours = |i: &reactive::Interface<verilog::Sort>| {
            names(&mut i.iter().map(|v| t.name(v).to_string()))
        };
        let theirs =
            |i: &reactive::Interface<Sort>| names(&mut i.iter().map(|v| b.name(v).to_string()));
        let p = path.display();
        assert_eq!(ours(m.extl()), theirs(n.extl()), "{p}");
        assert_eq!(ours(m.intf()), theirs(n.intf()), "{p}");
        assert_eq!(ours(m.prvt()), theirs(n.prvt()), "{p}");
        assert_eq!(m.atoms().len(), n.atoms().len(), "{p}");
        for (x, y) in m.atoms().iter().zip(n.atoms()) {
            assert_eq!(ours(x.ctrl()), theirs(y.ctrl()), "{p}");
            assert_eq!(ours(x.wait()), theirs(y.wait()), "{p}");
            assert_eq!(ours(x.read()), theirs(y.read()), "{p}");
            for v in y.ctrl().iter() {
                assert!(
                    matches!(v.dtype(), Sort::Word(_)),
                    "{p}: a variable of sort {}",
                    v.dtype()
                );
            }
        }
    }
}

/// Only the bridge ops occur: no comparison result reaches a variable
/// directly, and every `Ite` condition is a `Bool`.
#[test]
fn bool_stays_local() {
    let src = module(
        "m",
        "input logic [3:0] a, b, output logic x, output logic [3:0] y",
        "assign x = a < b;\nassign y = x ? a : b;",
    );
    let t = translate(&src).unwrap();
    let b = qfbv::bridge(&t).unwrap();
    let terms: Vec<_> = b
        .module
        .atoms()
        .iter()
        .flat_map(|a| a.update().iter())
        .collect();
    let ops: Vec<Op> = terms.iter().map(|t| *t.itype()).collect();
    assert!(ops.contains(&Op::ULt), "{ops:?}");
    for t in &terms {
        if *t.itype() == Op::Ite {
            assert_eq!(*t.read()[0].dtype(), Sort::Bool);
        }
    }
}

#[test]
fn wide_words_refused() {
    let src = module(
        "m",
        "input logic [69:0] a, input logic [3:0] b, output logic [3:0] y",
        "logic [69:0] w;\nassign w = a + 70'd1;\nassign y = w[3:0] ^ b;",
    );
    let t = translate(&src).unwrap();
    let e = qfbv::bridge(&t).err().expect("refused");
    assert_eq!(e.kind, ErrorKind::Unsupported);
    assert!(
        e.msg.contains("`a` (70 bits)") && e.msg.contains("`w` (70 bits)"),
        "{}",
        e.msg
    );
    assert!(!e.msg.contains("`b`"), "{}", e.msg);
}
