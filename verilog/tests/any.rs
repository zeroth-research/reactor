//! The rewriting into the ops of the `theory` crate (`any::desugar`) and
//! the export into `theory::any`: each `qfbv` op whose desugaring has cases
//! (1 bit, 64 bits, constant and variable operands, negative and most
//! negative constants), checked by z3 against the `qfbv` rounds; division
//! by a variable divisor on modules built by hand (designs only divide by
//! nonzero constants); the bridges between truth values and 1-bit words
//! leaving no terms; and the ops of exported modules. (Every example and
//! every accepted test design is checked the same way, see
//! `common::bridge_equal`.)

mod common;

use common::{bridge_equal, desugar_equal, exported_equal, module, translate};
use std::collections::{BTreeSet, HashMap};
use verilog::any::{self, Op};
use verilog::qfbv::{self, Sort};

/// Translates the combinational design with the ports `ports` and the body
/// `body`, checks every step down to the export, and returns the desugared
/// module.
fn check(ports: &str, body: &str) -> any::Desugared {
    let src = module("m", ports, body);
    let t = translate(&src).unwrap_or_else(|e| panic!("{src}\n{e}"));
    let label = format!("{src}\n");
    let (_, d) = bridge_equal(&label, &t).unwrap_or_else(|| panic!("{label}refused"));
    d
}

/// The ops of the update blocks of `d`.
fn ops(d: &any::Desugared) -> Vec<Op> {
    let atoms = d.module.atoms().iter();
    atoms
        .flat_map(|a| a.update().iter().map(|t| *t.itype()))
        .collect()
}

#[test]
fn concatenation_and_sign_extension() {
    for (h, l) in [(1, 1), (1, 63), (63, 1), (32, 32), (3, 5)] {
        check(
            &format!(
                "input logic [{}:0] a, input logic signed [{}:0] b, \
                 output logic [{}:0] x, output logic signed [63:0] y, z",
                h - 1,
                l - 1,
                h + l - 1
            ),
            "assign x = {a, b};\nassign y = b;\nassign z = $signed(a);",
        );
    }
}

#[test]
fn signed_comparisons_and_division() {
    for w in [1, 2, 8, 64] {
        let ports = format!(
            "input logic signed [{h}:0] a, b, output logic x, y, \
             output logic signed [{h}:0] q, r",
            h = w - 1
        );
        check(
            &ports,
            "assign x = a < b;\nassign y = a >= b;\nassign q = a / 1;\nassign r = a % 1;",
        );
    }
    for d in ["3", "-3", "-1", "-8'sd128", "8'sd127", "2"] {
        check(
            "input logic signed [7:0] a, output logic signed [7:0] q, r, \
             input logic [7:0] u, output logic [7:0] uq, ur",
            &format!(
                "assign q = a / {d};\nassign r = a % {d};\n\
                 assign uq = u / 8'd3;\nassign ur = u % 8'd3;"
            ),
        );
    }
}

#[test]
fn shifts_by_constants() {
    for w in [1, 2, 8, 64] {
        let ports = format!(
            "input logic [{h}:0] a, input logic signed [{h}:0] s, \
             output logic [{h}:0] x0, x1, x2, x3, output logic signed [{h}:0] y0, y1, y2",
            h = w - 1
        );
        let k = w - 1;
        check(
            &ports,
            &format!(
                "assign x0 = a << 1;\nassign x1 = a >> 1;\nassign x2 = a << {k};\n\
                 assign x3 = a >> {w};\nassign y0 = s >>> 1;\nassign y1 = s >>> {k};\n\
                 assign y2 = s >>> {};",
                w + 3
            ),
        );
    }
}

/// The module `z = op(a, b)` over `qfbv` on `w`-bit words, with `b` the
/// constant `divisor` if given.
fn hand_built(op: qfbv::Op, w: usize, divisor: Option<u64>) -> qfbv::Bridged {
    let word = Sort::Word(w);
    let (a, b, z) = (
        qfbv::Var::new(word),
        qfbv::Var::new(word),
        qfbv::Var::new(word),
    );
    let x = reactive::var::X;
    let mut terms = vec![];
    let d = match divisor {
        Some(value) => {
            let c = qfbv::Wire::new(word);
            terms.push(qfbv::term(qfbv::Op::Num { value, width: w }, [c], []).unwrap());
            c
        }
        None => x(&b),
    };
    terms.push(qfbv::term(op, [x(&z)], [x(&a), d]).unwrap());
    let scope: BTreeSet<qfbv::Var> = [a, b, z].into_iter().collect();
    let atom = qfbv::Atom::combinatorial::<qfbv::Op, _, _>(&scope, terms).unwrap();
    let module = qfbv::Module::partially_observable([atom], |_| false).unwrap();
    let names = HashMap::from([(a, "a".into()), (b, "b".into()), (z, "z".into())]);
    qfbv::Bridged::new(module, names)
}

#[test]
fn division_by_a_variable_or_zero() {
    use qfbv::Op::*;
    for op in [UDiv, URem, SDiv, SRem] {
        for w in [1, 3, 8, 64] {
            // z3 compares two dividers bit by bit: with a variable divisor
            // of 16 bits it does not finish in minutes
            let variable = (w <= 8).then_some(None);
            let constants = [Some(0), Some(1), Some(u64::MAX >> (64 - w))];
            for divisor in variable.into_iter().chain(constants) {
                let label = format!("{op} on {w} bits by {divisor:?}: ");
                let d = desugar_equal(&label, &hand_built(op, w, divisor));
                exported_equal(&label, &d);
                // every divisor is a nonzero constant or guarded
                let atom = &d.module.atoms()[0];
                let consts: HashMap<_, _> = (atom.update().iter())
                    .filter_map(|t| match t.itype() {
                        Op::Const { value, .. } => Some((t.write()[0], *value)),
                        _ => None,
                    })
                    .collect();
                for t in atom.update().iter() {
                    if matches!(t.itype(), Op::UDiv | Op::UMod) {
                        let divisor = t.read()[1];
                        let guarded = atom
                            .update()
                            .iter()
                            .any(|g| g.write() == [divisor] && *g.itype() == Op::Ite);
                        assert!(
                            guarded || consts.get(&divisor).is_some_and(|v| *v != 0),
                            "{label}a divisor that may be zero"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn truth_values_need_no_terms() {
    // a comparison writes the variable itself, a condition is read as it is
    let d = check(
        "input logic [7:0] a, b, input logic c, output logic x, output logic [7:0] y",
        "assign x = a < b;\nassign y = c ? a : b;",
    );
    let mut ops = ops(&d);
    ops.sort_by_key(|o| o.to_string());
    assert_eq!(ops, [Op::Ite, Op::ULt]);
}

#[test]
fn exported_ops() {
    // registers with reset, keep-value, constants and a variable shift: the
    // exported module has the ops of `theory::bv` (`Id`, never `SKIP`)
    let d = check(
        "input logic clk, rst, input logic [7:0] a, input logic [2:0] k, \
         output logic [7:0] q, x",
        "always_ff @(posedge clk) if (rst) q <= 8'd5; else q <= q + a;\nassign x = a << k;",
    );
    let e = exported_equal("", &d);
    let names: BTreeSet<String> = (e.module.atoms().iter())
        .flat_map(|a| {
            let init = a.init().iter().map(|t| t.itype().to_string());
            init.chain(a.update().iter().map(|t| t.itype().to_string()))
                .collect::<Vec<_>>()
        })
        .collect();
    assert!(names.iter().all(|n| n != "SKIP"), "{names:?}");
    assert!(
        names.contains("Id") && names.contains("Ite") && names.contains("Mul"),
        "{names:?}"
    );
}
