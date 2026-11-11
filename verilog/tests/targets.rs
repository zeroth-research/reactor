//! Assignments to selects with a variable index, in the shapes where
//! Yosys deviates from the standard (so they are not among the examples,
//! which the Yosys comparison checks): a `-:` select reaching below bit 0
//! and a select inside a concatenation. Values computed by hand from the
//! standard (bits outside the signal are not written) and checked against
//! iverilog when written; the AST semantics and the translated module must
//! give them.

use std::collections::BTreeMap;
use verilog::ast_interp::{Interp, Values};
use verilog::lower::lower_inst;
use verilog::rm_sim::{Sim, Valuation};
use verilog::{BitVec, elab, typing};

/// The design (also used by `tests/yosys.rs`).
const DESIGN: &str = include_str!("varwrites.sv");

/// Inputs and expected outputs, in hex.
const VECTORS: &[(&str, &str)] = &[
    // bits 1, 0 from e[2:1]; bit -1 not written
    ("n=1 a=0f e=2", "z=0d r=0d f=0"),
    ("n=0 a=0f e=4", "z=0f r=0e f=1"),
    // bits 9, 8 not written; r[9] does not exist
    ("n=9 a=00 e=7", "z=80 r=00 f=1"),
    ("n=6 a=ff e=2", "z=af r=bf f=0"),
];

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
fn deviating_shapes() {
    let inst = elab::load(DESIGN, "dev.sv").unwrap();
    let t = lower_inst(&inst).unwrap();
    let env = typing::check(&inst.module).unwrap();
    let width = |x: &str| env[x].ty.width;
    for (inputs, expected) in VECTORS {
        let inputs = values(inputs, width);
        let expected = values(expected, width);
        let got = Interp::new(&inst, &Values::new())
            .unwrap()
            .step(&inputs)
            .unwrap();
        let extl: Valuation = inputs.iter().map(|(n, v)| (t.var(n), v.clone())).collect();
        let sim: BTreeMap<String, BitVec> = Sim::new(&t.module)
            .step(&extl)
            .unwrap()
            .into_iter()
            .map(|(var, v)| (t.name(&var).to_string(), v))
            .collect();
        for (y, v) in &expected {
            assert_eq!(&got[y], v, "`{y}` with {inputs:?} (AST semantics)");
            assert_eq!(&sim[y], v, "`{y}` with {inputs:?} (translated module)");
        }
    }
}
