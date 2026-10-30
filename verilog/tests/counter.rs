//! A hand-built RM for an 8-bit counter with enable, as the translator
//! will produce it for
//!
//! ```systemverilog
//! module counter(input logic clk, input logic en,
//!                output logic [7:0] q, output logic full);
//!   always_ff @(posedge clk) if (en) q <= q + 8'd1;
//!   assign full = &q;
//! endmodule
//! ```
//!
//! The register `q` starts at the external `q_init`, so every initial
//! value is possible.

use base::var::X;
use verilog::rm_sim::{Sim, Valuation};
use verilog::{Atom, BitVec, Module, Sort, SvOp, Term, Var, Wire};

struct Counter {
    module: Module,
    en: Var,
    q_init: Var,
    q: Var,
    full: Var,
}

fn counter(atoms_in_wrong_order: bool) -> Result<Counter, String> {
    let en = Var::new(Sort(1));
    let q_init = Var::new(Sort(8));
    let q = Var::new(Sort(8));
    let full = Var::new(Sort(1));

    // always_ff: reads latched values, writes X(q); q keeps its value when !en
    let one = Wire::scalar(Sort(8));
    let inc = Wire::scalar(Sort(8));
    let init = [Term::function(SvOp::Id, [X(q)], [X(q_init)])?];
    let update = [
        Term::constant(SvOp::Lit(BitVec::new(8, 1u8).unwrap()), [one])?,
        Term::function(SvOp::Add, [inc], [*q, one])?,
        Term::function(SvOp::Ite, [X(q)], [*en, inc, *q])?,
    ];
    let reg = Atom::sequential(&[en, q_init, q], init, update)?;

    // assign: awaits X(q)
    let assign = [Term::function(SvOp::RedAnd, [X(full)], [X(q)])?];
    let comb = Atom::combinatorial(&[q, full], assign)?;

    let atoms = if atoms_in_wrong_order {
        [comb, reg]
    } else {
        [reg, comb]
    };
    let module = Module::observable(atoms)?;
    Ok(Counter {
        module,
        en,
        q_init,
        q,
        full,
    })
}

impl Counter {
    fn name(&self, var: &Var) -> &'static str {
        [
            (self.en, "en"),
            (self.q_init, "q_init"),
            (self.q, "q"),
            (self.full, "full"),
        ]
        .into_iter()
        .find(|(v, _)| v == var)
        .map_or("?", |(_, n)| n)
    }

    fn inputs(&self, en: bool, q_init: u8) -> Valuation {
        Valuation::from([
            (self.en, BitVec::new(1, en as u8).unwrap()),
            (self.q_init, BitVec::new(8, q_init).unwrap()),
        ])
    }
}

#[test]
fn structure() {
    let c = counter(false).unwrap();
    let names = |vars: &base::Interface<Sort>| {
        let mut names: Vec<_> = vars.iter().map(|v| c.name(v)).collect();
        names.sort();
        names
    };
    assert_eq!(names(c.module.extl()), ["en", "q_init"]);
    assert_eq!(names(c.module.intf()), ["full", "q"]);
    assert!(c.module.prvt().is_empty());
    println!("{}", c.module.with_varnames(|v| c.name(v)));
}

#[test]
fn atoms_must_follow_await_order() {
    assert!(counter(true).is_err());
}

#[test]
fn trace() {
    let c = counter(false).unwrap();
    let mut sim = Sim::new(&c.module);
    // (en given in this round, q_init, expected q, expected full)
    let rounds = [
        (true, 254, 254, false), // init round: q = q_init
        (false, 0, 255, true),   // en was 1 in the previous round
        (true, 0, 255, true),    // en was 0
        (true, 0, 0, false),     // wraps around
        (true, 0, 1, false),
    ];
    for (k, (en, q_init, q, full)) in rounds.into_iter().enumerate() {
        let next = sim.step(&c.inputs(en, q_init)).unwrap();
        assert_eq!(
            next[&c.q],
            BitVec::new(8, q as u8).unwrap(),
            "q in round {k}"
        );
        assert_eq!(
            next[&c.full],
            BitVec::new(1, full as u8).unwrap(),
            "full in round {k}"
        );
    }
}

/// Pseudo-random traces against a native model of the counter.
#[test]
fn random_traces() {
    let c = counter(false).unwrap();
    let mut seed: u64 = 0x2545_f491_4f6c_dd1d;
    let mut rand = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as u8
    };
    for _ in 0..50 {
        let mut sim = Sim::new(&c.module);
        let q_init = rand();
        let (mut q, mut en) = (q_init, false);
        for k in 0..100 {
            if k > 0 && en {
                q = q.wrapping_add(1);
            }
            en = rand() % 2 == 1;
            // q_init matters only in the init round
            let next = sim
                .step(&c.inputs(en, if k == 0 { q_init } else { rand() }))
                .unwrap();
            assert_eq!(next[&c.q], BitVec::new(8, q).unwrap());
            assert_eq!(next[&c.full], BitVec::new(1, (q == 255) as u8).unwrap());
        }
    }
}

#[test]
fn missing_input_is_an_error() {
    let c = counter(false).unwrap();
    let mut sim = Sim::new(&c.module);
    let mut inputs = c.inputs(true, 0);
    inputs.remove(&c.en);
    assert!(sim.step(&inputs).is_err());
}
