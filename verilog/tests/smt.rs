//! The SMT-LIB printer against `value::eval`: random ops on random values,
//! printed as closed terms; z3 must prove each term equal to the literal of
//! the value `eval` gives. Shifts are also checked exhaustively on small
//! widths. Skipped (with a message) if z3 is not found in `$Z3_BIN`,
//! `~/tools/oss-cad-suite/bin` or the `PATH`. And the exact text of a few
//! rounds of examples, which pins the documented format (what the rounds of
//! all examples mean is checked in `examples.rs`).

use num_bigint::BigUint;
use std::path::Path;
use verilog::smt::Phase;
use verilog::value::eval;
use verilog::{BitVec, Sort, SvOp, elab, lower, smt, z3};

struct Rand(u64);

impl Rand {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 16
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// A width, mostly small, sometimes around 32, 64 and 128 bits.
    fn width(&mut self) -> usize {
        const WIDE: [usize; 9] = [16, 31, 32, 33, 63, 64, 65, 100, 130];
        if self.below(4) == 0 {
            WIDE[self.below(WIDE.len())]
        } else {
            1 + self.below(9)
        }
    }

    /// A value of the given width, biased towards the edge cases 0, 1, all
    /// ones, the most negative and the largest signed value, and small
    /// values.
    fn bits(&mut self, width: usize) -> BitVec {
        match self.below(10) {
            0 => BitVec::zero(width),
            1 => BitVec::wrap(width, 1u8),
            2 => BitVec::from_signed(width, -1),
            3 => BitVec::wrap(width, BigUint::from(1u8) << (width - 1)),
            4 => BitVec::wrap(width, (BigUint::from(1u8) << (width - 1)) - 1u8),
            5 => BitVec::wrap(width, self.below(4)),
            _ => {
                let mut v = BigUint::from(0u8);
                for _ in 0..width.div_ceil(48) {
                    v = (v << 48) | BigUint::from(self.next());
                }
                BitVec::wrap(width, v)
            }
        }
    }

    fn nonzero(&mut self, width: usize) -> BitVec {
        loop {
            let v = self.bits(width);
            if !v.is_zero() {
                return v;
            }
        }
    }

    /// A shift amount: of any width, mostly around the shifted width `w`.
    fn amount(&mut self, w: usize) -> BitVec {
        let k = match self.below(3) {
            0 => self.width(),
            _ => 1 + self.below(8),
        };
        if self.below(2) == 0 {
            BitVec::wrap(k, w.saturating_sub(2) + self.below(5))
        } else {
            self.bits(k)
        }
    }
}

/// Every op but `Havoc` and `Zero`; the parameters are placeholders,
/// [`random_case`] picks them.
fn ops() -> Vec<SvOp> {
    use SvOp::*;
    let lit = Lit(BitVec::zero(1));
    #[rustfmt::skip]
    let ops = vec![
        lit, Id, Not, And, Or, Xor, Neg, Add, Sub, Mul, UDiv, URem, SDiv, SRem,
        Eq, Ne, ULt, ULe, SLt, SLe, RedAnd, RedOr, RedXor, Ite, Shl, LShr, AShr,
        Extract { hi: 0, lo: 0 }, Concat, ZExt(0), SExt(0),
    ];
    ops
}

/// An application of an op like `template` to random operands of a random
/// width.
fn random_case(rand: &mut Rand, template: &SvOp) -> (SvOp, Vec<BitVec>) {
    use SvOp::*;
    let w = rand.width();
    let a = rand.bits(w);
    match template {
        Lit(_) => (Lit(a), vec![]),
        Id | Not | Neg | RedAnd | RedOr | RedXor => (template.clone(), vec![a]),
        UDiv | URem | SDiv | SRem => (template.clone(), vec![a, rand.nonzero(w)]),
        Ite => {
            let c = rand.bits(1);
            (Ite, vec![c, a, rand.bits(w)])
        }
        Shl | LShr | AShr => (template.clone(), vec![a, rand.amount(w)]),
        Extract { .. } => {
            let lo = rand.below(w);
            let hi = lo + rand.below(w - lo);
            (Extract { hi, lo }, vec![a])
        }
        Concat => {
            let v = rand.width();
            (Concat, vec![a, rand.bits(v)])
        }
        ZExt(_) | SExt(_) => {
            let to = w + [0, 1, 2, rand.below(70)][rand.below(4)];
            let ext = if matches!(template, ZExt(_)) {
                ZExt(to)
            } else {
                SExt(to)
            };
            (ext, vec![a])
        }
        _ => (template.clone(), vec![a, rand.bits(w)]),
    }
}

/// All shifts of every value by every amount, for value and amount widths
/// 1 to 4.
fn shift_cases() -> Vec<(SvOp, Vec<BitVec>)> {
    let mut cases = vec![];
    for op in [SvOp::Shl, SvOp::LShr, SvOp::AShr] {
        for w in 1..=4 {
            for k in 1..=4 {
                for a in 0..1usize << w {
                    for b in 0..1usize << k {
                        let args = vec![BitVec::wrap(w, a), BitVec::wrap(k, b)];
                        cases.push((op.clone(), args));
                    }
                }
            }
        }
    }
    cases
}

/// Checks each case with z3: the printed term differs from the literal of
/// its value must be unsatisfiable. One line of the script per case, after
/// a one-line header, so a z3 error names the case by its line.
fn check(bin: &Path, cases: &[(SvOp, Vec<BitVec>)]) {
    let mut script = String::from("(set-logic QF_BV)\n");
    let mut shown = vec![];
    for (op, args) in cases {
        let operands: Vec<(String, Sort)> = args.iter().map(|v| (smt::lit(v), v.sort())).collect();
        let term = smt::apply(op, &operands);
        let value = eval(op, args).unwrap();
        script += &format!(
            "(push) (assert (distinct {term} {})) (check-sat) (pop)\n",
            smt::lit(&value)
        );
        let args: Vec<String> = args.iter().map(BitVec::to_string).collect();
        let term = match term.char_indices().nth(300) {
            Some((i, _)) => format!("{}…", &term[..i]),
            None => term,
        };
        shown.push(format!(
            "{op}({}) = {value}, printed as {term}",
            args.join(", ")
        ));
    }
    let out = z3::run(bin, &script).unwrap();
    if let Some(e) = out.iter().find(|l| l.starts_with("(error")) {
        panic!("z3: {e}\n(line 2 is the first case)");
    }
    assert_eq!(out.len(), cases.len(), "z3 output: {out:?}");
    let wrong: Vec<&String> = out
        .iter()
        .zip(&shown)
        .filter(|(answer, _)| *answer != "unsat")
        .map(|(_, case)| case)
        .collect();
    assert!(
        wrong.is_empty(),
        "{} of {} cases differ, e.g.:\n{}",
        wrong.len(),
        cases.len(),
        wrong
            .iter()
            .take(5)
            .map(|c| c.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn ops_match_eval() {
    let Some(bin) = z3::find() else {
        eprintln!("z3 not found: skipped");
        return;
    };
    let mut rand = Rand(0x9e37_79b9_7f4a_7c15);
    let cases: Vec<_> = ops()
        .iter()
        .flat_map(|op| std::iter::repeat_n(op, 150))
        .map(|op| random_case(&mut rand, op))
        .collect();
    check(&bin, &cases);
}

#[test]
fn shifts_match_eval() {
    let Some(bin) = z3::find() else {
        eprintln!("z3 not found: skipped");
        return;
    };
    check(&bin, &shift_cases());
}

#[test]
fn printed_forms() {
    use SvOp::*;
    let x = |w: usize| (format!("|x{w}|"), Sort(w));
    assert_eq!(smt::sort(Sort(8)), "(_ BitVec 8)");
    assert_eq!(smt::lit(&BitVec::wrap(8, 5u8)), "#b00000101");
    assert_eq!(smt::symbol("u.q@init"), "|u.q@init|");
    assert_eq!(
        smt::apply(&Ite, &[x(1), x(8), x(8)]),
        "(ite (= |x1| #b1) |x8| |x8|)"
    );
    assert_eq!(
        smt::apply(&ULt, &[x(8), x(8)]),
        "(ite (bvult |x8| |x8|) #b1 #b0)"
    );
    assert_eq!(smt::apply(&Shl, &[x(4), x(4)]), "(bvshl |x4| |x4|)");
    assert_eq!(
        smt::apply(&Shl, &[x(4), x(2)]),
        "(bvshl |x4| ((_ zero_extend 2) |x2|))"
    );
    assert_eq!(
        smt::apply(&AShr, &[x(2), x(3)]),
        "(bvashr |x2| (ite (bvult |x3| #b010) ((_ extract 1 0) |x3|) #b10))"
    );
    assert_eq!(smt::apply(&SExt(8), &[x(8)]), "|x8|");
    assert_eq!(smt::apply(&RedXor, &[x(1)]), "|x1|");
}

#[test]
#[should_panic(expected = "cannot be an SMT-LIB symbol")]
fn symbol_rejects_bar() {
    smt::symbol("a|b");
}

/// A `Havoc` term declares its wire (lowering produces none yet, so no
/// example has one).
#[test]
fn round_declares_havoc() {
    use base::var::X;
    use std::collections::{BTreeMap, HashMap};
    use verilog::lower::Translation;
    use verilog::{Atom, Module, Term, Var, Wire};
    let (a, y) = (Var::new(Sort(8)), Var::new(Sort(8)));
    let h = Wire::scalar(Sort(8));
    let assign = [
        Term::constant(SvOp::Havoc(Sort(8)), [h]).unwrap(),
        Term::function(SvOp::Xor, [X(y)], [X(a), h]).unwrap(),
    ];
    let atom = Atom::combinatorial(&[a, y], assign).unwrap();
    let t = Translation {
        name: "noise".into(),
        module: Module::observable([atom]).unwrap(),
        vars: BTreeMap::from([("a".into(), a), ("y".into(), y)]),
        clocks: vec![],
        registers: vec![],
        reset_registers: vec![],
        initialized: vec![],
        register_widths: Default::default(),
        unread_inputs: vec![],
        wire_names: HashMap::new(),
        instances: vec![],
    };
    let expected = "\
; v2rm-input a 8
(declare-const |X(a)| (_ BitVec 8))
; atom controls y
(declare-const |%1| (_ BitVec 8))
; v2rm-next y 8 interface
(define-fun |X(y)| () (_ BitVec 8) (bvxor |X(a)| |%1|))
";
    for phase in [Phase::Init, Phase::Update] {
        let r = smt::round(&t, phase);
        assert_eq!(r.text, expected);
        assert_eq!(
            (r.latched.len(), r.inputs, r.next),
            (0, vec!["a".to_string()], vec!["y".to_string()])
        );
    }
}

/// The round of an example, with the exact text of the format.
fn example_round(name: &str, phase: Phase) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("examples/{name}.sv"));
    let inst = elab::load_file(&path).unwrap();
    smt::round(&lower::lower_inst(&inst).unwrap(), phase).text
}

/// Latched values and inputs, the comments, and the init round.
#[test]
fn round_counter() {
    assert_eq!(
        example_round("counter", Phase::Update),
        "\
; v2rm-latched clk 1
(declare-const |*clk| (_ BitVec 1))
; v2rm-latched en 1
(declare-const |*en| (_ BitVec 1))
; v2rm-latched q 8
(declare-const |*q| (_ BitVec 8))
; v2rm-input clk 1
(declare-const |X(clk)| (_ BitVec 1))
; atom controls q
(define-fun |%1| () (_ BitVec 1) (bvnot |*clk|))
(define-fun |clk@pos| () (_ BitVec 1) (bvand |%1| |X(clk)|))
(define-fun |%2| () (_ BitVec 8) #b00000001)
(define-fun |%3| () (_ BitVec 8) (bvadd |*q| |%2|))
(define-fun |%4| () (_ BitVec 8) (ite (= |*en| #b1) |%3| |*q|))
; v2rm-next q 8 interface
(define-fun |X(q)| () (_ BitVec 8) (ite (= |clk@pos| #b1) |%4| |*q|))
; atom controls full
; v2rm-next full 1 interface
(define-fun |X(full)| () (_ BitVec 1) (ite (= |X(q)| #b11111111) #b1 #b0))
"
    );
    assert_eq!(
        example_round("counter", Phase::Init),
        "\
; v2rm-input q@init 8
(declare-const |X(q@init)| (_ BitVec 8))
; atom controls q
; v2rm-next q 8 interface
(define-fun |X(q)| () (_ BitVec 8) |X(q@init)|)
; atom controls full
; v2rm-next full 1 interface
(define-fun |X(full)| () (_ BitVec 1) (ite (= |X(q)| #b11111111) #b1 #b0))
"
    );
}

/// The intermediate values `sum@1`, `sum@2`, … of an `always_comb` block.
#[test]
fn round_satadd() {
    assert_eq!(
        example_round("satadd", Phase::Update),
        "\
; v2rm-input a 8
(declare-const |X(a)| (_ BitVec 8))
; v2rm-input b 8
(declare-const |X(b)| (_ BitVec 8))
; atom controls y, sat, sum
(define-fun |sum@1| () (_ BitVec 8) (bvadd |X(a)| |X(b)|))
(define-fun |sat@1| () (_ BitVec 1) (ite (bvult |sum@1| |X(a)|) #b1 #b0))
(define-fun |sum@2| () (_ BitVec 8) #b11111111)
; v2rm-next sum 8 private
(define-fun |X(sum)| () (_ BitVec 8) (ite (= |sat@1| #b1) |sum@2| |sum@1|))
; v2rm-next sat 1 interface
(define-fun |X(sat)| () (_ BitVec 1) |sat@1|)
; v2rm-next y 8 interface
(define-fun |X(y)| () (_ BitVec 8) |X(sum)|)
"
    );
}

/// A composed design: path names (also of the edge tests), the variable
/// `m1.l.d` of a connection, and `%k` numbered across atoms.
#[test]
fn round_nest() {
    assert_eq!(
        example_round("nest", Phase::Update),
        "\
; v2rm-latched clk 1
(declare-const |*clk| (_ BitVec 1))
; v2rm-latched m1.q 4
(declare-const |*m1.q| (_ BitVec 4))
; v2rm-latched m1.l.d 4
(declare-const |*m1.l.d| (_ BitVec 4))
; v2rm-latched m2.q 4
(declare-const |*m2.q| (_ BitVec 4))
; v2rm-latched m2.l.d 4
(declare-const |*m2.l.d| (_ BitVec 4))
; v2rm-input clk 1
(declare-const |X(clk)| (_ BitVec 1))
; v2rm-input a 4
(declare-const |X(a)| (_ BitVec 4))
; atom controls m1.q
(define-fun |%1| () (_ BitVec 1) (bvnot |*clk|))
(define-fun |m1.l.clk@pos| () (_ BitVec 1) (bvand |%1| |X(clk)|))
; v2rm-next m1.q 4 private
(define-fun |X(m1.q)| () (_ BitVec 4) (ite (= |m1.l.clk@pos| #b1) |*m1.l.d| |*m1.q|))
; atom controls m2.q
(define-fun |%2| () (_ BitVec 1) (bvnot |*clk|))
(define-fun |m2.l.clk@pos| () (_ BitVec 1) (bvand |%2| |X(clk)|))
; v2rm-next m2.q 4 private
(define-fun |X(m2.q)| () (_ BitVec 4) (ite (= |m2.l.clk@pos| #b1) |*m2.l.d| |*m2.q|))
; atom controls m1.s
(define-fun |%3| () (_ BitVec 4) #b0010)
; v2rm-next m1.s 4 private
(define-fun |X(m1.s)| () (_ BitVec 4) (bvadd |X(m1.q)| |%3|))
; atom controls m2.s
(define-fun |%4| () (_ BitVec 4) #b0111)
; v2rm-next m2.s 4 private
(define-fun |X(m2.s)| () (_ BitVec 4) (bvadd |X(m2.q)| |%4|))
; atom controls y
; v2rm-next y 4 interface
(define-fun |X(y)| () (_ BitVec 4) (bvadd |X(m1.s)| |X(m1.q)|))
; atom controls m1.l.d
; v2rm-next m1.l.d 4 private
(define-fun |X(m1.l.d)| () (_ BitVec 4) (bvxor |X(a)| |X(m1.s)|))
; atom controls z
; v2rm-next z 4 interface
(define-fun |X(z)| () (_ BitVec 4) (bvadd |X(m2.s)| |X(m2.q)|))
; atom controls m2.l.d
; v2rm-next m2.l.d 4 private
(define-fun |X(m2.l.d)| () (_ BitVec 4) (bvxor |X(y)| |X(m2.s)|))
"
    );
}
