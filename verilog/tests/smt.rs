//! The SMT-LIB printer against `value::eval`: random ops on random values,
//! printed as closed terms; z3 must prove each term equal to the literal of
//! the value `eval` gives. Shifts are also checked exhaustively on small
//! widths. Skipped (with a message) if z3 is not found in `$Z3_BIN`,
//! `~/tools/oss-cad-suite/bin` or the `PATH`.

use num_bigint::BigUint;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use verilog::value::eval;
use verilog::{BitVec, Sort, SvOp, smt};

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

fn z3_bin() -> Option<PathBuf> {
    let candidates = [
        std::env::var_os("Z3_BIN").map(PathBuf::from),
        std::env::var_os("HOME").map(|h| Path::new(&h).join("tools/oss-cad-suite/bin")),
    ];
    let found = candidates
        .into_iter()
        .flatten()
        .find(|dir| dir.join("z3").exists())
        .map(|dir| dir.join("z3"));
    found.or_else(|| {
        let ok = Command::new("z3").arg("--version").output().is_ok();
        ok.then(|| PathBuf::from("z3"))
    })
}

/// Runs z3 on `script` and returns its output lines.
fn z3(bin: &Path, script: &str) -> Vec<String> {
    let mut child = Command::new(bin)
        .args(["-smt2", "-in"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("z3 starts");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(script.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect()
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
    let out = z3(bin, &script);
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
    let Some(bin) = z3_bin() else {
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
    let Some(bin) = z3_bin() else {
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
