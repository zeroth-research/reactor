//! `SvOp`: type checking per op, and evaluation edge cases. On 8 bits, the
//! arithmetic, comparison, and shift ops are compared exhaustively with
//! Rust's `u8`/`i8` arithmetic.

use ::theory::Theory;
use num_bigint::BigUint;
use verilog::value::eval;
use verilog::{BitVec, Sort, SvOp};

fn check(op: SvOp, read: &[usize], write: usize) -> Result<(), String> {
    check_degree(op, read, (write, 0))
}

fn check_degree(op: SvOp, read: &[usize], (write, degree): (usize, u8)) -> Result<(), String> {
    let ok = |(w, d): (usize, u8)| Ok::<_, String>((Sort(w), d));
    op.check(read.iter().map(|w| ok((*w, 0))), [ok((write, degree))])
}

fn bv(width: usize, value: u128) -> BitVec {
    BitVec::new(width, value).expect("value fits")
}

fn ev(op: SvOp, args: &[BitVec]) -> BitVec {
    eval(&op, args).unwrap()
}

#[test]
fn check_accepts() {
    use SvOp::*;
    check(Lit(bv(8, 3)), &[], 8).unwrap();
    check(Id, &[8], 8).unwrap();
    check(Havoc(Sort(8)), &[], 8).unwrap();
    check_degree(Zero(Sort(8)), &[], (8, 1)).unwrap();
    check(Not, &[8], 8).unwrap();
    check(Add, &[8, 8], 8).unwrap();
    check(SLt, &[8, 8], 1).unwrap();
    check(RedXor, &[8], 1).unwrap();
    check(Ite, &[1, 8, 8], 8).unwrap();
    check(AShr, &[8, 3], 8).unwrap();
    check(Shl, &[8, 100], 8).unwrap();
    check(Extract { hi: 7, lo: 4 }, &[8], 4).unwrap();
    check(Extract { hi: 0, lo: 0 }, &[8], 1).unwrap();
    check(Concat, &[3, 5], 8).unwrap();
    check(ZExt(16), &[8], 16).unwrap();
    check(SExt(8), &[8], 8).unwrap();
}

#[test]
fn check_rejects() {
    use SvOp::*;
    // wrong result sort
    assert!(check(Lit(bv(8, 3)), &[], 4).is_err());
    assert!(check(Add, &[8, 8], 9).is_err());
    assert!(check(Eq, &[8, 8], 8).is_err());
    assert!(check(Concat, &[3, 5], 15).is_err());
    // operand sorts
    assert!(check(Add, &[8, 4], 8).is_err());
    assert!(check(Ite, &[2, 8, 8], 8).is_err());
    assert!(check(Ite, &[1, 8, 4], 8).is_err());
    assert!(check(Add, &[0, 0], 0).is_err());
    // arity
    assert!(check(Id, &[], 8).is_err());
    assert!(check(Not, &[8, 8], 8).is_err());
    assert!(check(Lit(bv(8, 3)), &[8], 8).is_err());
    // parameters
    assert!(check(Extract { hi: 8, lo: 0 }, &[8], 9).is_err());
    assert!(check(Extract { hi: 2, lo: 3 }, &[8], 1).is_err());
    assert!(check(ZExt(4), &[8], 4).is_err());
    assert!(check(SExt(4), &[8], 4).is_err());
    // degrees: only ZERO writes a derivative, and nothing reads one
    assert!(check_degree(Zero(Sort(8)), &[], (8, 0)).is_err());
    assert!(check_degree(Id, &[8], (8, 1)).is_err());
    let ok = |s: usize, d: u8| Ok::<_, String>((Sort(s), d));
    assert!(Id.check([ok(8, 1)], [ok(8, 0)]).is_err());
    // number of writes
    assert!(
        Add.check([ok(8, 0), ok(8, 0)], [ok(8, 0), ok(8, 0)])
            .is_err()
    );
    assert!(Add.check([ok(8, 0), ok(8, 0)], []).is_err());
}

#[test]
fn bitvec_construction() {
    assert!(BitVec::new(8, 255u32).is_some());
    assert!(BitVec::new(8, 256u32).is_none());
    assert!(BitVec::new(0, 0u32).is_none());
    assert_eq!(BitVec::wrap(8, 300u32), bv(8, 44));
    assert_eq!(BitVec::from_signed(8, -1), bv(8, 255));
    assert_eq!(BitVec::from_signed(8, -128), bv(8, 128));
    assert_eq!(bv(8, 128).signed(), (-128).into());
    assert_eq!(bv(8, 3).to_string(), "8'd3");
}

#[test]
fn eval_edge_cases() {
    use SvOp::*;
    assert_eq!(ev(Neg, &[bv(8, 0)]), bv(8, 0));
    assert_eq!(ev(Neg, &[bv(8, 128)]), bv(8, 128));
    assert_eq!(ev(Not, &[bv(4, 0b1010)]), bv(4, 0b0101));
    assert_eq!(ev(RedAnd, &[bv(4, 0b1111)]), bv(1, 1));
    assert_eq!(ev(RedAnd, &[bv(4, 0b1011)]), bv(1, 0));
    assert_eq!(ev(RedOr, &[bv(4, 0)]), bv(1, 0));
    assert_eq!(ev(RedXor, &[bv(4, 0b1011)]), bv(1, 1));
    assert_eq!(ev(Ite, &[bv(1, 1), bv(8, 1), bv(8, 2)]), bv(8, 1));
    assert_eq!(ev(Ite, &[bv(1, 0), bv(8, 1), bv(8, 2)]), bv(8, 2));
    assert_eq!(ev(Extract { hi: 7, lo: 4 }, &[bv(8, 0xab)]), bv(4, 0xa));
    assert_eq!(ev(Extract { hi: 3, lo: 0 }, &[bv(8, 0xab)]), bv(4, 0xb));
    assert_eq!(
        ev(Concat, &[bv(3, 0b101), bv(5, 0b00011)]),
        bv(8, 0b101_00011)
    );
    assert_eq!(ev(ZExt(16), &[bv(8, 0x80)]), bv(16, 0x0080));
    assert_eq!(ev(SExt(16), &[bv(8, 0x80)]), bv(16, 0xff80));
    assert_eq!(ev(SExt(16), &[bv(8, 0x7f)]), bv(16, 0x007f));
    // a shift amount wider than the operand, and one beyond `usize`
    let huge = BitVec::wrap(200, BigUint::from(1u8) << 150);
    assert_eq!(ev(Shl, &[bv(8, 0xff), bv(16, 1000)]), bv(8, 0));
    assert_eq!(ev(LShr, &[bv(8, 0xff), huge.clone()]), bv(8, 0));
    assert_eq!(ev(AShr, &[bv(8, 0x80), huge.clone()]), bv(8, 0xff));
    assert_eq!(ev(AShr, &[bv(8, 0x40), huge]), bv(8, 0));
    // wider than any machine integer
    let max = BitVec::wrap(200, (BigUint::from(1u8) << 200) - 1u8);
    assert_eq!(
        ev(Add, &[max.clone(), BitVec::wrap(200, 1u8)]),
        BitVec::zero(200)
    );
    assert_eq!(ev(SLt, &[max, BitVec::zero(200)]), bv(1, 1));
}

#[test]
fn eval_rejects() {
    assert!(eval(&SvOp::Havoc(Sort(8)), &[]).is_err());
    assert!(eval(&SvOp::Add, &[bv(8, 1), bv(4, 1)]).is_err());
    for op in [SvOp::UDiv, SvOp::URem, SvOp::SDiv, SvOp::SRem] {
        assert!(eval(&op, &[bv(8, 7), bv(8, 0)]).is_err(), "{op}");
    }
}

/// All 8-bit operand pairs, against `u8`/`i8`.
#[test]
fn eval_exhaustive_8bit() {
    use SvOp::*;
    let b = |x: bool| bv(1, x as u128);
    for x in 0..=255u8 {
        let a = bv(8, x.into());
        assert_eq!(
            ev(Neg, std::slice::from_ref(&a)),
            bv(8, x.wrapping_neg().into())
        );
        for y in 0..=255u8 {
            let c = bv(8, y.into());
            let (sx, sy) = (x as i8, y as i8);
            let args = [a.clone(), c.clone()];
            assert_eq!(ev(Add, &args), bv(8, x.wrapping_add(y).into()));
            assert_eq!(ev(Sub, &args), bv(8, x.wrapping_sub(y).into()));
            assert_eq!(ev(Mul, &args), bv(8, x.wrapping_mul(y).into()));
            assert_eq!(ev(And, &args), bv(8, (x & y).into()));
            assert_eq!(ev(Or, &args), bv(8, (x | y).into()));
            assert_eq!(ev(Xor, &args), bv(8, (x ^ y).into()));
            assert_eq!(ev(Eq, &args), b(x == y));
            assert_eq!(ev(Ne, &args), b(x != y));
            assert_eq!(ev(ULt, &args), b(x < y));
            assert_eq!(ev(ULe, &args), b(x <= y));
            assert_eq!(ev(SLt, &args), b(sx < sy));
            assert_eq!(ev(SLe, &args), b(sx <= sy));
            // shifts: the amount `y` is unsigned, and `≥ 8` shifts every bit out
            let k = y as u32;
            let shl = x.checked_shl(k).unwrap_or(0);
            let lshr = x.checked_shr(k).unwrap_or(0);
            let ashr = sx.checked_shr(k).unwrap_or(if sx < 0 { -1 } else { 0 }) as u8;
            if let (Some(q), Some(r)) = (x.checked_div(y), x.checked_rem(y)) {
                // Rust's `/` and `%` round toward zero like SystemVerilog's;
                // `-128 / -1` overflows `i8` and wraps
                assert_eq!(ev(UDiv, &args), bv(8, q.into()));
                assert_eq!(ev(URem, &args), bv(8, r.into()));
                assert_eq!(ev(SDiv, &args), bv(8, (sx.wrapping_div(sy) as u8).into()));
                assert_eq!(ev(SRem, &args), bv(8, (sx.wrapping_rem(sy) as u8).into()));
            }
            assert_eq!(ev(Shl, &args), bv(8, shl.into()));
            assert_eq!(ev(LShr, &args), bv(8, lshr.into()));
            assert_eq!(ev(AShr, &args), bv(8, ashr.into()));
        }
    }
}
