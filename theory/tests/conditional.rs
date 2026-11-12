//! `Conditional`: `ite` type-checks with a condition of the theory's
//! `boolean()` sort and rejects any other condition.

use theory::{Conditional, bv, lia, lra};

fn deg0<S>(s: S) -> Result<(S, u8), String> {
    Ok((s, 0))
}

fn check_ite<T: Conditional>(range: T::Sort, other: T::Sort)
where
    T::Sort: Copy,
{
    let b = T::boolean();
    let ite = T::ite(&range);
    assert!(
        ite.check([b, range, range].map(deg0), [range].map(deg0))
            .is_ok()
    );
    // the condition must be of the boolean sort
    assert!(
        ite.check([other, range, range].map(deg0), [range].map(deg0))
            .is_err()
    );
}

#[test]
fn lia_ite_reads_a_boolean() {
    check_ite::<lia::LIA>(lia::Sort::Int([2, 3]), lia::Sort::Int([1, 1]));
    check_ite::<lia::LIA>(lia::Sort::Bool([1, 1]), lia::Sort::Int([1, 1]));
}

#[test]
fn lra_ite_reads_a_boolean() {
    check_ite::<lra::LRA>(lra::Sort::Real([2, 3]), lra::Sort::Real([1, 1]));
}

#[test]
fn bv_ite_reads_a_bit() {
    check_ite::<bv::BV>(bv::Sort::BV(8, [2, 3]), bv::Sort::BV(8, [1, 1]));
}
