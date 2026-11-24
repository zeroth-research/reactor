//! Tests for the tape builder: the two example tapes rebuilt from
//! combinators, laws that must hold as equalities of frozen tapes, and
//! errors on boundaries that do not fit.
//!
//! Comments beside builder calls give the draft each call makes: its type,
//! or, where the strings carry named values, how the values flow through it.

mod common;

use base::boundary::{Boundary, Mono, Poly};
use common::*;
use dataflow::tape::{Builder, Draft, NodeKind, Tape};

fn m(sorts: &[Sort]) -> Mono<Sort> {
    Mono::from_iter(sorts.iter().copied())
}

/// The example of the module documentation, from combinators.
fn sampled_guard_built() -> Result<Tape<Sort, Op>, String> {
    let mut b = Builder::new(B);
    let rd = m(&[R, D]);

    // inl (x, d) => if gt0(x) then (let b = sample(d); if b then inl x else
    // inr ()) else inr ()
    let copy_x = b.wiring(rd.clone(), &[0, 1, 0])?; // x, d ↦ x, d, x
    let gt0 = b.lift(Op::Gt0)?; // R → B
    let gt0 = b.whisker(rd.clone(), gt0, m(&[])); // x, d, x ↦ x, d, gt0(x)
    let case = b.case(rd.clone()); // R ⊗ D ⊗ B → (R ⊗ D) ⊕ (R ⊗ D)
    let sample = b.lift(Op::Sample)?; // D → B
    let sample = b.whisker(m(&[R]), sample, m(&[])); // x, d ↦ x, sample(d)
    let case_r = b.case(m(&[R])); // R ⊗ B → R ⊕ R
    let keep = b.id(poly(&[&[R]])); // R → R
    let drop = b.discard(m(&[R])); // R → I
    let keep_or_drop = b.sum(keep, drop); // R ⊕ R → R ⊕ I
    let then = b.compose(sample, case_r)?;
    let then = b.compose(then, keep_or_drop)?; // R ⊗ D → R ⊕ I
    let drop_all = b.discard(rd); // R ⊗ D → I
    let no_r = b.initial(poly(&[&[R]])); // 0 → R
    let unit = b.id(poly(&[&[]])); // I → I
    let inr = b.sum(no_r, unit); // I → R ⊕ I
    let els = b.compose(drop_all, inr)?; // R ⊗ D → R ⊕ I
    let branches = b.sum(then, els); // (R ⊗ D) ⊕ (R ⊗ D) → R ⊕ I ⊕ R ⊕ I
    let join = b.merge(poly(&[&[R], &[]])); // R ⊕ I ⊕ R ⊕ I → R ⊕ I
    let left = b.compose(copy_x, gt0)?;
    let left = b.compose(left, case)?;
    let left = b.compose(left, branches)?;
    let left = b.compose(left, join)?; // R ⊗ D → R ⊕ I

    // inr () => inl zero()
    let zero = b.lift(Op::Lit("zero"))?; // I → R
    let keep = b.id(poly(&[&[R]])); // R → R
    let no_unit = b.initial(poly(&[&[]])); // 0 → I
    let inl = b.sum(keep, no_unit); // R → R ⊕ I
    let right = b.compose(zero, inl)?; // I → R ⊕ I

    let both = b.sum(left, right); // (R ⊗ D) ⊕ I → R ⊕ I ⊕ R ⊕ I
    let join = b.merge(poly(&[&[R], &[]])); // R ⊕ I ⊕ R ⊕ I → R ⊕ I
    let whole = b.compose(both, join)?; // (R ⊗ D) ⊕ I → R ⊕ I
    Ok(b.freeze(whole))
}

/// The example program, from combinators.
fn program_built() -> Result<Tape<Sort, Op>, String> {
    let mut b = Builder::new(B);
    let s = [R, R, R];

    // if x < y or x < z
    let guard = {
        let copies = b.wiring(m(&s), &[0, 1, 2, 0, 1, 0, 2])?; // x, y, z ↦ x, y, z, x, y, x, z
        let lt = b.lift(Op::Lt)?; // R ⊗ R → B
        let lt1 = b.whisker(m(&s), lt, m(&[R, R])); // x, y, z, x, y, x, z ↦ x, y, z, x<y, x, z
        let lt2 = b.whisker(m(&[R, R, R, B]), lt, m(&[])); // x, y, z, x<y, x, z ↦ x, y, z, x<y, x<z
        let or = b.lift(Op::Or)?; // B ⊗ B → B
        let or = b.whisker(m(&s), or, m(&[])); // x, y, z, x<y, x<z ↦ x, y, z, x<y or x<z
        let case = b.case(m(&s)); // R ⊗ R ⊗ R ⊗ B → (R ⊗ R ⊗ R) ⊕ (R ⊗ R ⊗ R)
        let guard = b.compose(copies, lt1)?;
        let guard = b.compose(guard, lt2)?;
        let guard = b.compose(guard, or)?;
        b.compose(guard, case)? // R ⊗ R ⊗ R → (R ⊗ R ⊗ R) ⊕ (R ⊗ R ⊗ R)
    };
    // new_x = x + 1.0
    let then = {
        let one = b.lift(Op::Lit("1.0"))?; // I → R
        let one = b.whisker(m(&s), one, m(&[])); // x, y, z ↦ x, y, z, 1.0
        let order = b.wiring(m(&[R, R, R, R]), &[0, 3, 1, 2])?; // x, y, z, 1.0 ↦ x, 1.0, y, z
        let add = b.lift(Op::Add)?; // R ⊗ R → R
        let add = b.whisker(m(&[]), add, m(&[R, R])); // x, 1.0, y, z ↦ x + 1.0, y, z
        let then = b.compose(one, order)?;
        b.compose(then, add)? // x, y, z ↦ x + 1.0, y, z
    };
    // new_x = 0.0
    let els = {
        let drop_x = b.wiring(m(&s), &[1, 2])?; // x, y, z ↦ y, z
        let zero = b.lift(Op::Lit("0.0"))?; // I → R
        let zero = b.whisker(m(&[]), zero, m(&[R, R])); // y, z ↦ 0.0, y, z
        b.compose(drop_x, zero)? // x, y, z ↦ 0.0, y, z
    };
    let branches = b.sum(then, els); // (R ⊗ R ⊗ R) ⊕ (R ⊗ R ⊗ R) → (R ⊗ R ⊗ R) ⊕ (R ⊗ R ⊗ R)
    let join = b.merge(poly(&[&s])); // (R ⊗ R ⊗ R) ⊕ (R ⊗ R ⊗ R) → R ⊗ R ⊗ R
    // with x = new_x: t = at_target = x == y or x == z, then the reward
    let tail = {
        let copies = b.wiring(m(&s), &[0, 0, 1, 0, 2])?; // x, y, z ↦ x, x, y, x, z
        let eq = b.lift(Op::Eq)?; // R ⊗ R → B
        let eq1 = b.whisker(m(&[R]), eq, m(&[R, R])); // x, x, y, x, z ↦ x, x==y, x, z
        let eq2 = b.whisker(m(&[R, B]), eq, m(&[])); // x, x==y, x, z ↦ x, x==y, x==z
        let or = b.lift(Op::Or)?; // B ⊗ B → B
        let or = b.whisker(m(&[R]), or, m(&[])); // x, x==y, x==z ↦ x, t
        let copy_at = b.wiring(m(&[R, B]), &[0, 1, 1])?; // x, t ↦ x, t, t
        let case = b.case(m(&[R, B])); // R ⊗ B ⊗ B → (R ⊗ B) ⊕ (R ⊗ B)
        let reward = |b: &mut Builder<Sort, Op>, lit| -> Result<Draft, String> {
            let value = b.lift(Op::Lit(lit))?; // I → R
            let value = b.whisker(m(&[R, B]), value, m(&[])); // x, t ↦ x, t, r
            let truncated = b.lift(Op::False)?; // I → B
            let truncated = b.whisker(m(&[R, B, R]), truncated, m(&[])); // x, t, r ↦ x, t, r, false
            let order = b.wiring(m(&[R, B, R, B]), &[0, 2, 1, 3])?; // x, t, r, false ↦ x, r, t, false
            let reward = b.compose(value, truncated)?;
            b.compose(reward, order) // x, t ↦ x, r, t, false
        };
        let hit = reward(&mut b, "1.0")?; // r = 1.0
        let miss = reward(&mut b, "0.0")?; // r = 0.0
        let rewards = b.sum(hit, miss); // (R ⊗ B) ⊕ (R ⊗ B) → (R ⊗ R ⊗ B ⊗ B) ⊕ (R ⊗ R ⊗ B ⊗ B)
        let join = b.merge(poly(&[&[R, R, B, B]])); // (R ⊗ R ⊗ B ⊗ B) ⊕ (R ⊗ R ⊗ B ⊗ B) → R ⊗ R ⊗ B ⊗ B
        let tail = b.compose(copies, eq1)?;
        let tail = b.compose(tail, eq2)?;
        let tail = b.compose(tail, or)?;
        let tail = b.compose(tail, copy_at)?;
        let tail = b.compose(tail, case)?;
        let tail = b.compose(tail, rewards)?;
        b.compose(tail, join)? // R ⊗ R ⊗ R → R ⊗ R ⊗ B ⊗ B
    };
    let whole = b.compose(guard, branches)?;
    let whole = b.compose(whole, join)?;
    let whole = b.compose(whole, tail)?; // R ⊗ R ⊗ R → R ⊗ R ⊗ B ⊗ B
    Ok(b.freeze(whole))
}

#[test]
fn rebuilds_the_sampled_guard() -> Result<(), String> {
    let expected = Tape::new(sampled_guard())?;
    assert_eq!(sampled_guard_built()?, expected);
    Ok(())
}

#[test]
fn rebuilds_the_program() -> Result<(), String> {
    let expected = Tape::new(program())?;
    assert_eq!(program_built()?, expected);
    Ok(())
}

#[test]
fn freezing_an_imported_tape_gives_it_back() -> Result<(), String> {
    for parts in [sampled_guard(), program()] {
        let tape = Tape::new(parts)?;
        let mut b = Builder::new(B);
        let draft = b.import(&tape)?;
        assert_eq!(b.freeze(draft), tape);
    }
    Ok(())
}

#[test]
fn drafts_have_boundaries() {
    let mut b: Builder<Sort, Op> = Builder::new(B);
    let case = b.case(m(&[R, D])); // R ⊗ D ⊗ B → (R ⊗ D) ⊕ (R ⊗ D)
    assert_eq!(b.draft(case).dom(), poly(&[&[R, D, B]]));
    assert_eq!(b.draft(case).cod(), poly(&[&[R, D], &[R, D]]));
    let frozen = b.freeze(case);
    assert_eq!(Boundary::dom(&frozen), b.draft(case).dom());
    assert_eq!(Boundary::cod(&frozen), b.draft(case).cod());
}

#[test]
fn identity_is_a_unit() -> Result<(), String> {
    let mut b = Builder::new(B);
    let lt = b.lift(Op::Lt)?; // R ⊗ R → B
    let t = b.whisker(m(&[R]), lt, m(&[])); // R ⊗ R ⊗ R → R ⊗ B
    let before = b.id(poly(&[&[R, R, R]])); // R ⊗ R ⊗ R → R ⊗ R ⊗ R
    let after = b.id(poly(&[&[R, B]])); // R ⊗ B → R ⊗ B
    let left = b.compose(before, t)?;
    let right = b.compose(t, after)?;
    assert_eq!(b.freeze(left), b.freeze(t));
    assert_eq!(b.freeze(right), b.freeze(t));
    Ok(())
}

#[test]
fn composition_is_associative() -> Result<(), String> {
    let mut b = Builder::new(B);
    let s = m(&[R, R]);
    let t1 = b.copy(s.clone()); // R ⊗ R → R ⊗ R ⊗ R ⊗ R
    let lt = b.lift(Op::Lt)?; // R ⊗ R → B
    let t2 = b.whisker(s.clone(), lt, m(&[])); // R ⊗ R ⊗ R ⊗ R → R ⊗ R ⊗ B
    let t3 = b.case(s); // R ⊗ R ⊗ B → (R ⊗ R) ⊕ (R ⊗ R)
    let first = b.compose(t1, t2)?;
    let left = b.compose(first, t3)?;
    let second = b.compose(t2, t3)?;
    let right = b.compose(t1, second)?;
    assert_eq!(b.freeze(left), b.freeze(right));
    Ok(())
}

#[test]
fn swapping_twice_is_the_identity() -> Result<(), String> {
    let mut b: Builder<Sort, Op> = Builder::new(B);
    let (p, q) = (poly(&[&[R], &[R, B]]), poly(&[&[D]]));
    let there = b.swap(p.clone(), q.clone()); // R ⊕ (R ⊗ B) ⊕ D → D ⊕ R ⊕ (R ⊗ B)
    let back = b.swap(q, p.clone()); // D ⊕ R ⊕ (R ⊗ B) → R ⊕ (R ⊗ B) ⊕ D
    let both = b.compose(there, back)?;
    let id = b.id(p + poly(&[&[D]])); // R ⊕ (R ⊗ B) ⊕ D → R ⊕ (R ⊗ B) ⊕ D
    assert_eq!(b.freeze(both), b.freeze(id));
    Ok(())
}

#[test]
fn crossing_twice_is_the_identity() -> Result<(), String> {
    let mut b: Builder<Sort, Op> = Builder::new(B);
    let (x, y) = (m(&[R, B]), m(&[D]));
    let there = b.cross(x.clone(), y.clone()); // R ⊗ B ⊗ D → D ⊗ R ⊗ B
    let back = b.cross(y, x); // D ⊗ R ⊗ B → R ⊗ B ⊗ D
    let both = b.compose(there, back)?;
    let id = b.id(poly(&[&[R, B, D]])); // R ⊗ B ⊗ D → R ⊗ B ⊗ D
    assert_eq!(b.freeze(both), b.freeze(id));
    Ok(())
}

#[test]
fn a_copy_with_one_side_discarded_is_the_identity() -> Result<(), String> {
    // copy ; (m ⊗ discard) = id = copy ; (discard ⊗ m)
    let mut b: Builder<Sort, Op> = Builder::new(B);
    let s = m(&[R, D]);
    let copy = b.copy(s.clone()); // x, d ↦ x, d, x, d
    let discard = b.discard(s.clone()); // R ⊗ D → I
    let keep_first = b.whisker(s.clone(), discard, m(&[])); // x, d, x', d' ↦ x, d
    let keep_second = b.whisker(m(&[]), discard, s); // x, d, x', d' ↦ x', d'
    let first = b.compose(copy, keep_first)?;
    let second = b.compose(copy, keep_second)?;
    let id = b.id(poly(&[&[R, D]])); // R ⊗ D → R ⊗ D
    assert_eq!(b.freeze(first), b.freeze(id));
    assert_eq!(b.freeze(second), b.freeze(id));
    Ok(())
}

#[test]
fn wiring_names_its_outputs() -> Result<(), String> {
    // with inputs x : R and d : D, output k carries input outputs[k]
    let mut b: Builder<Sort, Op> = Builder::new(B);
    let copy_x = b.wiring(m(&[R, D]), &[0, 1, 0])?; // x, d ↦ x, d, x
    assert_eq!(b.draft(copy_x).cod(), poly(&[&[R, D, R]]));
    let crossed = b.wiring(m(&[R, D]), &[1, 0])?; // x, d ↦ d, x
    let cross = b.cross(m(&[R]), m(&[D])); // x, d ↦ d, x
    assert_eq!(b.freeze(crossed), b.freeze(cross));
    Ok(())
}

#[test]
fn sum_and_composition_interchange() -> Result<(), String> {
    // (f ⊕ g) ; (h ⊕ k) = (f ; h) ⊕ (g ; k)
    let mut b = Builder::new(B);
    let lt = b.lift(Op::Lt)?; // R ⊗ R → B
    let f = b.whisker(m(&[R]), lt, m(&[])); // R ⊗ R ⊗ R → R ⊗ B
    let g = b.lift(Op::Lit("0.0"))?; // I → R
    let h = b.case(m(&[R])); // R ⊗ B → R ⊕ R
    let copy = b.copy(m(&[R])); // R → R ⊗ R
    let add = b.lift(Op::Add)?; // R ⊗ R → R
    let k = b.compose(copy, add)?; // R → R
    let fg = b.sum(f, g); // (R ⊗ R ⊗ R) ⊕ I → (R ⊗ B) ⊕ R
    let hk = b.sum(h, k); // (R ⊗ B) ⊕ R → R ⊕ R ⊕ R
    let left = b.compose(fg, hk)?;
    let fh = b.compose(f, h)?; // R ⊗ R ⊗ R → R ⊕ R
    let gk = b.compose(g, k)?; // I → R
    let right = b.sum(fh, gk);
    assert_eq!(b.freeze(left), b.freeze(right));
    Ok(())
}

/// `case ; (f ⊕ f) ; merge` and `discard the boolean ; f`, for an operation
/// `f : input → n`.
fn split_and_join(op: Op, input: Sort, n: &[Sort]) -> Result<[Tape<Sort, Op>; 2], String> {
    let mut b = Builder::new(B);
    let case = b.case(m(&[input])); // input ⊗ B → input ⊕ input
    let f = b.lift(op)?; // input → n
    let ff = b.sum(f, f); // input ⊕ input → n ⊕ n
    let join = b.merge(poly(&[n])); // n ⊕ n → n
    let left = b.compose(case, ff)?;
    let left = b.compose(left, join)?; // input ⊗ B → n
    let drop_b = b.wiring(m(&[input, B]), &[0])?; // input ⊗ B → input
    let right = b.compose(drop_b, f)?; // input ⊗ B → n
    Ok([b.freeze(left), b.freeze(right)])
}

#[test]
fn equal_branches_join() -> Result<(), String> {
    let [left, right] = split_and_join(Op::Gt0, R, &[B])?;
    assert_eq!(left, right);
    Ok(())
}

#[test]
fn equal_branches_join_even_when_they_sample() -> Result<(), String> {
    // each branch draws its own sample in the draft; both draws are made at
    // the same level on exclusive paths, so they are the same draw
    let [left, right] = split_and_join(Op::Sample, D, &[B])?;
    assert_eq!(left, right);
    Ok(())
}

#[test]
fn a_sample_is_drawn_only_where_it_is_used() -> Result<(), String> {
    // case(D) ; (sample ⊕ discard): the sample is drawn on the `hi` path only
    let mut b = Builder::new(B);
    let case = b.case(m(&[D])); // D ⊗ B → D ⊕ D
    let sample = b.lift(Op::Sample)?; // D → B
    let drop = b.discard(m(&[D])); // D → I
    let branches = b.sum(sample, drop); // D ⊕ D → B ⊕ I
    let whole = b.compose(case, branches)?; // D ⊗ B → B ⊕ I
    let tape = b.freeze(whole);
    let root = tape.arm(0).root();
    assert!(root.binds().is_empty());
    let (hi, lo) = root.children().unwrap();
    assert_eq!(hi.binds().len(), 1);
    assert!(lo.binds().is_empty());
    Ok(())
}

#[test]
fn two_samples_on_one_path_stay_independent() -> Result<(), String> {
    // (sample ⊗ sample) : D ⊗ D → B ⊗ B, from one distribution copied
    let mut b = Builder::new(B);
    let copy = b.copy(m(&[D])); // d ↦ d, d
    let sample = b.lift(Op::Sample)?; // D → B
    let first = b.whisker(m(&[]), sample, m(&[D])); // d, d ↦ sample(d), d
    let second = b.whisker(m(&[B]), sample, m(&[])); // a, d ↦ a, sample(d)
    let whole = b.compose(copy, first)?;
    let whole = b.compose(whole, second)?; // D → B ⊗ B
    let tape = b.freeze(whole);
    // two non-deterministic nodes, at levels 0 and 1
    assert_eq!(tape.nodes().len(), 2);
    assert_eq!(tape.arm(0).root().binds().len(), 2);
    Ok(())
}

#[test]
fn a_draw_made_on_both_branches_is_made_once() -> Result<(), String> {
    // case(D) ; (sample ⊕ sample) : D ⊗ B → B ⊕ B draws on both paths, with
    // the same distribution, so it is the same as drawing before the split
    let mut b = Builder::new(B);
    let case = b.case(m(&[D])); // D ⊗ B → D ⊕ D
    let sample = b.lift(Op::Sample)?; // D → B
    let both = b.sum(sample, sample); // D ⊕ D → B ⊕ B
    let after = b.compose(case, both)?; // D ⊗ B → B ⊕ B
    let first = b.whisker(m(&[]), sample, m(&[B])); // d, c ↦ sample(d), c
    let case_b = b.case(m(&[B])); // B ⊗ B → B ⊕ B
    let before = b.compose(first, case_b)?; // D ⊗ B → B ⊕ B
    let (after, before) = (b.freeze(after), b.freeze(before));
    assert_eq!(after, before);
    let root = after.arm(0).root();
    assert_eq!(root.binds().len(), 1);
    let (hi, lo) = root.children().unwrap();
    assert!(hi.binds().is_empty() && lo.binds().is_empty());
    Ok(())
}

#[test]
fn the_same_draw_at_different_depths_is_two_nodes() -> Result<(), String> {
    // D ⊗ D ⊗ B → B ⊕ I: on the `hi` path, sample the first distribution
    // and split on it, then sample the second; on the `lo` path, only sample
    // the second. Its level is 1 on one path and 0 on the other.
    let mut b = Builder::new(B);
    let sample = b.lift(Op::Sample)?; // D → B
    let case = b.case(m(&[D, D])); // d1, d2, c ↦ (d1, d2) ⊕ (d1, d2)
    let hi = {
        let first = b.whisker(m(&[]), sample, m(&[D])); // d1, d2 ↦ sample(d1), d2
        let order = b.cross(m(&[B]), m(&[D])); // a, d2 ↦ d2, a
        let case = b.case(m(&[D])); // d2, a ↦ d2 ⊕ d2
        let drop = b.discard(m(&[D])); // D → I
        let branches = b.sum(sample, drop); // d2 ⊕ d2 ↦ sample(d2) ⊕ ()
        let hi = b.compose(first, order)?;
        let hi = b.compose(hi, case)?;
        b.compose(hi, branches)? // D ⊗ D → B ⊕ I
    };
    let lo = {
        let second = b.wiring(m(&[D, D]), &[1])?; // d1, d2 ↦ d2
        let keep = b.id(poly(&[&[B]])); // B → B
        let no_unit = b.initial(poly(&[&[]])); // 0 → I
        let inl = b.sum(keep, no_unit); // B → B ⊕ I
        let lo = b.compose(second, sample)?;
        b.compose(lo, inl)? // D ⊗ D → B ⊕ I
    };
    let branches = b.sum(hi, lo); // (D ⊗ D) ⊕ (D ⊗ D) → B ⊕ I ⊕ B ⊕ I
    let join = b.merge(poly(&[&[B], &[]])); // B ⊕ I ⊕ B ⊕ I → B ⊕ I
    let whole = b.compose(case, branches)?;
    let whole = b.compose(whole, join)?; // D ⊗ D ⊗ B → B ⊕ I
    let tape = b.freeze(whole);
    let levels: Vec<u32> = tape
        .nodes()
        .iter()
        .filter_map(|n| match n.kind {
            NodeKind::NonDet { level, .. } => Some(level),
            _ => None,
        })
        .collect();
    assert_eq!(levels.len(), 3);
    assert_eq!(levels.iter().filter(|&&l| l == 0).count(), 2);
    Ok(())
}

#[test]
fn freezing_a_frozen_tape_gives_it_back() -> Result<(), String> {
    // Found by random testing. On both sides of a split on c, flip a and b,
    // and split on a. The two hi-hi paths join, and flip once more there.
    // The two hi-hi ends become one route only once the flips of both sides
    // are merged, which an analysis made before the merge does not see: it
    // bound the last flip at the root.
    let mut b = Builder::new(B);
    let flip = b.lift(Op::Flip)?; // I → B
    let side = |b: &mut Builder<Sort, Op>, keep_lo: bool| -> Result<Draft, String> {
        let second = b.whisker(m(&[B]), flip, m(&[])); // a ↦ a, b
        let order = b.cross(m(&[B]), m(&[B])); // a, b ↦ b, a
        let case = b.case(m(&[B])); // b, a ↦ b ⊕ b
        let keep = b.id(poly(&[&[B]])); // B → B
        let lo = if keep_lo {
            keep
        } else {
            b.discard(m(&[B])) // B → I
        };
        let branches = b.sum(keep, lo);
        let side = b.compose(flip, second)?;
        let side = b.compose(side, order)?;
        let side = b.compose(side, case)?;
        b.compose(side, branches) // I → B ⊕ B, or I → B ⊕ I
    };
    let hi = side(&mut b, true)?; // I → B ⊕ B
    let lo = side(&mut b, false)?; // I → B ⊕ I
    let case = b.case(m(&[])); // B → I ⊕ I
    let both = b.sum(hi, lo); // I ⊕ I → B ⊕ B ⊕ B ⊕ I
    let order = {
        let first = b.id(poly(&[&[B]])); // B → B
        let swap = b.swap(poly(&[&[B]]), poly(&[&[B]])); // B ⊕ B → B ⊕ B
        let last = b.id(poly(&[&[]])); // I → I
        let order = b.sum(first, swap);
        b.sum(order, last) // hi-hi, hi-lo, lo-hi, lo-lo ↦ hi-hi, lo-hi, hi-lo, lo-lo
    };
    let join = {
        let hi_hi = b.merge(poly(&[&[B]])); // B ⊕ B → B
        let rest = b.id(poly(&[&[B], &[]])); // B ⊕ I → B ⊕ I
        b.sum(hi_hi, rest) // B ⊕ B ⊕ B ⊕ I → B ⊕ B ⊕ I
    };
    let draw = {
        let flip_again = b.whisker(m(&[B]), flip, m(&[])); // b ↦ b, flip
        let rest = b.id(poly(&[&[B], &[]])); // B ⊕ I → B ⊕ I
        b.sum(flip_again, rest) // B ⊕ B ⊕ I → (B ⊗ B) ⊕ B ⊕ I
    };
    let whole = b.compose(case, both)?;
    let whole = b.compose(whole, order)?;
    let whole = b.compose(whole, join)?;
    let whole = b.compose(whole, draw)?; // B → (B ⊗ B) ⊕ B ⊕ I

    let tape = b.freeze(whole);
    // only the flips a and b are drawn before the split on c
    assert_eq!(tape.arm(0).root().binds().len(), 2);
    let again = b.import(&tape)?;
    assert_eq!(b.freeze(again), tape);
    Ok(())
}

#[test]
fn rejects_misfits() -> Result<(), String> {
    let mut b = Builder::new(B);
    let gt0 = b.lift(Op::Gt0)?; // R → B
    let lt = b.lift(Op::Lt)?; // R ⊗ R → B
    assert!(b.compose(gt0, lt).unwrap_err().contains("cannot feed"));
    assert!(b.wiring(m(&[R]), &[1]).unwrap_err().contains("no input 1"));
    Ok(())
}

#[test]
#[should_panic(expected = "another builder")]
fn rejects_drafts_of_another_builder() {
    let mut first: Builder<Sort, Op> = Builder::new(B);
    let mut second: Builder<Sort, Op> = Builder::new(B);
    let _ = first.lift(Op::Gt0); // the first draft of `first`
    let foreign = second.lift(Op::Sample).unwrap(); // the first draft of `second`
    first.freeze(foreign);
}

#[test]
fn rejects_operations_on_sums() {
    let mut b: Builder<Sort, Op> = Builder::new(B);
    // choice : B → I ⊕ I has a sum as codomain
    assert!(b.lift(Op::Choice).unwrap_err().contains("single product"));
}

#[test]
fn freezing_leaves_the_draft_usable() -> Result<(), String> {
    let mut b = Builder::new(B);
    let gt0 = b.lift(Op::Gt0)?; // R → B
    let first = b.freeze(gt0);
    let case = b.case(m(&[])); // B → I ⊕ I
    let more = b.compose(gt0, case)?; // R → I ⊕ I
    let _ = b.freeze(more);
    assert_eq!(b.freeze(gt0), first);
    Ok(())
}

#[test]
fn empty_domain_has_no_arms() {
    let mut b: Builder<Sort, Op> = Builder::new(B);
    let none = b.initial(Poly::from(m(&[R]))); // 0 → R
    let tape = b.freeze(none);
    assert_eq!(tape.arms().len(), 0);
    assert_eq!(Boundary::cod(&tape), poly(&[&[R]]));
}
