//! Tapes: constructors, operations, and the stochasticity invariant.

use base::boundary::{Boundary, Mono, Poly};
use base::semiring::{Probability, Semiring};
use dataflow::diagram::{Block, Term, Wire};
use dataflow::tape::Tape;
use std::ops::{Add, Mul};
use theory::lra::{LRA, Sort};

//============================================================
// Exact weights
//============================================================

/// Non-negative rationals `num / den` in lowest terms, so that equality of
/// weights is exact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Q {
    num: i64,
    den: i64,
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 { a.abs() } else { gcd(b, a % b) }
}

fn q(num: i64, den: i64) -> Q {
    let g = gcd(num, den).max(1);
    Q {
        num: num / g,
        den: den / g,
    }
}

impl Add for Q {
    type Output = Q;
    fn add(self, o: Q) -> Q {
        q(self.num * o.den + o.num * self.den, self.den * o.den)
    }
}

impl Mul for Q {
    type Output = Q;
    fn mul(self, o: Q) -> Q {
        q(self.num * o.num, self.den * o.den)
    }
}

impl Semiring for Q {
    fn zero() -> Q {
        q(0, 1)
    }
    fn one() -> Q {
        q(1, 1)
    }
}

// The derived order compares numerators first, which is the numeric order
// only for equal denominators, so `is_probability` checks the fraction
// directly.
impl Probability for Q {
    fn complement(&self) -> Q {
        q(self.den - self.num, self.den)
    }
    fn is_probability(&self) -> bool {
        self.num >= 0 && self.num <= self.den
    }
}

//============================================================
// Fixtures
//============================================================

type T = Tape<LRA, Q>;

const R: Sort = Sort::Real([1, 1]);
const B: Sort = Sort::Bool([1, 1]);

fn u() -> Poly<Sort> {
    Mono::from_iter([R]).into()
}

/// The tape of the single block `y := op(x)`, `R → R`.
fn unary(op: LRA) -> T {
    let (x, y) = (Wire::scalar(R), Wire::scalar(R));
    let t = Term::function(op, [y], [x]).unwrap();
    Tape::lift(Block::new([x], [y], [t]).unwrap()).unwrap()
}

fn f() -> T {
    unary(LRA::ReLU())
}

fn g() -> T {
    unary(LRA::Id())
}

fn seq(tapes: &[T]) -> T {
    let mut it = tapes.iter();
    let first = it.next().unwrap().clone();
    it.fold(first, |t, s| Tape::compose(&t, s).unwrap())
}

fn sum(tapes: &[T]) -> T {
    let mut it = tapes.iter();
    let first = it.next().unwrap().clone();
    it.fold(first, |t, s| Tape::sum(&t, s).unwrap())
}

/// An entry, up to wire renaming: its weight, its guard (by wire id), and
/// the instructions of its block.
type Shape = (Q, Vec<(usize, bool)>, Vec<String>);

/// The tape up to wire renaming and the order of the entries in a cell.
fn shape(t: &T) -> Vec<Vec<Vec<Shape>>> {
    (0..t.rows())
        .map(|i| {
            (0..t.cols())
                .map(|j| {
                    let mut cell: Vec<Shape> = t
                        .cell(i, j)
                        .iter()
                        .map(|e| {
                            let mut guard: Vec<_> =
                                e.guard().iter().map(|(w, p)| (w.id(), *p)).collect();
                            guard.sort();
                            let ops = e.block().iter().map(|t| t.itype().to_string()).collect();
                            (*e.weight(), guard, ops)
                        })
                        .collect();
                    cell.sort();
                    cell
                })
                .collect()
        })
        .collect()
}

/// Checks the invariant: under every valuation of `wires`, the weights of
/// the enabled entries of every column sum to `1`.
fn assert_stochastic(t: &T, wires: &[Wire<Sort>]) {
    for bits in 0..(1u32 << wires.len()) {
        let val = |w: &Wire<Sort>| {
            let k = wires
                .iter()
                .position(|v| v == w)
                .expect("unknown guard wire");
            bits & (1 << k) != 0
        };
        for j in 0..t.cols() {
            let mass = (0..t.rows())
                .flat_map(|i| t.cell(i, j))
                .filter(|e| e.is_enabled(val))
                .fold(Q::zero(), |m, e| m + *e.weight());
            assert_eq!(mass, Q::one(), "column {j} under valuation {bits:#b}");
        }
    }
}

/// Every entry's block has the type of its cell.
fn assert_well_typed(t: &T) {
    let read: Vec<_> = t.read().iter().cloned().collect();
    let write: Vec<_> = t.write().iter().cloned().collect();
    for (i, j, e) in t.entries() {
        assert_eq!(e.block().read_type(), read[j]);
        assert_eq!(e.block().write_type(), write[i]);
    }
}

//============================================================
// Constructors
//============================================================

#[test]
fn id_is_diagonal() {
    let sig: Poly<Sort> = Mono::from_iter([R]) + Mono::from_iter([B, R]);
    let id = T::id(sig.clone());
    assert_eq!(id.dom(), sig);
    assert_eq!(id.cod(), sig);
    assert_eq!(
        shape(&id),
        [
            [vec![(q(1, 1), vec![], vec![])], vec![]],
            [vec![], vec![(q(1, 1), vec![], vec![])]],
        ]
    );
    assert_well_typed(&id);
}

#[test]
fn merge_and_splits_have_the_expected_types() {
    let uu = u() + u();
    let b = Wire::scalar(B);
    let merge = T::merge(u());
    let ps = T::prob_split(u(), q(1, 3)).unwrap();
    let gs = T::guard_split(u(), b).unwrap();
    assert_eq!((merge.dom(), merge.cod()), (uu.clone(), u()));
    assert_eq!((ps.dom(), ps.cod()), (u(), uu.clone()));
    assert_eq!((gs.dom(), gs.cod()), (u(), uu));
    assert_eq!(
        shape(&ps),
        [
            [vec![(q(1, 3), vec![], vec![])]],
            [vec![(q(2, 3), vec![], vec![])]]
        ]
    );
    assert_eq!(
        shape(&gs),
        [
            [vec![(q(1, 1), vec![(b.id(), true)], vec![])]],
            [vec![(q(1, 1), vec![(b.id(), false)], vec![])]]
        ]
    );
    for t in [&merge, &ps, &gs] {
        assert_well_typed(t);
        assert_stochastic(t, &[b]);
    }
}

#[test]
fn prob_split_rejects_non_probabilities() {
    assert!(T::prob_split(u(), q(3, 2)).is_err());
    assert!(T::prob_split(u(), Q { num: -1, den: 2 }).is_err());
    assert!(Tape::<LRA, f64>::prob_split(u(), f64::NAN).is_err());
}

#[test]
fn certain_prob_split_behaves_as_identity() {
    let one = T::prob_split(u(), q(1, 1)).unwrap();
    assert_eq!(shape(&one), [[vec![(q(1, 1), vec![], vec![])]], [vec![]]]);
    let zero = T::prob_split(u(), q(0, 1)).unwrap();
    assert_eq!(shape(&zero), [[vec![]], [vec![(q(1, 1), vec![], vec![])]]]);
    // followed by a merge, either is the identity
    for t in [one, zero] {
        assert_eq!(shape(&seq(&[t, T::merge(u())])), shape(&T::id(u())));
    }
}

#[test]
fn guard_split_rejects_non_boolean_guards() {
    assert!(T::guard_split(u(), Wire::scalar(R)).is_err());
    assert!(T::guard_split(u(), Wire::covector(B)).is_err());
}

#[test]
fn swap_is_an_involution() {
    let p: Poly<Sort> = Mono::from_iter([R]).into();
    let r: Poly<Sort> = Mono::from_iter([B]) + Mono::from_iter([R, R]);
    let sw = T::swap(p.clone(), r.clone());
    assert_eq!(
        (sw.dom(), sw.cod()),
        (p.clone() + r.clone(), r.clone() + p.clone())
    );
    assert_well_typed(&sw);
    let back = seq(&[sw, T::swap(r.clone(), p.clone())]);
    assert_eq!(shape(&back), shape(&T::id(p + r)));
}

#[test]
fn lift_rejects_derivatives() {
    let x = Wire::covector(R);
    assert!(T::lift(Block::new([x], [x], []).unwrap()).is_err());
}

//============================================================
// Operations
//============================================================

#[test]
fn compose_checks_boundaries() {
    let bb: Poly<Sort> = Mono::from_iter([B]).into();
    assert!(T::compose(&T::id(u()), &T::id(bb)).is_err());
    // `u ⊕ u` is not `u`
    assert!(T::compose(&T::id(u() + u()), &f()).is_err());
}

#[test]
fn id_is_the_unit_of_compose() {
    let t = seq(&[T::prob_split(u(), q(1, 4)).unwrap(), sum(&[f(), g()])]);
    assert_eq!(shape(&seq(&[T::id(u()), t.clone()])), shape(&t));
    assert_eq!(shape(&seq(&[t.clone(), T::id(u() + u())])), shape(&t));
}

#[test]
fn a_cell_mixes_probabilistic_branches() {
    // prob_split ; (f ⊕ g) ; merge  =  ½·f + ½·g, in a single cell
    let t = seq(&[
        T::prob_split(u(), q(1, 2)).unwrap(),
        sum(&[f(), g()]),
        T::merge(u()),
    ]);
    assert_eq!(
        shape(&t),
        [[vec![
            (q(1, 2), vec![], vec!["Id".into()]),
            (q(1, 2), vec![], vec!["ReLU".into()])
        ]]]
    );
    assert_well_typed(&t);
    assert_stochastic(&t, &[]);
}

#[test]
fn a_cell_mixes_guarded_branches() {
    // guard_split(b) ; (f ⊕ g) ; merge  =  b·f + ¬b·g, in a single cell
    let b = Wire::scalar(B);
    let t = seq(&[
        T::guard_split(u(), b).unwrap(),
        sum(&[f(), g()]),
        T::merge(u()),
    ]);
    assert_eq!(
        shape(&t),
        [[vec![
            (q(1, 1), vec![(b.id(), false)], vec!["Id".into()]),
            (q(1, 1), vec![(b.id(), true)], vec!["ReLU".into()]),
        ]]]
    );
    assert_stochastic(&t, &[b]);
}

#[test]
fn contradictory_guards_are_dropped() {
    let b = Wire::scalar(B);
    let gs = || T::guard_split(u(), b).unwrap();
    let t = seq(&[gs(), sum(&[gs(), gs()])]);
    let lit = |p| (q(1, 1), vec![(b.id(), p)], vec![]);
    // b ∧ b, b ∧ ¬b, ¬b ∧ b, ¬b ∧ ¬b: the mixed ones are never enabled
    assert_eq!(
        shape(&t),
        [[vec![lit(true)]], [vec![]], [vec![]], [vec![lit(false)]]]
    );
    assert_stochastic(&t, &[b]);
}

#[test]
fn compose_is_associative() {
    let w = Wire::scalar(B);
    let a = T::prob_split(u(), q(1, 3)).unwrap();
    let b = sum(&[f(), T::guard_split(u(), w).unwrap()]);
    let c = sum(&[T::merge(u()), g()]);
    let left = seq(&[seq(&[a.clone(), b.clone()]), c.clone()]);
    let right = seq(&[a, seq(&[b, c])]);
    assert_eq!(shape(&left), shape(&right));
    assert_well_typed(&left);
}

#[test]
fn operations_keep_tapes_stochastic() {
    let (b, c) = (Wire::scalar(B), Wire::scalar(B));
    let t = seq(&[
        T::prob_split(u(), q(1, 3)).unwrap(),
        sum(&[
            T::guard_split(u(), b).unwrap(),
            T::prob_split(u(), q(1, 2)).unwrap(),
        ]),
        sum(&[f(), g(), T::guard_split(u(), c).unwrap(), T::id(u())]),
        sum(&[T::id(u()), T::merge(u()), T::swap(u(), u())]),
        sum(&[T::merge(u()), T::guard_split(u(), b).unwrap(), T::id(u())]),
    ]);
    assert_eq!(t.cols(), 1);
    assert_eq!(t.rows(), 4);
    assert_well_typed(&t);
    assert_stochastic(&t, &[b, c]);
}

#[test]
fn guard_wires_must_stay_external() {
    let w = Wire::scalar(B);
    let bb: Poly<Sort> = Mono::from_iter([B]).into();
    // a block that reads the guard wire
    let reads_w = T::lift(Block::new([w], [w], []).unwrap()).unwrap();
    let gs = T::guard_split(bb.clone(), w).unwrap();
    assert!(T::sum(&reads_w, &gs).is_err());
    assert!(T::compose(&gs, &sum(&[reads_w, T::id(bb)])).is_err());
}
