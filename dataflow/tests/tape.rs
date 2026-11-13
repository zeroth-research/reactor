//! Tapes: constructors, operations, conditions, and the stochasticity
//! invariant, checked by running the entries on concrete inputs.

use base::boundary::{Boundary, Mono, Poly};
use base::semiring::{Probability, Semiring};
use dataflow::diagram::{Block, Term, Wire};
use dataflow::tape::Tape;
use std::collections::HashMap;
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

fn poly<const N: usize>(sorts: [Sort; N]) -> Poly<Sort> {
    Mono::from_iter(sorts).into()
}

fn u() -> Poly<Sort> {
    poly([R])
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

/// The tape of the block that keeps the first `keep` of `sorts` and
/// discards the others.
fn keep<const N: usize>(sorts: [Sort; N], keep: usize) -> T {
    let ws = sorts.map(Wire::scalar);
    Tape::lift(Block::new(ws, ws[..keep].to_vec(), []).unwrap()).unwrap()
}

fn seq(tapes: &[T]) -> T {
    let mut it = tapes.iter();
    let first = it.next().unwrap().clone();
    it.fold(first, |t, s| Tape::compose(&t, s).unwrap())
}

fn sum(tapes: &[T]) -> T {
    let mut it = tapes.iter();
    let first = it.next().unwrap().clone();
    it.fold(first, |t, s| Tape::sum(&t, s))
}

/// An entry, up to wire renaming: its weight and the instructions of its
/// block.
type Shape = (Q, Vec<String>);

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
                            let ops = e.block().iter().map(|t| t.itype().to_string()).collect();
                            (*e.weight(), ops)
                        })
                        .collect();
                    cell.sort();
                    cell
                })
                .collect()
        })
        .collect()
}

fn ops(names: &[&str]) -> Vec<String> {
    names.iter().map(|s| s.to_string()).collect()
}

//============================================================
// Running entries
//============================================================

/// Runs a block on scalar inputs (booleans as `0` and `1`): its outputs, or
/// nothing if an `assume` fails. Only the itypes used in these tests are
/// supported.
fn run(block: &Block<LRA>, inputs: &[f64]) -> Option<Vec<f64>> {
    assert_eq!(block.read().len(), inputs.len());
    let mut val: HashMap<usize, f64> = HashMap::new();
    for (w, x) in block.read().iter().zip(inputs) {
        val.insert(w.id(), *x);
    }
    for term in block.iter() {
        let r: Vec<f64> = term.read().iter().map(|w| val[&w.id()]).collect();
        let bit = |b: bool| if b { 1.0 } else { 0.0 };
        let out = match term.itype() {
            LRA::Id() => r[0],
            LRA::ReLU() => r[0].max(0.0),
            LRA::Sub() => r[0] - r[1],
            LRA::Lt() => bit(r[0] < r[1]),
            LRA::Ite() => {
                if r[0] != 0.0 {
                    r[1]
                } else {
                    r[2]
                }
            }
            LRA::Assume(p) => {
                if (r[0] != 0.0) != *p {
                    return None;
                }
                continue;
            }
            op => panic!("run: unsupported itype {op}"),
        };
        val.insert(term.write()[0].id(), out);
    }
    Some(block.write().iter().map(|w| val[&w.id()]).collect())
}

/// The subdistribution of column `col` on `inputs`: the row, outputs and
/// weight of every entry whose `assume`s pass, sorted.
fn dist(t: &T, col: usize, inputs: &[f64]) -> Vec<(usize, Vec<f64>, Q)> {
    let mut d: Vec<_> = (0..t.rows())
        .flat_map(|i| t.cell(i, col).iter().map(move |e| (i, e)))
        .filter_map(|(i, e)| run(e.block(), inputs).map(|out| (i, out, *e.weight())))
        .collect();
    d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    d
}

fn mass(t: &T, col: usize, inputs: &[f64]) -> Q {
    dist(t, col, inputs)
        .iter()
        .fold(Q::zero(), |m, (_, _, w)| m + *w)
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
            [vec![(q(1, 1), vec![])], vec![]],
            [vec![], vec![(q(1, 1), vec![])]],
        ]
    );
    assert_well_typed(&id);
}

#[test]
fn merge_and_splits_have_the_expected_types() {
    let uu = u() + u();
    let merge = T::merge(u());
    let ps = T::prob_split(u(), q(1, 3)).unwrap();
    let gs = T::guard_split(u());
    assert_eq!((merge.dom(), merge.cod()), (uu.clone(), u()));
    assert_eq!((ps.dom(), ps.cod()), (u(), uu.clone()));
    assert_eq!((gs.dom(), gs.cod()), (poly([R, B]), uu));
    assert_eq!(
        shape(&ps),
        [[vec![(q(1, 3), vec![])]], [vec![(q(2, 3), vec![])]]]
    );
    assert_eq!(
        shape(&gs),
        [
            [vec![(q(1, 1), ops(&["Assume(true)"]))]],
            [vec![(q(1, 1), ops(&["Assume(false)"]))]]
        ]
    );
    for t in [&merge, &ps, &gs] {
        assert_well_typed(t);
    }
}

#[test]
fn guard_split_consumes_the_boolean() {
    let sig: Poly<Sort> = Mono::from_iter([R]) + Mono::from_iter([R, R]);
    let gs = T::guard_split(sig.clone());
    assert_eq!(
        gs.dom(),
        Mono::from_iter([R, B]) + Mono::from_iter([R, R, B])
    );
    assert_eq!(gs.cod(), sig.clone() + sig);
    assert_well_typed(&gs);
    // summand `R ⊗ B` goes to row 0 or 2, summand `R ⊗ R ⊗ B` to 1 or 3
    assert_eq!(dist(&gs, 0, &[5.0, 1.0]), [(0, vec![5.0], q(1, 1))]);
    assert_eq!(dist(&gs, 0, &[5.0, 0.0]), [(2, vec![5.0], q(1, 1))]);
    assert_eq!(
        dist(&gs, 1, &[5.0, 6.0, 1.0]),
        [(1, vec![5.0, 6.0], q(1, 1))]
    );
    assert_eq!(
        dist(&gs, 1, &[5.0, 6.0, 0.0]),
        [(3, vec![5.0, 6.0], q(1, 1))]
    );
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
    assert_eq!(shape(&one), [[vec![(q(1, 1), vec![])]], [vec![]]]);
    let zero = T::prob_split(u(), q(0, 1)).unwrap();
    assert_eq!(shape(&zero), [[vec![]], [vec![(q(1, 1), vec![])]]]);
    // followed by a merge, either is the identity
    for t in [one, zero] {
        assert_eq!(shape(&seq(&[t, T::merge(u())])), shape(&T::id(u())));
    }
}

#[test]
fn branch_rejects_non_boolean_conditions() {
    let (x, y) = (Wire::scalar(R), Wire::scalar(R));
    // writes a real
    let relu = Term::function(LRA::ReLU(), [y], [x]).unwrap();
    assert!(T::branch(&Block::new([x], [y], [relu]).unwrap()).is_err());
    // writes two booleans
    let b = Wire::scalar(B);
    assert!(T::branch(&Block::new([b], [b, b], []).unwrap()).is_err());
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

#[test]
fn lift_of_a_partial_block_loses_mass() {
    // `R ⊗ B → R` that runs only when the boolean is true
    let (x, b) = (Wire::scalar(R), Wire::scalar(B));
    let assume = Term::function(LRA::Assume(true), Vec::<Wire<Sort>>::new(), [b]).unwrap();
    let t = T::lift(Block::new([x, b], [x], [assume]).unwrap()).unwrap();
    assert_eq!(mass(&t, 0, &[3.0, 1.0]), q(1, 1));
    assert_eq!(mass(&t, 0, &[3.0, 0.0]), q(0, 1));
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
        [[vec![(q(1, 2), ops(&["Id"])), (q(1, 2), ops(&["ReLU"]))]]]
    );
    assert_well_typed(&t);
    assert_eq!(
        dist(&t, 0, &[-2.0]),
        [(0, vec![-2.0], q(1, 2)), (0, vec![0.0], q(1, 2))]
    );
}

#[test]
fn a_cell_mixes_guarded_branches() {
    // guard_split ; (f ⊕ g) ; merge  =  b·f + ¬b·g, in a single cell
    let t = seq(&[T::guard_split(u()), sum(&[f(), g()]), T::merge(u())]);
    assert_eq!((t.dom(), t.cod()), (poly([R, B]), u()));
    assert_eq!(
        shape(&t),
        [[vec![
            (q(1, 1), ops(&["Assume(false)", "Id"])),
            (q(1, 1), ops(&["Assume(true)", "ReLU"])),
        ]]]
    );
    // the raw weights sum to 2, but only one entry runs on each input
    assert_eq!(dist(&t, 0, &[-2.0, 1.0]), [(0, vec![0.0], q(1, 1))]);
    assert_eq!(dist(&t, 0, &[-2.0, 0.0]), [(0, vec![-2.0], q(1, 1))]);
}

#[test]
fn compose_is_associative() {
    let rb = poly([R, B]);
    let a = T::prob_split(rb.clone(), q(1, 3)).unwrap();
    let b = sum(&[T::guard_split(u()), T::id(rb.clone())]);
    let c = sum(&[T::merge(u()), T::id(rb)]);
    let left = seq(&[seq(&[a.clone(), b.clone()]), c.clone()]);
    let right = seq(&[a, seq(&[b, c])]);
    assert_eq!(shape(&left), shape(&right));
    assert_well_typed(&left);
}

#[test]
fn operations_keep_tapes_stochastic() {
    let rb = poly([R, B]);
    let rbb = poly([R, B, B]);
    let t = seq(&[
        T::prob_split(rbb.clone(), q(1, 3)).unwrap(),
        sum(&[T::guard_split(rb.clone()), keep([R, B, B], 2)]),
        sum(&[T::guard_split(u()), T::guard_split(u()), T::id(rb)]),
        sum(&[f(), g(), T::merge(u()), keep([R, B], 1)]),
        sum(&[T::merge(u()), T::swap(u(), u())]),
    ]);
    assert_eq!((t.dom(), t.cod()), (rbb, u() + u() + u()));
    assert_well_typed(&t);
    for b in [0.0, 1.0] {
        for c in [0.0, 1.0] {
            assert_eq!(mass(&t, 0, &[-1.0, b, c]), q(1, 1), "b = {b}, c = {c}");
        }
    }
}

//============================================================
// Conditions on internal values
//============================================================

/// The tape of a block `R ⊗ R → …` built by `body` from the wires `x, y`.
fn block2(
    body: impl FnOnce(Wire<Sort>, Wire<Sort>) -> (Vec<Wire<Sort>>, Vec<Term<LRA>>),
) -> Block<LRA> {
    let (x, y) = (Wire::scalar(R), Wire::scalar(R));
    let (write, terms) = body(x, y);
    Block::new([x, y], write, terms).unwrap()
}

/// `c := x < y`, the condition `R ⊗ R → B`.
fn less() -> Block<LRA> {
    block2(|x, y| {
        let c = Wire::scalar(B);
        (
            vec![c],
            vec![Term::function(LRA::Lt(), [c], [x, y]).unwrap()],
        )
    })
}

/// `z := op(args(x, y))`, keeping `y`: a tape `R ⊗ R → R ⊗ R`.
fn assign(op: LRA, args: fn(Wire<Sort>, Wire<Sort>) -> Vec<Wire<Sort>>) -> T {
    T::lift(block2(|x, y| {
        let z = Wire::scalar(R);
        (
            vec![z, y],
            vec![Term::function(op, [z], args(x, y)).unwrap()],
        )
    }))
    .unwrap()
}

#[test]
fn branch_on_an_internal_condition() {
    // if x < y {
    //     if flip(0.3) { z := relu(x) } else { z := x }
    // } else {
    //     z := x - y
    // }
    // return if z < y then z else y      (a data-level conditional: `ite`)
    let rr = poly([R, R]);
    let then = seq(&[
        T::prob_split(rr.clone(), q(3, 10)).unwrap(),
        sum(&[
            assign(LRA::ReLU(), |x, _| vec![x]),
            assign(LRA::Id(), |x, _| vec![x]),
        ]),
        T::merge(rr.clone()),
    ]);
    let other = assign(LRA::Sub(), |x, y| vec![x, y]);
    let min = T::lift(block2(|z, y| {
        let (b, w) = (Wire::scalar(B), Wire::scalar(R));
        let terms = vec![
            Term::function(LRA::Lt(), [b], [z, y]).unwrap(),
            Term::function(LRA::Ite(), [w], [b, z, y]).unwrap(),
        ];
        (vec![w], terms)
    }))
    .unwrap();
    let t = seq(&[
        T::branch(&less()).unwrap(),
        sum(&[then, other]),
        T::merge(rr),
        min,
    ]);
    assert_eq!((t.dom(), t.cod()), (poly([R, R]), u()));
    assert_well_typed(&t);
    let tail = ["Lt", "Ite"];
    let entry = |w, head: &[&str]| (w, ops(&[head, &tail[..]].concat()));
    assert_eq!(
        shape(&t),
        // sorted by the derived order of `Q`, numerator first
        [[vec![
            entry(q(1, 1), &["Lt", "Assume(false)", "Sub"]),
            entry(q(3, 10), &["Lt", "Assume(true)", "ReLU"]),
            entry(q(7, 10), &["Lt", "Assume(true)", "Id"]),
        ]]]
    );
    // the raw weights sum to 2; each input runs a distribution
    assert_eq!(
        dist(&t, 0, &[-1.0, 2.0]),
        [(0, vec![-1.0], q(7, 10)), (0, vec![0.0], q(3, 10))]
    );
    assert_eq!(dist(&t, 0, &[5.0, 2.0]), [(0, vec![2.0], q(1, 1))]);
    assert_eq!(dist(&t, 0, &[3.0, 2.0]), [(0, vec![1.0], q(1, 1))]);
}

#[test]
fn branch_routes_to_different_summands() {
    // if x < y { return Ok(relu(x)) } else { return Err }:  R ⊗ R → R ⊕ I
    let ok = T::lift(block2(|x, _| {
        let r = Wire::scalar(R);
        (
            vec![r],
            vec![Term::function(LRA::ReLU(), [r], [x]).unwrap()],
        )
    }))
    .unwrap();
    let err = T::lift(block2(|_, _| (vec![], vec![]))).unwrap();
    let t = seq(&[T::branch(&less()).unwrap(), sum(&[ok, err])]);
    let unit: Poly<Sort> = Mono::from_iter([]).into();
    assert_eq!((t.dom(), t.cod()), (poly([R, R]), u() + unit));
    assert_well_typed(&t);
    assert_eq!(dist(&t, 0, &[-1.0, 2.0]), [(0, vec![0.0], q(1, 1))]);
    assert_eq!(dist(&t, 0, &[5.0, 2.0]), [(1, vec![], q(1, 1))]);
}
