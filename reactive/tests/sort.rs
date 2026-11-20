//! A sort of this test's own, with operations on it, alongside the sorts
//! and operations of a logic of `theory::tensor`, non-linear real
//! arithmetic: angles, whose rates of change are reals. The logic's sorts
//! and operations are closed enums, so the test joins them with its own in
//! enums of its own, [`Sort`] and [`Op`], and builds its terms and atoms over
//! those. The structural operations, which a logic leaves to whoever
//! builds atoms over it, are the test's own, on all of its sorts.

mod common;

use base::boundary::{Boundary, Poly};
use base::differentiable::{Differentiable, T};
use base::mono;
use common::matrix::Matrix;
use dataflow::diagram::{Block, Term, Wire};
use reactive::var::{Var, X, d};
use std::fmt;
use theory::tensor::{real, sort};
use theory::{Combinatorial, Differential, Sequential, Tensor};

//============================================================
// The sort of angles
//============================================================

/// Matrices of angles of the given `[rows, cols]` shape: points on the
/// circle, each given by an angle in radians up to whole turns.
#[derive(Clone, Copy, PartialEq, Debug)]
struct Angle([usize; 2]);

impl fmt::Display for Angle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [i, j] = self.0;
        write!(f, "Angle({i}, {j})")
    }
}

/// An angle turns at a rate in radians per unit of time, which is a real
/// and not an angle: the tangent of the circle is the real line. So the
/// tangent of angles is a sort of the theory, the rates of change of
/// reals, and so are the tangents of those.
impl Differentiable for Angle {
    type Tangent = sort::Real;

    fn tangent(self) -> sort::Real {
        sort::Real::with_degree(self.0, 1)
    }
}

/// The sorts of this test: those of real arithmetic, and angles.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Sort {
    Theory(real::Sort),
    Angle(Angle),
}

/// Each sort's tangent: the logic's for its sorts, and reals for angles.
impl Differentiable for Sort {
    type Tangent = Sort;

    fn tangent(self) -> Sort {
        match self {
            Sort::Theory(s) => Sort::Theory(T(s)),
            Sort::Angle(a) => Sort::Theory(T(a).into()),
        }
    }
}

impl fmt::Display for Sort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sort::Theory(s) => s.fmt(f),
            Sort::Angle(a) => a.fmt(f),
        }
    }
}

/// Reals, angles and booleans of the given shape, and the rates of change
/// of reals, which are also those of angles.
fn real(shape: [usize; 2]) -> Sort {
    Sort::Theory(sort::Real::new(shape).into())
}

fn angle(shape: [usize; 2]) -> Sort {
    Sort::Angle(Angle(shape))
}

fn boolean(shape: [usize; 2]) -> Sort {
    Sort::Theory(sort::Bool::new(shape).into())
}

fn rate(shape: [usize; 2]) -> Sort {
    Sort::Theory(sort::Real::with_degree(shape, 1).into())
}

/// The shape of single values: the values here are all single.
const SCALAR: [usize; 2] = [1, 1];

//============================================================
// The operations on angles
//============================================================

/// The operations on matrices of angles, entrywise, which read and write
/// angles and reals.
#[derive(Debug)]
enum Angles {
    /// The angles of the given matrix, in radians.
    Radians(Matrix),
    /// `θ + φ`, the angles `θ` turned by the reals `φ`, in radians.
    Turn([usize; 2]),
    /// `cos θ`, the reals of the cosines of the angles `θ`.
    Cos([usize; 2]),
}

impl Boundary<Sort> for Angles {
    fn dom(&self) -> Poly<Sort> {
        match *self {
            Angles::Radians(_) => Poly::unit(),
            Angles::Turn(s) => mono![angle(s), real(s)].into(),
            Angles::Cos(s) => mono![angle(s)].into(),
        }
    }

    fn cod(&self) -> Poly<Sort> {
        match *self {
            Angles::Radians(ref m) => mono![angle(m.shape())].into(),
            Angles::Turn(s) => mono![angle(s)].into(),
            Angles::Cos(s) => mono![real(s)].into(),
        }
    }
}

impl fmt::Display for Angles {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Angles::Radians(m) => write!(f, "Radians({m})"),
            Angles::Turn(_) => write!(f, "Turn"),
            Angles::Cos(_) => write!(f, "Cos"),
        }
    }
}

/// The operations of this test: those of non-linear real arithmetic, on
/// its sorts, those on angles, and the structural operations, on all sorts.
#[derive(Debug)]
enum Op {
    Real(real::NRA<Matrix>),
    Angle(Angles),
    /// An arbitrary value of the sort, reading nothing.
    Havoc(Sort),
    /// The value of the sort read.
    Skip(Sort),
    /// The zero of the sort, reading nothing: the rate of change of a value
    /// that does not change.
    Zero(Sort),
}

impl Op {
    /// The operation of real arithmetic `op`.
    fn real(op: impl Into<real::NRA<Matrix>>) -> Op {
        Op::Real(op.into())
    }
}

impl Boundary<Sort> for Op {
    fn dom(&self) -> Poly<Sort> {
        match self {
            Op::Real(op) => op.dom().map(Sort::Theory),
            Op::Angle(op) => op.dom(),
            Op::Havoc(_) | Op::Zero(_) => Poly::unit(),
            Op::Skip(s) => mono![*s].into(),
        }
    }

    fn cod(&self) -> Poly<Sort> {
        match self {
            Op::Real(op) => op.cod().map(Sort::Theory),
            Op::Angle(op) => op.cod(),
            Op::Havoc(s) | Op::Skip(s) | Op::Zero(s) => mono![*s].into(),
        }
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::Real(op) => op.fmt(f),
            Op::Angle(op) => op.fmt(f),
            Op::Havoc(_) => write!(f, "Havoc"),
            Op::Skip(_) => write!(f, "Skip"),
            Op::Zero(_) => write!(f, "Zero"),
        }
    }
}

// The structural operations that the constructors of atoms add, on any
// sort; `Zero` is asked of the sorts of rates of change only, which are
// never angles, as the tangent of angles is a real.

impl Combinatorial<Sort> for Op {
    fn havoc(range: &Sort) -> Self {
        Op::Havoc(*range)
    }
}

impl Sequential<Sort> for Op {
    fn skip(range: &Sort) -> Self {
        Op::Skip(*range)
    }
}

impl Differential<Sort> for Op {
    fn zero(range: &Sort) -> Self {
        Op::Zero(*range)
    }
}

type Atom = reactive::Atom<Op, Op, Op, Sort>;

/// The wire that each term of a block of structural operations writes,
/// with the sort of its operation, in the order of the wires.
fn structural(block: &Block<Op, Sort>) -> Vec<(Wire<Sort>, Sort)> {
    let mut terms: Vec<_> = block
        .iter()
        .map(|term| {
            let (Op::Havoc(sort) | Op::Skip(sort) | Op::Zero(sort)) = term.itype() else {
                panic!("{} is not a structural operation", term.itype());
            };
            let [w] = term.write() else {
                panic!("{} writes one wire", term.itype());
            };
            (*w, *sort)
        })
        .collect();
    terms.sort_by_key(|(w, _)| w.id());
    terms
}

//============================================================
// Tests
//============================================================

#[test]
fn the_rates_of_change_of_angles_are_reals() {
    let theta = Var::new(angle(SCALAR));
    assert_eq!(*X(theta).dtype(), angle(SCALAR));
    assert_eq!(*d(theta).dtype(), rate(SCALAR));
    assert_eq!(d(theta).dtype().to_string(), "T^1(Real(1, 1))");
    // and the rates of change of those, reals of degree 2
    let accel = Sort::Theory(sort::Real::with_degree(SCALAR, 2).into());
    assert_eq!(T(T(angle(SCALAR))), accel);
}

/// `c = cos(θ + φ) < 0`: the operations on angles read and write angles
/// and reals, and the operations of real arithmetic take on the reals
/// they write.
#[test]
fn operations_on_angles_and_of_real_arithmetic_compose() {
    let (theta, phi) = (Wire::new(angle(SCALAR)), Wire::new(real(SCALAR)));
    let (turned, cos, zero, c) = (
        Wire::new(angle(SCALAR)),
        Wire::new(real(SCALAR)),
        Wire::new(real(SCALAR)),
        Wire::new(boolean(SCALAR)),
    );

    let radians = Op::Angle(Angles::Radians(Matrix::new([[0.5]])));
    assert_eq!(radians.cod(), Poly::from(mono![angle(SCALAR)]));
    assert!(Term::constant(radians, [theta]).is_ok());
    let turn = Op::Angle(Angles::Turn(SCALAR));
    assert!(Term::function(turn, [turned], [theta, phi]).is_ok());
    assert!(Term::function(Op::Angle(Angles::Cos(SCALAR)), [cos], [turned]).is_ok());
    assert!(Term::constant(Op::real(real::Const::new(Matrix::zeros(SCALAR))), [zero]).is_ok());
    assert!(Term::function(Op::real(real::Lt::new(SCALAR)), [c], [cos, zero]).is_ok());
}

/// An angle is not a real: neither the operations of real arithmetic nor
/// those on angles take one for the other, nor an angle for its rate of change.
#[test]
fn angles_and_reals_do_not_mix() {
    let (theta, phi) = (Wire::new(angle(SCALAR)), Wire::new(real(SCALAR)));

    let add = Op::real(real::Add::new(SCALAR));
    assert_eq!(
        Term::function(add, [Wire::new(real(SCALAR))], [theta, theta]).unwrap_err(),
        "expected Add to read Real(1, 1) ⊗ Real(1, 1), got Angle(1, 1) ⊗ Angle(1, 1)"
    );
    let turn = Op::Angle(Angles::Turn(SCALAR));
    assert_eq!(
        Term::function(turn, [Wire::new(angle(SCALAR))], [phi, theta]).unwrap_err(),
        "expected Turn to read Angle(1, 1) ⊗ Real(1, 1), got Real(1, 1) ⊗ Angle(1, 1)"
    );
    let cos = Op::Angle(Angles::Cos(SCALAR));
    assert_eq!(
        Term::function(cos, [Wire::new(angle(SCALAR))], [theta]).unwrap_err(),
        "expected Cos to write Real(1, 1), got Angle(1, 1)"
    );

    let x = Var::new(angle(SCALAR));
    let radians = Op::Angle(Angles::Radians(Matrix::zeros(SCALAR)));
    assert_eq!(
        Term::constant(radians, [d(x)]).unwrap_err(),
        "expected Radians([[0]]) to write Angle(1, 1), got T^1(Real(1, 1))"
    );
}

/// The hand of a clock, `θ' = θ + step` from `θ = 0`; the constructor adds
/// the zero rate of change of `θ`, the `Zero` of a real rate.
#[test]
fn a_sequential_atom_over_angles_zeroes_their_real_rates() {
    let theta = Var::new(angle(SCALAR));
    let zero = Op::Angle(Angles::Radians(Matrix::zeros(SCALAR)));
    let init = [Term::constant(zero, [X(theta)]).unwrap()];
    let step = Wire::new(real(SCALAR));
    let update = [
        Term::constant(Op::real(real::Const::new(Matrix::new([[0.1]]))), [step]).unwrap(),
        Term::function(Op::Angle(Angles::Turn(SCALAR)), [X(theta)], [*theta, step]).unwrap(),
    ];

    let atom = Atom::sequential(&[theta], init, update).unwrap();
    assert_eq!(atom.update().len(), 2);
    assert_eq!(structural(atom.delay()), [(d(theta), rate(SCALAR))]);
}

/// A wheel of radius `0.5` rolling along `s`, `dθ = 2·ds`: real
/// arithmetic's `Scale` of degree 1 maps the real rate of `s` to that of
/// `θ`, which is real too. The constructor adds the update `θ' = θ`, a
/// `Skip` of an angle.
#[test]
fn a_differential_atom_computes_the_rate_of_an_angle_on_reals() {
    let (theta, s) = (Var::new(angle(SCALAR)), Var::new(real(SCALAR)));
    let zero = Op::Angle(Angles::Radians(Matrix::zeros(SCALAR)));
    let init = [Term::constant(zero, [X(theta)]).unwrap()];
    let two = Matrix::new([[2.0]]);
    let roll = Op::real(real::Scale::with_degree(two, 1));
    let delay = [Term::function(roll, [d(theta)], [d(s)]).unwrap()];

    let atom = Atom::differential(&[theta, s], init, delay).unwrap();
    assert_eq!(structural(atom.update()), [(X(theta), angle(SCALAR))]);
    assert_eq!(atom.update().read(), [*theta]);
}

/// `θ' = θ + cos θ`, without init or delay: the constructor havocs `θ` with
/// a `Havoc` of an angle, and zeroes its rate with a `Zero` of a real rate.
#[test]
fn a_jump_atom_havocs_angles_and_zeroes_their_rates() {
    let theta = Var::new(angle(SCALAR));
    let cos = Wire::new(real(SCALAR));
    let update = [
        Term::function(Op::Angle(Angles::Cos(SCALAR)), [cos], [theta]).unwrap(),
        Term::function(Op::Angle(Angles::Turn(SCALAR)), [X(theta)], [*theta, cos]).unwrap(),
    ];

    let atom = Atom::jump(&[theta], update).unwrap();
    assert_eq!(structural(atom.init()), [(X(theta), angle(SCALAR))]);
    assert_eq!(structural(atom.delay()), [(d(theta), rate(SCALAR))]);
}
