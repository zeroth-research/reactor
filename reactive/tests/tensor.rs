//! Operations and atoms over a tensor type of the tests' own,
//! [`Matrix`]: the operations of `theory::tensor` are generic over
//! `theory::Tensor`, so any representation of matrices can back them, here
//! with no torch.

mod common;

use base::boundary::{Boundary, Poly};
use base::differentiable::T;
use common::matrix::Matrix;
use dataflow::diagram::{Block, Term, Wire};
use reactive::var::{Var, X, d};
use theory::tensor::{Havoc, Skip, Zero, bool, real, sort};

/// The operations of all theories, holding their data in `Matrix`es.
type Op = theory::tensor::Any<Matrix>;
type Atom = reactive::Atom<Op, Op, Op, sort::Any>;

fn real(op: impl Into<real::NRAT<Matrix>>) -> Op {
    Op::Real(op.into())
}

fn boolean(op: impl Into<bool::Prop<Matrix>>) -> Op {
    Op::Bool(op.into())
}

/// Columns of two reals: the sort of most values here.
fn vec2() -> sort::Any {
    sort::Real::new([2, 1]).into()
}

/// Columns of two booleans.
fn bool2() -> sort::Any {
    sort::Bool::new([2, 1]).into()
}

/// A single boolean, as the condition of a choice.
fn bool1() -> sort::Any {
    sort::Bool::new([1, 1]).into()
}

/// The identity on columns of two reals, and the zero column.
fn identity() -> Matrix {
    Matrix::new([[1.0, 0.0], [0.0, 1.0]])
}

fn zeros() -> Matrix {
    Matrix::new([[0.0], [0.0]])
}

/// The wire that each term of a block of structural operations writes,
/// with the sort of its operation, in the order of the wires.
fn structural(block: &Block<Op, sort::Any>) -> Vec<(Wire<sort::Any>, sort::Any)> {
    let mut terms: Vec<_> = block
        .iter()
        .map(|term| {
            let s = match term.itype() {
                Op::Havoc(op) => op.sort(),
                Op::Skip(op) => op.sort(),
                Op::Zero(op) => op.sort(),
                op => panic!("{op} is not a structural operation"),
            };
            let [w] = term.write() else {
                panic!("{} writes one wire", term.itype());
            };
            (*w, s)
        })
        .collect();
    terms.sort_by_key(|(w, _)| w.id());
    terms
}

#[test]
fn constants_and_linear_maps_take_their_sorts_from_the_tensor_shape() {
    let c = real(real::Const::new(Matrix::new([[1.0], [2.0]])));
    assert_eq!(c.cod(), Poly::from(base::mono![vec2()]));
    assert!(Term::constant(c, [Wire::new(vec2())]).is_ok());

    // `A` of shape `[3, 2]` and `B` of shape `[3, 1]` map `[2, 1]` to `[3, 1]`
    let a = Matrix::new([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]);
    let b = Matrix::new([[0.0], [0.0], [1.0]]);
    let l = real(real::Linear::new(a, b).unwrap());
    let vec3: sort::Any = sort::Real::new([3, 1]).into();
    assert_eq!(l.dom(), Poly::from(base::mono![vec2()]));
    assert_eq!(l.cod(), Poly::from(base::mono![vec3]));
    assert!(Term::function(l, [Wire::new(vec3)], [Wire::new(vec2())]).is_ok());
}

#[test]
fn operations_without_data_type_check_on_their_shape() {
    let (x, y) = (Wire::new(vec2()), Wire::new(vec2()));
    let (sum, pos, lt) = (Wire::new(vec2()), Wire::new(vec2()), Wire::new(bool2()));
    let all = Wire::new(bool1());

    assert!(Term::function(real(real::Add::new([2, 1])), [sum], [x, y]).is_ok());
    assert!(Term::function(real(real::ReLU::new([2, 1])), [pos], [sum]).is_ok());
    assert!(Term::function(real(real::Lt::new([2, 1])), [lt], [x, y]).is_ok());
    assert!(Term::function(boolean(bool::All::new([2, 1])), [all], [lt]).is_ok());
    let ite = || real(real::Ite::new([2, 1]));
    assert!(Term::function(ite(), [Wire::new(vec2())], [all, x, pos]).is_ok());
    // the condition is one boolean, not one per entry
    assert!(Term::function(ite(), [Wire::new(vec2())], [lt, x, pos]).is_err());
}

#[test]
fn structural_operations_take_any_sort_rates_included() {
    let (x, c) = (Var::new(vec2()), Var::new(bool2()));

    assert!(Term::constant(Op::Havoc(Havoc::new(vec2())), [X(x)]).is_ok());
    assert!(Term::function(Op::Skip(Skip::new(vec2())), [X(x)], [x]).is_ok());
    // the zero rates of change: of degree 1 for reals, and the singleton
    // `Zero` for booleans
    assert!(Term::constant(Op::Zero(Zero::new(T(vec2()))), [d(x)]).is_ok());
    assert_eq!(T(bool2()), sort::Any::from(sort::Zero));
    assert!(Term::constant(Op::Zero(Zero::new(T(bool2()))), [d(c)]).is_ok());
}

#[test]
fn a_constant_of_another_shape_than_its_wire_is_rejected() {
    let c = real(real::Const::new(Matrix::new([[1.0, 2.0]])));
    assert_eq!(
        Term::constant(c, [Wire::new(vec2())]).unwrap_err(),
        "expected [[1, 2]] to write Real(1, 2), got Real(2, 1)"
    );
}

#[test]
fn a_linear_map_whose_matrices_do_not_fit_its_wires_is_rejected() {
    // `A` of shape `[2, 3]` reads columns of three, not of two
    let a = Matrix::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    let l = real(real::Linear::new(a, zeros()).unwrap());
    assert_eq!(
        Term::function(l, [Wire::new(vec2())], [Wire::new(vec2())]).unwrap_err(),
        "expected Linear([[1, 0, 0], [0, 1, 0]], [[0], [0]]) to read Real(3, 1), got Real(2, 1)"
    );

    // `B` of shape `[2, 1]` writes columns of two, not of three
    let l = real(real::Linear::new(identity(), zeros()).unwrap());
    let vec3: sort::Any = sort::Real::new([3, 1]).into();
    assert_eq!(
        Term::function(l, [Wire::new(vec3)], [Wire::new(vec2())]).unwrap_err(),
        "expected Linear([[1, 0], [0, 1]], [[0], [0]]) to write Real(2, 1), got Real(3, 1)"
    );
}

#[test]
fn a_value_on_a_derivative_wire_is_rejected() {
    let x = Var::new(vec2());
    assert_eq!(
        Term::constant(Op::Zero(Zero::new(vec2())), [d(x)]).unwrap_err(),
        "expected Zero to write Real(2, 1), got T^1(Real(2, 1))"
    );
    assert_eq!(
        Term::constant(real(real::Const::new(zeros())), [d(x)]).unwrap_err(),
        "expected [[0], [0]] to write Real(2, 1), got T^1(Real(2, 1))"
    );
}

/// `x' = relu(A·x + B)`, from `x = [1, -1]`; the constructor adds the zero
/// rate of change of `x`.
#[test]
fn a_sequential_atom_holds_matrix_operations_and_a_zero_delay_of_the_tangent() {
    let x = Var::new(vec2());
    let init =
        [Term::constant(real(real::Const::new(Matrix::new([[1.0], [-1.0]]))), [X(x)]).unwrap()];
    let affine = Wire::new(vec2());
    let update = [
        Term::function(
            real(real::Linear::new(identity(), zeros()).unwrap()),
            [affine],
            [x],
        )
        .unwrap(),
        Term::function(real(real::ReLU::new([2, 1])), [X(x)], [affine]).unwrap(),
    ];

    let atom = Atom::sequential(&[x], init, update).unwrap();
    assert_eq!((atom.init().len(), atom.update().len()), (1, 2));
    assert_eq!(structural(atom.delay()), [(d(x), T(vec2()))]);
}

/// `dx = A·dt + dt`, a linear map of the rates of change of the input `t`;
/// the constructor adds the update `x' = x`.
#[test]
fn a_differential_atom_computes_on_rates_and_skips_its_values() {
    let (x, t) = (Var::new(vec2()), Var::new(vec2()));
    let init = [Term::constant(real(real::Const::new(zeros())), [X(x)]).unwrap()];
    let scaled = Wire::new(T(vec2()));
    let delay = [
        Term::function(
            real(real::Linear::with_degree(identity(), zeros(), 1).unwrap()),
            [scaled],
            [d(t)],
        )
        .unwrap(),
        Term::function(
            real(real::Add::with_degree([2, 1], 1)),
            [d(x)],
            [scaled, d(t)],
        )
        .unwrap(),
    ];

    let atom = Atom::differential(&[x, t], init, delay).unwrap();
    assert_eq!(atom.delay().len(), 2);
    assert_eq!(structural(atom.update()), [(X(x), vec2())]);
    assert_eq!(atom.update().read(), [*x]);
}

/// `c' = all(x < relu(x))` and `x' = if c then relu(x) else x`, without
/// init or delay: the constructor havocs both variables and adds their zero
/// rates of change, of degree 1 for `x` and the singleton `Zero` for `c`.
#[test]
fn a_jump_atom_havocs_its_values_and_zeroes_their_rates() {
    let (x, c) = (Var::new(vec2()), Var::new(bool1()));
    let (pos, lt) = (Wire::new(vec2()), Wire::new(bool2()));
    let update = [
        Term::function(real(real::ReLU::new([2, 1])), [pos], [x]).unwrap(),
        Term::function(real(real::Lt::new([2, 1])), [lt], [*x, pos]).unwrap(),
        Term::function(boolean(bool::All::new([2, 1])), [X(c)], [lt]).unwrap(),
        Term::function(real(real::Ite::new([2, 1])), [X(x)], [*c, pos, *x]).unwrap(),
    ];

    let atom = Atom::jump(&[x, c], update).unwrap();
    assert_eq!(structural(atom.init()), [(X(x), vec2()), (X(c), bool1())]);
    assert_eq!(
        structural(atom.delay()),
        [(d(x), T(vec2())), (d(c), sort::Any::from(sort::Zero))]
    );
}

/// The counter `x' = x + one`, on columns of two reals.
fn counter(x: Var<sort::Any>, one: Matrix) -> Result<Atom, String> {
    let init = Term::constant(real(real::Const::new(zeros())), [X(x)])?;
    let c = Wire::new(vec2());
    let update = [
        Term::constant(real(real::Const::new(one)), [c])?,
        Term::function(real(real::Add::new([2, 1])), [X(x)], [*x, c])?,
    ];
    Atom::sequential(&[x], [init], update)
}

/// The constructors of atoms take terms, which are checked as they are
/// built: an atom with an ill-typed term fails to build at that term.
#[test]
fn an_atom_with_an_ill_typed_term_fails_to_build() {
    assert!(counter(Var::new(vec2()), Matrix::new([[1.0], [1.0]])).is_ok());
    assert_eq!(
        counter(Var::new(vec2()), Matrix::new([[1.0]])).unwrap_err(),
        "expected [[1]] to write Real(1, 1), got Real(2, 1)"
    );
}
