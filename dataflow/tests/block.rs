use base::boundary::{Boundary, Poly};
use base::tangent::Tangent;
use dataflow::diagram::{Block, Term, Wire};
use theory::lra::{LRA, Sort};
use theory::{Combinatorial, Differential, Sequential, Theory};

fn real() -> Sort {
    Sort::real([1, 1])
}

fn wire() -> Wire<Sort> {
    Wire::new(real())
}

fn add(write: &Wire<Sort>, a: &Wire<Sort>, b: &Wire<Sort>) -> Term<LRA> {
    Term::function(LRA::Add(1, 1), [*write], [*a, *b]).unwrap()
}

fn id(write: &Wire<Sort>, read: &Wire<Sort>) -> Term<LRA> {
    Term::function(LRA::IdReal(1, 1), [*write], [*read]).unwrap()
}

fn discard(read: &Wire<Sort>) -> Term<LRA> {
    Term::function(
        LRA::discard(read.dtype()),
        Vec::<Wire<Sort>>::new(),
        [*read],
    )
    .unwrap()
}

#[test]
fn boundary_keeps_the_given_order() {
    // created in the reverse of the boundary order
    let (y, b, a) = (wire(), wire(), wire());
    let t = wire();

    // t = a + b, y = t: `t` is internal, not an output
    let block = Block::new([a, b], [y], [add(&t, &a, &b), id(&y, &t)]).unwrap();
    assert_eq!(block.read(), [a, b]);
    assert_eq!(block.write(), [y]);
    assert_eq!(
        block.dom(),
        [real(), real()].into_iter().collect::<Poly<_>>()
    );
    assert_eq!(block.cod(), [real()].into_iter().collect::<Poly<_>>());

    // the order is the given one, whichever it is
    let block = Block::new([b, a], [y], [add(&t, &a, &b), id(&y, &t)]).unwrap();
    assert_eq!(block.read(), [b, a]);
}

#[test]
fn every_input_is_read() {
    let (a, b, y) = (wire(), wire(), wire());
    assert!(Block::new([a, b], [y], [id(&y, &a)]).is_err());

    // an input without other use is discarded
    assert!(Block::new([a, b], [y], [id(&y, &a), discard(&b)]).is_ok());
    // an input may be read several times
    assert!(Block::new([a], [y], [add(&y, &a, &a)]).is_ok());
}

#[test]
fn every_output_is_written_once() {
    let (a, y, z) = (wire(), wire(), wire());
    assert!(Block::new([a], [y, z], [id(&y, &a)]).is_err());
    assert!(Block::new([a], [y], [id(&y, &a), id(&y, &a)]).is_err());
    // an output may be read by a later term
    assert!(Block::new([a], [y, z], [id(&y, &a), id(&z, &y)]).is_ok());
}

#[test]
fn terms_read_inputs_or_earlier_writes() {
    let (a, d, t, y) = (wire(), wire(), wire(), wire());
    // `d` is neither an input nor written by any term
    assert!(Block::new([a], [y], [add(&y, &a, &d)]).is_err());
    // `t` is read before it is written
    assert!(Block::new([a], [y], [id(&y, &t), id(&t, &a)]).is_err());
    assert!(Block::new([a], [y], [id(&t, &a), id(&y, &t)]).is_ok());
}

#[test]
fn inputs_are_not_written() {
    let (a, b, y) = (wire(), wire(), wire());
    let terms = [id(&b, &a), id(&y, &b)];
    assert!(Block::new([a, b], [y], terms).is_err());
}

#[test]
fn boundary_wires_are_distinct() {
    let (a, y) = (wire(), wire());
    assert!(Block::new([a, a], [y], [id(&y, &a)]).is_err());
    assert!(Block::new([a], [y, y], [id(&y, &a)]).is_err());
    assert!(Block::new([a], [a], [discard(&a)]).is_err());
}

#[test]
fn empty_boundary() {
    let block: Block<LRA> = Block::new([], [], []).unwrap();
    assert_eq!(block.dom(), Poly::unit());
    assert_eq!(block.cod(), Poly::unit());
}

#[test]
fn boundary_is_the_product_of_the_wire_sorts() {
    let real = Sort::real([1, 1]);
    let bool = Sort::Bool { shape: [1, 1] };
    let (a, b, c) = (Wire::new(real), Wire::new(real), Wire::new(bool));

    // c = a < b
    let lt = Term::function(LRA::Lt(1, 1), [c], [a, b]).unwrap();
    let block: Block<LRA> = Block::try_from_iter([Ok(lt)]).unwrap();
    assert_eq!(block.dom(), [real, real].into_iter().collect::<Poly<_>>());
    assert_eq!(block.cod(), [bool].into_iter().collect::<Poly<_>>());

    let empty: Block<LRA> = Block::empty();
    assert_eq!(empty.dom(), Poly::unit());
    assert_eq!(empty.cod(), Poly::unit());
}

fn itypes(block: &Block<LRA>) -> Vec<String> {
    block.iter().map(|t| t.itype().to_string()).collect()
}

#[test]
fn op_reads_all_inputs_and_writes_all_outputs() {
    let bool = Sort::Bool { shape: [1, 1] };
    let block = Block::op(LRA::Lt(1, 1), [real(), real()], [bool]).unwrap();
    assert_eq!(block.len(), 1);
    assert_eq!(
        block.dom(),
        [real(), real()].into_iter().collect::<Poly<_>>()
    );
    assert_eq!(block.cod(), [bool].into_iter().collect::<Poly<_>>());
    let term = block.get(0).unwrap();
    assert_eq!(term.read(), block.read());
    assert_eq!(term.write(), block.write());

    // the operation type-checks against the sorts
    assert!(Block::op(LRA::Lt(1, 1), [real()], [bool]).is_err());
}

#[test]
fn pointwise_reads_inputs_by_position() {
    // a, b, c -> a + a, c
    let ops = [
        (LRA::Add(1, 1), vec![0, 0], real()),
        (LRA::IdReal(1, 1), vec![2], real()),
    ];
    let block = Block::pointwise([real(); 3], ops).unwrap();
    assert_eq!(block.dom(), [real(); 3].into_iter().collect::<Poly<_>>());
    assert_eq!(block.cod(), [real(); 2].into_iter().collect::<Poly<_>>());
    let (a, c) = (block.read()[0], block.read()[2]);
    assert_eq!(block.get(0).unwrap().read(), [a, a]);
    assert_eq!(block.get(1).unwrap().read(), [c]);
    assert_eq!(block.get(0).unwrap().write(), [block.write()[0]]);
    assert_eq!(block.get(1).unwrap().write(), [block.write()[1]]);

    // `b` is read by no operation: it is discarded, after the operations
    assert_eq!(block.len(), 3);
    assert_eq!(block.get(2).unwrap().read(), [block.read()[1]]);
    let discard = LRA::discard(&real()).to_string();
    assert_eq!(itypes(&block)[2], discard);

    // a position beyond the inputs, and an operation not type-checking
    let beyond = [(LRA::IdReal(1, 1), vec![3], real())];
    assert!(Block::pointwise([real(); 3], beyond).is_err());
    let unary = [(LRA::Add(1, 1), vec![0], real())];
    assert!(Block::pointwise([real(); 3], unary).is_err());
}

#[test]
fn havoc_skip_and_zero_give_their_outputs_and_discard_the_rest() {
    let bool = Sort::Bool { shape: [1, 1] };
    let dreal = real().tangent();

    // a -> arbitrary, arbitrary
    let havoc = Block::havoc([real()], [real(), bool]).unwrap();
    assert_eq!(havoc.cod(), [real(), bool].into_iter().collect::<Poly<_>>());
    let expected = [
        LRA::havoc(&real()),
        LRA::havoc(&bool),
        LRA::discard(&real()),
    ];
    assert_eq!(itypes(&havoc), expected.map(|op| op.to_string()));

    // a, p -> p, a
    let skip = Block::skip([real(), bool], [1, 0]).unwrap();
    assert_eq!(skip.cod(), [bool, real()].into_iter().collect::<Poly<_>>());
    assert_eq!(skip.get(0).unwrap().read(), [skip.read()[1]]);
    assert_eq!(skip.get(1).unwrap().read(), [skip.read()[0]]);
    assert_eq!(
        itypes(&skip),
        [LRA::skip(&bool), LRA::skip(&real())].map(|op| op.to_string())
    );
    assert!(Block::<LRA>::skip([real()], [1]).is_err());

    // a -> 0
    let zero = Block::zero([real()], [dreal]).unwrap();
    assert_eq!(zero.cod(), [dreal].into_iter().collect::<Poly<_>>());
    assert_eq!(
        itypes(&zero),
        [LRA::zero(&dreal), LRA::discard(&real())].map(|op| op.to_string())
    );
    // a zero is a derivative
    assert!(Block::<LRA>::zero([], [real()]).is_err());
}

#[test]
fn boundary_names_follow_the_positions() {
    // t = a + b, y = t
    let (y, b, a, t) = (wire(), wire(), wire(), wire());
    let block = Block::new([a, b], [y], [add(&t, &a, &b), id(&y, &t)]).unwrap();

    let shown = block.with_boundary_names(["a", "b"], ["y"]).to_string();
    let (add, id) = (LRA::Add(1, 1), LRA::IdReal(1, 1));
    let t = t.untyped().to_string();
    assert_eq!(shown, format!("{t} := {add} (a, b)\ny := {id} {t}\n"));

    // the names left out print as the wires
    let shown = block
        .with_boundary_names(["a"], Vec::<&str>::new())
        .to_string();
    let (b, y) = (b.untyped().to_string(), y.untyped().to_string());
    assert_eq!(shown, format!("{t} := {add} (a, {b})\n{y} := {id} {t}\n"));
}
