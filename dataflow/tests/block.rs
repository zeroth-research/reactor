use base::Wire;
use base::boundary::{Boundary, Poly};
use dataflow::diagram::{Block, Term};
use theory::Theory;
use theory::lra::{LRA, Sort};

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
