use base::Wire;
use base::boundary::{Boundary, Poly};
use dataflow::diagram::{Block, Term};
use theory::lra::{LRA, Sort};

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
