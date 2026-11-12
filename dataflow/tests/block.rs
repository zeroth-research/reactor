//! Blocks with an explicit, ordered interface, and their composition.

use base::boundary::{Boundary, Mono};
use dataflow::diagram::{Block, Term, Wire};
use theory::lra::{LRA, Sort};

const R: Sort = Sort::Real([1, 1]);
const B: Sort = Sort::Bool([1, 1]);

/// `y := ReLU(x)` as a block `R → R`.
fn relu() -> Block<LRA> {
    let (x, y) = (Wire::scalar(R), Wire::scalar(R));
    let t = Term::function(LRA::ReLU(), [y], [x]).unwrap();
    Block::new([x], [y], [t]).unwrap()
}

fn itypes(b: &Block<LRA>) -> Vec<String> {
    b.iter().map(|t| t.itype().to_string()).collect()
}

#[test]
fn interface_is_ordered_as_given() {
    let (x, y) = (Wire::scalar(R), Wire::scalar(B));
    let b = Block::<LRA>::new([y, x], [x, y], []).unwrap();
    assert_eq!(b.read_type(), Mono::from_iter([B, R]));
    assert_eq!(b.write_type(), Mono::from_iter([R, B]));
    assert_eq!(b.dom(), Mono::from_iter([B, R]).into());
    assert_eq!(b.cod(), Mono::from_iter([R, B]).into());
}

#[test]
fn pass_through_discard_and_copy() {
    let (x, y) = (Wire::scalar(R), Wire::scalar(R));
    // discard `y`, copy `x`
    let b = Block::<LRA>::new([x, y], [x, x], []).unwrap();
    assert_eq!(b.read_type(), Mono::from_iter([R, R]));
    assert_eq!(b.write_type(), Mono::from_iter([R, R]));
    assert!(b.is_empty());
}

#[test]
fn ill_formed_interfaces_are_rejected() {
    let (x, y, z) = (Wire::scalar(R), Wire::scalar(R), Wire::scalar(R));
    let relu_yx = || Term::function(LRA::ReLU(), [y], [x]).unwrap();
    // duplicate read wire
    assert!(Block::<LRA>::new([x, x], [x], []).is_err());
    // a term reads a wire that is neither read nor written before
    assert!(Block::new([z], [y], [relu_yx()]).is_err());
    // a write wire is never written
    assert!(Block::new([x], [z], [relu_yx()]).is_err());
    // a term writes a read wire
    assert!(Block::new([x, y], [y], [relu_yx()]).is_err());
    // a wire is written twice
    assert!(Block::new([x], [y], [relu_yx(), relu_yx()]).is_err());
}

#[test]
fn id_passes_through_fresh_wires() {
    let sig = Mono::from_iter([R, B]);
    let id = Block::<LRA>::id(&sig);
    assert!(id.is_empty());
    assert_eq!(id.read(), id.write());
    assert_eq!(id.read_type(), sig);
    assert_ne!(Block::<LRA>::id(&sig).read(), id.read());
}

#[test]
fn compose_with_itself_renames() {
    let f = relu();
    let ff = Block::compose(&f, &f).unwrap();
    assert_eq!(itypes(&ff), ["ReLU", "ReLU"]);
    assert_eq!(ff.read(), f.read());
    // the second term reads what the first writes and writes a fresh wire
    let (t0, t1) = (ff.get(0).unwrap(), ff.get(1).unwrap());
    assert_eq!(t1.read(), t0.write());
    assert_ne!(t1.write(), t0.write());
    assert_eq!(ff.write(), t1.write());
    // and the result is well-formed
    let rebuilt = Block::new(ff.read().to_vec(), ff.write().to_vec(), ff.clone());
    assert!(rebuilt.is_ok());
}

#[test]
fn compose_with_id_is_neutral() {
    let f = relu();
    let id = Block::<LRA>::id(&Mono::from_iter([R]));
    for g in [
        Block::compose(&id, &f).unwrap(),
        Block::compose(&f, &id).unwrap(),
    ] {
        assert_eq!(itypes(&g), ["ReLU"]);
        assert_eq!(g.read_type(), f.read_type());
        assert_eq!(g.write_type(), f.write_type());
    }
}

#[test]
fn compose_checks_types() {
    let id_b = Block::<LRA>::id(&Mono::from_iter([B]));
    assert!(Block::compose(&relu(), &id_b).is_err());
}
