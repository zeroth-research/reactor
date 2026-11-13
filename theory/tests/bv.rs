//! Tests for the theory of bit-vectors.

#![cfg(feature = "torch")]

use theory::bv::Sort;

fn bv(bw: usize, r: usize, c: usize) -> Sort {
    Sort::BV(bw, [r, c])
}

// --- Type helpers ---

#[test]
fn type_shape_and_bw() {
    let t = bv(8, 2, 3);
    assert_eq!(t.shape(), &[2, 3]);
    assert_eq!(t.bw(), 8);

    let t = bv(16, 1, 4);
    assert_eq!(t.shape(), &[1, 4]);
    assert_eq!(t.bw(), 16);
}
