//! Tests for boundaries: polynomials of `⊗`-monomials under `⊕` and `⊗`.

use base::boundary::{Mono, Poly};

#[test]
fn boundaries() {
    // Bool = I ⊕ I, and ⊗ distributes: Bool ⊗ x has two summands
    let boolean = Poly::<char>::unit() + Poly::unit();
    let paired = boolean * Poly::from_iter(['x']);
    assert_eq!(paired, Poly::from_iter(['x']) + Poly::from_iter(['x']));

    // ⊗ of monomials is concatenation, with the unit I
    let xy = Poly::from_iter(['x']) * Poly::from_iter(['y']);
    assert_eq!(xy, Poly::from_iter(['x', 'y']));
    assert_eq!(xy.clone() * Poly::unit(), xy);

    // 0 annihilates
    assert_eq!(xy.clone() * Poly::zero(), Poly::zero());

    // operator syntax: (x + y) * z = xz ⊕ yz
    let (x, y, z) = (
        Mono::from_iter(['x']),
        Mono::from_iter(['y']),
        Mono::from_iter(['z']),
    );
    assert_eq!(
        (x + y) * Poly::from(z),
        Poly::from_iter([Mono::from_iter(['x', 'z']), Mono::from_iter(['y', 'z'])])
    );
    assert_eq!(
        Mono::from_iter(['x']) * Mono::from_iter(['y']),
        Mono::from_iter(['x', 'y'])
    );

    // ⊗ of sums keeps the order: (x + y) * (z + w) = xz ⊕ xw ⊕ yz ⊕ yw
    let (x, y, z, w) = (
        Mono::from_iter(['x']),
        Mono::from_iter(['y']),
        Mono::from_iter(['z']),
        Mono::from_iter(['w']),
    );
    assert_eq!(
        (x + y) * (z + w),
        Poly::from_iter([
            Mono::from_iter(['x', 'z']),
            Mono::from_iter(['x', 'w']),
            Mono::from_iter(['y', 'z']),
            Mono::from_iter(['y', 'w']),
        ])
    );
}
