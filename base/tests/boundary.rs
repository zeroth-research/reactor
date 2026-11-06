//! Tests for boundaries: polynomials of `⊗`-monomials under `⊕` and `⊗`.

use base::boundary::{Mono, Poly};

/// A monomial over single-character objects: `mono("xy")` is `x ⊗ y`.
fn mono(word: &str) -> Mono<char> {
    word.chars().collect()
}

/// A polynomial from its summands, in order: `poly(&["xy", "z"])` is
/// `xy ⊕ z`. The empty word `""` is the unit monomial `I`.
fn poly(words: &[&str]) -> Poly<char> {
    words.iter().map(|w| mono(w)).collect()
}

#[test]
fn boundaries() {
    // ⊗ of monomials is concatenation, with the unit I
    let xy = Mono::from_iter(['x']) * Mono::from_iter(['y']);
    assert_eq!(Poly::from(xy.clone()), Poly::from_iter(['x', 'y']));
    assert_eq!(xy.clone() * Mono::unit(), xy);

    // operator syntax: x * y = xy
    assert_eq!(
        Mono::from_iter(['x']) * Mono::from_iter(['y']),
        Mono::from_iter(['x', 'y'])
    );
}

#[test]
fn concatenation_of_monomials() {
    // ⊗ of two monomials is a monomial: its word is the first word followed
    // by the second
    let (m, n) = (mono("xy"), mono("zwv"));
    let mn = m.clone() * n.clone();
    assert_eq!(mn, mono("xyzwv"));
    assert_eq!(
        mn.iter().collect::<Vec<_>>(),
        m.iter().chain(n.iter()).collect::<Vec<_>>()
    );
}

#[test]
fn addition_of_polynomials() {
    // ⊕ of two polynomials is a polynomial: its summands are those of the
    // first followed by those of the second, unit monomials included
    let (p, q) = (poly(&["xy", ""]), poly(&["z", "wv", "u"]));
    let pq = p.clone() + q.clone();
    assert_eq!(pq, poly(&["xy", "", "z", "wv", "u"]));
    assert_eq!(
        pq.iter().collect::<Vec<_>>(),
        p.iter().chain(q.iter()).collect::<Vec<_>>()
    );

    // likewise for monomials: two summands, in order
    assert_eq!(mono("xy") + mono("z"), poly(&["xy", "z"]));
}

#[test]
fn unit_on_both_sides() {
    // I is the unit of ⊗ on both sides
    let m = mono("xy");
    assert_eq!(Mono::unit() * m.clone(), m);
    assert_eq!(m.clone() * Mono::unit(), m);

    // the empty word is I, and I ⊗ I = I
    assert_eq!(mono(""), Mono::unit());
    assert_eq!(Mono::unit() * mono(""), Mono::unit());

    // the sum of the empty word alone is I
    assert_eq!(poly(&[""]), Poly::unit());
}

#[test]
fn zero_on_both_sides() {
    // the empty sum is 0, and 0 ⊕ 0 = 0
    assert_eq!(poly(&[]), Poly::zero());
    assert_eq!(Poly::zero() + poly(&[]), Poly::zero());

    let p = poly(&["xy", "z"]);

    // 0 is the unit of ⊕ on both sides
    assert_eq!(Poly::zero() + p.clone(), p);
    assert_eq!(p.clone() + Poly::zero(), p);
}

#[test]
fn associativity() {
    let (p, q, r) = (poly(&["a", "b"]), poly(&["c", "d"]), poly(&["e", "f"]));
    assert_eq!((p.clone() + q.clone()) + r.clone(), p + (q + r));

    let (x, y, z) = (mono("ab"), mono("c"), mono("de"));
    assert_eq!((x.clone() * y.clone()) * z.clone(), x * (y * z));
}

#[test]
fn equality_is_structural() {
    // sums are lists, not sets: Bool = I ⊕ I is not I
    assert_ne!(Poly::<char>::unit() + Poly::unit(), Poly::unit());

    // ⊕ does not commute: the order of the alternatives is tracked
    assert_ne!(poly(&["x"]) + poly(&["y"]), poly(&["y"]) + poly(&["x"]));

    // nor does ⊗: words are ordered
    assert_ne!(mono("x") * mono("y"), mono("y") * mono("x"));

    // the impossible boundary 0 is not the empty boundary I
    assert_ne!(Poly::<char>::zero(), Poly::unit());
}

#[test]
fn collecting_nothing() {
    // an empty word collects into I: a single summand, the empty word
    assert_eq!(
        Poly::<char>::from_iter(std::iter::empty::<char>()),
        Poly::unit()
    );

    // an empty sum collects into 0: no summands at all
    assert_eq!(
        Poly::<char>::from_iter(std::iter::empty::<Mono<char>>()),
        Poly::zero()
    );

    // the unit monomial, injected, is the unit polynomial
    assert_eq!(Poly::from(Mono::<char>::unit()), Poly::unit());
}

#[test]
fn iteration_preserves_order() {
    // a monomial iterates over its word in order, by reference or by value
    let m = mono("xyz");
    assert_eq!(m.iter().collect::<String>(), "xyz");
    let mut word = String::new();
    for g in &m {
        word.push(*g);
    }
    assert_eq!(word, "xyz");
    assert_eq!(m.clone().into_iter().collect::<Mono<char>>(), m);

    // a polynomial iterates over its summands in order, unit monomials
    // included
    let p = poly(&["xy", "", "z"]);
    let words: Vec<String> = p.iter().map(|m| m.iter().collect()).collect();
    assert_eq!(words, ["xy", "", "z"]);
    let mut words = Vec::new();
    for m in &p {
        words.push(m.iter().collect::<String>());
    }
    assert_eq!(words, ["xy", "", "z"]);
    assert_eq!(p.clone().into_iter().collect::<Poly<char>>(), p);

    // 0 has no summands, I has one
    assert_eq!(Poly::<char>::zero().iter().count(), 0);
    assert_eq!(Poly::<char>::unit().iter().count(), 1);
}
